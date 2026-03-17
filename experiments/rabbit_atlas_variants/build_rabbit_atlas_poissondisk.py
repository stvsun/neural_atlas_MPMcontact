#!/usr/bin/env python3
"""
Build a meshfree rabbit atlas from point clouds using Poisson-disk-like seeding.

Outputs:
- canonical normalized point cloud with normals
- chart seeds, local frames, overlap memberships
- overlap graph and chart coloring for Schwarz iterations
"""

import argparse
import json
import math
import os
import random
import sys
import struct
import tarfile
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PLY_TYPE_MAP = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def parse_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with open(path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {path}")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        fmt = None
        n_vertices = None
        vertex_props: List[Tuple[str, str]] = []
        current_element = None

        for line in header_lines:
            toks = line.split()
            if not toks:
                continue
            if toks[0] == "format":
                fmt = toks[1]
            elif toks[0] == "element":
                current_element = toks[1]
                if current_element == "vertex":
                    n_vertices = int(toks[2])
            elif toks[0] == "property" and current_element == "vertex":
                if len(toks) >= 3 and toks[1] != "list":
                    p_type = toks[1]
                    p_name = toks[2]
                    vertex_props.append((p_name, p_type))

        if fmt is None or n_vertices is None or not vertex_props:
            raise ValueError(f"Unsupported PLY header: {path}")

        prop_names = [p[0] for p in vertex_props]
        xyz_names = ["x", "y", "z"]
        if not all(name in prop_names for name in xyz_names):
            raise ValueError(f"PLY vertex does not contain x/y/z: {path}")

        nx_present = all(name in prop_names for name in ["nx", "ny", "nz"])

        if fmt == "ascii":
            vals = []
            for _ in range(n_vertices):
                row = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(row) < len(vertex_props):
                    raise ValueError(f"Malformed ASCII PLY vertex row in {path}")
                vals.append([float(row[i]) for i in range(len(vertex_props))])
            arr = np.asarray(vals, dtype=float)
        elif fmt == "binary_little_endian":
            fmt_chars = []
            for _, p_type in vertex_props:
                if p_type not in PLY_TYPE_MAP:
                    raise ValueError(f"Unsupported PLY property type '{p_type}' in {path}")
                fmt_chars.append(PLY_TYPE_MAP[p_type])
            rec_fmt = "<" + "".join(fmt_chars)
            rec_size = struct.calcsize(rec_fmt)
            vals = []
            for _ in range(n_vertices):
                data = f.read(rec_size)
                if len(data) != rec_size:
                    raise ValueError(f"Unexpected EOF in PLY vertex data: {path}")
                vals.append(struct.unpack(rec_fmt, data))
            arr = np.asarray(vals, dtype=float)
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    idx_x = prop_names.index("x")
    idx_y = prop_names.index("y")
    idx_z = prop_names.index("z")
    points = arr[:, [idx_x, idx_y, idx_z]]

    normals = None
    if nx_present:
        idx_nx = prop_names.index("nx")
        idx_ny = prop_names.index("ny")
        idx_nz = prop_names.index("nz")
        normals = arr[:, [idx_nx, idx_ny, idx_nz]]

    return points.astype(float), normals.astype(float) if normals is not None else None


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lower = path.lower()
    if lower.endswith(".npz"):
        data = np.load(path)
        if "points" not in data:
            raise ValueError("NPZ must include key 'points'")
        points = np.asarray(data["points"], dtype=float)
        normals = np.asarray(data["normals"], dtype=float) if "normals" in data else None
        return points, normals

    if lower.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("NPY must have shape [N,3] or [N,6]")
        points = arr[:, :3]
        normals = arr[:, 3:6] if arr.shape[1] == 6 else None
        return points, normals

    if lower.endswith(".ply"):
        return parse_ply(path)

    if lower.endswith(".xyz") or lower.endswith(".txt") or lower.endswith(".csv"):
        arr = np.loadtxt(path, delimiter="," if lower.endswith(".csv") else None)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("Text/XYZ/CSV must have shape [N,3] or [N,6]")
        points = arr[:, :3]
        normals = arr[:, 3:6] if arr.shape[1] == 6 else None
        return points, normals

    raise ValueError(f"Unsupported point-cloud format: {path}")


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def estimate_normals(points: np.ndarray, k: int = 32, chunk: int = 2048) -> np.ndarray:
    pts = torch.tensor(points, dtype=torch.float64)
    n = pts.shape[0]
    out = np.zeros((n, 3), dtype=float)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        q = pts[s:e]
        d = torch.cdist(q, pts)
        nn = torch.topk(d, k=min(k + 1, n), largest=False, dim=1).indices[:, 1:]
        neigh = pts[nn]
        centered = neigh - neigh.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / max(1, neigh.shape[1])
        _, eigvec = torch.linalg.eigh(cov)
        normal = eigvec[:, :, 0]
        out[s:e] = normal.numpy()

    out = normalize_rows(out)
    center = np.mean(points, axis=0, keepdims=True)
    flip = np.sum((points - center) * out, axis=1) < 0
    out[flip] *= -1.0
    return out


def maybe_download_stanford_bunny(download_dir: str, url: str) -> Optional[str]:
    os.makedirs(download_dir, exist_ok=True)
    archive_path = os.path.join(download_dir, "bunny.tar.gz")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception:
        return None

    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.lower().endswith(".ply")]
            if not members:
                return None
            preferred = None
            for m in members:
                if "bun_zipper" in m.name.lower():
                    preferred = m
                    break
            member = preferred if preferred is not None else members[0]
            tf.extract(member, path=download_dir)
            return os.path.join(download_dir, member.name)
    except Exception:
        return None


def farthest_point_seeds(points: np.ndarray, n_seeds: int) -> np.ndarray:
    n = points.shape[0]
    if n_seeds >= n:
        return np.arange(n, dtype=int)
    first = np.random.randint(0, n)
    seeds = [first]
    min_dist = np.linalg.norm(points - points[first : first + 1], axis=1)
    for _ in range(1, n_seeds):
        nxt = int(np.argmax(min_dist))
        seeds.append(nxt)
        d = np.linalg.norm(points - points[nxt : nxt + 1], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.asarray(seeds, dtype=int)


def chart_frames(
    points: np.ndarray,
    normals: np.ndarray,
    seed_idx: np.ndarray,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1_list = []
    t2_list = []
    n_list = []
    for idx in seed_idx:
        s = points[idx : idx + 1]
        d = np.linalg.norm(points - s, axis=1)
        nn = np.argsort(d)[: max(8, k_neighbors)]
        neigh = points[nn]
        c = np.mean(neigh, axis=0, keepdims=True)
        centered = neigh - c
        cov = centered.T @ centered / max(1, centered.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov)
        nvec = eigvecs[:, 0]

        nref = np.mean(normals[nn], axis=0)
        if np.dot(nvec, nref) < 0:
            nvec = -nvec

        t1 = eigvecs[:, 2]
        t1 = t1 - nvec * np.dot(t1, nvec)
        t1 = t1 / max(np.linalg.norm(t1), 1e-12)
        t2 = np.cross(nvec, t1)
        t2 = t2 / max(np.linalg.norm(t2), 1e-12)

        t1_list.append(t1)
        t2_list.append(t2)
        n_list.append(nvec / max(np.linalg.norm(nvec), 1e-12))

    return np.asarray(t1_list), np.asarray(t2_list), np.asarray(n_list)


def overlap_membership(dist: np.ndarray, target_overlap: float) -> Tuple[np.ndarray, float, float]:
    dmin = np.min(dist, axis=1, keepdims=True)

    lo, hi = 0.0, 1.5
    best_alpha = 0.2
    best_mask = None
    best_gap = 1e9

    for _ in range(30):
        a = 0.5 * (lo + hi)
        mask = dist <= (dmin * (1.0 + a) + 1e-12)
        overlap = float(np.mean(np.sum(mask, axis=1) > 1))
        gap = abs(overlap - target_overlap)
        if gap < best_gap:
            best_gap = gap
            best_alpha = a
            best_mask = mask
        if overlap < target_overlap:
            lo = a
        else:
            hi = a

    if best_mask is None:
        best_mask = dist <= (dmin * (1.0 + best_alpha) + 1e-12)

    overlap = float(np.mean(np.sum(best_mask, axis=1) > 1))
    return best_mask, float(best_alpha), overlap


def build_overlap_graph(membership: np.ndarray) -> Dict[str, object]:
    n_points, n_charts = membership.shape
    adj = {i: set() for i in range(n_charts)}

    chart_sizes = membership.sum(axis=0)
    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            shared = int(np.sum(membership[:, i] & membership[:, j]))
            min_shared = max(20, int(0.01 * min(chart_sizes[i], chart_sizes[j] + 1)))
            if shared >= min_shared:
                adj[i].add(j)
                adj[j].add(i)

    return {
        "adj": {str(k): sorted(list(v)) for k, v in adj.items()},
    }


def bipartite_or_greedy_coloring(adj_dict: Dict[str, List[int]], n_charts: int) -> Tuple[bool, Dict[int, int], List[List[int]]]:
    adj = {int(k): [int(x) for x in v] for k, v in adj_dict.items()}
    color: Dict[int, int] = {}
    bip = True

    for s in range(n_charts):
        if s in color:
            continue
        color[s] = 0
        queue = [s]
        h = 0
        while h < len(queue):
            u = queue[h]
            h += 1
            for v in adj.get(u, []):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    bip = False

    if bip:
        groups = [[], []]
        for i in range(n_charts):
            groups[color.get(i, 0)].append(i)
        return True, color, groups

    color = {}
    for u in range(n_charts):
        used = {color[v] for v in adj.get(u, []) if v in color}
        c = 0
        while c in used:
            c += 1
        color[u] = c

    n_colors = max(color.values()) + 1 if color else 1
    groups = [[] for _ in range(n_colors)]
    for i in range(n_charts):
        groups[color[i]].append(i)
    return False, color, groups


def subsample(points: np.ndarray, normals: np.ndarray, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= n_max:
        return points, normals
    idx = np.random.choice(points.shape[0], size=n_max, replace=False)
    return points[idx], normals[idx]


def make_chart_plot(points: np.ndarray, primary: np.ndarray, out_path: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    n_charts = int(primary.max()) + 1
    cmap = plt.cm.get_cmap("tab20", n_charts)
    for i in range(n_charts):
        m = primary == i
        if np.any(m):
            ax.scatter(points[m, 0], points[m, 1], points[m, 2], s=1.5, color=cmap(i), alpha=0.8)
    ax.set_title("Atlas primary chart assignments")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rabbit atlas with Poisson-disk chart seeds")
    parser.add_argument("--points-file", default=None, help="Local point cloud path (.npz/.npy/.xyz/.ply)")
    parser.add_argument(
        "--download-url",
        default="https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        help="Stanford bunny archive URL",
    )
    parser.add_argument("--download-dir", default=None, help="Download cache dir (default: <output-dir>/downloads)")
    parser.add_argument("--allow-procedural-fallback", action="store_true")

    parser.add_argument("--n-charts", type=int, default=12)
    parser.add_argument("--overlap-target", type=float, default=0.20)
    parser.add_argument("--n-surface-max", type=int, default=30000)
    parser.add_argument("--frame-k", type=int, default=96)
    parser.add_argument("--normal-k", type=int, default=32)

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    download_dir = args.download_dir or os.path.join(args.output_dir, "downloads")

    source = None
    points = None
    normals = None

    if args.points_file is not None:
        points, normals = load_point_cloud(args.points_file)
        source = f"local:{args.points_file}"
    else:
        ply_path = maybe_download_stanford_bunny(download_dir=download_dir, url=args.download_url)
        if ply_path is not None:
            points, normals = load_point_cloud(ply_path)
            source = f"download:{ply_path}"

    if points is None:
        if not args.allow_procedural_fallback:
            raise RuntimeError(
                "No point cloud available. Provide --points-file or enable fallback with --allow-procedural-fallback."
            )
        try:
            from src.train_sdf_rabbit import generate_procedural_rabbit  # release layout
        except ModuleNotFoundError:
            from train_sdf_rabbit import generate_procedural_rabbit  # legacy fallback

        points, normals = generate_procedural_rabbit(args.n_surface_max)
        source = "procedural_fallback"

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must be [N,3]")

    if normals is None:
        normals = estimate_normals(points, k=args.normal_k)
    normals = normalize_rows(np.asarray(normals, dtype=float))

    points, normals = subsample(points, normals, n_max=args.n_surface_max)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    scale = float(np.max(maxs - mins))
    scale = max(scale, 1e-8)

    points_n = (points - center[None, :]) / scale
    normals_n = normalize_rows(normals)

    seed_idx = farthest_point_seeds(points_n, n_seeds=args.n_charts)
    seed_pts = points_n[seed_idx]

    dmat = np.linalg.norm(points_n[:, None, :] - seed_pts[None, :, :], axis=2)
    membership, overlap_alpha, overlap_ratio = overlap_membership(dmat, target_overlap=args.overlap_target)
    primary = np.argmin(dmat, axis=1)

    t1, t2, nvec = chart_frames(points_n, normals_n, seed_idx, k_neighbors=args.frame_k)

    support_r = np.zeros((args.n_charts,), dtype=float)
    for i in range(args.n_charts):
        m = membership[:, i]
        if not np.any(m):
            support_r[i] = float(np.quantile(dmat[:, i], 0.8))
        else:
            support_r[i] = float(np.quantile(dmat[m, i], 0.95))
        support_r[i] = max(support_r[i], 1e-4)

    graph = build_overlap_graph(membership.astype(bool))
    is_bip, color_map, color_groups = bipartite_or_greedy_coloring(graph["adj"], n_charts=args.n_charts)

    atlas_npz = os.path.join(args.output_dir, "rabbit_atlas_data.npz")
    np.savez_compressed(
        atlas_npz,
        points=points_n,
        normals=normals_n,
        seed_indices=seed_idx,
        seed_points=seed_pts,
        frame_t1=t1,
        frame_t2=t2,
        frame_n=nvec,
        membership=membership.astype(np.uint8),
        primary_chart=primary.astype(np.int64),
        support_radii=support_r,
        overlap_alpha=np.asarray([overlap_alpha], dtype=float),
        center=center,
        scale=np.asarray([scale], dtype=float),
    )

    canonical_npz = os.path.join(args.output_dir, "rabbit_points_normals.npz")
    np.savez_compressed(
        canonical_npz,
        points=points_n,
        normals=normals_n,
        center=center,
        scale=np.asarray([scale], dtype=float),
    )

    plot_path = os.path.join(args.output_dir, "atlas_primary_chart_assignment.png")
    make_chart_plot(points_n, primary, plot_path)

    meta = {
        "source": source,
        "n_points": int(points_n.shape[0]),
        "n_charts": int(args.n_charts),
        "overlap_target": float(args.overlap_target),
        "overlap_alpha": float(overlap_alpha),
        "overlap_ratio": float(overlap_ratio),
        "is_bipartite": bool(is_bip),
        "color_map": {str(k): int(v) for k, v in color_map.items()},
        "color_groups": [[int(x) for x in g] for g in color_groups],
        "overlap_graph": graph["adj"],
        "atlas_npz": atlas_npz,
        "canonical_npz": canonical_npz,
        "plot": plot_path,
    }
    meta_path = os.path.join(args.output_dir, "rabbit_atlas_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Atlas build complete")
    print(f"  source:          {source}")
    print(f"  n_points:        {points_n.shape[0]}")
    print(f"  n_charts:        {args.n_charts}")
    print(f"  overlap_ratio:   {overlap_ratio:.4f}")
    print(f"  bipartite:       {is_bip}")
    print(f"  atlas_npz:       {atlas_npz}")
    print(f"  meta_json:       {meta_path}")
    print(f"  chart_plot:      {plot_path}")


if __name__ == "__main__":
    main()
