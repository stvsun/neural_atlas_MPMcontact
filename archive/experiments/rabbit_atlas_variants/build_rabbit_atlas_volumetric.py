#!/usr/bin/env python3
"""
Build a volumetric rabbit atlas from an SDF network using interior seed sampling.

M6: Replaces the surface-based Poisson-disk atlas with a volumetric atlas whose
seeds are placed INSIDE the domain (SDF < 0) rather than on the surface.  With
axis-aligned frames (t1=e1, t2=e2, n=e3), the existing ChartDecoder already
implements a ball-chart  x = seed + xi + small_residual(xi) — no architecture
changes are needed.

Outputs (same NPZ keys as build_rabbit_atlas_poissondisk.py, backward-compatible):
- rabbit_atlas_data.npz   — interior points, identity frames, overlap membership
- rabbit_atlas_meta.json  — metadata incl. "volumetric": true
- atlas_primary_chart_assignment.png
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Minimal MLP (verbatim from train_rabbit_atlas.py) for SDF network
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(width, out_dim)
        for layer in self.hidden:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)


# ---------------------------------------------------------------------------
# SDF network wrapper
# ---------------------------------------------------------------------------

class _SDFNetVolumetric(torch.nn.Module):
    """Thin wrapper around an MLP used as a Signed-Distance-Function network."""

    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_sdf_net(
    path: str,
    device: torch.device,
) -> Tuple["_SDFNetVolumetric", np.ndarray, float]:
    """Load an SDF network from a checkpoint.

    Supports two checkpoint formats:
      Format A (train_sdf_rabbit.py): keys ``model_state``, ``model_kwargs``, ``center``, ``scale``
      Format B (simplified):          keys ``model``, ``center``, ``scale``, ``width``, ``depth``

    Returns
    -------
    net     : loaded, frozen _SDFNetVolumetric
    center  : np.ndarray shape (3,)  — physical-space centre used during SDF training
    scale   : float                  — physical-space scale used during SDF training
    """
    ckpt = torch.load(path, map_location=device)

    if "model_state" in ckpt:
        kw = ckpt.get("model_kwargs", {})
        width = int(kw.get("width", 128))
        depth = int(kw.get("depth", 6))
        net = _SDFNetVolumetric(width=width, depth=depth)
        net.load_state_dict(ckpt["model_state"])
    else:
        width = int(ckpt.get("width", 128))
        depth = int(ckpt.get("depth", 6))
        net = _SDFNetVolumetric(width=width, depth=depth)
        net.load_state_dict(ckpt["model"])

    net.to(device)
    net.eval()
    net.requires_grad_(False)

    center = np.array(ckpt["center"], dtype=float).reshape(3)
    scale = float(ckpt["scale"])
    return net, center, scale


# ---------------------------------------------------------------------------
# Interior point sampling
# ---------------------------------------------------------------------------

def sample_interior_points(
    sdf_net: "_SDFNetVolumetric",
    n_target: int,
    bbox_half: float,
    rejection_factor: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """Sample ``n_target`` points inside the domain (SDF < threshold).

    Candidates are drawn uniformly in ``[-bbox_half, bbox_half]^3``, which is
    the SDF network's *normalised* input space.  Points are accepted when
    ``sdf_net(x) < threshold``.

    Parameters
    ----------
    sdf_net         : frozen SDF network (takes normalised 3-D input)
    n_target        : desired number of interior points
    bbox_half       : half-extent of the sampling bounding cube (in SDF-normalised coords)
    rejection_factor: how many candidates to draw per required sample
    threshold       : SDF acceptance threshold (0 = exactly inside)
    device          : torch device

    Returns
    -------
    np.ndarray of shape (n_target, 3) in SDF-normalised coordinate space
    """
    accepted_parts: List[np.ndarray] = []
    n_accepted = 0
    attempts = 0
    max_attempts = 50

    while n_accepted < n_target and attempts < max_attempts:
        attempts += 1
        n_cand = max(8192, rejection_factor * n_target)
        x_norm = np.random.uniform(-bbox_half, bbox_half, (n_cand, 3)).astype(np.float64)
        with torch.no_grad():
            x_t = torch.tensor(x_norm, dtype=torch.float64, device=device)
            sdf = sdf_net(x_t).cpu().numpy()
        mask = sdf < threshold
        accepted_here = x_norm[mask]
        if accepted_here.shape[0] > 0:
            accepted_parts.append(accepted_here)
            n_accepted += accepted_here.shape[0]

        rate = mask.mean()
        if rate < 0.10 and attempts == 1:
            print(
                f"  [WARNING] Low SDF acceptance rate ({rate:.3f}). "
                "Consider increasing --rejection-factor or adjusting --bbox-half."
            )

    if n_accepted == 0:
        raise RuntimeError(
            "SDF rejection sampling produced 0 interior points. "
            "Check that --sdf-checkpoint is valid and --bbox-half covers the domain."
        )

    pts = np.concatenate(accepted_parts, axis=0)
    if pts.shape[0] < n_target:
        print(
            f"  [WARNING] Only {pts.shape[0]} interior points available; "
            f"requested {n_target}. Using all available points."
        )
        return pts
    idx = np.random.choice(pts.shape[0], size=n_target, replace=False)
    return pts[idx]


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def axis_aligned_frames(n_charts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return identity-frame vectors for each chart.

    With t1=e1, t2=e2, n=e3, the ChartDecoder computes
    ``base = seed + xi[0]*e1 + xi[1]*e2 + xi[2]*e3 = seed + xi``,
    which is a perfect 3-D ball chart centred at seed.
    """
    t1   = np.tile([1.0, 0.0, 0.0], (n_charts, 1))
    t2   = np.tile([0.0, 1.0, 0.0], (n_charts, 1))
    nvec = np.tile([0.0, 0.0, 1.0], (n_charts, 1))
    return t1, t2, nvec


def pca_frames(
    interior_pts: np.ndarray,
    seed_pts: np.ndarray,
    k: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct local frames from PCA of k nearest interior neighbours.

    The dominant eigenvector (largest eigenvalue) becomes t1, the second
    largest becomes t2, and n = t1 x t2 (right-handed system).
    """
    t1_list, t2_list, n_list = [], [], []
    for s in seed_pts:
        d = np.linalg.norm(interior_pts - s[None, :], axis=1)
        nn = np.argsort(d)[:max(8, k)]
        neigh = interior_pts[nn]
        c = np.mean(neigh, axis=0, keepdims=True)
        centered = neigh - c
        cov = centered.T @ centered / max(1, centered.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

        t1 = eigvecs[:, 2]  # largest eigenvector
        t2 = eigvecs[:, 1]  # middle eigenvector
        n  = np.cross(t1, t2)

        # normalise
        t1 = t1 / max(np.linalg.norm(t1), 1e-12)
        t2 = t2 / max(np.linalg.norm(t2), 1e-12)
        n  = n  / max(np.linalg.norm(n),  1e-12)

        # enforce right-handed
        if np.dot(np.cross(t1, t2), n) < 0:
            t2 = -t2
            n  = np.cross(t1, t2)
            n  = n / max(np.linalg.norm(n), 1e-12)

        t1_list.append(t1)
        t2_list.append(t2)
        n_list.append(n)

    return np.array(t1_list), np.array(t2_list), np.array(n_list)


# ---------------------------------------------------------------------------
# Reused verbatim from build_rabbit_atlas_poissondisk.py
# ---------------------------------------------------------------------------

def farthest_point_seeds(points: np.ndarray, n_seeds: int) -> np.ndarray:
    n = points.shape[0]
    if n_seeds >= n:
        return np.arange(n, dtype=int)
    first = np.random.randint(0, n)
    seeds = [first]
    min_dist = np.linalg.norm(points - points[first: first + 1], axis=1)
    for _ in range(1, n_seeds):
        nxt = int(np.argmax(min_dist))
        seeds.append(nxt)
        d = np.linalg.norm(points - points[nxt: nxt + 1], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.asarray(seeds, dtype=int)


def overlap_membership(
    dist: np.ndarray,
    target_overlap: float,
) -> Tuple[np.ndarray, float, float]:
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
    adj: Dict[int, set] = {i: set() for i in range(n_charts)}
    chart_sizes = membership.sum(axis=0)
    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            shared = int(np.sum(membership[:, i] & membership[:, j]))
            min_shared = max(20, int(0.01 * min(chart_sizes[i], chart_sizes[j] + 1)))
            if shared >= min_shared:
                adj[i].add(j)
                adj[j].add(i)
    return {"adj": {str(k): sorted(list(v)) for k, v in adj.items()}}


def bipartite_or_greedy_coloring(
    adj_dict: Dict[str, List[int]],
    n_charts: int,
) -> Tuple[bool, Dict[int, int], List[List[int]]]:
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
        groups: List[List[int]] = [[], []]
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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_chart_plot(points: np.ndarray, primary: np.ndarray, out_path: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    n_charts = int(primary.max()) + 1
    cmap = plt.cm.get_cmap("tab20", n_charts)
    for i in range(n_charts):
        m = primary == i
        if np.any(m):
            ax.scatter(
                points[m, 0], points[m, 1], points[m, 2],
                s=1.5, color=cmap(i), alpha=0.8,
            )
    ax.set_title("Volumetric atlas — primary chart assignments (interior points)")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a volumetric rabbit atlas using SDF interior seeding (M6)"
    )
    parser.add_argument(
        "--sdf-checkpoint",
        required=True,
        help="Path to a trained SDF checkpoint (.pt). "
             "Loaded with load_sdf_net(); supports both train_sdf_rabbit.py formats.",
    )
    parser.add_argument("--n-charts", type=int, default=12)
    parser.add_argument(
        "--n-interior-samples",
        type=int,
        default=50000,
        help="Number of interior points to sample via SDF rejection for the atlas point cloud.",
    )
    parser.add_argument("--overlap-target", type=float, default=0.20)
    parser.add_argument(
        "--frame-mode",
        choices=["axis_aligned", "pca"],
        default="axis_aligned",
        help="Frame mode: axis_aligned (t1=e1,t2=e2,n=e3) or pca (from neighbour covariance).",
    )
    parser.add_argument(
        "--frame-k",
        type=int,
        default=32,
        help="Number of neighbours for PCA frame estimation (only used when --frame-mode pca).",
    )
    parser.add_argument(
        "--rejection-factor",
        type=int,
        default=8,
        help="Candidate multiplier for SDF rejection sampling.",
    )
    parser.add_argument(
        "--sdf-threshold",
        type=float,
        default=0.0,
        help="SDF acceptance threshold: points with SDF < this value are considered interior.",
    )
    parser.add_argument(
        "--bbox-half",
        type=float,
        default=0.55,
        help="Half-extent of the sampling bounding cube in SDF-normalised space.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Load SDF network
    # ------------------------------------------------------------------
    print(f"Loading SDF network from: {args.sdf_checkpoint}")
    sdf_net, sdf_center, sdf_scale = load_sdf_net(args.sdf_checkpoint, device)
    print(f"  SDF center: {sdf_center}  scale: {sdf_scale:.6f}")

    # ------------------------------------------------------------------
    # 2. Sample interior points (in SDF-normalised space)
    # ------------------------------------------------------------------
    print(
        f"Sampling {args.n_interior_samples} interior points "
        f"(bbox_half={args.bbox_half}, rejection_factor={args.rejection_factor}) ..."
    )
    interior_pts = sample_interior_points(
        sdf_net=sdf_net,
        n_target=args.n_interior_samples,
        bbox_half=args.bbox_half,
        rejection_factor=args.rejection_factor,
        threshold=args.sdf_threshold,
        device=device,
    )
    n_points = interior_pts.shape[0]
    print(f"  Interior points obtained: {n_points}")

    # Placeholder normals (zeros) — not meaningful for interior points but
    # required for NPZ backward-compatibility with train_rabbit_atlas.py
    normals_placeholder = np.zeros_like(interior_pts)

    # ------------------------------------------------------------------
    # 3. Farthest-point seeding on interior points
    # ------------------------------------------------------------------
    print(f"Selecting {args.n_charts} chart seeds via farthest-point sampling ...")
    seed_idx = farthest_point_seeds(interior_pts, n_seeds=args.n_charts)
    seed_pts  = interior_pts[seed_idx]

    # Verify seeds are inside domain
    with torch.no_grad():
        seed_sdf = sdf_net(
            torch.tensor(seed_pts, dtype=torch.float64, device=device)
        ).cpu().numpy()
    n_outside = int(np.sum(seed_sdf >= args.sdf_threshold))
    if n_outside > 0:
        print(
            f"  [WARNING] {n_outside}/{args.n_charts} seeds have SDF >= threshold "
            f"({args.sdf_threshold}). Increase --n-interior-samples or adjust seeding."
        )
    else:
        print(f"  All {args.n_charts} seeds are strictly inside the domain.")

    # ------------------------------------------------------------------
    # 4. Distance matrix → overlap membership
    # ------------------------------------------------------------------
    print("Computing chart overlap membership ...")
    dmat = np.linalg.norm(
        interior_pts[:, None, :] - seed_pts[None, :, :], axis=2
    )  # (N, n_charts)
    membership, overlap_alpha, overlap_ratio = overlap_membership(dmat, target_overlap=args.overlap_target)
    primary = np.argmin(dmat, axis=1)

    # ------------------------------------------------------------------
    # 5. Support radii (95th-percentile within-chart distance)
    # ------------------------------------------------------------------
    support_r = np.zeros((args.n_charts,), dtype=float)
    for i in range(args.n_charts):
        m = membership[:, i]
        if not np.any(m):
            support_r[i] = float(np.quantile(dmat[:, i], 0.80))
        else:
            support_r[i] = float(np.quantile(dmat[m, i], 0.95))
        support_r[i] = max(support_r[i], 1e-4)

    # ------------------------------------------------------------------
    # 6. Frame construction
    # ------------------------------------------------------------------
    print(f"Constructing frames (mode={args.frame_mode}) ...")
    if args.frame_mode == "pca":
        t1, t2, nvec = pca_frames(interior_pts, seed_pts, k=args.frame_k)
    else:
        t1, t2, nvec = axis_aligned_frames(args.n_charts)

    # ------------------------------------------------------------------
    # 7. Overlap graph & Schwarz colouring
    # ------------------------------------------------------------------
    graph = build_overlap_graph(membership.astype(bool))
    is_bip, color_map, color_groups = bipartite_or_greedy_coloring(
        graph["adj"], n_charts=args.n_charts
    )

    # ------------------------------------------------------------------
    # 8. Save NPZ (backward-compatible with surface atlas + "interior_points" key)
    # ------------------------------------------------------------------
    atlas_npz = os.path.join(args.output_dir, "rabbit_atlas_data.npz")
    np.savez_compressed(
        atlas_npz,
        # Standard keys (same as surface atlas)
        points=interior_pts,              # interior points in SDF-normalised space
        normals=normals_placeholder,      # zeros placeholder
        seed_indices=seed_idx,
        seed_points=seed_pts,
        frame_t1=t1,
        frame_t2=t2,
        frame_n=nvec,
        membership=membership.astype(np.uint8),
        primary_chart=primary.astype(np.int64),
        support_radii=support_r,
        overlap_alpha=np.asarray([overlap_alpha], dtype=float),
        center=sdf_center,               # SDF physical-space centre
        scale=np.asarray([sdf_scale], dtype=float),  # SDF physical-space scale
        # M6-specific key: signals volumetric mode to train_rabbit_atlas.py
        interior_points=interior_pts,
    )
    print(f"Atlas NPZ saved: {atlas_npz}")

    # ------------------------------------------------------------------
    # 9. Visualisation
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "atlas_primary_chart_assignment.png")
    make_chart_plot(interior_pts, primary, plot_path)

    # ------------------------------------------------------------------
    # 10. Meta JSON
    # ------------------------------------------------------------------
    meta = {
        "source": f"volumetric_sdf:{args.sdf_checkpoint}",
        "volumetric": True,
        "frame_mode": args.frame_mode,
        "n_points": int(n_points),
        "n_charts": int(args.n_charts),
        "overlap_target": float(args.overlap_target),
        "overlap_alpha": float(overlap_alpha),
        "overlap_ratio": float(overlap_ratio),
        "sdf_threshold": float(args.sdf_threshold),
        "bbox_half": float(args.bbox_half),
        "is_bipartite": bool(is_bip),
        "color_map": {str(k): int(v) for k, v in color_map.items()},
        "color_groups": [[int(x) for x in g] for g in color_groups],
        "overlap_graph": graph["adj"],
        "atlas_npz": atlas_npz,
        "plot": plot_path,
    }
    meta_path = os.path.join(args.output_dir, "rabbit_atlas_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nVolumetric atlas build complete")
    print(f"  n_interior_points : {n_points}")
    print(f"  n_charts          : {args.n_charts}")
    print(f"  frame_mode        : {args.frame_mode}")
    print(f"  overlap_ratio     : {overlap_ratio:.4f}")
    print(f"  is_bipartite      : {is_bip}")
    print(f"  atlas_npz         : {atlas_npz}")
    print(f"  meta_json         : {meta_path}")
    print(f"  chart_plot        : {plot_path}")

    # Coverage sanity check
    uncovered = int(np.sum(membership.sum(axis=1) == 0))
    if uncovered > 0:
        print(f"  [WARNING] {uncovered} interior points are not covered by any chart!")
    else:
        print(f"  Coverage check: all {n_points} interior points covered by ≥1 chart.")


if __name__ == "__main__":
    main()
