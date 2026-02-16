#!/usr/bin/env python3
"""
Adaptive atlas refinement by splitting high-score charts into smaller charts.

This script keeps the original atlas format and emits:
- rabbit_atlas_data_refined.npz
- rabbit_atlas_meta_refined.json
- atlas_split_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)
for path in (EXPERIMENTS_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_rabbit_atlas_poissondisk import (
    bipartite_or_greedy_coloring,
    build_overlap_graph,
    chart_frames,
    farthest_point_seeds,
    overlap_membership,
)


torch.set_default_dtype(torch.float64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_chart_scores(
    n_charts: int,
    chart_sizes: np.ndarray,
    chart_score_json: Optional[str],
    baseline_metrics: Optional[str],
) -> Tuple[np.ndarray, str]:
    if chart_score_json is not None and os.path.isfile(chart_score_json):
        with open(chart_score_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if isinstance(payload.get("chart_scores"), list):
                arr = np.asarray(payload["chart_scores"], dtype=float).reshape(-1)
                if arr.shape[0] == n_charts:
                    return arr, "chart_score_json.chart_scores"
            if isinstance(payload.get("naPINN"), list):
                arr = np.asarray(payload["naPINN"], dtype=float).reshape(-1)
                if arr.shape[0] == n_charts:
                    return arr, "chart_score_json.naPINN"
            if isinstance(payload.get("per_chart"), list):
                arr = np.zeros((n_charts,), dtype=float)
                used = 0
                for row in payload["per_chart"]:
                    cid = int(row.get("chart_id", -1))
                    if 0 <= cid < n_charts:
                        v = row.get("naPINN")
                        if v is None:
                            v = row.get("score")
                        if v is None:
                            continue
                        arr[cid] = float(v)
                        used += 1
                if used > 0:
                    return arr, "chart_score_json.per_chart"

    if baseline_metrics is not None and os.path.isfile(baseline_metrics):
        with open(baseline_metrics, "r", encoding="utf-8") as f:
            payload = json.load(f)
        pcs = payload.get("per_chart")
        if isinstance(pcs, list):
            arr = np.zeros((n_charts,), dtype=float)
            used = 0
            med_sz = float(np.median(chart_sizes[chart_sizes > 0])) if np.any(chart_sizes > 0) else 1.0
            med_sz = max(med_sz, 1.0)
            for row in pcs:
                cid = int(row.get("chart_id", -1))
                if 0 <= cid < n_charts:
                    rel = row.get("relative_l2_error")
                    if rel is None:
                        continue
                    npts = float(row.get("n_points", chart_sizes[cid]))
                    # Proxy naPINN: per-chart rel-L2 with mild chart-size scaling.
                    scale = (max(npts, 1.0) / med_sz) ** 0.20
                    arr[cid] = float(rel) * scale
                    used += 1
            if used > 0:
                return arr, "baseline_metrics.per_chart_relative_l2_proxy"

    # Fallback deterministic score: prefer larger charts first.
    arr = chart_sizes.astype(float).copy()
    return arr, "fallback_chart_size"


def choose_split_parents(
    chart_scores: np.ndarray,
    chart_sizes: np.ndarray,
    k_split: int,
    min_points_to_split: int,
    n_charts_max: int,
    n_children_per_parent: int,
) -> List[int]:
    n_charts = chart_scores.shape[0]
    eligible = [i for i in range(n_charts) if int(chart_sizes[i]) >= int(min_points_to_split)]
    if not eligible:
        return []

    # Splitting one parent into c children increases chart count by (c-1).
    max_new = max(0, int(n_charts_max) - int(n_charts))
    if n_children_per_parent <= 1:
        max_parents = 0
    else:
        max_parents = max_new // (n_children_per_parent - 1)
    if max_parents <= 0:
        return []

    k_eff = max(0, min(int(k_split), len(eligible), max_parents))
    if k_eff <= 0:
        return []

    ranked = sorted(eligible, key=lambda i: float(chart_scores[i]), reverse=True)
    return ranked[:k_eff]


def remap_with_splits(
    points: np.ndarray,
    normals: np.ndarray,
    membership: np.ndarray,
    seed_points: np.ndarray,
    support_radii: np.ndarray,
    split_parents: List[int],
    n_children_per_parent: int,
    split_radius_scale: float,
    frame_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    n_points, n_charts = membership.shape
    parent_set = set(split_parents)

    keep_ids = [i for i in range(n_charts) if i not in parent_set]
    new_seed_idx: List[int] = []
    new_seed_pts: List[np.ndarray] = []
    new_support_r: List[float] = []
    new_parent_of_chart: List[int] = []
    old_to_new: Dict[int, List[int]] = {i: [] for i in range(n_charts)}

    # Keep charts.
    for i in keep_ids:
        new_idx = len(new_seed_idx)
        # Choose nearest point index to preserve seed index field.
        s = seed_points[i]
        d = np.linalg.norm(points - s[None, :], axis=1)
        sid = int(np.argmin(d))
        new_seed_idx.append(sid)
        new_seed_pts.append(points[sid].copy())
        new_support_r.append(float(support_radii[i]))
        new_parent_of_chart.append(int(i))
        old_to_new[i].append(new_idx)

    # Split parent charts.
    for p in split_parents:
        idx = np.where(membership[:, p])[0]
        if idx.size == 0:
            continue
        local_pts = points[idx]
        n_child = int(n_children_per_parent)
        if idx.size < n_child:
            # Duplicate with replacement if support is tiny.
            pick_local = np.random.choice(np.arange(idx.size), size=n_child, replace=True)
        else:
            pick_local = farthest_point_seeds(local_pts, n_child)
        child_seed_idx = idx[pick_local]
        for sid in child_seed_idx.tolist():
            new_idx = len(new_seed_idx)
            new_seed_idx.append(int(sid))
            new_seed_pts.append(points[sid].copy())
            new_support_r.append(float(max(1e-4, support_radii[p] * split_radius_scale)))
            new_parent_of_chart.append(int(p))
            old_to_new[p].append(new_idx)

    seed_idx_arr = np.asarray(new_seed_idx, dtype=np.int64)
    seed_pts_arr = np.asarray(new_seed_pts, dtype=float)
    support_r_arr = np.asarray(new_support_r, dtype=float)

    # Rebuild frames on refined seeds.
    t1, t2, nvec = chart_frames(points, normals, seed_idx_arr, k_neighbors=frame_k)

    # Rebuild membership from distances to refined seeds.
    dmat = np.linalg.norm(points[:, None, :] - seed_pts_arr[None, :, :], axis=2)
    membership_ref, overlap_alpha, overlap_ratio = overlap_membership(dmat, target_overlap=0.20)
    primary = np.argmin(dmat, axis=1)

    # Recompute support radii from refined memberships.
    n_ref = membership_ref.shape[1]
    support_r_ref = np.zeros((n_ref,), dtype=float)
    for i in range(n_ref):
        m = membership_ref[:, i]
        if np.any(m):
            support_r_ref[i] = float(np.quantile(dmat[m, i], 0.95))
        else:
            support_r_ref[i] = float(np.quantile(dmat[:, i], 0.80))
        support_r_ref[i] = max(support_r_ref[i], 1e-4)

    graph = build_overlap_graph(membership_ref.astype(bool))
    is_bip, color_map, color_groups = bipartite_or_greedy_coloring(graph["adj"], n_charts=n_ref)

    remap = {
        "old_to_new": {str(k): [int(x) for x in v] for k, v in old_to_new.items()},
        "new_parent": [int(x) for x in new_parent_of_chart],
        "split_parents": [int(x) for x in split_parents],
        "keep_ids": [int(x) for x in keep_ids],
        "n_old_charts": int(n_charts),
        "n_new_charts": int(n_ref),
        "overlap_alpha": float(overlap_alpha),
        "overlap_ratio": float(overlap_ratio),
        "is_bipartite": bool(is_bip),
        "color_groups": [[int(x) for x in g] for g in color_groups],
        "overlap_graph": graph["adj"],
    }

    return (
        seed_idx_arr,
        seed_pts_arr,
        t1,
        t2,
        nvec,
        membership_ref.astype(np.uint8),
        primary.astype(np.int64),
        support_r_ref,
        remap,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build refined rabbit atlas by splitting high-score charts")
    parser.add_argument("--atlas-data", required=True, help="Path to source rabbit_atlas_data.npz")
    parser.add_argument("--atlas-meta", default=None, help="Optional source rabbit_atlas_meta.json")
    parser.add_argument("--baseline-metrics", default=None, help="Optional baseline Poisson metrics JSON")
    parser.add_argument("--chart-score-json", default=None, help="Optional explicit chart score JSON")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--k-split", type=int, default=2)
    parser.add_argument("--n-children-per-parent", type=int, default=2)
    parser.add_argument("--min-points-to-split", type=int, default=1200)
    parser.add_argument("--split-radius-scale", type=float, default=0.72)
    parser.add_argument("--max-charts", type=int, default=20)
    parser.add_argument("--frame-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.atlas_data)
    points = np.asarray(data["points"], dtype=float)
    normals = np.asarray(data["normals"], dtype=float)
    seed_points = np.asarray(data["seed_points"], dtype=float)
    membership = np.asarray(data["membership"]).astype(bool)
    support_radii = np.asarray(data["support_radii"], dtype=float)

    n_points, n_charts = membership.shape
    chart_sizes = membership.sum(axis=0).astype(int)
    chart_scores, score_source = load_chart_scores(
        n_charts=n_charts,
        chart_sizes=chart_sizes,
        chart_score_json=args.chart_score_json,
        baseline_metrics=args.baseline_metrics,
    )

    split_parents = choose_split_parents(
        chart_scores=chart_scores,
        chart_sizes=chart_sizes,
        k_split=args.k_split,
        min_points_to_split=args.min_points_to_split,
        n_charts_max=args.max_charts,
        n_children_per_parent=args.n_children_per_parent,
    )

    if not split_parents:
        print("No eligible charts selected for splitting.")
        print("Writing passthrough refined atlas artifacts.")
        # Passthrough save.
        out_npz = os.path.join(args.output_dir, "rabbit_atlas_data_refined.npz")
        np.savez_compressed(
            out_npz,
            **{k: data[k] for k in data.files},
        )
        split_map = {
            "split_skipped": True,
            "reason": "no_eligible_chart",
            "score_source": score_source,
            "chart_scores": chart_scores.tolist(),
            "chart_sizes": chart_sizes.tolist(),
        }
        split_map_path = os.path.join(args.output_dir, "atlas_split_map.json")
        with open(split_map_path, "w", encoding="utf-8") as f:
            json.dump(split_map, f, indent=2)
        meta = {
            "source_atlas_data": args.atlas_data,
            "split_skipped": True,
            "score_source": score_source,
            "n_old_charts": int(n_charts),
            "n_new_charts": int(n_charts),
        }
        meta_path = os.path.join(args.output_dir, "rabbit_atlas_meta_refined.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  refined_atlas: {out_npz}")
        print(f"  split_map:     {split_map_path}")
        print(f"  meta:          {meta_path}")
        return

    (
        seed_idx_ref,
        seed_pts_ref,
        t1_ref,
        t2_ref,
        n_ref,
        membership_ref,
        primary_ref,
        support_r_ref,
        remap,
    ) = remap_with_splits(
        points=points,
        normals=normals,
        membership=membership,
        seed_points=seed_points,
        support_radii=support_radii,
        split_parents=split_parents,
        n_children_per_parent=args.n_children_per_parent,
        split_radius_scale=args.split_radius_scale,
        frame_k=args.frame_k,
    )

    out_npz = os.path.join(args.output_dir, "rabbit_atlas_data_refined.npz")
    payload = {k: data[k] for k in data.files if k not in {
        "seed_indices",
        "seed_points",
        "frame_t1",
        "frame_t2",
        "frame_n",
        "membership",
        "primary_chart",
        "support_radii",
    }}
    payload.update(
        {
            "seed_indices": seed_idx_ref,
            "seed_points": seed_pts_ref,
            "frame_t1": t1_ref,
            "frame_t2": t2_ref,
            "frame_n": n_ref,
            "membership": membership_ref.astype(np.uint8),
            "primary_chart": primary_ref.astype(np.int64),
            "support_radii": support_r_ref.astype(float),
        }
    )
    np.savez_compressed(out_npz, **payload)

    split_map = {
        "split_skipped": False,
        "score_source": score_source,
        "chart_scores": chart_scores.tolist(),
        "chart_sizes": chart_sizes.tolist(),
        **remap,
    }
    split_map_path = os.path.join(args.output_dir, "atlas_split_map.json")
    with open(split_map_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)

    meta = {
        "source_atlas_data": args.atlas_data,
        "source_atlas_meta": args.atlas_meta,
        "score_source": score_source,
        "split_parents": [int(x) for x in split_parents],
        "n_children_per_parent": int(args.n_children_per_parent),
        "min_points_to_split": int(args.min_points_to_split),
        "split_radius_scale": float(args.split_radius_scale),
        "n_old_charts": int(n_charts),
        "n_new_charts": int(membership_ref.shape[1]),
        "overlap_ratio": float(remap["overlap_ratio"]),
    }
    meta_path = os.path.join(args.output_dir, "rabbit_atlas_meta_refined.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Adaptive split atlas build complete")
    print(f"  score_source:  {score_source}")
    print(f"  split_parents: {split_parents}")
    print(f"  old_charts:    {n_charts}")
    print(f"  new_charts:    {membership_ref.shape[1]}")
    print(f"  refined_atlas: {out_npz}")
    print(f"  split_map:     {split_map_path}")
    print(f"  meta:          {meta_path}")


if __name__ == "__main__":
    main()
