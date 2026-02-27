#!/usr/bin/env python3
"""
Export Poisson-on-rabbit atlas results to ParaView-compatible VTK files.

The solution NPZ contains ~50 000 volumetric interior points (not just the
surface), so the resulting VTU files give full 3-D interior field access.
In ParaView you can:
  - Render the raw point cloud with "Point Gaussian" or "Sphere" glyphs.
  - Apply the "Slice" filter (X / Y / Z planes) to inspect interior cross-
    sections of the pressure field and error.
  - Apply "Threshold" on the `chart_id` field to isolate individual charts.
  - Apply "Glyph" on gradient (velocity) vectors.

Optionally an axis-aligned rectilinear-grid VTR file is produced by
nearest-neighbour interpolation, which allows the "Volume" renderer and
cleaner slice views.

Outputs (written to --output-dir):
    <prefix>_merged.vtu          All interior + evaluation points (merged)
    <prefix>_chart_<id>.vtu      Per-chart subset (12 files for 12-chart atlas)
    <prefix>_grid.vtr            Regular Cartesian grid (optional, --grid)
    <prefix>_manifest.json       Paths + metrics summary

Usage:
    python postprocessing/poisson_rabbit_paraview.py \\
        --solution-npz runs/attempt20c_compact/rabbit_poisson_schwarz_attempt20c_compact_solution.npz \\
        --output-dir  runs/attempt20c_compact/paraview \\
        --prefix      rabbit_poisson_compact \\
        --grid                      # also write rectilinear grid VTR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import write_vtu_points, write_vtu_rectilinear_grid, interpolate_to_grid


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export rabbit Poisson atlas results to ParaView VTK files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--solution-npz",
        required=True,
        help="Path to *_solution.npz produced by run_poisson_rabbit_atlas_schwarz.py",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for VTU / VTR output files",
    )
    p.add_argument(
        "--prefix",
        default="rabbit_poisson",
        help="Filename prefix for all output files",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Also export a regular Cartesian rectilinear grid (.vtr) suitable "
             "for Volume rendering and clean cross-section slices in ParaView",
    )
    p.add_argument(
        "--grid-nx", type=int, default=80, help="Grid resolution along X (for --grid)"
    )
    p.add_argument(
        "--grid-ny", type=int, default=80, help="Grid resolution along Y (for --grid)"
    )
    p.add_argument(
        "--grid-nz", type=int, default=80, help="Grid resolution along Z (for --grid)"
    )
    p.add_argument(
        "--no-per-chart",
        action="store_true",
        help="Skip writing per-chart VTU files (saves time for large runs)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ---- Load solution data -----------------------------------------------
    print(f"Loading {args.solution_npz} ...")
    data = np.load(args.solution_npz, allow_pickle=True)

    points         = np.asarray(data["points"],           dtype=float)   # (N, 3)
    u_pred         = np.asarray(data["u_pred"],           dtype=float).reshape(-1)
    u_true         = np.asarray(data["u_true"],           dtype=float).reshape(-1)
    u_error        = np.asarray(data["u_error"],          dtype=float).reshape(-1)
    u_error_mag    = np.asarray(data["u_error_mag"],      dtype=float).reshape(-1)
    chart_id       = np.asarray(data["chart_id"],         dtype=np.int32).reshape(-1)
    blend_weight   = np.asarray(data["blend_weight"],     dtype=float).reshape(-1)
    interface_res  = np.asarray(data["interface_residual"],dtype=float).reshape(-1)

    n_points = points.shape[0]
    n_charts = int(chart_id.max()) + 1
    print(f"  N = {n_points} interior points,  n_charts = {n_charts}")

    # Optional extra fields
    extra: Dict[str, np.ndarray] = {}
    for key in ("detail_score", "high_detail_mask"):
        if key in data:
            extra[key] = np.asarray(data[key], dtype=float).reshape(-1)

    # ---- Derived fields ---------------------------------------------------
    # Relative pointwise error (clamped to avoid division by zero)
    u_true_abs = np.abs(u_true)
    u_rel_err = u_error_mag / np.maximum(u_true_abs, 1e-12 * u_true_abs.max() + 1e-20)

    # ---- Build point_data dict for merged file ----------------------------
    point_data: Dict[str, np.ndarray] = {
        # Solution
        "u_pred":             u_pred,
        "u_true":             u_true,
        # Error
        "u_error":            u_error,
        "u_error_mag":        u_error_mag,
        "u_rel_error":        u_rel_err,
        # Atlas metadata
        "chart_id":           chart_id.astype(float),
        "blend_weight":       blend_weight,
        "interface_residual": interface_res,
    }
    point_data.update(extra)

    # ---- Write merged VTU -------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, f"{args.prefix}_merged.vtu")
    print(f"Writing merged VTU: {merged_path}")
    write_vtu_points(merged_path, points, point_data)

    # ---- Write per-chart VTUs ---------------------------------------------
    per_chart_paths: list[str] = []
    if not args.no_per_chart:
        for cid in range(n_charts):
            mask = chart_id == cid
            if mask.sum() == 0:
                continue
            chart_data = {k: v[mask] for k, v in point_data.items()}
            cpath = os.path.join(args.output_dir, f"{args.prefix}_chart_{cid:02d}.vtu")
            print(f"  Chart {cid:02d}: {mask.sum()} pts → {cpath}")
            write_vtu_points(cpath, points[mask], chart_data)
            per_chart_paths.append(cpath)

    # ---- Optional rectilinear grid (for Volume / Slice rendering) ---------
    grid_path: Optional[str] = None
    if args.grid:
        print(
            f"Interpolating to {args.grid_nx}×{args.grid_ny}×{args.grid_nz} grid "
            "(nearest-neighbour) ..."
        )
        scalar_fields = {
            "u_pred":       u_pred,
            "u_true":       u_true,
            "u_error":      u_error,
            "u_error_mag":  u_error_mag,
            "u_rel_error":  u_rel_err,
            "chart_id":     chart_id.astype(float),
            "blend_weight": blend_weight,
        }
        xg, yg, zg, gf = interpolate_to_grid(
            points, scalar_fields,
            nx=args.grid_nx, ny=args.grid_ny, nz=args.grid_nz,
            method="nearest",
        )
        grid_path = os.path.join(args.output_dir, f"{args.prefix}_grid.vtr")
        print(f"Writing rectilinear grid VTR: {grid_path}")
        write_vtu_rectilinear_grid(grid_path, xg, yg, zg, gf)

    # ---- Compute summary metrics ------------------------------------------
    rel_l2 = float(
        np.sqrt(np.mean(u_error ** 2)) / (np.sqrt(np.mean(u_true ** 2)) + 1e-30)
    )
    max_err = float(u_error_mag.max())

    # ---- Write manifest ---------------------------------------------------
    manifest = {
        "solution_npz":   os.path.abspath(args.solution_npz),
        "n_points":       n_points,
        "n_charts":       n_charts,
        "rel_l2_error":   rel_l2,
        "max_error":      max_err,
        "merged_vtu":     os.path.abspath(merged_path),
        "per_chart_vtu":  [os.path.abspath(p) for p in per_chart_paths],
        "grid_vtr":       os.path.abspath(grid_path) if grid_path else None,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {manifest_path}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s.  rel_L2 = {rel_l2 * 100:.3f}%,  "
        f"max_err = {max_err:.4f}"
    )
    print("\nParaView tips:")
    print("  1. Open the merged VTU.  Set Representation = 'Point Gaussian' or 'Sphere'.")
    print("  2. Color by 'u_error_mag' with a diverging colormap to see error hot-spots.")
    print("  3. Add 'Slice' filter (X/Y/Z normal) to see interior cross-sections.")
    print("  4. Use 'Threshold' on 'chart_id' to highlight individual atlas charts.")
    if grid_path:
        print(f"  5. Open {os.path.basename(grid_path)} and set Representation = 'Volume'")
        print("     for full 3-D volumetric rendering of the interior pressure field.")


if __name__ == "__main__":
    main()
