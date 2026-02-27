#!/usr/bin/env python3
"""
Export Elder-flow inverse results on the Stanford rabbit to ParaView VTK files.

The solution NPZ contains ~30 000 volumetric interior points.  All three
physical fields (pressure, concentration, velocity) are exported together with
their errors, enabling 3-D interior cross-section analysis in ParaView.

In ParaView you can:
  - Open the merged VTU, set Representation = 'Point Gaussian' or 'Sphere'.
  - Color by 'pressure', 'concentration', 'velocity_mag', or their error fields.
  - Apply 'Slice' filter (X/Y/Z) to inspect interior cross-sections.
  - Apply 'Threshold' on 'chart_id' to isolate individual atlas charts.
  - Apply 'Glyph' on 'velocity' (3-component) to show flow arrows.

Optionally a rectilinear-grid VTR is produced by nearest-neighbour
interpolation for clean Volume rendering and Slice views.

Additionally writes publication-quality convergence + parameter-recovery
figures (Matplotlib) to the --figures-dir directory.

Outputs:
    <output-dir>/<prefix>_merged.vtu         All interior points
    <output-dir>/<prefix>_chart_<id>.vtu     Per-chart subsets (12 charts)
    <output-dir>/<prefix>_grid.vtr           Regular grid (optional, --grid)
    <output-dir>/<prefix>_manifest.json

Usage:
    python postprocessing/elder_rabbit_paraview.py \\
        --run-dir  runs/rabbit_inverse_elder_globalfield_small \\
        --output-dir runs/rabbit_inverse_elder_globalfield_small/paraview2 \\
        --figures-dir figures \\
        --grid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    DOUBLE_COL_W,
    GOLDEN,
    PUB_COLORS,
    PUB_LINESTYLES,
    PUB_MARKERS,
    SINGLE_COL_W,
    set_pub_style,
    write_vtu_points,
    write_vtu_rectilinear_grid,
    interpolate_to_grid,
)

# Ground-truth permeability parameters
K0_TRUE   = 1.0
EIG_TRUE  = np.array([1.688, 1.0, 0.541])   # dominant eigenvalues


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export rabbit Elder inverse results to ParaView VTK files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        default="runs/rabbit_inverse_elder_globalfield_small",
        help="Run directory (must contain *_solution.npz and *_pressure_velocity_fields.npz)",
    )
    p.add_argument(
        "--output-dir",
        default="runs/rabbit_inverse_elder_globalfield_small/paraview2",
        help="Directory for VTU / VTR output files",
    )
    p.add_argument(
        "--prefix",
        default="rabbit_elder_inverse",
        help="Filename prefix for output files",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Also export a rectilinear grid VTR for Volume rendering in ParaView",
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
        help="Skip per-chart VTU files",
    )
    p.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for publication Matplotlib figures",
    )
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip the Matplotlib figure generation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Find NPZ files in run directory
# ---------------------------------------------------------------------------

def _find_npz(run_dir: str, suffix: str) -> Optional[str]:
    """Find the first file in run_dir ending with *suffix*."""
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(suffix):
            return os.path.join(run_dir, fn)
    return None


def _load_json(run_dir: str, suffix: str) -> Dict:
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(suffix):
            with open(os.path.join(run_dir, fn)) as fh:
                return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def export_vtk(args: argparse.Namespace) -> None:
    t0 = time.time()

    # ---- Load solution data (p, c, u fields) --------------------------------
    sol_path = _find_npz(args.run_dir, "_solution.npz")
    pv_path  = _find_npz(args.run_dir, "_pressure_velocity_fields.npz")

    if sol_path is None:
        print(f"ERROR: No *_solution.npz found in {args.run_dir}")
        return

    print(f"Loading {sol_path} ...")
    sol = np.load(sol_path, allow_pickle=True)

    points   = np.asarray(sol["points"],   dtype=float)   # (N, 3)
    chart_id = np.asarray(sol["chart_id"], dtype=np.int32).reshape(-1)
    n_points = points.shape[0]
    n_charts = int(chart_id.max()) + 1
    print(f"  N = {n_points} volumetric interior points,  n_charts = {n_charts}")

    # Build point_data ---------------------------------------------------
    point_data: Dict[str, np.ndarray] = {
        "chart_id": chart_id.astype(float),
    }

    # Pressure and concentration from solution NPZ
    for field, alias in [
        ("p_pred",     "pressure"),
        ("p_true",     "pressure_true"),
        ("p_error",    "pressure_error"),
        ("p_error_mag","pressure_error_mag"),
        ("c_pred",     "concentration"),
        ("c_true",     "concentration_true"),
        ("c_error",    "concentration_error"),
        ("c_error_mag","concentration_error_mag"),
    ]:
        if field in sol:
            point_data[field]  = np.asarray(sol[field], dtype=float).reshape(-1)
            point_data[alias]  = point_data[field]

    # Velocity: prefer pressure_velocity_fields.npz if available
    if pv_path is not None:
        print(f"Loading {pv_path} ...")
        pv = np.load(pv_path, allow_pickle=True)
        for field, alias in [
            ("velocity_pred",      "velocity"),
            ("velocity_true",      "velocity_true"),
            ("velocity_error",     "velocity_error"),
            ("velocity_error_mag", "velocity_error_mag"),
        ]:
            if field in pv:
                arr = np.asarray(pv[field], dtype=float)
                point_data[field] = arr
                point_data[alias] = arr
        # Also scalar velocity magnitude
        if "velocity_pred" in pv:
            vmag = np.linalg.norm(np.asarray(pv["velocity_pred"], dtype=float), axis=1)
            point_data["velocity_mag"] = vmag
        if "velocity_error_mag" in pv:
            point_data["velocity_error_mag"] = np.asarray(pv["velocity_error_mag"], dtype=float).reshape(-1)
    elif "u_pred" in sol:
        # Fall back to u in solution NPZ
        for field, alias in [
            ("u_pred",     "velocity"),
            ("u_true",     "velocity_true"),
            ("u_error",    "velocity_error"),
            ("u_error_mag","velocity_error_mag"),
        ]:
            if field in sol:
                arr = np.asarray(sol[field], dtype=float)
                point_data[field] = arr
                point_data[alias] = arr
        if "u_pred" in sol:
            vmag = np.linalg.norm(np.asarray(sol["u_pred"], dtype=float), axis=1)
            point_data["velocity_mag"] = vmag

    # Optional extra fields
    for key in ("detail_score", "high_detail_mask"):
        if key in sol:
            point_data[key] = np.asarray(sol[key], dtype=float).reshape(-1)

    # ---- Write merged VTU -------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, f"{args.prefix}_merged.vtu")
    print(f"Writing merged VTU: {merged_path}")
    write_vtu_points(merged_path, points, point_data)

    # ---- Write per-chart VTUs ---------------------------------------------
    per_chart_paths: List[str] = []
    if not args.no_per_chart:
        for cid in range(n_charts):
            mask = chart_id == cid
            if mask.sum() == 0:
                continue
            cdata = {}
            for k, v in point_data.items():
                v = np.asarray(v)
                if v.shape[0] == n_points:
                    cdata[k] = v[mask]
                else:
                    cdata[k] = v  # broadcast / metadata
            cpath = os.path.join(args.output_dir, f"{args.prefix}_chart_{cid:02d}.vtu")
            print(f"  Chart {cid:02d}: {mask.sum()} pts → {cpath}")
            write_vtu_points(cpath, points[mask], {k: v for k, v in cdata.items() if np.asarray(v).shape[0] == mask.sum()})
            per_chart_paths.append(cpath)

    # ---- Optional rectilinear grid ----------------------------------------
    grid_path: Optional[str] = None
    if args.grid:
        scalar_names = [
            "pressure", "pressure_error_mag",
            "concentration", "concentration_error_mag",
            "velocity_mag", "velocity_error_mag",
            "chart_id",
        ]
        scalar_fields = {k: np.asarray(v).reshape(-1)
                         for k, v in point_data.items()
                         if k in scalar_names and np.asarray(v).ndim == 1}
        print(f"Interpolating to {args.grid_nx}×{args.grid_ny}×{args.grid_nz} grid ...")
        xg, yg, zg, gf = interpolate_to_grid(
            points, scalar_fields,
            nx=args.grid_nx, ny=args.grid_ny, nz=args.grid_nz,
            method="nearest",
        )
        grid_path = os.path.join(args.output_dir, f"{args.prefix}_grid.vtr")
        print(f"Writing VTR: {grid_path}")
        write_vtu_rectilinear_grid(grid_path, xg, yg, zg, gf)

    # ---- Manifest ---------------------------------------------------------
    manifest = {
        "solution_npz":    os.path.abspath(sol_path) if sol_path else None,
        "pv_npz":          os.path.abspath(pv_path)  if pv_path  else None,
        "n_points":        n_points,
        "n_charts":        n_charts,
        "merged_vtu":      os.path.abspath(merged_path),
        "per_chart_vtu":   [os.path.abspath(p) for p in per_chart_paths],
        "grid_vtr":        os.path.abspath(grid_path) if grid_path else None,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {manifest_path}")

    elapsed = time.time() - t0
    print(f"\nVTK export done in {elapsed:.1f}s")
    print("\nParaView tips:")
    print("  1. Open the merged VTU.  Color by 'pressure' or 'concentration'.")
    print("  2. Apply 'Slice' (X/Y/Z) to see interior cross-sections.")
    print("  3. Apply 'Glyph' on 'velocity' (3-component) for flow arrows.")
    print("  4. 'Threshold' on 'chart_id' to isolate individual charts.")
    if grid_path:
        print(f"  5. Open {os.path.basename(grid_path)} → Volume renderer for pressure/concentration.")


# ---------------------------------------------------------------------------
# Publication figures (convergence + parameter recovery)
# ---------------------------------------------------------------------------

def make_figures(run_dir: str, figures_dir: str) -> None:
    """Generate convergence and parameter-recovery figures for the Elder run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_pub_style(fontsize=9, linewidth=1.5)

    hist  = _load_json(run_dir, "_history.json")
    metr  = _load_json(run_dir, "_metrics.json")

    if not hist:
        print(f"  No history JSON found in {run_dir} — skipping figures.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    iters = np.arange(1, len(hist.get("flow", [])) + 1)

    # ---- Figure 1: loss convergence (2x2) ---------------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 1.1),
        constrained_layout=True,
    )
    axes = axes.ravel()

    panels = [
        ("flow",        "Flow (Darcy) PDE loss",           "log",    None),
        ("trans",       "Transport PDE loss",               "log",    None),
        ("if_val",      "Interface value residual",         "log",    None),
        ("if_flux",     "Interface flux residual",          "log",    None),
    ]
    for pi, (key, ylabel, yscale, _) in enumerate(panels):
        ax = axes[pi]
        if key in hist:
            vals = np.asarray(hist[key], dtype=float)
            ax.plot(iters[:len(vals)], vals,
                    color=PUB_COLORS[0], linewidth=1.5)
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.text(0.02, 0.97, f"({chr(ord('a') + pi)})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold")

    fig.suptitle("Elder inverse on rabbit — Schwarz convergence",
                 fontsize=10, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir, f"elder_rabbit_convergence.{ext}"), dpi=300)
        print(f"Saved: {os.path.join(figures_dir, f'elder_rabbit_convergence.{ext}')}")
    plt.close(fig)

    # ---- Figure 2: parameter recovery (2x2) --------------------------------
    fig2, axes2 = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 1.1),
        constrained_layout=True,
    )
    axes2 = axes2.ravel()

    param_panels = [
        ("k0",           "Permeability $k_0$ estimate",     "linear", K0_TRUE),
        ("eig1",         "Eigenvalue $\\lambda_1$ estimate","linear", EIG_TRUE[0]),
        ("eig2",         "Eigenvalue $\\lambda_2$ estimate","linear", EIG_TRUE[1]),
        ("eig3",         "Eigenvalue $\\lambda_3$ estimate","linear", EIG_TRUE[2]),
    ]
    for pi, (key, ylabel, yscale, true_val) in enumerate(param_panels):
        ax = axes2[pi]
        if key in hist:
            vals = np.asarray(hist[key], dtype=float)
            ax.plot(iters[:len(vals)], vals,
                    color=PUB_COLORS[0], linewidth=1.5, label="Estimate")
            ax.axhline(true_val, color="black", linestyle="--",
                       linewidth=0.9, alpha=0.6,
                       label=f"True = {true_val:.3g}")
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.legend(loc="best", fontsize=7)
        ax.text(0.02, 0.97, f"({chr(ord('a') + pi)})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold")

    fig2.suptitle("Elder inverse on rabbit — permeability parameter recovery",
                  fontsize=10, fontweight="bold")
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(figures_dir, f"elder_rabbit_parameters.{ext}"), dpi=300)
        print(f"Saved: {os.path.join(figures_dir, f'elder_rabbit_parameters.{ext}')}")
    plt.close(fig2)

    # ---- Figure 3: error summary bar chart ---------------------------------
    fig3, ax3 = plt.subplots(
        figsize=(SINGLE_COL_W * 1.3, SINGLE_COL_W * GOLDEN * 1.6),
        constrained_layout=True,
    )

    # Collect final errors from metrics
    bar_labels = []
    bar_vals   = []
    bar_colors = []

    err_map = {
        "k0_rel_err":        ("$k_0$ rel. err.",    PUB_COLORS[0]),
        "eig_rel_mean":      ("Eig. rel. (mean)",   PUB_COLORS[1]),
        "axis_deg_mean":     ("Axis angle (°, mean)",PUB_COLORS[2]),
    }
    # Also pull from history last value
    for hist_key, (lbl, col) in err_map.items():
        val = None
        for mk in (hist_key, hist_key.replace("_err", "").replace("_mean", "")):
            if mk in metr:
                val = float(metr[mk])
                break
        if val is None and hist_key in hist:
            arr = np.asarray(hist[hist_key])
            val = float(arr[-1])
        if val is not None:
            bar_labels.append(lbl)
            bar_vals.append(val * 100.0 if "deg" not in hist_key else val)
            bar_colors.append(col)

    if bar_vals:
        bars = ax3.bar(range(len(bar_vals)), bar_vals,
                       color=bar_colors, edgecolor="white", linewidth=0.5)
        for rect, val in zip(bars, bar_vals):
            ax3.annotate(
                f"{val:.3g}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )
        ax3.set_xticks(range(len(bar_labels)))
        ax3.set_xticklabels(bar_labels, rotation=15, ha="right")
        ax3.set_ylabel("Error (%, or degrees)")
        ax3.set_title("Elder inverse — final parameter errors",
                      fontsize=9, fontweight="bold")
        for ext in ("pdf", "png"):
            fig3.savefig(os.path.join(figures_dir, f"elder_rabbit_error_summary.{ext}"), dpi=300)
            print(f"Saved: {os.path.join(figures_dir, f'elder_rabbit_error_summary.{ext}')}")
    plt.close(fig3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # VTK export
    export_vtk(args)

    # Publication figures
    if not args.skip_figures:
        print(f"\nGenerating publication figures → {args.figures_dir}")
        make_figures(args.run_dir, args.figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
