#!/usr/bin/env python3
"""
Export inverse neo-Hookean results on the torus to ParaView-compatible VTK files.

Background
──────────
The torus problem uses an 8-chart atlas to represent the torus surface and solve
the inverse neo-Hookean problem (recovering shear modulus μ and bulk modulus K).

Three solver variants are compared:
  1. Atlas direct PINN            (torus_inverse_mps_dense_v4)
  2. Schwarz dual — displacement  (torus_schwarz_dual_accurate_displacement_*)
  3. Schwarz dual — traction      (torus_schwarz_dual_accurate_traction_*)

The torus coordinates are already in physical space (major radius R ≈ 1,
minor radius r ≈ 0.35), so no SDF denormalization is needed.

Exported fields (surface boundary VTU):
    traction_true / traction_pred / traction_error  — traction vectors (3-comp)
    traction_error_mag                               — |error| scalar
    displacement / displacement_mag                  — boundary displacement
    chart_id                                         — atlas chart assignment
    (deformed_x, _y, _z)                             — deformed node positions

The dense_v4 NPZ also contains per-chart subsets (chart_00 … chart_07) with
the full chart-support overlap visible.

Existing VTU files in the run directories:
  • atlas run: *_boundary_full_chart_NN.vtu, *_boundary_full_error.vtu
  • Schwarz run: *_best_boundary_fields.vtu
These are already in physical space.  This script supplements them by writing
a single merged VTU from the NPZ for easy side-by-side comparison.

Outputs (written to --output-dir):
    <prefix>_surface_merged.vtu         Merged surface (all 16k points)
    <prefix>_surface_chart_<id>.vtu     Per-chart surface subsets
    <prefix>_surface_deformed.vtu       Deformed surface (if u_full present)
    <prefix>_manifest.json

Usage (defaults target the best atlas run):
    python postprocessing/torus_inverse_paraview.py

    # Custom paths:
    python postprocessing/torus_inverse_paraview.py \\
        --atlas-dir   runs/torus_inverse_mps_dense_v4 \\
        --disp-dir    runs/torus_schwarz_dual_accurate_displacement_20260215_140200 \\
        --tract-dir   runs/torus_schwarz_dual_accurate_traction_20260215_140800 \\
        --output-dir  paraview/torus_inverse \\
        --prefix      torus_inverse
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
    write_vtu_points,
    interpolate_to_grid,
    write_vtu_rectilinear_grid,
    DOUBLE_COL_W,
    GOLDEN,
    PUB_COLORS,
    PUB_LINESTYLES,
    PUB_MARKERS,
    set_pub_style,
)

# ---- Defaults ---------------------------------------------------------------
_DEFAULT_ATLAS_DIR = "runs/torus_inverse_mps_dense_v4"
_DEFAULT_DISP_DIR  = "runs/torus_schwarz_dual_accurate_displacement_20260215_140200"
_DEFAULT_TRACT_DIR = "runs/torus_schwarz_dual_accurate_traction_20260215_140800"
_DEFAULT_OUTPUT    = "paraview/torus_inverse"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export torus inverse neo-Hookean results to ParaView VTK files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--atlas-dir",  default=_DEFAULT_ATLAS_DIR,
                   help="Run directory for the atlas direct solve (contains *_obs.npz)")
    p.add_argument("--disp-dir",   default=_DEFAULT_DISP_DIR,
                   help="Run directory for Schwarz displacement-BC solve")
    p.add_argument("--tract-dir",  default=_DEFAULT_TRACT_DIR,
                   help="Run directory for Schwarz traction-BC solve")
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT,
                   help="Directory for VTU output files")
    p.add_argument("--prefix",     default="torus_inverse",
                   help="Filename prefix for output files")
    p.add_argument("--no-per-chart", action="store_true",
                   help="Skip per-chart VTU files")
    p.add_argument("--figures-dir", default="figures",
                   help="Directory for optional Matplotlib summary figure")
    p.add_argument("--skip-figures", action="store_true",
                   help="Skip the Matplotlib figure generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_npz(directory: str, suffix: str) -> Optional[str]:
    """Find the first *.npz file in *directory* whose name ends with *suffix*."""
    if not os.path.isdir(directory):
        return None
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(suffix):
            return os.path.join(directory, fn)
    return None


def _load_json(directory: str, suffix: str) -> Dict:
    if not os.path.isdir(directory):
        return {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(suffix):
            with open(os.path.join(directory, fn)) as fh:
                return json.load(fh)
    return {}


def _chart_id_from_weights(chart_weights: np.ndarray) -> np.ndarray:
    """Return the dominant chart index for each point (argmax of blend weights)."""
    return np.argmax(chart_weights, axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Export atlas run
# ---------------------------------------------------------------------------

def export_atlas(args: argparse.Namespace) -> Dict:
    """Export the dense atlas NPZ to VTU files.  Returns manifest sub-dict."""
    t0 = time.time()
    npz_path = _find_npz(args.atlas_dir, "_obs.npz")
    if npz_path is None:
        print(f"  WARNING: No *_obs.npz found in {args.atlas_dir} — skipping atlas export.")
        return {}

    print(f"\n=== Atlas export ===")
    print(f"Loading {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)

    # Use full boundary if available, else fall back to eval set
    if "x_full" in d:
        pts       = np.asarray(d["x_full"],       dtype=float)
        normals   = np.asarray(d["n_full"],        dtype=float)
        t_true    = np.asarray(d["t_full_true"],   dtype=float)
        t_pred    = np.asarray(d["t_full_pred"],   dtype=float)
        t_err     = np.asarray(d["t_full_error"],  dtype=float)
        cw        = np.asarray(d["chart_weights_full"], dtype=float)
        tag       = "full"
    else:
        pts       = np.asarray(d["x_eval"],        dtype=float)
        normals   = np.asarray(d["n_eval"],        dtype=float)
        t_true    = np.asarray(d["t_eval_true"],   dtype=float)
        t_pred    = np.asarray(d["t_eval_pred"],   dtype=float)
        t_err     = t_pred - t_true
        cw        = np.asarray(d["chart_weights_eval"], dtype=float)
        tag       = "eval"

    n_pts    = pts.shape[0]
    n_charts = cw.shape[1]
    chart_id = _chart_id_from_weights(cw)

    t_err_mag = np.linalg.norm(t_err, axis=1)
    t_true_mag = np.linalg.norm(t_true, axis=1)
    t_rel_err  = t_err_mag / np.maximum(t_true_mag, 1e-12 * t_true_mag.max() + 1e-20)

    print(f"  N = {n_pts} boundary points,  n_charts = {n_charts}")
    print(f"  Coords: x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
          f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
          f"z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")
    print(f"  Traction error: max={t_err_mag.max():.4e}, rel_l2={np.sqrt(np.mean(t_err**2)) / (np.sqrt(np.mean(t_true**2)) + 1e-30):.4e}")

    point_data: Dict[str, np.ndarray] = {
        "traction_true":      t_true,         # (N, 3) vector
        "traction_pred":      t_pred,
        "traction_error":     t_err,
        "traction_error_mag": t_err_mag,
        "traction_rel_error": t_rel_err,
        "normal":             normals,
        "chart_id":           chart_id.astype(float),
    }

    # Displacement fields (v3/v4 only)
    if "u_full" in d:
        u = np.asarray(d["u_full"], dtype=float)
        point_data["displacement"]     = u
        point_data["displacement_mag"] = np.asarray(d["u_full_mag"], dtype=float).reshape(-1) \
                                         if "u_full_mag" in d else np.linalg.norm(u, axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, f"{args.prefix}_atlas_surface.vtu")
    print(f"  Writing merged surface VTU: {merged_path}")
    write_vtu_points(merged_path, pts, point_data)

    # Per-chart
    per_chart_paths: List[str] = []
    if not args.no_per_chart:
        for cid in range(n_charts):
            mask = chart_id == cid
            if mask.sum() == 0:
                continue
            cdata = {k: (v[mask] if np.asarray(v).shape[0] == n_pts else v)
                     for k, v in point_data.items()}
            cpath = os.path.join(args.output_dir, f"{args.prefix}_atlas_chart_{cid:02d}.vtu")
            write_vtu_points(cpath, pts[mask], cdata)
            per_chart_paths.append(cpath)
        print(f"  Written {n_charts} per-chart VTU files.")

    # Deformed surface
    deformed_path: Optional[str] = None
    if "x_full_deformed" in d:
        x_def = np.asarray(d["x_full_deformed"], dtype=float)
        deformed_path = os.path.join(args.output_dir, f"{args.prefix}_atlas_deformed.vtu")
        write_vtu_points(deformed_path, x_def, point_data)
        print(f"  Written deformed surface VTU: {deformed_path}")

    elapsed = time.time() - t0
    print(f"  Atlas export done in {elapsed:.1f}s")

    return {
        "type": "atlas_direct",
        "npz": os.path.abspath(npz_path),
        "n_points": n_pts,
        "n_charts": n_charts,
        "merged_vtu": os.path.abspath(merged_path),
        "per_chart_vtu": [os.path.abspath(p) for p in per_chart_paths],
        "deformed_vtu": os.path.abspath(deformed_path) if deformed_path else None,
    }


# ---------------------------------------------------------------------------
# Export Schwarz run
# ---------------------------------------------------------------------------

def export_schwarz(run_dir: str, label: str,
                   output_dir: str, prefix: str) -> Dict:
    """Copy / symlink the existing best-boundary VTU into the output directory,
    and create a clean merged VTU from the Schwarz VTU if it exists."""
    t0 = time.time()
    print(f"\n=== Schwarz export: {label} ===")

    if not os.path.isdir(run_dir):
        print(f"  WARNING: {run_dir} not found — skipping.")
        return {}

    # Find best-boundary VTU
    best_vtu: Optional[str] = None
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith("_best_boundary_fields.vtu"):
            best_vtu = os.path.join(run_dir, fn)
            break

    os.makedirs(output_dir, exist_ok=True)
    out_vtu: Optional[str] = None
    if best_vtu is not None:
        out_vtu = os.path.join(output_dir, f"{prefix}_{label.replace(' ', '_')}_boundary.vtu")
        import shutil
        shutil.copy2(best_vtu, out_vtu)
        print(f"  Copied {os.path.basename(best_vtu)} → {os.path.basename(out_vtu)}")
    else:
        print(f"  No *_best_boundary_fields.vtu found in {run_dir}")

    elapsed = time.time() - t0
    print(f"  Schwarz export done in {elapsed:.1f}s")

    metrics = _load_json(run_dir, "_metrics.json")
    return {
        "type": f"schwarz_{label}",
        "run_dir": os.path.abspath(run_dir),
        "source_vtu": os.path.abspath(best_vtu) if best_vtu else None,
        "output_vtu": os.path.abspath(out_vtu) if out_vtu else None,
        "mu_rel_error_pct": metrics.get("mu_rel_error_percent"),
        "K_rel_error_pct":  metrics.get("K_rel_error_percent"),
    }


# ---------------------------------------------------------------------------
# Publication figures (traction error field map)
# ---------------------------------------------------------------------------

def make_error_map_figure(atlas_dir: str, output_dir: str, figures_dir: str,
                           prefix: str) -> None:
    """2-panel figure: (a) traction-error magnitude colour map on torus surface,
    (b) rel-error histogram across charts."""
    npz_path = _find_npz(atlas_dir, "_obs.npz")
    if npz_path is None:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    set_pub_style(fontsize=9, linewidth=1.2)
    d = np.load(npz_path, allow_pickle=True)

    key_pts  = "x_full"   if "x_full"       in d else "x_eval"
    key_terr = "t_full_error" if "t_full_error" in d else None

    if key_terr is None:
        print("  No traction error data — skipping error map figure.")
        return

    pts    = np.asarray(d[key_pts],   dtype=float)
    t_err  = np.asarray(d[key_terr],  dtype=float)
    t_true = np.asarray(d["t_full_true" if "t_full_true" in d else "t_eval_true"], dtype=float)
    cw     = np.asarray(d["chart_weights_full" if "chart_weights_full" in d else "chart_weights_eval"], dtype=float)
    chart_id = _chart_id_from_weights(cw)

    t_err_mag  = np.linalg.norm(t_err, axis=1)
    t_true_mag = np.linalg.norm(t_true, axis=1)
    t_rel_err  = t_err_mag / np.maximum(t_true_mag, 1e-12 * t_true_mag.max() + 1e-20)

    n_charts = cw.shape[1]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN),
        constrained_layout=True,
    )

    # (a) Traction error scatter in 3-D projected to (x, y)
    ax = axes[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1],
                    c=np.log10(t_err_mag + 1e-14),
                    cmap="hot_r", s=1.5, lw=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}|\mathbf{t}_\mathrm{pred} - \mathbf{t}_\mathrm{true}|$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Traction error on torus (top view)")
    ax.set_aspect("equal")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")

    # (b) Per-chart relative error box plot
    ax2 = axes[1]
    chart_data = [t_rel_err[chart_id == cid] * 100.0 for cid in range(n_charts)]
    ax2.boxplot(chart_data, patch_artist=True,
                boxprops=dict(facecolor=PUB_COLORS[0], alpha=0.6),
                medianprops=dict(color="black"))
    ax2.set_xlabel("Chart index")
    ax2.set_ylabel("Relative traction error (%)")
    ax2.set_title("Per-chart traction error distribution")
    ax2.set_yscale("log")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
             va="top", ha="left", fontsize=9, fontweight="bold")

    fig.suptitle("Inverse neo-Hookean on torus — traction field error",
                 fontsize=10, fontweight="bold")

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(figures_dir, f"torus_inverse_traction_error.{ext}")
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    manifest = {
        "atlas_direct": {},
        "schwarz_displacement": {},
        "schwarz_traction": {},
        "coords": "physical",
    }

    # Atlas export
    manifest["atlas_direct"] = export_atlas(args)

    # Schwarz exports
    manifest["schwarz_displacement"] = export_schwarz(
        args.disp_dir,  "disp",
        args.output_dir, args.prefix,
    )
    manifest["schwarz_traction"] = export_schwarz(
        args.tract_dir, "tract",
        args.output_dir, args.prefix,
    )

    # Write manifest
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest: {manifest_path}")

    # Publication figures
    if not args.skip_figures:
        print(f"\nGenerating publication figures → {args.figures_dir}")
        make_error_map_figure(
            args.atlas_dir, args.output_dir, args.figures_dir, args.prefix
        )

    elapsed = time.time() - t0
    print(f"\nAll torus exports done in {elapsed:.1f}s")
    print("\nParaView tips (physical space, torus):")
    print("  1. Open *_atlas_surface.vtu.  Color by 'traction_error_mag'.")
    print("  2. Open *_atlas_deformed.vtu to visualise the deformed shape.")
    print("  3. 'Threshold' on 'chart_id' to highlight individual atlas charts.")
    print("  4. Open the Schwarz *_boundary.vtu files for comparison.")
    print("  5. Apply 'Warp By Vector' with 'displacement' to animate deformation.")


if __name__ == "__main__":
    main()
