#!/usr/bin/env python3
"""Checkpoint-driven PyVista export for the torus forward BVP.

This script reconstructs the saved multi-chart torus state from a real
checkpoint, exports raw per-chart VTK data, and renders manuscript figures
directly from the FEM mesh (no griddata surface interpolation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

from experiments.torus_elastoplastic.reconstruct_torus_fields import (
    charts_to_multiblock,
    export_l2_projected_vtk,
    find_latest_checkpoint,
    interpolate_plastic_surface_fields,
    make_torus_surface,
    reconstruct_checkpoint,
    unique_points_to_polydata,
    unique_points_to_unstructured_grid,
)


OUT_DIR = Path(_REPO) / "manuscript" / "figures_cmame_core" / "example_forward_bvp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_POS = [(3.0, -2.5, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def render_panel(mesh, scalar, title, cmap="turbo", fmt="%.4f"):
    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        scalars=scalar,
        cmap=cmap,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": title,
            "title_font_size": 16,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": fmt,
            "position_x": 0.78,
            "position_y": 0.15,
            "width": 0.14,
            "height": 0.65,
            "color": "black",
        },
    )
    pl.camera_position = CAMERA_POS
    pl.screenshot("/tmp/_forward_bvp_panel.png", transparent_background=False)
    img = plt.imread("/tmp/_forward_bvp_panel.png")
    pl.close()
    return img


def _combine_raw_blocks(blocks: pv.MultiBlock):
    combined = blocks[0].copy()
    for block_idx in range(1, blocks.n_blocks):
        combined = combined.merge(blocks[block_idx], merge_points=False)
    return combined


def render_mesh_and_displacement(blocks: pv.MultiBlock, unique_poly: pv.PolyData, out_dir: Path):
    raw_combined = _combine_raw_blocks(blocks)

    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    surf_mesh = raw_combined.extract_surface(algorithm=None)
    pl.add_mesh(
        surf_mesh,
        scalars="chart_id",
        cmap="Set2",
        show_edges=True,
        edge_color="gray",
        line_width=0.3,
        scalar_bar_args={
            "title": "Chart ID",
            "title_font_size": 16,
            "label_font_size": 12,
            "n_labels": 8,
            "fmt": "%.0f",
            "position_x": 0.78,
            "position_y": 0.15,
            "width": 0.14,
            "height": 0.65,
            "color": "black",
        },
    )
    pl.camera_position = CAMERA_POS
    pl.screenshot("/tmp/_forward_bvp_mesh.png", transparent_background=False)
    img_mesh = plt.imread("/tmp/_forward_bvp_mesh.png")
    pl.close()

    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    smooth_surface, _, _ = make_torus_surface()
    pl.add_mesh(smooth_surface, color="lightgray", opacity=0.25, smooth_shading=True)

    n_pts = unique_poly.n_points
    stride = max(1, n_pts // 600)
    ids = np.arange(0, n_pts, stride)
    sub = unique_poly.extract_points(ids)
    u_mag = sub.point_data["displacement_mag"]
    max_u = float(np.max(np.asarray(u_mag))) if len(u_mag) else 0.0
    scale = 0.4 * 0.35 / max(max_u, 1e-30)
    arrows = sub.glyph(
        orient="displacement",
        scale="displacement_mag",
        factor=scale,
        geom=pv.Arrow(),
    )
    pl.add_mesh(
        arrows,
        scalars="displacement_mag",
        cmap="coolwarm",
        scalar_bar_args={
            "title": "|u|",
            "title_font_size": 16,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": "%.3e",
            "position_x": 0.78,
            "position_y": 0.15,
            "width": 0.14,
            "height": 0.65,
            "color": "black",
        },
    )
    pl.camera_position = CAMERA_POS
    pl.screenshot("/tmp/_forward_bvp_disp.png", transparent_background=False)
    img_disp = plt.imread("/tmp/_forward_bvp_disp.png")
    pl.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    for ax, img, label in zip(axes, [img_mesh, img_disp], ["(a)", "(b)"]):
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
        )
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"forward_bvp_mesh_displacement.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def export_vtk(result, out_dir: Path):
    vtu_dir = out_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    blocks = charts_to_multiblock(result)
    for chart_id, block in enumerate(blocks):
        if block is None:
            continue
        block.save(vtu_dir / f"chart_{chart_id:02d}.vtu")
    if blocks.n_blocks > 0:
        blocks.save(vtu_dir / "all_charts.vtm")
        # Merged VTU with nodal displacement (merge_points=False to keep charts separate)
        combined = blocks[0].copy()
        for bi in range(1, blocks.n_blocks):
            combined = combined.merge(blocks[bi], merge_points=False)
        combined.save(vtu_dir / "all_charts_merged.vtu")

    unique_poly = unique_points_to_polydata(result.unique_points)
    unique_poly.save(vtu_dir / "unique_points.vtp")
    unique_grid = unique_points_to_unstructured_grid(result.unique_points)
    unique_grid.save(vtu_dir / "unique_points.vtu")
    unique_surface_poly = unique_points_to_polydata(result.unique_surface_points)
    unique_surface_poly.save(vtu_dir / "unique_surface_points.vtp")
    unique_surface_grid = unique_points_to_unstructured_grid(result.unique_surface_points)
    unique_surface_grid.save(vtu_dir / "unique_surface_points.vtu")
    return blocks, unique_poly


def render_displacement_components(blocks: pv.MultiBlock, out_dir: Path):
    """Render u_x, u_y, u_z displacement panels from the raw FEM mesh."""
    combined = blocks[0].copy()
    for bi in range(1, blocks.n_blocks):
        combined = combined.merge(blocks[bi], merge_points=False)
    surf = combined.extract_surface(algorithm=None)

    # Split displacement vector into components
    disp = np.asarray(surf.point_data["displacement"])
    surf.point_data["u_x"] = disp[:, 0]
    surf.point_data["u_y"] = disp[:, 1]
    surf.point_data["u_z"] = disp[:, 2]

    panels = [
        ("u_x", r"$u_x$", "coolwarm", "%.3e"),
        ("u_y", r"$u_y$", "coolwarm", "%.3e"),
        ("u_z", r"$u_z$", "coolwarm", "%.3e"),
    ]
    images = [render_panel(surf, name, title, cmap, fmt) for name, title, cmap, fmt in panels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    for ax, img, label in zip(axes, images, ["(a)", "(b)", "(c)"]):
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
        )
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"forward_bvp_displacement_fields.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/torus_forward_bvp_debug_small_max25",
        help="Used when --checkpoint is omitted.",
    )
    parser.add_argument("--round-decimals", type=int, default=10)
    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint(Path(_REPO) / args.run_dir)
    )
    result = reconstruct_checkpoint(checkpoint, round_decimals=args.round_decimals)

    # Export raw per-chart VTU and unique point clouds
    blocks, unique_poly = export_vtk(result, OUT_DIR)
    render_mesh_and_displacement(blocks, unique_poly, OUT_DIR)

    # Render displacement component panels (u_x, u_y, u_z) from raw FEM mesh
    render_displacement_components(blocks, OUT_DIR)

    # L2-project F^p fields to nodes and export VTU
    surf = interpolate_plastic_surface_fields(result)
    export_l2_projected_vtk(surf, OUT_DIR)

    report = {
        "checkpoint": str(result.checkpoint_path),
        "classification": result.classification,
        "metrics": result.metrics,
    }
    (OUT_DIR / "forward_bvp_reconstruction_report.json").write_text(json.dumps(report, indent=2))
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Classification: {result.classification}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6e}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
