#!/usr/bin/env python3
"""
Export Poisson-on-rabbit atlas results to ParaView-compatible VTK files
in *physical (world) space*.

Background
──────────
The solution NPZ stores point coordinates in the SDF-normalised reference
frame used during training:

    x_norm = (x_physical − center) / scale

The inverse transform that recovers physical space is:

    x_physical = center + scale × x_norm

The center and scale are stored in the atlas-data NPZ produced by
build_rabbit_atlas_volumetric.py.  This script loads them and applies the
transform before writing any VTK output, so that the resulting files open
with the correct rabbit geometry in ParaView.

All 50 000 points in the solution NPZ are already filtered to lie inside the
rabbit domain (SDF < 0), so no additional culling is needed.

Surface mesh (--surface / --sdf-checkpoint):
  Evaluates the trained neural SDF on an 80³ grid, extracts the SDF=0
  isosurface with marching cubes, and exports a triangulated surface mesh VTU
  with solution scalars (u_pred, u_error_mag, …) mapped to surface vertices
  via nearest-neighbour interpolation from the interior point cloud.  Open
  this file in ParaView with the default 'Surface' representation to see the
  rabbit geometry immediately — no need for 'Point Gaussian' mode.

In ParaView:
  - <prefix>_surface.vtu  → open with Representation='Surface', colour by
    'u_error_mag' or 'u_pred' for instant rabbit geometry visualisation.
  - <prefix>_merged.vtu   → Representation='Point Gaussian' for the interior
    point cloud; use 'Slice' or 'Threshold' filters for cross-sections.
  - <prefix>_grid.vtr     → Representation='Volume' for volumetric rendering.

Outputs (written to --output-dir):
    <prefix>_merged.vtu          All interior points (physical coords)
    <prefix>_chart_<id>.vtu      Per-chart subsets  (physical coords)
    <prefix>_surface.vtu         Triangulated rabbit surface (optional, --surface)
    <prefix>_grid.vtr            Regular Cartesian grid (optional, --grid)
    <prefix>_manifest.json       Paths + metrics summary

Usage (default paths for the CompactChartNet best run):
    python postprocessing/poisson_rabbit_paraview.py \\
        --solution-npz runs/attempt20c_compact/rabbit_poisson_schwarz_attempt20c_compact_solution.npz \\
        --atlas-npz    runs/atlas_vol/rabbit_atlas_data.npz \\
        --output-dir   runs/attempt20c_compact/paraview \\
        --prefix       rabbit_poisson_compact \\
        --grid \\
        --surface \\
        --sdf-checkpoint runs/examples/meshfree_accel_20260212_194730/sdf/rabbit_sdf.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    write_vtu_points,
    write_vtu_surface_mesh,
    write_vtu_rectilinear_grid,
    interpolate_to_grid,
)

# Default atlas NPZ for the volumetric (50 000 point) rabbit runs
_DEFAULT_ATLAS_NPZ = "runs/atlas_vol/rabbit_atlas_data.npz"
_DEFAULT_SDF_CKPT  = "runs/examples/meshfree_accel_20260212_194730/sdf/rabbit_sdf.pt"


# ---------------------------------------------------------------------------
# SDF-normalisation helpers
# ---------------------------------------------------------------------------

def load_sdf_transform(atlas_npz: Optional[str]) -> Tuple[np.ndarray, float]:
    """Return (center, scale) from the atlas-data NPZ.

    Parameters
    ----------
    atlas_npz:
        Path to the atlas NPZ file that contains 'center' and 'scale' arrays.
        If None or the file is missing, returns (zeros, 1.0) — i.e. identity
        transform (no denormalization).

    Returns
    -------
    center : np.ndarray of shape (3,)
    scale  : float
    """
    if atlas_npz is None or not os.path.isfile(atlas_npz):
        print(f"  WARNING: atlas NPZ not found ({atlas_npz}); "
              "coordinates will remain in SDF-normalised space.")
        return np.zeros(3, dtype=float), 1.0

    d = np.load(atlas_npz, allow_pickle=True)
    center = np.asarray(d["center"], dtype=float).reshape(3)
    scale  = float(np.asarray(d["scale"]).reshape(-1)[0])
    return center, scale


def to_physical(points_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Convert SDF-normalised coordinates to physical space.

    x_physical = center + scale * x_norm
    """
    return center + scale * np.asarray(points_norm, dtype=float)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export rabbit Poisson atlas results to ParaView VTK files "
                    "in physical (world) space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--solution-npz",
        required=True,
        help="Path to *_solution.npz produced by run_poisson_rabbit_atlas_schwarz.py",
    )
    p.add_argument(
        "--atlas-npz",
        default=_DEFAULT_ATLAS_NPZ,
        help="Path to rabbit_atlas_data.npz containing the SDF normalisation "
             "'center' and 'scale'.  Used to convert coordinates to physical space.",
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
    p.add_argument(
        "--surface",
        action="store_true",
        help="Export a triangulated rabbit surface mesh VTU using marching cubes "
             "on the trained neural SDF.  Requires --sdf-checkpoint.",
    )
    p.add_argument(
        "--sdf-checkpoint",
        default=_DEFAULT_SDF_CKPT,
        help="Path to the SDF model checkpoint (rabbit_sdf.pt) used for surface "
             "extraction via marching cubes.",
    )
    p.add_argument(
        "--surface-grid", type=int, default=80,
        help="Grid resolution for marching cubes SDF evaluation (NxNxN, default 80)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# SDF-based surface extraction
# ---------------------------------------------------------------------------

def extract_sdf_surface(
    sdf_ckpt: str,
    grid_size: int = 80,
    half_extent: float = 0.60,
    batch_size: int = 32768,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate the neural SDF on a grid, run marching cubes, return surface.

    Returns
    -------
    verts_norm : (V, 3) float64
        Surface vertex coordinates in SDF-normalised space.
    faces : (F, 3) int32
        Triangle vertex index array.
    sdf_center : (3,) float64
        Center used by the SDF model.
    sdf_scale : float
        Scale used by the SDF model.
    """
    import torch

    class _MLP(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
            super().__init__()
            layers = [torch.nn.Linear(in_dim, width)]
            for _ in range(depth - 1):
                layers.append(torch.nn.Linear(width, width))
            self.hidden = torch.nn.ModuleList(layers)
            self.out = torch.nn.Linear(width, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for layer in self.hidden:
                h = torch.tanh(layer(h))
            return self.out(h)

    class _SDFNet(torch.nn.Module):
        def __init__(self, width: int = 128, depth: int = 6):
            super().__init__()
            self.net = _MLP(in_dim=3, out_dim=1, width=width, depth=depth)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("scikit-image is required for surface extraction: pip install scikit-image")

    ckpt = torch.load(sdf_ckpt, map_location="cpu", weights_only=False)
    kw = ckpt.get("model_kwargs", {"width": 128, "depth": 6})
    model = _SDFNet(width=kw.get("width", 128), depth=kw.get("depth", 6))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sdf_center = np.asarray(ckpt["center"], dtype=float).reshape(3)
    sdf_scale  = float(np.asarray(ckpt["scale"]).reshape(-1)[0])

    # Build 3-D grid in SDF-normalised space
    ax = np.linspace(-half_extent, half_extent, grid_size, dtype=np.float32)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")          # (G, G, G) each
    grid_pts = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1)  # (G³, 3)

    # Evaluate SDF in batches
    sdf_vals = np.empty(len(grid_pts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            chunk = torch.tensor(grid_pts[i:i + batch_size])
            sdf_vals[i:i + batch_size] = model(chunk).numpy()

    sdf_vol = sdf_vals.reshape(grid_size, grid_size, grid_size)

    # Marching cubes at SDF = 0 (rabbit surface)
    spacing = (ax[1] - ax[0],) * 3
    verts, faces, _, _ = marching_cubes(sdf_vol, level=0.0, spacing=spacing)
    # verts come out in index * spacing coordinates — shift to centred range
    verts = verts + ax[0]   # shift origin from [0, 0, 0] to [-half_extent, ...]
    verts = verts.astype(np.float64)
    faces = faces.astype(np.int32)

    print(f"  SDF marching cubes: {len(verts)} vertices, {len(faces)} triangles")
    return verts, faces, sdf_center, sdf_scale


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ---- Load SDF transform (center + scale) ------------------------------
    print(f"Loading atlas transform from {args.atlas_npz} ...")
    center, scale = load_sdf_transform(args.atlas_npz)
    print(f"  SDF normalisation: center={center}, scale={scale:.6f}")
    print(f"  Transform: x_physical = center + {scale:.4f} × x_norm")

    # ---- Load solution data -----------------------------------------------
    print(f"Loading {args.solution_npz} ...")
    data = np.load(args.solution_npz, allow_pickle=True)

    # Points are in SDF-normalised space — convert to physical space
    points_norm     = np.asarray(data["points"],            dtype=float)   # (N, 3)
    points_physical = to_physical(points_norm, center, scale)

    u_pred         = np.asarray(data["u_pred"],            dtype=float).reshape(-1)
    u_true         = np.asarray(data["u_true"],            dtype=float).reshape(-1)
    u_error        = np.asarray(data["u_error"],           dtype=float).reshape(-1)
    u_error_mag    = np.asarray(data["u_error_mag"],       dtype=float).reshape(-1)
    chart_id       = np.asarray(data["chart_id"],          dtype=np.int32).reshape(-1)
    blend_weight   = np.asarray(data["blend_weight"],      dtype=float).reshape(-1)
    interface_res  = np.asarray(data["interface_residual"],dtype=float).reshape(-1)

    n_points = points_physical.shape[0]
    n_charts = int(chart_id.max()) + 1

    print(f"  N = {n_points} interior points,  n_charts = {n_charts}")
    print(f"  Physical coords: "
          f"x=[{points_physical[:,0].min():.4f}, {points_physical[:,0].max():.4f}]  "
          f"y=[{points_physical[:,1].min():.4f}, {points_physical[:,1].max():.4f}]  "
          f"z=[{points_physical[:,2].min():.4f}, {points_physical[:,2].max():.4f}]")

    # Optional extra fields
    extra: Dict[str, np.ndarray] = {}
    for key in ("detail_score", "high_detail_mask"):
        if key in data:
            extra[key] = np.asarray(data[key], dtype=float).reshape(-1)

    # ---- Derived fields ---------------------------------------------------
    u_true_abs = np.abs(u_true)
    u_rel_err  = u_error_mag / np.maximum(u_true_abs, 1e-12 * u_true_abs.max() + 1e-20)

    # ---- Build point_data dict --------------------------------------------
    point_data: Dict[str, np.ndarray] = {
        "u_pred":             u_pred,
        "u_true":             u_true,
        "u_error":            u_error,
        "u_error_mag":        u_error_mag,
        "u_rel_error":        u_rel_err,
        "chart_id":           chart_id.astype(float),
        "blend_weight":       blend_weight,
        "interface_residual": interface_res,
    }
    point_data.update(extra)

    # ---- Write merged VTU (physical coords) -------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, f"{args.prefix}_merged.vtu")
    print(f"Writing merged VTU: {merged_path}")
    write_vtu_points(merged_path, points_physical, point_data)

    # ---- Write per-chart VTUs ---------------------------------------------
    per_chart_paths: list = []
    if not args.no_per_chart:
        for cid in range(n_charts):
            mask = chart_id == cid
            if mask.sum() == 0:
                continue
            chart_data = {k: v[mask] for k, v in point_data.items()}
            cpath = os.path.join(args.output_dir, f"{args.prefix}_chart_{cid:02d}.vtu")
            print(f"  Chart {cid:02d}: {mask.sum()} pts → {cpath}")
            write_vtu_points(cpath, points_physical[mask], chart_data)
            per_chart_paths.append(cpath)

    # ---- Optional rectilinear grid (physical coords) ----------------------
    grid_path: Optional[str] = None
    if args.grid:
        print(
            f"Interpolating to {args.grid_nx}×{args.grid_ny}×{args.grid_nz} grid "
            "(nearest-neighbour) in physical space ..."
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
            points_physical, scalar_fields,
            nx=args.grid_nx, ny=args.grid_ny, nz=args.grid_nz,
            method="nearest",
        )
        grid_path = os.path.join(args.output_dir, f"{args.prefix}_grid.vtr")
        print(f"Writing rectilinear grid VTR: {grid_path}")
        write_vtu_rectilinear_grid(grid_path, xg, yg, zg, gf)

    # ---- Optional surface mesh (SDF marching cubes) -----------------------
    surface_path: Optional[str] = None
    if args.surface:
        print(f"\nExtracting rabbit surface via neural SDF: {args.sdf_checkpoint} ...")
        verts_norm, faces, sdf_center, sdf_scale = extract_sdf_surface(
            args.sdf_checkpoint,
            grid_size=args.surface_grid,
        )
        # Convert SDF-normalised verts to physical space using the SDF model's
        # own center/scale (stored inside the checkpoint, not the atlas NPZ).
        verts_phys = sdf_center + sdf_scale * verts_norm
        # Map scalar fields to surface vertices via nearest-neighbour
        from scipy.spatial import cKDTree
        tree = cKDTree(points_physical)
        _, nn_idx = tree.query(verts_phys)
        surf_data: Dict[str, np.ndarray] = {k: v[nn_idx] for k, v in point_data.items()}
        surface_path = os.path.join(args.output_dir, f"{args.prefix}_surface.vtu")
        print(f"Writing surface VTU: {surface_path}")
        write_vtu_surface_mesh(surface_path, verts_phys, faces, surf_data)

    # ---- Summary metrics --------------------------------------------------
    rel_l2  = float(np.sqrt(np.mean(u_error**2)) / (np.sqrt(np.mean(u_true**2)) + 1e-30))
    max_err = float(u_error_mag.max())

    # ---- Manifest ---------------------------------------------------------
    manifest = {
        "solution_npz":   os.path.abspath(args.solution_npz),
        "atlas_npz":      os.path.abspath(args.atlas_npz) if os.path.isfile(args.atlas_npz) else None,
        "sdf_center":     center.tolist(),
        "sdf_scale":      scale,
        "n_points":       n_points,
        "n_charts":       n_charts,
        "rel_l2_error":   rel_l2,
        "max_error":      max_err,
        "coords":         "physical",
        "merged_vtu":     os.path.abspath(merged_path),
        "per_chart_vtu":  [os.path.abspath(p) for p in per_chart_paths],
        "grid_vtr":       os.path.abspath(grid_path) if grid_path else None,
        "surface_vtu":    os.path.abspath(surface_path) if surface_path else None,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {manifest_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  rel_L2 = {rel_l2*100:.3f}%,  max_err = {max_err:.4f}")
    print("\nParaView tips (physical space):")
    print("  1. Open the merged VTU.  Representation = 'Point Gaussian' or 'Sphere'.")
    print("     The rabbit shape should now be visible at its true physical dimensions.")
    print("  2. Color by 'u_error_mag' with a diverging colormap.")
    print("  3. Add 'Slice' filter (X/Y/Z) to see interior cross-sections.")
    print("  4. 'Threshold' on 'chart_id' to isolate individual atlas charts.")
    if grid_path:
        print(f"  5. Open {os.path.basename(grid_path)} → Representation = 'Volume'")
        print("     for full 3-D volumetric rendering (note: grid is bounding-box only;")
        print("     points outside the rabbit surface have values from nearest interior pt).")
    if surface_path:
        print(f"  6. Open {os.path.basename(surface_path)} → Representation = 'Surface'")
        print("     Colour by 'u_error_mag' or 'u_pred' to see the full rabbit geometry.")


if __name__ == "__main__":
    main()
