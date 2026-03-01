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

Note on physical coordinates vs. Elder rabbit
─────────────────────────────────────────────
The Poisson rabbit uses atlas_vol/rabbit_atlas_data.npz (scale ≈ 1.64),
whereas the Elder rabbit uses the Stanford-Bunny atlas (scale ≈ 0.156).
Consequently the Poisson physical domain spans ≈ 1.8 m while the Elder domain
spans ≈ 0.16 m.  The Poisson PINN was trained on a *procedural* union-of-
ellipsoids SDF — not the Stanford Bunny PLY — so its geometry appears as a
rounded ellipsoidal blob rather than a recognisable rabbit silhouette.
Both coordinate transforms are numerically correct; the visual difference is
a consequence of the different training domains.

Surface mesh (--surface):
  Two modes are available:

  1. PLY mesh (--ply-file, recommended for Stanford Bunny):
     Reads the original PLY mesh vertices and face connectivity directly.
     This preserves thin features such as rabbit ears that carry almost no
     interior points and are completely lost by voxelisation.  Solution
     scalars are mapped to PLY vertices by nearest-neighbour query from the
     interior point cloud.

  2. Voxel occupancy (default when --ply-file is omitted):
     Voxelises the 50 000 interior solution points onto a grid occupancy
     grid, applies morphological fill/close, then runs marching cubes.
     Produces a blobby outline that loses fine surface features.

In ParaView:
  - <prefix>_surface.vtu  → Representation='Surface', colour by 'u_error_mag'
  - <prefix>_merged.vtu   → Representation='Point Gaussian' for point cloud
  - <prefix>_grid.vtr     → Representation='Volume' for volumetric rendering

Outputs (written to --output-dir):
    <prefix>_merged.vtu          All interior points (physical coords)
    <prefix>_chart_<id>.vtu      Per-chart subsets  (physical coords)
    <prefix>_surface.vtu         Triangulated rabbit surface (optional, --surface)
    <prefix>_grid.vtr            Regular Cartesian grid (optional, --grid)
    <prefix>_manifest.json       Paths + metrics summary

Additionally writes publication-quality convergence figures (Matplotlib) to
--figures-dir, mirroring the figure output of elder_rabbit_paraview.py.

Usage (default paths for the CompactChartNet best run):
    python postprocessing/poisson_rabbit_paraview.py \\
        --solution-npz runs/attempt20c_compact/rabbit_poisson_schwarz_attempt20c_compact_solution.npz \\
        --atlas-npz    runs/atlas_vol/rabbit_atlas_data.npz \\
        --output-dir   paraview/rabbit_poisson_compact \\
        --prefix       rabbit_poisson_compact \\
        --grid \\
        --surface \\
        --figures-dir  figures
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    DOUBLE_COL_W,
    GOLDEN,
    PUB_COLORS,
    SINGLE_COL_W,
    set_pub_style,
    write_vtu_points,
    write_vtu_surface_mesh,
    write_vtu_rectilinear_grid,
    interpolate_to_grid,
)

# Default atlas NPZ for the volumetric (50 000 point) rabbit runs
_DEFAULT_ATLAS_NPZ = "runs/atlas_vol/rabbit_atlas_data.npz"


# ---------------------------------------------------------------------------
# JSON helpers (mirror elder_rabbit_paraview.py)
# ---------------------------------------------------------------------------

def _load_json(run_dir: str, suffix: str) -> dict:
    """Return the first JSON file in run_dir whose name ends with suffix."""
    if not os.path.isdir(run_dir):
        return {}
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(suffix):
            with open(os.path.join(run_dir, fn)) as fh:
                return json.load(fh)
    return {}


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
        help="Export a triangulated rabbit surface mesh VTU.  Voxelises the "
             "interior solution points onto a grid, fills gaps, and runs "
             "marching cubes.  No SDF checkpoint needed.",
    )
    p.add_argument(
        "--surface-grid", type=int, default=50,
        help="Resolution of the voxel occupancy grid for --surface (NxNxN). "
             "Default 50.  Voxel edge ≈ 0.022 normalised units for the 50k-point "
             "rabbit run; binary-closing bridges the remaining gaps.",
    )
    p.add_argument(
        "--surface-close-iters", type=int, default=3,
        help="Morphological binary-closing iterations for --surface.  "
             "Default 3 bridges ~1-voxel gaps between isolated occupied voxels.",
    )
    p.add_argument(
        "--surface-smooth-sigma", type=float, default=1.2,
        help="Gaussian smoothing sigma (voxels) applied before marching cubes "
             "for --surface.  Reduces staircase artefacts and triangle count. "
             "Default 1.2; set to 0 to disable.",
    )
    p.add_argument(
        "--ply-file", default=None,
        help="Path to the original PLY mesh file (e.g. bun_zipper.ply) whose "
             "vertices and face connectivity are used for --surface instead of "
             "the voxel-occupancy reconstruction.  Recommended for the Stanford "
             "Bunny: preserves thin features (ears, legs) that have almost no "
             "interior atlas points and are lost by voxelisation.  "
             "Example: runs/atlas_schwarz_20260212_210412/downloads/bunny/"
             "reconstruction/bun_zipper.ply",
    )
    p.add_argument(
        "--run-dir", default=None,
        help="Run directory containing *_history.json and *_metrics.json used "
             "for convergence figures.  Inferred from --solution-npz if omitted.",
    )
    p.add_argument(
        "--figures-dir", default="figures",
        help="Directory for publication-quality Matplotlib figures.",
    )
    p.add_argument(
        "--skip-figures", action="store_true",
        help="Skip Matplotlib figure generation (useful when matplotlib is "
             "unavailable or figures are not needed).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Occupancy-voxel surface extraction
# ---------------------------------------------------------------------------

def extract_surface_from_points(
    points_norm: np.ndarray,
    *,
    grid_size: int = 50,
    padding: float = 0.005,
    close_iters: int = 3,
    smooth_sigma: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelise interior points and run marching cubes to recover the surface.

    The interior solution points were filtered to lie inside the rabbit domain
    (SDF < 0) during the PINN solve.  Their outer envelope therefore traces
    the true rabbit surface.

    Key constraint: the voxel size must be ≥ the average inter-point spacing
    of the solution point cloud (~0.026 in normalised units for 50k points).
    grid_size=40 (voxel ≈ 0.028) gives >99% of filled voxels in one connected
    component; finer grids produce thousands of isolated floating voxels.

    Pipeline:
      1. Voxelise onto a grid_size³ occupancy grid.
      2. Extract the LARGEST connected component (discards isolated noise voxels).
      3. Apply binary_fill_holes to close enclosed bubbles.
      4. Apply binary_closing (close_iters) to bridge thin gaps (ears, legs).
      5. Apply Gaussian smoothing (smooth_sigma voxels) to eliminate staircase
         artefacts and reduce triangle count.
      6. Run marching_cubes at level=0.5.

    Parameters
    ----------
    points_norm:
        (N, 3) interior point cloud in SDF-normalised coordinates.
    grid_size:
        Resolution of the occupancy grid along each axis.  Default 40.
        Must be small enough that the voxel edge length ≥ inter-point spacing.
    padding:
        Small margin (normalised units) around the point-cloud bounding box.
    close_iters:
        Binary-closing iterations to bridge gaps in thin regions.
    smooth_sigma:
        Gaussian smoothing sigma (voxels) before marching cubes.

    Returns
    -------
    verts_norm : (V, 3) float64
        Surface vertex positions in SDF-normalised space.
    faces : (F, 3) int32
        Triangle face index array.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for --surface: pip install scikit-image"
        )
    from scipy.ndimage import (
        binary_fill_holes, binary_closing, gaussian_filter, label
    )

    lo = points_norm.min(axis=0) - padding
    hi = points_norm.max(axis=0) + padding
    ranges = [(lo[i], hi[i]) for i in range(3)]
    spacing = tuple(float((hi[i] - lo[i]) / (grid_size - 1)) for i in range(3))

    # 1. Build binary occupancy volume
    occupancy, _ = np.histogramdd(points_norm, bins=grid_size, range=ranges)
    binary = occupancy > 0

    # 2. Keep only the largest connected component (removes isolated noise)
    lbl, n_features = label(binary)
    if n_features > 1:
        comp_sizes = np.bincount(lbl.ravel())
        comp_sizes[0] = 0          # ignore background
        largest_label = int(comp_sizes.argmax())
        binary = lbl == largest_label
        print(f"  Connected components: {n_features}, "
              f"kept largest ({comp_sizes[largest_label]} voxels, "
              f"{comp_sizes[largest_label]/binary.sum()*100:.0f}% of filled)")

    # 3. Fill enclosed holes, then morphological close for thin features
    binary = binary_fill_holes(binary)
    binary = binary_closing(binary, iterations=close_iters)

    # 4. Gaussian smoothing: eliminates staircase, reduces triangle count
    smooth = gaussian_filter(binary.astype(np.float64), sigma=smooth_sigma)

    # 5. Marching cubes at level=0.5
    verts, faces, _, _ = marching_cubes(smooth, level=0.5, spacing=spacing)
    verts = verts + lo   # shift from index-origin to normalised lo[]
    print(
        f"  Voxel occupancy surface: {len(verts)} vertices, {len(faces)} triangles"
        f"  (grid {grid_size}³, sigma={smooth_sigma}, close_iters={close_iters})"
    )
    return verts.astype(np.float64), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# PLY mesh reader (vertices + face connectivity)
# ---------------------------------------------------------------------------

# Maps PLY property type names to (numpy dtype suffix, struct format char, byte size)
_PLY_NP: Dict[str, str] = {
    "char":    "i1",  "int8":    "i1",
    "uchar":   "u1",  "uint8":   "u1",
    "short":   "i2",  "int16":   "i2",
    "ushort":  "u2",  "uint16":  "u2",
    "int":     "i4",  "int32":   "i4",
    "uint":    "u4",  "uint32":  "u4",
    "float":   "f4",  "float32": "f4",
    "double":  "f8",  "float64": "f8",
}
_PLY_STRUCT: Dict[str, tuple] = {
    "char":    ("b", 1),  "int8":    ("b", 1),
    "uchar":   ("B", 1),  "uint8":   ("B", 1),
    "short":   ("h", 2),  "int16":   ("h", 2),
    "ushort":  ("H", 2),  "uint16":  ("H", 2),
    "int":     ("i", 4),  "int32":   ("i", 4),
    "uint":    ("I", 4),  "uint32":  ("I", 4),
    "float":   ("f", 4),  "float32": ("f", 4),
    "double":  ("d", 8),  "float64": ("d", 8),
}


def parse_ply_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a PLY file and return vertex positions and triangle faces.

    Supports ``binary_little_endian``, ``binary_big_endian``, and ``ascii``
    PLY formats.  Non-triangular faces (quads, n-gons) are tessellated into
    triangles using a fan from the first vertex.

    Parameters
    ----------
    path:
        Path to the ``.ply`` mesh file.

    Returns
    -------
    verts : (V, 3) float64
        Vertex ``(x, y, z)`` positions in the PLY file's coordinate system.
    faces : (F, 3) int32
        Triangle face index array.
    """
    with open(path, "rb") as fh:
        # ── parse ASCII header ──────────────────────────────────────────────
        header_lines: List[str] = []
        while True:
            raw = fh.readline()
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Detect binary / ascii format
        fmt = "ascii"
        for l in header_lines:
            if l.startswith("format"):
                parts = l.split()
                if len(parts) >= 2:
                    fmt = parts[1]   # e.g. "binary_little_endian"
                break
        endian = "<" if "little" in fmt else (">" if "big" in fmt else "=")

        # Parse element/property declarations
        n_verts: int = 0
        n_faces: int = 0
        vert_props: List[Tuple[str, str]] = []   # (name, ply_type_str)
        face_count_ptype: str = "uchar"
        face_idx_ptype:   str = "int"

        cur_elem = None
        for l in header_lines:
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1])
                cur_elem = "vertex"
            elif l.startswith("element face"):
                n_faces = int(l.split()[-1])
                cur_elem = "face"
            elif l.startswith("element"):
                cur_elem = "other"
            elif l.startswith("property list") and cur_elem == "face":
                parts = l.split()   # property list <count_type> <idx_type> <name>
                face_count_ptype = parts[2]
                face_idx_ptype   = parts[3]
            elif l.startswith("property") and "list" not in l and cur_elem == "vertex":
                parts = l.split()   # property <type> <name>
                vert_props.append((parts[2], parts[1]))

        # ── read vertices ────────────────────────────────────────────────────
        if fmt == "ascii":
            x_col = next(i for i, (n, _) in enumerate(vert_props) if n == "x")
            y_col = next(i for i, (n, _) in enumerate(vert_props) if n == "y")
            z_col = next(i for i, (n, _) in enumerate(vert_props) if n == "z")
            verts = np.zeros((n_verts, 3), dtype=np.float64)
            for vi in range(n_verts):
                row = fh.readline().decode("ascii").split()
                verts[vi, 0] = float(row[x_col])
                verts[vi, 1] = float(row[y_col])
                verts[vi, 2] = float(row[z_col])
        else:
            # Binary: build a numpy structured dtype and read all vertices at once
            dt = np.dtype(
                [(name, endian + _PLY_NP[ptype]) for name, ptype in vert_props]
            )
            raw = fh.read(n_verts * dt.itemsize)
            va = np.frombuffer(raw, dtype=dt)
            verts = np.column_stack([
                va["x"].astype(np.float64),
                va["y"].astype(np.float64),
                va["z"].astype(np.float64),
            ])

        # ── read faces ───────────────────────────────────────────────────────
        c_fmt_char, c_sz = _PLY_STRUCT[face_count_ptype]
        i_fmt_char, i_sz = _PLY_STRUCT[face_idx_ptype]
        c_struct = struct.Struct(endian + c_fmt_char)

        tri_faces: List[List[int]] = []
        if fmt == "ascii":
            for _ in range(n_faces):
                row = fh.readline().decode("ascii").split()
                cnt = int(row[0])
                idxs = [int(row[1 + k]) for k in range(cnt)]
                for k in range(1, cnt - 1):          # fan tessellation
                    tri_faces.append([idxs[0], idxs[k], idxs[k + 1]])
        else:
            for _ in range(n_faces):
                cnt = c_struct.unpack(fh.read(c_sz))[0]
                raw_idxs = fh.read(cnt * i_sz)
                idxs = list(struct.unpack(endian + i_fmt_char * cnt, raw_idxs))
                for k in range(1, cnt - 1):           # fan tessellation
                    tri_faces.append([idxs[0], idxs[k], idxs[k + 1]])

    faces_arr = (
        np.array(tri_faces, dtype=np.int32)
        if tri_faces
        else np.empty((0, 3), dtype=np.int32)
    )
    print(
        f"  PLY mesh '{os.path.basename(path)}': "
        f"{n_verts} vertices, {len(faces_arr)} triangles "
        f"(from {n_faces} face records)"
    )
    return verts.astype(np.float64), faces_arr


# ---------------------------------------------------------------------------
# Publication figures (convergence + error summary)
# ---------------------------------------------------------------------------

def make_figures(run_dir: str, figures_dir: str, final_metrics: dict) -> None:
    """Generate Poisson convergence and error-summary figures.

    Mirrors the figure output of elder_rabbit_paraview.py.

    Parameters
    ----------
    run_dir:
        Directory that contains *_history.json and *_metrics.json.
    figures_dir:
        Output directory for PDF/PNG figures.
    final_metrics:
        Dict with keys 'rel_l2' (float, 0-1) and 'max_err' (float).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style(fontsize=9, linewidth=1.5)

    hist = _load_json(run_dir, "_history.json")
    if not hist:
        print(f"  No *_history.json found in {run_dir} — skipping figures.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    iters = np.arange(1, len(next(iter(hist.values()))) + 1)

    # ---- Figure 1: convergence (2×2) --------------------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 1.1),
        constrained_layout=True,
    )
    axes = axes.ravel()

    panels = [
        ("global_residual", "PDE (Laplacian) loss"),
        ("bc_loss",         "Boundary condition loss"),
        ("interface_flux",  "Interface flux residual"),
        ("rel_l2_eval",     "Relative L² error (%)"),
    ]
    for pi, (key, ylabel) in enumerate(panels):
        ax = axes[pi]
        if key in hist:
            vals = np.asarray(hist[key], dtype=float)
            if key == "rel_l2_eval":
                vals = vals * 100.0   # convert to %
            ax.plot(iters[:len(vals)], vals,
                    color=PUB_COLORS[0], linewidth=1.5)
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.text(0.02, 0.97, f"({chr(ord('a') + pi)})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold")

    fig.suptitle("Poisson on rabbit — Schwarz convergence",
                 fontsize=10, fontweight="bold")
    for ext in ("pdf", "png"):
        path = os.path.join(figures_dir, f"poisson_rabbit_convergence.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)

    # ---- Figure 2: error summary bar chart --------------------------------
    fig2, ax2 = plt.subplots(
        figsize=(SINGLE_COL_W * 1.2, SINGLE_COL_W * GOLDEN * 1.5),
        constrained_layout=True,
    )

    bar_labels = ["Rel-L² error (%)", "Max abs. error"]
    bar_vals   = [final_metrics.get("rel_l2", 0.0) * 100.0,
                  final_metrics.get("max_err", 0.0)]
    bar_colors = [PUB_COLORS[0], PUB_COLORS[1]]

    bars = ax2.bar(range(len(bar_vals)), bar_vals,
                   color=bar_colors, edgecolor="white", linewidth=0.5)
    for rect, val in zip(bars, bar_vals):
        ax2.annotate(
            f"{val:.3g}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    ax2.set_xticks(range(len(bar_labels)))
    ax2.set_xticklabels(bar_labels, rotation=10, ha="right")
    ax2.set_ylabel("Error")
    ax2.set_title("Poisson rabbit — final errors",
                  fontsize=9, fontweight="bold")

    for ext in ("pdf", "png"):
        path = os.path.join(figures_dir, f"poisson_rabbit_error_summary.{ext}")
        fig2.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig2)


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

    n_points = points_physical.shape[0]
    n_charts = int(chart_id.max()) + 1

    # Optional fields — present in most runs but may be absent in older NPZs
    blend_weight  = np.asarray(
        data["blend_weight"] if "blend_weight" in data else np.zeros(n_points),
        dtype=float).reshape(-1)
    interface_res = np.asarray(
        data["interface_residual"] if "interface_residual" in data else np.zeros(n_points),
        dtype=float).reshape(-1)

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

    # ---- Optional surface mesh --------------------------------------------
    surface_path: Optional[str] = None
    if args.surface:
        from scipy.spatial import cKDTree

        if args.ply_file:
            # ── PLY mesh path (recommended for Stanford Bunny) ─────────────
            # Uses original mesh connectivity → preserves ears, legs, and all
            # thin features that voxel-occupancy reconstruction loses.
            print(f"\nLoading PLY mesh for surface: {args.ply_file}")
            if not os.path.isfile(args.ply_file):
                raise FileNotFoundError(
                    f"--ply-file not found: {args.ply_file}"
                )
            verts_phys, faces = parse_ply_mesh(args.ply_file)
            print(
                f"  PLY bounding box: "
                f"x=[{verts_phys[:,0].min():.4f}, {verts_phys[:,0].max():.4f}]  "
                f"y=[{verts_phys[:,1].min():.4f}, {verts_phys[:,1].max():.4f}]  "
                f"z=[{verts_phys[:,2].min():.4f}, {verts_phys[:,2].max():.4f}]"
            )
        else:
            # ── Voxel-occupancy fallback (original behaviour) ──────────────
            print(
                f"\nExtracting rabbit surface from interior points "
                f"(grid {args.surface_grid}³, sigma={args.surface_smooth_sigma}, "
                f"close_iters={args.surface_close_iters}) ..."
            )
            verts_norm, faces = extract_surface_from_points(
                points_norm,
                grid_size=args.surface_grid,
                close_iters=args.surface_close_iters,
                smooth_sigma=args.surface_smooth_sigma,
            )
            # Convert SDF-normalised verts to physical space
            verts_phys = to_physical(verts_norm, center, scale)

        # Map scalar fields to surface vertices via nearest-neighbour from
        # the interior point cloud (works for both PLY and voxel surfaces).
        print(f"  Mapping {len(point_data)} scalar fields to surface vertices "
              f"(KD-tree over {len(points_physical)} interior points) ...")
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

    # ---- Publication figures ----------------------------------------------
    if not args.skip_figures:
        run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.solution_npz))
        print(f"\nGenerating convergence figures → {args.figures_dir}")
        make_figures(run_dir, args.figures_dir,
                     {"rel_l2": rel_l2, "max_err": max_err})

    print("\nDone.")


if __name__ == "__main__":
    main()
