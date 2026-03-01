#!/usr/bin/env python3
"""
Generate publication-quality 3-D geometry figures directly in Python.

Two rendering back-ends are used:
  • PyVista (VTK-based) — off-screen PNG renders with SSAO/ambient-occlusion,
    custom camera angles, multi-panel layouts, scalar bar, etc.
  • Plotly — interactive 3-D HTML figures (open in any browser) plus
    static PNG exports.

Both back-ends work headlessly (no display required).

Figures produced
────────────────
  1. rabbit_poisson_3d.png / .html
       Interior solution cloud coloured by u_error_mag
       (4 panels: iso view, top, front, side)
  2. rabbit_poisson_chart_mosaic.png
       All 12 atlas charts as individual coloured sub-plots (PyVista)
  3. rabbit_elder_3d.png / .html
       Interior Elder fields: pressure + concentration
  4. torus_inverse_3d.png / .html
       Torus surface coloured by traction_error_mag, deformed overlay
  5. torus_inverse_charts_3d.png
       8 torus atlas charts with individual colours

Output directory:  figures/3d/

Usage (defaults — uses the pre-built NPZ and VTK files):
    python postprocessing/render_3d_figures.py

    # Override paths:
    python postprocessing/render_3d_figures.py \\
        --poisson-npz  runs/bunny_poisson/rabbit_poisson_schwarz_bunny_vol_solution.npz \\
        --atlas-npz    runs/atlas_bunny_vol/rabbit_atlas_data.npz \\
        --ply-file     runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply \\
        --elder-dir    runs/rabbit_inverse_elder_globalfield_small \\
        --elder-atlas  runs/atlas_schwarz_20260213_002517/rabbit_atlas_data.npz \\
        --torus-dir    runs/torus_inverse_mps_dense_v4 \\
        --output-dir   figures/3d
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import PUB_COLORS, set_pub_style

# ---------------------------------------------------------------------------
# Defaults  (updated to Stanford Bunny domain — bunny_vol atlas, scale≈0.156)
# ---------------------------------------------------------------------------
_POISSON_NPZ   = "runs/bunny_poisson/rabbit_poisson_schwarz_bunny_vol_solution.npz"
_POISSON_VTR   = "runs/bunny_poisson/paraview/rabbit_poisson_bunny_grid.vtr"
_ATLAS_NPZ     = "runs/atlas_bunny_vol/rabbit_atlas_data.npz"
_ELDER_DIR     = "runs/rabbit_inverse_elder_globalfield_small"
_ELDER_ATLAS   = "runs/atlas_schwarz_20260213_002517/rabbit_atlas_data.npz"
_TORUS_DIR     = "runs/torus_inverse_mps_dense_v4"
_OUTPUT_DIR    = "figures/3d"

# chart palette (12 colours for rabbit, 8 for torus)
_CHART_COLORS_12 = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff",
]
_CHART_COLORS_8 = _CHART_COLORS_12[:8]


# ---------------------------------------------------------------------------
# SDF helpers (same as paraview scripts)
# ---------------------------------------------------------------------------

def load_sdf_transform(atlas_npz: Optional[str]) -> Tuple[np.ndarray, float]:
    if atlas_npz is None or not os.path.isfile(atlas_npz):
        return np.zeros(3, dtype=float), 1.0
    d = np.load(atlas_npz, allow_pickle=True)
    center = np.asarray(d["center"], dtype=float).reshape(3)
    scale  = float(np.asarray(d["scale"]).reshape(-1)[0])
    return center, scale


def to_physical(pts_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return center + scale * np.asarray(pts_norm, dtype=float)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 3-D geometry figures using PyVista and Plotly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--poisson-npz",  default=_POISSON_NPZ)
    p.add_argument("--poisson-vtr",  default=_POISSON_VTR,
                   help="VTR rectilinear grid for Poisson volume-slice renders")
    p.add_argument("--atlas-npz",    default=_ATLAS_NPZ)
    p.add_argument("--ply-file",     default=None,
                   help="Stanford Bunny PLY mesh (e.g. bun_zipper.ply).  When "
                        "supplied, surface figures use the original mesh connectivity "
                        "instead of the alpha-shape reconstruction from interior points, "
                        "preserving thin features like ears.  "
                        "Example: runs/atlas_schwarz_20260212_210412/downloads/bunny/"
                        "reconstruction/bun_zipper.ply")
    p.add_argument("--elder-dir",    default=_ELDER_DIR)
    p.add_argument("--elder-atlas",  default=_ELDER_ATLAS)
    p.add_argument("--torus-dir",    default=_TORUS_DIR)
    p.add_argument("--output-dir",   default=_OUTPUT_DIR)
    p.add_argument("--no-pyvista",   action="store_true", help="Skip PyVista renders")
    p.add_argument("--no-plotly",    action="store_true", help="Skip Plotly renders")
    return p.parse_args()


# ===========================================================================
# PLY mesh reader  (vertices + face connectivity)
# ===========================================================================

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

    Supports binary_little_endian, binary_big_endian, and ASCII PLY.
    Non-triangular faces are tessellated by fan from first vertex.

    Returns
    -------
    verts : (V, 3) float64  — vertex (x, y, z) in the PLY coordinate system
    faces : (F, 3) int32   — triangle connectivity
    """
    with open(path, "rb") as fh:
        header_lines: List[str] = []
        while True:
            raw = fh.readline()
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        for l in header_lines:
            if l.startswith("format"):
                parts = l.split()
                if len(parts) >= 2:
                    fmt = parts[1]
                break
        endian = "<" if "little" in fmt else (">" if "big" in fmt else "=")

        n_verts: int = 0
        n_faces: int = 0
        vert_props: List[Tuple[str, str]] = []
        face_count_ptype = "uchar"
        face_idx_ptype   = "int"
        cur_elem = None
        for l in header_lines:
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1]);  cur_elem = "vertex"
            elif l.startswith("element face"):
                n_faces = int(l.split()[-1]);  cur_elem = "face"
            elif l.startswith("element"):
                cur_elem = "other"
            elif l.startswith("property list") and cur_elem == "face":
                parts = l.split()
                face_count_ptype = parts[2];  face_idx_ptype = parts[3]
            elif l.startswith("property") and "list" not in l and cur_elem == "vertex":
                parts = l.split()
                vert_props.append((parts[2], parts[1]))

        if fmt == "ascii":
            x_col = next(i for i, (n, _) in enumerate(vert_props) if n == "x")
            y_col = next(i for i, (n, _) in enumerate(vert_props) if n == "y")
            z_col = next(i for i, (n, _) in enumerate(vert_props) if n == "z")
            verts = np.zeros((n_verts, 3), dtype=np.float64)
            for vi in range(n_verts):
                row = fh.readline().decode("ascii").split()
                verts[vi] = [float(row[x_col]), float(row[y_col]), float(row[z_col])]
        else:
            dt = np.dtype([(name, endian + _PLY_NP[ptype]) for name, ptype in vert_props])
            raw = fh.read(n_verts * dt.itemsize)
            va = np.frombuffer(raw, dtype=dt)
            verts = np.column_stack([va["x"].astype(np.float64),
                                     va["y"].astype(np.float64),
                                     va["z"].astype(np.float64)])

        c_fmt_char, c_sz = _PLY_STRUCT[face_count_ptype]
        i_fmt_char, i_sz = _PLY_STRUCT[face_idx_ptype]
        c_struct = struct.Struct(endian + c_fmt_char)
        tri_faces: List[List[int]] = []
        if fmt == "ascii":
            for _ in range(n_faces):
                row = fh.readline().decode("ascii").split()
                cnt = int(row[0]);  idxs = [int(row[1 + k]) for k in range(cnt)]
                for k in range(1, cnt - 1):
                    tri_faces.append([idxs[0], idxs[k], idxs[k + 1]])
        else:
            for _ in range(n_faces):
                cnt = c_struct.unpack(fh.read(c_sz))[0]
                idxs = list(struct.unpack(endian + i_fmt_char * cnt, fh.read(cnt * i_sz)))
                for k in range(1, cnt - 1):
                    tri_faces.append([idxs[0], idxs[k], idxs[k + 1]])

    faces_arr = (np.array(tri_faces, dtype=np.int32) if tri_faces
                 else np.empty((0, 3), dtype=np.int32))
    print(f"  PLY mesh '{os.path.basename(path)}': "
          f"{n_verts} verts, {len(faces_arr)} triangles")
    return verts.astype(np.float64), faces_arr


def _load_ply_surface(
    ply_path: str,
    pts_phys: np.ndarray,
    scalars_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load a PLY mesh and map interior-point scalars onto surface vertices.

    Parameters
    ----------
    ply_path : str
        Path to the PLY file (physical coordinates).
    pts_phys : (N, 3)
        Interior atlas points in physical space.
    scalars_dict : dict[str, (N,) array]
        Per-interior-point scalar arrays to map onto the surface.

    Returns
    -------
    ply_verts : (V, 3) float64   — PLY vertex positions (physical space)
    ply_faces : (F, 3) int32     — PLY triangle face indices
    surf_data : dict             — scalars interpolated onto PLY vertices via
                                   nearest-neighbour from interior pts_phys
    """
    from scipy.spatial import cKDTree
    ply_verts, ply_faces = parse_ply_mesh(ply_path)
    tree = cKDTree(pts_phys)
    _, nn_idx = tree.query(ply_verts)
    surf_data = {k: v[nn_idx] for k, v in scalars_dict.items()}
    print(f"  PLY surface: {len(ply_verts)} verts, {len(ply_faces)} faces; "
          f"scalars mapped from {len(pts_phys)} interior pts")
    return ply_verts, ply_faces, surf_data


def _pv_surf_from_ply(ply_verts: np.ndarray,
                       ply_faces: np.ndarray) -> "pyvista.PolyData":
    """Build a PyVista PolyData surface from PLY verts + faces."""
    import pyvista as pv
    faces_pv = np.hstack(
        [np.full((len(ply_faces), 1), 3, dtype=np.int32), ply_faces]
    ).ravel()
    return pv.PolyData(ply_verts.astype(np.float32), faces_pv)


# ===========================================================================
# PyVista rendering helpers
# ===========================================================================

def _pv_available() -> bool:
    try:
        import pyvista
        return True
    except ImportError:
        return False


def _make_pv_cloud(pts: np.ndarray, scalars: np.ndarray, scalar_name: str):
    """Create a PyVista PolyData point cloud with one scalar array."""
    import pyvista as pv
    cloud = pv.PolyData(pts.astype(np.float32))
    cloud[scalar_name] = scalars.astype(np.float32)
    return cloud


def pyvista_4panel(pts: np.ndarray, scalars: np.ndarray, scalar_name: str,
                   cmap: str, scalar_label: str,
                   title: str, output_path: str,
                   log_scale: bool = False,
                   point_size: float = 3.0,
                   y_up: bool = False) -> None:
    """Render a 4-panel (iso, top, front, side) view of a point cloud.

    Camera is auto-fitted to the actual data bounds so the geometry fills
    the viewport regardless of coordinate scale.

    Parameters
    ----------
    y_up : bool
        When True the camera uses Y as the up axis (Stanford bunny convention).
        When False (default) the standard Z-up convention is used.
    """
    import pyvista as pv
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    cloud = _make_pv_cloud(pts, scalars, scalar_name)

    # View methods: each panel uses a different orthographic angle
    if y_up:
        # Y is the vertical axis (ears point +Y, feet at -Y).
        # view_vector(direction, viewup) looks from that direction; reset_camera
        # then auto-fits the zoom while preserving the orientation.
        view_fns = [
            ("Isometric",  lambda p: p.view_vector([1, 0.7, 1],  viewup=(0, 1, 0))),
            ("Front (Z-)", lambda p: p.view_vector([0, 0, 1],    viewup=(0, 1, 0))),
            ("Side (X+)",  lambda p: p.view_vector([1, 0, 0],    viewup=(0, 1, 0))),
            ("Top (Y↑)",   lambda p: p.view_vector([0, 1, 0],    viewup=(0, 0, -1))),
        ]
    else:
        view_fns = [
            ("Isometric",  lambda p: p.view_isometric()),
            ("Top (Z↑)",   lambda p: p.view_xy()),
            ("Front (Y)",  lambda p: p.view_xz()),
            ("Side (X)",   lambda p: p.view_yz()),
        ]

    pl = pv.Plotter(shape=(2, 2), off_screen=True,
                    window_size=(1600, 1200))
    pl.set_background("white")

    plot_kwargs = dict(
        scalars=scalar_name,
        cmap=cmap,
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar_label, "n_labels": 5,
                         "fmt": "%.2e", "title_font_size": 14,
                         "label_font_size": 11},
        log_scale=log_scale,
    )

    for idx, (name, view_fn) in enumerate(view_fns):
        r, c = divmod(idx, 2)
        pl.subplot(r, c)
        pl.add_mesh(cloud, **plot_kwargs)
        view_fn(pl)
        pl.reset_camera()
        pl.add_text(name, font_size=10, color="black")

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista 4-panel: {output_path}")


def pyvista_chart_mosaic(pts: np.ndarray, chart_id: np.ndarray,
                          n_charts: int, colors: List[str],
                          title: str, output_path: str,
                          point_size: float = 3.0,
                          y_up: bool = False) -> None:
    """One sub-plot per chart in its assigned colour.

    Each sub-panel shows the FULL point cloud (gray, small) so the chart
    context is clear, with the individual chart highlighted in its colour.
    Camera is reset per sub-plot so the whole cloud fills the frame.
    """
    import pyvista as pv
    pv.global_theme.background = "white"

    ncols = 4
    nrows = int(np.ceil(n_charts / ncols))
    pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                    window_size=(ncols * 500, nrows * 400))
    pl.set_background("white")

    full_cloud = pv.PolyData(pts.astype(np.float32))  # background context

    for cid in range(n_charts):
        r, c = divmod(cid, ncols)
        pl.subplot(r, c)
        # light gray background of all points
        pl.add_mesh(full_cloud, color="#dddddd",
                    point_size=max(1, point_size - 1),
                    render_points_as_spheres=True, opacity=0.3)
        mask = chart_id == cid
        if mask.sum() > 0:
            chart_cloud = pv.PolyData(pts[mask].astype(np.float32))
            pl.add_mesh(chart_cloud,
                        color=colors[cid % len(colors)],
                        point_size=point_size,
                        render_points_as_spheres=True)
        if y_up:
            pl.view_vector([1, 0.7, 1], viewup=(0, 1, 0))
        else:
            pl.view_isometric()
        pl.reset_camera()
        pl.add_text(f"Chart {cid}", font_size=9, color="black")

    # Hide empty subplots
    for cid in range(n_charts, nrows * ncols):
        r, c = divmod(cid, ncols)
        pl.subplot(r, c)
        pl.add_text("", font_size=1)

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista chart mosaic: {output_path}")


def pyvista_two_scalars(pts: np.ndarray,
                         scalar1: np.ndarray, name1: str, cmap1: str,
                         scalar2: np.ndarray, name2: str, cmap2: str,
                         suptitle: str, output_path: str,
                         point_size: float = 3.0,
                         y_up: bool = False) -> None:
    """Two-panel side-by-side view of the same point cloud with different scalars."""
    import pyvista as pv
    pv.global_theme.background = "white"

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 700))
    pl.set_background("white")

    for col, (scalars, sname, cmap) in enumerate(
        [(scalar1, name1, cmap1), (scalar2, name2, cmap2)]
    ):
        pl.subplot(0, col)
        cloud = _make_pv_cloud(pts, scalars, sname)
        pl.add_mesh(cloud,
                    scalars=sname,
                    cmap=cmap,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": sname.replace("_", " "),
                                     "fmt": "%.2e",
                                     "title_font_size": 13,
                                     "label_font_size": 10})
        if y_up:
            pl.view_vector([1, 0.7, 1], viewup=(0, 1, 0))
        else:
            pl.view_isometric()
        pl.reset_camera()
        pl.add_text(sname.replace("_", " "), font_size=11, color="black")

    pl.add_title(suptitle, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista two-scalar: {output_path}")


def pyvista_rabbit_surface(pts: np.ndarray, scalars: np.ndarray,
                            scalar_name: str, cmap: str, scalar_label: str,
                            title: str, output_path: str,
                            y_up: bool = True,
                            alpha: Optional[float] = None) -> None:
    """Reconstruct the rabbit surface from interior points using Delaunay 3D
    alpha-shapes, then render the resulting surface mesh coloured by the
    scalar field.

    The Delaunay 3-D alpha-shape keeps only those tetrahedra whose circumsphere
    radius ≤ ``alpha``.  Extracting the outer boundary of the tetrahedral mesh
    gives a tight triangulated surface that follows all concavities of the
    rabbit (neck, inter-ear gap, paws).  Scalar values are mapped from the
    original cloud to the surface vertices by nearest-neighbour look-up.

    Parameters
    ----------
    alpha : float or None
        Alpha-shape parameter (VTK units).  ``None`` (default) estimates it
        automatically from point density: alpha ≈ 4 × average point spacing.
    y_up : bool
        Use Y as the vertical axis (Stanford bunny convention).
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    # ---- Estimate alpha from point density if not given --------------------
    if alpha is None:
        extent      = pts.max(axis=0) - pts.min(axis=0)
        volume_est  = np.prod(extent) * 0.45          # ~45 % fill for rabbit
        avg_spacing = (volume_est / max(len(pts), 1)) ** (1.0 / 3.0)
        alpha = 4.0 * avg_spacing
        print(f"  Alpha-shape: auto alpha = {alpha:.4f}  "
              f"({len(pts)} pts, avg spacing ≈ {avg_spacing:.4f})")

    cloud = pv.PolyData(pts.astype(np.float32))
    cloud[scalar_name] = scalars.astype(np.float32)

    # ---- Surface reconstruction --------------------------------------------
    print(f"  Building Delaunay 3-D alpha-shape (alpha={alpha:.4f}) …")
    tet     = cloud.delaunay_3d(alpha=alpha)
    surface = tet.extract_surface(algorithm=None).clean()
    print(f"  Surface: {surface.n_points} vertices, {surface.n_cells} cells")

    # Map scalars to surface by nearest-neighbour (alpha-shape may add vertices
    # whose exact position isn't in the original cloud)
    tree     = cKDTree(pts)
    _, idx   = tree.query(np.asarray(surface.points))
    surface[scalar_name] = scalars[idx].astype(np.float32)

    # ---- Camera configurations ---------------------------------------------
    if y_up:
        view_fns = [
            ("Isometric",  lambda p: p.view_vector([1, 0.7, 1], viewup=(0, 1, 0))),
            ("Front (Z-)", lambda p: p.view_vector([0, 0, 1],   viewup=(0, 1, 0))),
            ("Side (X+)",  lambda p: p.view_vector([1, 0, 0],   viewup=(0, 1, 0))),
            ("Top (Y↑)",   lambda p: p.view_vector([0, 1, 0],   viewup=(0, 0, -1))),
        ]
    else:
        view_fns = [
            ("Isometric",  lambda p: p.view_isometric()),
            ("Top (Z↑)",   lambda p: p.view_xy()),
            ("Front (Y)",  lambda p: p.view_xz()),
            ("Side (X)",   lambda p: p.view_yz()),
        ]

    plot_kwargs = dict(
        scalars=scalar_name,
        cmap=cmap,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar_label, "n_labels": 5,
                         "fmt": "%.2e", "title_font_size": 14,
                         "label_font_size": 11},
        smooth_shading=True,
    )

    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    for idx, (name, view_fn) in enumerate(view_fns):
        r, c = divmod(idx, 2)
        pl.subplot(r, c)
        pl.add_mesh(surface, **plot_kwargs)
        view_fn(pl)
        pl.reset_camera()
        pl.add_text(name, font_size=10, color="black")

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista rabbit surface: {output_path}")


def pyvista_ply_scalar_surface(
    ply_verts: np.ndarray,
    ply_faces: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str,
    cmap: str,
    scalar_label: str,
    title: str,
    output_path: str,
    y_up: bool = True,
) -> None:
    """Render a PLY surface mesh coloured by a scalar field (4-panel layout).

    Unlike ``pyvista_rabbit_surface`` which reconstructs the surface from
    interior points via Delaunay alpha-shapes, this function renders the
    original PLY mesh connectivity directly.  This preserves thin features
    such as rabbit ears that carry almost no interior points and are lost by
    the alpha-shape approach.

    Parameters
    ----------
    ply_verts, ply_faces : outputs of ``parse_ply_mesh``
    scalars : (V,) array of scalar values already interpolated onto PLY verts
    """
    import pyvista as pv
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    surf = _pv_surf_from_ply(ply_verts, ply_faces)
    surf[scalar_name] = scalars.astype(np.float32)

    if y_up:
        view_fns = [
            ("Isometric",  lambda p: p.view_vector([1, 0.7, 1], viewup=(0, 1, 0))),
            ("Front (Z-)", lambda p: p.view_vector([0, 0, 1],   viewup=(0, 1, 0))),
            ("Side (X+)",  lambda p: p.view_vector([1, 0, 0],   viewup=(0, 1, 0))),
            ("Top (Y↑)",   lambda p: p.view_vector([0, 1, 0],   viewup=(0, 0, -1))),
        ]
    else:
        view_fns = [
            ("Isometric", lambda p: p.view_isometric()),
            ("Top (Z↑)",  lambda p: p.view_xy()),
            ("Front (Y)", lambda p: p.view_xz()),
            ("Side (X)",  lambda p: p.view_yz()),
        ]

    plot_kwargs = dict(
        scalars=scalar_name, cmap=cmap,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar_label, "n_labels": 5,
                         "fmt": "%.2e", "title_font_size": 14,
                         "label_font_size": 11},
        smooth_shading=True,
    )

    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")
    for idx, (name, view_fn) in enumerate(view_fns):
        r, c = divmod(idx, 2)
        pl.subplot(r, c)
        pl.add_mesh(surf, **plot_kwargs)
        view_fn(pl)
        pl.reset_camera()
        pl.add_text(name, font_size=10, color="black")

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista PLY scalar surface: {output_path}")


def pyvista_ply_chart_surface(
    ply_verts: np.ndarray,
    ply_faces: np.ndarray,
    vert_chart_id: np.ndarray,
    n_charts: int,
    colors: List[str],
    title: str,
    output_path: str,
    y_up: bool = True,
) -> None:
    """Render PLY surface with chart assignment — one sub-panel per chart.

    For each chart, the full surface is shown in light gray and the faces
    belonging to that chart are highlighted in the chart's colour.  A face
    is assigned to a chart when the plurality of its three vertices belongs
    to that chart (tie goes to lower chart index).

    This replaces ``pyvista_chart_mosaic`` (interior-point blobs) with a
    proper surface view that reveals which part of the Stanford Bunny each
    atlas chart covers.
    """
    import pyvista as pv
    pv.global_theme.background = "white"

    # Assign each face to the chart whose id is most common among its 3 verts
    vert_ids = vert_chart_id[ply_faces]      # (F, 3) int
    # Use mode (plurality) per row — simple approach: take the chart id that
    # appears in ≥1 of the 3 vertices with smallest id on ties
    face_chart_id = np.array([
        np.bincount(row, minlength=n_charts).argmax()
        for row in vert_ids
    ], dtype=np.int32)

    full_surf = _pv_surf_from_ply(ply_verts, ply_faces)

    ncols = 4
    nrows = int(np.ceil(n_charts / ncols))
    pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                    window_size=(ncols * 500, nrows * 400))
    pl.set_background("white")

    for cid in range(n_charts):
        r, c = divmod(cid, ncols)
        pl.subplot(r, c)
        pl.add_mesh(full_surf, color="#dddddd", opacity=0.25, show_edges=False)

        mask = face_chart_id == cid
        if mask.sum() > 0:
            cf = ply_faces[mask]
            chart_faces_pv = np.hstack(
                [np.full((len(cf), 1), 3, dtype=np.int32), cf]
            ).ravel()
            chart_surf = pv.PolyData(ply_verts.astype(np.float32), chart_faces_pv)
            pl.add_mesh(chart_surf, color=colors[cid % len(colors)],
                        opacity=0.92, smooth_shading=True, show_edges=False)

        if y_up:
            pl.view_vector([1, 0.7, 1], viewup=(0, 1, 0))
        else:
            pl.view_isometric()
        pl.reset_camera()
        pl.add_text(f"Chart {cid}  ({(face_chart_id == cid).sum()} faces)",
                    font_size=8, color="black")

    for cid in range(n_charts, nrows * ncols):
        r, c = divmod(cid, ncols)
        pl.subplot(r, c)
        pl.add_text("", font_size=1)

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista PLY chart surface: {output_path}")


def pyvista_point_slices(pts: np.ndarray, scalars: np.ndarray,
                          scalar_name: str, cmap: str, scalar_label: str,
                          title: str, output_path: str,
                          y_up: bool = True,
                          slab_fraction: float = 0.12,
                          point_size: float = 5.0) -> None:
    """Render 3 orthogonal cross-section slices of an interior point cloud.

    The approach uses a thin slab of the original interior points around each
    midplane.  Because these points are already filtered to lie INSIDE the
    domain, the rabbit boundary appears naturally as the outer edge of the
    coloured region (blank/white outside the rabbit, coloured inside).

    Layout: 2×2
      [0,0] YZ slab (x≈mid) — viewed face-on from +X
      [0,1] XZ slab (y≈mid) — viewed face-on from +Y (top-down)
      [1,0] XY slab (z≈mid) — viewed face-on from +Z (front)
      [1,1] Isometric of all three slabs together

    Parameters
    ----------
    pts : ndarray, shape (N, 3)
        Interior point coordinates in physical space.
    scalars : ndarray, shape (N,)
        Scalar field values at each point.
    slab_fraction : float
        Half-thickness of each slab as a fraction of the axis range.
        Increase to include more points; decrease for sharper slices.
    y_up : bool
        Use Y as up axis (default True for Stanford bunny).
    """
    import pyvista as pv
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    mn  = pts.min(axis=0)
    mx  = pts.max(axis=0)
    rng = mx - mn
    mid = (mn + mx) / 2

    clim = [float(scalars.min()), float(scalars.max())]

    # Each slab: axis index, view direction, viewup vector, label
    if y_up:
        slab_defs = [
            (0, "YZ slab\n(x = mid)", [1, 0, 0], (0, 1, 0)),   # view from +X
            (1, "XZ slab\n(y = mid)", [0, 1, 0], (0, 0, -1)),  # view from +Y (top)
            (2, "XY slab\n(z = mid)", [0, 0, 1], (0, 1, 0)),   # view from +Z (front)
        ]
    else:
        slab_defs = [
            (0, "YZ slab\n(x = mid)", [1, 0, 0], (0, 0, 1)),
            (1, "XZ slab\n(y = mid)", [0, 1, 0], (0, 0, 1)),
            (2, "XY slab\n(z = mid)", [0, 0, 1], (0, 1, 0)),
        ]

    iso_up = (0, 1, 0) if y_up else (0, 0, 1)

    plot_kwargs = dict(
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar_label, "n_labels": 5,
                         "fmt": "%.2e", "title_font_size": 14,
                         "label_font_size": 11},
    )

    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    slab_clouds: list = []

    for idx, (axis, name, view_dir, up) in enumerate(slab_defs):
        half = slab_fraction * rng[axis]
        mask = np.abs(pts[:, axis] - mid[axis]) <= half
        slab_pts = pts[mask].astype(np.float32)
        slab_sc  = scalars[mask].astype(np.float32)

        cloud = pv.PolyData(slab_pts)
        cloud[scalar_name] = slab_sc
        slab_clouds.append(cloud)

        r, c = divmod(idx, 2)
        pl.subplot(r, c)
        pl.add_mesh(cloud, **plot_kwargs)
        pl.view_vector(view_dir, viewup=up)
        pl.reset_camera()
        pl.add_text(f"{name}  ({mask.sum()} pts)", font_size=9, color="black")

    # Panel [1,1]: all 3 slabs together, isometric view
    pl.subplot(1, 1)
    for cloud in slab_clouds:
        pl.add_mesh(cloud, **plot_kwargs)
    pl.view_vector([1, 0.7, 1], viewup=iso_up)
    pl.reset_camera()
    pl.add_text("All slabs (isometric)", font_size=9, color="black")

    pl.add_title(title, font_size=12, color="black")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved PyVista point slices: {output_path}")


# ===========================================================================
# Plotly rendering helpers
# ===========================================================================

def plotly_scatter3d(pts: np.ndarray, scalars: np.ndarray,
                      scalar_name: str, colorscale: str,
                      title: str, output_html: str,
                      output_png: Optional[str] = None,
                      marker_size: int = 2,
                      log_color: bool = False,
                      opacity: float = 0.85) -> None:
    """Create an interactive Plotly 3-D scatter plot."""
    import plotly.graph_objects as go

    color_vals = np.log10(scalars + 1e-14) if log_color else scalars
    color_label = f"log₁₀({scalar_name})" if log_color else scalar_name

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=color_vals,
            colorscale=colorscale,
            colorbar=dict(title=color_label, thickness=15),
            opacity=opacity,
        ),
        hovertemplate=(
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            f"<br>{scalar_name}=%{{marker.color:.3e}}<extra></extra>"
        ),
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"  Saved Plotly HTML: {output_html}")

    if output_png is not None:
        try:
            fig.write_image(output_png, width=1200, height=800, scale=2)
            print(f"  Saved Plotly PNG: {output_png}")
        except Exception as e:
            print(f"  WARNING: Could not save PNG ({e}). "
                  "Install kaleido: pip install kaleido")


def plotly_chart_scatter3d(pts: np.ndarray, chart_id: np.ndarray,
                            n_charts: int, colors: List[str],
                            title: str, output_html: str,
                            output_png: Optional[str] = None,
                            marker_size: int = 2,
                            opacity: float = 0.8) -> None:
    """Plotly 3-D scatter with one trace per atlas chart."""
    import plotly.graph_objects as go

    traces = []
    for cid in range(n_charts):
        mask = chart_id == cid
        if mask.sum() == 0:
            continue
        p = pts[mask]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="markers",
            name=f"Chart {cid}",
            marker=dict(
                size=marker_size,
                color=colors[cid % len(colors)],
                opacity=opacity,
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
        legend=dict(title="Chart", font=dict(size=10)),
    )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"  Saved Plotly chart HTML: {output_html}")
    if output_png:
        try:
            fig.write_image(output_png, width=1200, height=900, scale=2)
            print(f"  Saved Plotly chart PNG: {output_png}")
        except Exception as e:
            print(f"  WARNING: PNG export failed ({e})")


def _save_plotly_fig(fig, output_html: str,
                     output_png: Optional[str] = None,
                     w: int = 1200, h: int = 900) -> None:
    """Write a Plotly figure to HTML and optionally PNG."""
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"  Saved Plotly HTML: {output_html}")
    if output_png:
        try:
            fig.write_image(output_png, width=w, height=h, scale=2)
            print(f"  Saved Plotly PNG: {output_png}")
        except Exception as e:
            print(f"  WARNING: PNG export failed ({e})")


def plotly_ply_scalar_mesh3d(
    ply_verts: np.ndarray,
    ply_faces: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str,
    colorscale: str,
    title: str,
    output_html: str,
    output_png: Optional[str] = None,
    opacity: float = 0.95,
) -> None:
    """Interactive Plotly surface mesh (Mesh3d) coloured by a scalar field.

    Uses the original PLY face connectivity, so thin features like rabbit
    ears are rendered correctly.  Solution scalars must already be mapped
    onto PLY vertices (e.g. via ``_load_ply_surface``).
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Mesh3d(
        x=ply_verts[:, 0].tolist(),
        y=ply_verts[:, 1].tolist(),
        z=ply_verts[:, 2].tolist(),
        i=ply_faces[:, 0].tolist(),
        j=ply_faces[:, 1].tolist(),
        k=ply_faces[:, 2].tolist(),
        intensity=scalars.tolist(),
        colorscale=colorscale,
        colorbar=dict(title=scalar_name.replace("_", " "), thickness=15),
        opacity=opacity,
        name=scalar_name,
        showscale=True,
        hovertemplate=(
            "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}"
            f"<br>{scalar_name}=%{{intensity:.3e}}<extra></extra>"
        ),
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )
    _save_plotly_fig(fig, output_html, output_png)


def plotly_ply_chart_mesh3d(
    ply_verts: np.ndarray,
    ply_faces: np.ndarray,
    face_chart_id: np.ndarray,
    n_charts: int,
    colors: List[str],
    title: str,
    output_html: str,
    output_png: Optional[str] = None,
) -> None:
    """Interactive Plotly Mesh3d with one trace per atlas chart.

    Each chart's faces are drawn as a separate Mesh3d trace in its chart
    colour, enabling toggle-visibility in the legend.  ``face_chart_id``
    must be a per-FACE assignment (see ``pyvista_ply_chart_surface`` for
    how to compute it from per-vertex chart_id).
    """
    import plotly.graph_objects as go

    traces = []
    for cid in range(n_charts):
        cf = ply_faces[face_chart_id == cid]
        if len(cf) == 0:
            continue
        traces.append(go.Mesh3d(
            x=ply_verts[:, 0].tolist(),
            y=ply_verts[:, 1].tolist(),
            z=ply_verts[:, 2].tolist(),
            i=cf[:, 0].tolist(),
            j=cf[:, 1].tolist(),
            k=cf[:, 2].tolist(),
            color=colors[cid % len(colors)],
            opacity=0.92,
            name=f"Chart {cid}",
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
        legend=dict(title="Chart", font=dict(size=10)),
    )
    _save_plotly_fig(fig, output_html, output_png)


def plotly_two_scalars_3d(pts: np.ndarray,
                           scalars1: np.ndarray, name1: str, cmap1: str,
                           scalars2: np.ndarray, name2: str, cmap2: str,
                           title: str, output_html: str,
                           output_png: Optional[str] = None) -> None:
    """Two-subplot Plotly 3-D figure for two different scalar fields."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[name1.replace("_", " "), name2.replace("_", " ")],
    )

    for col, (scalars, name, cmap) in enumerate(
        [(scalars1, name1, cmap1), (scalars2, name2, cmap2)], start=1
    ):
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=scalars,
                    colorscale=cmap,
                    colorbar=dict(
                        title=name.replace("_", " "),
                        thickness=12,
                        x=0.45 if col == 1 else 1.0,
                    ),
                    opacity=0.85,
                ),
                showlegend=False,
            ),
            row=1, col=col,
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data"),
        scene2=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                    aspectmode="data"),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"  Saved Plotly two-scalar HTML: {output_html}")
    if output_png:
        try:
            fig.write_image(output_png, width=1600, height=800, scale=2)
            print(f"  Saved Plotly two-scalar PNG: {output_png}")
        except Exception as e:
            print(f"  WARNING: PNG export failed ({e})")


# ===========================================================================
# Matplotlib 2-D projection helpers (for volumetric interior point clouds)
# ===========================================================================

def matplotlib_volume_projections(
        pts: np.ndarray, scalars: np.ndarray,
        scalar_label: str, cmap: str,
        title: str, output_path: str,
        n_bins: int = 80,
        log_scale: bool = False,
        y_up: bool = True) -> None:
    """Render three 2-D mean-projection views of a volumetric interior point cloud.

    For each of the three face directions, all interior points are projected
    onto that plane and binned into a 2-D grid.  Each bin is coloured by the
    **mean** scalar value of the points that project into it; bins with no
    points appear white, making the rabbit boundary immediately visible.

    Unlike a 3-D scatter which forms an opaque solid mass, this gives a clean
    "CT-slice projection" style view that clearly shows both the domain shape
    and the field distribution.

    Layout (y_up=True, Stanford bunny with Y vertical):
      [0] Side view  — project onto YZ plane  (view from +X)
      [1] Front view — project onto XY plane  (view from +Z)
      [2] Top view   — project onto XZ plane  (view from +Y)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mn  = pts.min(axis=0)
    mx  = pts.max(axis=0)

    # Axes for each projection: (horiz_axis, vert_axis, xlabel, ylabel, title)
    if y_up:
        projections = [
            (2, 1, "z", "y", "Side view\n(project onto YZ, view from +X)"),
            (0, 1, "x", "y", "Front view\n(project onto XY, view from +Z)"),
            (0, 2, "x", "z", "Top view\n(project onto XZ, view from +Y)"),
        ]
    else:
        projections = [
            (0, 1, "x", "y", "XY plane (view from +Z)"),
            (0, 2, "x", "z", "XZ plane (view from +Y)"),
            (1, 2, "y", "z", "YZ plane (view from +X)"),
        ]

    sc = np.log10(scalars + 1e-14) if log_scale else scalars
    vmin, vmax = sc.min(), sc.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13)

    for ax, (ha, va, xl, yl, sub_title) in zip(axes, projections):
        h_vals = pts[:, ha]
        v_vals = pts[:, va]

        # Build bin edges
        h_edges = np.linspace(mn[ha], mx[ha], n_bins + 1)
        v_edges = np.linspace(mn[va], mx[va], n_bins + 1)

        # Digitise: bin index for each point (0 = out of range, clamp)
        hi = np.clip(np.searchsorted(h_edges, h_vals, side='right') - 1, 0, n_bins - 1)
        vi = np.clip(np.searchsorted(v_edges, v_vals, side='right') - 1, 0, n_bins - 1)

        # Accumulate sum and count
        grid_sum   = np.zeros((n_bins, n_bins), dtype=float)
        grid_count = np.zeros((n_bins, n_bins), dtype=int)
        np.add.at(grid_sum,   (vi, hi), sc)
        np.add.at(grid_count, (vi, hi), 1)

        # Mean: NaN where no data
        with np.errstate(invalid='ignore'):
            grid_mean = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

        im = ax.imshow(
            grid_mean,
            extent=[mn[ha], mx[ha], mn[va], mx[va]],
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label=scalar_label, shrink=0.85)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(sub_title, fontsize=10)
        ax.set_facecolor("white")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved matplotlib projections: {output_path}")


def matplotlib_chart_projections(
        pts: np.ndarray, chart_id: np.ndarray,
        n_charts: int, colors: List[str],
        title: str, output_path: str,
        n_bins: int = 80,
        y_up: bool = True) -> None:
    """Side-view and front-view 2-D projections of the volumetric atlas charts.

    Each chart is shown with its colour (density = fraction of bins occupied
    by that chart), on a light-gray background of the full cloud, giving a
    clear picture of how the 12 volumetric charts partition the rabbit interior.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)

    # Two projections: side (YZ) and front (XY)
    if y_up:
        projs = [
            (2, 1, "z", "y", "Side view (YZ)"),
            (0, 1, "x", "y", "Front view (XY)"),
        ]
    else:
        projs = [
            (0, 1, "x", "y", "XY"),
            (0, 2, "x", "z", "XZ"),
        ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)

    for ax, (ha, va, xl, yl, sub_title) in zip(axes, projs):
        h_edges = np.linspace(mn[ha], mx[ha], n_bins + 1)
        v_edges = np.linspace(mn[va], mx[va], n_bins + 1)

        # Background: occupied bins (any chart)
        hi_all = np.clip(np.searchsorted(h_edges, pts[:, ha], side='right') - 1, 0, n_bins - 1)
        vi_all = np.clip(np.searchsorted(v_edges, pts[:, va], side='right') - 1, 0, n_bins - 1)

        occ = np.zeros((n_bins, n_bins), dtype=bool)
        occ[vi_all, hi_all] = True

        # Plot light gray background for occupied bins
        bg = np.full((n_bins, n_bins, 4), [1., 1., 1., 0.], dtype=float)
        bg[occ] = [0.88, 0.88, 0.88, 1.]
        ax.imshow(bg, extent=[mn[ha], mx[ha], mn[va], mx[va]],
                  origin="lower", aspect="equal", interpolation="nearest")

        # Plot each chart on top
        for cid in range(n_charts):
            mask = chart_id == cid
            if mask.sum() == 0:
                continue
            hi = np.clip(np.searchsorted(h_edges, pts[mask, ha], side='right') - 1, 0, n_bins - 1)
            vi = np.clip(np.searchsorted(v_edges, pts[mask, va], side='right') - 1, 0, n_bins - 1)
            cnt = np.zeros((n_bins, n_bins), dtype=int)
            np.add.at(cnt, (vi, hi), 1)

            # Convert hex to RGBA
            r, g, b = tuple(int(colors[cid % len(colors)].lstrip('#')[i:i+2], 16)/255
                            for i in (0, 2, 4))
            overlay = np.zeros((n_bins, n_bins, 4), dtype=float)
            overlay[cnt > 0] = [r, g, b, 0.75]
            ax.imshow(overlay, extent=[mn[ha], mx[ha], mn[va], mx[va]],
                      origin="lower", aspect="equal", interpolation="nearest")

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(sub_title, fontsize=10)

    legend_patches = [Patch(color=colors[i % len(colors)], label=f"Chart {i}")
                      for i in range(n_charts)]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=min(n_charts, 6), fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved matplotlib chart projections: {output_path}")


# ===========================================================================
# Per-example rendering
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.  Rabbit Poisson
# ---------------------------------------------------------------------------

def render_rabbit_poisson(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.poisson_npz):
        print(f"  Skipping Poisson rabbit (NPZ not found: {args.poisson_npz})")
        return

    print("\n=== Rabbit Poisson 3-D figures ===")
    center, scale = load_sdf_transform(args.atlas_npz)
    data = np.load(args.poisson_npz, allow_pickle=True)
    pts = to_physical(np.asarray(data["points"], dtype=float), center, scale)
    u_pred      = np.asarray(data["u_pred"],        dtype=float).reshape(-1)
    u_error_mag = np.asarray(data["u_error_mag"],   dtype=float).reshape(-1)
    chart_id    = np.asarray(data["chart_id"],       dtype=np.int32).reshape(-1)
    n_charts    = int(chart_id.max()) + 1

    print(f"  {len(pts)} interior pts  |  n_charts={n_charts}  |  "
          f"physical x=[{pts[:,0].min():.4f},{pts[:,0].max():.4f}]  "
          f"y=[{pts[:,1].min():.4f},{pts[:,1].max():.4f}]")

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── Optionally load PLY mesh for surface renders ─────────────────────────
    # When --ply-file is given we use the original Stanford Bunny mesh topology
    # so thin features (ears, legs) are preserved.  Without it we fall back to
    # the Delaunay 3-D alpha-shape reconstruction from interior points.
    ply_verts:      Optional[np.ndarray] = None
    ply_faces:      Optional[np.ndarray] = None
    surf_u_err:     Optional[np.ndarray] = None
    surf_u_pred:    Optional[np.ndarray] = None
    face_chart_id:  Optional[np.ndarray] = None   # per-face chart assignment

    if args.ply_file and os.path.isfile(args.ply_file):
        scalars_map = {"u_error_mag": u_error_mag, "u_pred": u_pred,
                       "chart_id": chart_id.astype(float)}
        ply_verts, ply_faces, surf_data = _load_ply_surface(
            args.ply_file, pts, scalars_map)
        surf_u_err   = surf_data["u_error_mag"]
        surf_u_pred  = surf_data["u_pred"]
        vert_chart   = surf_data["chart_id"].round().astype(np.int32)
        # Per-face chart assignment: plurality vote over 3 vertex chart ids
        vert_ids_per_face = vert_chart[ply_faces]   # (F, 3)
        face_chart_id = np.array([
            np.bincount(row, minlength=n_charts).argmax()
            for row in vert_ids_per_face
        ], dtype=np.int32)
        print(f"  PLY surface loaded — {len(ply_verts)} verts, "
              f"{len(ply_faces)} faces, chart faces: "
              + ", ".join(f"C{c}={int((face_chart_id==c).sum())}"
                          for c in range(n_charts)))
    elif args.ply_file:
        print(f"  WARNING: --ply-file not found: {args.ply_file}  "
              "— falling back to alpha-shape surface reconstruction")

    # ── PyVista renders ───────────────────────────────────────────────────────
    if _pv_available() and not args.no_pyvista:
        if ply_verts is not None:
            # PLY-based surface — correct topology, preserves ears
            pyvista_ply_scalar_surface(
                ply_verts, ply_faces, surf_u_err, "u_error_mag",
                cmap="hot_r", scalar_label="|u error|",
                title="Rabbit Poisson — solution error on PLY surface",
                output_path=os.path.join(out, "rabbit_poisson_error_surface.png"),
                y_up=True,
            )
            pyvista_ply_scalar_surface(
                ply_verts, ply_faces, surf_u_pred, "u_pred",
                cmap="viridis", scalar_label="u predicted",
                title="Rabbit Poisson — predicted solution on PLY surface",
                output_path=os.path.join(out, "rabbit_poisson_upred_surface.png"),
                y_up=True,
            )
            pyvista_ply_chart_surface(
                ply_verts, ply_faces, face_chart_id, n_charts, _CHART_COLORS_12,
                title="Rabbit Poisson — atlas chart coverage on PLY surface",
                output_path=os.path.join(out, "rabbit_poisson_charts.png"),
                y_up=True,
            )
        else:
            # Fallback: Delaunay alpha-shape reconstruction from interior points
            pyvista_rabbit_surface(
                pts, u_error_mag, "u_error_mag",
                cmap="hot_r", scalar_label="|u error|",
                title="Rabbit Poisson — solution error (physical space)",
                output_path=os.path.join(out, "rabbit_poisson_error_surface.png"),
                y_up=True,
            )
            pyvista_rabbit_surface(
                pts, u_pred, "u_pred",
                cmap="viridis", scalar_label="u predicted",
                title="Rabbit Poisson — predicted solution (physical space)",
                output_path=os.path.join(out, "rabbit_poisson_upred_surface.png"),
                y_up=True,
            )
            pyvista_chart_mosaic(
                pts, chart_id, n_charts, _CHART_COLORS_12,
                title="Rabbit Poisson — 12 volumetric atlas charts (physical space)",
                output_path=os.path.join(out, "rabbit_poisson_charts.png"),
                point_size=2, y_up=True,
            )

    # ── Plotly interactive renders ────────────────────────────────────────────
    if not args.no_plotly:
        if ply_verts is not None:
            # PLY-based surface Mesh3d — crisp interactive renders with ears
            plotly_ply_scalar_mesh3d(
                ply_verts, ply_faces, surf_u_err, "u_error_mag",
                colorscale="Hot",
                title="Rabbit Poisson — |u error| on Stanford Bunny surface",
                output_html=os.path.join(out, "rabbit_poisson_error.html"),
                output_png=os.path.join(out, "rabbit_poisson_error_plotly.png"),
            )
            plotly_ply_scalar_mesh3d(
                ply_verts, ply_faces, surf_u_pred, "u_pred",
                colorscale="Viridis",
                title="Rabbit Poisson — predicted solution on Stanford Bunny surface",
                output_html=os.path.join(out, "rabbit_poisson_upred.html"),
                output_png=os.path.join(out, "rabbit_poisson_upred_plotly.png"),
            )
            plotly_ply_chart_mesh3d(
                ply_verts, ply_faces, face_chart_id, n_charts, _CHART_COLORS_12,
                title="Rabbit Poisson — atlas chart coverage on Stanford Bunny surface",
                output_html=os.path.join(out, "rabbit_poisson_charts.html"),
                output_png=os.path.join(out, "rabbit_poisson_charts_plotly.png"),
            )
        else:
            # Fallback: semi-transparent interior point cloud
            plotly_scatter3d(
                pts, u_error_mag, "u_error_mag",
                colorscale="Hot",
                title="Rabbit Poisson — |u error| (interactive 3-D)",
                output_html=os.path.join(out, "rabbit_poisson_error.html"),
                output_png=os.path.join(out, "rabbit_poisson_error_plotly.png"),
                opacity=0.25,
            )
            plotly_scatter3d(
                pts, u_pred, "u_pred",
                colorscale="Viridis",
                title="Rabbit Poisson — predicted solution (interactive 3-D)",
                output_html=os.path.join(out, "rabbit_poisson_upred.html"),
                output_png=os.path.join(out, "rabbit_poisson_upred_plotly.png"),
                opacity=0.25,
            )
            plotly_chart_scatter3d(
                pts, chart_id, n_charts, _CHART_COLORS_12,
                title="Rabbit Poisson — atlas charts (interactive 3-D)",
                output_html=os.path.join(out, "rabbit_poisson_charts.html"),
                output_png=os.path.join(out, "rabbit_poisson_charts_plotly.png"),
                opacity=0.25,
            )


# ---------------------------------------------------------------------------
# 2.  Rabbit Elder
# ---------------------------------------------------------------------------

def render_rabbit_elder(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.elder_dir):
        print(f"  Skipping Elder rabbit (dir not found: {args.elder_dir})")
        return

    print("\n=== Rabbit Elder 3-D figures ===")
    center, scale = load_sdf_transform(args.elder_atlas)

    # Find solution NPZ
    sol_path = None
    pv_path  = None
    for fn in sorted(os.listdir(args.elder_dir)):
        if fn.endswith("_solution.npz") and sol_path is None:
            sol_path = os.path.join(args.elder_dir, fn)
        if fn.endswith("_pressure_velocity_fields.npz") and pv_path is None:
            pv_path = os.path.join(args.elder_dir, fn)

    if sol_path is None:
        print(f"  No *_solution.npz found in {args.elder_dir} — skipping.")
        return

    sol = np.load(sol_path, allow_pickle=True)
    pts = to_physical(np.asarray(sol["points"], dtype=float), center, scale)

    fields: Dict[str, np.ndarray] = {}
    for k in ("p_pred", "p_error_mag", "c_pred", "c_error_mag"):
        if k in sol:
            fields[k] = np.asarray(sol[k], dtype=float).reshape(-1)

    if pv_path is not None:
        pv_data = np.load(pv_path, allow_pickle=True)
        for k in ("velocity_pred", "velocity_error_mag"):
            if k in pv_data:
                arr = np.asarray(pv_data[k], dtype=float)
                if arr.ndim > 1:
                    fields[k + "_mag"] = np.linalg.norm(arr, axis=1)
                else:
                    fields[k] = arr.reshape(-1)
        if "velocity_pred" in pv_data:
            fields["velocity_mag"] = np.linalg.norm(
                np.asarray(pv_data["velocity_pred"], dtype=float), axis=1)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # PyVista — Elder atlas uses surface points so the rabbit silhouette is
    # visible directly.  Use y_up=True so the rabbit appears right-side-up
    # (Stanford bunny: ears point in +Y direction).
    if _pv_available() and not args.no_pyvista and "p_pred" in fields:
        if "c_pred" in fields:
            pyvista_two_scalars(
                pts,
                fields["p_pred"], "pressure",  "RdBu_r",
                fields["c_pred"], "concentration", "viridis",
                suptitle="Rabbit Elder — pressure & concentration (physical space)",
                output_path=os.path.join(out, "rabbit_elder_fields.png"),
                point_size=3, y_up=True,
            )
        pyvista_4panel(
            pts, fields["p_pred"], "pressure",
            cmap="RdBu_r", scalar_label="pressure",
            title="Rabbit Elder — pressure field (physical space)",
            output_path=os.path.join(out, "rabbit_elder_pressure_4panel.png"),
            point_size=3, y_up=True,
        )
        if "p_error_mag" in fields:
            pyvista_4panel(
                pts, fields["p_error_mag"], "pressure_error",
                cmap="hot_r", scalar_label="|p error|",
                title="Rabbit Elder — pressure error (physical space)",
                output_path=os.path.join(out, "rabbit_elder_error_4panel.png"),
                point_size=3, y_up=True,
            )

    # Plotly
    if not args.no_plotly and "p_pred" in fields:
        plotly_two_scalars_3d(
            pts,
            fields["p_pred"], "pressure",  "RdBu",
            fields.get("c_pred", fields["p_pred"]), "concentration", "Viridis",
            title="Rabbit Elder — pressure & concentration (interactive)",
            output_html=os.path.join(out, "rabbit_elder_fields.html"),
            output_png=os.path.join(out, "rabbit_elder_fields_plotly.png"),
        )
        if "p_error_mag" in fields:
            plotly_scatter3d(
                pts, fields["p_error_mag"], "pressure_error_mag",
                colorscale="Hot",
                title="Rabbit Elder — pressure error (interactive)",
                output_html=os.path.join(out, "rabbit_elder_error.html"),
                output_png=os.path.join(out, "rabbit_elder_error_plotly.png"),
            )


# ---------------------------------------------------------------------------
# 3.  Torus inverse
# ---------------------------------------------------------------------------

def render_torus_inverse(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.torus_dir):
        print(f"  Skipping torus inverse (dir not found: {args.torus_dir})")
        return

    print("\n=== Torus inverse 3-D figures ===")
    # Find obs NPZ
    npz_path = None
    for fn in sorted(os.listdir(args.torus_dir)):
        if fn.endswith("_obs.npz"):
            npz_path = os.path.join(args.torus_dir, fn)
            break
    if npz_path is None:
        print(f"  No *_obs.npz in {args.torus_dir} — skipping.")
        return

    d = np.load(npz_path, allow_pickle=True)
    key_pts = "x_full" if "x_full" in d else "x_eval"
    pts     = np.asarray(d[key_pts], dtype=float)

    key_terr = "t_full_error"
    t_err = np.asarray(d[key_terr], dtype=float) if key_terr in d else None
    t_true = np.asarray(
        d["t_full_true" if "t_full_true" in d else "t_eval_true"], dtype=float)

    cw       = np.asarray(d["chart_weights_full" if "chart_weights_full" in d else "chart_weights_eval"], dtype=float)
    chart_id = np.argmax(cw, axis=1).astype(np.int32)
    n_charts = cw.shape[1]

    t_err_mag = np.linalg.norm(t_err, axis=1) if t_err is not None \
                else np.zeros(len(pts))

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # PyVista
    if _pv_available() and not args.no_pyvista:
        pyvista_4panel(
            pts, t_err_mag, "traction_error_mag",
            cmap="hot_r", scalar_label="|traction error|",
            title="Torus inverse neo-Hookean — traction error",
            output_path=os.path.join(out, "torus_inverse_error_4panel.png"),
            log_scale=False, point_size=5,
        )
        pyvista_chart_mosaic(
            pts, chart_id, n_charts, _CHART_COLORS_8,
            title="Torus inverse — 8 atlas charts",
            output_path=os.path.join(out, "torus_inverse_charts.png"),
            point_size=4,
        )
        # Deformed overlay (if present)
        if "x_full_deformed" in d and "u_full_mag" in d:
            import pyvista as pv
            pv.global_theme.background = "white"
            pts_ref = pts
            pts_def = np.asarray(d["x_full_deformed"], dtype=float)
            u_mag   = np.asarray(d["u_full_mag"], dtype=float).reshape(-1)

            pl = pv.Plotter(shape=(1, 2), off_screen=True,
                            window_size=(1600, 700))
            pl.set_background("white")

            center = pts_ref.mean(axis=0)
            cam = [(center[0], center[1], center[2] + 3.5),
                   tuple(center), (0, 1, 0)]

            pl.subplot(0, 0)
            ref_cloud = pv.PolyData(pts_ref.astype(np.float32))
            ref_cloud["u_mag"] = u_mag.astype(np.float32)
            pl.add_mesh(ref_cloud, scalars="u_mag", cmap="plasma",
                        point_size=4, render_points_as_spheres=True,
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "|u|", "fmt": "%.2e"})
            pl.view_isometric()
            pl.reset_camera()
            pl.add_text("Reference (coloured by |u|)", font_size=10, color="black")

            pl.subplot(0, 1)
            def_cloud = pv.PolyData(pts_def.astype(np.float32))
            def_cloud["u_mag"] = u_mag.astype(np.float32)
            pl.add_mesh(def_cloud, scalars="u_mag", cmap="plasma",
                        point_size=4, render_points_as_spheres=True,
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "|u|", "fmt": "%.2e"})
            pl.view_isometric()
            pl.reset_camera()
            pl.add_text("Deformed configuration", font_size=10, color="black")

            pl.add_title("Torus inverse — reference vs deformed", font_size=12)
            def_path = os.path.join(out, "torus_inverse_deformation.png")
            pl.screenshot(def_path)
            pl.close()
            print(f"  Saved PyVista deformation: {def_path}")

    # Plotly
    if not args.no_plotly:
        plotly_scatter3d(
            pts, t_err_mag, "traction_error_mag",
            colorscale="Hot",
            title="Torus inverse — |traction error| (interactive)",
            output_html=os.path.join(out, "torus_inverse_error.html"),
            output_png=os.path.join(out, "torus_inverse_error_plotly.png"),
        )
        plotly_chart_scatter3d(
            pts, chart_id, n_charts, _CHART_COLORS_8,
            title="Torus inverse — atlas charts (interactive)",
            output_html=os.path.join(out, "torus_inverse_charts.html"),
            output_png=os.path.join(out, "torus_inverse_charts_plotly.png"),
        )

        # Deformed vs reference overlay
        if "x_full_deformed" in d:
            import plotly.graph_objects as go
            pts_def = np.asarray(d["x_full_deformed"], dtype=float)
            u_mag   = np.asarray(d["u_full_mag"], dtype=float).reshape(-1) \
                      if "u_full_mag" in d else np.zeros(len(pts))
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", name="Reference",
                marker=dict(size=2, color=u_mag, colorscale="Plasma",
                            opacity=0.5, showscale=False),
            ))
            fig.add_trace(go.Scatter3d(
                x=pts_def[:, 0], y=pts_def[:, 1], z=pts_def[:, 2],
                mode="markers", name="Deformed",
                marker=dict(size=2, color=u_mag, colorscale="Plasma",
                            opacity=0.8,
                            colorbar=dict(title="|u|", thickness=12)),
            ))
            fig.update_layout(
                title="Torus inverse — reference & deformed",
                scene=dict(aspectmode="data"),
                template="plotly_white",
            )
            deform_html = os.path.join(out, "torus_inverse_deformation.html")
            fig.write_html(deform_html, include_plotlyjs="cdn")
            print(f"  Saved Plotly deformation HTML: {deform_html}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"PyVista available: {_pv_available()} (skip={args.no_pyvista})")

    try:
        import plotly
        plotly_ok = True
    except ImportError:
        plotly_ok = False
    print(f"Plotly available: {plotly_ok} (skip={args.no_plotly})")

    render_rabbit_poisson(args)
    render_rabbit_elder(args)
    render_torus_inverse(args)

    print(f"\nAll 3-D figures saved to: {args.output_dir}")
    print("\nHow to view:")
    print("  PNG files: open in any image viewer / include in LaTeX directly.")
    print("  HTML files: open in browser for interactive rotation/zoom.")
    print("  (Plotly HTML is self-contained — no server required.)")


if __name__ == "__main__":
    main()
