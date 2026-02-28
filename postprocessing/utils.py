#!/usr/bin/env python3
"""
Shared utilities for postprocessing and figure generation.

Provides:
- write_vtu_points()  : ASCII VTK UnstructuredGrid (point cloud) writer
- write_vtu_rectilinear_grid() : structured regular-grid VTU writer (for
  volumetric slice views in ParaView)
- set_pub_style()     : publication-quality matplotlib rcParams
- add_colorbar()      : helper to attach a colorbar to an Axes
- plot_convergence()  : generic multi-curve convergence plotter
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# VTK / ParaView export helpers
# ---------------------------------------------------------------------------

def write_vtu_points(
    path: str,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray],
    *,
    float_fmt: str = "%.8g",
) -> None:
    """Write an ASCII VTK UnstructuredGrid with VTK_VERTEX cells.

    Parameters
    ----------
    path:
        Output .vtu file path (parent directories are created automatically).
    points:
        (N, 3) array of 3-D coordinates.
    point_data:
        Dictionary mapping field names to either
        - 1-D array of length N (scalar field), or
        - (N, k) array (vector / tensor field with k components).
    float_fmt:
        printf-style format string used when writing floating-point data.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)  # 1 = VTK_VERTEX

    def _fmt(arr: np.ndarray) -> str:
        return " ".join(float_fmt % v for v in np.asarray(arr).reshape(-1))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        # ---- Points --------------------------------------------------------
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + _fmt(points) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        # ---- Cells ---------------------------------------------------------
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(map(str, connectivity)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(map(str, offsets)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(map(str, cell_types)) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        # ---- PointData -----------------------------------------------------
        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = arr.shape[1]
            else:
                raise ValueError(f"Unsupported data shape for '{name}': {arr.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + _fmt(arr) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def write_vtu_surface_mesh(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    point_data: Dict[str, np.ndarray],
    *,
    float_fmt: str = "%.8g",
) -> None:
    """Write an ASCII VTK UnstructuredGrid with VTK_TRIANGLE cells.

    Parameters
    ----------
    path:
        Output .vtu file path.
    vertices:
        (V, 3) array of vertex coordinates.
    faces:
        (F, 3) integer array of triangle vertex indices.
    point_data:
        Dictionary of field names → 1-D arrays of length V (scalar at each vertex).
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    nv = vertices.shape[0]
    nf = faces.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    connectivity = faces.reshape(-1)                          # 3*F indices
    offsets      = np.arange(3, 3 * nf + 1, 3, dtype=np.int32)
    cell_types   = np.full(nf, 5, dtype=np.uint8)             # 5 = VTK_TRIANGLE

    def _fmt(arr: np.ndarray) -> str:
        return " ".join(float_fmt % v for v in np.asarray(arr).reshape(-1))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{nv}" NumberOfCells="{nf}">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + _fmt(vertices) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(map(str, connectivity)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(map(str, offsets)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(map(str, cell_types)) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr, dtype=float)
            ncomp = 1 if arr.ndim == 1 else arr.shape[1]
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + _fmt(arr) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def write_vtu_rectilinear_grid(
    path: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    cell_data: Dict[str, np.ndarray],
    *,
    float_fmt: str = "%.8g",
) -> None:
    """Write an ASCII VTK RectilinearGrid for volumetric slice views.

    Parameters
    ----------
    path:
        Output .vtr file path.
    x_coords, y_coords, z_coords:
        1-D sorted coordinate arrays along each axis. The grid has
        shape (Nx, Ny, Nz) in Fortran (column-major) order.
    cell_data:
        Dictionary of field names → (Nx*Ny*Nz,) flat arrays (C-order:
        z varies fastest, then y, then x).
    float_fmt:
        printf-style float format string.
    """
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    n_total = nx * ny * nz

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _fmt(arr: np.ndarray) -> str:
        return " ".join(float_fmt % v for v in np.asarray(arr).reshape(-1))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')
        ext = f'0 {nx - 1} 0 {ny - 1} 0 {nz - 1}'
        f.write(f'  <RectilinearGrid WholeExtent="{ext}">\n')
        f.write(f'    <Piece Extent="{ext}">\n')
        f.write("      <Coordinates>\n")
        for name, coords in [("x_coords", x_coords), ("y_coords", y_coords), ("z_coords", z_coords)]:
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            f.write("          " + _fmt(coords) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </Coordinates>\n")
        f.write("      <PointData>\n")
        for name, arr in cell_data.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = arr.shape[1]
            else:
                raise ValueError(f"Unsupported data shape for '{name}': {arr.shape}")
            assert arr.shape[0] == n_total, (
                f"Field '{name}': expected {n_total} values, got {arr.shape[0]}"
            )
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + _fmt(arr) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </RectilinearGrid>\n")
        f.write("</VTKFile>\n")


def interpolate_to_grid(
    points: np.ndarray,
    fields: Dict[str, np.ndarray],
    *,
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
    method: str = "nearest",
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Scatter-interpolate point-cloud fields onto a regular Cartesian grid.

    Parameters
    ----------
    points:
        (N, 3) array of 3-D coordinates.
    fields:
        Dictionary of field-name → 1-D array (length N).
    nx, ny, nz:
        Number of grid points along x, y, z axes.
    method:
        Interpolation method: 'nearest' (fast) or 'linear'.
    fill_value:
        Value used for grid points outside the convex hull (linear only).

    Returns
    -------
    xg, yg, zg:
        1-D coordinate arrays of the output grid.
    grid_fields:
        Dictionary with same keys as *fields*; values are flat (C-order)
        arrays of length nx*ny*nz.
    """
    try:
        from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for interpolate_to_grid().")

    points = np.asarray(points, dtype=float)
    xg = np.linspace(points[:, 0].min(), points[:, 0].max(), nx)
    yg = np.linspace(points[:, 1].min(), points[:, 1].max(), ny)
    zg = np.linspace(points[:, 2].min(), points[:, 2].max(), nz)

    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing="ij")
    query_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    grid_fields: Dict[str, np.ndarray] = {}
    for name, vals in fields.items():
        vals = np.asarray(vals, dtype=float).reshape(-1)
        if method == "nearest":
            interp = NearestNDInterpolator(points, vals)
            grid_vals = interp(query_pts)
        else:
            interp = LinearNDInterpolator(points, vals, fill_value=fill_value)
            grid_vals = interp(query_pts)
        grid_fields[name] = grid_vals  # length nx*ny*nz, C-order (z fast)

    return xg, yg, zg, grid_fields


# ---------------------------------------------------------------------------
# Matplotlib publication style
# ---------------------------------------------------------------------------

# Journal-standard figure widths (inches)
SINGLE_COL_W = 3.5   # ~ 89 mm  (CMAME / IJNME single column)
DOUBLE_COL_W = 7.16  # ~182 mm  (CMAME / IJNME double column)
GOLDEN = 0.618


def set_pub_style(
    *,
    fontsize: float = 9.0,
    linewidth: float = 1.2,
    usetex: bool = False,
) -> None:
    """Apply publication-quality matplotlib rcParams.

    Parameters
    ----------
    fontsize:
        Base font size (pt).  Axis labels → fontsize + 1, ticks → fontsize - 1.
    linewidth:
        Default line width for curves.
    usetex:
        If True and a LaTeX installation is available, use LaTeX for text.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Palatino"],
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize + 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "legend.title_fontsize": fontsize - 1,
        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        # Lines
        "lines.linewidth": linewidth,
        "lines.markersize": 4,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        # LaTeX
        "text.usetex": usetex,
    })


# Publication colour cycle (colour-blind-friendly, print-safe)
PUB_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple-pink
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

PUB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
PUB_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def add_colorbar(
    fig,
    ax,
    im,
    *,
    label: str = "",
    shrink: float = 0.85,
    pad: float = 0.04,
    format_str: Optional[str] = None,
) -> Any:
    """Attach a colorbar to *ax* and return it."""
    import matplotlib.pyplot as plt

    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad)
    cb.set_label(label)
    if format_str is not None:
        from matplotlib.ticker import FormatStrFormatter
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(format_str))
    cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    return cb


# ---------------------------------------------------------------------------
# Generic convergence plot helper
# ---------------------------------------------------------------------------

def plot_convergence(
    ax,
    history: Dict[str, List[float]],
    keys: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    yscale: str = "log",
    xlabel: str = "Schwarz iteration",
    ylabel: str = "",
    colors: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    markers: Optional[Sequence[str]] = None,
    mark_min: bool = False,
    mark_min_key: Optional[str] = None,
) -> None:
    """Plot one or more convergence curves from a history dictionary.

    Parameters
    ----------
    ax:
        Matplotlib Axes instance.
    history:
        Dictionary mapping metric names to lists of values (one per
        iteration / epoch).
    keys:
        Which keys from *history* to plot.
    labels:
        Display labels for the legend (defaults to *keys*).
    yscale:
        Axis scale: 'log' or 'linear'.
    mark_min:
        If True, annotate the minimum of the first key with a vertical line.
    mark_min_key:
        If provided, annotate the minimum of this key instead.
    """
    if colors is None:
        colors = PUB_COLORS
    if linestyles is None:
        linestyles = PUB_LINESTYLES
    if markers is None:
        markers = [None] * len(keys)  # type: ignore[assignment]
    if labels is None:
        labels = list(keys)

    for i, (key, label) in enumerate(zip(keys, labels)):
        if key not in history:
            continue
        vals = np.asarray(history[key], dtype=float)
        iters = np.arange(1, len(vals) + 1)
        ax.plot(
            iters,
            vals,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],  # type: ignore[index]
            markevery=max(1, len(vals) // 10),
            label=label,
        )

    if mark_min:
        ref_key = mark_min_key or keys[0]
        if ref_key in history:
            vals = np.asarray(history[ref_key], dtype=float)
            idx_best = int(np.argmin(vals))
            ax.axvline(idx_best + 1, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"iter {idx_best + 1}",
                xy=(idx_best + 1, vals[idx_best]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=plt.rcParams["xtick.labelsize"],
                color="gray",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.legend(loc="best")


# ---------------------------------------------------------------------------
# Convenience: load history + metrics JSON
# ---------------------------------------------------------------------------

def load_run(run_dir: str, prefix: str) -> Tuple[Dict, Dict]:
    """Load *_history.json and *_metrics.json from a run directory.

    Returns (history_dict, metrics_dict).  Missing files return empty dicts.
    """
    import json

    hist_path = os.path.join(run_dir, f"{prefix}_history.json")
    metr_path = os.path.join(run_dir, f"{prefix}_metrics.json")

    history: Dict = {}
    if os.path.isfile(hist_path):
        with open(hist_path) as fh:
            history = json.load(fh)

    metrics: Dict = {}
    if os.path.isfile(metr_path):
        with open(metr_path) as fh:
            metrics = json.load(fh)

    return history, metrics
