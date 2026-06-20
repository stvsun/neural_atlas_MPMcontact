"""3-D PyVista visualization of two evolving rough rock-joint surfaces under shear.

Renders an upper and a lower height-field surface (e.g. the two mating faces of a
fractal rock joint) as VTK structured grids, colored by a per-vertex field (traction,
gap, contact pressure, ...). Supports a single professional still and an animated GIF
sequence showing the surfaces sliding past one another with an exploded aperture.

Environment notes (pinned versions: pyvista 0.38.6, vtk 9.3.1, matplotlib 3.4.1,
imageio 2.6.1 with NO imageio-ffmpeg). The implementation bakes in the workarounds:

  * StructuredGrid height fields use Fortran-order point/scalar raveling with
    ``dimensions = [nx, ny, 1]``.
  * Colormaps are always passed as matplotlib ``Colormap`` *objects*, never strings
    (string cmaps raise on this matplotlib build).
  * Animation defaults to GIF (``open_gif`` / ``write_frame`` / ``close``); mp4 is not
    available because imageio-ffmpeg is absent and is offered only behind a guarded
    fallback.
  * Off-screen rendering throughout (``off_screen=True``), white background, FXAA
    anti-aliasing, depth peeling for translucent overlaps.

Run as a script for a self-contained synthetic self-test::

    PYTHONPATH=. python3 postprocessing/surface_anim_3d.py
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless backend; we only need colormap objects
import matplotlib.cm as cm

import pyvista as pv


# ---------------------------------------------------------------------------
# Colormap helper
# ---------------------------------------------------------------------------
def _resolve_cmap(cmap) -> "matplotlib.colors.Colormap":
    """Return a matplotlib Colormap *object*.

    PyVista 0.38.6 on matplotlib 3.4.1 raises if a colormap *string* is passed
    through to the VTK lookup-table builder, so we always materialize an object.
    Accepts either a name (str) or an already-resolved Colormap.
    """
    if isinstance(cmap, str):
        return cm.get_cmap(cmap)
    return cmap


# ---------------------------------------------------------------------------
# Structured-grid construction / update
# ---------------------------------------------------------------------------
def make_height_surface(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    name: str = "scalar",
) -> pv.StructuredGrid:
    """Build a :class:`pyvista.StructuredGrid` from a height field ``Z(x, y)``.

    Parameters
    ----------
    x, y : 1-D arrays of length ``nx`` and ``ny`` (the grid axes).
    Z : 2-D array of shape ``(nx, ny)`` giving the surface elevation; indexed
        ``Z[i, j]`` for ``x[i]``, ``y[j]`` (i.e. ``indexing='ij'``).
    scalars : optional 2-D array shaped like ``Z`` used to color the surface.
        Defaults to ``Z`` itself when omitted.
    name : key under which the scalar field is stored on the grid.

    Returns
    -------
    pyvista.StructuredGrid with ``dimensions = [nx, ny, 1]`` (single z-layer)
    and Fortran-order points so the VTK topology matches the ``ij`` field.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    nx, ny = Z.shape
    if x.shape[0] != nx or y.shape[0] != ny:
        raise ValueError(
            f"x ({x.shape[0]}), y ({y.shape[0]}) must match Z {Z.shape}"
        )

    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.c_[
        X.ravel(order="F"),
        Y.ravel(order="F"),
        Z.ravel(order="F"),
    ]

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, 1]

    if scalars is None:
        scalars = Z
    grid[name] = np.asarray(scalars, dtype=float).ravel(order="F")
    return grid


def update_surface(
    grid: pv.StructuredGrid,
    Z: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    name: str = "scalar",
) -> pv.StructuredGrid:
    """Mutate ``grid`` in place to a new elevation ``Z`` and (optionally) scalars.

    Reassigns ``grid.points`` wholesale (mutating the z-column alone does not
    reliably trigger a render refresh in pyvista 0.38.6) and re-sets the scalar
    array under the *same* ``name`` so a fixed ``clim`` keeps the colorbar stable.
    """
    Z = np.asarray(Z, dtype=float)
    pts = np.array(grid.points, dtype=float)  # copy: (nx*ny, 3), F-order layout
    pts[:, 2] = Z.ravel(order="F")
    grid.points = pts

    if scalars is None:
        scalars = Z
    grid[name] = np.asarray(scalars, dtype=float).ravel(order="F")
    return grid


# ---------------------------------------------------------------------------
# Plotter construction / mesh styling
# ---------------------------------------------------------------------------
def new_plotter(
    window_size: Sequence[int] = (1100, 800),
    off_screen: bool = True,
) -> pv.Plotter:
    """Return a configured off-screen :class:`pyvista.Plotter`.

    White background, FXAA anti-aliasing, and 8-pass depth peeling so the
    translucent upper surface composites correctly over the lower one.
    """
    pl = pv.Plotter(window_size=list(window_size), off_screen=off_screen)
    pl.set_background("white")
    try:
        pl.enable_anti_aliasing("fxaa")
    except Exception:
        # Older signature: enable_anti_aliasing() with no mode argument.
        try:
            pl.enable_anti_aliasing()
        except Exception:
            pass
    try:
        pl.enable_depth_peeling(8)
    except Exception:
        pass
    return pl


def add_surface(
    pl: pv.Plotter,
    grid: pv.StructuredGrid,
    name: str,
    cmap="viridis",
    clim=None,
    opacity: float = 1.0,
    show_edges: bool = False,
    sbar_title: str = "",
):
    """Add ``grid`` to the plotter, colored by scalar ``name``.

    Returns the VTK actor so the caller can later toggle visibility, etc.
    The colormap is always resolved to a matplotlib Colormap object.
    """
    scalar_bar_args = dict(
        title=sbar_title,
        vertical=True,
        n_labels=5,
        title_font_size=18,
        label_font_size=14,
        color="black",
        fmt="%.3g",
        position_x=0.86,
        position_y=0.20,
        width=0.08,
        height=0.60,
    )
    actor = pl.add_mesh(
        grid,
        scalars=name,
        cmap=_resolve_cmap(cmap),
        clim=clim,
        opacity=opacity,
        show_edges=show_edges,
        edge_color="gray",
        smooth_shading=True,
        specular=0.3,
        specular_power=15,
        ambient=0.25,
        diffuse=0.75,
        scalar_bar_args=scalar_bar_args,
        show_scalar_bar=bool(sbar_title),
    )
    return actor


# ---------------------------------------------------------------------------
# Camera / lighting helper
# ---------------------------------------------------------------------------
def _frame_camera(pl: pv.Plotter, azimuth: float = -55.0, elevation: float = 28.0):
    """Set a low-elevation oblique view that flatters a rough joint surface.

    We set the camera explicitly (rather than ``view_isometric`` + nudges) so the
    vertical relief and the exploded joint aperture read as genuinely 3-D instead
    of collapsing to a near-top-down map.
    """
    bounds = pl.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])

    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    r = 2.4 * span
    eye = (
        cx + r * np.cos(el) * np.cos(az),
        cy + r * np.cos(el) * np.sin(az),
        cz + r * np.sin(el),
    )
    pl.camera_position = [eye, (cx, cy, cz), (0, 0, 1)]
    try:
        pl.camera.zoom(1.35)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Static still
# ---------------------------------------------------------------------------
def still_two_surfaces(
    x: np.ndarray,
    y: np.ndarray,
    lower_Z: np.ndarray,
    upper_Z: np.ndarray,
    scalar: Optional[np.ndarray] = None,
    out_png: str = "figures/two_surfaces_still.png",
    explode_dz: float = 0.0,
    cmap="coolwarm",
    clim=None,
    sbar_title: str = "traction (MPa)",
    window_size: Sequence[int] = (1100, 800),
    azimuth: float = -55.0,
    elevation: float = 28.0,
) -> str:
    """Render a single professional still of the two joint surfaces to ``out_png``.

    The lower surface is opaque; the upper surface is drawn at the same scalar
    coloring, lifted by ``explode_dz`` and made slightly translucent to expose
    the joint aperture. Returns the output path.
    """
    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lower_Z = np.asarray(lower_Z, dtype=float)
    upper_Z = np.asarray(upper_Z, dtype=float)
    if scalar is None:
        scalar = upper_Z

    if clim is None:
        s = np.asarray(scalar, dtype=float)
        clim = [float(np.nanmin(s)), float(np.nanmax(s))]
        if clim[0] == clim[1]:
            clim = [clim[0] - 0.5, clim[1] + 0.5]

    lower = make_height_surface(x, y, lower_Z, scalars=scalar, name="field")
    upper = make_height_surface(
        x, y, upper_Z + explode_dz, scalars=scalar, name="field"
    )

    pl = new_plotter(window_size=window_size, off_screen=True)
    add_surface(
        pl, lower, "field", cmap=cmap, clim=clim, opacity=1.0,
        sbar_title=sbar_title,
    )
    add_surface(
        pl, upper, "field", cmap=cmap, clim=clim,
        opacity=0.65 if explode_dz != 0.0 else 1.0, sbar_title="",
    )

    pl.add_light(pv.Light(position=(2, -2, 4), light_type="scene light", intensity=0.6))
    _frame_camera(pl, azimuth=azimuth, elevation=elevation)
    pl.screenshot(out_png)
    pl.close()
    return out_png


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def animate_two_surfaces(
    x: np.ndarray,
    y: np.ndarray,
    lower_Z_seq: Sequence[np.ndarray],
    upper_Z_seq: Sequence[np.ndarray],
    scalar_seq: Sequence[np.ndarray],
    out_gif: str,
    fps: int = 12,
    explode_dz: float = 0.0,
    sbar_title: str = "traction (MPa)",
    cmap="coolwarm",
    clim=None,
    window_size: Sequence[int] = (1100, 800),
    azimuth: float = -55.0,
    elevation: float = 28.0,
    try_mp4: bool = False,
) -> str:
    """Animate a sequence of two-surface frames to an animated GIF.

    Parameters
    ----------
    lower_Z_seq, upper_Z_seq : sequences of ``(nx, ny)`` height fields (one per
        frame). The upper surface is translated by ``+explode_dz`` to reveal the
        joint aperture.
    scalar_seq : sequence of ``(nx, ny)`` scalar fields used to color both
        surfaces at frame ``k`` (e.g. local traction or gap).
    out_gif : output path (``.gif``). A fixed ``clim`` is computed across the
        whole sequence (unless supplied) so the colorbar does not flicker.
    try_mp4 : if True, attempt an mp4 via ``open_movie`` first and fall back to
        GIF when imageio-ffmpeg is unavailable (it is, on this environment).

    Returns
    -------
    The path actually written (always the GIF unless an mp4 succeeded).
    """
    n_frames = len(lower_Z_seq)
    if not (len(upper_Z_seq) == n_frames == len(scalar_seq)):
        raise ValueError("lower_Z_seq, upper_Z_seq, scalar_seq must be equal length")
    if n_frames == 0:
        raise ValueError("empty frame sequence")

    out_dir = os.path.dirname(os.path.abspath(out_gif))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Fixed color limits across the whole sequence.
    if clim is None:
        all_s = np.concatenate([np.asarray(s, float).ravel() for s in scalar_seq])
        clim = [float(np.nanmin(all_s)), float(np.nanmax(all_s))]
        if clim[0] == clim[1]:
            clim = [clim[0] - 0.5, clim[1] + 0.5]

    lower0 = np.asarray(lower_Z_seq[0], dtype=float)
    upper0 = np.asarray(upper_Z_seq[0], dtype=float)
    s0 = np.asarray(scalar_seq[0], dtype=float)

    lower = make_height_surface(x, y, lower0, scalars=s0, name="field")
    upper = make_height_surface(x, y, upper0 + explode_dz, scalars=s0, name="field")

    pl = new_plotter(window_size=window_size, off_screen=True)
    add_surface(
        pl, lower, "field", cmap=cmap, clim=clim, opacity=1.0,
        sbar_title=sbar_title,
    )
    add_surface(
        pl, upper, "field", cmap=cmap, clim=clim,
        opacity=0.65 if explode_dz != 0.0 else 1.0, sbar_title="",
    )
    pl.add_light(pv.Light(position=(2, -2, 4), light_type="scene light", intensity=0.6))
    _frame_camera(pl, azimuth=azimuth, elevation=elevation)

    # Choose the writer. mp4 needs imageio-ffmpeg, which is absent here, so it is
    # opt-in and guarded; we always end up with a GIF unless mp4 truly works.
    out_path = out_gif
    opened_movie = False
    if try_mp4 and out_gif.lower().endswith((".mp4", ".mov", ".avi")):
        try:
            pl.open_movie(out_gif, framerate=fps)
            opened_movie = True
            out_path = out_gif
        except Exception:
            opened_movie = False
    if not opened_movie:
        gif_path = out_gif
        if not gif_path.lower().endswith(".gif"):
            gif_path = os.path.splitext(gif_path)[0] + ".gif"
        pl.open_gif(gif_path, fps=fps)
        out_path = gif_path

    # Render the opening frame, then step through the rest.
    pl.write_frame()
    for k in range(1, n_frames):
        lo = np.asarray(lower_Z_seq[k], dtype=float)
        up = np.asarray(upper_Z_seq[k], dtype=float)
        sc = np.asarray(scalar_seq[k], dtype=float)
        update_surface(lower, lo, scalars=sc, name="field")
        update_surface(upper, up + explode_dz, scalars=sc, name="field")
        pl.write_frame()

    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _synth_rough_surface(x, y, seed=0, amp=1.0):
    """A few sine modes summed into a rough height field on the ``ij`` grid."""
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Lx = x.max() - x.min() + 1e-9
    Ly = y.max() - y.min() + 1e-9
    Z = np.zeros_like(X)
    for _ in range(5):
        kx = rng.integers(1, 5)
        ky = rng.integers(1, 5)
        phx = rng.uniform(0, 2 * np.pi)
        phy = rng.uniform(0, 2 * np.pi)
        a = amp * rng.uniform(0.3, 1.0) / (kx + ky)
        Z += a * np.sin(2 * np.pi * kx * X / Lx + phx) * np.cos(
            2 * np.pi * ky * Y / Ly + phy
        )
    return Z


def _self_test():
    figdir = "figures"
    os.makedirs(figdir, exist_ok=True)
    gif_path = os.path.join(figdir, "_surface_anim_selftest.gif")
    png_path = os.path.join(figdir, "_surface_anim_selftest.png")

    n = 60
    x = np.linspace(0.0, 10.0, n)
    y = np.linspace(0.0, 10.0, n)

    # Two mating rough faces: the lower face plus a thin "mold" upper face.
    base_lower = _synth_rough_surface(x, y, seed=1, amp=1.0)
    base_upper = base_lower + 0.8 + 0.15 * _synth_rough_surface(x, y, seed=2, amp=1.0)

    n_frames = 16
    shear_total = 3.0  # grid units the upper face slides in +x over the run
    dx = x[1] - x[0]

    lower_seq, upper_seq, scalar_seq = [], [], []
    for k in range(n_frames):
        shift = shear_total * k / (n_frames - 1)
        shift_cells = int(round(shift / dx))
        # Slide the upper face in +x by rolling its height field along axis 0.
        up_k = np.roll(base_upper, shift_cells, axis=0)
        lo_k = base_lower
        # Synthetic scalar = vertical gap between the (exploded-free) faces.
        gap = up_k - lo_k
        lower_seq.append(lo_k)
        upper_seq.append(up_k)
        scalar_seq.append(gap)

    clim = [
        float(min(s.min() for s in scalar_seq)),
        float(max(s.max() for s in scalar_seq)),
    ]

    out_gif = animate_two_surfaces(
        x, y, lower_seq, upper_seq, scalar_seq,
        out_gif=gif_path, fps=12, explode_dz=1.5,
        sbar_title="gap (mm)", cmap="coolwarm", clim=clim,
    )

    out_png = still_two_surfaces(
        x, y, lower_seq[0], upper_seq[-1], scalar=scalar_seq[-1],
        out_png=png_path, explode_dz=1.5, cmap="coolwarm", clim=clim,
        sbar_title="gap (mm)",
    )

    # Validate outputs exist and are non-trivial in size.
    assert os.path.exists(out_gif), f"GIF not written: {out_gif}"
    assert os.path.exists(out_png), f"PNG not written: {out_png}"
    gif_size = os.path.getsize(out_gif)
    png_size = os.path.getsize(out_png)
    assert gif_size > 20_000, f"GIF too small ({gif_size} bytes) — render likely blank"
    assert png_size > 10_000, f"PNG too small ({png_size} bytes) — render likely blank"

    print(f"  GIF : {out_gif}  ({gif_size:,} bytes, {n_frames} frames)")
    print(f"  PNG : {out_png}  ({png_size:,} bytes)")
    print("surface_anim_3d self-test PASSED")


if __name__ == "__main__":
    _self_test()
