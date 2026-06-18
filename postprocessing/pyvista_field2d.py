"""Thin PyVista helper to render a flat 2D analytical scalar field, paper-style.

Renders a Cartesian field (e.g. an analytical stress component on a disc or a
3x3 array) as a flat, top-down, filled color map with a jet colormap and a
styled scalar bar -- matching the look of the Liu & Sun (2020) ILS-MPM figures.

Uses raw VTK mappers + a self-contained jet lookup table (no matplotlib
dependency, mirroring nineO_examples/pyvista_pub.py::make_lut), so it works with
the repo's matplotlib 3.4 / pyvista 0.38 environment.

Design choices (see docs/contact_verification_manual.md):
- StructuredGrid at z=0 with a NaN geometry mask (transparent outside the body
  via the LUT NaN color alpha=0).
- Flat lighting (ambient=1, diffuse=0) so a planar slice is not Phong-shaded.
- Parallel projection + top-down (``view_xy``) for a 2D contour appearance.
- Symmetric 2/98-percentile color limits so the 1/r contact singularities do
  not blow out the range.
"""
from __future__ import annotations

import os

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


# ---------------------------------------------------------------------------
# Lookup table (jet / coolwarm), no matplotlib
# ---------------------------------------------------------------------------

def _jet_rgb(t):
    if t < 0.125:
        r, g, b = 0.0, 0.0, 0.5 + 4 * t
    elif t < 0.375:
        r, g, b = 0.0, (t - 0.125) * 4, 1.0
    elif t < 0.625:
        r, g, b = (t - 0.375) * 4, 1.0, 1.0 - (t - 0.375) * 4
    elif t < 0.875:
        r, g, b = 1.0, 1.0 - (t - 0.625) * 4, 0.0
    else:
        r, g, b = 1.0 - (t - 0.875) * 4, 0.0, 0.0
    return min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1)


def make_lut(clim, n_colors=256, preset="jet"):
    """vtk.vtkLookupTable over *clim* with transparent NaN color."""
    import vtk

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    for i in range(n_colors):
        r, g, b = _jet_rgb(i / (n_colors - 1))
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetRange(clim[0], clim[1])
    lut.SetNanColor(0.0, 0.0, 0.0, 0.0)   # masked (NaN) -> transparent
    lut.Build()
    return lut


# ---------------------------------------------------------------------------
# Geometry / clim helpers
# ---------------------------------------------------------------------------

def symmetric_clim(field, mask=None, pct=(2.0, 98.0), symmetric=True):
    """Robust color limits from finite (optionally masked) values."""
    f = np.asarray(field, dtype=float).copy()
    if mask is not None:
        f = np.where(mask, f, np.nan)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, pct)
    if symmetric:
        m = max(abs(lo), abs(hi)) or 1.0
        return (-m, m)
    if hi - lo < 1e-30:
        hi = lo + 1.0
    return (float(lo), float(hi))


def circle_outline(xc, yc, R, n=240, z=1e-2):
    """Closed-ring PolyData for overlaying a crisp body boundary."""
    import pyvista as pv

    th = np.linspace(0.0, 2.0 * np.pi, n)
    pts = np.column_stack([xc + R * np.cos(th), yc + R * np.sin(th), np.full(n, z)])
    lines = np.hstack([[n], np.arange(n)])
    return pv.PolyData(pts, lines=lines)


def _structured_grid(X, Y, field):
    """Flat (z=0) StructuredGrid; X, Y, field are (nx, ny) from 'ij' meshgrid."""
    import pyvista as pv

    grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
    grid["field"] = np.asarray(field, dtype=float).ravel(order="F")
    return grid


def _field_actor(grid, clim):
    """Raw VTK flat-lit actor for the field (bypasses matplotlib cmap path)."""
    import vtk

    lut = make_lut(clim)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("field")
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(clim[0], clim[1])
    mapper.InterpolateScalarsBeforeMappingOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetAmbient(1.0)        # flat, unlit color for a planar slice
    prop.SetDiffuse(0.0)
    prop.SetSpecular(0.0)
    return actor, lut


def _scalar_bar(lut, title):
    import vtk

    sbar = vtk.vtkScalarBarActor()
    sbar.SetLookupTable(lut)
    sbar.SetTitle(title)
    sbar.SetNumberOfLabels(6)
    sbar.SetWidth(0.07)
    sbar.SetHeight(0.6)
    sbar.SetPosition(0.88, 0.2)
    sbar.UnconstrainedFontSizeOn()
    sbar.SetLabelFormat("%.2g")
    for prop, fs, bold in ((sbar.GetTitleTextProperty(), 16, True),
                           (sbar.GetLabelTextProperty(), 13, False)):
        prop.SetColor(0, 0, 0)
        prop.SetFontFamilyToArial()
        prop.SetFontSize(fs)
        prop.SetShadow(0)
        (prop.BoldOn if bold else prop.BoldOff)()
    return sbar


def _title_actor(text):
    import vtk

    t = vtk.vtkTextActor()
    t.SetInput(text)
    t.GetTextProperty().SetFontSize(16)
    t.GetTextProperty().SetColor(0, 0, 0)
    t.GetTextProperty().SetFontFamilyToArial()
    t.GetTextProperty().BoldOn()
    t.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    t.SetPosition(0.03, 0.93)
    return t


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------

def _prep(X, Y, field, mask, clim, symmetric, pct):
    field = np.asarray(field, dtype=float).copy()
    if mask is not None:
        field = np.where(mask, field, np.nan)
    if clim is None:
        clim = symmetric_clim(field, pct=pct, symmetric=symmetric)
    return field, clim


def render_field_2d(X, Y, field, *, filename, mask=None, clim=None,
                    scalar_bar_title="", title="", outlines=None,
                    window_size=(1100, 1000), symmetric=True, pct=(2.0, 98.0),
                    dpi_scale=2, zoom=1.25):
    """Render a 2D scalar field to ``figures/<filename>``; return the path."""
    import pyvista as pv

    pv.global_theme.background = "white"
    field, clim = _prep(X, Y, field, mask, clim, symmetric, pct)
    grid = _structured_grid(X, Y, field)

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.enable_parallel_projection()
    actor, lut = _field_actor(grid, clim)
    pl.add_actor(actor)
    pl.add_actor(_scalar_bar(lut, scalar_bar_title))
    if outlines:
        for (xc, yc, Rc) in outlines:
            pl.add_mesh(circle_outline(xc, yc, Rc), color="black", line_width=1.5)
    if title:
        pl.add_actor(_title_actor(title))
    pl.view_xy()
    pl.camera.zoom(zoom)

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    pl.screenshot(path, scale=dpi_scale)
    pl.close()
    return path


def render_two_panel(panels, *, filename, window_size=(2000, 1000), dpi_scale=2,
                     zoom=1.2):
    """Render two field panels side-by-side (e.g. Fig.16 S1 and sigma_xy).

    ``panels`` is a list of two dicts with keys: X, Y, field, mask, clim,
    scalar_bar_title, title, outlines, symmetric.
    """
    import pyvista as pv

    pv.global_theme.background = "white"
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=window_size, border=False)
    for col, p in enumerate(panels):
        pl.subplot(0, col)
        pl.enable_parallel_projection()
        field, clim = _prep(p["X"], p["Y"], p["field"], p.get("mask"),
                            p.get("clim"), p.get("symmetric", True), (2.0, 98.0))
        grid = _structured_grid(p["X"], p["Y"], field)
        actor, lut = _field_actor(grid, clim)
        pl.add_actor(actor)
        pl.add_actor(_scalar_bar(lut, p.get("scalar_bar_title", "")))
        for (xc, yc, Rc) in p.get("outlines", []) or []:
            pl.add_mesh(circle_outline(xc, yc, Rc), color="black", line_width=1.0)
        if p.get("title"):
            pl.add_actor(_title_actor(p["title"]))
        pl.view_xy()
        pl.camera.zoom(zoom)

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    pl.screenshot(path, scale=dpi_scale)
    pl.close()
    return path
