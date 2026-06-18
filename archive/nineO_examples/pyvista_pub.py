#!/usr/bin/env python3
"""Publication-quality PyVista postprocessing for challenge problems.

Builds proper tetrahedral UnstructuredGrid meshes from chart FEM solvers,
extracts outer surfaces, warps by displacement, and renders von Mises
stress contours with professional styling suitable for journal figures.

Usage:
    python nineO_examples/pyvista_pub.py           # replot all 9 challenges
    python nineO_examples/pyvista_pub.py 1 4 8      # replot specific ones
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT_FIG, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────

def compute_von_mises(sigma):
    """Von Mises from (N, 3, 3) stress tensor."""
    tr = sigma[:, 0, 0] + sigma[:, 1, 1] + sigma[:, 2, 2]
    s = sigma.copy()
    for i in range(3):
        s[:, i, i] -= tr / 3.0
    return np.sqrt(1.5 * np.sum(s * s, axis=(1, 2)))


def build_tet_mesh(solver, u, stress_fn, warp_factor=1.0):
    """Build a VTK UnstructuredGrid from a chart solver + displacement.

    Returns
    -------
    pv.UnstructuredGrid with point data: von_mises, displacement, chart_id
    """
    import pyvista as pv
    import vtk

    nodes_phys = solver.nodes_phys.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    elements = solver.elements.cpu().numpy()

    # Deformed positions
    deformed = nodes_phys + warp_factor * u_np

    # Stress (element-wise) -> average to nodes
    F_def = solver.compute_F(u)
    sigma_el = stress_fn(F_def).detach().cpu().numpy()
    vm_el = compute_von_mises(sigma_el)

    # Average element values to nodes
    vm_nodes = np.zeros(len(nodes_phys))
    count = np.zeros(len(nodes_phys))
    for ei in range(len(elements)):
        for ni in elements[ei]:
            vm_nodes[ni] += vm_el[ei]
            count[ni] += 1
    count[count == 0] = 1
    vm_nodes /= count

    # Build VTK UnstructuredGrid
    n_tets = len(elements)
    # VTK cell format: [n_pts, p0, p1, p2, p3, n_pts, ...]
    cells = np.hstack([np.full((n_tets, 1), 4, dtype=np.int64), elements]).ravel()
    cell_types = np.full(n_tets, vtk.VTK_TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, deformed)
    grid.point_data["von_mises"] = vm_nodes
    grid.point_data["displacement_mag"] = np.linalg.norm(u_np, axis=1)

    # Also store cell data
    grid.cell_data["von_mises_cell"] = vm_el

    return grid


def combine_chart_meshes(chart_solvers, u_charts, stress_fn, n_charts,
                         warp_factor=1.0):
    """Build and merge tet meshes from all charts into one UnstructuredGrid."""
    import pyvista as pv

    meshes = []
    for ci in range(n_charts):
        if u_charts[ci] is None:
            continue
        grid = build_tet_mesh(chart_solvers[ci], u_charts[ci], stress_fn,
                              warp_factor=warp_factor)
        # Tag chart ID
        grid.point_data["chart_id"] = np.full(grid.n_points, ci, dtype=float)
        meshes.append(grid)

    if not meshes:
        return None

    combined = meshes[0]
    for m in meshes[1:]:
        combined = combined.merge(m)

    return combined


def make_lut(n_colors=256, preset="coolwarm"):
    """Build a VTK lookup table (bypasses matplotlib requirement).

    Presets: coolwarm, jet, viridis, bwr, plasma
    """
    import vtk

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)

    if preset == "coolwarm":
        # Blue → White → Red (diverging, publication standard)
        for i in range(n_colors):
            t = i / (n_colors - 1)
            if t < 0.5:
                s = t * 2  # 0 → 1
                r = 0.23 + 0.77 * s
                g = 0.30 + 0.70 * s
                b = 0.75 + 0.25 * s
            else:
                s = (t - 0.5) * 2  # 0 → 1
                r = 1.0
                g = 1.0 - 0.75 * s
                b = 1.0 - 0.85 * s
            lut.SetTableValue(i, r, g, b, 1.0)
    elif preset == "jet":
        for i in range(n_colors):
            t = i / (n_colors - 1)
            if t < 0.125:
                r, g, b = 0, 0, 0.5 + 4*t
            elif t < 0.375:
                r, g, b = 0, (t-0.125)*4, 1
            elif t < 0.625:
                r, g, b = (t-0.375)*4, 1, 1-(t-0.375)*4
            elif t < 0.875:
                r, g, b = 1, 1-(t-0.625)*4, 0
            else:
                r, g, b = 1-(t-0.875)*4, 0, 0
            lut.SetTableValue(i, min(max(r,0),1), min(max(g,0),1),
                              min(max(b,0),1), 1.0)
    elif preset == "viridis":
        # Simplified viridis
        anchors = [(0.267,0.004,0.329), (0.283,0.141,0.458), (0.231,0.322,0.546),
                    (0.171,0.484,0.558), (0.127,0.567,0.551), (0.134,0.658,0.517),
                    (0.267,0.749,0.441), (0.478,0.821,0.318), (0.741,0.873,0.150),
                    (0.993,0.906,0.144)]
        for i in range(n_colors):
            t = i / (n_colors - 1) * (len(anchors) - 1)
            idx = int(t)
            frac = t - idx
            if idx >= len(anchors) - 1:
                idx = len(anchors) - 2; frac = 1.0
            r = anchors[idx][0] + frac * (anchors[idx+1][0] - anchors[idx][0])
            g = anchors[idx][1] + frac * (anchors[idx+1][1] - anchors[idx][1])
            b = anchors[idx][2] + frac * (anchors[idx+1][2] - anchors[idx][2])
            lut.SetTableValue(i, r, g, b, 1.0)
    elif preset == "plasma":
        anchors = [(0.050,0.030,0.528), (0.295,0.011,0.630), (0.500,0.007,0.651),
                    (0.658,0.134,0.588), (0.798,0.280,0.470), (0.900,0.424,0.360),
                    (0.965,0.582,0.253), (0.988,0.753,0.145), (0.940,0.975,0.131)]
        for i in range(n_colors):
            t = i / (n_colors - 1) * (len(anchors) - 1)
            idx = int(t)
            frac = t - idx
            if idx >= len(anchors) - 1:
                idx = len(anchors) - 2; frac = 1.0
            r = anchors[idx][0] + frac * (anchors[idx+1][0] - anchors[idx][0])
            g = anchors[idx][1] + frac * (anchors[idx+1][1] - anchors[idx][1])
            b = anchors[idx][2] + frac * (anchors[idx+1][2] - anchors[idx][2])
            lut.SetTableValue(i, r, g, b, 1.0)
    else:
        # Default blue-white-red
        return make_lut(n_colors, "coolwarm")

    lut.Build()
    return lut


def render_pub_figure(
    mesh,
    scalars="von_mises",
    title="",
    filename="figure.png",
    clim=None,
    colormap="coolwarm",
    camera_position=None,
    window_size=(1600, 1000),
    show_edges=False,
    opacity=1.0,
    n_labels=6,
    scalar_bar_title="von Mises (MPa)",
    zoom=1.0,
    show_axes=True,
    font_size=14,
    dpi_scale=1,
    multi_view=False,
):
    """Render a publication-quality figure of a surface mesh with scalar field.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid or pv.PolyData
        The mesh to render (surface already extracted).
    scalars : str
        Name of the scalar field to plot.
    title : str
        Figure title.
    filename : str
        Output filename (in figures/).
    clim : tuple or None
        Colorbar limits.
    colormap : str
        Colormap preset: coolwarm, jet, viridis, plasma
    camera_position : str or list
        Camera position: 'xy', 'xz', 'yz', 'iso', or pyvista camera tuple.
    window_size : tuple
        Pixel dimensions.
    show_edges : bool
        Show mesh wireframe.
    """
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'

    if multi_view:
        return _render_multi_view(mesh, scalars, title, filename, clim,
                                  colormap, window_size, show_edges,
                                  scalar_bar_title, n_labels)

    # Extract surface from volume mesh
    if hasattr(mesh, 'extract_surface'):
        try:
            surface = mesh.extract_surface()
        except Exception:
            surface = mesh
    else:
        surface = mesh

    # Transfer scalars to surface if needed
    if scalars not in surface.point_data and scalars in mesh.point_data:
        surface.point_data[scalars] = mesh.point_data[scalars][:surface.n_points]

    # Clim
    if clim is None:
        vals = surface.point_data.get(scalars, np.zeros(surface.n_points))
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
        if vmax - vmin < 1e-10:
            vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-10:
            vmax = vmin + 1.0
        clim = (vmin, vmax)

    # Build custom lookup table
    lut = make_lut(256, colormap)
    lut.SetRange(clim[0], clim[1])

    # Plotter
    pl = pv.Plotter(off_screen=True, window_size=window_size)

    # Add mesh with VTK mapper for coloring
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(surface)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(scalars)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(clim[0], clim[1])
    mapper.InterpolateScalarsBeforeMappingOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if show_edges:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
        actor.GetProperty().SetLineWidth(0.3)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(30)
    actor.GetProperty().SetAmbient(0.15)
    actor.GetProperty().SetDiffuse(0.75)
    actor.GetProperty().SetOpacity(opacity)

    pl.add_actor(actor)

    # Scalar bar
    sbar = vtk.vtkScalarBarActor()
    sbar.SetLookupTable(lut)
    sbar.SetTitle(scalar_bar_title)
    sbar.SetNumberOfLabels(n_labels)
    sbar.SetWidth(0.06)
    sbar.SetHeight(0.55)
    sbar.SetPosition(0.90, 0.22)
    sbar.SetBarRatio(0.25)
    sbar.UnconstrainedFontSizeOn()

    # Style the scalar bar text
    title_prop = sbar.GetTitleTextProperty()
    title_prop.SetFontSize(font_size)
    title_prop.SetColor(0, 0, 0)
    title_prop.SetFontFamilyToArial()
    title_prop.BoldOn()
    title_prop.ItalicOff()
    title_prop.SetShadow(0)

    label_prop = sbar.GetLabelTextProperty()
    label_prop.SetFontSize(font_size - 2)
    label_prop.SetColor(0, 0, 0)
    label_prop.SetFontFamilyToArial()
    label_prop.BoldOff()
    label_prop.ItalicOff()
    label_prop.SetShadow(0)

    sbar.SetLabelFormat("%.1f")
    pl.add_actor(sbar)

    # Title
    if title:
        title_actor = vtk.vtkTextActor()
        title_actor.SetInput(title)
        title_actor.GetTextProperty().SetFontSize(font_size + 2)
        title_actor.GetTextProperty().SetColor(0, 0, 0)
        title_actor.GetTextProperty().SetFontFamilyToArial()
        title_actor.GetTextProperty().BoldOn()
        title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        title_actor.SetPosition(0.02, 0.94)
        pl.add_actor(title_actor)

    # Axes
    if show_axes:
        pl.add_axes(line_width=2, labels_off=False)

    # Camera
    if camera_position == 'iso' or camera_position is None:
        pl.view_isometric()
    elif camera_position == 'xy':
        pl.view_xy()
    elif camera_position == 'xz':
        pl.view_xz()
    elif camera_position == 'yz':
        pl.view_yz()
    elif isinstance(camera_position, (list, tuple)):
        pl.camera_position = camera_position
    else:
        pl.view_isometric()

    pl.camera.zoom(zoom)

    path = os.path.join(OUT_FIG, filename)
    pl.screenshot(path, scale=dpi_scale)
    pl.close()
    print(f"  Saved: {path}")
    return path


def _render_multi_view(mesh, scalars, title, filename, clim, colormap,
                       window_size, show_edges, scalar_bar_title, n_labels):
    """Render 2x2 multi-view: iso, front, side, top."""
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'

    if hasattr(mesh, 'extract_surface'):
        try:
            surface = mesh.extract_surface()
        except Exception:
            surface = mesh
    else:
        surface = mesh

    if clim is None:
        vals = surface.point_data.get(scalars, np.zeros(surface.n_points))
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
        if vmax - vmin < 1e-10:
            vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-10:
            vmax = vmin + 1.0
        clim = (vmin, vmax)

    lut = make_lut(256, colormap)
    lut.SetRange(clim[0], clim[1])

    views = [
        ("Isometric", "iso"),
        ("Front (XY)", "xy"),
        ("Side (XZ)", "xz"),
        ("Top (YZ)", "yz"),
    ]

    pl = pv.Plotter(off_screen=True, shape=(2, 2),
                     window_size=(window_size[0]*2, window_size[1]*2))

    for idx, (label, view) in enumerate(views):
        row, col = divmod(idx, 2)
        pl.subplot(row, col)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(surface)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(scalars)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(clim[0], clim[1])
        mapper.InterpolateScalarsBeforeMappingOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetAmbient(0.15)
        actor.GetProperty().SetDiffuse(0.75)
        if show_edges:
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetLineWidth(0.3)
        pl.add_actor(actor)

        # View label
        txt = vtk.vtkTextActor()
        txt.SetInput(label)
        txt.GetTextProperty().SetFontSize(14)
        txt.GetTextProperty().SetColor(0.3, 0.3, 0.3)
        txt.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        txt.SetPosition(0.02, 0.92)
        pl.add_actor(txt)

        if view == 'iso':
            pl.view_isometric()
        elif view == 'xy':
            pl.view_xy()
        elif view == 'xz':
            pl.view_xz()
        elif view == 'yz':
            pl.view_yz()

        pl.camera.zoom(1.1)

        # Add scalar bar to first subplot only
        if idx == 0:
            sbar = vtk.vtkScalarBarActor()
            sbar.SetLookupTable(lut)
            sbar.SetTitle(scalar_bar_title)
            sbar.SetNumberOfLabels(n_labels)
            sbar.SetWidth(0.06)
            sbar.SetHeight(0.5)
            sbar.SetPosition(0.90, 0.25)
            sbar.SetLabelFormat("%.1f")
            tp = sbar.GetTitleTextProperty()
            tp.SetFontSize(12); tp.SetColor(0,0,0); tp.BoldOn(); tp.SetShadow(0)
            lp = sbar.GetLabelTextProperty()
            lp.SetFontSize(10); lp.SetColor(0,0,0); lp.SetShadow(0)
            pl.add_actor(sbar)

    path = os.path.join(OUT_FIG, filename)
    pl.screenshot(path, scale=2)
    pl.close()
    print(f"  Saved multi-view: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────
# Per-challenge runners
# ─────────────────────────────────────────────────────────────────────

def _setup():
    """Common imports and solver setup."""
    import torch
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.robin_schwarz import RobinSchwarzSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder, TubeSectorDecoder
    from solvers.fracture_criteria import drucker_prager_F
    return {
        'torch': torch, 'np': np,
        'ChartVectorFEMSolver': ChartVectorFEMSolver,
        'RobinSchwarzSolver': RobinSchwarzSolver,
        'make_linear_elastic_small_strain': make_linear_elastic_small_strain,
        'BoxDecoder': BoxDecoder, 'CrackTipDecoder': CrackTipDecoder,
        'TubeSectorDecoder': TubeSectorDecoder,
        'drucker_prager_F': drucker_prager_F,
    }


def run_challenge_1(ctx):
    """Uniaxial tension: cylindrical rod."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']

    E = 70e3; nu = 0.22; L = 15.0; R = 2.0

    class RodSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R; d_z = np.maximum(-x_np[:, 2], x_np[:, 2] - L)
            out = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            v = out + ins
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = RodSDF()
    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    solvers = []; decoders = []; seeds = []
    overlap = 0.3; z_span = L / 2 * (1 + overlap)
    for ci in range(2):
        z_c = L * (ci + 0.5) / 2
        dec = BoxDecoder(center=(0, 0, z_c), half_extents=(R*1.2, R*1.2, z_span/2)).double()
        s = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec,
                                  decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                  device="cpu", dtype=torch.float64)
        solvers.append(s); decoders.append(dec); seeds.append([0, 0, z_c])

    crack_dec = CrackTipDecoder.from_crack_tip([0,0,L/2], [1,0,0], [0,0,1], radius=R*0.8).double()
    s_crack = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=crack_dec,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)
    solvers.append(s_crack); decoders.append(crack_dec); seeds.append([0, 0, L/2])

    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    eps = 40.0 / E * 1.15  # just past nucleation
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.ones(n, dtype=bool)
        u[:, 0] = -nu * eps * np_phys[:, 0]
        u[:, 1] = -nu * eps * np_phys[:, 1]
        u[:, 2] = eps * np_phys[:, 2]
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-3)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=50.0)
    render_pub_figure(mesh, title="Challenge 1: Uniaxial Tension",
                      filename="challenge_1_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.3)
    render_pub_figure(mesh, title="Challenge 1: Uniaxial Tension",
                      filename="challenge_1_multiview.png",
                      colormap="coolwarm", multi_view=True)
    return mesh


def run_challenge_2(ctx):
    """Biaxial tension: circular plate."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']
    from benchmarks.fracture.biaxial_tension import sdf_circular_plate

    E = 70e3; nu = 0.22; R = 5.0; T = 0.5

    class PlateOracle:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            v = sdf_circular_plate(x_np, R=R, L=T)
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = PlateOracle()
    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec_bulk = BoxDecoder(center=(0, 0, 0), half_extents=(R*1.1, R*1.1, T*0.6)).double()
    s_bulk = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec_bulk,
                                   decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                   device="cpu", dtype=torch.float64)
    crack_dec = CrackTipDecoder.from_crack_tip([0,0,0], [1,0,0], [0,1,0], radius=R*0.5).double()
    s_crack = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=crack_dec,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)

    solvers = [s_bulk, s_crack]; decoders = [dec_bulk, crack_dec]
    seeds_t = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float64)

    sigma_bs = 27.03; eps = sigma_bs * (1 - nu) / E * 1.1
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.ones(n, dtype=bool)
        u[:, 0] = eps * np_phys[:, 0]; u[:, 1] = eps * np_phys[:, 1]
        u[:, 2] = -2*nu/(1-nu) * eps * np_phys[:, 2]
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1],[0]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-3)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=100.0)
    render_pub_figure(mesh, title="Challenge 2: Biaxial Tension",
                      filename="challenge_2_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.3,
                      camera_position='iso')
    return mesh


def run_challenge_3(ctx):
    """Torsion: thin-walled tube."""
    import math
    torch = ctx['torch']; TubeSectorDecoder = ctx['TubeSectorDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']

    E = 70e3; nu = 0.22; mu = E / (2*(1+nu))
    r_mid = 2.925; t_wall = 0.15; L = 5.0

    solvers = []; decoders = []; seeds = []
    for ti in range(4):
        theta_c = ti * math.pi / 2
        dec = TubeSectorDecoder(theta_center=theta_c, theta_span=math.pi/1.5,
                                 r_mid=r_mid, t_half=t_wall/2, z_center=L/2, L_half=L/2).double()
        s = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec,
                                  decoder_kwargs={}, device="cpu", dtype=torch.float64)
        solvers.append(s); decoders.append(dec)
        seeds.append([r_mid*math.cos(theta_c), r_mid*math.sin(theta_c), L/2])

    tip = [r_mid, 0, L/2]
    crack_dec = CrackTipDecoder.from_crack_tip(tip, [-1/math.sqrt(2),0,1/math.sqrt(2)],
                                                [1,0,0], radius=t_wall*2).double()
    s_crack = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=crack_dec,
                                    decoder_kwargs={}, device="cpu", dtype=torch.float64)
    solvers.append(s_crack); decoders.append(crack_dec); seeds.append(tip)

    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[(i+1)%4, (i-1)%4, 4] for i in range(4)]
    neighbors.append([0,1,2,3])

    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)
    alpha = 44.4 * L / (mu * r_mid) * 1.1

    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        z = np_phys[:, 2]; tol = L * 0.05
        rho = np.sqrt(np_phys[:, 0]**2 + np_phys[:, 1]**2)
        theta = np.arctan2(np_phys[:, 1], np_phys[:, 0])
        z0 = z < tol; m[z0] = True
        zL = z > L - tol; m[zL] = True
        u_th = alpha * rho[zL]
        u[zL, 0] = -u_th * np.sin(theta[zL])
        u[zL, 1] = u_th * np.cos(theta[zL])
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=neighbors, robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=20.0)
    render_pub_figure(mesh, title="Challenge 3: Torsion",
                      filename="challenge_3_von_mises_pub.png",
                      colormap="plasma", show_edges=True, zoom=1.3)
    return mesh


def run_challenge_4(ctx):
    """Pure shear: strip with edge crack."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']
    from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle

    E = 70e3; nu = 0.22; W = 25.0; H = 5.0; B = 0.5; A = 10.0
    sdf = CrackedPlateSDFOracle(a=A, W=W, H=H, T=B, delta=0.02)
    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec_l = BoxDecoder(center=(-W/2, 0, 0), half_extents=(W/2+0.1, H/2+0.1, B/2+0.1)).double()
    s_l = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec_l,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    dec_r = BoxDecoder(center=(W/2, 0, 0), half_extents=(W/2+0.1, H/2+0.1, B/2+0.1)).double()
    s_r = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec_r,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    tip_x = -W + A
    crack_dec = CrackTipDecoder.from_crack_tip([tip_x,0,0], [1,0,0], [0,1,0], radius=2.0).double()
    s_c = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_l, s_r, s_c]; decoders = [dec_l, dec_r, crack_dec]
    seeds_t = torch.tensor([[-W/2,0,0],[W/2,0,0],[tip_x,0,0]], dtype=torch.float64)

    import math
    Gc = 0.01
    h_crit = math.sqrt(Gc * H * 4 * (1-nu**2) / E)
    h = h_crit * 1.5

    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        y = np_phys[:, 1]; tol = H/2 * 0.05
        top = y > H/2 - tol; m[top] = True; u[top, 1] = h/2
        bot = y < -H/2 + tol; m[bot] = True; u[bot, 1] = -h/2
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=500.0)
    render_pub_figure(mesh, title="Challenge 4: Pure Shear",
                      filename="challenge_4_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.2,
                      camera_position='xy')
    return mesh


def run_challenge_5(ctx):
    """Single edge notch: strip with parameterized crack."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']
    from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle

    E = 70e3; nu = 0.22; W_half = 2.5; H = 5.0; B = 0.25; A = 0.5
    sdf = CrackedPlateSDFOracle(a=A, W=W_half, H=H, T=B, delta=0.01)
    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec = BoxDecoder(center=(0,0,0), half_extents=(W_half+0.1, H/2+0.1, B/2+0.1)).double()
    s_b = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    tip_x = -W_half + A
    crack_dec = CrackTipDecoder.from_crack_tip([tip_x,0,0], [1,0,0], [0,1,0],
                                                radius=min(A*0.5, 0.3)).double()
    s_c = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_b, s_c]; decoders = [dec, crack_dec]
    seeds_t = torch.tensor([[0,0,0],[tip_x,0,0]], dtype=torch.float64)

    eps = 40.0 / E  # at sigma_ts
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        y = np_phys[:, 1]; tol = H/2 * 0.05
        top = y > H/2 - tol; m[top] = True; u[top, 1] = eps * H/2
        bot = y < -H/2 + tol; m[bot] = True
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1],[0]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=200.0)
    render_pub_figure(mesh, title="Challenge 5: Single Edge Notch",
                      filename="challenge_5_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.2,
                      camera_position='xy')
    return mesh


def run_challenge_6(ctx):
    """Indentation: cylindrical block with flat punch."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']

    E = 70e3; nu = 0.22; R_block = 25.0; L_block = 25.0; R_punch = 1.0

    class BlockSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R_block; d_z = np.maximum(-x_np[:, 2], x_np[:, 2] - L_block)
            out = np.sqrt(np.maximum(d_r,0)**2 + np.maximum(d_z,0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            return torch.tensor(out+ins, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else out+ins

    sdf = BlockSDF(); stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec_t = BoxDecoder(center=(0,0,L_block*0.8), half_extents=(R_punch*5, R_punch*5, L_block*0.25)).double()
    s_t = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_t,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    dec_b = BoxDecoder(center=(0,0,L_block*0.3), half_extents=(R_block*0.5, R_block*0.5, L_block*0.35)).double()
    s_b = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_b,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    crack_dec = CrackTipDecoder.from_crack_tip([R_punch*1.2,0,L_block], [0,1,0], [0,0,-1],
                                                radius=R_punch*0.5).double()
    s_c = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_t, s_b, s_c]; decoders = [dec_t, dec_b, crack_dec]
    seeds_t = torch.tensor([[0,0,L_block*0.8],[0,0,L_block*0.3],[R_punch*1.2,0,L_block]], dtype=torch.float64)

    delta = 0.01
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        z = np_phys[:, 2]; r = np.sqrt(np_phys[:, 0]**2 + np_phys[:, 1]**2)
        bot = z < L_block * 0.05; m[bot] = True
        top_punch = (z > L_block - L_block*0.05) & (r < R_punch*1.1)
        m[top_punch] = True; u[top_punch, 2] = -delta
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=1000.0)
    render_pub_figure(mesh, title="Challenge 6: Indentation",
                      filename="challenge_6_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.2)
    return mesh


def run_challenge_7(ctx):
    """Poker-chip: circular disk under hydrostatic tension."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']

    mu = 0.52; lam = 85.77; E = mu*(3*lam+2*mu)/(lam+mu); nu = lam/(2*(lam+mu))
    R = 5.0; L_avg = 1.35

    class DiskSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R; d_z = np.abs(x_np[:, 2]) - L_avg/2
            out = np.sqrt(np.maximum(d_r,0)**2 + np.maximum(d_z,0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            return torch.tensor(out+ins, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else out+ins

    sdf = DiskSDF(); stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec = BoxDecoder(center=(0,0,0), half_extents=(R*1.1, R*1.1, L_avg*0.6)).double()
    s_b = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    crack_dec = CrackTipDecoder.from_crack_tip([0,0,0], [1,0,0], [0,0,1], radius=R*0.3).double()
    s_c = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_b, s_c]; decoders = [dec, crack_dec]
    seeds_t = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float64)

    delta = 0.03
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        z = np_phys[:, 2]; tol = L_avg/2 * 0.1
        top = z > L_avg/2 - tol; m[top] = True; u[top, 2] = delta/2
        bot = z < -L_avg/2 + tol; m[bot] = True; u[bot, 2] = -delta/2
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1],[0]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=5.0)
    render_pub_figure(mesh, title="Challenge 7: Poker-Chip",
                      filename="challenge_7_von_mises_pub.png",
                      colormap="viridis", show_edges=True, zoom=1.2)
    return mesh


def run_challenge_8(ctx):
    """DCB: double cantilever beam with pre-crack."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']
    from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
    import math

    E = 70e3; nu = 0.22; Gc = 0.01; L = 55.0; H = 20.0; B = 2.5; A = 25.0
    W_half = L / 2; h_arm = H / 2
    sdf = CrackedPlateSDFOracle(a=A, W=W_half, H=H/2, T=B, delta=0.05)
    stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec_l = BoxDecoder(center=(-L/4,0,0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
    s_l = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec_l,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    dec_r = BoxDecoder(center=(L/4,0,0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
    s_r = ChartVectorFEMSolver(n_cells=10, support_r=1.0, chart_decoder=dec_r,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    tip_x = -W_half + A
    crack_dec = CrackTipDecoder.from_crack_tip([tip_x,0,0], [1,0,0], [0,1,0], radius=2.0).double()
    s_c = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_l, s_r, s_c]; decoders = [dec_l, dec_r, crack_dec]
    seeds_t = torch.tensor([[-L/4,0,0],[L/4,0,0],[tip_x,0,0]], dtype=torch.float64)

    I_arm = B * h_arm**3 / 12
    F_crit = B * math.sqrt(E * Gc * h_arm**3 / (12 * A**2))
    C_A = 2 * A**3 / (3 * E * I_arm)
    delta = F_crit * C_A * 2.0
    pin_x = -W_half + 1.5

    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        x = np_phys[:, 0]; y = np_phys[:, 1]
        right = x > W_half - 1.0; m[right] = True
        top_pin = (np.abs(x - pin_x) < 2.0) & (np.abs(y - H/6) < 2.0) & (y > 0)
        m[top_pin] = True; u[top_pin, 1] = delta
        bot_pin = (np.abs(x - pin_x) < 2.0) & (np.abs(y + H/6) < 2.0) & (y < 0)
        m[bot_pin] = True; u[bot_pin, 1] = -delta
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=500.0)
    render_pub_figure(mesh, title="Challenge 8: Double Cantilever Beam",
                      filename="challenge_8_von_mises_pub.png",
                      colormap="coolwarm", show_edges=True, zoom=1.2,
                      camera_position='xy')
    return mesh


def run_challenge_9(ctx):
    """Trousers: sheet with pre-crack (Mode III)."""
    torch = ctx['torch']; BoxDecoder = ctx['BoxDecoder']
    CrackTipDecoder = ctx['CrackTipDecoder']
    ChartVectorFEMSolver = ctx['ChartVectorFEMSolver']
    RobinSchwarzSolver = ctx['RobinSchwarzSolver']

    mu = 0.52; lam = 85.77; E = mu*(3*lam+2*mu)/(lam+mu); nu = lam/(2*(lam+mu))
    L = 100.0; W = 40.0; B = 1.0; A = 50.0

    class SheetSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            dx = np.abs(x_np[:, 0]) - W/2; dy_lo = -x_np[:, 1]; dy_hi = x_np[:, 1] - L
            dz = np.abs(x_np[:, 2]) - B/2
            d_max = np.maximum(np.maximum(dx, np.maximum(dy_lo, dy_hi)), dz)
            out = np.sqrt(np.maximum(dx,0)**2 + np.maximum(np.maximum(dy_lo, dy_hi),0)**2 + np.maximum(dz,0)**2)
            ins = np.minimum(d_max, 0)
            # Crack slit
            crack_dx = np.abs(x_np[:, 0]) - 0.02
            crack_dy = np.maximum(-x_np[:, 1], x_np[:, 1] - A)
            crack_out = np.sqrt(np.maximum(crack_dx,0)**2 + np.maximum(crack_dy,0)**2)
            crack_in = np.minimum(np.maximum(crack_dx, crack_dy), 0)
            v = np.maximum(out + ins, -(crack_out + crack_in))
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = SheetSDF(); stress_fn, tangent_fn = ctx['make_linear_elastic_small_strain'](E, nu)

    dec_l = BoxDecoder(center=(-W/4, L/2, 0), half_extents=(W/4+0.1, L/2+0.1, B/2+0.1)).double()
    s_l = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_l,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    dec_r = BoxDecoder(center=(W/4, L/2, 0), half_extents=(W/4+0.1, L/2+0.1, B/2+0.1)).double()
    s_r = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_r,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)
    crack_dec = CrackTipDecoder.from_crack_tip([0,A,0], [0,1,0], [0,0,1], radius=B*2).double()
    s_c = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=crack_dec,
                                decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                device="cpu", dtype=torch.float64)

    solvers = [s_l, s_r, s_c]; decoders = [dec_l, dec_r, crack_dec]
    seeds_t = torch.tensor([[-W/4,L/2,0],[W/4,L/2,0],[0,A,0]], dtype=torch.float64)

    delta = 0.5
    def bc_fn(np_phys):
        n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
        y = np_phys[:, 1]; x = np_phys[:, 0]; tol = L * 0.02
        top = y > L - tol; m[top] = True
        u[top & (x < 0), 2] = -delta; u[top & (x >= 0), 2] = delta
        bot = y < tol; m[bot] = True
        return u, m

    robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
    u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=25, tol=1e-2)

    mesh = combine_chart_meshes(solvers, u_charts, stress_fn, len(solvers), warp_factor=2.0)
    render_pub_figure(mesh, title="Challenge 9: Trousers Test",
                      filename="challenge_9_von_mises_pub.png",
                      colormap="viridis", show_edges=True, zoom=1.1)
    return mesh


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

RUNNERS = {
    1: ("Uniaxial Tension", run_challenge_1),
    2: ("Biaxial Tension", run_challenge_2),
    3: ("Torsion", run_challenge_3),
    4: ("Pure Shear", run_challenge_4),
    5: ("Single Edge Notch", run_challenge_5),
    6: ("Indentation", run_challenge_6),
    7: ("Poker-Chip", run_challenge_7),
    8: ("Double Cantilever Beam", run_challenge_8),
    9: ("Trousers", run_challenge_9),
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problems = [int(x) for x in sys.argv[1:]]
    else:
        problems = list(range(1, 10))

    ctx = _setup()

    for p in problems:
        name, runner = RUNNERS[p]
        print(f"\n{'='*60}")
        print(f"  Challenge {p}: {name}")
        print(f"{'='*60}")
        try:
            runner(ctx)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
