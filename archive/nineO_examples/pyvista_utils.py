"""PyVista plotting utilities for challenge problem visualization.

Provides von Mises stress plotting on deformed configuration using PyVista
off-screen rendering. All figures are saved to the figures/ directory.
"""

import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT_FIG, exist_ok=True)


def compute_von_mises(sigma):
    """Compute von Mises stress from stress tensor (N, 3, 3).

    sigma_vm = sqrt(3/2 * s_ij * s_ij) where s = sigma - 1/3*tr(sigma)*I
    """
    if sigma.ndim == 2:
        raise ValueError("Expected (N, 3, 3) stress tensor")

    tr_sigma = sigma[:, 0, 0] + sigma[:, 1, 1] + sigma[:, 2, 2]
    s = sigma.copy()
    for i in range(3):
        s[:, i, i] -= tr_sigma / 3.0

    s_sq = np.sum(s * s, axis=(1, 2))
    return np.sqrt(1.5 * s_sq)


def _make_vtk_lut(n_colors=256):
    """Build a blue-green-red lookup table using VTK directly."""
    import vtk
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    lut.Build()
    for i in range(n_colors):
        t = i / (n_colors - 1)
        # Blue -> Cyan -> Green -> Yellow -> Red
        if t < 0.25:
            r, g, b = 0, 4*t, 1
        elif t < 0.5:
            r, g, b = 0, 1, 1 - 4*(t-0.25)
        elif t < 0.75:
            r, g, b = 4*(t-0.5), 1, 0
        else:
            r, g, b = 1, 1 - 4*(t-0.75), 0
        lut.SetTableValue(i, r, g, b, 1.0)
    return lut


def plot_von_mises_deformed(
    nodes,
    u_disp,
    sigma,
    chart_ids=None,
    title="",
    filename="von_mises.png",
    warp_factor=1.0,
    clim=None,
    window_size=(1200, 800),
    show_edges=False,
    camera_position=None,
):
    """Plot von Mises stress on deformed configuration via PyVista.

    Parameters
    ----------
    nodes : ndarray (N, 3) -- undeformed node positions
    u_disp : ndarray (N, 3) -- displacement field
    sigma : ndarray (N, 3, 3) or ndarray (N,) -- stress tensor or pre-computed VM
    chart_ids : ndarray (N,) optional -- chart ID for coloring
    title : str
    filename : str -- saved to figures/
    warp_factor : float -- magnification factor for displacements
    clim : tuple (min, max) optional -- colorbar limits
    """
    try:
        import pyvista as pv
        import vtk
    except ImportError:
        print(f"  [WARN] PyVista not available, skipping plot: {filename}")
        return None

    # Compute von Mises if full tensor provided
    if sigma.ndim == 3 and sigma.shape[1:] == (3, 3):
        vm = compute_von_mises(sigma)
    elif sigma.ndim == 1:
        vm = sigma
    else:
        print(f"  [WARN] Unexpected sigma shape {sigma.shape}, skipping plot")
        return None

    # Deformed positions
    deformed = nodes + warp_factor * u_disp

    # Create point cloud
    cloud = pv.PolyData(deformed)
    cloud["von_mises"] = vm
    cloud["displacement_mag"] = np.linalg.norm(u_disp, axis=1)
    if chart_ids is not None:
        cloud["chart_id"] = chart_ids.astype(float)

    # Build VTK lookup table (avoids matplotlib dependency)
    lut = _make_vtk_lut(256)
    if clim is not None:
        lut.SetRange(clim[0], clim[1])
    else:
        lut.SetRange(vm.min(), vm.max())

    # Off-screen rendering
    pv.global_theme.background = 'white'
    pl = pv.Plotter(off_screen=True, window_size=window_size)

    # Point cloud rendering with VTK lookup table
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cloud)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("von_mises")
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(vm.min() if clim is None else clim[0],
                          vm.max() if clim is None else clim[1])

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    actor.GetProperty().SetRepresentationToPoints()

    pl.add_actor(actor)

    # Add scalar bar manually
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetTitle("von Mises (MPa)")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetWidth(0.08)
    scalar_bar.SetHeight(0.5)
    scalar_bar.SetPosition(0.88, 0.25)
    pl.add_actor(scalar_bar)

    pl.add_title(title, font_size=10)

    if camera_position is not None:
        pl.camera_position = camera_position
    else:
        pl.camera.zoom(1.2)

    path = os.path.join(OUT_FIG, filename)
    pl.screenshot(path)
    pl.close()
    print(f"  Saved PyVista plot: {path}")
    return path


def collect_chart_data(chart_solvers, u_charts, stress_fn, n_charts=None):
    """Collect nodes, displacements, stress from all charts.

    Returns
    -------
    nodes, u_disp, sigma, chart_ids : ndarrays
    """
    if n_charts is None:
        n_charts = len(chart_solvers)

    all_nodes = []
    all_u = []
    all_sigma = []
    all_cid = []

    for ci in range(n_charts):
        if u_charts[ci] is None:
            continue
        solver = chart_solvers[ci]
        nodes_np = solver.nodes_phys.detach().cpu().numpy()
        u_np = u_charts[ci].detach().cpu().numpy()

        F_def = solver.compute_F(u_charts[ci])
        sigma = stress_fn(F_def).detach().cpu().numpy()

        # Stress is at elements; average to nodes
        if len(sigma) != len(nodes_np):
            sigma_nodes = np.zeros((len(nodes_np), 3, 3))
            count = np.zeros(len(nodes_np))
            elements = solver.elements.cpu().numpy()
            for ei in range(len(elements)):
                for ni in elements[ei]:
                    sigma_nodes[ni] += sigma[ei]
                    count[ni] += 1
            count[count == 0] = 1
            sigma_nodes /= count[:, None, None]
        else:
            sigma_nodes = sigma

        all_nodes.append(nodes_np)
        all_u.append(u_np)
        all_sigma.append(sigma_nodes)
        all_cid.append(np.full(len(nodes_np), ci))

    if not all_nodes:
        return None, None, None, None

    return (np.concatenate(all_nodes), np.concatenate(all_u),
            np.concatenate(all_sigma), np.concatenate(all_cid))
