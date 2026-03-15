from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

from pub_style import DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH, apply_publication_style, save_figure

REPO_ROOT = Path('/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry')
MANUSCRIPT_DIR = REPO_ROOT / 'manuscript'
FIG_DIR = MANUSCRIPT_DIR / 'figures_cmame_core' / 'example3_torus_inverse_original'

RUN_DIR = REPO_ROOT / 'runs' / 'torus_inverse_mps_dense_v4'
VTU_PATH = RUN_DIR / 'torus_inverse_neohookean_atlas_boundary_full_error.vtu'
HISTORY_JSON = RUN_DIR / 'torus_inverse_neohookean_atlas_history.json'
METRICS_JSON = RUN_DIR / 'torus_inverse_neohookean_atlas_metrics.json'

FIELDS_PNG = FIG_DIR / 'example3_torus_inverse_fields_pub.png'
FIELDS_PDF = FIG_DIR / 'example3_torus_inverse_fields_pub.pdf'
CONV_PNG = FIG_DIR / 'example3_torus_inverse_convergence_pub.png'
CONV_PDF = FIG_DIR / 'example3_torus_inverse_convergence_pub.pdf'

CAMERA_POSITION = [(3.2, -2.7, 1.45), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def load_data() -> tuple[pv.DataSet, dict, dict]:
    mesh = pv.read(VTU_PATH)
    hist = json.loads(HISTORY_JSON.read_text())
    metrics = json.loads(METRICS_JSON.read_text())
    return mesh, hist, metrics


def periodic_griddata(phi: np.ndarray, theta: np.ndarray, values: np.ndarray, phi_grid: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    pts = np.column_stack([phi, theta])
    periodic_pts = []
    periodic_vals = []
    for dphi in (-2 * np.pi, 0.0, 2 * np.pi):
        for dtheta in (-2 * np.pi, 0.0, 2 * np.pi):
            periodic_pts.append(pts + np.array([dphi, dtheta]))
            periodic_vals.append(values)
    periodic_pts = np.vstack(periodic_pts)
    periodic_vals = np.concatenate(periodic_vals)
    target = np.column_stack([phi_grid.ravel(), theta_grid.ravel()])
    out = griddata(periodic_pts, periodic_vals, target, method='linear')
    mask = np.isnan(out)
    if mask.any():
        out[mask] = griddata(periodic_pts, periodic_vals, target[mask], method='nearest')
    return out.reshape(phi_grid.shape)


def build_structured_surfaces(mesh: pv.DataSet) -> tuple[pv.PolyData, pv.PolyData, tuple[float, float], tuple[float, float]]:
    phi = np.mod(np.asarray(mesh['phi'], dtype=float), 2 * np.pi)
    theta = np.mod(np.asarray(mesh['theta'], dtype=float), 2 * np.pi)
    pts = np.asarray(mesh.points, dtype=float)
    x_def = np.asarray(mesh['x_deformed'], dtype=float)
    disp_mag = np.asarray(mesh['displacement_mag'], dtype=float)
    trac_err = np.asarray(mesh['traction_error_mag'], dtype=float)

    n_phi = 280
    n_theta = 160
    phi_lin = np.linspace(0.0, 2 * np.pi, n_phi + 1, endpoint=True)
    theta_lin = np.linspace(0.0, 2 * np.pi, n_theta + 1, endpoint=True)
    PHI, THETA = np.meshgrid(phi_lin, theta_lin)

    X = periodic_griddata(phi, theta, pts[:, 0], PHI, THETA)
    Y = periodic_griddata(phi, theta, pts[:, 1], PHI, THETA)
    Z = periodic_griddata(phi, theta, pts[:, 2], PHI, THETA)
    XD = periodic_griddata(phi, theta, x_def[:, 0], PHI, THETA)
    YD = periodic_griddata(phi, theta, x_def[:, 1], PHI, THETA)
    ZD = periodic_griddata(phi, theta, x_def[:, 2], PHI, THETA)
    DISP = periodic_griddata(phi, theta, disp_mag, PHI, THETA)
    TRAC = periodic_griddata(phi, theta, trac_err, PHI, THETA)

    # Enforce periodic closure explicitly so the visual seam does not appear in
    # the reconstructed surface.
    for arr in (X, Y, Z, XD, YD, ZD, DISP, TRAC):
        arr[:, -1] = arr[:, 0]
        arr[-1, :] = arr[0, :]

    sgrid_ref = pv.StructuredGrid(X, Y, Z)
    sgrid_def = pv.StructuredGrid(XD, YD, ZD)
    surf_ref = sgrid_ref.extract_surface(algorithm='dataset_surface').triangulate()
    surf_def = sgrid_def.extract_surface(algorithm='dataset_surface').triangulate()
    surf_ref['traction_error_mag'] = TRAC.ravel(order='F')
    surf_def['displacement_mag'] = DISP.ravel(order='F')
    return surf_ref, surf_def, (float(np.nanmin(DISP)), float(np.nanmax(DISP))), (float(np.nanmin(TRAC)), float(np.nanmax(TRAC)))


def render_surface(mesh: pv.PolyData, scalars: str, cmap: str, clim: tuple[float, float]) -> np.ndarray:
    pl = pv.Plotter(off_screen=True, window_size=(1200, 880), lighting='three lights')
    pl.set_background('white')
    pl.enable_anti_aliasing('fxaa')
    pl.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.55,
        diffuse=0.42,
        specular=0.0,
    )
    pl.camera_position = CAMERA_POSITION
    pl.camera.zoom(1.18)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def build_field_figure(surf_ref: pv.PolyData, surf_def: pv.PolyData, disp_clim: tuple[float, float], trac_clim: tuple[float, float]) -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    disp_img = render_surface(surf_def, 'displacement_mag', 'viridis', disp_clim)
    trac_img = render_surface(surf_ref, 'traction_error_mag', 'magma', trac_clim)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.95))
    panels = [
        (axes[0], disp_img, 'Deformed configuration (displacement magnitude)', 'viridis', disp_clim, r'$|u|$'),
        (axes[1], trac_img, 'Boundary traction-error magnitude', 'magma', trac_clim, r'$|t-t^{\mathrm{obs}}|$'),
    ]
    for ax, img, title, cmap, clim, cbar_label in panels:
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, pad=4)
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.06)
        cbar.ax.tick_params(labelsize=6.6)
        cbar.set_label(cbar_label, fontsize=7.0)

    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.15, wspace=0.10)
    save_figure(fig, FIELDS_PNG, FIELDS_PDF)
    plt.close(fig)


def build_convergence_figure(hist: dict, metrics: dict) -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    iters = np.arange(1, len(hist['mu_guess']) + 1, dtype=float)
    mu = np.asarray(hist['mu_guess'], dtype=float)
    K = np.asarray(hist['K_guess'], dtype=float)
    trac = np.asarray(hist['traction'], dtype=float)

    mu_true = float(metrics['mu_true'])
    K_true = float(metrics['K_true'])
    mu_err_pct = 100.0 * np.abs(mu - mu_true) / mu_true
    K_err_pct = 100.0 * np.abs(K - K_true) / K_true

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.65))
    ax.semilogy(iters, np.maximum(mu_err_pct, 1e-12), label=r'$\mu$ relative error (\%)', color='#1f77b4')
    ax.semilogy(iters, np.maximum(K_err_pct, 1e-12), label=r'$K$ relative error (\%)', color='#d62728')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter relative error (%)')
    ax.set_title('Inverse-parameter convergence', pad=4)
    ax.grid(True, which='both', alpha=0.35)

    ax2 = ax.twinx()
    ax2.semilogy(iters, np.maximum(trac, 1e-16), label='traction mismatch', color='#444444', ls='--')
    ax2.set_ylabel(r'$\mathcal{L}_{\mathrm{trac}}$')
    ax2.tick_params(axis='y', colors='#444444')

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=6.5, frameon=False)

    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.20)
    save_figure(fig, CONV_PNG, CONV_PDF)
    plt.close(fig)


def main() -> None:
    mesh, hist, metrics = load_data()
    surf_ref, surf_def, disp_clim, trac_clim = build_structured_surfaces(mesh)
    build_field_figure(surf_ref, surf_def, disp_clim, trac_clim)
    build_convergence_figure(hist, metrics)


if __name__ == '__main__':
    main()
