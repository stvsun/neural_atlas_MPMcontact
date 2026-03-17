from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

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

TWO_PI = 2.0 * np.pi


def load_data() -> tuple[pv.DataSet, dict, dict]:
    mesh = pv.read(VTU_PATH)
    hist = json.loads(HISTORY_JSON.read_text())
    metrics = json.loads(METRICS_JSON.read_text())
    return mesh, hist, metrics


def wrap_theta(theta: np.ndarray) -> np.ndarray:
    return ((theta + np.pi) % TWO_PI) - np.pi


def periodic_binned_field(
    phi: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    *,
    n_phi: int = 320,
    n_theta: int = 180,
    sigma: float = 1.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.mod(np.asarray(phi, dtype=float), TWO_PI)
    theta = wrap_theta(np.asarray(theta, dtype=float))
    values = np.asarray(values, dtype=float)

    phi_edges = np.linspace(0.0, TWO_PI, n_phi + 1)
    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)

    weighted, _, _ = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges], weights=values)
    counts, _, _ = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges])

    weighted_s = gaussian_filter(weighted, sigma=sigma, mode='wrap')
    counts_s = gaussian_filter(counts, sigma=sigma, mode='wrap')
    field = weighted_s / np.maximum(counts_s, 1.0e-12)

    return phi_edges, theta_edges, field


def angle_ticks() -> tuple[list[float], list[str], list[float], list[str]]:
    xticks = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, TWO_PI]
    xlabels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    yticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    ylabels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    return xticks, xlabels, yticks, ylabels


def style_map_axis(ax: plt.Axes, *, show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    xticks, xlabels, yticks, ylabels = angle_ticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    if show_xlabel:
        ax.set_xlabel(r'Major angle $\phi$')
    if show_ylabel:
        ax.set_ylabel(r'Minor angle $\theta$')
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(False)


def build_field_figure(mesh: pv.DataSet) -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    phi = np.asarray(mesh['phi'], dtype=float)
    theta = np.asarray(mesh['theta'], dtype=float)
    disp = np.asarray(mesh['displacement_mag'], dtype=float)
    trac = np.asarray(mesh['traction_error_mag'], dtype=float)

    disp_x, disp_y, disp_field = periodic_binned_field(phi, theta, disp, n_phi=320, n_theta=180, sigma=1.0)
    trac_x, trac_y, trac_field = periodic_binned_field(phi, theta, trac, n_phi=320, n_theta=180, sigma=1.0)

    disp_norm = mpl.colors.Normalize(vmin=0.0, vmax=float(np.nanpercentile(disp, 99.8)))
    trac_norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanmax(trac)))

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.85), sharey=True)

    mesh_disp = axes[0].pcolormesh(disp_x, disp_y, disp_field, shading='auto', cmap='viridis', norm=disp_norm, rasterized=True)
    style_map_axis(axes[0], show_xlabel=True, show_ylabel=True)
    axes[0].set_title(r'(a) Boundary displacement magnitude $|u|$', loc='left', pad=3)

    mesh_trac = axes[1].pcolormesh(trac_x, trac_y, trac_field, shading='auto', cmap='magma', norm=trac_norm, rasterized=True)
    style_map_axis(axes[1], show_xlabel=True, show_ylabel=False)
    axes[1].set_title(r'(b) Boundary traction error $|t-t^{\mathrm{obs}}|$', loc='left', pad=3)

    cbar1 = fig.colorbar(mesh_disp, ax=axes[0], orientation='horizontal', fraction=0.055, pad=0.18)
    cbar1.locator = mpl.ticker.MaxNLocator(5)
    cbar1.update_ticks()
    cbar1.set_label(r'$|u|$')
    cbar2 = fig.colorbar(mesh_trac, ax=axes[1], orientation='horizontal', fraction=0.055, pad=0.18)
    cbar2.locator = mpl.ticker.MaxNLocator(4)
    cbar2.formatter = mpl.ticker.FormatStrFormatter('%.1e')
    cbar2.update_ticks()
    cbar2.set_label(r'$|t-t^{\mathrm{obs}}|$')

    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.24, wspace=0.08)
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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COLUMN_WIDTH, 3.35),
        sharex=True,
        gridspec_kw={'height_ratios': [1.3, 1.0]},
    )

    axes[0].semilogy(iters, np.maximum(mu_err_pct, 1e-12), color='#1f77b4', label=r'$\mu$ relative error')
    axes[0].semilogy(iters, np.maximum(K_err_pct, 1e-12), color='#d62728', label=r'$K$ relative error')
    axes[0].set_ylabel('Relative error (%)')
    axes[0].set_title('(a) Parameter recovery', loc='left', pad=2)
    axes[0].legend(loc='upper right')

    axes[1].semilogy(iters, np.maximum(trac, 1e-16), color='#444444')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\mathcal{L}_{\mathrm{trac}}$')
    axes[1].set_title('(b) Traction mismatch', loc='left', pad=2)

    for ax in axes:
        ax.grid(True, which='both', alpha=0.35)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.14, hspace=0.18)
    save_figure(fig, CONV_PNG, CONV_PDF)
    plt.close(fig)


def main() -> None:
    mesh, hist, metrics = load_data()
    build_field_figure(mesh)
    build_convergence_figure(hist, metrics)


if __name__ == '__main__':
    main()
