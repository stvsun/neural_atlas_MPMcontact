from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure

REPO_ROOT = Path('/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry')
MANUSCRIPT_DIR = REPO_ROOT / 'manuscript'
FIG_DIR = MANUSCRIPT_DIR / 'figures_cmame_core' / 'example4_torus_inverse_schwarz_dual'

TRACTION_DIR = REPO_ROOT / 'runs' / 'torus_schwarz_dual_accurate_traction_20260215_140800'
DISP_DIR = REPO_ROOT / 'runs' / 'torus_schwarz_dual_accurate_displacement_20260215_140200'

TRACTION_VTU = TRACTION_DIR / 'torus_inverse_neohookean_schwarz_traction_accurate_best_boundary_fields.vtu'
DISP_VTU = DISP_DIR / 'torus_inverse_neohookean_schwarz_displacement_accurate_best_boundary_fields.vtu'
TRACTION_HISTORY = TRACTION_DIR / 'torus_inverse_neohookean_schwarz_traction_accurate_history.json'
DISP_HISTORY = DISP_DIR / 'torus_inverse_neohookean_schwarz_displacement_accurate_history.json'
TRACTION_METRICS = TRACTION_DIR / 'torus_inverse_neohookean_schwarz_traction_accurate_metrics.json'
DISP_METRICS = DISP_DIR / 'torus_inverse_neohookean_schwarz_displacement_accurate_metrics.json'

UNCERTAINTY_PNG = FIG_DIR / 'torus_inverse_uncertainty.png'
UNCERTAINTY_PDF = FIG_DIR / 'torus_inverse_uncertainty.pdf'
TRACTION_MAP_PNG = FIG_DIR / 'torus_inverse_traction_error.png'
TRACTION_MAP_PDF = FIG_DIR / 'torus_inverse_traction_error.pdf'

MU_TRUE = 1.8
K_TRUE = 25.0
TWO_PI = 2.0 * np.pi


def wrap_theta(theta: np.ndarray) -> np.ndarray:
    return ((theta + np.pi) % TWO_PI) - np.pi


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def infer_theta_from_points(points: np.ndarray) -> np.ndarray:
    rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    major_radius = float(np.mean(rho))
    return wrap_theta(np.arctan2(points[:, 2], rho - major_radius))


def periodic_binned_field(
    phi: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    *,
    n_phi: int = 320,
    n_theta: int = 180,
    sigma: float = 1.0,
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


def build_uncertainty_figure() -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    disp_hist = load_json(DISP_HISTORY)
    trac_hist = load_json(TRACTION_HISTORY)

    curves = [
        ('Displacement data', disp_hist, '#1273b6', '-'),
        ('Traction data', trac_hist, '#d95f02', '--'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.55), sharex=False)
    panel_cfg = [
        ('mu_mean', 'mu_std', r'(a) Shear modulus $\mu$', MU_TRUE),
        ('K_mean', 'K_std', r'(b) Bulk modulus $K$', K_TRUE),
    ]

    for ax, (mean_key, std_key, title, truth) in zip(axes, panel_cfg):
        for label, hist, color, linestyle in curves:
            mean = np.asarray(hist[mean_key], dtype=float)
            std = np.asarray(hist[std_key], dtype=float)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.16)
        ax.axhline(truth, color='#444444', linestyle=':', linewidth=1.0)
        ax.set_title(title, loc='left', pad=3)
        ax.set_xlabel('Epoch')
        ax.grid(True, which='both', alpha=0.35)

    axes[0].set_ylabel(r'$\mu$')
    axes[1].set_ylabel(r'$K$')
    axes[1].legend(loc='upper right')

    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.20, wspace=0.20)
    save_figure(fig, UNCERTAINTY_PNG, UNCERTAINTY_PDF)
    plt.close(fig)


def build_traction_maps_figure() -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meshes = [
        ('(a) Traction-data Schwarz', pv.read(TRACTION_VTU)),
        ('(b) Displacement-data Schwarz', pv.read(DISP_VTU)),
    ]

    prepared = []
    vmax = 0.0
    for title, mesh in meshes:
        phi = np.asarray(mesh['phi'], dtype=float)
        theta = infer_theta_from_points(np.asarray(mesh.points, dtype=float))
        traction_error = np.asarray(mesh['traction_error_mag'], dtype=float)
        grid_x, grid_y, field = periodic_binned_field(phi, theta, traction_error, n_phi=320, n_theta=180, sigma=1.0)
        vmax = max(vmax, float(np.nanmax(traction_error)))
        prepared.append((title, grid_x, grid_y, field))

    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN_WIDTH * 0.47, 4.05), sharex=True, sharey=True)
    mappable = None
    for ax, (title, grid_x, grid_y, field) in zip(axes, prepared):
        mappable = ax.pcolormesh(grid_x, grid_y, field, shading='auto', cmap='magma', norm=norm, rasterized=True)
        style_map_axis(ax, show_xlabel=ax is axes[-1], show_ylabel=True)
        ax.set_title(title, loc='left', pad=3)

    cbar = fig.colorbar(mappable, ax=axes, orientation='vertical', fraction=0.05, pad=0.03)
    cbar.locator = mpl.ticker.MaxNLocator(5)
    cbar.update_ticks()
    cbar.set_label(r'$|t-t^{\mathrm{obs}}|$')

    fig.subplots_adjust(left=0.14, right=0.88, top=0.96, bottom=0.12, hspace=0.16)
    save_figure(fig, TRACTION_MAP_PNG, TRACTION_MAP_PDF)
    plt.close(fig)


def main() -> None:
    build_uncertainty_figure()
    build_traction_maps_figure()


if __name__ == '__main__':
    main()
