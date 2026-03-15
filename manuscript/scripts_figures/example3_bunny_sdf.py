from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import pyvista as pv

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure

REPO_ROOT = Path('/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry')
MANUSCRIPT_DIR = REPO_ROOT / 'manuscript'
OUT_DIR = MANUSCRIPT_DIR / 'figures_cmame_core' / 'example3_bunny_sdf'
CACHE_DIR = MANUSCRIPT_DIR / 'scripts_figures' / 'cache'

SDF_RUN = REPO_ROOT / 'runs' / 'bunny_sdf_v3'
SLICES_PNG = SDF_RUN / 'rabbit_sdf_slices.png'
HISTORY_JSON = SDF_RUN / 'rabbit_sdf_history.json'
META_JSON = SDF_RUN / 'rabbit_sdf_meta.json'
SURFACE_VTU = REPO_ROOT / 'paraview' / 'bunny_poisson_8chart_intpre20k' / 'bunny_poisson_8chart_intpre20k_surface.vtu'

FIG1_PNG = OUT_DIR / 'example3_bunny_sdf_learning_pub.png'
FIG1_PDF = OUT_DIR / 'example3_bunny_sdf_learning_pub.pdf'
FIG2_PNG = OUT_DIR / 'example3_bunny_sdf_surface_pub.png'
FIG2_PDF = OUT_DIR / 'example3_bunny_sdf_surface_pub.pdf'

CAMERA_VECTOR = [1.0, 0.72, 1.08]
VIEWUP = (0.0, 1.0, 0.0)


def crop_white_border(path: Path, pad: int = 16) -> np.ndarray:
    img = Image.open(path).convert('RGBA')
    arr = np.asarray(img)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    mask = (alpha > 0) & (np.any(rgb < 250, axis=-1))
    if not np.any(mask):
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad + 1, arr.shape[0])
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def render_surface(
    mesh: pv.DataSet,
    scalar: str,
    cmap: str,
    clim: tuple[float, float],
    out_path: Path,
    categorical: bool = False,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800), lighting='three lights')
    plotter.set_background('white')
    plotter.enable_anti_aliasing('fxaa')
    kwargs = dict(
        scalars=scalar,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.28,
        diffuse=0.76,
        specular=0.08,
    )
    if categorical:
        kwargs['categories'] = True
    plotter.add_mesh(mesh, **kwargs)
    plotter.view_vector(CAMERA_VECTOR, viewup=VIEWUP)
    plotter.reset_camera()
    plotter.camera.zoom(1.28)
    plotter.show(screenshot=str(out_path), auto_close=True)


def add_image_panel(ax, image: np.ndarray, title: str, label: str) -> None:
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, pad=4)
    ax.text(
        0.03, 0.97, label, transform=ax.transAxes,
        ha='left', va='top', fontsize=8, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, boxstyle='round,pad=0.18')
    )


def add_cbar(ax, cmap, norm, label: str) -> None:
    cax = inset_axes(
        ax, width='58%', height='5.5%', loc='lower center',
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0.0
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6.8, width=0.6, length=2.2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(label, fontsize=7.0, labelpad=2)


def add_losses_panel(ax) -> None:
    hist = json.loads(HISTORY_JSON.read_text())
    total = np.asarray(hist['total'], dtype=float)
    surface = np.asarray(hist['surface'], dtype=float)
    normal = np.asarray(hist['normal'], dtype=float)
    eikonal = np.asarray(hist['eikonal'], dtype=float)
    sign = np.asarray(hist['sign'], dtype=float)
    step = max(1, len(total) // 700)
    epochs = np.arange(1, len(total) + 1, dtype=int)[::step]

    ax.semilogy(epochs, total[::step], label='total', color='#1f77b4')
    ax.semilogy(epochs, surface[::step], label='surface', color='#d62728')
    ax.semilogy(epochs, normal[::step], label='normal', color='#2ca02c')
    ax.semilogy(epochs, eikonal[::step], label='Eikonal', color='#9467bd')
    ax.semilogy(epochs, sign[::step], label='sign-anchor', color='#8c564b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    ax.set_title('SDF training diagnostics', pad=4)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='upper right', fontsize=6.2, ncol=1, frameon=False)
    ax.text(
        0.03, 0.97, '(b)', transform=ax.transAxes,
        ha='left', va='top', fontsize=8, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, boxstyle='round,pad=0.18')
    )


def build_learning_figure(slices_img: np.ndarray) -> None:
    meta = json.loads(META_JSON.read_text())
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.15))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.995, top=0.91, bottom=0.13, wspace=0.14)

    ax0 = fig.add_subplot(gs[0, 0])
    add_image_panel(ax0, slices_img, 'Learned signed-distance slices', '(a)')

    ax1 = fig.add_subplot(gs[0, 1])
    add_losses_panel(ax1)

    fig.text(
        0.5, 0.03,
        (
            f"Canonical run: bunny_sdf_v3  |  surface loss = {meta['final_surface']:.3e}, "
            f"normal loss = {meta['final_normal']:.3e}, sign loss = {meta['final_sign']:.3e}"
        ),
        ha='center', va='bottom', fontsize=7.0
    )
    save_figure(fig, FIG1_PNG, FIG1_PDF)
    plt.close(fig)


def build_surface_figure(chart_img: np.ndarray, blend_img: np.ndarray) -> None:
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.05))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.995, top=0.92, bottom=0.16, wspace=0.10)

    ax0 = fig.add_subplot(gs[0, 0])
    add_image_panel(ax0, chart_img, 'Downstream surface chart partition', '(a)')
    chart_norm = mpl.colors.Normalize(vmin=0, vmax=7)
    add_cbar(ax0, plt.get_cmap('tab10'), chart_norm, 'chart id')

    ax1 = fig.add_subplot(gs[0, 1])
    add_image_panel(ax1, blend_img, 'Dominant blend weight on the Bunny surface', '(b)')
    blend_norm = mpl.colors.Normalize(vmin=0.48, vmax=1.0)
    add_cbar(ax1, plt.get_cmap('viridis'), blend_norm, r'$\max_i\,\omega_i$')

    save_figure(fig, FIG2_PNG, FIG2_PDF)
    plt.close(fig)


def build_figures() -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    surface = pv.read(SURFACE_VTU)

    chart_img_path = CACHE_DIR / 'example3_bunny_chartid.png'
    blend_img_path = CACHE_DIR / 'example3_bunny_blend.png'
    render_surface(surface, 'chart_id', 'tab10', (0, 7), chart_img_path, categorical=True)
    render_surface(surface, 'blend_weight', 'viridis', (0.48, 1.0), blend_img_path, categorical=False)

    slices_img = crop_white_border(SLICES_PNG)
    chart_img = crop_white_border(chart_img_path)
    blend_img = crop_white_border(blend_img_path)

    build_learning_figure(slices_img)
    build_surface_figure(chart_img, blend_img)


if __name__ == '__main__':
    build_figures()
