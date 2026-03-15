from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import pyvista as pv

from pub_style import apply_publication_style, save_figure


REPO_ROOT = Path("/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry")
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
OUT_DIR = MANUSCRIPT_DIR / "figures_cmame_core" / "example5_rabbit_elder"
CACHE_DIR = MANUSCRIPT_DIR / "scripts_figures" / "cache"

SOLUTION_NPZ = (
    REPO_ROOT
    / "runs"
    / "rabbit_inverse_elder_globalfield_small"
    / "rabbit_inverse_elder_atlas_schwarz_globalfield_small_solution.npz"
)
SURFACE_VTP = CACHE_DIR / "example5_rabbit_elder_surface.vtp"
FIG_FIELDS_PNG = OUT_DIR / "rabbit_elder_fields.png"
FIG_FIELDS_PDF = OUT_DIR / "rabbit_elder_fields.pdf"
FIG_VEL_PNG = OUT_DIR / "rabbit_elder_velocity_error.png"
FIG_VEL_PDF = OUT_DIR / "rabbit_elder_velocity_error.pdf"


def crop_white_border(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGBA")
    arr = np.asarray(img)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    mask = (alpha > 0) & (np.any(rgb < 250, axis=-1))
    if not np.any(mask):
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 20
    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad + 1, arr.shape[0])
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def load_or_build_surface() -> pv.PolyData:
    if SURFACE_VTP.exists():
        return pv.read(SURFACE_VTP)

    data = np.load(SOLUTION_NPZ, allow_pickle=True)
    points = data["points"].astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["pressure_pred"] = data["p_pred"].astype(np.float32)
    cloud["concentration"] = data["c_pred"].astype(np.float32)
    cloud["velocity_error_mag"] = data["u_error_mag"].astype(np.float32)
    surface = cloud.reconstruct_surface(nbr_sz=30, sample_spacing=0.012).clean()
    surface = surface.connectivity(extraction_mode="largest")
    surface = surface.interpolate(cloud, radius=0.03, sharpness=2.0)
    SURFACE_VTP.parent.mkdir(parents=True, exist_ok=True)
    surface.save(SURFACE_VTP)
    return surface


def render_panel(mesh: pv.DataSet, scalar: str, cmap: str, clim: tuple[float, float], out_path: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=scalar,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.25,
        diffuse=0.78,
        specular=0.10,
    )
    plotter.view_vector([1.0, 0.7, 1.0], viewup=(0.0, 1.0, 0.0))
    plotter.reset_camera()
    plotter.camera.zoom(1.15)
    plotter.show(screenshot=str(out_path), auto_close=True)


def add_image_panel(ax, image: np.ndarray, title: str, cmap, norm, cbar_label: str) -> None:
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, pad=4)
    cax = inset_axes(
        ax,
        width="62%",
        height="5.6%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6.8, width=0.6, length=2.2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(cbar_label, fontsize=7.0, labelpad=2)


def build_fields_figure(mesh: pv.PolyData) -> None:
    pressure = np.asarray(mesh["pressure_pred"])
    concentration = np.asarray(mesh["concentration"])

    p_abs = float(max(abs(np.quantile(pressure, 0.005)), abs(np.quantile(pressure, 0.995))))
    c_lo = float(np.quantile(concentration, 0.005))
    c_hi = float(np.quantile(concentration, 0.995))

    p_norm = TwoSlopeNorm(vmin=-p_abs, vcenter=0.0, vmax=p_abs)
    c_norm = Normalize(vmin=c_lo, vmax=c_hi)

    p_img_path = CACHE_DIR / "example5_pressure.png"
    c_img_path = CACHE_DIR / "example5_concentration.png"
    render_panel(mesh, "pressure_pred", "coolwarm", (-p_abs, p_abs), p_img_path)
    render_panel(mesh, "concentration", "viridis", (c_lo, c_hi), c_img_path)

    pressure_img = crop_white_border(p_img_path)
    concentration_img = crop_white_border(c_img_path)

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.7))
    add_image_panel(axes[0], pressure_img, "Pressure", plt.get_cmap("coolwarm"), p_norm, r"$p_h$")
    add_image_panel(axes[1], concentration_img, "Concentration", plt.get_cmap("viridis"), c_norm, r"$c_h$")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.18, wspace=0.03)
    save_figure(fig, FIG_FIELDS_PNG, FIG_FIELDS_PDF)
    plt.close(fig)


def build_velocity_figure(mesh: pv.PolyData) -> None:
    velocity_error = np.asarray(mesh["velocity_error_mag"])
    vel_hi = float(np.quantile(velocity_error, 0.995))
    vel_norm = Normalize(vmin=0.0, vmax=vel_hi)

    vel_img_path = CACHE_DIR / "example5_velocity_error.png"
    render_panel(mesh, "velocity_error_mag", "viridis", (0.0, vel_hi), vel_img_path)
    vel_img = crop_white_border(vel_img_path)

    fig, ax = plt.subplots(1, 1, figsize=(2.7, 2.7))
    add_image_panel(ax, vel_img, "Velocity error magnitude", plt.get_cmap("viridis"), vel_norm, r"$\|\mathbf{v}_h-\mathbf{v}^\ast\|$")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.18)
    save_figure(fig, FIG_VEL_PNG, FIG_VEL_PDF)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Generate publication-ready rabbit Elder figures.")
    parser.parse_args()

    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mesh = load_or_build_surface()
    build_fields_figure(mesh)
    build_velocity_figure(mesh)


if __name__ == "__main__":
    main()
