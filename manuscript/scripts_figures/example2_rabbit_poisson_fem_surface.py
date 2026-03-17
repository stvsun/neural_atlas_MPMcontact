from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import pyvista as pv
import torch

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure


REPO_ROOT = Path("/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry")
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
OUT_DIR = MANUSCRIPT_DIR / "figures_cmame_core" / "example2_rabbit_poisson"
CACHE_DIR = MANUSCRIPT_DIR / "scripts_figures" / "cache"

FEM_SOLUTION = REPO_ROOT / "runs" / "fem_highres" / "rabbit_poisson_schwarz_fem_highres_n56_solution.npz"
ATLAS_DATA = REPO_ROOT / "runs" / "atlas_vol" / "rabbit_atlas_data.npz"
ATLAS_CKPT = REPO_ROOT / "runs" / "atlas_vol_trained" / "rabbit_atlas_trained.pt"
BOUNDARY_RAW = REPO_ROOT / "runs" / "poisson_rabbit_accurate_warmstart_20260214_232536" / "rabbit_poisson_pubready_boundary_raw.vtu"

RECON_SURFACE = CACHE_DIR / "example2_rabbit_reconstruct.vtp"
INTERP_SURFACE = CACHE_DIR / "example2_rabbit_fem_surface_interpolated.vtp"

FIG_PATH = OUT_DIR / "example2_rabbit_poisson_fem_surface_pub.png"
FIG_PDF = OUT_DIR / "example2_rabbit_poisson_fem_surface_pub.pdf"

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from run_poisson_rabbit_atlas_schwarz import load_atlas_models  # noqa: E402


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


def local_coords_np(
    x: np.ndarray,
    seed: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    n_vec: np.ndarray,
) -> np.ndarray:
    d = x - seed[np.newaxis, :]
    return np.stack([d @ t1, d @ t2, d @ n_vec], axis=-1)


def evaluate_mask_np(
    xi: np.ndarray,
    mask_net: torch.nn.Module,
    support_r: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    xi_t = torch.tensor(xi, device=device, dtype=dtype)
    with torch.no_grad():
        logit = mask_net(xi_t, chart_scale=support_r)
    return logit.cpu().numpy()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compute_chart_entropy(points: np.ndarray) -> np.ndarray:
    atlas = np.load(ATLAS_DATA)
    seeds = np.asarray(atlas["seed_points"], dtype=np.float64)
    t1s = np.asarray(atlas["frame_t1"], dtype=np.float64)
    t2s = np.asarray(atlas["frame_t2"], dtype=np.float64)
    nvecs = np.asarray(atlas["frame_n"], dtype=np.float64)
    support_radii = np.asarray(atlas["support_radii"], dtype=np.float64)

    device = torch.device("cpu")
    dtype = torch.float64
    _, masks, _ = load_atlas_models(str(ATLAS_CKPT), device=device, dtype=dtype)
    support_r_t = [
        torch.tensor(float(r), device=device, dtype=dtype) for r in support_radii
    ]

    logits_all = np.zeros((points.shape[0], len(masks)), dtype=np.float64)
    for i, mask in enumerate(masks):
        xi_i = local_coords_np(points, seeds[i], t1s[i], t2s[i], nvecs[i])
        logits_all[:, i] = evaluate_mask_np(xi_i, mask, support_r_t[i], device, dtype)

    weights = softmax_np(logits_all)
    eps = 1e-12
    return -np.sum(weights * np.log(weights + eps), axis=1)


def load_or_build_surface() -> pv.PolyData:
    if INTERP_SURFACE.exists():
        return pv.read(INTERP_SURFACE)

    data = np.load(FEM_SOLUTION)
    points = np.asarray(data["points"], dtype=np.float64)
    pressure = np.asarray(data["u_pred"], dtype=np.float64)
    error = np.asarray(data["u_error_mag"], dtype=np.float64)
    entropy = compute_chart_entropy(points)

    cloud = pv.PolyData(points)
    cloud["pressure"] = pressure
    cloud["pressure_error_abs"] = error
    cloud["chart_entropy"] = entropy

    if RECON_SURFACE.exists():
        surface = pv.read(RECON_SURFACE)
    else:
        boundary_points = pv.read(BOUNDARY_RAW)
        surface = pv.PolyData(boundary_points.points).reconstruct_surface(nbr_sz=20, sample_spacing=0.01)
        RECON_SURFACE.parent.mkdir(parents=True, exist_ok=True)
        surface.save(RECON_SURFACE)

    interpolated = surface.interpolate(cloud, radius=0.03, sharpness=2.0)
    interpolated.save(INTERP_SURFACE)
    return interpolated


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


def add_image_panel(ax, image: np.ndarray, title: str, label: str, cmap, norm, cbar_label: str):
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, pad=4)
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.18"),
    )
    cax = inset_axes(
        ax,
        width="58%",
        height="5.5%",
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


def build_figure() -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mesh = load_or_build_surface()
    pressure = np.asarray(mesh["pressure"])
    error = np.asarray(mesh["pressure_error_abs"])
    entropy = np.asarray(mesh["chart_entropy"])

    pred_abs = float(max(abs(np.quantile(pressure, 0.005)), abs(np.quantile(pressure, 0.995))))
    err_max = float(np.quantile(error, 0.995))
    entropy_max = float(np.quantile(entropy, 0.995))

    pred_norm = TwoSlopeNorm(vmin=-pred_abs, vcenter=0.0, vmax=pred_abs)
    err_norm = Normalize(vmin=0.0, vmax=err_max)
    entropy_norm = Normalize(vmin=0.0, vmax=entropy_max)

    pred_img_path = CACHE_DIR / "example2_fem_pred.png"
    err_img_path = CACHE_DIR / "example2_fem_err.png"
    entropy_img_path = CACHE_DIR / "example2_fem_entropy.png"
    render_panel(mesh, "pressure", "coolwarm", (-pred_abs, pred_abs), pred_img_path)
    render_panel(mesh, "pressure_error_abs", "magma", (0.0, err_max), err_img_path)
    render_panel(mesh, "chart_entropy", "magma", (0.0, entropy_max), entropy_img_path)

    pred_img = crop_white_border(pred_img_path)
    err_img = crop_white_border(err_img_path)
    entropy_img = crop_white_border(entropy_img_path)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.95))
    add_image_panel(axes[0], pred_img, "Predicted scalar field", "(a)", plt.get_cmap("coolwarm"), pred_norm, r"$u_h$")
    add_image_panel(axes[1], err_img, "Absolute error", "(b)", plt.get_cmap("magma"), err_norm, r"$|u_h-u^\ast|$")
    add_image_panel(axes[2], entropy_img, "Chart entropy", "(c)", plt.get_cmap("magma"), entropy_norm, r"$H(\omega)$")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.18, wspace=0.04)
    save_figure(fig, FIG_PATH, FIG_PDF)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Generate publication-ready rabbit FEM surface figure.")
    parser.parse_args()
    build_figure()


if __name__ == "__main__":
    main()
