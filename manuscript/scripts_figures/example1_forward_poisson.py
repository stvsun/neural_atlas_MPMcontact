from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.tri as mtri
import numpy as np
import torch

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure


REPO_ROOT = Path("/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry")
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pinn_3d_ellipsoid_mapped_sphere as ellipsoid_mod  # noqa: E402
import run_poisson_star3d_mapped as star_mod  # noqa: E402


ELLIPSOID_CACHE = REPO_ROOT / "manuscript" / "scripts_figures" / "cache" / "example1_ellipsoid_checkpoint.pt"
ELLIPSOID_RUN_LOG = REPO_ROOT / "runs" / "examples" / "ellipsoid3d_20260212_180204" / "run.log"
STAR_MODEL_PATH = REPO_ROOT / "runs" / "examples" / "meshfree_accel_20260212_194730" / "poisson" / "poisson_star3d_pinn.pt"
STAR_MAPPING_PATH = REPO_ROOT / "runs" / "examples" / "meshfree_accel_20260212_194730" / "map_star_retry" / "mapping_star.pt"
OUT_DIR = REPO_ROOT / "manuscript" / "figures_cmame_core" / "example1_forward_poisson"


def train_or_load_ellipsoid_model() -> tuple[torch.nn.Module, dict, dict]:
    ellipsoid_mod.set_seed(42)
    device = torch.device("cpu")
    dtype = torch.float64

    if ELLIPSOID_CACHE.exists():
        ckpt = torch.load(ELLIPSOID_CACHE, map_location=device)
        model = ellipsoid_mod.PINN3D(width=ckpt["model_kwargs"]["width"], depth=ckpt["model_kwargs"]["depth"]).to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt["history"], ckpt["metrics"]

    model = ellipsoid_mod.PINN3D(width=64, depth=4).to(device=device, dtype=dtype)
    history = ellipsoid_mod.train_pinn(
        model=model,
        a_axis=2.0,
        b_axis=1.4,
        c_axis=0.9,
        n_epochs=3000,
        lr=1e-3,
        n_int=512,
        n_bc=256,
        bc_weight=5.0,
        target_total_loss=1e-4,
        log_every=300,
    )
    metrics = ellipsoid_mod.evaluate_model(model, n_eval=20000)
    ELLIPSOID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {"width": 64, "depth": 4},
            "history": history,
            "metrics": metrics,
        },
        ELLIPSOID_CACHE,
    )
    model.eval()
    return model, history, metrics


def load_star_models() -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    device = torch.device("cpu")
    dtype = torch.float64

    star_ckpt = torch.load(STAR_MODEL_PATH, map_location=device)
    model = star_mod.ScalarPINN(width=star_ckpt["model_kwargs"]["width"], depth=star_ckpt["model_kwargs"]["depth"]).to(device=device, dtype=dtype)
    model.load_state_dict(star_ckpt["model_state"])
    model.eval()

    map_ckpt = torch.load(STAR_MAPPING_PATH, map_location=device)
    mapping = star_mod.MappingNet(
        width=map_ckpt["model_kwargs"]["width"],
        depth=map_ckpt["model_kwargs"]["depth"],
        disp_cap=map_ckpt["model_kwargs"]["disp_cap"],
    ).to(device=device, dtype=dtype)
    mapping.load_state_dict(map_ckpt["model_state"])
    mapping.eval()

    for net in (model, mapping):
        for param in net.parameters():
            param.requires_grad_(False)
    return model, mapping, star_ckpt["metrics"]


def finite_limits(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    return float(vals.min()), float(vals.max())


def build_regular_disk_triangulation(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nrows, ncols = mask.shape
    valid = np.flatnonzero(mask.ravel())
    index = -np.ones(mask.shape, dtype=int)
    index.ravel()[valid] = np.arange(valid.size)
    triangles: list[tuple[int, int, int]] = []
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            if all(mask[c] for c in corners):
                a = index[i, j]
                b = index[i + 1, j]
                c = index[i, j + 1]
                d = index[i + 1, j + 1]
                triangles.append((a, b, c))
                triangles.append((b, d, c))
    return valid, np.asarray(triangles, dtype=int)


def add_panel_mesh(ax, mesh, title: str, panel_label: str, x_label: str, y_label: str, cbar_label: str):
    ax.set_title(title, pad=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.text(
        0.03,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.18"),
    )
    cax = inset_axes(ax, width="4.8%", height="58%", loc="center right", borderpad=0.9)
    cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=6.6, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(cbar_label, fontsize=6.9, labelpad=4)


def build_ellipsoid_arrays(model: torch.nn.Module):
    xi, eta, pred_map, exact_map, err_map = ellipsoid_mod.mapped_slice_prediction(model, n_grid=280, zeta=0.0)
    x_phys, y_phys, pred_phys = ellipsoid_mod.physical_slice_prediction(model, a_axis=2.0, b_axis=1.4, c_axis=0.9, n_grid=280, z_phys=0.0)
    exact_phys = np.full_like(pred_phys, np.nan)
    mask = np.isfinite(pred_phys)
    exact_phys[mask] = 1.0 - (x_phys[mask] / 2.0) ** 2 - (y_phys[mask] / 1.4) ** 2
    err_phys = np.abs(pred_phys - exact_phys)
    return {
        "mapped": (xi, eta, pred_map, err_map),
        "physical": (x_phys, y_phys, pred_phys, err_phys),
    }


def build_star_arrays(model: torch.nn.Module, mapping: torch.nn.Module):
    line = np.linspace(-1.0, 1.0, 280)
    y1, y2 = np.meshgrid(line, line)
    mask = (y1**2 + y2**2) <= 1.0
    valid, triangles = build_regular_disk_triangulation(mask)
    ref_pts = np.column_stack([y1.ravel()[valid], y2.ravel()[valid], np.zeros(valid.size)])
    ref_t = torch.tensor(ref_pts, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    with torch.no_grad():
        x_phys = mapping(ref_t).cpu().numpy()
        pred = model(ref_t).cpu().numpy().reshape(-1)
        exact = star_mod.exact_solution(torch.tensor(x_phys, dtype=ref_t.dtype)).cpu().numpy().reshape(-1)
    err = np.abs(pred - exact)

    pred_map = np.full_like(y1, np.nan, dtype=float)
    err_map = np.full_like(y1, np.nan, dtype=float)
    pred_map.ravel()[valid] = pred
    err_map.ravel()[valid] = err

    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1], triangles)
    boundary_theta = np.linspace(0.0, 2.0 * np.pi, 721)
    boundary_ref = np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta), np.zeros(boundary_theta.size)])
    with torch.no_grad():
        boundary_phys = mapping(torch.tensor(boundary_ref, dtype=ref_t.dtype)).cpu().numpy()
    return {
        "mapped": (y1, y2, pred_map, err_map),
        "physical": (tri, x_phys, pred, err, boundary_phys),
    }


def plot_ellipsoid_figure(output_png: Path, output_pdf: Path, model: torch.nn.Module, metrics: dict):
    arrays = build_ellipsoid_arrays(model)
    pred_vals = [arrays["mapped"][2], arrays["physical"][2]]
    err_vals = [arrays["mapped"][3], arrays["physical"][3]]
    pred_lim = max(max(abs(finite_limits(v)[0]), abs(finite_limits(v)[1])) for v in pred_vals)
    err_lim = max(finite_limits(v)[1] for v in err_vals)
    pred_norm = TwoSlopeNorm(vmin=-pred_lim, vcenter=0.0, vmax=pred_lim)
    err_norm = Normalize(vmin=0.0, vmax=err_lim)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 6.15))
    xi, eta, pred_map, err_map = arrays["mapped"]
    x_phys, y_phys, pred_phys, err_phys = arrays["physical"]

    mesh = axes[0, 0].pcolormesh(xi, eta, np.ma.masked_invalid(pred_map), shading="auto", cmap="coolwarm", norm=pred_norm, rasterized=True)
    axes[0, 0].add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[0, 0].set_xlim(-1.02, 1.02); axes[0, 0].set_ylim(-1.02, 1.02)
    add_panel_mesh(axes[0, 0], mesh, "Mapped domain: predicted field", "(a)", r"$\zeta_1$", r"$\zeta_2$", r"$u_h$")

    mesh = axes[0, 1].pcolormesh(x_phys, y_phys, np.ma.masked_invalid(pred_phys), shading="auto", cmap="coolwarm", norm=pred_norm, rasterized=True)
    axes[0, 1].add_patch(Ellipse((0.0, 0.0), width=4.0, height=2.8, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[0, 1].set_xlim(-2.05, 2.05); axes[0, 1].set_ylim(-1.45, 1.45)
    add_panel_mesh(axes[0, 1], mesh, "Physical domain: predicted field", "(b)", r"$x_1$", r"$x_2$", r"$u_h$")

    mesh = axes[1, 0].pcolormesh(xi, eta, np.ma.masked_invalid(err_map), shading="auto", cmap="magma", norm=err_norm, rasterized=True)
    axes[1, 0].add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[1, 0].set_xlim(-1.02, 1.02); axes[1, 0].set_ylim(-1.02, 1.02)
    add_panel_mesh(axes[1, 0], mesh, "Mapped domain: absolute error", "(c)", r"$\zeta_1$", r"$\zeta_2$", r"$|u_h-u^\ast|$")

    mesh = axes[1, 1].pcolormesh(x_phys, y_phys, np.ma.masked_invalid(err_phys), shading="auto", cmap="magma", norm=err_norm, rasterized=True)
    axes[1, 1].add_patch(Ellipse((0.0, 0.0), width=4.0, height=2.8, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[1, 1].set_xlim(-2.05, 2.05); axes[1, 1].set_ylim(-1.45, 1.45)
    add_panel_mesh(axes[1, 1], mesh, "Physical domain: absolute error", "(d)", r"$x_1$", r"$x_2$", r"$|u_h-u^\ast|$")

    for ax in axes.flat:
        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.075, wspace=0.22, hspace=0.20)
    save_figure(fig, output_png, output_pdf)
    plt.close(fig)

    return {
        "metrics": metrics,
        "figure_type": "ellipsoid",
        "panels": ["mapped_predicted", "physical_predicted", "mapped_error", "physical_error"],
    }


def plot_star_figure(output_png: Path, output_pdf: Path, model: torch.nn.Module, mapping: torch.nn.Module, metrics: dict):
    arrays = build_star_arrays(model, mapping)
    pred_vals = [arrays["mapped"][2], arrays["physical"][2]]
    err_vals = [arrays["mapped"][3], arrays["physical"][3]]
    pred_lim = max(max(abs(finite_limits(v)[0]), abs(finite_limits(v)[1])) for v in pred_vals)
    err_lim = max(finite_limits(v)[1] for v in err_vals)
    pred_norm = TwoSlopeNorm(vmin=-pred_lim, vcenter=0.0, vmax=pred_lim)
    err_norm = Normalize(vmin=0.0, vmax=err_lim)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 6.15))
    y1, y2, pred_map, err_map = arrays["mapped"]
    tri, x_phys, pred_phys, err_phys, boundary_phys = arrays["physical"]

    mesh = axes[0, 0].pcolormesh(y1, y2, np.ma.masked_invalid(pred_map), shading="auto", cmap="coolwarm", norm=pred_norm, rasterized=True)
    axes[0, 0].add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[0, 0].set_xlim(-1.02, 1.02); axes[0, 0].set_ylim(-1.02, 1.02)
    add_panel_mesh(axes[0, 0], mesh, "Mapped domain: predicted field", "(a)", r"$\zeta_1$", r"$\zeta_2$", r"$u_h$")

    mesh = axes[0, 1].tripcolor(tri, pred_phys, shading="gouraud", cmap="coolwarm", norm=pred_norm, rasterized=True)
    axes[0, 1].plot(boundary_phys[:, 0], boundary_phys[:, 1], color="#202020", linewidth=0.9)
    axes[0, 1].set_xlim(boundary_phys[:, 0].min() - 0.04, boundary_phys[:, 0].max() + 0.04)
    axes[0, 1].set_ylim(boundary_phys[:, 1].min() - 0.04, boundary_phys[:, 1].max() + 0.04)
    add_panel_mesh(axes[0, 1], mesh, "Physical domain: projected central slice", "(b)", r"$x_1$", r"$x_2$", r"$u_h$")

    mesh = axes[1, 0].pcolormesh(y1, y2, np.ma.masked_invalid(err_map), shading="auto", cmap="magma", norm=err_norm, rasterized=True)
    axes[1, 0].add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.9, edgecolor="#202020"))
    axes[1, 0].set_xlim(-1.02, 1.02); axes[1, 0].set_ylim(-1.02, 1.02)
    add_panel_mesh(axes[1, 0], mesh, "Mapped domain: absolute error", "(c)", r"$\zeta_1$", r"$\zeta_2$", r"$|u_h-u^\ast|$")

    mesh = axes[1, 1].tripcolor(tri, err_phys, shading="gouraud", cmap="magma", norm=err_norm, rasterized=True)
    axes[1, 1].plot(boundary_phys[:, 0], boundary_phys[:, 1], color="#202020", linewidth=0.9)
    axes[1, 1].set_xlim(boundary_phys[:, 0].min() - 0.04, boundary_phys[:, 0].max() + 0.04)
    axes[1, 1].set_ylim(boundary_phys[:, 1].min() - 0.04, boundary_phys[:, 1].max() + 0.04)
    add_panel_mesh(axes[1, 1], mesh, "Physical domain: projected error", "(d)", r"$x_1$", r"$x_2$", r"$|u_h-u^\ast|$")

    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.075, wspace=0.22, hspace=0.20)
    save_figure(fig, output_png, output_pdf)
    plt.close(fig)

    return {
        "metrics": metrics,
        "figure_type": "star",
        "panels": ["mapped_predicted", "physical_predicted", "mapped_error", "physical_error"],
        "note": "Physical-domain panels show the projected image of the central reference slice under the learned global map.",
    }


def main() -> None:
    parser = ArgumentParser(description="Generate publication-ready Example 1 figures.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    apply_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ell_model, _, ell_metrics = train_or_load_ellipsoid_model()
    star_model, star_mapping, star_metrics = load_star_models()

    ell_summary = plot_ellipsoid_figure(
        args.out_dir / "example1_ellipsoid_pub.png",
        args.out_dir / "example1_ellipsoid_pub.pdf",
        ell_model,
        ell_metrics,
    )
    star_summary = plot_star_figure(
        args.out_dir / "example1_star_pub.png",
        args.out_dir / "example1_star_pub.pdf",
        star_model,
        star_mapping,
        star_metrics,
    )

    (args.out_dir / "example1_forward_poisson_pub_metrics.json").write_text(
        json.dumps(
            {
                "ellipsoid": ell_summary,
                "star": star_summary,
                "ellipsoid_log": str(ELLIPSOID_RUN_LOG),
                "star_checkpoint": str(STAR_MODEL_PATH),
                "star_mapping_checkpoint": str(STAR_MAPPING_PATH),
                "notes": "Ellipsoid model regenerated from canonical logged settings because the original run did not save a checkpoint.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
