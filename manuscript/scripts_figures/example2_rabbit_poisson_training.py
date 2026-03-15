from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure


REPO_ROOT = Path("/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry")
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
OUT_DIR = MANUSCRIPT_DIR / "figures_cmame_core" / "example2_rabbit_poisson"

HISTORY_JSON = REPO_ROOT / "runs" / "attempt20c_compact" / "rabbit_poisson_schwarz_attempt20c_compact_history.json"
METRICS_JSON = REPO_ROOT / "runs" / "attempt20c_compact" / "rabbit_poisson_schwarz_attempt20c_compact_metrics.json"

LOSS_FIG_PNG = OUT_DIR / "example2_rabbit_poisson_losses_pub.png"
LOSS_FIG_PDF = OUT_DIR / "example2_rabbit_poisson_losses_pub.pdf"
CHART_FIG_PNG = OUT_DIR / "example2_rabbit_poisson_chart_errors_pub.png"
CHART_FIG_PDF = OUT_DIR / "example2_rabbit_poisson_chart_errors_pub.pdf"

# Canonical attempt20c_compact settings, recorded in RESULTS.md and the
# successful run scripts that reproduce the benchmark.
W_PDE = 5.0
W_BC = 1.0
W_IF_VAL = 2.0


def load_history() -> dict:
    return json.loads(HISTORY_JSON.read_text())


def load_metrics() -> dict:
    return json.loads(METRICS_JSON.read_text())


def build_loss_figure(hist: dict) -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    iters = np.arange(1, len(hist["global_residual"]) + 1, dtype=float)
    pde = np.asarray(hist["global_residual"], dtype=float)
    bc = np.asarray(hist["bc_loss"], dtype=float)
    if_val = np.asarray(hist["interface_value"], dtype=float)
    if_flux = np.asarray(hist["interface_flux"], dtype=float)
    w_if_flux = np.asarray(hist["w_interface_flux_eff"], dtype=float)
    rel_l2 = np.asarray(hist["rel_l2_eval"], dtype=float)

    pde_w = W_PDE * pde
    bc_w = W_BC * bc
    if_val_w = W_IF_VAL * if_val
    if_flux_w = w_if_flux * if_flux
    total_proxy = pde_w + bc_w + if_val_w + if_flux_w

    palette = sns.color_palette("deep", 6)
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH, 4.45))
    panels = [
        (axes[0, 0], np.maximum(total_proxy, 1e-12), "Weighted total objective", r"$\mathcal{L}_{\mathrm{total}}$"),
        (axes[0, 1], np.maximum(pde, 1e-12), "PDE residual loss", r"$\mathcal{L}_{\mathrm{pde}}$"),
        (axes[0, 2], np.maximum(bc, 1e-12), "Boundary loss", r"$\mathcal{L}_{\mathrm{bc}}$"),
        (axes[1, 0], np.maximum(if_val, 1e-12), "Interface value loss", r"$\mathcal{L}_{\mathrm{if,val}}$"),
        (axes[1, 1], np.maximum(if_flux, 1e-12), "Interface flux loss", r"$\mathcal{L}_{\mathrm{if,flux}}$"),
        (axes[1, 2], np.maximum(rel_l2, 1e-12), "Evaluation relative $L^2$", r"$\mathrm{rel}\,L^2$"),
    ]

    for idx, (ax, values, title, ylabel) in enumerate(panels):
        ax.semilogy(iters, values, color=palette[idx], lw=1.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.35)

    axes[0, 0].text(
        0.03,
        0.06,
        r"$\mathcal{L}_{\mathrm{total}} = 5\mathcal{L}_{\mathrm{pde}} + \mathcal{L}_{\mathrm{bc}} + 2\mathcal{L}_{\mathrm{if,val}} + w_{\mathrm{if,flux}}\mathcal{L}_{\mathrm{if,flux}}$",
        transform=axes[0, 0].transAxes,
        fontsize=6.1,
        color="#444444",
    )

    fig.subplots_adjust(left=0.075, right=0.98, top=0.91, bottom=0.12, wspace=0.34, hspace=0.38)
    save_figure(fig, LOSS_FIG_PNG, LOSS_FIG_PDF)
    plt.close(fig)


def build_chart_error_figure(metrics: dict) -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_chart = metrics["per_chart"]
    chart_ids = np.asarray([item["chart_id"] for item in per_chart], dtype=int)
    rel_l2 = np.asarray([item["relative_l2_error"] for item in per_chart], dtype=float)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH * 0.58, 2.75))
    bars = ax.bar(chart_ids, rel_l2, color=sns.color_palette("crest", len(chart_ids)), width=0.78)
    worst_idx = int(np.argmax(rel_l2))
    bars[worst_idx].set_edgecolor("#111111")
    bars[worst_idx].set_linewidth(1.0)
    ax.set_xlabel("Chart ID")
    ax.set_ylabel(r"Final relative $L^2$ error")
    ax.set_title("Final chartwise error at the selected checkpoint", pad=4)
    ax.set_xticks(chart_ids)
    ax.set_ylim(0.0, max(rel_l2) * 1.22)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(np.mean(rel_l2), color="#444444", lw=1.0, ls="--", alpha=0.7)
    ax.text(
        0.985,
        0.95,
        rf"mean = {np.mean(rel_l2):.3e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color="#444444",
    )
    ax.annotate(
        f"worst chart = {chart_ids[worst_idx]}",
        xy=(chart_ids[worst_idx], rel_l2[worst_idx]),
        xytext=(chart_ids[worst_idx] - 2.0, rel_l2[worst_idx] * 1.10),
        fontsize=6.4,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"),
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
    save_figure(fig, CHART_FIG_PNG, CHART_FIG_PDF)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Generate rabbit Poisson training-history figure.")
    parser.parse_args()
    hist = load_history()
    metrics = load_metrics()
    build_loss_figure(hist)
    build_chart_error_figure(metrics)


if __name__ == "__main__":
    main()
