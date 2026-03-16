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
RUN_DIR = REPO_ROOT / "runs" / "fem_sweep"
HIGHRES_DIR = REPO_ROOT / "runs" / "fem_highres"
PINN_METRICS = REPO_ROOT / "runs" / "attempt20c_compact" / "rabbit_poisson_schwarz_attempt20c_compact_metrics.json"

FIG_PNG = OUT_DIR / "example2_rabbit_poisson_fem_refinement_pub.png"
FIG_PDF = OUT_DIR / "example2_rabbit_poisson_fem_refinement_pub.pdf"

RUNS = [
    (8, RUN_DIR / "rabbit_poisson_schwarz_fem_sweep_n8_metrics.json"),
    (16, RUN_DIR / "rabbit_poisson_schwarz_fem_sweep_n16_metrics.json"),
    (24, RUN_DIR / "rabbit_poisson_schwarz_fem_sweep_n24_metrics.json"),
    (32, RUN_DIR / "rabbit_poisson_schwarz_fem_sweep_n32_metrics.json"),
    (48, HIGHRES_DIR / "rabbit_poisson_schwarz_fem_highres_n48_metrics.json"),
    (56, HIGHRES_DIR / "rabbit_poisson_schwarz_fem_highres_n56_metrics.json"),
]


def load_series() -> dict[str, np.ndarray]:
    n_cells = []
    h = []
    rel_l2 = []
    max_error = []
    dofs = []
    runtime = []
    for n, path in RUNS:
        data = json.loads(path.read_text())
        n_cells.append(n)
        h.append(float(data["h"]))
        rel_l2.append(100.0 * float(data["relative_l2_error"]))
        max_error.append(100.0 * float(data["max_error"]))
        dofs.append(int(data["total_dofs"]))
        runtime.append(float(data["runtime_seconds"]))
    return {
        "n_cells": np.asarray(n_cells, dtype=int),
        "h": np.asarray(h, dtype=float),
        "rel_l2_percent": np.asarray(rel_l2, dtype=float),
        "rel_l2": np.asarray(rel_l2, dtype=float) / 100.0,
        "max_error_percent": np.asarray(max_error, dtype=float),
        "max_error": np.asarray(max_error, dtype=float) / 100.0,
        "dofs": np.asarray(dofs, dtype=int),
        "runtime": np.asarray(runtime, dtype=float),
    }


def build_refinement_figure(series: dict[str, np.ndarray]) -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("deep", 3)
    h = series["h"]
    rel_l2 = series["rel_l2"]
    n_cells = series["n_cells"]
    pinn_rel_l2 = float(json.loads(PINN_METRICS.read_text())["global"]["relative_l2_error"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH * 0.66, 3.1))
    ax.loglog(h, rel_l2, marker="o", color=palette[0], lw=1.8, label=r"atlas-FEM relative $L^2$")

    h_ref = np.array([h[0], h[-1]])
    oh2_ref = rel_l2[-1] * (h_ref / h[-1]) ** 2
    ax.loglog(h_ref, oh2_ref, ls="--", color=palette[1], lw=1.4, label=r"$O(h^2)$ reference")
    ax.axhline(
        pinn_rel_l2,
        color=palette[2],
        ls=":",
        lw=1.5,
        label=rf"compact PINN ({pinn_rel_l2:.2e})",
    )

    ax.invert_xaxis()
    ax.set_xlabel(r"Chart FEM mesh size $h$")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_title("Rabbit atlas-FEM refinement", pad=4)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.0)

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(h)
    top.set_xticklabels([str(n) for n in n_cells])
    top.set_xlabel(r"Structured-grid resolution $n_{\mathrm{cells}}$")

    for x, y, n in zip(h, rel_l2, n_cells):
        ax.annotate(
            f"n={n}",
            xy=(x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=palette[0],
        )

    slope = np.log(rel_l2[-2] / rel_l2[-1]) / np.log(h[-2] / h[-1])
    ax.text(
        0.05,
        0.12,
        rf"fine-level trend: $O(h^{{{slope:.2f}}})$",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
    )

    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.20, top=0.83)
    save_figure(fig, FIG_PNG, FIG_PDF)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Generate publication-ready rabbit atlas-FEM refinement figure.")
    parser.parse_args()
    build_refinement_figure(load_series())


if __name__ == "__main__":
    main()
