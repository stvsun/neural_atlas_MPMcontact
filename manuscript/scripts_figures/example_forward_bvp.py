#!/usr/bin/env python3
"""Generate publication figures for the 8-chart Schwarz forward elastoplastic BVP.

Reads experiment data from runs/torus_forward_bvp_experiments/ and creates:
  - forward_bvp_diagnostics.pdf: 2x2 convergence/diagnostics figure
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "runs" / "torus_forward_bvp_experiments"
OUT_DIR = REPO / "manuscript" / "figures_cmame_core" / "example_forward_bvp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def build_diagnostics_figure():
    """2x2 figure: (a) ep_bar vs step for 3 mesh sizes, (b) interface jump vs step,
    (c) load-step sensitivity, (d) timing comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=200)

    # (a) Accumulated plastic strain vs load step for 3 mesh sizes
    ax = axes[0, 0]
    colors = {"4": "#1f77b4", "6": "#ff7f0e", "8": "#2ca02c"}
    for nc in [4, 6, 8]:
        h = load_history(f"history_ncells{nc}.json")
        steps = np.arange(1, len(h["max_ep_bar"]) + 1)
        ax.plot(steps, h["max_ep_bar"], color=colors[str(nc)],
                linewidth=1.5, label=f"$n_{{\\mathrm{{cells}}}}={nc}$")
    ax.set_xlabel("Load step", fontsize=11)
    ax.set_ylabel(r"$\max\,\bar{\varepsilon}_p$", fontsize=11)
    ax.set_title("(a) Accumulated plastic strain", fontsize=12, pad=6)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (b) Interface jump vs load step
    ax = axes[0, 1]
    for nc in [4, 6, 8]:
        h = load_history(f"history_ncells{nc}.json")
        steps = np.arange(1, len(h["interface_jump"]) + 1)
        ax.semilogy(steps, h["interface_jump"], color=colors[str(nc)],
                    linewidth=1.5, label=f"$n_{{\\mathrm{{cells}}}}={nc}$")
    ax.set_xlabel("Load step", fontsize=11)
    ax.set_ylabel("Interface displacement jump", fontsize=11)
    ax.set_title("(b) Schwarz interface continuity", fontsize=12, pad=6)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # (c) Load-step sensitivity: final ep_bar vs steps_per_half
    ax = axes[1, 0]
    with open(DATA_DIR / "timestep_sensitivity.json") as f:
        ts = json.load(f)
    sphs = [10, 20, 40]
    ep_bars = [ts[str(s)]["final_max_ep_bar"] for s in sphs]
    ax.plot(sphs, ep_bars, "ko-", markersize=8, linewidth=1.5)
    ax.set_xlabel("Steps per half-cycle", fontsize=11)
    ax.set_ylabel(r"Final $\max\,\bar{\varepsilon}_p$", fontsize=11)
    ax.set_title("(c) Load-step convergence", fontsize=12, pad=6)
    ax.set_xticks(sphs)
    ax.grid(True, alpha=0.3)
    # Add percentage annotation
    spread = (max(ep_bars) - min(ep_bars)) / np.mean(ep_bars) * 100
    ax.text(0.95, 0.05, f"spread: {spread:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    # (d) Timing comparison (mesh refinement)
    ax = axes[1, 1]
    with open(DATA_DIR / "mesh_refinement.json") as f:
        mr = json.load(f)
    ncs = [4, 6, 8]
    dofs = [mr[str(nc)]["total_dof"] for nc in ncs]
    times = [mr[str(nc)]["avg_time_per_step"] for nc in ncs]
    ax.loglog(dofs, times, "s-", color="#d62728", markersize=8, linewidth=1.5)
    for nc, d, t in zip(ncs, dofs, times):
        ax.annotate(f"$n={nc}$", (d, t), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Total DOF (8 charts)", fontsize=11)
    ax.set_ylabel("Wall time per step (s)", fontsize=11)
    ax.set_title("(d) Computational cost scaling", fontsize=12, pad=6)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"forward_bvp_diagnostics.{ext}",
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved forward_bvp_diagnostics.{{png,pdf}}")


def build_loading_history_figure():
    """Single-panel figure: applied delta and max|u| vs step for n_cells=6."""
    h = load_history("history_ncells6.json")
    steps = np.arange(1, len(h["delta"]) + 1)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3.5), dpi=200)
    color1, color2 = "#1f77b4", "#d62728"
    ax1.plot(steps, np.array(h["delta"]) * 1000, color=color1, linewidth=1.5,
             label=r"$\delta$ (prescribed)")
    ax1.set_xlabel("Load step", fontsize=11)
    ax1.set_ylabel(r"$\delta$ ($\times 10^{-3}$)", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(steps, np.array(h["max_ep_bar"]), color=color2, linewidth=1.5,
             label=r"$\max\,\bar{\varepsilon}_p$")
    ax2.set_ylabel(r"$\max\,\bar{\varepsilon}_p$", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Loading history and plastic strain accumulation ($n_{\\mathrm{cells}}=6$)",
                   fontsize=12, pad=6)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"forward_bvp_loading.{ext}",
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved forward_bvp_loading.{{png,pdf}}")


if __name__ == "__main__":
    build_diagnostics_figure()
    build_loading_history_figure()
    print("Done.")
