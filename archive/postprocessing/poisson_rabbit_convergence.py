#!/usr/bin/env python3
"""
Publication-quality convergence figures for the Poisson-on-rabbit benchmark.

Produces a multi-panel figure comparing:
  (a) Relative L2 error vs Schwarz iteration  (W4 dense MLP vs CompactChartNet)
  (b) Global PDE residual vs iteration
  (c) Interface value residual vs iteration
  (d) Interface flux residual vs iteration

Optionally marks the best-checkpoint iteration for each run.

Output: <output-dir>/<prefix>_convergence.pdf  (300 dpi PNG also saved)

Usage (with default run paths):
    python postprocessing/poisson_rabbit_convergence.py

Custom paths:
    python postprocessing/poisson_rabbit_convergence.py \\
        --run-dirs runs/attempt19_w4 runs/attempt20c_compact \\
        --prefixes rabbit_poisson_schwarz_attempt19_w4 \\
                   rabbit_poisson_schwarz_attempt20c_compact \\
        --labels "Dense MLP (W4)" "CompactChartNet" \\
        --output-dir figures \\
        --output-prefix poisson_rabbit_convergence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    DOUBLE_COL_W,
    GOLDEN,
    PUB_COLORS,
    PUB_LINESTYLES,
    PUB_MARKERS,
    set_pub_style,
)

# ---- Default run configurations -------------------------------------------
DEFAULT_RUNS = [
    {
        "run_dir": "runs/attempt19_w4",
        "prefix":  "rabbit_poisson_schwarz_attempt19_w4",
        "label":   "Dense MLP (W4)",
    },
    {
        "run_dir": "runs/attempt20c_compact",
        "prefix":  "rabbit_poisson_schwarz_attempt20c_compact",
        "label":   "CompactChartNet",
    },
]

# Metrics to plot and their axis labels
PANELS = [
    ("rel_l2_eval",     "Relative $L_2$ error",         "linear"),
    ("global_residual", "Global PDE residual",           "log"),
    ("interface_value", "Interface value residual",      "log"),
    ("interface_flux",  "Interface flux residual",       "log"),
]


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Schwarz convergence history for the rabbit Poisson benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        default=[r["run_dir"] for r in DEFAULT_RUNS],
        help="List of run directories (one per method)",
    )
    p.add_argument(
        "--prefixes",
        nargs="+",
        default=[r["prefix"] for r in DEFAULT_RUNS],
        help="Filename prefixes for history / metrics JSON files",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=[r["label"] for r in DEFAULT_RUNS],
        help="Legend labels for each run",
    )
    p.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for output figure files",
    )
    p.add_argument(
        "--output-prefix",
        default="poisson_rabbit_convergence",
        help="Filename prefix for output figures",
    )
    p.add_argument(
        "--mark-best",
        action="store_true",
        default=True,
        help="Mark the best-checkpoint iteration with a vertical dashed line",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_history(run_dir: str, prefix: str) -> Dict:
    hist_path = os.path.join(run_dir, f"{prefix}_history.json")
    if not os.path.isfile(hist_path):
        print(f"  WARNING: {hist_path} not found — skipping.")
        return {}
    with open(hist_path) as fh:
        return json.load(fh)


def load_metrics(run_dir: str, prefix: str) -> Dict:
    metr_path = os.path.join(run_dir, f"{prefix}_metrics.json")
    if not os.path.isfile(metr_path):
        return {}
    with open(metr_path) as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_pub_style(fontsize=9, linewidth=1.5)

    # ---- Load histories ---------------------------------------------------
    runs: List[Dict] = []
    for rd, pref, lbl in zip(args.run_dirs, args.prefixes, args.labels):
        hist = load_history(rd, pref)
        metr = load_metrics(rd, pref)
        if not hist:
            continue
        runs.append({"label": lbl, "history": hist, "metrics": metr})

    if not runs:
        print("No history data found — exiting.")
        return

    n_panels = len(PANELS)
    fig_w = DOUBLE_COL_W
    fig_h = fig_w * GOLDEN * 1.2
    fig, axes = plt.subplots(
        2, 2,
        figsize=(fig_w, fig_h),
        sharex=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    panel_labels = [r"(\textit{a})", r"(\textit{b})", r"(\textit{c})", r"(\textit{d})"]
    # Fall back to plain text if LaTeX off
    panel_labels_plain = ["(a)", "(b)", "(c)", "(d)"]

    for pi, (key, ylabel, yscale) in enumerate(PANELS):
        ax = axes[pi]
        any_plotted = False

        for ri, run in enumerate(runs):
            hist = run["history"]
            if key not in hist:
                continue
            vals = np.asarray(hist[key], dtype=float)
            iters = np.arange(1, len(vals) + 1)

            # Convert rel_l2 to percentage for display
            if key == "rel_l2_eval":
                vals = vals * 100.0

            ax.plot(
                iters,
                vals,
                color=PUB_COLORS[ri % len(PUB_COLORS)],
                linestyle=PUB_LINESTYLES[ri % len(PUB_LINESTYLES)],
                marker=PUB_MARKERS[ri % len(PUB_MARKERS)],
                markevery=max(1, len(vals) // 8),
                markersize=4,
                label=run["label"],
            )
            any_plotted = True

            # Mark best iteration
            if args.mark_best and key == "rel_l2_eval":
                idx_best = int(np.argmin(vals))
                ax.axvline(
                    idx_best + 1,
                    color=PUB_COLORS[ri % len(PUB_COLORS)],
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.7,
                )
                ax.annotate(
                    f"{vals[idx_best]:.2f}%",
                    xy=(idx_best + 1, vals[idx_best]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=PUB_COLORS[ri % len(PUB_COLORS)],
                )

        if not any_plotted:
            ax.set_visible(False)
            continue

        ax.set_xlabel("Schwarz iteration")
        if key == "rel_l2_eval":
            ax.set_ylabel("Relative $L_2$ error (%)")
            ax.set_yscale("linear")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        else:
            ax.set_ylabel(ylabel)
            ax.set_yscale(yscale)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.legend(loc="upper right", handlelength=2)

        # Panel label
        plbl = panel_labels_plain[pi]
        ax.text(
            0.02, 0.97,
            plbl,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9, fontweight="bold",
        )

    # ---- Title + save -----------------------------------------------------
    fig.suptitle(
        "Schwarz PINN convergence — forward Poisson on Stanford rabbit",
        fontsize=10, fontweight="bold",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(args.output_dir, f"{args.output_prefix}.{ext}")
        fig.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
