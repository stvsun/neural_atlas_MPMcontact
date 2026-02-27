#!/usr/bin/env python3
"""
Combined multi-example convergence figure for all successful numerical examples.

Produces a single journal-ready figure with four sub-panels, one per problem:
  (a) Poisson on rabbit      — relative L2 error vs Schwarz iteration
                               (Dense MLP W4 vs CompactChartNet side-by-side)
  (b) Elder inverse on rabbit — permeability k0 relative error vs iteration
  (c) Torus inverse — atlas  — μ estimate vs epoch
  (d) Torus inverse — Schwarz — μ mean ± std and K mean ± std vs epoch

A second figure gives a benchmark bar-chart summary:

  Fig. 2  (1 row, grouped bars):
    Each group = one numerical example; each bar = a key metric.

Outputs (in --output-dir):
    all_convergence.pdf / .png      (4-panel convergence)
    all_benchmark_summary.pdf/.png  (grouped bar chart)

Usage:
    # Auto-discovers all default run dirs:
    python postprocessing/plot_all_convergence.py

    # Override:
    python postprocessing/plot_all_convergence.py \\
        --output-dir figures \\
        --poisson-dirs  runs/attempt19_w4  runs/attempt20c_compact \\
        --poisson-prefixes  rabbit_poisson_schwarz_attempt19_w4 \\
                            rabbit_poisson_schwarz_attempt20c_compact \\
        --poisson-labels "Dense MLP (W4)" "CompactChartNet"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    DOUBLE_COL_W,
    GOLDEN,
    PUB_COLORS,
    PUB_LINESTYLES,
    PUB_MARKERS,
    SINGLE_COL_W,
    set_pub_style,
)


# ---------------------------------------------------------------------------
# Default run configurations
# ---------------------------------------------------------------------------

POISSON_DEFAULTS = [
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

ELDER_DEFAULT = {
    "run_dir": "runs/rabbit_inverse_elder_globalfield_small",
    # history prefix auto-detected
}

TORUS_DEFAULTS = {
    "atlas": "runs/torus_inverse_mps",
    "disp":  "runs/torus_schwarz_dual_accurate_displacement_20260215_140200",
    "tract": "runs/torus_schwarz_dual_accurate_traction_20260215_140800",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hist(run_dir: str, prefix: Optional[str] = None) -> Tuple[Dict, str]:
    """Return (history_dict, label) for a run directory."""
    if not os.path.isdir(run_dir):
        return {}, ""
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith("_history.json"):
            if prefix and not fn.startswith(prefix):
                continue
            with open(os.path.join(run_dir, fn)) as fh:
                return json.load(fh), fn.replace("_history.json", "")
    return {}, ""


def _load_metrics(run_dir: str, prefix: Optional[str] = None) -> Dict:
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith("_metrics.json"):
            if prefix and not fn.startswith(prefix):
                continue
            with open(os.path.join(run_dir, fn)) as fh:
                return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined convergence figure for all numerical examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="figures")
    p.add_argument(
        "--poisson-dirs",
        nargs="+",
        default=[r["run_dir"] for r in POISSON_DEFAULTS],
    )
    p.add_argument(
        "--poisson-prefixes",
        nargs="+",
        default=[r["prefix"] for r in POISSON_DEFAULTS],
    )
    p.add_argument(
        "--poisson-labels",
        nargs="+",
        default=[r["label"] for r in POISSON_DEFAULTS],
    )
    p.add_argument("--elder-dir",  default=ELDER_DEFAULT["run_dir"])
    p.add_argument("--torus-atlas-dir",  default=TORUS_DEFAULTS["atlas"])
    p.add_argument("--torus-disp-dir",   default=TORUS_DEFAULTS["disp"])
    p.add_argument("--torus-tract-dir",  default=TORUS_DEFAULTS["tract"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Figure 1: 4-panel convergence overview
# ---------------------------------------------------------------------------

def fig_all_convergence(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_pub_style(fontsize=9, linewidth=1.5)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 1.15),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # ---- Panel (a): Poisson rabbit rel_l2 --------------------------------
    ax = axes[0]
    for ri, (rd, pref, lbl) in enumerate(zip(
        args.poisson_dirs, args.poisson_prefixes, args.poisson_labels
    )):
        hist, _ = _load_hist(rd, pref)
        if "rel_l2_eval" not in hist:
            continue
        vals = np.asarray(hist["rel_l2_eval"], dtype=float) * 100.0
        iters = np.arange(1, len(vals) + 1)
        ax.plot(
            iters, vals,
            color=PUB_COLORS[ri % len(PUB_COLORS)],
            linestyle=PUB_LINESTYLES[ri % len(PUB_LINESTYLES)],
            marker=PUB_MARKERS[ri % len(PUB_MARKERS)],
            markevery=max(1, len(vals) // 8),
            markersize=4,
            label=lbl,
        )
        # annotate best
        best_i = int(np.argmin(vals))
        ax.annotate(
            f"{vals[best_i]:.2f}%",
            xy=(best_i + 1, vals[best_i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=PUB_COLORS[ri % len(PUB_COLORS)],
        )
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.set_yscale("linear")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("Poisson — rabbit", fontsize=8)

    # ---- Panel (b): Elder rabbit k0 error vs iteration -------------------
    ax = axes[1]
    hist_elder, _ = _load_hist(args.elder_dir)
    if "k0_rel" in hist_elder:
        vals = np.asarray(hist_elder["k0_rel"], dtype=float) * 100.0
        # Deduplicate (Elder history repeats each iter 3 times due to update scheme)
        # Keep unique consecutive values
        keep = [0] + [i for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
        vals_u = vals[keep]
        iters_u = np.arange(1, len(vals_u) + 1)
        ax.plot(iters_u, vals_u,
                color=PUB_COLORS[0], linewidth=1.5,
                marker="o", markersize=4, label="$k_0$ rel. error")
    if "eig_rel_mean" in hist_elder:
        vals = np.asarray(hist_elder["eig_rel_mean"], dtype=float) * 100.0
        keep = [0] + [i for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
        vals_u = vals[keep]
        iters_u = np.arange(1, len(vals_u) + 1)
        ax.plot(iters_u, vals_u,
                color=PUB_COLORS[1], linestyle="--", linewidth=1.5,
                marker="s", markersize=4, label="Eigenvalue rel. error (mean)")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Parameter relative error (%)")
    ax.set_yscale("linear")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("Elder inverse — rabbit", fontsize=8)

    # ---- Panel (c): Torus atlas — μ and K convergence --------------------
    ax = axes[2]
    hist_atlas, _ = _load_hist(args.torus_atlas_dir)
    mu_true, K_true = 1.8, 25.0
    if "mu_guess" in hist_atlas:
        mu_vals = np.asarray(hist_atlas["mu_guess"], dtype=float)
        ep = np.arange(1, len(mu_vals) + 1)
        mu_err = np.abs(mu_vals - mu_true) / mu_true * 100.0
        ax.plot(ep, mu_err,
                color=PUB_COLORS[0], linewidth=1.5,
                marker="o", markevery=max(1, len(ep) // 8), markersize=4,
                label=r"$|\mu - \mu^\star| / \mu^\star$ (%)")
    if "K_guess" in hist_atlas:
        K_vals = np.asarray(hist_atlas["K_guess"], dtype=float)
        ep = np.arange(1, len(K_vals) + 1)
        K_err = np.abs(K_vals - K_true) / K_true * 100.0
        ax.plot(ep, K_err,
                color=PUB_COLORS[1], linestyle="--", linewidth=1.5,
                marker="s", markevery=max(1, len(ep) // 8), markersize=4,
                label=r"$|K - K^\star| / K^\star$ (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter relative error (%)")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("Neo-Hookean inverse — torus (atlas)", fontsize=8)

    # ---- Panel (d): Torus Schwarz — μ and K mean ± std ------------------
    ax = axes[3]
    for ri, (rd, lbl) in enumerate([
        (args.torus_disp_dir,  "Schwarz (disp.)"),
        (args.torus_tract_dir, "Schwarz (tract.)"),
    ]):
        hist_s, _ = _load_hist(rd)
        if "mu_mean" not in hist_s:
            continue
        mu = np.asarray(hist_s["mu_mean"], dtype=float)
        mu_err = np.abs(mu - mu_true) / mu_true * 100.0
        ep = np.arange(1, len(mu) + 1)
        color = PUB_COLORS[ri % len(PUB_COLORS)]
        ls    = PUB_LINESTYLES[ri % len(PUB_LINESTYLES)]
        ax.plot(ep, mu_err, color=color, linestyle=ls, linewidth=1.5,
                marker=PUB_MARKERS[ri], markevery=max(1, len(ep) // 8),
                markersize=4, label=lbl + r" ($\mu$)")
        if "mu_std" in hist_s:
            sd = np.asarray(hist_s["mu_std"], dtype=float) / mu_true * 100.0
            ax.fill_between(ep,
                            np.maximum(mu_err - sd, 1e-15),
                            mu_err + sd,
                            color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$|\mu - \mu^\star| / \mu^\star$ (%)")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.02, 0.97, "(d)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("Neo-Hookean inverse — torus (Schwarz)", fontsize=8)

    # ---- Save -------------------------------------------------------------
    fig.suptitle(
        "Convergence histories — all numerical examples",
        fontsize=10, fontweight="bold",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(args.output_dir, f"all_convergence.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: grouped benchmark bar chart
# ---------------------------------------------------------------------------

def fig_benchmark_summary(args: argparse.Namespace) -> None:
    """Bar-chart summary of final errors across all examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style(fontsize=9, linewidth=1.2)

    # Collect final metrics ------------------------------------------------
    groups = []

    # Poisson rabbit (best run = attempt20c_compact)
    metr = _load_metrics(
        args.poisson_dirs[-1],  # last entry = best
        args.poisson_prefixes[-1],
    )
    if metr:
        rel_l2 = float(metr.get("relative_l2_error", 0.0)) * 100.0
        max_err = float(metr.get("max_error", 0.0)) * 100.0
        groups.append({
            "label":  "Poisson\nrabbit\n(best)",
            "bars": [
                ("Rel-$L_2$ (%)", rel_l2,  PUB_COLORS[0]),
                ("Max err (%)",   max_err,  PUB_COLORS[1]),
            ],
        })

    # Torus inverse — atlas
    metr_atlas = _load_metrics(args.torus_atlas_dir)
    if metr_atlas:
        mu_err = abs(float(metr_atlas.get("mu_final", 1.8)) - 1.8) / 1.8 * 100.0
        K_err  = abs(float(metr_atlas.get("K_final",  25.0)) - 25.0) / 25.0 * 100.0
        groups.append({
            "label": "Neo-Hookean\ntorus\n(atlas)",
            "bars": [
                (r"$\mu$ err (%)", mu_err, PUB_COLORS[0]),
                (r"$K$ err (%)",   K_err,  PUB_COLORS[1]),
            ],
        })

    # Torus inverse — Schwarz displacement (best)
    metr_disp = _load_metrics(args.torus_disp_dir)
    if metr_disp:
        mu_err = abs(float(metr_disp.get("mu_mean", 1.8)) - 1.8) / 1.8 * 100.0
        K_err  = abs(float(metr_disp.get("K_mean",  25.0)) - 25.0) / 25.0 * 100.0
        groups.append({
            "label": "Neo-Hookean\ntorus\n(Schwarz disp.)",
            "bars": [
                (r"$\mu$ err (%)", mu_err, PUB_COLORS[0]),
                (r"$K$ err (%)",   K_err,  PUB_COLORS[1]),
            ],
        })

    # Elder rabbit
    metr_elder = _load_metrics(args.elder_dir)
    if metr_elder:
        k0_err  = float(metr_elder.get("k0_rel_err", 0.0)) * 100.0
        eig_err = float(metr_elder.get("eig_rel_mean", 0.0)) * 100.0
        ax_deg  = float(metr_elder.get("axis_deg_mean", 0.0))
        groups.append({
            "label": "Elder inverse\nrabbit",
            "bars": [
                (r"$k_0$ err (%)",      k0_err,  PUB_COLORS[0]),
                ("Eig. err (%,mean)",   eig_err, PUB_COLORS[1]),
                ("Axis angle (°,mean)", ax_deg,  PUB_COLORS[2]),
            ],
        })

    if not groups:
        print("No metrics found for summary figure.")
        return

    # Determine max bars per group
    max_bars = max(len(g["bars"]) for g in groups)
    n_groups = len(groups)

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 0.9),
        constrained_layout=True,
    )

    x = np.arange(n_groups)
    total_width = 0.7
    bar_w = total_width / max_bars

    for bi in range(max_bars):
        xs, ys, cols, lbls = [], [], [], []
        for gi, g in enumerate(groups):
            if bi < len(g["bars"]):
                lbl, val, col = g["bars"][bi]
                xs.append(gi + (bi - (max_bars - 1) / 2) * bar_w)
                ys.append(val)
                cols.append(col)
                lbls.append(lbl)
        if xs:
            bars = ax.bar(xs, ys, bar_w * 0.9,
                          color=cols, edgecolor="white", linewidth=0.5,
                          label=lbls[0])
            for rect, val in zip(bars, ys):
                ax.annotate(
                    f"{val:.3g}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    # Build legend from first group's bar labels
    from matplotlib.patches import Patch
    legend_items = []
    if groups:
        for _, lbl_val_col in enumerate(groups[0]["bars"]):
            legend_items.append(Patch(color=lbl_val_col[2], label=lbl_val_col[0]))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8)
    ax.set_ylabel("Error (% or degrees)")
    ax.set_yscale("log")
    ax.set_title("Benchmark summary — all numerical examples",
                 fontsize=9, fontweight="bold")

    os.makedirs(args.output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(args.output_dir, f"all_benchmark_summary.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print("Generating combined convergence figure ...")
    fig_all_convergence(args)
    print("\nGenerating benchmark summary figure ...")
    fig_benchmark_summary(args)
    print("\nAll figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
