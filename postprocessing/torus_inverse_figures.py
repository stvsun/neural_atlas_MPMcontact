#!/usr/bin/env python3
"""
Publication-quality figures for the inverse neo-Hookean problem on a torus.

Three solver variants are compared:
  1. Atlas direct PINN   (torus_inverse_mps)
  2. Schwarz dual — displacement BC  (torus_schwarz_dual_accurate_displacement_*)
  3. Schwarz dual — traction  BC  (torus_schwarz_dual_accurate_traction_*)

Figure panels produced
──────────────────────
  Fig. 1  (2×2):  Convergence histories
    (a)  μ estimate vs epoch / iteration
    (b)  K estimate vs epoch / iteration
    (c)  Observation relative L2 vs epoch / iteration
    (d)  Training loss vs epoch / iteration

  Fig. 2  (1×3):  Parameter-recovery summary bar chart
    Left bar group   : μ error (%)
    Centre bar group : K error (%)
    Right bar group  : Observation rel-L2 (%)

  Fig. 3  (1×2):  Uncertainty band plots for the Schwarz displacement run
    (a)  μ_mean ± μ_std vs epoch (shows chart-to-chart consistency)
    (b)  K_mean ± K_std vs epoch

Output directory:  figures/
Filenames:
    torus_inverse_convergence.pdf / .png
    torus_inverse_summary.pdf     / .png
    torus_inverse_uncertainty.pdf / .png

Usage:
    python postprocessing/torus_inverse_figures.py

    # Override default run directories:
    python postprocessing/torus_inverse_figures.py \\
        --atlas-dir   runs/torus_inverse_mps \\
        --disp-dir    runs/torus_schwarz_dual_accurate_displacement_20260215_140200 \\
        --tract-dir   runs/torus_schwarz_dual_accurate_traction_20260215_140800 \\
        --output-dir  figures
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

# Ground-truth material parameters
MU_TRUE = 1.8
K_TRUE  = 25.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce publication figures for the torus inverse neo-Hookean problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--atlas-dir",
        default="runs/torus_inverse_mps",
        help="Run directory for atlas-direct solve",
    )
    p.add_argument(
        "--disp-dir",
        default="runs/torus_schwarz_dual_accurate_displacement_20260215_140200",
        help="Run directory for Schwarz displacement-BC solve",
    )
    p.add_argument(
        "--tract-dir",
        default="runs/torus_schwarz_dual_accurate_traction_20260215_140800",
        help="Run directory for Schwarz traction-BC solve",
    )
    p.add_argument("--output-dir", default="figures", help="Output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(run_dir: str) -> Tuple[Dict, Dict]:
    """Return (history_dict, metrics_dict) for a run directory."""
    if not os.path.isdir(run_dir):
        return {}, {}
    # Find the first *_history.json and *_metrics.json
    hist, metr = {}, {}
    for fn in os.listdir(run_dir):
        path = os.path.join(run_dir, fn)
        if fn.endswith("_history.json"):
            with open(path) as fh:
                hist = json.load(fh)
        elif fn.endswith("_metrics.json"):
            with open(path) as fh:
                metr = json.load(fh)
    return hist, metr


def _param_err_pct(metrics: Dict, key: str, true_val: float) -> Optional[float]:
    """Extract a parameter's final relative error (%) from metrics.

    Tries pre-computed *_rel_error_percent keys first (most accurate), then
    falls back to estimating from the stored parameter value.
    """
    # 1. Pre-computed percentage error stored directly
    for pct_key in (f"{key}_rel_error_percent", f"{key}_rel_error_pct"):
        if pct_key in metrics:
            return float(metrics[pct_key])

    # 2. Estimate from stored parameter value (atlas: mu_final, Schwarz: mu_mean_final)
    for candidate in (f"{key}_final", f"{key}_mean_final", f"{key}_mean", key):
        if candidate in metrics:
            est = float(metrics[candidate])
            return abs(est - true_val) / abs(true_val) * 100.0
    return None


# ---------------------------------------------------------------------------
# Figure 1: convergence histories
# ---------------------------------------------------------------------------

def fig_convergence(
    runs: List[Dict],
    output_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_pub_style(fontsize=9, linewidth=1.5)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN * 1.1),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # Panel configuration: (key_in_history, ylabel, yscale, true_value_hline)
    panel_cfg = [
        ("mu_mean",    r"Shear modulus $\mu$ estimate",   "linear", MU_TRUE),
        ("K_mean",     r"Bulk modulus $K$ estimate",       "linear", K_TRUE),
        ("obs_rel_l2", r"Observation rel-$L_2$ error",    "log",    None),
        ("loss",       r"Total training loss",             "log",    None),
    ]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for pi, (key, ylabel, yscale, hline) in enumerate(panel_cfg):
        ax = axes[pi]

        for ri, run in enumerate(runs):
            hist = run["history"]
            if key not in hist:
                # Try mu_guess for atlas run
                alt = "mu_guess" if key == "mu_mean" else ("K_guess" if key == "K_mean" else None)
                if alt and alt in hist:
                    key_used = alt
                else:
                    continue
            else:
                key_used = key

            vals = np.asarray(hist[key_used], dtype=float)
            epochs = np.arange(1, len(vals) + 1)
            ax.plot(
                epochs, vals,
                color=PUB_COLORS[ri % len(PUB_COLORS)],
                linestyle=PUB_LINESTYLES[ri % len(PUB_LINESTYLES)],
                marker=PUB_MARKERS[ri % len(PUB_MARKERS)],
                markevery=max(1, len(vals) // 8),
                markersize=4,
                label=run["label"],
                zorder=3,
            )

        # Ground-truth horizontal reference
        if hline is not None:
            ax.axhline(hline, color="black", linestyle="--", linewidth=0.9,
                       alpha=0.5, label=f"True = {hline:.1f}", zorder=2)

        ax.set_xlabel("Epoch / iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.legend(loc="best", fontsize=7)
        ax.text(
            0.02, 0.97, panel_labels[pi],
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "Inverse neo-Hookean on torus — training convergence",
        fontsize=10, fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"torus_inverse_convergence.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: parameter-recovery summary bar chart
# ---------------------------------------------------------------------------

def fig_summary(
    runs: List[Dict],
    output_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_pub_style(fontsize=9, linewidth=1.2)

    n_runs = len(runs)

    # Collect final errors
    mu_errs:  List[Optional[float]] = []
    K_errs:   List[Optional[float]] = []
    obs_rl2:  List[Optional[float]] = []

    for run in runs:
        metr = run["metrics"]
        mu_errs.append(_param_err_pct(metr, "mu", MU_TRUE))
        K_errs.append(_param_err_pct(metr, "K",  K_TRUE))
        # obs_rel_l2 — key name varies by solver variant
        for cand in ("obs_rel_l2_final", "obs_rel_l2", "traction_rel_l2",
                     "traction_rel_l2_final", "displacement_rel_l2"):
            if cand in metr:
                obs_rl2.append(float(metr[cand]) * 100.0)
                break
        else:
            obs_rl2.append(None)

    labels = [r["label"] for r in runs]
    x = np.arange(n_runs)
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_W * 0.75, DOUBLE_COL_W * 0.75 * GOLDEN * 1.2),
        constrained_layout=True,
    )

    def _vals(lst: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else 0.0 for v in lst])

    bar1 = ax.bar(x - width, _vals(mu_errs),  width, label=r"$\mu$ error (%)",
                  color=PUB_COLORS[0], edgecolor="white", linewidth=0.5)
    bar2 = ax.bar(x,          _vals(K_errs),   width, label=r"$K$ error (%)",
                  color=PUB_COLORS[1], edgecolor="white", linewidth=0.5)
    bar3 = ax.bar(x + width,  _vals(obs_rl2),  width, label=r"Obs. rel-$L_2$ (%)",
                  color=PUB_COLORS[2], edgecolor="white", linewidth=0.5)

    # Value annotations on top of bars
    for bars in (bar1, bar2, bar3):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3g}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7,
                )

    ax.set_yscale("log")
    ax.set_ylabel("Error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Inverse neo-Hookean on torus — final parameter errors", fontsize=9, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"torus_inverse_summary.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: μ and K uncertainty bands (Schwarz runs only)
# ---------------------------------------------------------------------------

def fig_uncertainty(
    runs: List[Dict],
    output_dir: str,
) -> None:
    """Plot mean ± std bands for μ and K for runs that track _std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style(fontsize=9, linewidth=1.5)

    # Runs with std data
    std_runs = [
        r for r in runs
        if "mu_std" in r["history"] and "K_std" in r["history"]
    ]
    if not std_runs:
        print("No runs with mu_std / K_std — skipping uncertainty figure.")
        return

    n_panels = 2
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(DOUBLE_COL_W, DOUBLE_COL_W * GOLDEN),
        constrained_layout=True,
    )

    param_cfg = [
        ("mu_mean", "mu_std", r"Shear modulus $\mu$",  MU_TRUE, "(a)"),
        ("K_mean",  "K_std",  r"Bulk modulus $K$",      K_TRUE,  "(b)"),
    ]

    for pi, (mean_key, std_key, ylabel, true_val, plbl) in enumerate(param_cfg):
        ax = axes[pi]
        for ri, run in enumerate(std_runs):
            hist = run["history"]
            if mean_key not in hist:
                continue
            mu = np.asarray(hist[mean_key], dtype=float)
            sd = np.asarray(hist[std_key],  dtype=float)
            ep = np.arange(1, len(mu) + 1)
            color = PUB_COLORS[ri % len(PUB_COLORS)]
            ax.plot(ep, mu, color=color,
                    linestyle=PUB_LINESTYLES[ri % len(PUB_LINESTYLES)],
                    label=run["label"])
            ax.fill_between(ep, mu - sd, mu + sd,
                            color=color, alpha=0.20)

        ax.axhline(true_val, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.5, label=f"True = {true_val:.1f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=7)
        ax.text(0.02, 0.97, plbl, transform=ax.transAxes,
                va="top", ha="left", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Chart-to-chart consistency: mean ± std across Schwarz charts",
        fontsize=9, fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"torus_inverse_uncertainty.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_cfgs = [
        {"run_dir": args.atlas_dir, "label": "Atlas direct"},
        {"run_dir": args.disp_dir,  "label": "Schwarz (disp. BC)"},
        {"run_dir": args.tract_dir, "label": "Schwarz (traction BC)"},
    ]

    runs: List[Dict] = []
    for cfg in run_cfgs:
        hist, metr = _load(cfg["run_dir"])
        if hist:
            runs.append({"label": cfg["label"], "history": hist, "metrics": metr})
        else:
            print(f"  WARNING: No history found in {cfg['run_dir']}")

    if not runs:
        print("No data loaded — exiting.")
        return

    print(f"Loaded {len(runs)} run(s): {[r['label'] for r in runs]}")

    fig_convergence(runs, args.output_dir)
    fig_summary(runs, args.output_dir)
    fig_uncertainty(runs, args.output_dir)

    print("\nAll torus inverse figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
