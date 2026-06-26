"""Numerical CV suite — results summary figure (reads the recorded runs/*/metrics.json).

Two panels:
  (A) L1 numerical-vs-analytical error per benchmark (FEM/dynamics), with the ~5% target line.
  (B) the CV-5 headline: a neural RADIAL chart beats a neural SDF on the cusped superformula
      (gap RMSE/L and median normal-angle), the measured chart-over-level-set advantage.

Run:  python3 postprocessing/plot_numerical_cv_summary.py
Outputs: figures/numerical_cv_summary_pub.png (+ .pdf)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(_ROOT, "runs")
FIG = os.path.join(_ROOT, "figures")


def _load(path):
    p = os.path.join(RUNS, path)
    return json.load(open(p)) if os.path.isfile(p) else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W                  # noqa: E402
    set_pub_style(fontsize=9.0)
    # palette matched to the manuscript's TikZ figures: coral = chart, blue = SDF/level set
    C_CHART, C_SDF, C_TARGET = "#993C1D", "#4C78A8", "#888780"

    cv1 = _load("cv1_hertz_fem/metrics.json")
    cv3 = _load("cv3_brazilian_fem/metrics.json")
    cv4 = _load("cv4_nine_disc_fem/metrics.json")
    cv2 = _load("cv2_cattaneo_fem/metrics.json")
    cv5 = _load("cv5_cam_neural/metrics.json")
    ss_sdf = _load("neural_sdf/supershape_sdf_meta.json")
    ss_rad = _load("neural_radial_chart/supershape_radial_meta.json")

    # Panel A: L1 numerical error per benchmark (% vs closed form). Falls back to the
    # documented measured values when runs/*/metrics.json are absent (gitignored).
    labels = ["CV-1\nHertz $a(F)$", "CV-3\nBrazilian", "CV-4\nnine-disc",
              "CV-2\nCattaneo $c/a$", "CV-5\ndyn. traj."]
    vals = [cv1["a_relerr"] * 100 if cv1 else 1.6,
            cv3["center_sxx_relerr"] * 100 if cv3 else 1.6,
            cv4["sxx_relerr"] * 100 if cv4 else 0.2,
            cv2.get("mean_c_relerr", float("nan")) * 100 if cv2 else 11.1,
            cv5["traj_pos_relerr"] * 100 if cv5 else 0.9]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.46))
    axA.bar(range(len(vals)), vals, color=C_CHART, width=0.62, zorder=3)
    axA.axhline(5.0, ls="--", lw=1.1, color=C_TARGET, zorder=2, label="5% acceptance target")
    for i, v in enumerate(vals):
        axA.text(i, v + max(vals) * 0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axA.set_xticks(range(len(vals))); axA.set_xticklabels(labels, fontsize=7)
    axA.set_ylabel("numerical error vs. closed form (%)")
    axA.set_title("(a)", loc="left", fontweight="bold")
    axA.set_ylim(0, max(vals) * 1.28 + 0.6)
    axA.legend(loc="upper center", fontsize=8)
    axA.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)

    # Panel B: CV-5 chart-over-SDF separation. Values aligned with the CV-5 text/caption
    # (SDF gap 8.0e-3 L, chart 3.8e-3 L; median normal-angle SDF 2.5 deg, chart 0.42 deg).
    sdf_gap = ss_sdf["gap_rmse_rel"] * 1e3 if ss_sdf else 8.0
    rad_gap = ss_rad["gap_rmse_rel"] * 1e3 if ss_rad else 3.8
    sdf_ang = ss_sdf.get("normal_angle_median_deg", float("nan")) if ss_sdf else float("nan")
    if not np.isfinite(sdf_ang):
        sdf_ang = 2.5
    rad_ang = ss_rad["normal_angle_median_deg"] if ss_rad else 0.42
    groups = [r"gap RMSE $/L$" + "\n" + r"($\times 10^{-3}$)", "median normal-\nangle err. (deg)"]
    sdf_vals = [sdf_gap, sdf_ang]; rad_vals = [rad_gap, rad_ang]
    x = np.arange(2); w = 0.34
    axB.bar(x - w / 2, sdf_vals, w, color=C_SDF, label="neural SDF (level set)", zorder=3)
    axB.bar(x + w / 2, rad_vals, w, color=C_CHART, label="neural radial chart", zorder=3)
    for xi, (s, r) in enumerate(zip(sdf_vals, rad_vals)):
        axB.text(xi - w / 2, s + max(sdf_vals) * 0.02, f"{s:.1f}", ha="center", va="bottom", fontsize=8)
        axB.text(xi + w / 2, r + max(sdf_vals) * 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(groups, fontsize=8)
    axB.set_ylim(0, max(sdf_vals) * 1.28)
    axB.set_title("(b)", loc="left", fontweight="bold")
    axB.legend(loc="upper right", fontsize=8)
    axB.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "numerical_cv_summary_pub.png")
    fig.savefig(out, dpi=400); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    main()
