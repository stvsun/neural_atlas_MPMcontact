#!/usr/bin/env python3
"""OT advantage hero figure (loop 1) — measure-coupling transition map vs conventional gap.

ONE publication figure, three panels, every number drawn from the authoritative
verified set (docs/ot_benchmark/final_report.md table 1 + the recorded
runs/{cv8_deformable_ot,cv9_nbody_array_ot}/metrics.json). NO number is invented
here: the dictionaries below transcribe the verified results, the CV-8 mesh-
convergence curve is loaded from the run file when present, and the script asserts
internal consistency against runs/ before plotting.

  (a) Seven-case head-to-head. Conventional node-to-surface lumped-penalty gap vs the
      OT measure-coupling (Brenier transition-map) gap, each driven to its converged
      resolution, as a relative error (log scale, lower is better). Each pair is
      annotated with the measured advantage factor (3.2x, 370x, 6.3x, 1.4x,
      grid-indep., convergent, machine-prec.). The three geometry-dominated cases
      (CV-5/6/7) fall to machine/grid floors where a resolution-bound gap function is
      frozen; those OT bars are drawn at a plotting floor and labelled with their true
      value, and CV-6's non-convergent conventional baseline is drawn hatched.

  (b) Non-matching patch test — the mechanism, and the one rigorously mesh-matched
      head-to-head (mortar and lumped assembly share a single mesh). A constant
      interface pressure is driven across a deliberately non-conforming sawtooth
      interface. Node-lumped projection leaves a 67.3x pressure non-uniformity
      (range over mean); the OT mortar coupling transmits the same uniform field to
      1.4e-16 (mass-marginal error 0.0) because the gap and traction are
      mass-preserving FIELDS.

  (c) The two-body unlock — the regime OT is built for. The same coupling solves
      contact between two DEFORMABLE bodies through a symmetric SPSD 4-block mortar
      tangent (no rigid master). CV-8 deformable Hertz converges monotonically under
      mesh refinement (half-width relerr 5.14->2.75%), and the conservation/consistency
      structure holds to machine precision: global force balance ~1e-19 (CV-8) /
      3.7e-15 (CV-9a 3x3 array), the analytic tangent matches its finite difference to
      3.45e-11, and the CV-9a centre disc recovers the equibiaxial mean stress to 0.58%.

Run:    cd <repo> && python3 postprocessing/fig_ot_advantage_loop1.py
Output: figures/fig_ot_advantage_loop1.png (+ .pdf)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
RUNS = os.path.join(_ROOT, "runs")
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))

# --- palette: coral = OT measure-coupling, blue = conventional gap (matches the ---
# --- manuscript TikZ/figure convention chart=coral, level-set/conventional=blue) ---
C_OT = "#993C1D"      # OT measure-coupling transition map
C_CONV = "#4C78A8"    # conventional node-to-surface lumped-penalty gap
C_TARGET = "#888780"  # neutral grey for reference lines
C_INK = "#222222"


# ---------------------------------------------------------------------------
# Authoritative verified set (docs/ot_benchmark/final_report.md, table 1).
# conv=None marks the CV-6 floor case where the conventional model is
# resolution-bound (frozen / non-convergent, no single relerr) and the OT value is
# a machine/grid floor; these are drawn hatched / at a plotting floor and annotated
# with the true symbolic margin.
# ---------------------------------------------------------------------------
CV = [
    # key,    short label,                 conv,     ot,        advantage label, floor?
    ("CV-1", "Hertz\nline\n$a(F)$",        1.59e-2,  0.50e-2,   r"$3.2\times$",          False),
    ("CV-2", "Cattaneo\n$c/a$",            11.15e-2, 0.03e-2,   r"$\mathbf{370\times}$", False),
    ("CV-3", "Brazilian\n$\\sigma_{xx}$",  1.62e-2,  0.23e-2,   r"$6.3\times$",          False),
    ("CV-4", "nine-disc\n$\\sigma_{xx}$",  0.11e-2,  0.077e-2,  r"$1.4\times$",          False),
    ("CV-5", "superformula\ncusps",        1.99e-2,  5.3e-13,   "grid-\nindep.",         True),
    ("CV-6", "Koch\nfractal",              None,     2.6e-16,   "frozen\n$\\to$ conv.",  True),
    ("CV-7", "Patton\n$\\mu_{\\mathrm{app}}$", 1.5e-4, 3.5e-14, "machine\nprec.",        True),
]

FLOOR_PLOT = 3e-15        # where machine-floor OT bars are drawn (below true values)
CONV_FROZEN_PLOT = 0.42   # where the CV-6 frozen conventional bar is drawn (top region)

# ---------------------------------------------------------------------------
# Two-body deformable-deformable evidence (panel c). CV-8 mesh-convergence curve is
# loaded from runs/ when present; otherwise the documented values stand.
# ---------------------------------------------------------------------------
CV8_NX_FALLBACK = [96, 128, 160, 192]
CV8_A_RELERR_FALLBACK = [5.143e-2, 4.136e-2, 2.961e-2, 2.752e-2]   # half-width relerr


def _load_cv8_convergence():
    """Return (nx, a_relerr, p0_relerr_finest, fb_finest) from the CV-8 run file."""
    p = os.path.join(RUNS, "cv8_deformable_ot", "metrics.json")
    if not os.path.isfile(p):
        return (np.array(CV8_NX_FALLBACK, float),
                np.array(CV8_A_RELERR_FALLBACK, float), 5.82e-2, 9.8e-19)
    tab = json.load(open(p))["hertz_convergence"]["table"]
    nx = np.array([r["nx"] for r in tab], float)
    a = np.array([r["a_relerr"] for r in tab], float)
    p0 = float(tab[-1]["p0_relerr"])
    fb = float(min(abs(r["force_balance"]) for r in tab))
    return nx, a, p0, fb


def _verify_against_runs():
    """Cross-check the transcribed numbers against runs/ when present."""
    p8 = os.path.join(RUNS, "cv8_deformable_ot", "metrics.json")
    if os.path.isfile(p8):
        d8 = json.load(open(p8))
        pt = d8["patch_test"]
        assert abs(pt["lumped_uniformity_rel"] - 67.32701213725079) < 1e-6, pt["lumped_uniformity_rel"]
        assert pt["coupling_transmit_err"] < 1e-14, pt["coupling_transmit_err"]
        a_fin = d8["hertz_convergence"]["table"][-1]["a_relerr"]
        assert abs(a_fin - 0.02752) < 5e-4, a_fin     # 2.75% finest
    p9 = os.path.join(RUNS, "cv9_nbody_array_ot", "metrics.json")
    if os.path.isfile(p9):
        d9 = json.load(open(p9))
        assert abs(d9["center_mean_relerr"] - 0.0058) < 5e-4, d9["center_mean_relerr"]
        assert d9["global_balance"] < 1e-13, d9["global_balance"]
        assert d9["converged"] is True


def _sci(v, sig=1):
    """Render v as $m\\times10^{e}$ LaTeX with sig mantissa digits."""
    mant, exp = (f"%.{sig}e" % v).split("e")
    return rf"${mant}\!\times\!10^{{{int(exp)}}}$"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from utils import set_pub_style, DOUBLE_COL_W  # noqa: E402

    _verify_against_runs()
    cv8_nx, cv8_a, cv8_p0, cv8_fb = _load_cv8_convergence()
    set_pub_style(fontsize=9.0)

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.40))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 0.95, 1.25], wspace=0.46)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # =====================================================================
    # Panel (a): seven-case head-to-head, log-scale relative error.
    # =====================================================================
    n = len(CV)
    x = np.arange(n)
    w = 0.38

    conv_plot, conv_frozen, ot_plot = [], [], []
    for _, _, conv, ot, _, _ in CV:
        if conv is None:
            conv_plot.append(CONV_FROZEN_PLOT)
            conv_frozen.append(True)
        else:
            conv_plot.append(conv)
            conv_frozen.append(False)
        ot_plot.append(max(ot, FLOOR_PLOT))

    for xi, (v, frozen) in enumerate(zip(conv_plot, conv_frozen)):
        axA.bar(xi - w / 2, v, w, color=C_CONV, zorder=3,
                hatch="////" if frozen else None,
                edgecolor=C_CONV if frozen else "white",
                linewidth=0.8 if frozen else 0.0)
    axA.bar(x + w / 2, ot_plot, w, color=C_OT, zorder=3)

    axA.set_yscale("log")
    axA.set_ylim(1e-16, 50.0)
    axA.set_xlim(-0.7, n - 0.3)
    axA.set_xticks(x)
    axA.set_xticklabels([c[1] for c in CV], fontsize=6.6)
    axA.set_ylabel("relative error  (lower is better)")
    axA.set_title("(a)  conventional gap vs. OT coupling",
                  loc="left", fontweight="bold", fontsize=8.2)
    axA.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    axA.set_axisbelow(True)
    axA.set_yticks([1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15])

    # 5% acceptance reference (drawn low so it never collides with the tag band)
    axA.axhline(5e-2, ls="--", lw=1.0, color=C_TARGET, zorder=2)
    axA.text(-0.62, 6.0e-2, "5% target", ha="left", va="bottom",
             fontsize=6.4, color=C_TARGET)

    # advantage-tag band lives in a clear strip near the top of the axis
    tag_y = 7.0
    for xi, (_, _, conv, ot, adv, _) in enumerate(CV):
        # OT true value, written just above the OT bar top
        if ot >= 1e-4:
            otxt = f"{ot*100:.2g}%"
        else:
            otxt = _sci(ot)
        axA.text(xi + w / 2, max(ot, FLOOR_PLOT) * 2.0, otxt, ha="center",
                 va="bottom", fontsize=6.0, color=C_OT, rotation=90)
        # conventional value, just above its bar top
        if conv is not None:
            axA.text(xi - w / 2, conv * 1.3, f"{conv*100:.2g}%", ha="center",
                     va="bottom", fontsize=6.0, color=C_CONV, rotation=90)
        else:
            axA.text(xi - w / 2, CONV_FROZEN_PLOT * 1.15, "frozen", ha="center",
                     va="bottom", fontsize=6.0, color=C_CONV, rotation=90)
        # advantage tag in the top band
        axA.text(xi, tag_y, adv, ha="center", va="center", fontsize=6.6,
                 fontweight="bold", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.16", fc="#F4ECE6",
                           ec=C_OT, lw=0.5))

    legend_handles = [
        Patch(facecolor=C_CONV, label="conventional node-to-surface gap"),
        Patch(facecolor=C_OT, label="OT measure-coupling (Brenier map)"),
    ]
    axA.legend(handles=legend_handles, loc="lower center", fontsize=6.4,
               ncol=1, framealpha=0.92, handlelength=1.2, borderpad=0.4)

    # =====================================================================
    # Panel (b): non-matching patch test — the mechanism (and only mesh-matched test).
    # =====================================================================
    pt_lumped = 67.32701213725079    # relative pressure non-uniformity (range/mean)
    pt_ot = 1.3877787648367726e-16   # OT mortar transmit error (uniform field exact)

    labels = ["node-\nlumped", "OT\nmortar"]
    vals = [pt_lumped, max(pt_ot, FLOOR_PLOT)]
    xb = np.arange(2)
    axB.bar(xb, vals, 0.58, color=[C_CONV, C_OT], zorder=3)
    axB.set_yscale("log")
    axB.set_ylim(1e-16, 5e3)
    axB.set_xticks(xb)
    axB.set_xticklabels(labels, fontsize=7.0)
    axB.set_yticks([1e2, 1e-1, 1e-4, 1e-7, 1e-10, 1e-13, 1e-16])
    axB.set_ylabel("pressure non-uniformity  (range / mean)", fontsize=7.6)
    axB.set_title("(b)  patch test", loc="left",
                  fontweight="bold", fontsize=8.2)
    axB.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    axB.set_axisbelow(True)

    axB.text(0, pt_lumped * 2.0, "67.3", ha="center", va="bottom",
             fontsize=7.6, color=C_CONV, fontweight="bold")
    axB.text(1, max(pt_ot, FLOOR_PLOT) * 2.2, _sci(pt_ot),
             ha="center", va="bottom", fontsize=6.8, color=C_OT, fontweight="bold")
    axB.annotate("mass-preserving\nfield $\\Rightarrow$\nexact transmission",
                 xy=(1, max(pt_ot, FLOOR_PLOT) * 60), xytext=(0.45, 4e-7),
                 ha="center", va="center", fontsize=6.0, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7))

    # =====================================================================
    # Panel (c): the two-body deformable-deformable unlock.
    # =====================================================================
    axC.plot(cv8_nx, cv8_a * 100, "o-", color=C_OT, lw=1.3, ms=4.5, zorder=4,
             label="CV-8 Hertz half-width $a$")
    for xi, yi in zip(cv8_nx, cv8_a * 100):
        axC.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=5.8, color=C_OT)
    axC.set_xlabel("mesh resolution  $n_x$", fontsize=7.6)
    axC.set_ylabel("CV-8 half-width relerr  (%)", fontsize=7.6)
    axC.set_title("(c)  two deformable bodies", loc="left",
                  fontweight="bold", fontsize=8.2)
    axC.set_xlim(cv8_nx[0] - 16, cv8_nx[-1] + 16)
    axC.set_ylim(0, max(cv8_a) * 100 * 1.30)
    axC.grid(True, ls=":", lw=0.4, alpha=0.5)
    axC.set_axisbelow(True)
    axC.legend(loc="upper right", fontsize=6.2, framealpha=0.92,
               handlelength=1.4, borderpad=0.35)

    # machine-precision conservation / consistency ledger (verified values). usetex=False,
    # so keep TeX only inside $...$ and avoid \text/\hspace/\bf (mathtext lacks them).
    # A single round box holds a blank first line (reserves space for the title) plus the
    # body; the bold OT-coloured title is then drawn on that reserved line, measured from
    # the rendered box extent so there is no hand-tuned offset and no residue.
    fs = 5.2
    title = "two-body machine-precision ledger"
    rows = [
        r"force balance  $1.3{\times}10^{-19}$ (CV-8)",
        r"force balance  $3.7{\times}10^{-15}$ (CV-9a)",
        r"tangent FD  $3.45{\times}10^{-11}$",
        r"symmetric SPSD 4-block tangent",
        r"CV-9a centre mean stress  0.58%",
    ]
    bbox = dict(boxstyle="round,pad=0.42", fc="#F4ECE6", ec=C_OT, lw=0.5)
    t_body = axC.text(0.05, 0.035, "\n".join([" "] + rows), transform=axC.transAxes,
                      ha="left", va="bottom", fontsize=fs, color=C_INK,
                      linespacing=1.6, bbox=bbox)
    fig.canvas.draw()
    ext = t_body.get_window_extent()
    inv = axC.transAxes.inverted()
    (xL, _), (_, yT) = inv.transform((ext.x0, ext.y0)), inv.transform((ext.x1, ext.y1))
    x_in = inv.transform((ext.x0 + 5, 0))[0] - inv.transform((ext.x0, 0))[0]
    y_in = inv.transform((0, ext.y1))[1] - inv.transform((0, ext.y1 - 5))[1]
    axC.text(xL + x_in, yT - y_in, title, transform=axC.transAxes,
             ha="left", va="top", fontsize=fs, color=C_OT, fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_ot_advantage_loop1.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("  CV-8 finest a relerr:", f"{cv8_a[-1]*100:.2f}%",
          "| p0:", f"{cv8_p0*100:.2f}%", "| force balance:", f"{cv8_fb:.1e}")
    print("  Saved:", out)
    print("  Saved:", out.replace(".png", ".pdf"))
    return out


if __name__ == "__main__":
    main()
