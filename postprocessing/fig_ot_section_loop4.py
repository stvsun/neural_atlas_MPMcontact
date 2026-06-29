#!/usr/bin/env python3
r"""OT-section schematic (loop 4) — the Monge--Kantorovich transport plan / Brenier map.

ONE publication schematic for the dedicated optimal-transport section.  It draws the
measure-theoretic object that the existing two-limits sketch (Figure~tm-ot-limits) and
the composite (transition_map_composite_pub) do NOT show: the two boundary measures
mu_A, mu_B carried by the slave/master faces, the admissible mass-preserving
correspondence chi = tau_AB that couples them, the quantile-matching mechanism
F_B(tau_AB(x)) = F_A(x) that selects the Brenier map, and the two regimes
(closest-point projection vs arclength-monotone rearrangement) as the two
specialisations of that one coupling.

This figure is PURELY SCHEMATIC: it carries NO measured benchmark number (those live in
fig_ot_advantage_loop1).  The only quantities drawn are the analytic profiles h_A, h_B,
their arclength densities sqrt(1+h'^2), and the exact monotone rearrangement
tau = F_B^{-1} o F_A computed from them — the same construction MonotoneCoupling1D uses
(arclength CDF, then interp the quantile), so the curves are faithful to the code, not
hand-drawn.

Notation matches main.tex appendix app:ot:
    d mu_X      = sqrt(1 + |h_X'|^2) dx          (eq:ot-measure, arclength measure)
    F_X(theta)  = int d mu_X                      (eq:ot-brenier-1d, cumulative arclength)
    chi = tau_AB= F_B^{-1} o F_A                   (eq:ot-brenier-1d, the Brenier map in 1-D)
    g_N         = (x - tau_AB(x)) . n              (eq:ot-gap)
    two limits  : closest-point pi_B (partial / convex) ; arclength-monotone (conforming)

Palette matches the manuscript TikZ convention: coral #993C1D = chart A / slave / OT,
blue #4C78A8 = chart B / master / level-set-class.

Run:    cd <repo> && python3 postprocessing/fig_ot_section_loop4.py
Output: figures/fig_ot_section_loop4.png (+ .pdf)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))

# --- palette (matches main.tex \definecolor): coral = chart A / slave / OT, ---
# --- blue = chart B / master.  Greys neutral. ---
C_A = "#993C1D"       # slave A  (coralStroke)
C_A_FILL = "#E7C7B6"  # light coral fill
C_B = "#4C78A8"       # master B (cBlue)
C_B_FILL = "#C5D3E4"  # light blue fill
C_INK = "#222222"
C_GREY = "#888780"
C_CHI = "#6A6A6A"     # neutral grey for the correspondence (the coupling itself)


# ---------------------------------------------------------------------------
# Analytic faces and the EXACT 1-D monotone (Brenier) rearrangement built the
# same way MonotoneCoupling1D does it: arclength density -> cumulative arclength
# CDF -> tau = F_B^{-1} o F_A by quantile interpolation.  Two unequal-length
# faces, renormalised to equal total mass (the admissibility condition of
# eq:ot-measure / app:ot:gap).
# ---------------------------------------------------------------------------
def _faces(n=801):
    xa = np.linspace(0.0, 1.0, n)
    xb = np.linspace(0.0, 1.0, n)
    # slave A: a rough face whose roughness is CONCENTRATED on the left half, so the
    # arclength density (and hence the cumulative arclength F_A) is strongly front-
    # loaded; master B: a gentle, nearly symmetric face.  The contrasting densities
    # make the coupling chi visibly NON-IDENTITY (the arrows tilt) and separate the
    # two CDFs in panel (b) so the quantile bridge is legible.
    ramp = np.clip(1.5 - 1.4 * xa, 0.05, 1.0)            # taper roughness left->right
    ha = ramp * (0.085 * np.sin(2.0 * np.pi * 2.3 * xa)
                 + 0.045 * np.sin(2.0 * np.pi * 4.2 * xa + 0.6))
    hb = 0.055 * np.sin(2.0 * np.pi * 1.0 * xb + 0.30)
    return xa, ha, xb, hb


def _arclength_cdf(x, h):
    """Normalised cumulative arclength F_hat (CDF of the arclength measure)."""
    dh = np.gradient(h, x)
    dens = np.sqrt(1.0 + dh ** 2)            # d mu = sqrt(1+h'^2) dx  (eq:ot-measure)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x))])
    return dens, F / F[-1]                    # renormalise -> equal total mass


def _brenier_map(xa, ha, xb, hb):
    """tau_AB = F_B^{-1} o F_A on the slave parameter line (the same as code)."""
    dens_a, Fa = _arclength_cdf(xa, ha)
    dens_b, Fb = _arclength_cdf(xb, hb)
    q = np.interp(xa, xa, Fa)                 # quantile q = F_A(x)
    tau = np.interp(q, Fb, xb)                # x_master = F_B^{-1}(q)
    return dens_a, Fa, dens_b, Fb, tau


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from utils import set_pub_style, DOUBLE_COL_W  # noqa: E402

    set_pub_style(fontsize=9.0)
    plt.rcParams["axes.grid"] = False

    xa, ha, xb, hb = _faces()
    dens_a, Fa, dens_b, Fb, tau = _brenier_map(xa, ha, xb, hb)

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.35], wspace=0.40)
    axA = fig.add_subplot(gs[0, 0])   # transport plan: the two measures + chi
    axB = fig.add_subplot(gs[0, 1])   # quantile matching mechanism
    axC = fig.add_subplot(gs[0, 2])   # the two limits

    # =====================================================================
    # Panel (a): the transport plan chi coupling mu_A and mu_B.
    #   - slave face A (bottom) carries density mu_A (shaded thickness)
    #   - master face B (top)   carries density mu_B
    #   - a mass-preserving sample of correspondence arrows x -> tau_AB(x)
    # =====================================================================
    yA0, yB0 = 0.0, 1.0
    sc = 0.30   # vertical scale for face undulation
    # face curves
    axA.plot(xa, yA0 + sc * ha, color=C_A, lw=1.6, zorder=5)
    axA.plot(xb, yB0 + sc * hb, color=C_B, lw=1.6, zorder=5)

    # density ribbon: draw the arclength measure as a variable-thickness band
    da = 0.060 * (dens_a / dens_a.max())
    db = 0.060 * (dens_b / dens_b.max())
    axA.fill_between(xa, yA0 + sc * ha - da, yA0 + sc * ha, color=C_A_FILL,
                     edgecolor="none", zorder=3)
    axA.fill_between(xb, yB0 + sc * hb, yB0 + sc * hb + db, color=C_B_FILL,
                     edgecolor="none", zorder=3)

    # correspondence arrows on a MASS-preserving sample: equal-quantile points
    # q_k -> x = F_A^{-1}(q_k) on A, tau = F_B^{-1}(q_k) on B  (so the arrows
    # carry equal mass, visualising chi_# mu_A = mu_B).
    qk = np.linspace(0.06, 0.94, 9)
    xA_k = np.interp(qk, Fa, xa)
    xB_k = np.interp(qk, Fb, xb)
    yA_k = yA0 + sc * np.interp(xA_k, xa, ha)
    yB_k = yB0 + sc * np.interp(xB_k, xb, hb)
    for xk, yk, xk2, yk2 in zip(xA_k, yA_k, xB_k, yB_k):
        arr = FancyArrowPatch((xk, yk + 0.012), (xk2, yk2 - 0.012),
                              arrowstyle="-|>", mutation_scale=6.5,
                              lw=0.9, color=C_CHI, alpha=0.85, zorder=4,
                              shrinkA=0, shrinkB=0)
        axA.add_patch(arr)
    # highlight one correspondence pair in ink
    axA.scatter(xA_k[4], yA_k[4], s=16, color=C_A, zorder=6)
    axA.scatter(xB_k[4], yB_k[4], s=16, color=C_B, zorder=6)
    axA.annotate(r"$\chi=\tau_{AB}$", xy=(0.5 * (xA_k[4] + xB_k[4]),
                 0.5 * (yA_k[4] + yB_k[4])), xytext=(0.62, 0.52),
                 fontsize=8.5, color=C_INK,
                 arrowprops=dict(arrowstyle="-", color=C_CHI, lw=0.6))

    axA.text(0.50, yA0 - 0.155, r"$d\mu_A=\sqrt{1+|h_A'|^2}\,dx$",
             fontsize=7.0, color=C_A, ha="center", va="center")
    axA.text(0.50, yB0 + 0.165, r"$d\mu_B=\sqrt{1+|h_B'|^2}\,dx$",
             fontsize=7.0, color=C_B, ha="center", va="center")
    axA.text(0.985, yA0 - 0.07, r"$\Gamma_c^A$ (slave)", fontsize=7.2, color=C_A,
             ha="right", va="center", fontweight="bold")
    axA.text(0.985, yB0 + 0.075, r"$\Gamma_c^B$ (master)", fontsize=7.2, color=C_B,
             ha="right", va="center", fontweight="bold")
    axA.set_xlim(-0.02, 1.02)
    axA.set_ylim(yA0 - 0.22, yB0 + 0.24)
    axA.axis("off")
    axA.set_title(r"(a)  transport plan $\chi_{\#}\hat\mu_A=\hat\mu_B$",
                  loc="left", fontweight="bold", fontsize=8.4)

    # =====================================================================
    # Panel (b): the mechanism — equal cumulative-arclength quantiles.
    #   F_A(x) and F_B(.) ; the Brenier map sends x to tau with F_B(tau)=F_A(x).
    # =====================================================================
    axB.plot(xa, Fa, color=C_A, lw=1.6, label=r"$F_A$")
    axB.plot(xb, Fb, color=C_B, lw=1.6, label=r"$F_B$")
    # pick a representative slave point, draw the quantile bridge
    x0 = 0.32
    q0 = float(np.interp(x0, xa, Fa))
    t0 = float(np.interp(q0, Fb, xb))
    axB.plot([x0, x0], [0, q0], color=C_A, lw=0.8, ls=":", zorder=2)
    axB.plot([0, t0], [q0, q0], color=C_GREY, lw=0.8, ls="--", zorder=2)
    axB.plot([t0, t0], [q0, 0], color=C_B, lw=0.8, ls=":", zorder=2)
    axB.scatter([x0], [q0], s=18, color=C_A, zorder=5)
    axB.scatter([t0], [q0], s=18, color=C_B, zorder=5)
    axB.annotate("", xy=(t0, 0.05), xytext=(x0, 0.05),
                 arrowprops=dict(arrowstyle="-|>", color=C_CHI, lw=1.0))
    axB.text(0.5 * (x0 + t0), 0.085, r"$\tau_{AB}$", ha="center", va="bottom",
             fontsize=8.0, color=C_INK)
    axB.text(x0 - 0.015, -0.05, r"$x$", ha="right", va="top", fontsize=7.6, color=C_A)
    axB.text(t0 + 0.02, -0.05, r"$\tau_{AB}(x)$", ha="left", va="top",
             fontsize=7.0, color=C_B)
    axB.text(0.045, q0 + 0.035, r"$F_B(\tau_{AB})=F_A(x)$", fontsize=6.8,
             color=C_GREY, ha="left", va="bottom")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1.02)
    axB.set_xlabel("surface parameter", fontsize=7.6)
    axB.set_ylabel("cumulative arclength  $F_X$", fontsize=7.6)
    axB.set_title(r"(b)  Brenier map $\tau_{AB}=F_B^{-1}\!\circ F_A$",
                  loc="left", fontweight="bold", fontsize=8.4)
    axB.legend(loc="lower right", fontsize=7.0, framealpha=0.9, handlelength=1.3,
               borderpad=0.35)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # =====================================================================
    # Panel (c): the two limits as specialisations of the one coupling.
    #   top sub-panel  : partial / convex  -> closest-point pi_B, g_N=|p-c|-R
    #   bot sub-panel  : conforming / rough -> arclength-monotone
    # =====================================================================
    axC.axis("off")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.set_title(r"(c)  two limits of one coupling", loc="left",
                  fontweight="bold", fontsize=8.4)

    # -- top: partial-support / convex master = closest-point projection --
    # indenter arc (master B) and a flat slave A; load localises at one foot
    th = np.linspace(np.deg2rad(205), np.deg2rad(335), 80)
    cx, cy, R = 0.50, 0.93, 0.30
    axC.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=C_B, lw=1.5)
    axC.scatter([cx], [cy], s=8, color=C_B, zorder=5)
    axC.text(cx + 0.02, cy - 0.005, r"$\mathbf{c}$", fontsize=7.0, color=C_B,
             ha="left", va="center")
    axC.add_patch(Rectangle((0.10, 0.575), 0.80, 0.045, facecolor=C_A_FILL,
                            edgecolor=C_A, lw=1.2, zorder=3))
    # the single foot p under the apex + the gap radius
    py = 0.620
    axC.annotate("", xy=(cx, py), xytext=(cx, cy - R + 0.005),
                 arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.7, ls=(0, (2, 2))))
    axC.scatter([cx], [py], s=18, color=C_A, zorder=6)
    axC.annotate("", xy=(cx, py + 0.10), xytext=(cx, py),
                 arrowprops=dict(arrowstyle="-|>", color=C_B, lw=1.1))
    axC.text(cx + 0.025, py + 0.02, r"$\mathbf{p}$", fontsize=7.0, color=C_A,
             ha="left", va="bottom")
    axC.text(0.5, 0.535, r"partial $\Rightarrow$ closest point  "
             r"$\pi_B$,  $g_N=\|\mathbf{p}-\mathbf{c}\|-R$",
             fontsize=6.6, color=C_INK, ha="center", va="top")

    # divider
    axC.plot([0.04, 0.96], [0.50, 0.50], color="0.85", lw=0.7)

    # -- bottom: conforming / rough = arclength-monotone (mated saw arcs) --
    xt = np.array([0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.90])
    yt = np.array([0.27, 0.37, 0.27, 0.37, 0.27, 0.37, 0.32])
    axC.plot(xt, yt + 0.045, color=C_B, lw=1.4)            # master face
    axC.plot(xt, yt, color=C_A, lw=1.4)                    # slave face (mated)
    for xx, yy in zip(0.5 * (xt[:-1] + xt[1:]), 0.5 * (yt[:-1] + yt[1:])):
        axC.annotate("", xy=(xx, yy + 0.045), xytext=(xx, yy),
                     arrowprops=dict(arrowstyle="-|>", color=C_CHI, lw=0.7))
    axC.text(0.5, 0.165, r"conforming $\Rightarrow$ arclength-monotone  "
             r"$F_B^{-1}\!\circ F_A$",
             fontsize=6.6, color=C_INK, ha="center", va="top")
    axC.text(0.5, 0.105, r"every slave arc carries its mated partner",
             fontsize=6.0, color=C_GREY, ha="center", va="top")

    fig.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.10)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_ot_section_loop4.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)

    # report a couple of self-consistency facts (no benchmark numbers claimed)
    marg = float(np.max(np.abs(np.interp(tau, xb, Fb) - Fa)))
    trapz = getattr(np, "trapezoid", np.trapz)
    print("  faces: len(A)=%.4f  len(B)=%.4f (renormalised to equal mass)"
          % (trapz(dens_a, xa), trapz(dens_b, xb)))
    print("  quantile identity max|F_B(tau)-F_A| = %.2e (machine, schematic build)" % marg)
    print("  Saved:", out)
    print("  Saved:", out.replace(".png", ".pdf"))
    return out


if __name__ == "__main__":
    main()
