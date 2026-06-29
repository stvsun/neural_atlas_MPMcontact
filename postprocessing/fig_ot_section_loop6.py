#!/usr/bin/env python3
r"""OT-section schematic (loop 6) — the UNBALANCED / partial optimal-transport mechanism.

ONE publication schematic for the dedicated optimal-transport section that draws the
object the existing figures do NOT: the *partial* (unbalanced) optimal-transport problem
that selects the closest-point map for a localised convex indenter.  The body text states
the closest-point = partial-support limit through the marginal-relaxed Kantorovich
functional (eq:ot-unbalanced in app:ot:limits) and a one-line Chizat-2018 citation; this
figure makes that functional VISIBLE.  It is complementary to, not a duplicate of:
  - fig_ot_section_loop4  : the BALANCED Brenier picture (two measures, the plan
    chi_# mu_A = mu_B, quantile matching F_B^{-1} o F_A, and the two-limit geometry);
  - fig:tm-ot-limits (TikZ): the two limits as mated-arc / indenter sketches.

Loop 6 instead isolates the single under-illustrated step: WHY the global (balanced)
arclength map is inadmissible under partial contact, and HOW the unbalanced functional

    inf_{pi>=0}  int c(x,y) dpi
                 + lambda KL((P_A)_# pi || mu_A_hat)
                 + lambda KL((P_B)_# pi || mu_B_hat)               (eq:ot-unbalanced)

relaxes the hard marginals so the optimal plan may create/destroy mass, concentrate its
support on the touching band, and there restrict (for a convex master) to the
closest-point projection pi_B = phi_B^{-1} o phi_A, with gap g_N = ||p - c|| - R.  The
restriction is well posed OFF the master's medial axis, where the metric projection is
single-valued (Santambrogio 2015, S1.3) — drawn explicitly as the locus where pi_B is
multivalued.

Three panels:
  (a) BALANCED map smears.  A localised convex indenter on a flat slave.  The hard
      marginal forces every slave point onto a master partner, dragging the untouched
      flanks onto spurious indenter-edge feet (the failure mode the body text names).
  (b) UNBALANCED relaxation.  The KL-penalised plan creates/destroys mass off the
      touching band: support collapses onto the contact band, the open interface is left
      unmatched, and the restriction is the closest-point projection with g_N=||p-c||-R.
  (c) WELL-POSED off the medial axis.  The convex master's metric projection is
      single-valued for every slave point except on the medial axis (the cusp locus),
      where pi_B is multivalued; the chart inversion never forms that axis.

PURELY SCHEMATIC: carries NO measured benchmark number.  Geometry (indenter arc, flat
slave, closest-point feet, medial axis of a convex arc) is computed analytically so the
feet and the equidistant locus are faithful, not hand-placed.

Palette matches the manuscript TikZ convention: coral #993C1D = chart A / slave / OT,
blue #4C78A8 = chart B / master.

Run:    cd <repo> && python3 postprocessing/fig_ot_section_loop6.py
Output: figures/fig_ot_section_loop6.png (+ .pdf)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))

# --- palette (matches main.tex \definecolor) ---
C_A = "#993C1D"        # slave A  (coralStroke)
C_A_FILL = "#E7C7B6"   # light coral fill
C_B = "#4C78A8"        # master B (cBlue)
C_B_FILL = "#C5D3E4"   # light blue fill
C_INK = "#222222"
C_GREY = "#888780"
C_CHI = "#6A6A6A"      # neutral grey for the correspondence
C_BAD = "#B23B2E"      # spurious / inadmissible (warm red)
C_MED = "#8E44AD"      # medial axis (distinct purple, only used once)


# ---------------------------------------------------------------------------
# Analytic geometry: a convex circular indenter (master B) of centre c, radius R,
# pressed onto a flat slave face A at height y = y_slave.  The closest-point
# projection of a flat-face point x onto the lower arc is exact and gives both the
# foot and the gap g_N = ||p - c|| - R.
# ---------------------------------------------------------------------------
CX, CY, R = 0.0, 0.62, 0.55          # indenter centre + radius (panel-local coords)
Y_SLAVE = 0.0                        # flat slave height
HALF_W = 0.95                        # slave half-width


def _indenter_lower_arc(n=200, span_deg=150.0):
    """Lower face of the convex master arc, centred under the apex."""
    a = np.deg2rad(span_deg)
    th = np.linspace(-np.pi / 2 - a / 2, -np.pi / 2 + a / 2, n)
    return CX + R * np.cos(th), CY + R * np.sin(th)


def _closest_point_on_arc(px):
    """Closest point on the convex arc to a flat-slave point (px, Y_SLAVE).

    For a circle the metric projection is along the centre ray: the foot is
    c + R*(p - c)/||p - c||, and the signed gap is ||p - c|| - R.
    """
    p = np.array([px, Y_SLAVE])
    d = p - np.array([CX, CY])
    r = np.hypot(*d)
    foot = np.array([CX, CY]) + R * d / r
    gap = r - R
    return foot, gap


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
    from utils import set_pub_style, DOUBLE_COL_W  # noqa: E402

    set_pub_style(fontsize=9.0)
    plt.rcParams["axes.grid"] = False

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.44))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.04)
    axA = fig.add_subplot(gs[0, 0])   # balanced map smears
    axB = fig.add_subplot(gs[0, 1])   # unbalanced relaxation
    axC = fig.add_subplot(gs[0, 2])   # well-posed off the medial axis

    for ax in (axA, axB, axC):
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.42, 1.50)
        ax.set_aspect("equal")
        ax.axis("off")

    def _panel_title(ax, text):
        """Centred in-axes title that cannot overflow into the neighbour panel."""
        ax.text(0.0, 1.44, text, fontsize=8.0, fontweight="bold", color=C_INK,
                ha="center", va="center")

    arc_x, arc_y = _indenter_lower_arc()

    # touching band: where the arc apex is within a small clearance of the slave
    band = HALF_W * 0.34                                   # half-width of the contact band

    # =====================================================================
    # Panel (a): the BALANCED (hard-marginal) map smears a partial load.
    #   Every slave point is forced onto a master partner; the untouched flanks
    #   are dragged onto spurious indenter-edge feet (inadmissible).
    # =====================================================================
    axA.plot(arc_x, arc_y, color=C_B, lw=1.6, zorder=5)
    axA.fill_between(arc_x, arc_y, arc_y + 0.10, color=C_B_FILL, edgecolor="none",
                     zorder=2)
    axA.scatter([CX], [CY], s=8, color=C_B, zorder=6)
    axA.text(CX + 0.04, CY, r"$\mathbf{c}$", fontsize=7.4, color=C_B, ha="left",
             va="center")
    axA.add_patch(Rectangle((-HALF_W, Y_SLAVE - 0.085), 2 * HALF_W, 0.085,
                            facecolor=C_A_FILL, edgecolor=C_A, lw=1.3, zorder=3))

    # sample slave points across the whole face; the BALANCED map insists each
    # has a master partner, so the flank points fan onto the arc edge (spurious).
    xs = np.linspace(-0.82, 0.82, 9)
    arc_edge_x = arc_x[0], arc_x[-1]
    arc_edge_y = arc_y[0], arc_y[-1]
    for xk in xs:
        in_band = abs(xk) <= band
        if in_band:
            foot, _ = _closest_point_on_arc(xk)
            col, lw, alpha = C_CHI, 0.9, 0.95
        else:
            # forced onto the nearest arc EDGE -> a long, slanted, spurious arrow
            ei = 0 if xk < 0 else 1
            foot = np.array([arc_edge_x[ei], arc_edge_y[ei]])
            col, lw, alpha = C_BAD, 1.0, 0.95
        arr = FancyArrowPatch((xk, Y_SLAVE + 0.004), (foot[0], foot[1]),
                              arrowstyle="-|>", mutation_scale=6.5, lw=lw,
                              color=col, alpha=alpha, zorder=4,
                              shrinkA=0, shrinkB=1.5)
        axA.add_patch(arr)
        axA.scatter([xk], [Y_SLAVE], s=10, color=C_A, zorder=5)

    axA.text(0.0, 1.20, r"hard marginal $\chi_{\#}\hat\mu_A=\hat\mu_B$",
             fontsize=7.2, color=C_INK, ha="center", va="center")
    axA.text(0.0, -0.30, r"flanks smeared onto the indenter edge",
             fontsize=6.8, color=C_BAD, ha="center", va="center")
    axA.text(-0.985, 0.92, r"$\Gamma_c^B$", fontsize=7.4, color=C_B, ha="left",
             va="center", fontweight="bold")
    axA.text(-0.985, Y_SLAVE - 0.22, r"$\Gamma_c^A$", fontsize=7.4, color=C_A,
             ha="left", va="center", fontweight="bold")
    _panel_title(axA, r"(a)  balanced map is inadmissible")

    # =====================================================================
    # Panel (b): the UNBALANCED relaxation (eq:ot-unbalanced).
    #   The KL-penalised marginals let the plan create/destroy mass off the
    #   touching band; support collapses onto the band; the restriction is the
    #   closest-point projection with g_N = ||p - c|| - R.
    # =====================================================================
    axB.plot(arc_x, arc_y, color=C_B, lw=1.6, zorder=5)
    axB.fill_between(arc_x, arc_y, arc_y + 0.10, color=C_B_FILL, edgecolor="none",
                     zorder=2)
    axB.scatter([CX], [CY], s=8, color=C_B, zorder=6)
    axB.text(CX + 0.04, CY, r"$\mathbf{c}$", fontsize=7.4, color=C_B, ha="left",
             va="center")
    # slave face: touching band solid, untouched flanks faded (mass destroyed)
    axB.add_patch(Rectangle((-band, Y_SLAVE - 0.085), 2 * band, 0.085,
                            facecolor=C_A_FILL, edgecolor=C_A, lw=1.3, zorder=4))
    axB.add_patch(Rectangle((-HALF_W, Y_SLAVE - 0.085), HALF_W - band, 0.085,
                            facecolor="none", edgecolor=C_GREY, lw=0.8, ls=(0, (3, 2)),
                            zorder=3))
    axB.add_patch(Rectangle((band, Y_SLAVE - 0.085), HALF_W - band, 0.085,
                            facecolor="none", edgecolor=C_GREY, lw=0.8, ls=(0, (3, 2)),
                            zorder=3))

    # correspondence ONLY on the touching band = closest-point feet
    xs_band = np.linspace(-band * 0.92, band * 0.92, 5)
    for xk in xs_band:
        foot, gap = _closest_point_on_arc(xk)
        arr = FancyArrowPatch((xk, Y_SLAVE + 0.004), (foot[0], foot[1]),
                              arrowstyle="-|>", mutation_scale=6.5, lw=0.95,
                              color=C_CHI, alpha=0.95, zorder=5, shrinkA=0, shrinkB=1.5)
        axB.add_patch(arr)
        axB.scatter([xk], [Y_SLAVE], s=10, color=C_A, zorder=6)
    # the central foot p and the explicit gap radius g_N = ||p - c|| - R
    foot0, gap0 = _closest_point_on_arc(0.0)
    axB.scatter([0.0], [Y_SLAVE], s=22, color=C_A, zorder=7, ec=C_INK, lw=0.5)
    axB.scatter([foot0[0]], [foot0[1]], s=18, color=C_B, zorder=7, ec=C_INK, lw=0.4)
    axB.annotate("", xy=(0.0, foot0[1]), xytext=(0.0, Y_SLAVE),
                 arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.9))
    # gap label set off to the right with a thin leader so it clears the arrows
    axB.annotate(r"$g_N=\|\mathbf{p}-\mathbf{c}\|-R$",
                 xy=(0.012, 0.5 * (Y_SLAVE + foot0[1])), xytext=(0.40, 0.20),
                 fontsize=6.8, color=C_INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=C_INK, lw=0.5))
    axB.text(-0.055, Y_SLAVE + 0.01, r"$\mathbf{p}$", fontsize=7.2, color=C_A,
             ha="right", va="bottom")
    # "mass destroyed" tags on the faded flanks
    axB.text(-0.62, Y_SLAVE - 0.23, r"mass destroyed", fontsize=6.0, color=C_GREY,
             ha="center", va="center", style="italic")
    axB.text(0.62, Y_SLAVE - 0.23, r"mass destroyed", fontsize=6.0, color=C_GREY,
             ha="center", va="center", style="italic")

    axB.text(0.0, 1.20, r"$+\,\lambda\,\mathrm{KL}((P_A)_{\#}\pi\,\|\,\hat\mu_A)"
             r"+\lambda\,\mathrm{KL}((P_B)_{\#}\pi\,\|\,\hat\mu_B)$",
             fontsize=6.4, color=C_INK, ha="center", va="center")
    axB.text(0.0, 1.04, r"finite $\lambda$: support $\to$ touching band $=\pi_B$",
             fontsize=6.6, color=C_B, ha="center", va="center")
    _panel_title(axB, r"(b)  unbalanced relaxation")

    # =====================================================================
    # Panel (c): the restriction is WELL POSED off the medial axis.
    #   A convex master's metric projection is single-valued for every slave
    #   point except on the medial axis (here the centre ray of a NON-convex
    #   notch), where pi_B is multivalued.  We draw a convex arc + a re-entrant
    #   notch to show both: single-valued on the convex part, multivalued on the
    #   medial axis of the notch.
    # =====================================================================
    # two stacked sub-cases, partitioned by a faint divider, so the convex
    # (single-valued) case and the re-entrant (multivalued) case never overlap.
    axC.plot([-1.0, 1.0], [0.55, 0.55], color="0.85", lw=0.7, zorder=1)
    axC.add_patch(Rectangle((-HALF_W, Y_SLAVE - 0.085), 2 * HALF_W, 0.085,
                            facecolor=C_A_FILL, edgecolor=C_A, lw=1.3, zorder=3))

    # -- LOWER sub-case: convex master -> SINGLE-VALUED metric projection --
    th = np.linspace(np.deg2rad(208), np.deg2rad(332), 120)
    ccx, ccy, cR = 0.0, 0.50, 0.34
    cax = ccx + cR * np.cos(th)
    cay = ccy + cR * np.sin(th)
    axC.plot(cax, cay, color=C_B, lw=1.6, zorder=5)
    axC.fill_between(cax, cay, cay + 0.07, color=C_B_FILL, edgecolor="none", zorder=2)
    for xk in (-0.30, 0.0, 0.30):
        d = np.array([xk - ccx, Y_SLAVE - ccy])
        foot = np.array([ccx, ccy]) + cR * d / np.hypot(*d)
        axC.add_patch(FancyArrowPatch((xk, Y_SLAVE + 0.004), (foot[0], foot[1]),
                      arrowstyle="-|>", mutation_scale=5.5, lw=0.9, color=C_CHI,
                      zorder=5, shrinkA=0, shrinkB=1.2))
        axC.scatter([xk], [Y_SLAVE], s=9, color=C_A, zorder=6)
    axC.text(0.0, Y_SLAVE - 0.24, r"convex master $\Rightarrow\ \pi_B$ single-valued",
             fontsize=6.2, color=C_INK, ha="center", va="center")

    # -- UPPER sub-case: re-entrant master -> medial axis, pi_B MULTIVALUED --
    base = 0.78                                            # vertical base of the wedge demo
    apex = np.array([0.0, base + 0.04])                   # vee apex (concave, points down)
    wallL = np.array([-0.34, base + 0.42])
    wallR = np.array([0.34, base + 0.42])
    axC.plot([wallL[0], apex[0], wallR[0]], [wallL[1], apex[1], wallR[1]],
             color=C_B, lw=1.6, zorder=6)
    axC.fill_between([wallL[0], apex[0], wallR[0]],
                     [wallL[1], apex[1], wallR[1]],
                     [wallL[1] + 0.06, apex[1] + 0.06, wallR[1] + 0.06],
                     color=C_B_FILL, edgecolor="none", zorder=2)
    # medial axis: the equidistant ray from the apex along the vee bisector
    med_bot = np.array([0.0, base - 0.16])
    axC.plot([apex[0], med_bot[0]], [apex[1], med_bot[1]], color=C_MED,
             lw=1.1, ls=(0, (4, 2)), zorder=5)
    axC.text(apex[0] + 0.05, 0.5 * (apex[1] + med_bot[1]) + 0.04, "medial axis",
             fontsize=6.0, color=C_MED, ha="left", va="center")
    # a query point on the medial axis with TWO equal feet (multivalued)
    pm = med_bot
    for w in (wallL, wallR):
        seg = w - apex
        t = np.clip(np.dot(pm - apex, seg) / np.dot(seg, seg), 0.0, 1.0)
        f = apex + t * seg
        axC.add_patch(FancyArrowPatch((pm[0], pm[1] + 0.004), (f[0], f[1]),
                      arrowstyle="-|>", mutation_scale=5.5, lw=0.9, color=C_BAD,
                      alpha=0.95, zorder=6, shrinkA=0, shrinkB=1.0))
    axC.scatter([pm[0]], [pm[1]], s=13, color=C_A, zorder=7, ec=C_INK, lw=0.4)
    axC.text(0.0, base - 0.27, r"re-entrant $\Rightarrow\ \pi_B$ multivalued",
             fontsize=6.2, color=C_BAD, ha="center", va="center")

    _panel_title(axC, r"(c)  well posed off the medial axis")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.04)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_ot_section_loop6.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)

    # self-consistency: the closest-point gap is exactly ||p - c|| - R (no number
    # is a benchmark claim; this only checks the drawn geometry is faithful).
    _, gap_apex = _closest_point_on_arc(0.0)
    expect = abs(CY - R - Y_SLAVE)
    print("  closest-point apex gap g_N = %.4f  (||p-c||-R = %.4f, match %.1e)"
          % (gap_apex, CY - R - Y_SLAVE, abs(gap_apex - (CY - R - Y_SLAVE))))
    print("  Saved:", out)
    print("  Saved:", out.replace(".png", ".pdf"))
    return out


if __name__ == "__main__":
    main()
