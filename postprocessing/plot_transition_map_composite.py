#!/usr/bin/env python3
"""Figure 2 (redesigned): the composite mapping, contact mechanics in physical AND mapped coordinates.

A diptych. LEFT (a) = the PHYSICAL configuration: two bodies in near-contact, with the contact point
x=phi_A(theta_A), the mating foot phi_B(psi_B), the signed normal gap g_N along n, the traction
t=t_N n + t_T, and the iterative closest-point search a level set would need (de-emphasised, grey).
CENTRE = a frameless gutter carrying the bold inter-body transition map tau_AB=pi_B o phi_A and the
small commuting triangle theta_A --phi_A--> x --pi_B--> psi_B that states tau_AB=pi_B o phi_A.
RIGHT = the MAPPED / reference coordinates: (b) the 1-D parameter strips Theta_A over Theta_B with the
carets theta_A, psi_B and the in-chart tau_AB (parameter -> parameter, never into physical space); and
(c) the pulled-back gap gbar(theta_A)=g_N^B(phi_A(theta_A)), whose open sub-level set {gbar<0} is the
active set -- a union of arcs (Remark, multi-arc completeness). Footer: deltaW_c depends on the
geometry only through (g_N, n), so the two coordinate pictures are interchangeable at the force layer.

Run:  <venv>/bin/python postprocessing/plot_transition_map_composite.py
Out:  figures/transition_map_composite_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")

CORAL = "#BE5536"   # body A / chart A / active set
BLUE  = "#3E6E9E"   # body B / chart B / mate
INK   = "#222222"   # kinematic primitives: n, t, tau_AB, gbar, dots
MUTE  = "#9a958a"   # the measured-but-incidental: g_N dimension, the search we replace, guides
LIGHT = "#f4f2ee"   # equation-box fill


# ---- physical-edge geometry (functions over x so the asperity is controllable) --------------
XC = 0.42                      # shared contact column
def yA(x):                     # body A lower belly (locally flat at XC so n points straight down)
    return 0.635 + 0.020 * np.cos(2 * np.pi * (x - XC) / 0.95)
def yB(x):                     # body B crown carrying a single asperity near XC
    return 0.400 + 0.052 * np.exp(-((x - 0.455) / 0.11) ** 2) + 0.018 * np.exp(-((x - 0.30) / 0.07) ** 2)


def _arrow(ax, p0, p1, color, lw=1.5, ms=12, style="-|>", ls="-", rad=0.0, alpha=1.0, zorder=5):
    from matplotlib.patches import FancyArrowPatch
    cs = f"arc3,rad={rad}" if rad else "arc3"
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        ls=ls, alpha=alpha, shrinkA=0, shrinkB=0, connectionstyle=cs,
                        capstyle="round", zorder=zorder)
    ax.add_patch(a)
    return a


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Arc
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=8.5)

    fig = plt.figure(figsize=(DOUBLE_COL_W, 3.62))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.06, 0.40, 1.04],
                             wspace=0.05, left=0.012, right=0.988, top=0.90, bottom=0.15)
    axP = fig.add_subplot(outer[0, 0])
    axG = fig.add_subplot(outer[0, 1])
    rg = outer[0, 2].subgridspec(2, 1, height_ratios=[0.58, 0.42], hspace=0.55)
    axTh = fig.add_subplot(rg[0])
    axGap = fig.add_subplot(rg[1])
    for ax in (axP, axG, axTh):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axP.set_aspect("equal")

    # ============================ (a) PHYSICAL ============================
    xs = np.linspace(-0.02, 1.02, 400)
    # body A: fill above its belly up out of frame; body B: fill below its crown
    axP.fill_between(xs, yA(xs), 1.16, color=CORAL, alpha=0.12, lw=0, zorder=1)
    axP.plot(xs, yA(xs), color=CORAL, lw=2.2, zorder=3)
    axP.fill_between(xs, -0.16, yB(xs), color=BLUE, alpha=0.12, lw=0, zorder=1)
    axP.plot(xs, yB(xs), color=BLUE, lw=2.2, zorder=3)
    axP.text(0.07, 0.93, "body $A$", color=CORAL, fontsize=8.5, fontweight="bold", va="top")
    axP.text(0.07, 0.07, "body $B$", color=BLUE, fontsize=8.5, fontweight="bold", va="bottom")
    axP.text(0.90, yA(0.90) + 0.045, r"$\partial\Omega_A$", color=CORAL, fontsize=9, ha="center")
    axP.text(0.90, yB(0.90) - 0.055, r"$\partial\Omega_B$", color=BLUE, fontsize=9, ha="center")

    xpt = np.array([XC, yA(XC)])                       # contact point x = phi_A(theta_A)
    n = np.array([0.0, -1.0])                           # outward unit normal of A at x (flat belly)
    foot = np.array([XC, yB(XC)])                       # mating foot phi_B(psi_B), along n
    ymid = 0.5 * (yA(XC) + yB(XC))
    # outward normal of A, into the gap
    _arrow(axP, xpt, xpt + 0.165 * n, INK, lw=1.6, ms=12, zorder=6)
    axP.text(XC + 0.032, yA(XC) - 0.135, r"$\mathbf{n}$", color=INK, fontsize=9.5, ha="left")
    # contact traction reaction on A:  t = t_N n + t_T  (stub up into A, plus a tangential barb)
    _arrow(axP, xpt, xpt - 0.12 * n, INK, lw=2.4, ms=11, zorder=6)
    _arrow(axP, xpt + np.array([0.0, 0.018]), xpt + np.array([0.085, 0.018]), INK, lw=1.4, ms=9, zorder=6)
    axP.text(XC, yA(XC) + 0.16, r"$\mathbf{t}=t_N\mathbf{n}+\mathbf{t}_T$", color=INK, fontsize=8.5, ha="center")
    axP.text(XC + 0.105, yA(XC) + 0.04, r"$\mathbf{t}_T$", color=INK, fontsize=8, ha="left")
    # signed normal gap g_N, collinear with n, offset to the right
    off = np.array([0.072, 0.0])
    _arrow(axP, xpt + off, foot + off, MUTE, lw=1.3, ms=9, style="<|-|>", zorder=5)
    axP.text(XC + 0.105, ymid, r"$g_N$", color=MUTE, fontsize=9.5, ha="left", va="center")
    # closest-point search (subordinate; drawn to the LEFT, faint)
    for sx in (0.330, 0.365):                          # competing trial feet a search iterates over
        _arrow(axP, xpt, np.array([sx, yB(sx)]), MUTE, lw=0.8, ms=6, ls=(0, (3, 2)), alpha=0.9, zorder=4)
        axP.plot(sx, yB(sx) + 0.012, "o", ms=2.6, mfc="none", mec=MUTE, mew=0.8, zorder=4)
    axP.text(0.185, ymid + 0.015, "closest-point\nsearch (iterative)", color=MUTE, fontsize=6.6,
             style="italic", ha="center", va="center", linespacing=0.95)
    # markers + labels
    axP.plot(*xpt, "o", ms=6, color=INK, zorder=7)
    axP.plot(*foot, "s", ms=6, color=BLUE, mec="white", mew=0.6, zorder=7)
    axP.text(XC - 0.05, yA(XC) + 0.004, r"$\mathbf{x}=\varphi_A(\theta_A)$", color=INK, fontsize=8.5,
             ha="right", va="center")
    axP.text(XC, yB(XC) - 0.052, r"$\varphi_B(\psi_B)$", color=BLUE, fontsize=8.5, ha="center", va="top")
    axP.text(0.02, 1.09, "(a) physical coordinates", fontsize=9, fontweight="bold", va="top")
    axP.text(0.02, 1.015, "mate by closest-point search", fontsize=6.8, color=MUTE, style="italic", va="top")

    # ============================ GUTTER: composite map ============================
    yb = 0.70
    _arrow(axG, (0.04, yb), (0.96, yb), INK, lw=2.2, ms=14, zorder=6)
    axG.text(0.5, yb + 0.085, r"$\tau_{AB}=\pi_B\circ\varphi_A$", color=INK, fontsize=10.5, ha="center")
    axG.text(0.5, yb - 0.072, r"evaluate $\varphi_A$, invert $\varphi_B$: no search", color=MUTE,
             fontsize=6.4, ha="center", style="italic")
    # small commuting triangle: theta_A --phi_A--> x --pi_B--> psi_B,  hypotenuse = tau_AB
    tL, tR, tA = np.array([0.13, 0.40]), np.array([0.87, 0.40]), np.array([0.50, 0.12])
    _arrow(axG, tL, tA, CORAL, lw=1.2, ms=8, rad=0.0, zorder=5)
    _arrow(axG, tA, tR, BLUE, lw=1.2, ms=8, rad=0.0, zorder=5)
    _arrow(axG, tL, tR, INK, lw=1.4, ms=9, rad=-0.16, zorder=5)
    axG.text(tL[0] - 0.015, tL[1] + 0.012, r"$\theta_A$", color=CORAL, fontsize=8.5, ha="right", va="bottom")
    axG.text(tR[0] + 0.015, tR[1] + 0.012, r"$\psi_B$", color=BLUE, fontsize=8.5, ha="left", va="bottom")
    axG.text(tA[0], tA[1] - 0.045, r"$\mathbf{x}$", color=INK, fontsize=8.5, ha="center", va="top")
    axG.text(0.265, 0.295, r"$\varphi_A$", color=CORAL, fontsize=8, ha="center")
    axG.text(0.735, 0.295, r"$\pi_B$", color=BLUE, fontsize=8, ha="center")
    axG.text(0.50, 0.485, r"$\tau_{AB}$", color=INK, fontsize=8, ha="center")

    # ============================ (b) REFERENCE STRIPS ============================
    axTh.text(0.0, 1.14, "(b) reference coordinates", fontsize=9, fontweight="bold", va="top")
    x0, x1 = 0.08, 0.92
    thA_x, psB_x = 0.40, 0.50                            # caret positions (non-trivial reparametrisation)
    def strip(yc, color, lab):
        axTh.add_patch(FancyBboxPatch((x0, yc - 0.055), x1 - x0, 0.11,
                       boxstyle="round,pad=0.006,rounding_size=0.04", mutation_aspect=0.4,
                       fc=color, ec=color, lw=1.6, alpha=0.12, zorder=2))
        axTh.plot([x0, x1], [yc, yc], color=color, lw=1.6, zorder=3, solid_capstyle="round")
        axTh.text(x1 + 0.03, yc, lab, color=color, fontsize=9.5, va="center", ha="left")
    strip(0.72, CORAL, r"$\Theta_A$")
    strip(0.24, BLUE,  r"$\Theta_B$")
    axTh.plot(thA_x, 0.72, marker="^", ms=8, color=CORAL, zorder=5)
    axTh.text(thA_x, 0.86, r"$\theta_A$", color=CORAL, fontsize=9, ha="center")
    axTh.plot(psB_x, 0.24, marker="^", ms=8, color=BLUE, zorder=5)
    axTh.text(psB_x, 0.085, r"$\psi_B$", color=BLUE, fontsize=9, ha="center", va="top")
    _arrow(axTh, (thA_x, 0.665), (psB_x, 0.30), INK, lw=1.7, ms=12, rad=-0.30, zorder=4)
    axTh.text(0.30, 0.47, r"$\tau_{AB}$", color=INK, fontsize=9, ha="center")
    axTh.text(0.5, -0.03, r"direct chart composition:  parameter $\to$ parameter",
              color=MUTE, fontsize=6.8, ha="center", style="italic", va="top")

    # ============================ (c) PULLED-BACK GAP ============================
    axGap.set_xlim(0, 1)
    t = np.linspace(0.03, 0.97, 500)
    c0, (b1, t1, w1), (b2, t2, w2) = 0.135, (0.40, 0.40, 0.052), (0.30, 0.70, 0.048)
    gbar = c0 - b1 * np.exp(-((t - t1) / w1) ** 2) - b2 * np.exp(-((t - t2) / w2) ** 2)
    axGap.axhline(0.0, color=MUTE, lw=0.9, ls=(0, (4, 2)), zorder=1)
    axGap.text(0.97, 0.012, r"$\bar g=0$", color=MUTE, fontsize=7, ha="right", va="bottom")
    axGap.fill_between(t, gbar, 0, where=gbar < 0, color=CORAL, alpha=0.22, lw=0, zorder=2)
    axGap.plot(t, gbar, color=INK, lw=1.6, zorder=4)
    axGap.set_ylim(gbar.min() - 0.155, c0 + 0.06)
    # active-set brackets under the curve (two disjoint sub-zero arcs -> union of arcs)
    ylo = gbar.min() - 0.05
    for (tc, wc, bb) in ((t1, w1, b1), (t2, w2, b2)):
        half = wc * np.sqrt(np.log(bb / c0))
        axGap.plot([tc - half, tc + half], [ylo, ylo], color=CORAL, lw=2.8, solid_capstyle="butt", zorder=4)
    axGap.text(0.5, gbar.min() - 0.098, r"$\{\theta_A:\bar g<0\}$  active set (union of arcs)",
               color=CORAL, fontsize=6.8, ha="center", va="top")
    # guide column tying the physical contact to its pulled-back gap value
    axGap.axvline(t1, color=MUTE, lw=0.8, ls=(0, (1, 2)), zorder=1)
    axGap.plot(t1, gbar[np.argmin(np.abs(t - t1))], "o", ms=4.5, mfc="white", mec=INK, mew=1.1, zorder=5)
    axGap.text(0.015, 1.16, r"(c) pulled-back gap  $\bar g(\theta_A)=g_N^{B}(\varphi_A(\theta_A))$",
               transform=axGap.transAxes, fontsize=8.0, fontweight="bold", va="top")
    axGap.set_xlabel(r"$\theta_A$", fontsize=9, labelpad=1.5)
    axGap.set_xticks([]); axGap.set_yticks([])
    axGap.grid(False)
    for sp in ("top", "right"):
        axGap.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axGap.spines[sp].set_color(MUTE); axGap.spines[sp].set_linewidth(0.9)

    # ============================ FOOTER punchline ============================
    fig.text(0.5, 0.022,
             r"$\delta W_c=\int_{\Gamma_c}(t_N\,\delta g_N+\mathbf{t}_T\!\cdot\delta\mathbf{g}_T)\,dA$"
             r"   depends on the geometry only through $(g_N,\mathbf{n})$",
             ha="center", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.42", fc=LIGHT, ec=MUTE, lw=0.8))

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "transition_map_composite_pub.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
