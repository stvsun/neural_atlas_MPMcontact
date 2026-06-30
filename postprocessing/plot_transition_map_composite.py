#!/usr/bin/env python3
"""Figure 2: the composite transition map -- contact detection as chart composition, not a search.

A two-panel schematic carrying ONE idea, so a reviewer reads it at a glance.
  (a) PHYSICAL coordinates: two bodies in near-contact.  The contact point x=phi_A(theta_A) on
      partial Omega_A is carried to its mate phi_B(psi_B) on partial Omega_B by inverting B's chart
      (phi_B^{-1}), fixing the signed normal gap g_N along n.  The iterative closest-point search a
      level set / node-to-segment scheme would need is drawn faint -- it is the step the composition
      replaces.
  (b) CHART coordinates: the commuting triangle theta_A --phi_A--> x --phi_B^{-1}--> psi_B, whose
      hypotenuse is the inter-body transition map tau_AB = phi_B^{-1} o phi_A = pi_B o phi_A.  One
      chart evaluation and one chart inversion give the mate; no closest-point iteration.

The two earlier panels (the reference-strip reparametrisation and the pulled-back gap / multi-arc
active set) were removed: they diluted the message, and the multi-arc point has its own figure.

Run:  <venv>/bin/python postprocessing/plot_transition_map_composite.py
Out:  figures/transition_map_composite_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")

CORAL = "#BE5536"   # body A / chart A
BLUE  = "#3E6E9E"   # body B / chart B / mate
INK   = "#222222"   # kinematic primitives: x, gap, tau_AB
MUTE  = "#9a958a"   # the search we replace; secondary guides
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
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=8.5)

    fig = plt.figure(figsize=(DOUBLE_COL_W, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.06,
                          left=0.015, right=0.985, top=0.90, bottom=0.04)
    axP = fig.add_subplot(gs[0, 0])
    axD = fig.add_subplot(gs[0, 1])
    for ax in (axP, axD):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axP.set_aspect("equal")

    # ==================== (a) PHYSICAL: the mate is found by inverting B's chart ====================
    xs = np.linspace(-0.02, 1.02, 400)
    axP.fill_between(xs, yA(xs), 1.18, color=CORAL, alpha=0.12, lw=0, zorder=1)
    axP.plot(xs, yA(xs), color=CORAL, lw=2.3, zorder=3)
    axP.fill_between(xs, -0.18, yB(xs), color=BLUE, alpha=0.12, lw=0, zorder=1)
    axP.plot(xs, yB(xs), color=BLUE, lw=2.3, zorder=3)
    axP.text(0.06, 0.95, "body $A$", color=CORAL, fontsize=9, fontweight="bold", va="top")
    axP.text(0.06, 0.05, "body $B$", color=BLUE, fontsize=9, fontweight="bold", va="bottom")
    axP.text(0.93, yA(0.93) + 0.05, r"$\partial\Omega_A$", color=CORAL, fontsize=9, ha="center")
    axP.text(0.93, yB(0.93) - 0.06, r"$\partial\Omega_B$", color=BLUE, fontsize=9, ha="center")

    xpt = np.array([XC, yA(XC)]); n = np.array([0.0, -1.0]); foot = np.array([XC, yB(XC)])
    ymid = 0.5 * (yA(XC) + yB(XC))
    # the composite step in physical space: x -> its mate on B by inverting B's chart
    _arrow(axP, xpt, foot, INK, lw=1.8, ms=12, zorder=6)
    axP.text(XC - 0.02, ymid + 0.015, r"$\varphi_B^{-1}$", color=INK, fontsize=9.5, ha="right", va="center")
    # signed normal gap (measured along the surface normal n), offset to the right of the column
    off = np.array([0.085, 0.0])
    _arrow(axP, xpt + off, foot + off, INK, lw=1.2, ms=9, style="<|-|>", zorder=5)
    axP.text(XC + 0.115, ymid, r"$g_N$ along $\mathbf{n}$", color=INK, fontsize=9, ha="left", va="center")
    # faint closest-point search -- the iterative step the composition replaces
    for sx in (0.325, 0.360):
        _arrow(axP, xpt, np.array([sx, yB(sx)]), MUTE, lw=0.8, ms=6, ls=(0, (3, 2)), alpha=0.85, zorder=4)
        axP.plot(sx, yB(sx) + 0.012, "o", ms=2.6, mfc="none", mec=MUTE, mew=0.8, zorder=4)
    axP.text(0.115, ymid - 0.05, "closest-point search\n(iterative; replaced)", color=MUTE, fontsize=6.6,
             style="italic", ha="center", va="center", linespacing=0.95)
    # markers + labels
    axP.plot(*xpt, "o", ms=6.5, color=INK, zorder=7)
    axP.plot(*foot, "s", ms=6.5, color=BLUE, mec="white", mew=0.6, zorder=7)
    axP.text(XC - 0.05, yA(XC) + 0.01, r"$\mathbf{x}=\varphi_A(\theta_A)$", color=INK, fontsize=8.5,
             ha="right", va="center")
    axP.text(XC + 0.015, yB(XC) - 0.05, r"$\varphi_B(\psi_B)$", color=BLUE, fontsize=8.5, ha="left", va="top")
    axP.text(0.02, 1.10, "(a) physical coordinates", fontsize=9.5, fontweight="bold", va="top")

    # ============ (b) CHART coordinates: the mate is a composition of two charts (no search) ============
    axD.text(0.02, 0.995, "(b) chart coordinates: the transition map", fontsize=9.5,
             fontweight="bold", va="top")
    thA = np.array([0.17, 0.70]); psB = np.array([0.83, 0.70]); xx = np.array([0.50, 0.30])
    # composite path:  theta_A --phi_A--> x --phi_B^{-1}--> psi_B
    _arrow(axD, thA + np.array([0.035, -0.03]), xx + np.array([-0.055, 0.05]), CORAL, lw=1.6, ms=11, zorder=5)
    _arrow(axD, xx + np.array([0.055, 0.05]), psB + np.array([-0.035, -0.03]), BLUE, lw=1.6, ms=11, zorder=5)
    # the shortcut tau_AB across the top
    _arrow(axD, thA + np.array([0.075, 0.015]), psB + np.array([-0.075, 0.015]), INK, lw=2.1, ms=13,
           rad=-0.16, zorder=5)
    # nodes
    axD.plot(*thA, "^", ms=9, color=CORAL, mec="white", mew=0.5, zorder=7)
    axD.plot(*psB, "^", ms=9, color=BLUE, mec="white", mew=0.5, zorder=7)
    axD.plot(*xx, "o", ms=7, color=INK, zorder=7)
    axD.text(thA[0] - 0.015, thA[1] + 0.035, r"$\theta_A\in\Theta_A$", color=CORAL, fontsize=8.8,
             ha="left", va="bottom")
    axD.text(psB[0] + 0.015, psB[1] + 0.035, r"$\psi_B\in\Theta_B$", color=BLUE, fontsize=8.8,
             ha="right", va="bottom")
    axD.text(xx[0], xx[1] - 0.055, r"$\mathbf{x}$ (physical)", color=INK, fontsize=8.5, ha="center", va="top")
    # arrow labels
    axD.text(0.265, 0.50, r"$\varphi_A$", color=CORAL, fontsize=9.5, ha="right", va="center")
    axD.text(0.735, 0.50, r"$\varphi_B^{-1}$", color=BLUE, fontsize=9.5, ha="left", va="center")
    axD.text(0.50, 0.885, r"$\tau_{AB}$", color=INK, fontsize=11, ha="center", va="bottom", fontweight="bold")
    # the statement
    axD.text(0.5, 0.045, r"$\tau_{AB}=\varphi_B^{-1}\circ\varphi_A=\pi_B\circ\varphi_A$"
             "\none chart evaluation + one inversion: no search",
             ha="center", va="bottom", fontsize=7.7, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="0.7", lw=0.8))

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "transition_map_composite_pub.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
