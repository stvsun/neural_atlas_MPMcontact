"""Figure 3: the Fourier-feature chart network.

A single, clean architecture diagram.  The raw coordinate x is lifted by a FIXED bank of sinusoids
gamma(x)=[cos(2*pi*B x), sin(2*pi*B x)] (the Fourier features, coral) and the resulting features feed
a TRAINED multilayer perceptron (grey) that outputs the chart value h(x).  The fixed encoding is what
lets the same MLP represent the sharp, high-frequency detail a plain coordinate network low-passes;
that spectral bias is stated in the NTK remark and measured on the real granite joint in the dedicated
spectral-bias figure.

Run:  <venv>/bin/python postprocessing/plot_fourier_mechanism.py
Out:  figures/fourier_mechanism_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
FOUR = "#BE5536"; INK = "#222222"; MUTE = "#9a958a"; GREY = "#6f6f6f"


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=9.5)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.80, DOUBLE_COL_W * 0.42))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    X_IN, X_FF, X_H1, X_H2, X_OUT = 0.08, 0.33, 0.57, 0.74, 0.93
    TOP, BOT, MID = 0.86, 0.42, 0.64

    def ys(n):
        return [MID] if n == 1 else list(np.linspace(TOP, BOT, n))
    def node(x, y, ec, r=0.020, fc="white", lw=1.3, z=6):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))

    ff_y, h1_y, h2_y = ys(6), ys(5), ys(5)

    # ---- edges (under the nodes) ----
    for yy in ff_y:                                        # fixed Fourier encoding: x -> features
        ax.plot([X_IN, X_FF], [MID, yy], color=FOUR, lw=0.9, alpha=0.55, zorder=1)
    for ya in ff_y:                                        # trained dense MLP edges
        for yb in h1_y:
            ax.plot([X_FF, X_H1], [ya, yb], color="0.74", lw=0.32, alpha=0.55, zorder=1)
    for ya in h1_y:
        for yb in h2_y:
            ax.plot([X_H1, X_H2], [ya, yb], color="0.74", lw=0.32, alpha=0.55, zorder=1)
    for yb in h2_y:
        ax.plot([X_H2, X_OUT], [yb, MID], color="0.74", lw=0.32, alpha=0.55, zorder=1)

    # ---- nodes ----
    node(X_IN, MID, INK, r=0.027); ax.text(X_IN, MID, r"$x$", ha="center", va="center", fontsize=10, zorder=7)
    for yy in ff_y:
        node(X_FF, yy, FOUR, r=0.018, fc="#fbeee8")
    for yy in h1_y:
        node(X_H1, yy, GREY, r=0.016, fc="#efefef")
    for yy in h2_y:
        node(X_H2, yy, GREY, r=0.016, fc="#efefef")
    node(X_OUT, MID, INK, r=0.027); ax.text(X_OUT, MID, r"$h$", ha="center", va="center", fontsize=10, zorder=7)

    # ---- column headers ----
    ax.text(X_IN, TOP + 0.07, "coordinate", ha="center", fontsize=7.4, color=MUTE)
    ax.text(X_FF, TOP + 0.07, r"$\gamma(x)$", ha="center", fontsize=11, color=FOUR, fontweight="bold")
    ax.text(0.5 * (X_H1 + X_H2), TOP + 0.07, "MLP", ha="center", fontsize=10, color=GREY, fontweight="bold")
    ax.text(X_OUT, TOP + 0.07, r"chart value", ha="center", fontsize=7.4, color=MUTE)

    # ---- a compact fixed sinusoid bank, conveying the cos/sin frequencies B ----
    xb = np.linspace(0, 1, 140)
    for (b, yo, c) in [(1.0, 0.37, "#d8b6a9"), (2.4, 0.30, "#c98a76"), (4.5, 0.23, FOUR)]:
        ax.plot(X_FF - 0.055 + 0.11 * xb, yo + 0.020 * np.sin(2 * np.pi * b * xb), color=c, lw=1.2, zorder=2)
    ax.annotate("", xy=(X_FF + 0.085, 0.21), xytext=(X_FF + 0.085, 0.39),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.8))
    ax.text(X_FF + 0.105, 0.30, "freq.", fontsize=6.4, color=MUTE, rotation=90, va="center")

    # ---- stage labels ----
    ax.text(X_FF, 0.045, "fixed Fourier features\n" r"$[\cos,\sin](2\pi B x)$",
            ha="center", va="bottom", fontsize=7.4, color=FOUR)
    ax.text(0.5 * (X_H1 + X_H2), 0.075, "trained weights", ha="center", va="bottom", fontsize=7.4, color=GREY)

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fourier_mechanism_pub.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
