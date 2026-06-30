"""Figure 3: how the Fourier-feature encoding works, against a plain coordinate MLP.

Two panels carrying one idea, so a reviewer reads it at a glance:
 (a) the encoding -- a plain network sees the raw coordinate x; a Fourier network first lifts x
     through a fixed bank of sinusoids gamma(x)=[cos(2*pi*B x), sin(2*pi*B x)] before the same MLP.
 (b) the effect -- a sharp one-dimensional target (a smooth carrier with a localised high-frequency
     burst): the plain (low-pass) network smooths the sharp feature, the Fourier-feature network
     resolves it.

The earlier neural-tangent-kernel-spectrum panel was removed: it is the most abstract view, the
spectral-bias mechanism it drew is stated in Remark (NTK) in the text, and the bias is *measured*
directly on the real granite joint in the dedicated spectral-bias figure.

Run:  <venv>/bin/python postprocessing/plot_fourier_mechanism.py
Out:  figures/fourier_mechanism_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
PLAIN = "#3E6E9E"; FOUR = "#BE5536"; INK = "#222222"; MUTE = "#9a958a"


def target(x):
    # smooth low-frequency carrier + a localised high-frequency burst (the "sharp" detail)
    return (0.55 * np.sin(2 * np.pi * 1.3 * x)
            + 0.42 * np.exp(-((x - 0.52) / 0.10) ** 2) * np.sin(2 * np.pi * 9.0 * x))


def krr_fit(xtr, ytr, xte, kernel, lam=1e-4):
    Ktr = kernel(xtr[:, None], xtr[None, :])
    Kte = kernel(xte[:, None], xtr[None, :])
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(xtr)), ytr)
    return Kte @ alpha


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=9.5)

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.40))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.22)

    # ================== (a) the encoding: a fixed Fourier bank before the same MLP ==================
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a) the encoding", fontsize=9.5, fontweight="bold", loc="left")

    def box(x, y, w, h, text, fc, ec, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x0, y0, x1, y1, c=INK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.2, color=c))

    # plain row (top): x -> MLP -> h(x)
    ax.text(0.3, 9.15, "plain coordinate network", fontsize=8.0, color=PLAIN, fontweight="bold")
    box(0.6, 7.5, 1.4, 1.2, r"$x$", "white", PLAIN); arrow(2.1, 8.1, 3.3, 8.1)
    box(3.3, 7.5, 2.4, 1.2, "MLP", "#eef2f6", PLAIN); arrow(5.8, 8.1, 7.0, 8.1)
    box(7.0, 7.5, 1.8, 1.2, r"$h(x)$", "white", PLAIN)

    # Fourier row (bottom): x -> gamma(x) bank -> same MLP -> h
    ax.text(0.3, 5.55, "Fourier-feature network", fontsize=8.0, color=FOUR, fontweight="bold")
    box(0.4, 3.8, 1.1, 1.2, r"$x$", "white", FOUR); arrow(1.6, 4.4, 2.5, 4.4)
    box(2.5, 3.55, 2.8, 1.7, r"$\gamma(x)=$" + "\n" + r"$[\cos,\sin](2\pi B x)$", "#fbeee8", FOUR, 7.2)
    arrow(5.4, 4.4, 6.3, 4.4)
    box(6.3, 3.8, 1.6, 1.2, "MLP", "#fbeee8", FOUR); arrow(8.0, 4.4, 8.8, 4.4)
    box(8.8, 3.8, 1.0, 1.2, r"$h$", "white", FOUR, 8.5)
    # the fixed sinusoidal bank under gamma, at increasing frequency
    xb = np.linspace(0, 1, 200)
    for (b, yoff, c) in [(1, 2.55, "#d6b3a6"), (3, 1.65, "#c98a76"), (7, 0.75, FOUR)]:
        ax.plot(2.55 + 2.7 * xb, yoff + 0.34 * np.sin(2 * np.pi * b * xb), color=c, lw=1.3)
    ax.annotate("", xy=(5.45, 2.95), xytext=(5.45, 0.55), arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.9))
    ax.text(5.72, 1.75, "higher freq.", fontsize=6.8, color=MUTE, rotation=90, va="center")
    ax.text(3.95, -0.15, r"fixed sinusoidal bank $B$", fontsize=7.5, color=FOUR, ha="center")

    # ================== (b) the effect: plain smooths a sharp feature; the bank resolves it ==================
    ax = fig.add_subplot(gs[0, 1])
    xte = np.linspace(0, 1, 600)
    rng = np.random.RandomState(0)
    xtr = np.sort(rng.uniform(0, 1, 60)); ytr = target(xtr)
    # plain MLP ~ low-pass (broad) kernel; Fourier features ~ band-limited cosine kernel to a cutoff
    ell = 0.16
    k_plain = lambda a, b: np.exp(-((a - b) / ell) ** 2)
    Bbank = np.linspace(0.0, 11.0, 40)
    k_four = lambda a, b: np.mean(np.cos(2 * np.pi * Bbank[None, None, :] * (a[..., None] - b[..., None])), axis=-1)
    fp = krr_fit(xtr, ytr, xte, k_plain, lam=1e-3)
    ff = krr_fit(xtr, ytr, xte, k_four, lam=1e-5)
    ax.plot(xte, target(xte), color=INK, lw=2.6, alpha=0.85, label="target (sharp feature)")
    ax.plot(xte, fp, color=PLAIN, lw=2.0, ls=(0, (5, 2)), label="plain: smooths it")
    ax.plot(xte, ff, color=FOUR, lw=1.8, label="Fourier features: resolves it")
    ax.plot(xtr, ytr, "o", ms=2.4, color="0.45", alpha=0.55)
    ax.set_title("(b) the effect", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel(r"coordinate $x$"); ax.set_ylabel(r"learned $h(x)$")
    ax.set_xticks([0, 0.5, 1])
    ax.legend(loc="lower center", fontsize=7.8, framealpha=0.95, handlelength=1.7)
    ax.set_ylim(-1.12, 0.82)
    ax.annotate("the sharp\nhigh-frequency\nfeature", xy=(0.52, -0.82), xytext=(0.80, -0.30),
                fontsize=7.0, color=MUTE, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fourier_mechanism_pub.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
