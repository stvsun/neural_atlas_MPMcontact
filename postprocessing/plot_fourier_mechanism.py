"""Figure 3: how the Fourier-feature encoding works, as opposed to a plain coordinate MLP.

Three illustrative panels:
 (a) the encoding -- a plain network sees the raw coordinate x; a Fourier network sees x lifted
     through a fixed bank of sinusoids gamma(x)=[cos(2*pi*B x), sin(2*pi*B x)] before the MLP.
 (b) the effect -- kernel-regression fit of a sharp 1-D target: the plain (low-pass) kernel
     smooths the sharp feature, the Fourier kernel resolves it.
 (c) the mechanism -- the neural-tangent-kernel spectrum: a plain MLP is Laplace-like and
     low-passes high frequencies (spectral bias); the Fourier bank flattens the spectrum to a
     tunable cutoff so the same frequencies become learnable.

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

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)

    # ============ (a) the encoding pipeline ============
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a)", fontsize=9.5, fontweight="bold", loc="left")

    def box(x, y, w, h, text, fc, ec, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.1, color=INK))
    # plain row
    ax.text(0.2, 8.7, "plain", fontsize=8.5, color=PLAIN, fontweight="bold")
    box(1.4, 7.7, 1.5, 1.2, r"$x$", "white", PLAIN)
    arrow(3.0, 8.3, 4.4, 8.3)
    box(4.4, 7.7, 2.3, 1.2, r"MLP", "#eef2f6", PLAIN)
    arrow(6.8, 8.3, 8.1, 8.3)
    box(8.1, 7.7, 1.6, 1.2, r"$h(x)$", "white", PLAIN)
    # Fourier row
    ax.text(0.2, 5.6, "Fourier", fontsize=8.5, color=FOUR, fontweight="bold")
    box(1.4, 4.4, 1.2, 1.2, r"$x$", "white", FOUR)
    arrow(2.7, 5.0, 3.7, 5.0)
    box(3.7, 4.2, 2.5, 1.6, r"$\gamma(x)$" + "\n" + r"$[\cos,\sin](2\pi Bx)$", "#fbeee8", FOUR, 7.2)
    arrow(6.3, 5.0, 6.8, 5.0)
    box(6.8, 4.4, 1.2, 1.2, r"MLP", "#fbeee8", FOUR)
    arrow(8.1, 5.0, 8.7, 5.0)
    box(8.7, 4.4, 1.2, 1.2, r"$h(x)$", "white", FOUR, 8.0)
    # the sinusoidal bank, drawn under the gamma box
    xb = np.linspace(0, 1, 200)
    for j, (b, yoff, c) in enumerate([(1, 3.0, "#caa"), (3, 2.0, "#c77"), (7, 1.0, FOUR)]):
        ax.plot(3.5 + 3.0 * xb, yoff + 0.42 * np.sin(2 * np.pi * b * xb), color=c, lw=1.1)
    ax.text(6.7, 2.0, r"increasing", fontsize=7, color=MUTE, rotation=90, va="center")
    ax.annotate("", xy=(6.55, 3.4), xytext=(6.55, 0.7),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.9))
    ax.text(5.0, 0.05, "fixed sinusoidal bank $B$", fontsize=7.5, color=FOUR, ha="center")

    # ============ (b) the effect: a sharp 1-D fit ============
    ax = fig.add_subplot(gs[0, 1])
    xte = np.linspace(0, 1, 600)
    rng = np.random.RandomState(0)
    xtr = np.sort(rng.uniform(0, 1, 60)); ytr = target(xtr)
    # plain MLP ~ low-pass (broad) kernel; Fourier features ~ band-limited cosine kernel to cutoff
    ell = 0.16
    k_plain = lambda a, b: np.exp(-((a - b) / ell) ** 2)
    Bbank = np.linspace(0.0, 11.0, 40)
    k_four = lambda a, b: np.mean(np.cos(2 * np.pi * Bbank[None, None, :] * (a[..., None] - b[..., None])), axis=-1)
    fp = krr_fit(xtr, ytr, xte, k_plain, lam=1e-3)
    ff = krr_fit(xtr, ytr, xte, k_four, lam=1e-5)
    ax.plot(xte, target(xte), color=INK, lw=2.4, alpha=0.85, label="target")
    ax.plot(xte, fp, color=PLAIN, lw=1.8, ls=(0, (5, 2)), label="plain (low-pass)")
    ax.plot(xte, ff, color=FOUR, lw=1.6, label="Fourier features")
    ax.plot(xtr, ytr, "o", ms=2.4, color="0.45", alpha=0.6)
    ax.set_title("(b)", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$h(x)$")
    ax.set_xticks([0, 0.5, 1]); ax.legend(loc="lower center", fontsize=7.2, ncol=1,
                                          framealpha=0.95, handlelength=1.4)
    ax.set_ylim(-1.08, 0.78)
    ax.annotate("plain smooths\nthe sharp detail", xy=(0.50, -0.80), xytext=(0.74, 0.34),
                fontsize=7.2, color=PLAIN, ha="center",
                arrowprops=dict(arrowstyle="->", color=PLAIN, lw=0.8))

    # ============ (c) the mechanism: NTK spectrum ============
    ax = fig.add_subplot(gs[0, 2])
    k = np.logspace(0, 2.3, 200)
    plain_spec = k ** (-2.0)                 # Laplace-like NTK, lambda(k) ~ |k|^-(d+1), d=1
    cutoff = 30.0
    four_spec = np.where(k <= cutoff, 0.55, 0.55 * (cutoff / k) ** 4)
    ax.loglog(k, plain_spec / plain_spec[0], color=PLAIN, lw=2.0, label="plain MLP NTK")
    ax.loglog(k, four_spec / four_spec[0], color=FOUR, lw=2.0, label="Fourier-feature kernel")
    ax.axvline(cutoff, color=FOUR, lw=0.8, ls=":")
    ax.text(cutoff * 1.05, 1.3e-3, r"cutoff $2\pi\sigma$", fontsize=7.2, color=FOUR, rotation=90,
            va="bottom")
    ax.fill_between(k, plain_spec / plain_spec[0], 1e-5, where=(k > 12),
                    color=PLAIN, alpha=0.07)
    ax.text(40, 6e-3, "high frequencies\nunlearned\n(spectral bias)", fontsize=7.0,
            color=PLAIN, ha="center")
    ax.set_title("(c)", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel(r"frequency $k$"); ax.set_ylabel("kernel power (norm.)")
    ax.set_ylim(1e-4, 2.0); ax.legend(loc="upper right", fontsize=7.2, framealpha=0.95)

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
