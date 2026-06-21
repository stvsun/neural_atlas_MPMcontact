"""Illustrations for the Fourier-feature chart TRAINING manual (docs/fourier_feature_chart_training.md).

This trains the repository's OWN height chart (`solvers/contact/profile_chart_2d.py::NeuralHeight1D`)
on the REAL Inada-granite rock-joint profile (`data/inada_joint/inada_rough_profile.npz`), twice — once
with Fourier features (`plain=False`) and once without (`plain=True`, the spectral-bias baseline) — using
the exact optimizer / schedule / loss of `fit_height_chart`, and records the RMSE history so the training
dynamics can be drawn.  The result is a genuine, reproducible demonstration that Fourier features are what
let the chart resolve the asperities (not the chart parametrization alone).

Outputs (figures/):
    fourier_training_pipeline_pub.png  — the training pipeline x -> features -> MLP -> h, + the 3 banks
    fourier_training_curves_pub.png    — RMSE vs iteration (Fourier vs plain) + final reconstructions

Run:  python3 postprocessing/plot_fourier_training.py   (trains on CPU/float64; ~1-2 min)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from solvers.contact.profile_chart_2d import NeuralHeight1D

FIG = os.path.join(_ROOT, "figures")
DATA = os.path.join(_ROOT, "data", "inada_joint", "inada_rough_profile.npz")

BG = "#ffffff"
INK = "#1b2430"
MUTE = "#6b7686"
FOUR = "#1f9e6e"     # Fourier (chart) green
PLAIN = "#b0485f"    # plain coordinate-MLP (spectral bias) red
ACC = "#2f6fb0"      # accent blue
GEOM = "#2f6fb0"
GAUSS = "#d2691e"
HARM = "#7a3fb0"
GRID = "#dfe4ea"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTE, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "cm",
})


# ====================================================================================================
# training: replicate fit_height_chart EXACTLY but record the RMSE history
# ====================================================================================================
def _train_record(x, z, plain, f_max, n_freq, iters, lr=2e-3, log_every=25):
    """Mirror solvers/contact/profile_chart_2d.fit_height_chart exactly, logging the RECONSTRUCTION
    RMSE (% of RMS) per log_every.  A measured rock-joint profile is a deterministic surface to
    *represent* (the production metric, verification manual §11.9), not a distribution to predict —
    so the chart is fit to, and scored on, the measured samples.  ``f_max`` is kept below the sampling
    Nyquist so the band-limited chart is well-posed (not over-parametrized)."""
    torch.manual_seed(0)
    xt = torch.from_numpy(np.asarray(x, float)); zt = torch.from_numpy(np.asarray(z, float))
    rms = float(zt.std())
    chart = NeuralHeight1D(float(xt.min()), float(xt.max()), f_max=f_max, n_freq=n_freq,
                           width=128, depth=4, base=float(zt.mean()), plain=plain).double()
    opt = torch.optim.Adam(chart.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    n = xt.shape[0]; batch = min(4096, n)
    hist_it, hist_rmse = [], []
    for it in range(iters):
        idx = torch.randint(0, n, (batch,))
        loss = torch.mean((chart(xt[idx]) - zt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % log_every == 0 or it == iters - 1:
            with torch.no_grad():
                r = float(torch.sqrt(torch.mean((chart(xt) - zt) ** 2)))
            hist_it.append(it); hist_rmse.append(100.0 * r / rms)
    with torch.no_grad():
        recon = chart(xt).numpy()
    return chart, np.array(hist_it), np.array(hist_rmse), recon, rms


def _mean_angle(x, z):
    sl = np.gradient(z, x)
    return float(np.rad2deg(np.arctan(np.abs(sl))).mean())


# ====================================================================================================
# Figure 1 — the training pipeline + the three Fourier banks
# ====================================================================================================
def _box(ax, xy, w, h, text, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.6, mutation_aspect=1.0, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3)


def fig_pipeline():
    fig = plt.figure(figsize=(13.2, 5.6))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)

    # --- (a) the pipeline flow ---
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y0, h = 0.34, 0.34
    boxes = [
        (0.005, "$x$\n(profile coord.)", "#eef2f6", MUTE, 0.105),
        (0.135, r"normalize" + "\n" + r"$\tilde x\in[-1,1]$", "#eef2f6", MUTE, 0.135),
        (0.300, r"$\gamma(x)=[\cos 2\pi B\tilde x,\ \sin 2\pi B\tilde x]$" + "\n(fixed Fourier bank $B$)",
         "#e7f5ef", FOUR, 0.250),
        (0.580, r"MLP$_\theta$" + "\n(tanh, width 128, depth 4)", "#eef2f6", ACC, 0.190),
        (0.800, r"$h_\theta(x)=\mathrm{base}+\Delta$", "#e7f5ef", FOUR, 0.195),
    ]
    centers = []
    for x0, txt, fc, ec, w in boxes:
        _box(ax, (x0, y0), w, h, txt, fc, ec, fs=10.5)
        centers.append((x0, x0 + w))
    for (l, r), (l2, r2) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((r, y0 + h / 2), (l2, y0 + h / 2), arrowstyle="-|>",
                                     mutation_scale=14, color=INK, lw=1.5))
    ax.text(0.5, 0.93, "Training pipeline of a Fourier-feature chart (MSE regression onto the surface "
            "samples)", ha="center", fontsize=12.5, color=INK)
    ax.text(0.425, 0.05, r"only $\theta$ (and base) are trained; the bank $B$ is FIXED — it injects the "
            r"high frequencies the MLP cannot learn on its own", ha="center", fontsize=9.5, color=MUTE)

    # --- (b) the three frequency banks on a shared log axis ---
    ax = fig.add_subplot(gs[1, 0])
    geom = np.geomspace(0.5, 1500.0, 192)                              # NeuralHeight1D bank
    g = torch.Generator().manual_seed(0)
    B2 = (torch.randn(256, 2, generator=g, dtype=torch.float64) * 8.0).numpy()
    gauss = np.linalg.norm(B2, axis=1)                                # |B| of the 2-D Gaussian bank
    harm = np.arange(1, 17)                                           # NeuralRho2D harmonics
    ax.scatter(geom, np.full_like(geom, 3.0), s=14, color=GEOM, marker="|",
               label="deterministic geometric (1-D height, $K{=}192$, $f_{\\max}{=}1500$)")
    ax.scatter(gauss, np.full_like(gauss, 2.0), s=12, color=GAUSS, alpha=0.55, marker="o",
               label="Gaussian random (2-D surface, $K{=}256$, $\\sigma{=}8$)")
    ax.scatter(harm, np.full_like(harm, 1.0), s=42, color=HARM, marker="|",
               label="harmonic (2-D radial chart, $k{=}1..16$)")
    ax.set_xscale("log"); ax.set_xlim(0.3, 2500); ax.set_ylim(0.4, 3.7)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["harmonic", "Gaussian", "geometric"], fontsize=10)
    ax.set_xlabel("frequency  (cycles across the normalized domain, log scale)")
    ax.set_title("The three Fourier banks — each tiles a band of frequencies, handing them to the MLP "
                 "directly", fontsize=11.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.4, framealpha=0.95, ncol=1)
    for s in ax.spines.values():
        s.set_color(MUTE)

    out = os.path.join(FIG, "fourier_training_pipeline_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ====================================================================================================
# Figure 2 — real training: curves + reconstructions (Fourier vs plain) on the real Inada profile
# ====================================================================================================
def fig_training(iters=2500, n_freq=128, f_max=700.0):
    d = np.load(DATA)
    x = d["x_mm"].astype(float)
    z = d["footwall_mm"].astype(float)
    z = z - z.mean()
    print(f"  Inada profile: {x.size} pts, span {x.max()-x.min():.1f} mm, RMS {z.std():.3f} mm "
          f"(Nyquist {x.size//2} cycles; f_max={f_max:.0f})")

    print("  training Fourier-feature chart (plain=False) ...")
    chF, itF, rmF, reconF, rms = _train_record(x, z, False, f_max, n_freq, iters)
    print(f"    reconstruction RMSE = {rmF[-1]:.3f}% of RMS  ({rmF[-1]/100*rms*1e3:.1f} um)")
    print("  training plain coordinate MLP (plain=True, spectral-bias baseline) ...")
    chP, itP, rmP, reconP, _ = _train_record(x, z, True, f_max, n_freq, iters)
    print(f"    reconstruction RMSE = {rmP[-1]:.3f}% of RMS  ({rmP[-1]/100*rms*1e3:.1f} um)")

    iA, iD = _mean_angle(x, z), _mean_angle(x, reconF)
    iP = _mean_angle(x, reconP)

    fig = plt.figure(figsize=(13.4, 5.2))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.22)

    # (a) training curves
    ax = fig.add_subplot(gs[0, 0])
    umP = rmP[-1] / 100 * rms * 1e3
    ax.plot(itF, rmF, color=FOUR, lw=2.2, label=f"Fourier features (final {rmF[-1]:.2g}% of RMS)")
    ax.plot(itP, rmP, color=PLAIN, lw=2.2,
            label=f"plain coord. MLP (final {rmP[-1]:.1f}% = {umP:.0f} $\\mu$m)")
    ax.axhline(rmP[-1], color=PLAIN, lw=0.8, ls=(0, (3, 3)), alpha=0.6)
    ax.set_yscale("log"); ax.set_xlabel("training iteration")
    ax.set_ylabel("reconstruction RMSE  (% of profile RMS)")
    ax.set_title("(a) reconstruction of the real Inada joint vs iteration\n(identical "
                 "width/depth/optimizer; only the input encoding differs)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.grid(True, which="both", color=GRID, lw=0.55)
    ax.annotate("spectral-bias floor:\nthe plain MLP cannot\nresolve the asperities",
                (0.5, 0.62), xycoords="axes fraction", fontsize=9.2, color=PLAIN, ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff4f0", ec=PLAIN, lw=0.9))
    for s in ax.spines.values():
        s.set_color(MUTE)

    # (b) reconstruction (a representative window)
    ax = fig.add_subplot(gs[0, 1])
    x0 = x.min() + 0.32 * (x.max() - x.min())
    m = (x >= x0) & (x <= x0 + 12.0)
    ax.plot(x[m], z[m], color=INK, lw=2.6, alpha=0.85, label=f"real Inada profile ($\\bar i$={iA:.0f}°)")
    ax.plot(x[m], reconF[m], color=FOUR, lw=1.6, label=f"Fourier chart ($\\bar i$={iD:.0f}°)")
    ax.plot(x[m], reconP[m], color=PLAIN, lw=2.0, ls=(0, (4, 2)),
            label=f"plain MLP ($\\bar i$={iP:.0f}°, smoothed)")
    ax.set_xlabel("position  $x$ (mm)"); ax.set_ylabel("height  $h$ (mm)")
    ax.set_title("(b) learned surface vs the real joint — the Fourier chart tracks the asperities,\n"
                 "the plain MLP low-passes them (under-resolved slopes → wrong friction angle)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, color=GRID, lw=0.55)
    for s in ax.spines.values():
        s.set_color(MUTE)

    fig.suptitle("Training a Fourier-feature height chart on a real rock joint "
                 "(solvers/contact/profile_chart_2d.py::NeuralHeight1D)", fontsize=13, y=1.03)
    out = os.path.join(FIG, "fourier_training_curves_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out, dict(rmse_fourier=float(rmF[-1]), rmse_plain=float(rmP[-1]),
                     ang_true=iA, ang_fourier=iD, ang_plain=iP, rms_mm=float(rms))


# ====================================================================================================
# Figure 3 — the Fourier-feature network architecture
# ====================================================================================================
def _layer(ax, x, ys, color, r=0.018, ec=None, lw=1.4, z=4):
    pts = []
    for y in ys:
        ax.add_patch(plt.Circle((x, y), r, fc=color, ec=ec or color, lw=lw, zorder=z))
        pts.append((x, y))
    return pts


def _connect(ax, A, B, color="#c7ccd4", lw=0.45, alpha=0.7):
    for (xa, ya) in A:
        for (xb, yb) in B:
            ax.plot([xa, xb], [ya, yb], color=color, lw=lw, alpha=alpha, zorder=1)


def fig_architecture():
    """Layered diagram of the Fourier-feature chart network (NeuralHeight1D and its siblings)."""
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def col(n, lo=0.30, hi=0.74):
        if n == 1:
            return [0.52]
        return list(np.linspace(lo, hi, n))

    # --- input + normalize ---
    inp = _layer(ax, 0.045, [0.52], "#9aa4b2", r=0.022, ec=INK)
    ax.text(0.045, 0.40, "input\ncoord. $x$", ha="center", va="top", fontsize=10, color=INK)
    ax.add_patch(FancyBboxPatch((0.105, 0.46), 0.085, 0.12, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc="#eef2f6", ec=MUTE, lw=1.4, zorder=3))
    ax.text(0.1475, 0.52, r"normalize" + "\n" + r"$\tilde x\in[-1,1]$", ha="center", va="center",
            fontsize=9.2, color=INK, zorder=4)

    # --- Fourier feature layer (cos block, dots, sin block) ---
    xF = 0.305
    cos_y = [0.69, 0.645, 0.60]
    sin_y = [0.44, 0.395, 0.35]
    feat = _layer(ax, xF, cos_y, GAUSS, r=0.016, ec=GAUSS)
    feat += _layer(ax, xF, sin_y, GAUSS, r=0.016, ec=GAUSS)
    ax.text(xF, 0.525, r"$\vdots$", ha="center", va="center", fontsize=15, color=GAUSS)
    ax.text(xF - 0.052, 0.645, r"$\cos$", ha="right", va="center", fontsize=9, color=GAUSS)
    ax.text(xF - 0.052, 0.395, r"$\sin$", ha="right", va="center", fontsize=9, color=GAUSS)
    ax.plot([0.19, xF - 0.018], [0.52, 0.52], color="#c7ccd4", lw=0.8, zorder=1)
    ax.text(xF, 0.80, "Fourier features\n" + r"$\gamma(x)=[\cos 2\pi B\tilde x,\ \sin 2\pi B\tilde x]$"
            + "\n$2K=384$  (bank $B$, $K{=}192$)", ha="center", va="bottom", fontsize=9.6, color=INK)

    # --- 4 hidden layers (tanh, width 128) ---
    hx = [0.45, 0.555, 0.66, 0.765]
    prev = feat
    hidden_cols = []
    for i, x in enumerate(hx):
        ys = col(6)
        h = _layer(ax, x, ys, ACC, r=0.015, ec=ACC)
        ax.text(x, 0.27, r"$\vdots$", ha="center", va="center", fontsize=14, color=ACC)
        _connect(ax, prev, h)
        prev = h; hidden_cols.append(h)
    ax.text(0.6075, 0.80, r"$\mathrm{MLP}_\theta$:  4 hidden layers, width 128, $\tanh$",
            ha="center", va="bottom", fontsize=10.5, color=ACC)

    # --- output + base ---
    out = _layer(ax, 0.86, [0.52], FOUR, r=0.018, ec=FOUR)
    _connect(ax, prev, out)
    ax.add_patch(FancyArrowPatch((0.878, 0.52), (0.93, 0.52), arrowstyle="-|>", mutation_scale=13,
                                 color=INK, lw=1.5, zorder=4))
    ax.text(0.905, 0.565, r"$+\,\mathrm{base}$", ha="center", fontsize=9.5, color=INK)
    ax.text(0.965, 0.52, r"$h_\theta(x)$", ha="center", va="center", fontsize=13, color=FOUR)
    ax.text(0.86, 0.40, r"$\Delta$", ha="center", va="top", fontsize=11, color=FOUR)

    # --- FIXED vs TRAINED bands ---
    ax.add_patch(FancyBboxPatch((0.245, 0.14), 0.12, 0.045, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#fdeee2", ec=GAUSS, lw=1.3, zorder=3))
    ax.text(0.305, 0.1625, "FIXED  (bank $B$ not trained)", ha="center", va="center", fontsize=9, color=GAUSS)
    ax.add_patch(FancyBboxPatch((0.40, 0.14), 0.50, 0.045, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#e9f0f8", ec=ACC, lw=1.3, zorder=3))
    ax.text(0.65, 0.1625, r"TRAINED  (weights $\theta$ + scalar base; Adam + cosine, MSE)",
            ha="center", va="center", fontsize=9, color=ACC)

    ax.text(0.5, 0.055,
            r"$h_\theta(x)=\mathrm{base}+\mathrm{MLP}_\theta(\gamma(x)),\quad$"
            r"$\mathrm{MLP}=\mathrm{Lin}(384\!\to\!128)\,\tanh\,[\mathrm{Lin}(128\!\to\!128)\,\tanh]^{\times3}"
            r"\,\mathrm{Lin}(128\!\to\!1)$",
            ha="center", va="center", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f6f8fa", ec=MUTE, lw=1.0))
    ax.text(0.5, 0.005, r"(the 2-D radial chart wraps the output in softplus to keep $\rho>0$; "
            r"the 2-D/3-D charts use a Gaussian bank $B\sim\mathcal{N}(0,\sigma^2)$ instead of the geometric one)",
            ha="center", va="center", fontsize=8.6, color=MUTE)

    ax.set_title("Architecture of a Fourier-feature coordinate chart  "
                 "(solvers/contact/profile_chart_2d.py::NeuralHeight1D)", fontsize=12.5, color=INK, y=1.0)
    out_path = os.path.join(FIG, "fourier_training_architecture_pub.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(FIG, exist_ok=True)
    print("generating Fourier-feature training illustrations ...")
    print("  ", fig_architecture())
    print("  ", fig_pipeline())
    p, stats = fig_training()
    print("  ", p)
    print("  stats:", {k: round(v, 3) for k, v in stats.items()})
    print("done.")


if __name__ == "__main__":
    main()
