"""Publication figure + shear animation for the rock-joint direct-shear capstone.

Reads runs/rock_joint_capstone/{results.json,surfaces.npz} (from
benchmarks/contact/cv_numerical/rock_joint_capstone.py) and the Inada data, and renders:
  * figures/rock_joint_capstone_pub.png  — the 6-panel hero figure
  * figures/rock_joint_shear.gif          — the 2-panel shear animation

Run:  python3 postprocessing/plot_rock_joint_capstone.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
sys.path.insert(0, _ROOT)
RUN = os.path.join(_ROOT, "runs", "rock_joint_capstone", "results.json")
SURF = os.path.join(_ROOT, "runs", "rock_joint_capstone", "surfaces.npz")
DATA = os.path.join(_ROOT, "data", "inada_joint")
RAW = os.path.join(_ROOT, "downloads", "inada_granite")
FIG = os.path.join(_ROOT, "figures")

CHART_C = "#0072B2"      # blue  — chart / resolved
SDF_C = "#D55E00"        # vermillion — ambient SDF / level set
MEAS_C = "#333333"       # measured surface


def _load_profile(tag):
    d = np.load(os.path.join(DATA, f"inada_{tag}_profile.npz"))
    return d["x_mm"].astype(float), d["footwall_mm"].astype(float)


def _load_map(tag="rough", every=8):
    """Downsampled 2-D height map from the raw CSV (or None if absent)."""
    f = os.path.join(RAW, f"{tag}_footwall.csv")
    if not os.path.exists(f):
        return None
    import pandas as pd
    Z = pd.read_csv(f, header=None).to_numpy(dtype=float)[::every, ::every]
    return Z


def hero_figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.66))
    g = R["geometry"]

    # (A) the real 2-D fractal joint surface --------------------------------------------------------
    Z = _load_map("rough")
    ax = axs[0, 0]
    if Z is not None:
        dx = 0.0234 * 8
        im = ax.imshow(Z, cmap="terrain", extent=[0, Z.shape[1] * dx, 0, Z.shape[0] * dx],
                       aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="height (mm)")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    else:
        x, z = _load_profile("rough"); ax.plot(x, z, color=MEAS_C, lw=0.4); ax.set_xlim(x.min(), x.max())
    ax.set_title("(a) Real Inada rock joint\n$D\\approx$2.2, RMS 1.7 mm, 23.4 $\\mu$m", fontsize=8)

    # (B) Patton method verification ----------------------------------------------------------------
    pat = R["patton"]
    saw = json.load(open(os.path.join(_ROOT, "runs", "rock_joint_sawtooth", "history.json")))
    ux = np.array(saw["history"]["ux"]); mu = np.array(saw["history"]["mu_app"])
    ax = axs[0, 1]
    ax.plot(ux, mu, color=CHART_C, lw=1.4, label="emergent $\\mu_{app}$")
    ax.axhline(pat["patton_mu_pred"], ls="--", color="k", lw=1.1,
               label="Patton $\\tan(\\phi_b+i)$=%.3f" % pat["patton_mu_pred"])
    ax.set_ylim(0, max(mu) * 1.25)
    ax.set_title(f"(b) Method exact: sawtooth\n$\\to$ Patton law ({pat['mu_app_relerr_vs_patton']*100:.2f}% error)",
                 fontsize=8)
    ax.set_xlabel("shear $u_x$ (mm)"); ax.set_ylabel("$\\mu_{app}=\\tau/\\sigma_n$")
    ax.legend(fontsize=6, loc="lower right")

    # (C) chart vs ambient SDF — zoom; shade the asperities the level set erases --------------------
    ax = axs[0, 2]
    if os.path.exists(SURF):
        s = np.load(SURF); xs, zm, hc, hsdf = s["x"], s["z_measured"], s["h_chart"], s["h_sdf"]
        x0 = xs.min() + 0.42 * (xs.max() - xs.min()); win = (xs >= x0) & (xs <= x0 + 7.0)
        xw = xs[win]
        ax.fill_between(xw, hsdf[win], zm[win], color="#bbbbbb", alpha=0.55,
                        label="asperities the SDF erases")
        ax.plot(xw, zm[win], color=MEAS_C, lw=1.5, label="measured", zorder=3)
        ax.plot(xw, hc[win], color=CHART_C, lw=0.9, label="chart", zorder=4)
        ax.plot(xw, hsdf[win], color=SDF_C, lw=1.4, ls="--", label="neural SDF", zorder=2)
        ax.set_xlim(x0, x0 + 7.0); ax.set_xlabel("x (mm)"); ax.set_ylabel("height (mm)")
        ax.legend(fontsize=5.6, loc="best")
    ax.set_title(f"(c) Level set smooths slopes\nrecon {g['sdf']['recon_rmse_mm']*1e3:.0f} vs "
                 f"{g['chart']['recon_rmse_mm']*1e3:.1f} $\\mu$m", fontsize=8)

    # (D) shear consequence -------------------------------------------------------------------------
    sc = R["shear_curves"]; up = R["shear_consequence"]["peak_mu_underprediction"]
    ax = axs[1, 0]
    ax.plot(sc["ux"], sc["mu_chart"], color=CHART_C, lw=1.1, label="chart-resolved")
    ax.plot(sc["ux"], sc["mu_sdf"], color=SDF_C, lw=1.1, label="neural SDF (smoothed)")
    ax.set_title(f"(d) SDF under-predicts peak\nshear strength by {up*100:.0f}%", fontsize=8)
    ax.set_xlabel("shear $u_x$ (mm)"); ax.set_ylabel("$\\mu_{app}$"); ax.legend(fontsize=6)

    # (E) roughness ensemble: dilation mean +/- std -------------------------------------------------
    ax = axs[1, 1]; ens = R.get("ensemble", {})
    if ens:
        for tag, c in (("rough", "#8c4a2f"), ("smooth", "#3b7ea1")):
            ss = ens[tag]; uxg = np.array(ss["ux"]); m = np.array(ss["dil_mean"]); sd = np.array(ss["dil_std"])
            ax.plot(uxg, m, color=c, lw=1.3, label=f"{tag} (RMS {ss['rms_mm']:.2f} mm)")
            ax.fill_between(uxg, m - sd, m + sd, color=c, alpha=0.20)
        ax.set_title(f"(e) Rougher $\\to$ more dilation\n(ensemble of {ens['rough']['n_rows']} real lines)",
                     fontsize=8)
    ax.set_xlabel("shear $u_x$ (mm)"); ax.set_ylabel("dilation (mm)"); ax.legend(fontsize=6, loc="upper left")

    # (F) headline numbers --------------------------------------------------------------------------
    ax = axs[1, 2]; ax.axis("off")
    rows = [
        ("Patton anchor", f"{pat['mu_app_relerr_vs_patton']*100:.2f}%  (exact)"),
        ("chart recon", f"{g['chart']['recon_rmse_mm']*1e3:.1f} $\\mu$m  ({g['chart']['n_params']//1000}k par)"),
        ("ambient SDF recon", f"{g['sdf']['recon_rmse_mm']*1e3:.0f} $\\mu$m  ({g['sdf']['n_params']//1000}k par)"),
        ("asperity angle", f"{g['ref']['mean_angle']:.0f}$^\\circ$ true $\\to$ {g['sdf']['mean_angle']:.0f}$^\\circ$ SDF"),
        ("strength under-pred.", f"{up*100:.0f}%"),
    ]
    if ens:
        rows.append(("dilation rough/smooth",
                     f"{ens['rough']['dil_total_mean']:.1f}/{ens['smooth']['dil_total_mean']:.1f} mm"))
    ax.text(0.5, 1.02, "neural atlas vs level set", ha="center", va="top",
            fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    for k, (lab, val) in enumerate(rows):
        y = 0.86 - k * 0.16
        ax.text(0.02, y, lab, fontsize=7, transform=ax.transAxes)
        ax.text(0.98, y, val, fontsize=7, ha="right", fontweight="bold", transform=ax.transAxes,
                color=CHART_C if "chart" in lab else (SDF_C if "SDF" in lab else "#000"))

    fig.suptitle("Direct shear of a real fractal rock joint: the chart resolves the asperities the "
                 "level set smooths away", y=1.01, fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "rock_joint_capstone_pub.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  saved", out)


def shear_gif(n_frames=56, shear_total=8.0, sigma_n=1.0, mu=0.3):
    """Two panels: (top) the rough mated joint shearing with lit contacts; (bottom) the dilation
    curve tracing out as it rides over the asperities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scipy.ndimage import gaussian_filter1d
    from benchmarks.contact.cv_numerical import rock_joint_shear as rjs

    x, z0 = _load_profile("rough")
    z = gaussian_filter1d(z0, 1.5, mode="nearest")
    m = (x >= 20.0) & (x <= 50.0); xw, zw = x[m], z[m]
    lower = rjs.bake_samples(xw, zw); eps_n = 5.0e4
    span = xw[-1] - xw[0]; amp = zw.max() - zw.min()
    y_lo = float(zw.max() - zw.min() - 2 * amp); y_hi = float(zw.max() - zw.min() + 2 * amp)
    uxs = np.linspace(0, shear_total, n_frames)

    # precompute y(ux) and dilation(ux)
    ys = []
    for ux in uxs:
        up = dict(x=xw.copy(), h=zw.copy(), hp=lower["hp"].copy())
        L_ov = max(span - abs(ux), 0.25 * span)
        ys.append(rjs.solve_y_equilibrium(ux, lower, up, eps_n, mu, sigma_n * L_ov, (y_lo, y_hi)))
    ys = np.array(ys); dil = ys - ys[0]

    fig, (axS, axD) = plt.subplots(2, 1, figsize=(6.4, 4.6),
                                   gridspec_kw=dict(height_ratios=[2.2, 1.0]))

    def frame(k):
        axS.clear(); axD.clear()
        ux = uxs[k]; y = ys[k]
        axS.fill_between(xw, zw - 6, zw, color="#b9a06a", alpha=0.95)              # footwall
        Xu = xw + ux; Zu = y + zw
        axS.fill_between(Xu, Zu, Zu + 6, color="#9aa0a6", alpha=0.9)               # hangingwall
        hL, hpL, valid = rjs._interp(lower, Xu)
        gap_n = (Zu - hL) / np.sqrt(1 + hpL ** 2)
        act = valid & (gap_n < 0)
        axS.plot(Xu[act], Zu[act], "o", ms=2.6, color="#d2382c", zorder=5)
        axS.set_xlim(xw[0], xw[-1] + shear_total); axS.set_ylim(zw.min() - 3, zw.max() + amp + 4)
        axS.set_title(f"Rock-joint direct shear   $u_x$={ux:4.2f} mm   dilation={dil[k]:+4.2f} mm   "
                      f"contacts={int(act.sum())}", fontsize=9)
        axS.set_xlabel("x (mm)"); axS.set_ylabel("z (mm)")
        axD.plot(uxs[:k + 1], dil[:k + 1], color="#0072B2", lw=1.8)
        axD.plot(ux, dil[k], "o", ms=5, color="#0072B2")
        axD.set_xlim(0, shear_total); axD.set_ylim(min(dil) - 0.1, max(dil) * 1.1 + 0.1)
        axD.set_xlabel("shear $u_x$ (mm)"); axD.set_ylabel("dilation (mm)")
        axD.set_title("the joint opens up (dilates) as asperities ride over each other", fontsize=8)
        fig.tight_layout()

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=90)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "rock_joint_shear.gif")
    anim.save(out, writer=PillowWriter(fps=11)); plt.close(fig)
    print("  saved", out)


def main():
    if not os.path.exists(RUN):
        print(f"  {RUN} not found — run rock_joint_capstone.py first"); return
    R = json.load(open(RUN))
    hero_figure(R)
    shear_gif()


if __name__ == "__main__":
    main()
