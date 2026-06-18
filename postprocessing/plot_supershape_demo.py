"""Visualize the superformula cam-drive demo + the honest chart-vs-SDF comparison.

Reads runs/supershape_cam_drive/history.json (produced by
benchmarks/contact/supershape_cam_drive.py) and produces:
  - figures/supershape_cam_drive.gif      (animation: cam A spins, follower B is driven)
  - figures/supershape_summary_pub.png    (alpha_B, omega_B, n_contacts, energy vs time)
  - figures/supershape_chart_vs_sdf_pub.png  (3-panel: radial-gap field, true Euclidean
        SDF with medial-axis ridge, and the angle between the radial normal and the true
        surface normal -- shows BOTH the smoothness advantage AND the radial bias)

Run:  python3 postprocessing/plot_supershape_demo.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers.contact import supershape as ss             # noqa: E402
from utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W, SINGLE_COL_W, GOLDEN  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RUN = os.path.join("runs", "supershape_cam_drive", "history.json")


def _load():
    with open(RUN) as f:
        data = json.load(f)
    m = data["meta"]
    pA = ss.SuperParams(**m["params_A"])
    pB = ss.SuperParams(**m["params_B"])
    return data["history"], m, pA, pB


def animate(stride=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    set_pub_style()
    hist, meta, pA, pB = _load()
    frames = list(range(0, len(hist), stride))
    th = np.linspace(0, 2 * np.pi, 400)
    cA = np.array([0.0, 0.0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 4.2)
    ax.set_ylim(-3.2, 2.6)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def draw(fi):
        ax.clear()
        ax.set_aspect("equal"); ax.set_xlim(-2.2, 4.2); ax.set_ylim(-3.2, 2.6)
        h = hist[fi]
        bA = ss.boundary(th, cA, h["alpha_A"], pA)
        cB = np.array([h["cB_x"], h["cB_y"]])
        bB = ss.boundary(th, cB, h["alpha_B"], pB)
        ax.fill(bA[:, 0], bA[:, 1], fc="#d98e5a", ec="k", lw=1.0, alpha=0.9)
        ax.fill(bB[:, 0], bB[:, 1], fc="#5a8fd9", ec="k", lw=1.0, alpha=0.9)
        # mark contact (B-boundary points inside A)
        g, _ = ss.radial_gap(bB, cA, h["alpha_A"], pA)
        pen = bB[g < 0]
        if len(pen):
            ax.plot(pen[:, 0], pen[:, 1], "r.", ms=3, zorder=5)
        ax.plot(*cA, "k+", ms=8)
        ax.plot(cB[0], cB[1], "k+", ms=8)
        ax.set_title("t = %.3f s   |   contact arcs: %d" % (h["time"], h["n_contacts"]))
        ax.set_xlabel("x"); ax.set_ylabel("y")

    anim = FuncAnimation(fig, draw, frames=frames, interval=60)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "supershape_cam_drive.gif")
    anim.save(path, writer=PillowWriter(fps=18))
    plt.close(fig)
    return path


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()
    hist, meta, pA, pB = _load()
    t = np.array([h["time"] for h in hist])
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.7))
    axs[0, 0].plot(t, [h["cB_y"] for h in hist], color=PUB_COLORS[0], label=r"$c_{B,y}$")
    axs[0, 0].plot(t, [h["cB_x"] for h in hist], color=PUB_COLORS[2], label=r"$c_{B,x}$")
    axs[0, 0].set_ylabel("follower center"); axs[0, 0].legend()
    axs[0, 1].plot(t, [h["alpha_B"] for h in hist], color=PUB_COLORS[1])
    axs[0, 1].set_ylabel(r"$\alpha_B$ (rad)")
    axs[1, 0].plot(t, [h["n_contacts"] for h in hist], color=PUB_COLORS[3])
    axs[1, 0].set_ylabel("# contact arcs"); axs[1, 0].set_xlabel("time (s)")
    axs[1, 1].plot(t, [h["ke_B"] for h in hist], color=PUB_COLORS[0], label="KE$_B$")
    axs[1, 1].plot(t, [h["energy_injected"] for h in hist], color=PUB_COLORS[1], ls="--", label="injected")
    axs[1, 1].plot(t, [h["energy_dissipated"] for h in hist], color=PUB_COLORS[4], ls=":", label="dissipated")
    axs[1, 1].set_ylabel("energy"); axs[1, 1].set_xlabel("time (s)"); axs[1, 1].legend()
    for ax in axs.ravel():
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Superformula cam drives the free follower")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "supershape_summary_pub.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_vs_sdf(n=300):
    """Honest 3-panel: radial-gap field, true Euclidean SDF (medial axis), normal-angle bias."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()
    _, _, _pA, pB = _load()
    p = pB
    c = np.array([0.0, 0.0])
    R = ss.radius(np.linspace(0, 2 * np.pi, 400), p).max() * 1.6
    xs = np.linspace(-R, R, n); ys = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)

    # (1) radial-gap field + its normal
    g_rad, grad = ss.radial_gap(P, c, 0.0, p)
    n_rad = grad / np.clip(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12, None)

    # (2) true Euclidean SDF + nearest-foot surface normal (dense theta)
    thb = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    B = ss.boundary(thb, c, 0.0, p)               # (Nb,2)
    Nrm = ss.outward_normal(thb, c, 0.0, p)        # (Nb,2)
    sdf = np.empty(len(P)); n_true = np.empty((len(P), 2))
    inside = ss.inside(P, c, 0.0, p)
    # chunked nearest-boundary search
    for i0 in range(0, len(P), 4000):
        chunk = P[i0:i0 + 4000]
        d2 = np.sum((chunk[:, None, :] - B[None, :, :]) ** 2, axis=2)
        k = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(chunk)), k])
        sdf[i0:i0 + 4000] = dist
        n_true[i0:i0 + 4000] = Nrm[k]
    sdf = np.where(inside, -sdf, sdf)

    ang = np.degrees(np.arccos(np.clip(np.sum(n_rad * n_true, axis=1), -1, 1)))

    Grad = g_rad.reshape(X.shape)
    Sdf = sdf.reshape(X.shape)
    Ang = np.where(inside, np.nan, ang).reshape(X.shape)   # show bias outside the body

    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.36))
    bnd = ss.boundary(np.linspace(0, 2 * np.pi, 400), c, 0.0, p)
    for ax in axs:
        ax.set_aspect("equal"); ax.plot(bnd[:, 0], bnd[:, 1], "k-", lw=1.0); ax.axis("off")
    im0 = axs[0].contourf(X, Y, Grad, levels=24, cmap="coolwarm",
                          vmin=-np.nanpercentile(np.abs(Grad), 98), vmax=np.nanpercentile(np.abs(Grad), 98))
    axs[0].contour(X, Y, Grad, levels=[0], colors="k", linewidths=0.6)
    axs[0].set_title("radial-chart gap $g_B$\n(smooth, radial iso-contours)")
    fig.colorbar(im0, ax=axs[0], shrink=0.7)
    im1 = axs[1].contourf(X, Y, Sdf, levels=24, cmap="coolwarm",
                          vmin=-np.nanpercentile(np.abs(Sdf), 98), vmax=np.nanpercentile(np.abs(Sdf), 98))
    axs[1].set_title("true Euclidean SDF\n(medial-axis ridge in concavities)")
    fig.colorbar(im1, ax=axs[1], shrink=0.7)
    im2 = axs[2].contourf(X, Y, Ang, levels=24, cmap="magma")
    axs[2].set_title("angle: radial normal vs\ntrue surface normal (deg)")
    fig.colorbar(im2, ax=axs[2], shrink=0.7)
    fig.suptitle("Chart radial gap vs true SDF: smoothness advantage AND radial bias")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "supershape_chart_vs_sdf_pub.png")
    fig.savefig(path); plt.close(fig)
    return path


def main():
    for fn in (summary, chart_vs_sdf, animate):
        try:
            print("  Saved:", fn())
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
