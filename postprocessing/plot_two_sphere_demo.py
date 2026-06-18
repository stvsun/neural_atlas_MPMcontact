"""Two-sphere collision GIF: level-set (SDF) detector == transition-map (chart) detector.

Runs the elastic two-ball MPM collision with the level-set-free radial-chart
detector (`solvers/contact/chart_gap.py::evaluate_gap_chart`) and animates the
actual particle clouds, the sphere boundaries, and the active contact set. Because
the sphere chart reproduces the analytic sphere SDF to machine precision (see
`benchmarks/contact/chart_vs_sdf_detection.py`), the level-set detector produces
identical frames — hence the title "level-set == transition-map".

The velocity panel shows the full collision arc (the v_A / v_B curves cross =
rebound). GIF idiom (FuncAnimation + PillowWriter) mirrors plot_supershape_demo.py.

Run:  python3 postprocessing/plot_two_sphere_demo.py
Output: figures/two_sphere_collision.gif
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_pub_style, PUB_COLORS                              # noqa: E402
from solvers.contact.chart_gap import SphereRho, evaluate_gap_chart      # noqa: E402
from solvers.contact.penalty import compute_contact_force, contact_stable_dt  # noqa: E402
from solvers.mpm.chart_mpm_solver import ChartMPMSolver                  # noqa: E402
from solvers.mpm.particles import MaterialPointCloud                     # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
R = 0.07

torch.set_default_dtype(torch.float64)


def simulate(v_app=0.4, eps_n=2.0e7, gap0=0.012, n_steps=5500, n_per_axis=5, stride=70):
    """Run the collision with the chart detector; record per-frame clouds + v(t)."""
    dens = 1000.0
    cA0 = torch.tensor([-(R + gap0 / 2), 0.0, 0.0])
    cB0 = torch.tensor([+(R + gap0 / 2), 0.0, 0.0])
    oA, oB = SphereRho(R, cA0.tolist()), SphereRho(R, cB0.tolist())
    sA = ChartMPMSolver(n_cells=16, extent=0.6, gravity=None, bc_type="free")
    sB = ChartMPMSolver(n_cells=16, extent=0.6, gravity=None, bc_type="free")
    pA = MaterialPointCloud.create_uniform(n_per_axis=n_per_axis, extent=R, density=dens)
    pA.xi += cA0; pA.v[:, 0] = +v_app
    pB = MaterialPointCloud.create_uniform(n_per_axis=n_per_axis, extent=R, density=dens)
    pB.xi += cB0; pB.v[:, 0] = -v_app
    # display-only spherical mask: show the inscribed ball of points (the full cube
    # cloud is still simulated; this just makes the render read as two spheres).
    sphA = ((pA.xi - cA0).norm(dim=1) <= R * 1.02).numpy().copy()
    sphB = ((pB.xi - cB0).norm(dim=1) <= R * 1.02).numpy().copy()
    dt = min(1.5e-5, contact_stable_dt(eps_n, min(pA.mass.min().item(), pB.mass.min().item())))

    def com(p):
        m = p.mass.unsqueeze(1); return (m * p.xi).sum(0) / p.mass.sum()

    def vc(p):
        m = p.mass.unsqueeze(1); return ((m * p.v).sum(0) / p.mass.sum())[0].item()

    frames, t_all, vA_all, vB_all = [], [], [], []
    for i in range(n_steps):
        oA.center = com(pA).clone(); oB.center = com(pB).clone()
        gA, nA = evaluate_gap_chart(pA.xi.detach(), oB)
        gB, nB = evaluate_gap_chart(pB.xi.detach(), oA)
        cfA = compute_contact_force(gA, nA, pA.current_volume, eps_n)
        cfB = compute_contact_force(gB, nB, pB.current_volume, eps_n)
        sA.step(pA, dt, contact_force=cfA); sB.step(pB, dt, contact_force=cfB)
        t = (i + 1) * dt
        t_all.append(t * 1e3); vA_all.append(vc(pA)); vB_all.append(vc(pB))
        if i % stride == 0 or i == n_steps - 1:
            frames.append(dict(
                t=t * 1e3,
                xA=pA.xi.detach().numpy().copy(), xB=pB.xi.detach().numpy().copy(),
                cA=com(pA).numpy().copy(), cB=com(pB).numpy().copy(),
                mA=(gA < 0).numpy().copy(), mB=(gB < 0).numpy().copy(),
                vA=vc(pA), vB=vc(pB),
                nc=int((gA < 0).sum() + (gB < 0).sum()),
            ))
    return frames, np.array(t_all), np.array(vA_all), np.array(vB_all), sphA, sphB


def render(frames, t_all, vA_all, vB_all, sphA, sphB, out=None, dpi=95):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageSequence

    if out is None:
        out = os.path.join(FIG_DIR, "two_sphere_collision.gif")
    set_pub_style()
    cblue, cverm, cred = PUB_COLORS[0], PUB_COLORS[1], "#D7191C"
    allc = np.array([f["cA"][0] for f in frames] + [f["cB"][0] for f in frames])
    xlo, xhi = allc.min() - R - 0.02, allc.max() + R + 0.02

    fig = plt.figure(figsize=(5.2, 5.8), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.35)
    axm = fig.add_subplot(gs[0]); axv = fig.add_subplot(gs[1])

    def draw(fi):
        f = frames[fi]
        axm.clear()
        axm.set_aspect("equal"); axm.set_xlim(xlo, xhi)
        axm.set_ylim(-(xhi - xlo) / 2, (xhi - xlo) / 2)
        for c, fc in ((f["cA"], cblue), (f["cB"], cverm)):
            axm.add_patch(Circle((c[0], c[1]), R, fc=fc, ec="k", lw=1.2, alpha=0.12, zorder=1))
        xA, xB = f["xA"], f["xB"]
        axm.scatter(xA[sphA, 0], xA[sphA, 1], s=16, c=cblue, edgecolors="none", zorder=2)
        axm.scatter(xB[sphB, 0], xB[sphB, 1], s=16, c=cverm, edgecolors="none", zorder=2)
        mA, mB = f["mA"] & sphA, f["mB"] & sphB
        if mA.any():
            axm.scatter(xA[mA, 0], xA[mA, 1], s=30, c=cred, zorder=4)
        if mB.any():
            axm.scatter(xB[mB, 0], xB[mB, 1], s=30, c=cred, zorder=4)
        axm.plot(f["cA"][0], f["cA"][1], "k+", ms=9); axm.plot(f["cB"][0], f["cB"][1], "k+", ms=9)
        axm.set_xlabel("x"); axm.set_ylabel("y")
        axm.set_title("t = %5.1f ms    active contact particles: %d" % (f["t"], f["nc"]))

        axv.clear()
        axv.plot(t_all, vA_all, color=cblue, lw=1.8, label="$v_A$")
        axv.plot(t_all, vB_all, color=cverm, lw=1.8, label="$v_B$")
        axv.axvline(f["t"], color="0.5", lw=1.0, ls="--")
        axv.plot(f["t"], f["vA"], "o", color=cblue, ms=6)
        axv.plot(f["t"], f["vB"], "o", color=cverm, ms=6)
        axv.set_xlabel("time (ms)"); axv.set_ylabel("COM $v_x$")
        axv.legend(loc="center right", fontsize=9, frameon=False)
        axv.grid(True, alpha=0.3)
        fig.suptitle("Two-sphere collision · level-set ≡ transition-map",
                     y=0.98, fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=55)
    os.makedirs(FIG_DIR, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=18), dpi=dpi)   # else savefig.dpi (300) is used
    plt.close(fig)

    # post-optimize: one SHARED palette for every frame so optimize=True can diff
    # frames (per-frame palettes defeat it). Cuts size ~5x.
    im = Image.open(out)
    rgb = [fr.copy().convert("RGB") for fr in ImageSequence.Iterator(im)]
    pal = rgb[len(rgb) // 2].quantize(colors=48, method=Image.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.NONE) for fr in rgb]
    q[0].save(out, save_all=True, append_images=q[1:], loop=0,
              duration=int(1000 / 18), optimize=True, disposal=2)
    return out


def animate():
    """Simulate, trim to the contact window, and write the GIF."""
    frames, t_all, vA_all, vB_all, sphA, sphB = simulate()
    ncs = [f["nc"] for f in frames]
    act = [i for i, n in enumerate(ncs) if n > 0]
    if act:                                   # trim to the contact window + margins
        lo, hi = max(0, act[0] - 4), min(len(frames), act[-1] + 10)
        frames = frames[lo:hi]
    path = render(frames, t_all, vA_all, vB_all, sphA, sphB)
    print("v_A: %.3f -> %.3f (rebound=%s)  peak active=%d  frames=%d  ->  %s  (%d KB)" % (
        vA_all[0], vA_all[-1], vA_all[-1] < 0, max(ncs), len(frames),
        path, os.path.getsize(path) // 1024))
    return path


if __name__ == "__main__":
    animate()
