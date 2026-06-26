"""Figure 1: two descriptions of the same body.

(a) Level set -- the body as an IMPLICIT field phi:R^2->R, drawn with many nested iso-contours
    so the reader sees it fills space (the boundary is just one level, phi=0).
(b) Coordinate chart -- the body's boundary as an EXPLICIT map phi:Theta->dOmega from a reference
    parameter domain, drawn with arrows that carry reference points to their physical images
    (and the inverse pi=phi^{-1}).

Run:  <venv>/bin/python postprocessing/plot_levelset_vs_atlas.py
Out:  figures/levelset_vs_atlas_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
CORAL = "#BE5536"; BLUE = "#3E6E9E"; INK = "#222222"; GRID = "#cfcabd"


def radius(theta):
    return 1.0 + 0.30 * np.cos(3 * theta) + 0.09 * np.cos(5 * theta)


def boundary(n=600, c=(0.0, 0.0), scale=1.0):
    th = np.linspace(0, 2 * np.pi, n)
    r = radius(th) * scale
    return np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)]), th


def signed_distance(grid_xy, bpts, inside_mask_fn, chunk=4000):
    A = bpts[:-1]; B = bpts[1:]; AB = B - A
    ab2 = np.maximum((AB * AB).sum(-1), 1e-30)
    d = np.empty(len(grid_xy))
    for i in range(0, len(grid_xy), chunk):
        P = grid_xy[i:i + chunk]
        t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / ab2[None], 0, 1)
        proj = A[None] + t[..., None] * AB[None]
        d[i:i + chunk] = np.linalg.norm(P[:, None, :] - proj, axis=-1).min(1)
    sign = np.where(inside_mask_fn(grid_xy), -1.0, 1.0)
    return d * sign


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=10.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.50))

    # ================= (a) LEVEL SET: implicit field with many contours =================
    bpts, _ = boundary(n=700, scale=1.0)
    path = Path(bpts)
    g = np.linspace(-2.1, 2.1, 320)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    phi = signed_distance(pts, bpts, lambda P: path.contains_points(P)).reshape(X.shape)

    levels = np.arange(-1.6, 1.81, 0.4)
    cf = axL.contourf(X, Y, phi, levels=np.linspace(-1.8, 1.8, 40), cmap="RdBu_r", alpha=0.92,
                      extend="both")
    axL.contour(X, Y, phi, levels=levels, colors=INK, linewidths=0.6, alpha=0.5)
    axL.contour(X, Y, phi, levels=[0.0], colors="black", linewidths=2.2)   # the boundary phi=0
    axL.set_aspect("equal"); axL.set_xlim(-2.1, 2.1); axL.set_ylim(-2.1, 2.1)
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_title(r"(a) Level set: implicit field $\phi:\mathbb{R}^d\!\to\!\mathbb{R}$",
                  fontsize=11, fontweight="bold", pad=8)
    axL.text(0.0, 0.0, r"$\phi<0$", ha="center", va="center", fontsize=9.5, color="#1c3a57",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axL.annotate(r"$\partial\Omega=\{\phi=0\}$", xy=(0.60, 1.02), xytext=(1.18, 1.74),
                 ha="center", fontsize=8.5, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82))
    axL.text(-1.55, 1.70, r"$\phi>0$", ha="center", va="center", fontsize=9.5, color=INK)
    axL.text(0.5, -0.085, "one field over all space; the boundary is a single level",
             transform=axL.transAxes, ha="center", va="top", fontsize=8.0, color="0.3")
    cb = fig.colorbar(cf, ax=axL, fraction=0.046, pad=0.03)
    cb.set_label(r"$\phi$ (signed distance)", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb.add_lines(axL.contour(X, Y, phi, levels=[0.0], colors="black"))

    # ================= (b) CHART: explicit reference->physical map =================
    # reference parameter domain Theta = S^1 (left), physical boundary dOmega (right)
    ref_c = np.array([-1.65, 0.0]); phys_c = np.array([1.95, 0.0])
    tref = np.linspace(0, 2 * np.pi, 400)
    axR.plot(ref_c[0] + 0.95 * np.cos(tref), ref_c[1] + 0.95 * np.sin(tref),
             color=BLUE, lw=1.6, ls=(0, (5, 2)))
    axR.text(ref_c[0], ref_c[1], r"$\Theta=\mathbb{S}^{d-1}$", ha="center", va="center",
             fontsize=10, color=BLUE)
    bP, _ = boundary(n=500, c=phys_c, scale=1.0)
    axR.fill(bP[:, 0], bP[:, 1], facecolor=CORAL + "55", edgecolor=CORAL, lw=2.0)
    axR.text(phys_c[0], phys_c[1] - 0.05, r"$\partial\Omega$", ha="center", va="center",
             fontsize=11, color=CORAL)

    # mapping arrows: theta_i on the reference circle -> phi(theta_i) on the boundary
    samp = np.linspace(0, 2 * np.pi, 7)[:-1] + 0.15
    for k, a in enumerate(samp):
        p0 = ref_c + 0.95 * np.array([np.cos(a), np.sin(a)])
        p1 = phys_c + radius(a) * np.array([np.cos(a), np.sin(a)])
        axR.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=0.16",
                                      arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                      color=INK, alpha=0.75))
        axR.plot(*p0, "o", ms=3, color=BLUE)
        axR.plot(*p1, "o", ms=3, color=CORAL)
    # label the bundle of arrows
    axR.text(0.13, 0.74, r"$\varphi:\Theta\!\to\!\partial\Omega$", transform=axR.transAxes,
             fontsize=11, color=INK, ha="center",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.6))
    axR.text(0.13, 0.26, r"$\pi=\varphi^{-1}$", transform=axR.transAxes,
             fontsize=10, color="0.45", ha="center")
    axR.set_aspect("equal"); axR.set_xlim(-2.9, 3.3); axR.set_ylim(-1.7, 1.7)
    axR.set_xticks([]); axR.set_yticks([])
    for s in axR.spines.values():
        s.set_visible(False)
    axR.set_title(r"(b) Coordinate chart: boundary map $\varphi:\Theta\!\to\!\partial\Omega$",
                  fontsize=11, fontweight="bold")
    axR.text(0.5, -0.085, "only the surface, parametrised by its own coordinates",
             transform=axR.transAxes, ha="center", va="top", fontsize=8.0, color="0.3")

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "levelset_vs_atlas_pub.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
