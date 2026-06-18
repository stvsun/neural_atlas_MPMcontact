"""Re-plot Liu & Sun (2020) ILS-MPM nine-disc figures (analytical reference).

- Fig.15  geometry & BCs schematic (matplotlib): 3x3 array of discs (R=1.7) in
          four confining plates under isotropic compression.
- Fig.16  two-panel field (PyVista): max principal stress S1 and shear sigma_xy
          over the full 3x3 array, reconstructed by tiling the identical
          4-load unit-cell field across the 9 disc centres.

By the symmetry reduction (docs/hertz_derivation/nine_disc_atlas.py) every disc
carries the same field: four equal inward contact forces N at N,S,E,W.
Run:  python3 postprocessing/plot_liusun_nine_disc.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_fields as cf                       # noqa: E402
import pyvista_field2d as pf                       # noqa: E402
from utils import set_pub_style, SINGLE_COL_W     # noqa: E402

FIG_DIR = pf.FIG_DIR

R = 1.7        # disc radius (mm)
T = 1.0        # thickness (mm)
N = 1000.0     # representative contact force per thickness (sets the stress scale)


def _tiled_fields(n=540):
    """Assemble S1 and sigma_xy over the 3x3 array (NaN outside the discs)."""
    centers = [(2 * R * i, 2 * R * j) for j in (1, 0, -1) for i in (-1, 0, 1)]
    span = 3.0 * R
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    S1 = np.full_like(X, np.nan)
    SXY = np.full_like(X, np.nan)
    mask_all = np.zeros_like(X, dtype=bool)
    for (xc, yc) in centers:
        sxx, syy, sxy = cf.nine_disc_unit_cell_field(X - xc, Y - yc, R, N, T,
                                                     r_floor=0.02 * R)
        s1, _s2, _tmax = cf.principal_stresses(sxx, syy, sxy)
        dmask = (X - xc) ** 2 + (Y - yc) ** 2 <= R ** 2
        S1[dmask] = s1[dmask]
        SXY[dmask] = sxy[dmask]
        mask_all |= dmask
    outlines = [(xc, yc, R) for (xc, yc) in centers]
    return X, Y, S1, SXY, mask_all, outlines


def figure_16_S1_Sxy():
    X, Y, S1, SXY, mask_all, outlines = _tiled_fields()
    panels = [
        dict(X=X, Y=Y, field=S1, mask=mask_all, outlines=outlines, symmetric=False,
             clim=pf.symmetric_clim(S1, mask=mask_all, pct=(2, 98), symmetric=False),
             scalar_bar_title="S1 (MPa)", title="max principal stress"),
        dict(X=X, Y=Y, field=SXY, mask=mask_all, outlines=outlines, symmetric=True,
             scalar_bar_title="Sxy (MPa)", title="shear stress"),
    ]
    return pf.render_two_panel(panels, filename="liusun_fig16_nine_disc_S1_Sxy_pub.png",
                               window_size=(2100, 1050))


def figure_15_schematic():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, FancyArrow

    set_pub_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))
    centers = [(2 * R * i, 2 * R * j) for j in (1, 0, -1) for i in (-1, 0, 1)]
    for (xc, yc) in centers:
        ax.add_patch(Circle((xc, yc), R, fc="#e8e3c8", ec="k", lw=1.0))
    half = 3.0 * R
    th = 1.0
    ax.add_patch(Rectangle((-half, half), 2 * half, th, fc="#bcd4e6", ec="k", lw=0.8))
    ax.add_patch(Rectangle((-half, -half - th), 2 * half, th, fc="#bcd4e6", ec="k", lw=0.8))
    ax.add_patch(Rectangle((-half - th, -half), th, 2 * half, fc="#bcd4e6", ec="k", lw=0.8))
    ax.add_patch(Rectangle((half, -half), th, 2 * half, fc="#bcd4e6", ec="k", lw=0.8))
    for s, (dx, dy, x0, y0) in zip("TBLR", [(0, -1, 0, half + 2.5), (0, 1, 0, -half - 2.5),
                                            (1, 0, -half - 2.5, 0), (-1, 0, half + 2.5, 0)]):
        ax.add_patch(FancyArrow(x0, y0, 1.3 * dx, 1.3 * dy, width=0.08, head_width=0.7,
                                head_length=0.7, color="k"))
    ax.text(0, half + 3.4, "$d$", ha="center")
    ax.annotate("", xy=(R, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.text(R / 2, 0.3, "R=1.7", ha="center", fontsize=7)
    ax.set_xlim(-half - 4, half + 4)
    ax.set_ylim(-half - 4, half + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Nine discs: isotropic compression")
    path = os.path.join(FIG_DIR, "liusun_fig15_nine_disc_schematic_pub.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    for fn in (figure_15_schematic, figure_16_S1_Sxy):
        try:
            print("  Saved:", fn())
        except Exception as exc:                   # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
