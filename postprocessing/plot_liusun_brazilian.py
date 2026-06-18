"""Re-plot Liu & Sun (2020) ILS-MPM Brazilian-disc figures (analytical reference).

- Fig.21  geometry & BCs schematic (matplotlib)
- Fig.22  contact radius a vs force F (matplotlib; standard Hertz + the paper's
          Eq.46, which we showed is 4x low; numerical-overlay slot left empty)
- Fig.23  sigma_yy field on the disc (PyVista, analytical Flamant superposition)

Fields come from postprocessing/contact_fields.py (verified vs the SymPy
derivations in docs/hertz_derivation/brazilian_disc_atlas.py).
Run:  python3 postprocessing/plot_liusun_brazilian.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_fields as cf                       # noqa: E402
import pyvista_field2d as pf                       # noqa: E402
from utils import set_pub_style, add_colorbar, PUB_COLORS, SINGLE_COL_W, GOLDEN  # noqa: E402

FIG_DIR = pf.FIG_DIR

# Paper geometry/material (Fig.21, sec 5.3.1): disc R=10 mm, E=100 GPa, nu=0.3
R = 10.0
E = 1.0e5      # N/mm^2 (100 GPa)
NU = 0.3
T = 1.0        # thickness (mm); 2D per-thickness


def figure_23_sigma_yy(n=420, F=1000.0):
    """Brazilian disc sigma_yy field (compression along the loaded diameter)."""
    xs = np.linspace(-R, R, n)
    ys = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    sxx, syy, sxy = cf.brazilian_field(X, Y, R, F, T, r_floor=0.015 * R)
    mask = (X ** 2 + Y ** 2) <= R ** 2
    # mostly-compressive field -> non-symmetric clim with max ~ 0 (compression=blue)
    clim = pf.symmetric_clim(syy, mask=mask, pct=(1.0, 99.0), symmetric=False)
    return pf.render_field_2d(
        X, Y, syy, filename="liusun_fig23_brazilian_sigma_yy_pub.png",
        mask=mask, clim=clim, scalar_bar_title="Syy (MPa)",
        title="Brazilian disc:  sigma_yy", outlines=[(0.0, 0.0, R)],
        symmetric=False, window_size=(1100, 1050),
    )


def figure_22_aF_curve():
    """Contact half-width a vs force F: standard Hertz + paper Eq.46."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()
    Fline = np.linspace(1.0, 6.0e4, 400)          # line load N/mm
    a_std = cf.brazilian_contact_halfwidth(Fline, R, E, NU, T)
    a_eq46 = 0.5 * np.sqrt(Fline * (1.0 - NU ** 2) * R / (np.pi * E))   # paper Eq.46

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * GOLDEN))
    ax.plot(a_std, Fline, color=PUB_COLORS[0], label="Standard Hertz")
    ax.plot(a_eq46, Fline, color=PUB_COLORS[1], ls="--", label="Eq. 46 (paper)")
    # numerical-overlay slot (to be filled by the implementation later):
    # ax.plot(a_num, F_num, 'o', mfc='none', color=PUB_COLORS[7], label="Num. solution")
    ax.set_xlabel("contact half-width $a$ (mm)")
    ax.set_ylabel("force $F$ (N per mm thickness)")
    ax.set_title("Brazilian disc: $a$–$F$")
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = os.path.join(FIG_DIR, "liusun_fig22_brazilian_aF_pub.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    return path


def figure_21_schematic():
    """Geometry & BC schematic: disc R=10 between two rigid platens."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, FancyArrow

    set_pub_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))
    W = 14.0
    ax.add_patch(Circle((0, 0), R, fill=True, fc="#e8e3c8", ec="k", lw=1.2))
    ax.add_patch(Rectangle((-W / 2, R), W, 2.0, fc="#bcd4e6", ec="k", lw=1.0))      # top platen
    ax.add_patch(Rectangle((-W / 2, -R - 2.0), W, 2.0, fc="#bcd4e6", ec="k", lw=1.0))  # bottom
    for xx in np.linspace(-4, 4, 5):                                                # load arrows
        ax.add_patch(FancyArrow(xx, R + 3.2, 0, -1.0, width=0.06, head_width=0.5,
                                head_length=0.5, color="k"))
    ax.text(0, R + 4.0, r"$\Delta u_y$", ha="center", va="bottom")
    ax.text(0, -R - 3.0, "fixed", ha="center", va="top")
    ax.annotate("", xy=(R, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.text(R / 2, 0.5, "R=10", ha="center")
    ax.set_xlim(-W / 2 - 1, W / 2 + 1)
    ax.set_ylim(-R - 5, R + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Brazilian disc: geometry & BCs")
    path = os.path.join(FIG_DIR, "liusun_fig21_brazilian_schematic_pub.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    for fn in (figure_21_schematic, figure_22_aF_curve, figure_23_sigma_yy):
        try:
            print("  Saved:", fn())
        except Exception as exc:                   # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
