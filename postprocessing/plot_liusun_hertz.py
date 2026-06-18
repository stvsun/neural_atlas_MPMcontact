"""Re-plot Liu & Sun (2020) ILS-MPM Hertz-contact figures (analytical reference).

- Fig.12  geometry & BCs schematic (matplotlib): curved body 1 (R=10) on block.
- Fig.13  vertical stress sigma_yy field (PyVista): line-contact subsurface
          stress (McEwen / Johnson eq 4.49) in both bodies near the contact.
- Fig.14  contact tractions sigma_n, sigma_t along the contact surface
          (matplotlib; Hertz pressure + Cattaneo-Mindlin partial slip;
          numerical-overlay slots left empty).

Fields from postprocessing/contact_fields.py (verified vs the SymPy derivations
in docs/hertz_derivation/hertz_transition_map.py).
Run:  python3 postprocessing/plot_liusun_hertz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_fields as cf                       # noqa: E402
import pyvista_field2d as pf                       # noqa: E402
from utils import set_pub_style, PUB_COLORS, SINGLE_COL_W, GOLDEN  # noqa: E402

FIG_DIR = pf.FIG_DIR

# Two elastic bodies, E=100 GPa, nu=0.3; curved body radius R=10 mm (Fig.12).
R = 10.0
E = 1.0e5
NU = 0.3
ESTAR = E / (2.0 * (1.0 - NU ** 2))   # two identical elastic bodies
PLINE = 1550.0                        # line load N/mm -> p0 ~ 1.6 GPa (paper scale)
MU = 0.5
QFRAC = 0.5                           # Q/(mu P): partial slip


def figure_13_sigma_yy(n=500):
    a, p0 = cf.line_contact_params(PLINE, R, ESTAR)
    W, H = 4.0, 4.0
    xs = np.linspace(-W, W, n)
    ys = np.linspace(-H, H, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # vertical stress sigma_yy = sigma_z(x, |y|) (mirror about the contact plane)
    _sx, sigyy, _txz = cf.line_contact_subsurface(X, Y, a, p0)
    # geometry mask: lower block (y<=0) + curved upper body (y >= x^2/2R)
    gap = X ** 2 / (2.0 * R)
    mask = (Y <= 0.0) | (Y >= gap)
    clim = pf.symmetric_clim(sigyy, mask=mask, pct=(1.0, 99.0), symmetric=False)
    return pf.render_field_2d(
        X, Y, sigyy, filename="liusun_fig13_hertz_sigma_yy_pub.png",
        mask=mask, clim=clim, scalar_bar_title="Syy (MPa)",
        title="Hertz contact:  sigma_yy", symmetric=False,
        window_size=(1100, 1100),
    )


def figure_14_traction():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()
    a, p0 = cf.line_contact_params(PLINE, R, ESTAR)
    P = PLINE
    Q = QFRAC * MU * P
    c = cf.cattaneo_stick_radius(Q, MU, P, a)
    x = np.linspace(-a, a, 600)
    sig_n = -cf.hertz_pressure(x, a, p0)              # contact pressure (compressive)
    sig_t = cf.cattaneo_traction(x, a, c, MU, p0)     # tangential traction

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * GOLDEN))
    ax.plot(x, sig_n, color=PUB_COLORS[1], label=r"$\sigma_n$ (ana.)")
    ax.plot(x, sig_t, color=PUB_COLORS[0], label=r"$\sigma_t$ (ana.)")
    ax.axvline(c, color="0.6", ls=":", lw=0.8)
    ax.axvline(-c, color="0.6", ls=":", lw=0.8)
    ax.text(c, ax.get_ylim()[1] * 0.9, " stick", fontsize=7, color="0.4")
    # numerical-overlay slots (filled by the implementation later):
    # ax.plot(x_num, sn_num, 's', mfc='none', color=PUB_COLORS[1], label=r"$\sigma_n$ (num.)")
    # ax.plot(x_num, st_num, 'o', mfc='none', color=PUB_COLORS[0], label=r"$\sigma_t$ (num.)")
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel(r"traction (MPa)")
    ax.set_title("Hertz contact tractions")
    ax.legend(loc="lower center", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = os.path.join(FIG_DIR, "liusun_fig14_hertz_traction_pub.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    return path


def figure_12_schematic():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow

    set_pub_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))
    W = 8.0
    # lower block (body 2)
    ax.add_patch(Rectangle((-W / 2, -4), W, 4, fc="#bcd4e6", ec="k", lw=1.0))
    # upper body (body 1) with curved bottom y = x^2/2R
    xs = np.linspace(-W / 2, W / 2, 200)
    bottom = xs ** 2 / (2.0 * R)
    upper_x = np.concatenate([xs, [W / 2, -W / 2]])
    upper_y = np.concatenate([bottom, [4.0, 4.0]])
    ax.fill(upper_x, upper_y, fc="#e8e3c8", ec="k", lw=1.0)
    for xx in np.linspace(-2.5, 2.5, 5):
        ax.add_patch(FancyArrow(xx, 5.0, 0, -0.9, width=0.05, head_width=0.35,
                                head_length=0.35, color="k"))
    ax.text(0, 5.4, r"$u_y$", ha="center")
    ax.text(0, -4.5, "fixed", ha="center", va="top")
    ax.annotate("R=10", xy=(2.0, 4.0 / (R)), xytext=(2.5, 2.0),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.text(-W / 2 - 0.3, -2, "Body 2", rotation=90, va="center", fontsize=8)
    ax.text(-W / 2 - 0.3, 2, "Body 1", rotation=90, va="center", fontsize=8)
    ax.set_xlim(-W / 2 - 1, W / 2 + 1)
    ax.set_ylim(-5, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Hertz contact: geometry & BCs")
    path = os.path.join(FIG_DIR, "liusun_fig12_hertz_schematic_pub.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    for fn in (figure_12_schematic, figure_13_sigma_yy, figure_14_traction):
        try:
            print("  Saved:", fn())
        except Exception as exc:                   # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
