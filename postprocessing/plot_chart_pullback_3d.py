"""Figure for Section 3.3 (3-D): reduction of the governing equations to the reference chart.

A lit 3-D scene, composited with LaTeX labels in matplotlib: a reference cube U=[-r,r]^3 in
chart coordinates xi is mapped to the physical chart Omega_alpha by the trained decoder
x = D_theta(xi). A highlighted reference element and its coordinate triad are carried to the
deformed element by the chart Jacobian J = dD_theta/dxi; the internal virtual work is integrated
on U with the Piola transform P_chart = |det J| P J^{-T} (annotated).

Run:  <venv>/bin/python postprocessing/plot_chart_pullback_3d.py
Out:  figures/chart_pullback_3d_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
CORAL = "#BE5536"; BLUE = "#3E6E9E"; GRAYB = "#9C978B"; INK = "#2b2b2b"
REFC = "#8C8779"; CHC = "#D98E6E"


def warp(P, center=(1.75, 0.0, 0.10)):
    """Smooth bend of the local cube into a curved physical chart (a gently arced block)."""
    a = 0.42 * P[:, 0]                       # along-x -> arc angle
    R = 1.30
    rr = R + P[:, 2] * 0.98
    X = rr * np.sin(a)
    Z = rr * (1.0 - np.cos(a))
    Y = P[:, 1] * 0.92 * (1.0 - 0.10 * P[:, 0])
    return np.column_stack([X, Y, Z]) + np.asarray(center)


def grid_cube(half=0.85, n=7, center=(-2.5, 0, 0)):
    g = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sg = pv.StructuredGrid(X, Y, Z)
    sg.points = sg.points + np.asarray(center)
    return sg, half


def lights(pl):
    pl.add_light(pv.Light(position=(4, -5, 6), focal_point=(0, 0, 0), intensity=0.80))
    pl.add_light(pv.Light(position=(-5, -2, 3), focal_point=(0, 0, 0), intensity=0.34))
    pl.add_light(pv.Light(position=(0, 6, 4), focal_point=(0, 0, 0), intensity=0.30))
    pl.add_light(pv.Light(position=(2, -4, -5), focal_point=(0, 0, 0), intensity=0.30))


def trim(img, bg=248):
    mask = (img[:, :, :3] < bg).any(2)
    ys, xs = np.where(mask); p = 8
    return img[max(ys.min() - p, 0):ys.max() + p, max(xs.min() - p, 0):xs.max() + p]


def render(size=(1500, 760)):
    REF_CENTER = (-2.5, 0, 0)
    half = 0.85
    sg, half = grid_cube(half=half, n=7, center=REF_CENTER)

    pl = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    pl.set_background("white")
    MAT = dict(ambient=0.36, diffuse=0.85, smooth_shading=False, specular=0.25,
               show_scalar_bar=False)

    # --- reference cube U (chart coordinates) ---
    pl.add_mesh(sg, color=REFC, opacity=0.5, show_edges=True, edge_color="#5f5b50",
                line_width=1.0, **MAT)
    # highlighted reference element (one corner cell)
    he = 0.85 * 2 / 6  # one cell width
    e0 = np.array(REF_CENTER) + np.array([half - he, half - he, half - he])
    elem_ref = pv.Box(bounds=(e0[0], e0[0] + he, e0[1], e0[1] + he, e0[2], e0[2] + he))
    pl.add_mesh(elem_ref, color=CORAL, opacity=0.95, show_edges=True, edge_color="#5a2414",
                line_width=1.4, **MAT)

    # --- physical chart Omega_alpha = D_theta(U) ---
    wp = sg.copy(); wp.points = warp(sg.points - np.array(REF_CENTER))
    pl.add_mesh(wp, color=CHC, opacity=0.55, show_edges=True, edge_color="#9c4a2e",
                line_width=1.0, **MAT)
    # image of the highlighted element (8 corners warped, as a hexahedron)
    loc = np.array([[half - he, half - he, half - he], [half, half - he, half - he],
                    [half, half, half - he], [half - he, half, half - he],
                    [half - he, half - he, half], [half, half - he, half],
                    [half, half, half], [half - he, half, half]])
    img_corners = warp(loc)
    hexa = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: np.arange(8)[None]}, img_corners)
    pl.add_mesh(hexa, color=CORAL, opacity=0.97, show_edges=True, edge_color="#5a2414",
                line_width=1.6, **MAT)

    # --- coordinate triad on the reference element, pushed forward by J on the image ---
    p0 = np.array([half - 0.5 * he, half - 0.5 * he, half - 0.5 * he])
    axes = np.eye(3) * 0.62
    cols = [CORAL, "#2e7d32", BLUE]
    # reference triad (xi-axes)
    for k in range(3):
        s = np.array(REF_CENTER) + p0
        pl.add_mesh(pv.Arrow(start=s, direction=axes[k], scale=0.62, tip_length=0.32,
                             tip_radius=0.08, shaft_radius=0.03), color=cols[k], **MAT)
    # pushed triad (J columns) on the image element, via finite differences of warp
    hfd = 1e-3
    base = warp(p0[None])[0]
    for k in range(3):
        Jcol = (warp((p0 + hfd * np.eye(3)[k])[None])[0] - warp((p0 - hfd * np.eye(3)[k])[None])[0]) / (2 * hfd)
        Jcol = Jcol / np.linalg.norm(Jcol) * 0.62
        pl.add_mesh(pv.Arrow(start=base, direction=Jcol, scale=0.62, tip_length=0.32,
                             tip_radius=0.08, shaft_radius=0.03), color=cols[k], **MAT)

    # --- decoder arrow D_theta between the two ---
    a = np.array([-1.30, 0.0, 0.45]); b = np.array([0.60, 0.0, 0.45])
    pl.add_mesh(pv.Tube(pointa=a, pointb=b, radius=0.035), color=INK, **MAT)
    pl.add_mesh(pv.Cone(center=b + 0.08 * np.array([1, 0, 0]), direction=(1, 0, 0),
                        height=0.22, radius=0.09), color=INK, **MAT)

    lights(pl)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(0.5, -8.2, 2.5), (0.0, 0, 0.25), (0, 0, 1)]
    pl.camera.zoom(1.55)
    img = pl.screenshot(None, return_img=True, scale=2)
    pl.close()
    return trim(img)


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=10.0)
    im = render()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.50))
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    def T(fx, fy, s, **kw):
        ax.text(fx, fy, s, transform=ax.transAxes, ha="center", va="center", **kw)
    T(0.135, 0.16, r"reference chart  $U=[-r,r]^d$", fontsize=11, color="#4a4639")
    T(0.135, 0.07, r"$\xi$", fontsize=12, color="#4a4639")
    T(0.80, 0.13, r"physical chart  $\Omega^\alpha$", fontsize=11, color=CORAL)
    T(0.46, 0.66, r"$x = D_\theta(\xi)$", fontsize=12.5, color=INK)
    T(0.30, 0.86, r"$J=\partial D_\theta/\partial\xi$", fontsize=11, color=INK)
    T(0.80, 0.92, r"$\int_{\Omega^\alpha}\!\! P:\nabla_{\! x}\delta u\,dV"
                  r"=\int_{U}\! P_{\mathrm{chart}}:\nabla_{\!\xi}\delta u\,d\xi$",
      fontsize=11, color=INK,
      bbox=dict(boxstyle="round,pad=0.3", fc="#fbf7f2", ec="0.7", lw=0.7))
    T(0.80, 0.80, r"$ P_{\mathrm{chart}}=|\!\det J|\, P\, J^{-\top}$",
      fontsize=10.5, color=INK)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "chart_pullback_3d_pub.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
