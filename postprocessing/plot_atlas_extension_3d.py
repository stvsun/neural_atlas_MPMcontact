"""Figure 2 (3-D): the contact formulation as an extension of the neural atlas.

Two lit 3-D panels, composited with LaTeX labels in matplotlib:
  (a) intra-body atlas  -- one body Omega covered by two overlapping volumetric charts
      Omega_i, Omega_j; the intra-body transition map psi_ij = pi_j o phi_i re-expresses a
      point of the overlap Omega_ij in the neighbouring chart.
  (b) inter-body transition map -- two bodies Omega_A, Omega_B in near-contact; the same
      composition across the gap, tau_AB = phi_B^{-1} o phi_A, carries a boundary point
      x = phi_A(theta_A) to its mate phi_B(psi_B) and returns the gap g_N and normal n.

Run:  <venv>/bin/python postprocessing/plot_atlas_extension_3d.py
Out:  figures/atlas_extension_3d_pub.png (+ .pdf)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")

CORAL = "#BE5536"; BLUE = "#3E6E9E"; GRAYB = "#A9A498"; INK = "#2b2b2b"


MAT = dict(ambient=0.34, diffuse=0.86, smooth_shading=True, specular=0.4,
           specular_power=20, show_scalar_bar=False)


def ellipsoid(center, axes, theta=0.0, nsub=5):
    s = pv.Icosphere(radius=1.0, nsub=nsub)
    P = s.points.copy()
    P = P * np.asarray(axes)
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])
    s.points = P @ R.T + np.asarray(center)
    s.compute_normals(inplace=True, auto_orient_normals=True)
    return s


def lights(pl):
    pl.add_light(pv.Light(position=(5, -4, 6), focal_point=(0, 0, 0), intensity=0.80))
    pl.add_light(pv.Light(position=(-5, -2, 3), focal_point=(0, 0, 0), intensity=0.34))
    pl.add_light(pv.Light(position=(0, 6, 4), focal_point=(0, 0, 0), intensity=0.30))
    pl.add_light(pv.Light(position=(2, -5, -5), focal_point=(0, 0, 0), intensity=0.34))


def trim(img, bg=248):
    mask = (img[:, :, :3] < bg).any(2)
    ys, xs = np.where(mask)
    p = 8
    return img[max(ys.min() - p, 0):ys.max() + p, max(xs.min() - p, 0):xs.max() + p]


def panel_atlas(size=(820, 760)):
    pl = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    pl.set_background("white")
    # body Omega: large translucent lumpy ellipsoid
    body = ellipsoid((0, 0, 0), (1.9, 1.5, 1.35), theta=0.3)
    pl.add_mesh(body, color=GRAYB, opacity=0.16, smooth_shading=True, specular=0.15,
                ambient=0.5, show_scalar_bar=False)
    pl.add_mesh(body, color=GRAYB, style="wireframe", line_width=0.35, opacity=0.05,
                show_scalar_bar=False)
    # two overlapping charts
    ci = ellipsoid((-0.55, 0.30, 0.18), (0.95, 0.78, 0.72))
    cj = ellipsoid((0.55, -0.28, -0.16), (1.0, 0.82, 0.74))
    pl.add_mesh(cj, color=BLUE, opacity=0.60, **MAT)
    pl.add_mesh(ci, color=CORAL, opacity=0.66, **MAT)
    # intra-body transition arrow psi_ij in the overlap
    a = np.array([-0.05, 0.06, 0.12]); b = np.array([0.30, -0.10, -0.02])
    pl.add_mesh(pv.Tube(pointa=a, pointb=b, radius=0.028), color=INK, show_scalar_bar=False)
    pl.add_mesh(pv.Cone(center=b + 0.06 * (b - a) / np.linalg.norm(b - a),
                        direction=b - a, height=0.16, radius=0.07), color=INK,
                show_scalar_bar=False)
    lights(pl)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(4.6, -4.8, 3.0), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(1.35)
    img = pl.screenshot(None, return_img=True, scale=2)
    pl.close()
    return trim(img)


def panel_contact(size=(820, 760)):
    pl = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    pl.set_background("white")
    # body A (coral) above, body B (blue) below, facing with a thin gap
    A = ellipsoid((0, 0, 1.18), (1.5, 1.2, 0.8))
    B = ellipsoid((0, 0, -1.18), (1.5, 1.2, 0.8))
    pl.add_mesh(B, color=BLUE, **MAT)
    pl.add_mesh(A, color=CORAL, **MAT)
    # contact point x on A's lower pole, foot on B's upper pole
    x_pt = np.array([0.0, 0.0, 0.40])      # on A's underside
    foot = np.array([0.0, 0.0, -0.40])     # on B's top
    pl.add_mesh(pv.Sphere(radius=0.07, center=x_pt), color=INK, show_scalar_bar=False)
    pl.add_mesh(pv.Sphere(radius=0.07, center=foot), color=INK, show_scalar_bar=False)
    # gap g_N (thin tube between x and foot)
    pl.add_mesh(pv.Tube(pointa=x_pt, pointb=foot, radius=0.018), color="#6b6b6b",
                show_scalar_bar=False)
    # outward normal n of B at the foot (pointing up toward A)
    nrm = np.array([0.0, 0.0, 1.0])
    pl.add_mesh(pv.Arrow(start=foot, direction=nrm, scale=0.95, tip_length=0.28,
                         tip_radius=0.07, shaft_radius=0.028), color=BLUE, show_scalar_bar=False)
    lights(pl)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(4.8, -4.4, 1.2), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(1.3)
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

    imA = panel_atlas(); imB = panel_contact()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.52))
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # ---- (a) intra-body atlas ----
    axL.imshow(imA)
    axL.set_title("(a) intra-body atlas", fontsize=11, fontweight="bold", pad=4)
    def Lt(fx, fy, s, **kw):
        axL.text(fx, fy, s, transform=axL.transAxes, ha="center", va="center", **kw)
    Lt(0.86, 0.90, r"$\Omega$", fontsize=13, color="#6c6757")
    Lt(0.265, 0.80, r"$\Omega_i$", fontsize=12, color=CORAL,
       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    Lt(0.70, 0.205, r"$\Omega_j$", fontsize=12, color=BLUE,
       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    Lt(0.50, 0.585, r"$\Omega_{ij}$", fontsize=10, color=INK,
       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    Lt(0.50, 0.40, r"$\psi_{ij}=\pi_j\!\circ\!\varphi_i$", fontsize=11, color=INK,
       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.6))

    # ---- (b) inter-body transition map ----
    axR.imshow(imB)
    axR.set_title("(b) inter-body transition map (contact)", fontsize=11, fontweight="bold", pad=4)
    def Rt(fx, fy, s, **kw):
        axR.text(fx, fy, s, transform=axR.transAxes, ha="center", va="center", **kw)
    Rt(0.20, 0.86, r"$\Omega_A$", fontsize=13, color=CORAL)
    Rt(0.20, 0.14, r"$\Omega_B$", fontsize=13, color=BLUE)
    Rt(0.70, 0.62, r"$\mathbf{x}=\varphi_A(\theta_A)$", fontsize=10.5, color=INK)
    Rt(0.72, 0.38, r"$\varphi_B(\psi_B)$", fontsize=10.5, color=INK)
    Rt(0.40, 0.50, r"$g_N$", fontsize=11, color="#5a5a5a")
    Rt(0.595, 0.66, r"$\mathbf{n}$", fontsize=12, color=BLUE)
    Rt(0.50, 0.05, r"$\tau_{AB}=\varphi_B^{-1}\!\circ\!\varphi_A$", fontsize=11.5, color=INK,
       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.6))

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "atlas_extension_3d_pub.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    build()
