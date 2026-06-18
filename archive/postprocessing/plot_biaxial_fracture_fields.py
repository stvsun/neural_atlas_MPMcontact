#!/usr/bin/env python3
"""Plot strain and phase fields for biaxial tension fracture nucleation.

Reproduces the spirit of Figs. 5(c) and 6(c) from Kamarei et al. (2026):
contour plots of the phase field over the plate at the moment of fracture
nucleation, showing where the crack forms.

Since we don't run a phase-field simulation, we construct the fields
analytically:
  - Pre-fracture: uniform biaxial strain field
  - At fracture: a through-crack nucleates at an arbitrary location
    (we choose a diametral crack for illustration)
  - The strain concentrates at the crack tips

Uses PyVista for 3D rendering and matplotlib for 2D slices.

Usage:
    python postprocessing/plot_biaxial_fracture_fields.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=9, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Material constants
GLASS_E = 70e3    # MPa
GLASS_nu = 0.22
GLASS_sigma_bs = 27.0  # MPa
R_plate = 5.0     # mm
L_plate = 0.25    # mm


def biaxial_strain_at_fracture():
    """Uniform biaxial strain at the moment of fracture."""
    return GLASS_sigma_bs * (1 - GLASS_nu) / GLASS_E


def crack_tip_strain_field(x1, x2, crack_tip, K_eff, E, nu):
    """Near-tip strain field (Mode I asymptotic) around a crack tip.

    epsilon_yy ~ K / (E * sqrt(2*pi*r)) * cos(theta/2) * (1 + sin(theta/2)*sin(3*theta/2))
    """
    dx = x1 - crack_tip[0]
    dy = x2 - crack_tip[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-6)
    theta = np.arctan2(dy, dx)

    cos_h = np.cos(theta / 2)
    sin_h = np.sin(theta / 2)
    sin_3h = np.sin(3 * theta / 2)

    eps_yy = K_eff / (E * np.sqrt(2 * np.pi * r)) * cos_h * (1 + sin_h * sin_3h)
    eps_xx = K_eff / (E * np.sqrt(2 * np.pi * r)) * cos_h * (1 - sin_h * sin_3h)
    eps_xy = K_eff / (E * np.sqrt(2 * np.pi * r)) * sin_h * cos_h * np.cos(3 * theta / 2)

    return eps_xx, eps_yy, eps_xy


def figure_strain_fields():
    """Multi-panel figure showing strain fields at fracture nucleation."""

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.7))

    nx, ny = 300, 300
    x1 = np.linspace(-R_plate * 1.05, R_plate * 1.05, nx)
    x2 = np.linspace(-R_plate * 1.05, R_plate * 1.05, ny)
    X1, X2 = np.meshgrid(x1, x2)

    # Circular plate mask
    r_grid = np.sqrt(X1**2 + X2**2)
    inside = r_grid <= R_plate

    eps_uniform = biaxial_strain_at_fracture()

    # Crack parameters: diametral crack along x1 axis
    crack_half_len = R_plate * 0.4
    crack_tip_L = np.array([-crack_half_len, 0.0])
    crack_tip_R = np.array([crack_half_len, 0.0])

    # Effective K for a through crack in a biaxial field
    K_eff = GLASS_sigma_bs * math.sqrt(math.pi * crack_half_len) * 1.12

    # ── Row 1: Pre-fracture (uniform fields) ─────────────────────────

    # (a) Strain epsilon_11 — uniform
    ax = axes[0, 0]
    eps_11_pre = np.full_like(X1, eps_uniform)
    eps_11_pre[~inside] = np.nan
    cf = ax.contourf(X1, X2, eps_11_pre * 1e3, levels=20, cmap="YlOrRd")
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    cax.set_ylabel("$\\times 10^{-3}$", fontsize=6)
    cax.tick_params(labelsize=5)
    ax.set_title("(a) $\\varepsilon_{11}$ pre-fracture", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # (b) Strain epsilon_22 — uniform (same as 11 for equi-biaxial)
    ax = axes[0, 1]
    eps_22_pre = np.full_like(X1, eps_uniform)
    eps_22_pre[~inside] = np.nan
    cf = ax.contourf(X1, X2, eps_22_pre * 1e3, levels=20, cmap="YlOrRd")
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    cax.set_ylabel("$\\times 10^{-3}$", fontsize=6)
    cax.tick_params(labelsize=5)
    ax.set_title("(b) $\\varepsilon_{22}$ pre-fracture", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # (c) Phase field v — intact (v=1 everywhere)
    ax = axes[0, 2]
    v_pre = np.ones_like(X1)
    v_pre[~inside] = np.nan
    cf = ax.contourf(X1, X2, v_pre, levels=np.linspace(0, 1, 20), cmap="bone_r")
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    cax.tick_params(labelsize=5)
    ax.set_title("(c) Phase field $v$ (intact)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # Add loading arrows (inside the plate to avoid clipping)
    for angle in [0, 90, 180, 270]:
        rad = math.radians(angle)
        x0 = R_plate * 0.7 * math.cos(rad)
        y0 = R_plate * 0.7 * math.sin(rad)
        dx = 0.6 * math.cos(rad)
        dy = 0.6 * math.sin(rad)
        axes[0, 0].annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # ── Row 2: Post-fracture (with crack) ────────────────────────────

    # Crack-tip singular strain field superimposed on uniform
    eps_xx_L, eps_yy_L, eps_xy_L = crack_tip_strain_field(
        X1, X2, crack_tip_L, K_eff, GLASS_E, GLASS_nu)
    eps_xx_R, eps_yy_R, eps_xy_R = crack_tip_strain_field(
        X1, X2, crack_tip_R, K_eff, GLASS_E, GLASS_nu)

    # Phase field: v=0 on crack, v=1 elsewhere, smooth transition
    crack_width = 0.15  # regularization width (like epsilon in phase-field)
    dist_to_crack = np.abs(X2)  # distance to crack line (y=0)
    on_crack = (X1 >= -crack_half_len) & (X1 <= crack_half_len)
    v_post = np.ones_like(X1)
    crack_zone = on_crack & (dist_to_crack < crack_width * 3)
    v_post[crack_zone] = 1.0 - np.exp(-0.5 * (dist_to_crack[crack_zone] / crack_width)**2)
    v_post[on_crack & (dist_to_crack < crack_width * 0.5)] = 0.0
    v_post[~inside] = np.nan

    # Total strain = uniform + crack-tip singular (modulated by v)
    eps_11_post = eps_uniform + (eps_xx_L + eps_xx_R) * v_post
    eps_22_post = eps_uniform + (eps_yy_L + eps_yy_R) * v_post
    eps_11_post[~inside] = np.nan
    eps_22_post[~inside] = np.nan

    vmax_strain = min(np.nanmax(eps_22_post) * 1e3, eps_uniform * 1e3 * 8)

    # (d) Strain epsilon_11 — post-fracture
    ax = axes[1, 0]
    cf = ax.contourf(X1, X2, np.clip(eps_11_post * 1e3, 0, vmax_strain),
                     levels=np.linspace(0, vmax_strain, 25), cmap="YlOrRd")
    ax.plot([-crack_half_len, crack_half_len], [0, 0], "k-", lw=2.5)
    ax.plot([-crack_half_len, crack_half_len], [0, 0], "ko", ms=3)
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    cax.set_ylabel("$\\times 10^{-3}$", fontsize=6)
    cax.tick_params(labelsize=5)
    ax.set_title("(d) $\\varepsilon_{11}$ post-fracture", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # (e) Strain epsilon_22 — post-fracture
    ax = axes[1, 1]
    cf = ax.contourf(X1, X2, np.clip(eps_22_post * 1e3, 0, vmax_strain),
                     levels=np.linspace(0, vmax_strain, 25), cmap="YlOrRd")
    ax.plot([-crack_half_len, crack_half_len], [0, 0], "k-", lw=2.5)
    ax.plot([-crack_half_len, crack_half_len], [0, 0], "ko", ms=3)
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    cax.set_ylabel("$\\times 10^{-3}$", fontsize=6)
    cax.tick_params(labelsize=5)
    ax.set_title("(e) $\\varepsilon_{22}$ post-fracture", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # (f) Phase field v — cracked
    ax = axes[1, 2]
    cf = ax.contourf(X1, X2, v_post, levels=np.linspace(0, 1, 25), cmap="bone_r")
    ax.plot([-crack_half_len, crack_half_len], [0, 0], "r-", lw=2.0)
    ax.add_patch(Circle((0, 0), R_plate, fill=False, ec="k", lw=1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cf, cax=cax)
    cax.tick_params(labelsize=5)
    ax.set_title("(f) Phase field $v$ (fractured)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=5)

    # Annotate crack
    ax.annotate("crack ($v$=0)", xy=(0, 0), xytext=(2.5, 3.0),
                fontsize=6, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    fig.suptitle(
        "Biaxial tension: strain and phase fields at fracture nucleation\n"
        "(soda-lime glass, $\\sigma_{bs}$ = 27 MPa, Kamarei et al. 2026)",
        fontsize=9, y=1.03,
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "biaxial_fracture_fields.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

    path_pdf = os.path.join(OUT_DIR, "biaxial_fracture_fields.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    print(f"Saved: {path_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    figure_strain_fields()
