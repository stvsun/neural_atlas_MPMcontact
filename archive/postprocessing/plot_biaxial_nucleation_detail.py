#!/usr/bin/env python3
"""Detailed comparison of biaxial nucleation results with Fig. 5 of Kamarei et al.

Produces a 2-row figure:
  Row 1: stress-strain (AT1 + sharp + our DP model), Drucker-Prager F vs strain
  Row 2: epsilon_11 field at 6 key time steps through nucleation

Usage:
    python postprocessing/plot_biaxial_nucleation_detail.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyvista as pv
pv.OFF_SCREEN = True

from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W
from solvers.fracture_criteria import derived_biaxial_strength, griffith_K_Ic
from benchmarks.fracture.biaxial_tension import GLASS

set_pub_style(fontsize=8, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
VTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "biaxial_nucleation")

E = GLASS["E"]
nu = GLASS["nu"]
sigma_ts = GLASS["sigma_ts"]
sigma_hs = GLASS["sigma_hs"]
sigma_bs = derived_biaxial_strength(sigma_ts, sigma_hs)
G_c = GLASS["G_c"]
R = 5.0

eps_bs = sigma_bs * (1 - nu) / E
delta_bs = eps_bs * R

fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.85))

# ── Row 1: stress-strain + Drucker-Prager ─────────────────────────

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)

# Read simulation history from VTU files
n_steps = 80
strains, stresses, F_dp_vals = [], [], []
nucleation_step = None

for step in range(n_steps):
    fpath = os.path.join(VTU_DIR, f"biaxial_{step:04d}.vtu")
    if not os.path.exists(fpath):
        break
    mesh = pv.read(fpath)
    s_app = mesh.point_data["applied_stress"][0]
    f_dp = mesh.point_data["F_drucker_prager"][0]
    eps = mesh.point_data["epsilon_11"].mean()

    strains.append(eps)
    stresses.append(s_app)
    F_dp_vals.append(f_dp)

    if f_dp >= 0 and nucleation_step is None and s_app == 0:
        nucleation_step = step

strains = np.array(strains)
stresses = np.array(stresses)
F_dp_vals = np.array(F_dp_vals)

# Sharp solution
eps_fine = np.linspace(0, eps_bs * 1.3, 300)
S_sharp = np.where(eps_fine < eps_bs, E * eps_fine / (1 - nu), 0.0)

# AT1 predictions
def at1_bs(eps_r):
    return math.sqrt(3 * G_c * E / (16 * (1 - nu) * eps_r))

# Panel (a): stress-strain
ax1.plot(eps_fine * 1e3, S_sharp, "k-", lw=2.0, label="Sharp (exact)")
for eps_r, col, ls in [(0.04, PUB_COLORS[1], "--"), (0.08, PUB_COLORS[2], "-."), (0.16, PUB_COLORS[3], ":")]:
    sbs = at1_bs(eps_r)
    eps_at1 = sbs * (1 - nu) / E
    S_at1 = np.where(eps_fine < eps_at1, E * eps_fine / (1 - nu), 0.0)
    lbl = f"AT1 $\\varepsilon$={eps_r}" + (" (fitted)" if eps_r == 0.16 else "")
    ax1.plot(eps_fine * 1e3, S_at1, ls, color=col, lw=0.9, label=lbl)
ax1.plot(strains * 1e3, stresses, "o", color=PUB_COLORS[0], ms=2, markevery=3,
         label="Our model (DP)", zorder=5)
ax1.axhline(sigma_bs, color="gray", ls=":", lw=0.5, alpha=0.5)
ax1.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)", fontsize=7)
ax1.set_ylabel("$S$ (MPa)", fontsize=7)
ax1.set_title("(a) Stress-strain", fontsize=8)
ax1.legend(fontsize=4.5, loc="upper left")
ax1.set_ylim(0, 50)
ax1.set_xlim(0, eps_bs * 1.3 * 1e3)
ax1.grid(True, alpha=0.15)
ax1.tick_params(labelsize=5)

# Panel (b): F(sigma) vs strain
ax2.plot(strains * 1e3, F_dp_vals, "-", color=PUB_COLORS[0], lw=1.2)
ax2.axhline(0, color="red", ls="--", lw=0.8, label="$\\mathcal{F}=0$")
ax2.fill_between(strains * 1e3, F_dp_vals, 0,
                 where=F_dp_vals >= 0, color="red", alpha=0.1)
ax2.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)", fontsize=7)
ax2.set_ylabel("$\\mathcal{F}(\\sigma)$", fontsize=7)
ax2.set_title("(b) Drucker-Prager $\\mathcal{F}$", fontsize=8)
ax2.legend(fontsize=5)
ax2.grid(True, alpha=0.15)
ax2.tick_params(labelsize=5)

# Panel (c): epsilon_11 at nucleation
step_nuc = nucleation_step if nucleation_step else 61
mesh_nuc = pv.read(os.path.join(VTU_DIR, f"biaxial_{step_nuc:04d}.vtu"))
x_p = mesh_nuc.points[:, 0]
y_p = mesh_nuc.points[:, 1]
eps11_nuc = mesh_nuc.point_data["epsilon_11"]
phase_nuc = mesh_nuc.point_data["phase_field"]

sc = ax3.scatter(x_p, y_p, c=np.clip(eps11_nuc, -5e-4, 5e-4), s=1.0,
                 cmap="RdBu_r", vmin=-5e-4, vmax=5e-4, rasterized=True)
crack_mask = phase_nuc < 0.5
if crack_mask.any():
    ax3.scatter(x_p[crack_mask], y_p[crack_mask], c="black", s=1.5, zorder=5)
ax3.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.6))
ax3.set_aspect("equal")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(sc, cax=cax)
cb.set_label("$\\varepsilon_{11}$", fontsize=6)
cb.ax.tick_params(labelsize=4)
ax3.set_title(f"(c) $\\varepsilon_{{11}}$ at nucleation (step {step_nuc})", fontsize=8)
ax3.set_xlim(-R * 1.1, R * 1.1)
ax3.set_ylim(-R * 1.1, R * 1.1)
ax3.tick_params(labelsize=5)

# ── Row 2: epsilon_11 at 6 key steps ─────────────────────────────

key_steps = [0, 20, 40, 58, step_nuc, min(step_nuc + 5, n_steps - 1)]
key_labels = ["Step 0\n$S$=0", "Step 20", "Step 40", "Step 58",
              f"Step {step_nuc}\n(nucleation)", f"Step {step_nuc+5}\n(post-frac)"]

eps_clim = 5e-4

for idx, (step, label) in enumerate(zip(key_steps, key_labels)):
    ax = fig.add_subplot(2, 6, 7 + idx)
    fpath = os.path.join(VTU_DIR, f"biaxial_{step:04d}.vtu")
    if not os.path.exists(fpath):
        ax.set_visible(False)
        continue

    mesh = pv.read(fpath)
    xp = mesh.points[:, 0]
    yp = mesh.points[:, 1]
    eps11 = mesh.point_data["epsilon_11"]
    phase = mesh.point_data["phase_field"]

    sc = ax.scatter(xp, yp, c=np.clip(eps11, -eps_clim, eps_clim), s=0.3,
                    cmap="RdBu_r", vmin=-eps_clim, vmax=eps_clim, rasterized=True)
    crack_m = phase < 0.5
    if crack_m.any():
        ax.scatter(xp[crack_m], yp[crack_m], c="black", s=0.5, zorder=5)
    ax.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.4))
    ax.set_aspect("equal")
    ax.set_xlim(-R * 1.15, R * 1.15)
    ax.set_ylim(-R * 1.15, R * 1.15)
    ax.set_title(label, fontsize=5, pad=1)
    ax.tick_params(labelsize=3)
    if idx == 0:
        ax.set_ylabel("$x_2$", fontsize=5)

# Shared colorbar for row 2
cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.35])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-eps_clim, eps_clim))
sm.set_array([])
cb2 = fig.colorbar(sm, cax=cbar_ax)
cb2.set_label("$\\varepsilon_{11}$", fontsize=6)
cb2.ax.tick_params(labelsize=4)

fig.suptitle(
    "Biaxial tension with Drucker-Prager nucleation (soda-lime glass)\n"
    f"$\\sigma_{{bs}}^{{DP}}$ = {sigma_bs:.2f} MPa — cf. Kamarei et al. (2026) Fig. 5",
    fontsize=9, y=1.0,
)

fig.subplots_adjust(hspace=0.45, wspace=0.4, right=0.90)

path = os.path.join(OUT_DIR, "biaxial_nucleation_detail.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")

path_pdf = os.path.join(OUT_DIR, "biaxial_nucleation_detail.pdf")
fig.savefig(path_pdf, bbox_inches="tight")
print(f"Saved: {path_pdf}")
plt.close(fig)
