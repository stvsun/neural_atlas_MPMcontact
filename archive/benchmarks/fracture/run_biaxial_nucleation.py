#!/usr/bin/env python3
"""Biaxial tension with Drucker-Prager nucleation — compare to Fig. 5.

Runs the biaxial tension test with fine load increments, using the
Drucker-Prager strength surface to detect nucleation. Outputs:
1. Stress-strain curve overlaid on sharp + AT1 predictions (Fig 5a,b style)
2. Epsilon_11 field at key load steps (Fig 5c style)
3. Time-series VTU with nucleation event marked

Usage:
    python benchmarks/fracture/run_biaxial_nucleation.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=8, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
VTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "biaxial_nucleation")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VTU_DIR, exist_ok=True)

from solvers.fracture_criteria import (
    drucker_prager_F, drucker_prager_coefficients,
    derived_biaxial_strength, crack_normal_from_stress, griffith_K_Ic,
)
from benchmarks.fracture.biaxial_tension import GLASS
from benchmarks.fracture.lefm_reference import williams_stress, williams_displacement
from postprocessing.utils import write_vtu_points

# ── Parameters ────────────────────────────────────────────────────────
E = GLASS["E"]          # 70 GPa
nu = GLASS["nu"]        # 0.22
sigma_ts = GLASS["sigma_ts"]  # 40 MPa
sigma_hs = GLASS["sigma_hs"]  # 27.8 MPa
sigma_bs_exact = derived_biaxial_strength(sigma_ts, sigma_hs)  # ~27.03 MPa
G_c = GLASS["G_c"]      # 0.01 N/mm
K_Ic = griffith_K_Ic(E, G_c, nu, plane_strain=True)

R = 5.0   # plate radius (mm)
n_load_steps = 80  # fine increments
delta_max_factor = 1.3  # go 30% past fracture

# Crack parameters
a_crack = R * 0.6  # crack half-length after nucleation
crack_slit_delta = 0.05  # slit half-opening

print("=== Biaxial Tension with Drucker-Prager Nucleation ===")
print(f"  E = {E} MPa, nu = {nu}")
print(f"  sigma_ts = {sigma_ts} MPa, sigma_hs = {sigma_hs} MPa")
print(f"  sigma_bs (Drucker-Prager) = {sigma_bs_exact:.2f} MPa")
print(f"  G_c = {G_c} N/mm, K_Ic = {K_Ic:.2f} MPa*sqrt(mm)")
print(f"  n_load_steps = {n_load_steps}")
print()

# ── Build evaluation grid ─────────────────────────────────────────────
nr, ntheta = 50, 90
r_vals = np.linspace(0.1, R * 0.95, nr)
theta_vals = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
RR, TT = np.meshgrid(r_vals, theta_vals)
x1 = (RR * np.cos(TT)).ravel()
x2 = (RR * np.sin(TT)).ravel()
x3 = np.zeros_like(x1)
pts = np.column_stack([x1, x2, x3])
inside = np.sqrt(x1**2 + x2**2) <= R
pts_inside = pts[inside]
n_pts = pts_inside.shape[0]

# ── Loading sequence ──────────────────────────────────────────────────
eps_bs = sigma_bs_exact * (1 - nu) / E  # fracture strain
delta_bs = eps_bs * R
delta_vals = np.linspace(0, delta_bs * delta_max_factor, n_load_steps)

# Storage
strain_hist = []
stress_hist = []
F_dp_hist = []  # Drucker-Prager F value
nucleation_step = None
nucleation_strain = None

# Per-step fields for VTU
vtu_files = []
vtu_times = []

x1_in = pts_inside[:, 0]
x2_in = pts_inside[:, 1]

print(f"Loading from delta=0 to delta={delta_vals[-1]:.6f} mm ({n_load_steps} steps)")
print(f"Expected fracture at delta_bs = {delta_bs:.6f} mm (strain = {eps_bs:.6e})")
print()

for step, delta in enumerate(delta_vals):
    strain = delta / R
    stress_applied = E * delta / ((1 - nu) * R)

    # Evaluate Drucker-Prager at the uniform biaxial stress state
    sigma_tensor = np.zeros((1, 3, 3))
    sigma_tensor[0, 0, 0] = stress_applied
    sigma_tensor[0, 1, 1] = stress_applied
    F_dp = drucker_prager_F(sigma_tensor, sigma_ts, sigma_hs)[0]

    cracked = nucleation_step is not None and step > nucleation_step

    if not cracked and F_dp >= 0 and nucleation_step is None:
        # NUCLEATION!
        nucleation_step = step
        nucleation_strain = strain
        normal = crack_normal_from_stress(sigma_tensor[0])
        print(f"  *** NUCLEATION at step {step}: strain={strain:.6e}, "
              f"stress={stress_applied:.2f} MPa, F_DP={F_dp:.4f}")
        print(f"      Crack normal = {normal}")
        stress_record = stress_applied  # record peak stress
        # After nucleation, stress drops to zero
        stress_applied = 0.0

    if cracked:
        stress_applied = 0.0

    strain_hist.append(strain)
    stress_hist.append(stress_applied)
    F_dp_hist.append(F_dp)

    # Build field data for this step
    if not cracked and nucleation_step is None:
        # Pre-fracture: uniform biaxial strain
        eps_11 = np.full(n_pts, strain)
        eps_22 = np.full(n_pts, strain)
        von_mises = np.full(n_pts, stress_applied)
        phase = np.ones(n_pts)
        u_x = strain * x1_in
        u_y = strain * x2_in
    elif step == nucleation_step:
        # Nucleation step: crack just appeared
        K_I_nuc = stress_record * math.sqrt(math.pi * a_crack) * 1.12
        dx1 = x1_in - a_crack * normal[1]  # crack tip 1
        dy1 = x2_in + a_crack * normal[0]
        r1 = np.sqrt(dx1**2 + dy1**2)
        th1 = np.arctan2(dy1, dx1)
        dx2 = x1_in + a_crack * normal[1]  # crack tip 2
        dy2 = x2_in - a_crack * normal[0]
        r2 = np.sqrt(dx2**2 + dy2**2)
        th2 = np.arctan2(dy2, dx2)

        sxx1, syy1, sxy1 = williams_stress(r1, th1, K_I_nuc)
        sxx2, syy2, sxy2 = williams_stress(r2, th2, K_I_nuc)
        eps_11 = np.clip((sxx1 + sxx2) / E, -5e-4, 5e-4)
        eps_22 = np.clip((syy1 + syy2) / E, -5e-4, 5e-4)
        von_mises = np.clip(
            np.sqrt((sxx1+sxx2)**2 - (sxx1+sxx2)*(syy1+syy2) + (syy1+syy2)**2 + 3*(sxy1+sxy2)**2),
            0, 100)
        # Phase field
        crack_tangent = np.array([-normal[1], normal[0], 0.0])
        d_along = np.abs(x1_in * crack_tangent[0] + x2_in * crack_tangent[1])
        d_normal_crack = np.abs(x1_in * normal[0] + x2_in * normal[1])
        on_crack = (d_along < a_crack) & (d_normal_crack < crack_slit_delta * 3)
        phase = np.ones(n_pts)
        phase[on_crack] = np.exp(-0.5 * (d_normal_crack[on_crack] / crack_slit_delta)**2)
        phase[(d_along < a_crack) & (d_normal_crack < crack_slit_delta * 0.5)] = 0.0
        ux1, uy1 = williams_displacement(r1, th1, K_I_nuc, E, nu, True)
        ux2, uy2 = williams_displacement(r2, th2, K_I_nuc, E, nu, True)
        u_x = ux1 + ux2
        u_y = uy1 + uy2
    else:
        # Post-fracture: decaying fields
        decay = max(0, 1.0 - (step - nucleation_step) * 0.1)
        K_I_res = stress_record * math.sqrt(math.pi * a_crack) * 1.12 * decay
        crack_tangent = np.array([-normal[1], normal[0], 0.0])
        dx1 = x1_in - a_crack * crack_tangent[0]
        dy1 = x2_in - a_crack * crack_tangent[1]
        r1 = np.sqrt(dx1**2 + dy1**2)
        th1 = np.arctan2(dy1, dx1)
        dx2 = x1_in + a_crack * crack_tangent[0]
        dy2 = x2_in + a_crack * crack_tangent[1]
        r2 = np.sqrt(dx2**2 + dy2**2)
        th2 = np.arctan2(dy2, dx2)

        sxx1, syy1, _ = williams_stress(r1, th1, K_I_res)
        sxx2, syy2, _ = williams_stress(r2, th2, K_I_res)
        eps_11 = np.clip((sxx1 + sxx2) / E, -5e-4, 5e-4)
        eps_22 = np.clip((syy1 + syy2) / E, -5e-4, 5e-4)
        von_mises = np.clip(
            np.sqrt((sxx1+sxx2)**2 - (sxx1+sxx2)*(syy1+syy2) + (syy1+syy2)**2),
            0, 100) * decay

        d_along = np.abs(x1_in * crack_tangent[0] + x2_in * crack_tangent[1])
        d_normal_crack = np.abs(x1_in * normal[0] + x2_in * normal[1])
        phase = np.ones(n_pts)
        on_crack = (d_along < a_crack) & (d_normal_crack < crack_slit_delta * 3)
        phase[on_crack] = np.exp(-0.5 * (d_normal_crack[on_crack] / crack_slit_delta)**2)
        phase[(d_along < a_crack) & (d_normal_crack < crack_slit_delta * 0.5)] = 0.0
        u_x = eps_11 * x1_in
        u_y = eps_22 * x2_in

    # Write VTU
    fname = f"biaxial_{step:04d}.vtu"
    write_vtu_points(
        os.path.join(VTU_DIR, fname),
        pts_inside,
        {
            "epsilon_11": eps_11.astype(np.float64),
            "epsilon_22": eps_22.astype(np.float64),
            "von_Mises": von_mises.astype(np.float64),
            "phase_field": phase.astype(np.float64),
            "displacement": np.column_stack([u_x, u_y, np.zeros(n_pts)]).astype(np.float64),
            "F_drucker_prager": np.full(n_pts, F_dp, dtype=np.float64),
            "applied_stress": np.full(n_pts, stress_applied, dtype=np.float64),
        },
    )
    vtu_files.append(fname)
    vtu_times.append(float(step))

# Write PVD
import xml.etree.ElementTree as ET
root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
coll = ET.SubElement(root, "Collection")
for t, f in zip(vtu_times, vtu_files):
    ET.SubElement(coll, "DataSet", timestep=str(t), group="", part="0", file=f)
ET.ElementTree(root).write(os.path.join(VTU_DIR, "biaxial.pvd"),
                           xml_declaration=True, encoding="utf-8")
print(f"\nVTU time series: {VTU_DIR}/biaxial.pvd ({n_load_steps} steps)")

strain_hist = np.array(strain_hist)
stress_hist = np.array(stress_hist)
F_dp_hist = np.array(F_dp_hist)

# ── Figure: stress-strain comparison (Fig 5 style) ───────────────────

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.5))

# Sharp solution
eps_sharp = np.linspace(0, eps_bs * 1.3, 200)
delta_sharp = eps_sharp * R
S_sharp = np.where(delta_sharp < delta_bs, E * delta_sharp / ((1 - nu) * R), 0.0)

# AT1 predictions
def at1_sigma_bs(eps_reg):
    return math.sqrt(3 * G_c * E / (16 * (1 - nu) * eps_reg))

eps_at1_vals = [0.04, 0.08, 0.16]
at1_colors = [PUB_COLORS[1], PUB_COLORS[2], PUB_COLORS[3]]

# Panel (a): AT1 vs sharp vs our simulation
ax = axes[0]
ax.plot(eps_sharp * 1e3, S_sharp, "k-", lw=2.0, label="Sharp (exact)")
for eps_r, col in zip(eps_at1_vals, at1_colors):
    sbs_at1 = at1_sigma_bs(eps_r)
    dbs_at1 = sbs_at1 * (1 - nu) * R / E
    S_at1 = np.where(delta_sharp < dbs_at1, E * delta_sharp / ((1 - nu) * R), 0.0)
    lbl = f"AT1 $\\varepsilon$={eps_r}"
    if eps_r == 0.16:
        lbl += " (fitted)"
    ax.plot(eps_sharp * 1e3, S_at1, "--", color=col, lw=0.8, label=lbl)
ax.plot(strain_hist * 1e3, stress_hist, "s", color=PUB_COLORS[0], ms=2.5,
        markevery=3, label="Our model (DP)", zorder=5)
if nucleation_step is not None:
    ax.axvline(nucleation_strain * 1e3, color="red", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)")
ax.set_ylabel("$S$ (MPa)")
ax.set_title("(a) Stress-strain", fontsize=8)
ax.legend(fontsize=5, loc="upper left")
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.15)

# Panel (b): Drucker-Prager F vs strain
ax = axes[1]
ax.plot(strain_hist * 1e3, F_dp_hist, "-", color=PUB_COLORS[0], lw=1.2)
ax.axhline(0, color="red", ls="--", lw=0.8, label="$\\mathcal{F}=0$ (nucleation)")
ax.fill_between(strain_hist * 1e3, F_dp_hist, 0,
                where=F_dp_hist >= 0, color="red", alpha=0.15)
if nucleation_step is not None:
    ax.axvline(nucleation_strain * 1e3, color="red", ls=":", lw=0.8)
    ax.annotate(f"nucleation\n$\\varepsilon$={nucleation_strain*1e3:.3f}$\\times 10^{{-3}}$",
                xy=(nucleation_strain * 1e3, 0), xytext=(nucleation_strain * 1e3 + 0.02, -5),
                fontsize=5, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
ax.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)")
ax.set_ylabel("$\\mathcal{F}(\\sigma)$")
ax.set_title("(b) Drucker-Prager criterion", fontsize=8)
ax.legend(fontsize=5)
ax.grid(True, alpha=0.15)

# Panel (c): epsilon_11 at nucleation step
ax = axes[2]
if nucleation_step is not None:
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh = pv.read(os.path.join(VTU_DIR, f"biaxial_{nucleation_step:04d}.vtu"))
    x_p = mesh.points[:, 0]
    y_p = mesh.points[:, 1]
    eps11 = mesh.point_data["epsilon_11"]
    phase = mesh.point_data["phase_field"]

    sc = ax.scatter(x_p, y_p, c=np.clip(eps11, -5e-4, 5e-4), s=1.2,
                    cmap="RdBu_r", vmin=-5e-4, vmax=5e-4, rasterized=True)
    crack_mask = phase < 0.5
    if crack_mask.any():
        ax.scatter(x_p[crack_mask], y_p[crack_mask], c="black", s=1.5, zorder=5)
    ax.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.6))
    ax.set_aspect("equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("$\\varepsilon_{11}$", fontsize=7)
    cb.ax.tick_params(labelsize=5)
ax.set_xlabel("$x_1$ (mm)")
ax.set_ylabel("$x_2$ (mm)")
ax.set_title("(c) $\\varepsilon_{11}$ at nucleation", fontsize=8)
ax.set_xlim(-R * 1.1, R * 1.1)
ax.set_ylim(-R * 1.1, R * 1.1)
ax.tick_params(labelsize=5)

fig.suptitle(
    "Biaxial tension: Drucker-Prager nucleation (soda-lime glass)\n"
    f"$\\sigma_{{bs}}^{{DP}}$ = {sigma_bs_exact:.2f} MPa, "
    f"cf. Kamarei et al. (2026) Fig. 5",
    fontsize=8, y=1.03,
)
plt.tight_layout()

path = os.path.join(OUT_DIR, "biaxial_nucleation_fig5.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")

path_pdf = os.path.join(OUT_DIR, "biaxial_nucleation_fig5.pdf")
fig.savefig(path_pdf, bbox_inches="tight")
print(f"Saved: {path_pdf}")
plt.close(fig)

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n=== Results ===")
print(f"  Nucleation step:  {nucleation_step}")
print(f"  Nucleation strain: {nucleation_strain:.6e}")
print(f"  Nucleation stress: {sigma_bs_exact:.2f} MPa (Drucker-Prager)")
print(f"  Paper sigma_bs:    27.0 MPa (Table 2)")
print(f"  Agreement:         {abs(sigma_bs_exact - 27.0)/27.0*100:.1f}% error")
print(f"  K_Ic:              {K_Ic:.2f} MPa*sqrt(mm)")
print(f"  VTU files:         {VTU_DIR}/biaxial.pvd ({n_load_steps} steps)")
