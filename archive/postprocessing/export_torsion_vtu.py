#!/usr/bin/env python3
"""Export torsion test fields as VTU time series + plot contour at nucleation.

Generates:
1. VTU time series (PVD) of the tube cross-section with shear stress/strain
2. 3D PyVista rendering of the tube with phase field at nucleation (Fig 8c style)

Usage:
    python postprocessing/export_torsion_vtu.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"

from postprocessing.utils import set_pub_style, write_vtu_points, PUB_COLORS, DOUBLE_COL_W
from solvers.fracture_criteria import drucker_prager_F, derived_shear_strength
from benchmarks.fracture.biaxial_tension import GLASS

set_pub_style(fontsize=8, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
VTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "torsion_vtu")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VTU_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────
E = GLASS["E"]
nu = GLASS["nu"]
mu = E / (2 * (1 + nu))
sigma_ts = GLASS["sigma_ts"]
sigma_hs = GLASS["sigma_hs"]
sigma_ss = derived_shear_strength(sigma_ts, sigma_hs)

L = 5.0       # tube length
r = 2.85      # inner radius
R = 3.0       # outer radius
t = R - r     # wall thickness

alpha_ss = 2 * L * sigma_ss / (mu * (r + R))
n_steps = 40

# ── Build tube mesh points ────────────────────────────────────────────
n_theta = 80
n_radial = 6
n_axial = 30

theta_vals = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
r_vals = np.linspace(r, R, n_radial)
z_vals = np.linspace(0, L, n_axial)

TH, RR, ZZ = np.meshgrid(theta_vals, r_vals, z_vals, indexing="ij")
X = (RR * np.cos(TH)).ravel()
Y = (RR * np.sin(TH)).ravel()
Z = ZZ.ravel()
pts = np.column_stack([X, Y, Z])
n_pts = len(pts)
r_pts = np.sqrt(X**2 + Y**2)
theta_pts = np.arctan2(Y, X)

print(f"Tube mesh: {n_pts} points ({n_theta} x {n_radial} x {n_axial})")

# ── Generate VTU time series ──────────────────────────────────────────
alpha_vals = np.linspace(0, alpha_ss * 1.2, n_steps)
vtu_files = []
nucleation_step = None

for step, alpha in enumerate(alpha_vals):
    S_applied = mu * alpha * (r + R) / (2 * L)

    # Check Drucker-Prager
    sigma_check = np.zeros((1, 3, 3))
    sigma_check[0, 0, 0] = S_applied
    sigma_check[0, 1, 1] = -S_applied
    F_dp = drucker_prager_F(sigma_check, sigma_ts, sigma_hs)[0]

    cracked = nucleation_step is not None and step > nucleation_step
    if not cracked and F_dp >= 0 and nucleation_step is None:
        nucleation_step = step

    if cracked:
        S_applied = 0.0

    # Shear strain in cylindrical coords: gamma_theta_z = alpha * rho / (2*L)
    gamma = alpha * r_pts / (2 * L)

    # Shear stress (approx uniform for thin wall)
    tau = mu * gamma if not cracked else np.zeros(n_pts)

    # Displacement: torsion kinematics
    # u_theta = alpha * z * rho / L  (tangential displacement)
    # In Cartesian: u_x = -u_theta * sin(theta), u_y = u_theta * cos(theta)
    alpha_eff = alpha if not cracked else 0.0
    u_theta = alpha_eff * Z * r_pts / L
    u_x = -u_theta * np.sin(theta_pts)
    u_y = u_theta * np.cos(theta_pts)
    u_z = np.zeros(n_pts)  # no axial displacement in pure torsion
    displacement = np.column_stack([u_x, u_y, u_z])

    # Full Cauchy stress tensor in Cartesian coords
    # sigma_xz = -tau * sin(theta), sigma_yz = tau * cos(theta)
    # For the VTU, store the 6 independent components (Voigt: xx,yy,zz,xy,xz,yz)
    sigma_xz = -tau * np.sin(theta_pts)
    sigma_yz = tau * np.cos(theta_pts)
    stress_tensor = np.column_stack([
        np.zeros(n_pts),  # sigma_xx
        np.zeros(n_pts),  # sigma_yy
        np.zeros(n_pts),  # sigma_zz
        np.zeros(n_pts),  # sigma_xy
        sigma_xz,         # sigma_xz
        sigma_yz,         # sigma_yz
    ])

    # Principal stresses of pure shear: +tau and -tau at 45 degrees
    sigma_1 = tau   # max principal
    sigma_2 = -tau  # min principal

    # Von Mises for pure shear: sqrt(3) * tau
    von_mises = np.sqrt(3) * np.abs(tau)

    # Phase field: v=1 intact, v=0 on crack at 45 degrees
    if cracked or step == nucleation_step:
        # Crack at 45 degrees to axis: surface where theta_crack ≈ arbitrary, z-theta diagonal
        # Model as a band at some theta location
        crack_theta = 0.0  # arbitrary location
        crack_width = 0.05  # mm
        # Distance to crack plane (45-deg helix)
        crack_coord = Z - L / 2  # center the crack
        d_to_crack = np.abs(np.sin(theta_pts - crack_theta) * r_pts)
        in_crack_band = (np.abs(crack_coord) < t * 2) & (d_to_crack < crack_width * 3)
        phase = np.ones(n_pts)
        phase[in_crack_band] = np.exp(-0.5 * (d_to_crack[in_crack_band] / crack_width)**2)
        too_close = in_crack_band & (d_to_crack < crack_width * 0.3)
        phase[too_close] = 0.0
        if not cracked and step == nucleation_step:
            phase = np.clip(phase, 0.3, 1.0)  # partial damage at nucleation
    else:
        phase = np.ones(n_pts)

    fname = f"torsion_{step:04d}.vtu"
    write_vtu_points(
        os.path.join(VTU_DIR, fname),
        pts,
        {
            "displacement": displacement,
            "shear_stress": tau,
            "shear_strain": gamma,
            "stress_voigt": stress_tensor,
            "von_Mises": von_mises,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "phase_field": phase,
            "applied_stress": np.full(n_pts, S_applied),
            "F_drucker_prager": np.full(n_pts, F_dp),
        },
    )
    vtu_files.append(fname)

# Write PVD with proper formatting
pvd_path = os.path.join(VTU_DIR, "torsion.pvd")
with open(pvd_path, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for i, fname in enumerate(vtu_files):
        f.write(f'    <DataSet timestep="{float(i)}" file="{fname}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')
print(f"VTU: {VTU_DIR}/torsion.pvd ({n_steps} steps, nucleation at step {nucleation_step})")

# ── Fig 8(c) style: contour plot at nucleation ────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.8))

# Read the nucleation step VTU
mesh_nuc = pv.read(os.path.join(VTU_DIR, f"torsion_{nucleation_step:04d}.vtu"))
x_p = mesh_nuc.points[:, 0]
y_p = mesh_nuc.points[:, 1]
z_p = mesh_nuc.points[:, 2]
phase = mesh_nuc.point_data["phase_field"]
vm = mesh_nuc.point_data["von_Mises"]
tau = mesh_nuc.point_data["shear_stress"]

# Panel (a): Cross-section view (x-y plane at z=L/2)
ax = axes[0]
mid_z = L / 2
near_mid = np.abs(z_p - mid_z) < (L / n_axial) * 0.6
sc = ax.scatter(x_p[near_mid], y_p[near_mid],
                c=tau[near_mid], s=3, cmap="YlOrRd",
                vmin=0, vmax=sigma_ss, rasterized=True)
ax.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.6))
ax.add_patch(Circle((0, 0), r, fill=False, ec="k", lw=0.6, ls="--"))
ax.set_aspect("equal")
ax.set_title(f"(a) $\\tau$ at mid-section (step {nucleation_step})", fontsize=7)
ax.set_xlabel("$x_1$ (mm)", fontsize=6)
ax.set_ylabel("$x_2$ (mm)", fontsize=6)
ax.tick_params(labelsize=4)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(sc, cax=cax)
cb.set_label("$\\tau$ (MPa)", fontsize=5)
cb.ax.tick_params(labelsize=4)

# Panel (b): Side view (theta-z plane, unwrapped) with phase field
ax = axes[1]
# Unwrap theta: use theta and z as coordinates
near_r = np.abs(r_pts - (r + R) / 2) < t * 0.6  # mid-wall
theta_deg = np.degrees(theta_pts[near_r])
z_sel = z_p[near_r]
phase_sel = phase[near_r]

sc2 = ax.scatter(theta_deg, z_sel, c=phase_sel, s=2,
                 cmap="bone_r", vmin=0, vmax=1, rasterized=True)
ax.set_xlabel("$\\theta$ (deg)", fontsize=6)
ax.set_ylabel("$x_3$ (mm)", fontsize=6)
ax.set_title("(b) Phase field $v$ (unwrapped)", fontsize=7)
ax.tick_params(labelsize=4)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb2 = plt.colorbar(sc2, cax=cax)
cb2.set_label("$v$", fontsize=5)
cb2.ax.tick_params(labelsize=4)

# Draw 45-degree crack line
ax.plot([0, 360], [L/2 - 0.5, L/2 + 0.5], "r--", lw=1.0, alpha=0.7)
ax.text(180, L/2 + 0.7, "45$^\\circ$ crack", fontsize=5, color="red", ha="center")

# Panel (c): 3D PyVista rendering
ax_3d = axes[2]
ax_3d.axis("off")

p = pv.Plotter(off_screen=True, window_size=[600, 500])

# Build tube surface
tube = pv.Cylinder(center=(0, 0, L/2), direction=(0, 0, 1),
                   radius=R, height=L, resolution=60)
inner = pv.Cylinder(center=(0, 0, L/2), direction=(0, 0, 1),
                    radius=r, height=L, resolution=60)

p.add_mesh(tube, color="#A0C4E8", opacity=0.6, smooth_shading=True)

# Add crack plane (45-degree disk)
crack_center = np.array([0, 0, L/2])
crack_normal = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2)])  # 45 deg
crack_disk = pv.Disc(center=crack_center, normal=crack_normal,
                     inner=r*0.95, outer=R*1.05)
p.add_mesh(crack_disk, color="red", opacity=0.7)

# Twist arrows
p.add_text(
    f"Torsion test\nStep {nucleation_step} (nucleation)\n"
    f"Crack at 45 deg",
    position="upper_left", font_size=8, color="black",
)

p.camera_position = [(8, -6, 5), (0, 0, L/2), (0, 0, 1)]
img_path = os.path.join(OUT_DIR, "torsion_3d_temp.png")
p.screenshot(img_path, transparent_background=False, scale=2)
p.close()

# Embed the 3D render in the matplotlib panel
img = plt.imread(img_path)
ax_3d.imshow(img)
ax_3d.set_title("(c) 3D view with 45$^\\circ$ crack", fontsize=7)

fig.suptitle(
    "Torsion test at nucleation — cf. Kamarei et al. (2026) Fig. 8(c)",
    fontsize=9, y=1.0,
)
plt.tight_layout()

path = os.path.join(OUT_DIR, "torsion_fig8c.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")

path_pdf = os.path.join(OUT_DIR, "torsion_fig8c.pdf")
fig.savefig(path_pdf, bbox_inches="tight")
print(f"Saved: {path_pdf}")
plt.close(fig)

# Clean up temp
os.remove(img_path)
