#!/usr/bin/env python3
"""Torsion test with REAL multi-chart FEM solve + parallel Schwarz.

Uses 4 coordinate charts around the tube circumference, each covering
a 120-degree arc with overlap. The charts are solved in parallel via
ThreadPoolExecutor and coupled via multiplicative Schwarz.

At each load step:
  1. Apply torsion BCs on boundary nodes of each chart
  2. Solve each chart in parallel (Newton-Raphson)
  3. Exchange boundary displacements (Schwarz coupling)
  4. Extract Cauchy stress, check Drucker-Prager
  5. Output VTU per chart per step

Usage:
    python benchmarks/fracture/run_torsion_chart_fem.py
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver
from solvers.fem.linear_elastic import make_linear_elastic
from solvers.fem.analytic_decoders import TubeSectorDecoder
from solvers.fracture_criteria import (
    drucker_prager_F, cauchy_from_first_piola, derived_shear_strength,
)
from benchmarks.fracture.biaxial_tension import GLASS
from postprocessing.utils import write_vtu_points

OUT_VTU = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "torsion_chart_fem")
OUT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
os.makedirs(OUT_VTU, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ── Material ──────────────────────────────────────────────────────────
E = GLASS["E"]
nu = GLASS["nu"]
mu_mat = E / (2 * (1 + nu))
sigma_ts = GLASS["sigma_ts"]
sigma_hs = GLASS["sigma_hs"]
sigma_ss = derived_shear_strength(sigma_ts, sigma_hs)

# ── Geometry ──────────────────────────────────────────────────────────
L = 5.0       # tube length
r_inner = 2.85
R_outer = 3.0
r_mid = (r_inner + R_outer) / 2
t_wall = R_outer - r_inner

# ── Build 4 coordinate charts ────────────────────────────────────────
# Each chart covers a sector of the tube + axial extent
# Charts are centered at 0, 90, 180, 270 degrees with 120-deg coverage
# This gives ~30-degree overlap between adjacent charts

n_cells = 12  # per chart
chart_r = 1.0  # chart reference domain radius

# Use 4 charts around the circumference, each covering full axial length
# More charts = finer circumferential resolution, but each has full z-coverage
# so both tube ends are captured by every chart
n_circ = 4
n_axial = 1  # single axial layer (full length) — avoids z-coupling issues
n_charts = n_circ * n_axial

print("=== Torsion Test: Multi-Chart Parallel FEM ===")
print(f"  Tube: L={L}, r={r_inner}, R={R_outer}, t={t_wall:.2f} mm")
print(f"  Material: E={E}, nu={nu}, mu={mu_mat:.1f} MPa")
print(f"  sigma_ss (DP) = {sigma_ss:.2f} MPa")
print(f"  Charts: {n_charts}, n_cells={n_cells}")
print()

# Create chart solvers with analytical TubeSectorDecoder
# 8 charts: 4 circumferential x 2 axial, each covering a smaller region
chart_solvers = []
chart_decoders = []
chart_seeds = []
theta_span = math.pi / 1.5  # 120 degrees per chart (30-deg overlap between neighbors)

theta_centers = [i * math.pi / 2 for i in range(n_circ)]  # 0, 90, 180, 270

for ti, theta_c in enumerate(theta_centers):
    decoder = TubeSectorDecoder(
        theta_center=theta_c,
        theta_span=theta_span,
        r_mid=r_mid,
        t_half=t_wall / 2,  # exact wall thickness
        z_center=L / 2,
        L_half=L / 2,  # full axial length
    ).double()

    solver = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=chart_r,
        chart_decoder=decoder,
        decoder_kwargs={},
        device="cpu", dtype=torch.float64,
    )

    chart_solvers.append(solver)
    chart_decoders.append(decoder)
    chart_seeds.append([r_mid * math.cos(theta_c), r_mid * math.sin(theta_c), L / 2])

seeds_t = torch.tensor(chart_seeds, dtype=torch.float64)

# Neighbor graph: circular (each chart neighbors its two adjacent sectors)
neighbors = [
    [(ti + 1) % n_charts, (ti - 1) % n_charts]
    for ti in range(n_charts)
]

total_nodes = sum(s.n_nodes for s in chart_solvers)
total_elements = sum(s.n_elements for s in chart_solvers)
print(f"  Total: {total_nodes} nodes, {total_elements} elements across {n_charts} charts")
print(f"  Nodes per chart: {[s.n_nodes for s in chart_solvers]}")

stress_fn, tangent_fn = make_linear_elastic(E, nu)

# ── Loading ───────────────────────────────────────────────────────────
alpha_ss = 2 * L * sigma_ss / (mu_mat * (r_inner + R_outer))
n_load_steps = 30
alpha_max = alpha_ss * 1.2
alpha_steps = np.linspace(0, alpha_max, n_load_steps + 1)[1:]

print(f"  alpha_ss = {alpha_ss:.6f} rad ({math.degrees(alpha_ss):.4f} deg)")
print(f"  Load steps: {n_load_steps}")
print()

# ── Torsion BC function ──────────────────────────────────────────────
def make_torsion_bc(alpha):
    """Create BC function for torsion angle alpha.

    Only constrain the z=0 and z=L faces (axial ends).
    z=0: fixed (u=0). z=L: prescribed twist u_theta = alpha*rho.
    Circumferential and radial faces are FREE (traction-free).
    """
    def phys_bc_fn(nodes_phys):
        n = len(nodes_phys)
        u_bc = np.zeros((n, 3))
        mask = np.zeros(n, dtype=bool)

        x, y, z = nodes_phys[:, 0], nodes_phys[:, 1], nodes_phys[:, 2]
        rho = np.sqrt(x**2 + y**2)
        theta_node = np.arctan2(y, x)

        tol = L * 0.05
        z0_face = z < tol          # z=0 face: fixed
        zL_face = z > L - tol      # z=L face: twist applied

        # Fix z=0 face completely
        mask[z0_face] = True
        u_bc[z0_face] = 0.0

        # Apply twist on z=L face
        mask[zL_face] = True
        u_theta = alpha * rho[zL_face]
        u_bc[zL_face, 0] = -u_theta * np.sin(theta_node[zL_face])
        u_bc[zL_face, 1] = u_theta * np.cos(theta_node[zL_face])
        u_bc[zL_face, 2] = 0.0

        return u_bc, mask
    return phys_bc_fn

# ── Incremental solve ─────────────────────────────────────────────────
strain_hist = []
stress_hist = []
F_dp_hist = []
nucleation_step = None
vtu_files = []

t_start = time.time()

for step, alpha in enumerate(alpha_steps):
    gamma = alpha * (r_inner + R_outer) / (4 * L)  # average shear strain
    S_expected = mu_mat * alpha * (r_inner + R_outer) / (2 * L)

    t0 = time.time()

    phys_bc_fn = make_torsion_bc(alpha)

    # Build Schwarz solver (parallel)
    schwarz = SchwarzVectorFEMSolver(
        chart_solvers=chart_solvers,
        seeds=seeds_t,
        decoders=chart_decoders,
        neighbors=neighbors,
        parallel=True,
        n_workers=4,
    )

    # Solve with under-relaxation for stable Schwarz convergence
    u_charts = schwarz.solve(
        stress_fn, tangent_fn, phys_bc_fn,
        max_schwarz_iters=20,
        tol=1e-3,
        newton_max_iter=10,
        newton_tol=1e-8,
        relaxation=0.3,  # damped to prevent oscillation
    )

    dt_solve = time.time() - t0

    # Extract stress from all charts
    all_sigma = []
    all_nodes_phys = []
    all_u = []
    all_chart_id = []

    for ci in range(n_charts):
        if u_charts[ci] is None:
            continue
        F_def = chart_solvers[ci].compute_F(u_charts[ci])
        P = stress_fn(F_def)
        sigma = cauchy_from_first_piola(P.detach().numpy(), F_def.detach().numpy())
        all_sigma.append(sigma)

        # Scatter to nodes for VTU
        nodes_ci = chart_solvers[ci].nodes_phys.numpy()
        u_ci = u_charts[ci].detach().numpy()
        all_nodes_phys.append(nodes_ci)
        all_u.append(u_ci)
        all_chart_id.append(np.full(len(nodes_ci), ci))

    # Concatenate all charts for global analysis
    if all_sigma:
        sigma_all = np.concatenate(all_sigma, axis=0)

        # Average shear stress magnitude
        tau_all = np.sqrt(sigma_all[:, 0, 2]**2 + sigma_all[:, 1, 2]**2)
        tau_mean = tau_all.mean()

        # Drucker-Prager
        F_dp = drucker_prager_F(sigma_all, sigma_ts, sigma_hs)
        F_dp_max = F_dp.max()
    else:
        tau_mean = 0.0
        F_dp_max = -999.0

    strain_hist.append(gamma)
    if nucleation_step is None:
        stress_hist.append(tau_mean)
    else:
        stress_hist.append(0.0)
    F_dp_hist.append(F_dp_max)

    # Check nucleation
    nucleated = False
    if F_dp_max >= 0 and nucleation_step is None:
        nucleation_step = step
        nucleated = True
        print(f"  *** NUCLEATION at step {step}: gamma={gamma:.6e}, "
              f"tau={tau_mean:.2f} MPa, F_DP={F_dp_max:.4f}")

    # Write VTU (all charts combined)
    if all_nodes_phys:
        nodes_combined = np.concatenate(all_nodes_phys, axis=0)
        u_combined = np.concatenate(all_u, axis=0)
        chart_ids = np.concatenate(all_chart_id, axis=0)

        # Scatter element stress to nodes per chart
        tau_nodal = np.zeros(len(nodes_combined))
        vm_nodal = np.zeros(len(nodes_combined))
        fdp_nodal = np.zeros(len(nodes_combined))
        offset = 0
        for ci in range(n_charts):
            if u_charts[ci] is None:
                continue
            n_ci = chart_solvers[ci].n_nodes
            n_el_ci = chart_solvers[ci].n_elements
            elems_ci = chart_solvers[ci].elements.numpy()
            sigma_ci = all_sigma[ci] if ci < len(all_sigma) else np.zeros((0, 3, 3))

            tau_el = np.sqrt(sigma_ci[:, 0, 2]**2 + sigma_ci[:, 1, 2]**2) if len(sigma_ci) > 0 else np.zeros(0)
            vm_el = np.sqrt(0.5 * ((sigma_ci[:, 0, 0] - sigma_ci[:, 1, 1])**2 +
                                    (sigma_ci[:, 1, 1] - sigma_ci[:, 2, 2])**2 +
                                    (sigma_ci[:, 2, 2] - sigma_ci[:, 0, 0])**2 +
                                    6 * (sigma_ci[:, 0, 1]**2 + sigma_ci[:, 1, 2]**2 + sigma_ci[:, 0, 2]**2))) if len(sigma_ci) > 0 else np.zeros(0)
            fdp_el = drucker_prager_F(sigma_ci, sigma_ts, sigma_hs) if len(sigma_ci) > 0 else np.zeros(0)

            count = np.zeros(n_ci)
            for e in range(min(n_el_ci, len(tau_el))):
                for ni in elems_ci[e]:
                    tau_nodal[offset + ni] += tau_el[e]
                    vm_nodal[offset + ni] += vm_el[e]
                    fdp_nodal[offset + ni] += fdp_el[e]
                    count[ni] += 1
            count = np.maximum(count, 1)
            tau_nodal[offset:offset+n_ci] /= count
            vm_nodal[offset:offset+n_ci] /= count
            fdp_nodal[offset:offset+n_ci] /= count
            offset += n_ci

        fname = f"torsion_{step:04d}.vtu"
        write_vtu_points(
            os.path.join(OUT_VTU, fname),
            nodes_combined,
            {
                "displacement": u_combined,
                "shear_stress": tau_nodal,
                "von_Mises": vm_nodal,
                "F_drucker_prager": fdp_nodal,
                "chart_id": chart_ids,
                "applied_alpha": np.full(len(nodes_combined), alpha),
            },
        )
        vtu_files.append(fname)

    status = "NUCLEATE" if nucleated else ("post" if nucleation_step is not None and step > nucleation_step else "loading")
    print(f"  Step {step:3d}/{n_load_steps-1} | gamma={gamma:.4e} | tau={tau_mean:.2f} MPa | "
          f"F_DP={F_dp_max:.2f} | {status} | {dt_solve:.2f}s")

    if nucleation_step is not None and step > nucleation_step + 2:
        break

# Write PVD
pvd_path = os.path.join(OUT_VTU, "torsion.pvd")
with open(pvd_path, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for i, fname in enumerate(vtu_files):
        f.write(f'    <DataSet timestep="{float(i)}" file="{fname}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

total_time = time.time() - t_start
print(f"\n  Total time: {total_time:.1f}s ({len(vtu_files)} steps)")
print(f"  VTU: {pvd_path}")

# ── PyVista visualization ─────────────────────────────────────────────
print("\nGenerating PyVista visualizations...")

import pyvista as pv
pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"

# Pick a mid-loading step and the nucleation step
mid_step = min(n_load_steps // 2, len(vtu_files) - 1)
nuc_step = nucleation_step if nucleation_step is not None and nucleation_step < len(vtu_files) else len(vtu_files) - 1

steps_to_show = [0, mid_step, nuc_step]
labels = ["Step 0 (unloaded)", f"Step {mid_step} (loading)", f"Step {nuc_step} (nucleation)"]

p = pv.Plotter(shape=(1, 3), window_size=[1800, 600], off_screen=True)

for idx, (step_i, label) in enumerate(zip(steps_to_show, labels)):
    p.subplot(0, idx)
    mesh = pv.read(os.path.join(OUT_VTU, f"torsion_{step_i:04d}.vtu"))

    # Color by chart_id (each chart a different color)
    chart_ids = mesh.point_data["chart_id"]
    colors = np.zeros((mesh.n_points, 4), dtype=np.uint8)
    palette = [(76, 155, 232), (232, 133, 76), (92, 184, 92), (204, 121, 167)]
    for ci in range(n_charts):
        mask = chart_ids == ci
        if mask.any():
            colors[mask, :3] = palette[ci % len(palette)]
            colors[mask, 3] = 200

    # Warp by displacement (amplified for visibility)
    disp = mesh.point_data["displacement"]
    amp = 500.0  # amplification factor
    warped = mesh.copy()
    warped.points = mesh.points + disp * amp

    warped.point_data["colors"] = colors
    try:
        p.add_mesh(warped, scalars="colors", rgba=True, point_size=4,
                   render_points_as_spheres=True)
    except Exception:
        # Fallback if rgba fails
        p.add_mesh(warped, color="#A0C4E8", point_size=4,
                   render_points_as_spheres=True)

    p.add_text(label, position="upper_left", font_size=9, color="black")
    p.camera_position = [(10, -8, 6), (0, 0, L/2), (0, 0, 1)]

pv_path = os.path.join(OUT_FIG, "torsion_chart_fem_3d.png")
p.screenshot(pv_path, transparent_background=False, scale=2)
p.close()
print(f"Saved: {pv_path}")

# ── Matplotlib stress-strain ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W
set_pub_style(fontsize=8, usetex=False)

strain_hist = np.array(strain_hist)
stress_hist = np.array(stress_hist)

# Exact solution
gamma_exact = np.linspace(0, alpha_ss * (r_inner+R_outer)/(4*L) * 1.2, 200)
S_exact = mu_mat * gamma_exact * 2  # tau = mu * gamma, S = 2*tau for avg

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W * 0.7, 2.5))

ax = axes[0]
gamma_sharp = np.linspace(0, alpha_ss * (r_inner+R_outer)/(4*L) * 1.2, 200)
alpha_sharp = gamma_sharp * 4 * L / (r_inner + R_outer)
S_sharp = np.where(alpha_sharp < alpha_ss,
                   mu_mat * alpha_sharp * (r_inner + R_outer) / (2 * L),
                   0.0)
ax.plot(gamma_sharp * 1e3, S_sharp, "k-", lw=2.0, label="Sharp (exact)")
ax.plot(strain_hist * 1e3, stress_hist, "o-", color=PUB_COLORS[0],
        ms=3, lw=0.8, label=f"Chart FEM ({n_charts} charts)")
ax.axhline(sigma_ss, color="gray", ls=":", lw=0.5)
ax.set_xlabel("$\\gamma$ ($\\times 10^{-3}$)")
ax.set_ylabel("$\\tau$ (MPa)")
ax.set_title("(a) Shear stress-strain", fontsize=8)
ax.legend(fontsize=5)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

ax = axes[1]
ax.axis("off")
info = (
    f"Charts: {n_charts}\n"
    f"Nodes/chart: {chart_solvers[0].n_nodes}\n"
    f"Total nodes: {total_nodes}\n"
    f"Parallel: 4 threads\n"
    f"Total time: {total_time:.1f}s\n"
)
if nucleation_step is not None:
    info += (
        f"\nNucleation: step {nucleation_step}\n"
        f"tau = {stress_hist[nucleation_step]:.1f} MPa\n"
        f"sigma_ss = {sigma_ss:.1f} MPa\n"
        f"Error: {abs(stress_hist[nucleation_step]-sigma_ss)/sigma_ss*100:.1f}%"
    )
ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=6,
        va="center", family="monospace",
        bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray"))
ax.set_title("(b) Summary", fontsize=8)

fig.suptitle("Torsion: multi-chart parallel FEM solve", fontsize=9, y=1.0)
plt.tight_layout()

fig_path = os.path.join(OUT_FIG, "torsion_chart_fem.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close(fig)

print(f"\n=== Result ===")
if nucleation_step is not None:
    print(f"  tau (FEM): {stress_hist[nucleation_step]:.2f} MPa")
    print(f"  sigma_ss (DP): {sigma_ss:.2f} MPa")
    print(f"  Error: {abs(stress_hist[nucleation_step]-sigma_ss)/sigma_ss*100:.1f}%")
