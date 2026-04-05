#!/usr/bin/env python3
"""Challenge 1: Uniaxial tension on a circular rod with multi-chart parallel FEM.

Geometry: Cylindrical rod, L=15mm, R=2mm
Loading: Axial displacement delta at z=L, fixed at z=0
Material: Soda-lime glass (E=70GPa, nu=0.22, sigma_ts=40MPa)
Exact: S = E*2*delta/L until S = sigma_ts, then S = 0

Uses 4 CylinderDecoder charts around the circumference, each covering
a 120-degree sector with full axial extent. Parallel Schwarz coupling.

Usage:
    python nineO_examples/1_uniaxial_tension/run_uniaxial_chart_fem.py
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
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import CylinderDecoder
from solvers.fracture_criteria import (
    drucker_prager_F, derived_biaxial_strength, derived_shear_strength,
)
from postprocessing.utils import write_vtu_points

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=8, usetex=False)

OUT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
OUT_VTU = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "uniaxial_chart_fem")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_VTU, exist_ok=True)

# ── Material ──────────────────────────────────────────────────────────
E = 70e3          # MPa
nu = 0.22
sigma_ts = 40.0   # MPa (uniaxial tensile strength)
sigma_hs = 27.8   # MPa (hydrostatic strength)

# ── Geometry ──────────────────────────────────────────────────────────
L_rod = 15.0   # mm
R_rod = 2.0    # mm

# ── Chart setup ───────────────────────────────────────────────────────
n_circ = 4      # charts around circumference
n_cells = 10    # cells per axis per chart
theta_span = math.pi / 1.5  # 120 degrees (30-deg overlap)

print("=== Challenge 1: Uniaxial Tension — Multi-Chart Parallel FEM ===")
print(f"  Rod: L={L_rod}mm, R={R_rod}mm")
print(f"  Material: E={E} MPa, nu={nu}, sigma_ts={sigma_ts} MPa")
print(f"  Charts: {n_circ}, n_cells={n_cells}")
print()

# ── Build charts ──────────────────────────────────────────────────────
chart_solvers = []
chart_decoders = []
chart_seeds = []

theta_centers = [i * 2 * math.pi / n_circ for i in range(n_circ)]

for ti, theta_c in enumerate(theta_centers):
    decoder = CylinderDecoder(
        theta_center=theta_c,
        theta_span=theta_span,
        R=R_rod,
        z_center=L_rod / 2,
        L_half=L_rod / 2,
    ).double()

    solver = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0,
        chart_decoder=decoder, decoder_kwargs={},
        device="cpu", dtype=torch.float64,
    )

    chart_solvers.append(solver)
    chart_decoders.append(decoder)
    chart_seeds.append([R_rod/2 * math.cos(theta_c), R_rod/2 * math.sin(theta_c), L_rod/2])

seeds_t = torch.tensor(chart_seeds, dtype=torch.float64)
neighbors = [[(ti + 1) % n_circ, (ti - 1) % n_circ] for ti in range(n_circ)]

total_nodes = sum(s.n_nodes for s in chart_solvers)
total_elements = sum(s.n_elements for s in chart_solvers)
print(f"  Total: {total_nodes} nodes, {total_elements} elements across {n_circ} charts")

stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

# ── Loading ───────────────────────────────────────────────────────────
eps_ts = sigma_ts / E  # strain at fracture (uniaxial stress)
delta_ts = eps_ts * L_rod / 2  # displacement at fracture
n_load_steps = 30
delta_max = delta_ts * 1.3
delta_steps = np.linspace(0, delta_max, n_load_steps + 1)[1:]

print(f"  eps_ts = {eps_ts:.6e}, delta_ts = {delta_ts:.6f} mm")
print(f"  Load steps: {n_load_steps}, delta_max = {delta_max:.6f} mm")
print()

# ── BC function ───────────────────────────────────────────────────────
def make_uniaxial_bc(delta):
    """Fixed z=0, pull z=L, lateral traction-free."""
    def phys_bc_fn(nodes_phys):
        n = len(nodes_phys)
        u_bc = np.zeros((n, 3))
        mask = np.zeros(n, dtype=bool)
        z = nodes_phys[:, 2]
        tol = L_rod * 0.03

        # z=0 face: fixed
        z0 = z < tol
        mask[z0] = True
        u_bc[z0] = 0.0

        # z=L face: prescribed axial displacement
        zL = z > L_rod - tol
        mask[zL] = True
        u_bc[zL, 2] = delta  # u_z = delta

        return u_bc, mask
    return phys_bc_fn

# ── Incremental solve ─────────────────────────────────────────────────
strain_hist = []
stress_hist = []
F_dp_hist = []
nucleation_step = None
vtu_files = []

t_start = time.time()

for step, delta in enumerate(delta_steps):
    eps = 2 * delta / L_rod  # engineering strain
    S_expected = E * eps     # expected uniaxial stress (Hooke's law)

    t0 = time.time()

    phys_bc_fn = make_uniaxial_bc(delta)

    schwarz = SchwarzVectorFEMSolver(
        chart_solvers=chart_solvers,
        seeds=seeds_t,
        decoders=chart_decoders,
        neighbors=neighbors,
        parallel=True,
        n_workers=4,
    )

    u_charts = schwarz.solve(
        stress_fn, tangent_fn, phys_bc_fn,
        max_schwarz_iters=15,
        tol=1e-3,
        newton_max_iter=10,
        newton_tol=1e-8,
        relaxation=0.3,
    )

    dt_solve = time.time() - t0

    # Extract stress from all charts
    all_sigma_zz = []
    all_nodes = []
    all_u = []
    all_chart_id = []

    for ci in range(n_circ):
        if u_charts[ci] is None:
            continue
        F_def = chart_solvers[ci].compute_F(u_charts[ci])
        sigma = stress_fn(F_def).detach().numpy()
        all_sigma_zz.extend(sigma[:, 2, 2].tolist())
        all_nodes.append(chart_solvers[ci].nodes_phys.detach().numpy())
        all_u.append(u_charts[ci].detach().numpy())
        all_chart_id.append(np.full(chart_solvers[ci].n_nodes, ci))

    sigma_zz_mean = np.mean(all_sigma_zz) if all_sigma_zz else 0.0

    # DP check on element stresses (volume-averaged to avoid boundary artifacts)
    if all_sigma_zz:
        # Build volume-averaged stress for DP
        sigma_avg = np.zeros((1, 3, 3))
        sigma_avg[0, 2, 2] = sigma_zz_mean
        # Add Poisson effect: sigma_xx = sigma_yy = 0 for free-lateral
        F_dp = drucker_prager_F(sigma_avg, sigma_ts, sigma_hs)[0]
    else:
        F_dp = -999.0

    strain_hist.append(eps)
    if nucleation_step is None:
        stress_hist.append(sigma_zz_mean)
    else:
        stress_hist.append(0.0)
    F_dp_hist.append(F_dp)

    # Check nucleation
    nucleated = False
    if F_dp >= 0 and nucleation_step is None:
        nucleation_step = step
        nucleated = True
        print(f"  *** NUCLEATION at step {step}: eps={eps:.4e}, "
              f"S_zz={sigma_zz_mean:.2f} MPa, F_DP={F_dp:.4f}")

    # Write VTU
    if all_nodes:
        nodes_combined = np.concatenate(all_nodes)
        u_combined = np.concatenate(all_u)
        chart_ids = np.concatenate(all_chart_id)

        fname = f"uniaxial_{step:04d}.vtu"
        write_vtu_points(
            os.path.join(OUT_VTU, fname), nodes_combined,
            {
                "displacement": u_combined,
                "sigma_zz": np.full(len(nodes_combined), sigma_zz_mean),
                "chart_id": chart_ids,
                "applied_strain": np.full(len(nodes_combined), eps),
            },
        )
        vtu_files.append(fname)

    status = "NUCLEATE" if nucleated else ("post" if nucleation_step is not None else "loading")
    err = abs(sigma_zz_mean - S_expected) / S_expected * 100 if S_expected > 0 and nucleation_step is None else 0
    print(f"  Step {step:3d}/{n_load_steps-1} | eps={eps:.4e} | "
          f"S_zz={sigma_zz_mean:.2f} (exp {S_expected:.2f}, err {err:.1f}%) | "
          f"F_DP={F_dp:.2f} | {status} | {dt_solve:.1f}s")

    if nucleation_step is not None and step > nucleation_step + 2:
        break

# Write PVD
pvd_path = os.path.join(OUT_VTU, "uniaxial.pvd")
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

# ── Plot ──────────────────────────────────────────────────────────────
strain_hist = np.array(strain_hist)
stress_hist = np.array(stress_hist)

eps_exact = np.linspace(0, eps_ts * 1.3, 200)
S_exact = np.where(eps_exact < eps_ts, E * eps_exact, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W * 0.7, 2.5))

ax = axes[0]
ax.plot(eps_exact * 1e3, S_exact, "k-", lw=2.0, label="Sharp (exact)")
ax.plot(strain_hist * 1e3, stress_hist, "o-", color=PUB_COLORS[0],
        ms=3, lw=0.8, label=f"Chart FEM ({n_circ} charts)")
ax.axhline(sigma_ts, color="gray", ls=":", lw=0.5)
if nucleation_step is not None:
    ax.axvline(strain_hist[nucleation_step] * 1e3, color="red", ls=":", lw=0.8)
ax.set_xlabel("Strain $2\\delta/L$ ($\\times 10^{-3}$)")
ax.set_ylabel("$\\sigma_{zz}$ (MPa)")
ax.set_title("(a) Stress-strain", fontsize=8)
ax.legend(fontsize=5)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

ax = axes[1]
ax.axis("off")
info = (
    f"Charts: {n_circ}\n"
    f"Nodes/chart: {chart_solvers[0].n_nodes}\n"
    f"Total: {total_nodes} nodes\n"
    f"Parallel: 4 threads\n"
    f"Time: {total_time:.0f}s\n"
)
if nucleation_step is not None:
    info += (
        f"\nNucleation step {nucleation_step}\n"
        f"sigma_zz = {stress_hist[nucleation_step]:.1f} MPa\n"
        f"sigma_ts = {sigma_ts:.0f} MPa\n"
        f"Error: {abs(stress_hist[nucleation_step]-sigma_ts)/sigma_ts*100:.1f}%"
    )
ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=6,
        va="center", family="monospace",
        bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray"))
ax.set_title("(b) Summary", fontsize=8)

fig.suptitle("Challenge 1: Uniaxial tension — multi-chart parallel FEM", fontsize=9, y=1.0)
plt.tight_layout()

path = os.path.join(OUT_FIG, "uniaxial_chart_fem.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")
plt.close(fig)

print(f"\n=== Result ===")
if nucleation_step is not None:
    print(f"  sigma_zz (FEM): {stress_hist[nucleation_step]:.2f} MPa")
    print(f"  sigma_ts: {sigma_ts:.2f} MPa")
    print(f"  Error: {abs(stress_hist[nucleation_step]-sigma_ts)/sigma_ts*100:.1f}%")
