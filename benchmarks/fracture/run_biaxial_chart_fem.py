#!/usr/bin/env python3
"""Biaxial tension on a circular plate using REAL chart FEM solve.

This is the actual incremental FEM simulation — no analytical shortcuts.
At each load step:
  1. Apply biaxial displacement BCs on boundary nodes
  2. Solve K*u = f via ChartVectorFEMSolver.solve_nonlinear()
  3. Extract Cauchy stress from FEM displacement
  4. Check Drucker-Prager nucleation criterion
  5. Output VTU + record stress-strain

Geometry: circular plate modeled as a cube chart with SDF filtering.
Material: soda-lime glass (E=70GPa, nu=0.22, sigma_ts=40MPa, sigma_hs=27.8MPa).

Usage:
    python benchmarks/fracture/run_biaxial_chart_fem.py
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.linear_elastic import make_linear_elastic
from solvers.fracture_criteria import (
    drucker_prager_F, cauchy_from_first_piola,
    derived_biaxial_strength, crack_normal_from_stress,
)
from benchmarks.fracture.biaxial_tension import sdf_circular_plate, GLASS
from postprocessing.utils import write_vtu_points

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=8, usetex=False)

# ── Output directories ───────────────────────────────────────────────
OUT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
OUT_VTU = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "biaxial_chart_fem")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_VTU, exist_ok=True)

# ── Material ──────────────────────────────────────────────────────────
E = GLASS["E"]            # 70000 MPa
nu = GLASS["nu"]          # 0.22
sigma_ts = GLASS["sigma_ts"]  # 40 MPa
sigma_hs = GLASS["sigma_hs"]  # 27.8 MPa
sigma_bs = derived_biaxial_strength(sigma_ts, sigma_hs)  # ~27.03 MPa

lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# ── Geometry ──────────────────────────────────────────────────────────
R_plate = 5.0   # mm
T_plate = 0.5   # mm

# SDF oracle for the circular plate
class PlateSDFOracle:
    def sdf(self, x):
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        vals = sdf_circular_plate(x_np, R=R_plate, L=T_plate)
        if isinstance(x, torch.Tensor):
            return torch.tensor(vals, dtype=x.dtype, device=x.device)
        return vals

sdf_oracle = PlateSDFOracle()

# ── Build FEM solver ──────────────────────────────────────────────────
# The chart covers [-R*1.2, R*1.2]^2 x [-T, T], SDF filters to plate
n_cells = 10  # resolution per axis
support_r = R_plate * 1.2  # chart radius

print("=== Biaxial Tension: Real Chart FEM Solve ===")
print(f"  Material: E={E} MPa, nu={nu}")
print(f"  sigma_ts={sigma_ts}, sigma_hs={sigma_hs}, sigma_bs(DP)={sigma_bs:.2f} MPa")
print(f"  Plate: R={R_plate}, T={T_plate} mm")
print(f"  Mesh: n_cells={n_cells}, support_r={support_r:.1f}")
print()

t_start = time.time()

solver = ChartVectorFEMSolver(
    n_cells=n_cells,
    support_r=support_r,
    sdf_oracle=sdf_oracle,
    sdf_threshold=-0.01,
    mesh_extent=1.0,
    device="cpu",
    dtype=torch.float64,
)

print(f"  FEM mesh: {solver.n_nodes} nodes, {solver.n_elements} elements, "
      f"{solver.boundary_mask.sum().item()} boundary nodes")
print(f"  Mesh build time: {time.time() - t_start:.2f}s")

if solver.n_nodes == 0:
    print("ERROR: No mesh nodes — SDF filtering removed everything.")
    print("Try increasing n_cells or support_r.")
    sys.exit(1)

stress_fn, tangent_fn = make_linear_elastic(E, nu)

# ── Identify boundary faces ──────────────────────────────────────────
nodes = solver.nodes  # reference coordinates
nodes_phys = solver.nodes_phys  # physical coordinates (same for identity decoder)
nodes_np = nodes.detach().cpu().numpy()

# For the SDF-filtered plate: boundary = nodes on the plate surface
# The SDF boundary detection classifies nodes near the SDF=0 level set

# ── Loading parameters ───────────────────────────────────────────────
# Biaxial strain: eps = delta / R
# At fracture: sigma_bs = E*eps/(1-nu) => eps_bs = sigma_bs*(1-nu)/E
eps_bs = sigma_bs * (1 - nu) / E
delta_bs = eps_bs * R_plate
n_load_steps = 40
delta_max = delta_bs * 1.3  # go 30% past expected fracture
delta_steps = np.linspace(0, delta_max, n_load_steps + 1)[1:]  # skip zero

print(f"  Expected fracture at eps_bs={eps_bs:.6e}, delta_bs={delta_bs:.6f} mm")
print(f"  Loading: {n_load_steps} steps, delta_max={delta_max:.6f} mm")
print()

# ── Incremental solve ─────────────────────────────────────────────────
strain_hist = []
stress_hist = []
F_dp_hist = []
nucleation_step = None
vtu_files = []

for step, delta in enumerate(delta_steps):
    eps = delta / R_plate
    t0 = time.time()

    # BCs: equi-biaxial displacement on ALL boundary nodes
    # u_x = eps * x, u_y = eps * y, u_z = 0 (plane strain)
    bc_mask = solver.boundary_mask
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
    u_bc[:, 0] = eps * nodes[:, 0]
    u_bc[:, 1] = eps * nodes[:, 1]
    u_bc[:, 2] = 0.0

    f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

    # Solve
    u = solver.solve_nonlinear(
        stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
        max_iter=15, tol=1e-9,
    )

    # Extract stress
    F_def = solver.compute_F(u)
    P = stress_fn(F_def)
    P_np = P.detach().cpu().numpy()
    F_np = F_def.detach().cpu().numpy()
    sigma = cauchy_from_first_piola(P_np, F_np)

    # Average biaxial stress: S = (sigma_xx + sigma_yy) / 2
    sigma_xx_mean = sigma[:, 0, 0].mean()
    sigma_yy_mean = sigma[:, 1, 1].mean()
    S_biaxial = (sigma_xx_mean + sigma_yy_mean) / 2

    # Drucker-Prager check
    F_dp = drucker_prager_F(sigma, sigma_ts, sigma_hs)
    F_dp_max = F_dp.max()

    dt_solve = time.time() - t0

    # Record
    strain_hist.append(eps)
    if nucleation_step is None:
        stress_hist.append(S_biaxial)
    else:
        stress_hist.append(0.0)
    F_dp_hist.append(F_dp_max)

    # Check nucleation
    nucleated_now = False
    if F_dp_max >= 0 and nucleation_step is None:
        nucleation_step = step
        nucleated_now = True
        max_elem = np.argmax(F_dp)
        crack_normal = crack_normal_from_stress(sigma[max_elem])
        print(f"  *** NUCLEATION at step {step}: eps={eps:.6e}, "
              f"S={S_biaxial:.2f} MPa, F_DP={F_dp_max:.4f}")
        print(f"      Crack normal = {crack_normal}")

    # Write VTU
    u_np = u.detach().cpu().numpy()

    # Scatter element stress to nodes
    sigma_xx_nodal = np.zeros(solver.n_nodes)
    sigma_yy_nodal = np.zeros(solver.n_nodes)
    von_mises_nodal = np.zeros(solver.n_nodes)
    F_dp_nodal = np.zeros(solver.n_nodes)
    count = np.zeros(solver.n_nodes)
    elements = solver.elements.detach().cpu().numpy()

    s_vm = np.sqrt(0.5 * ((sigma[:, 0, 0] - sigma[:, 1, 1])**2
                          + (sigma[:, 1, 1] - sigma[:, 2, 2])**2
                          + (sigma[:, 2, 2] - sigma[:, 0, 0])**2
                          + 6 * (sigma[:, 0, 1]**2 + sigma[:, 1, 2]**2 + sigma[:, 0, 2]**2)))

    for e in range(solver.n_elements):
        for ni in elements[e]:
            sigma_xx_nodal[ni] += sigma[e, 0, 0]
            sigma_yy_nodal[ni] += sigma[e, 1, 1]
            von_mises_nodal[ni] += s_vm[e]
            F_dp_nodal[ni] += F_dp[e]
            count[ni] += 1
    count = np.maximum(count, 1)
    sigma_xx_nodal /= count
    sigma_yy_nodal /= count
    von_mises_nodal /= count
    F_dp_nodal /= count

    fname = f"biaxial_{step:04d}.vtu"
    write_vtu_points(
        os.path.join(OUT_VTU, fname),
        nodes_np,
        {
            "displacement": u_np,
            "sigma_xx": sigma_xx_nodal,
            "sigma_yy": sigma_yy_nodal,
            "von_Mises": von_mises_nodal,
            "F_drucker_prager": F_dp_nodal,
            "applied_strain": np.full(solver.n_nodes, eps),
            "biaxial_stress": np.full(solver.n_nodes, S_biaxial if nucleation_step is None or nucleation_step == step else 0.0),
        },
    )
    vtu_files.append(fname)

    status = "NUCLEATE" if nucleated_now else ("post" if nucleation_step is not None and step > nucleation_step else "loading")
    print(f"  Step {step:3d}/{n_load_steps-1} | eps={eps:.4e} | S={S_biaxial:.2f} MPa | "
          f"F_DP={F_dp_max:.4f} | {status} | {dt_solve:.2f}s")

    if nucleation_step is not None and step > nucleation_step + 3:
        print("  (stopping — post-fracture)")
        break

# Write PVD
pvd_path = os.path.join(OUT_VTU, "biaxial.pvd")
with open(pvd_path, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for i, fname in enumerate(vtu_files):
        f.write(f'    <DataSet timestep="{float(i)}" file="{fname}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

total_time = time.time() - t_start
print(f"\n  Total time: {total_time:.1f}s")
print(f"  VTU: {pvd_path} ({len(vtu_files)} steps)")

# ── Plot comparison with exact solution ───────────────────────────────
strain_hist = np.array(strain_hist)
stress_hist = np.array(stress_hist)
F_dp_hist = np.array(F_dp_hist)

# Exact sharp solution
eps_exact = np.linspace(0, eps_bs * 1.3, 200)
S_exact = np.where(eps_exact < eps_bs, E * eps_exact / (1 - nu), 0.0)

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.8))

# (a) Stress-strain
ax = axes[0]
ax.plot(eps_exact * 1e3, S_exact, "k-", lw=2.0, label="Sharp (exact)")
ax.plot(strain_hist * 1e3, stress_hist, "o-", color=PUB_COLORS[0],
        ms=3, lw=0.8, label="Chart FEM (real solve)")
ax.axhline(sigma_bs, color="gray", ls=":", lw=0.5, alpha=0.5)
if nucleation_step is not None:
    ax.axvline(strain_hist[nucleation_step] * 1e3, color="red", ls=":", lw=0.8)
ax.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)")
ax.set_ylabel("$S$ (MPa)")
ax.set_title("(a) Stress-strain (chart FEM)", fontsize=8)
ax.legend(fontsize=5)
ax.set_ylim(0, 35)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

# (b) Drucker-Prager
ax = axes[1]
ax.plot(strain_hist * 1e3, F_dp_hist, "-", color=PUB_COLORS[0], lw=1.2)
ax.axhline(0, color="red", ls="--", lw=0.8, label="$\\mathcal{F}=0$")
ax.fill_between(strain_hist * 1e3, F_dp_hist, 0,
                where=F_dp_hist >= 0, color="red", alpha=0.1)
ax.set_xlabel("$\\delta/R$ ($\\times 10^{-3}$)")
ax.set_ylabel("$\\mathcal{F}(\\sigma)$")
ax.set_title("(b) Drucker-Prager (from FEM stress)", fontsize=8)
ax.legend(fontsize=5)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

# (c) Summary
ax = axes[2]
ax.axis("off")
if nucleation_step is not None:
    nuc_eps = strain_hist[nucleation_step]
    nuc_S = stress_hist[nucleation_step]
    info = (
        f"Mesh: {solver.n_nodes} nodes, {solver.n_elements} tets\n"
        f"n_cells = {n_cells}\n"
        f"Load steps: {len(vtu_files)}\n"
        f"\nNucleation:\n"
        f"  Step {nucleation_step}\n"
        f"  $\\varepsilon$ = {nuc_eps*1e3:.4f} $\\times 10^{{-3}}$\n"
        f"  $S$ = {nuc_S:.2f} MPa\n"
        f"  $\\sigma_{{bs}}^{{DP}}$ = {sigma_bs:.2f} MPa\n"
        f"  Error: {abs(nuc_S - sigma_bs)/sigma_bs*100:.1f}%\n"
        f"\nTotal time: {total_time:.1f}s"
    )
else:
    info = f"No nucleation detected in {n_load_steps} steps"
ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=6,
        va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="gray"))
ax.set_title("(c) Summary", fontsize=8)

fig.suptitle(
    "Biaxial tension: REAL chart FEM solve (soda-lime glass)\n"
    f"cf. Kamarei et al. (2026) Fig. 5",
    fontsize=9, y=1.02,
)
plt.tight_layout()

path = os.path.join(OUT_FIG, "biaxial_chart_fem.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")

path_pdf = os.path.join(OUT_FIG, "biaxial_chart_fem.pdf")
fig.savefig(path_pdf, bbox_inches="tight")
print(f"Saved: {path_pdf}")
plt.close(fig)

print(f"\n=== Result ===")
if nucleation_step is not None:
    print(f"  Nucleation stress (FEM): {stress_hist[nucleation_step]:.2f} MPa")
    print(f"  Expected (DP):           {sigma_bs:.2f} MPa")
    print(f"  Error:                   {abs(stress_hist[nucleation_step] - sigma_bs)/sigma_bs*100:.1f}%")
