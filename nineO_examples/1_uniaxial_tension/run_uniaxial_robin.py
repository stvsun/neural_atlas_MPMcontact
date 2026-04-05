#!/usr/bin/env python3
"""Challenge 1: Uniaxial tension with Robin DD on BoxDecoder charts.

Uses BoxDecoder (Cartesian mapping, no axis singularity) with SDF
filtering for the cylindrical rod, and Robin parallel DD for coupling.

Usage:
    python nineO_examples/1_uniaxial_tension/run_uniaxial_robin.py
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import BoxDecoder
from solvers.fracture_criteria import drucker_prager_F
from postprocessing.utils import write_vtu_points

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W
set_pub_style(fontsize=8, usetex=False)

OUT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
OUT_VTU = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "uniaxial_robin")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_VTU, exist_ok=True)

# ── Material ──────────────────────────────────────────────────────────
E = 70e3; nu = 0.22
sigma_ts = 40.0; sigma_hs = 27.8

# ── Geometry ──────────────────────────────────────────────────────────
L = 15.0; R = 2.0

# SDF for cylindrical rod
class RodSDF:
    def sdf(self, x):
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
        d_radial = r - R
        d_axial = np.maximum(-x_np[:, 2], x_np[:, 2] - L)
        # Intersection of cylinder and z-slab
        outside = np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_axial, 0)**2)
        inside = np.minimum(np.maximum(d_radial, d_axial), 0)
        vals = outside + inside
        if isinstance(x, torch.Tensor):
            return torch.tensor(vals, dtype=x.dtype, device=x.device)
        return vals

sdf = RodSDF()

# ── Charts: 2 BoxDecoders stacked along z ─────────────────────────────
n_cells = 10
n_charts = 2
overlap = 0.3  # 30% overlap

z_span = L / n_charts * (1 + overlap)
chart_solvers = []
chart_decoders = []
chart_seeds = []

for ci in range(n_charts):
    z_center = L * (ci + 0.5) / n_charts
    dec = BoxDecoder(
        center=(0, 0, z_center),
        half_extents=(R * 1.2, R * 1.2, z_span / 2),
    ).double()

    solver = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0,
        chart_decoder=dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01,
        device="cpu", dtype=torch.float64,
    )

    chart_solvers.append(solver)
    chart_decoders.append(dec)
    chart_seeds.append([0, 0, z_center])

seeds_t = torch.tensor(chart_seeds, dtype=torch.float64)
neighbors = [[1], [0]]

total_nodes = sum(s.n_nodes for s in chart_solvers)
total_elements = sum(s.n_elements for s in chart_solvers)

print("=== Challenge 1: Uniaxial Tension — Robin DD + BoxDecoder + SDF ===")
print(f"  Rod: L={L}, R={R} mm")
print(f"  Material: E={E}, nu={nu}, sigma_ts={sigma_ts}")
print(f"  Charts: {n_charts} BoxDecoder, n_cells={n_cells}")
print(f"  Total: {total_nodes} nodes, {total_elements} elements")
print(f"  Nodes per chart: {[s.n_nodes for s in chart_solvers]}")
print()

stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

# ── Loading ───────────────────────────────────────────────────────────
eps_ts = sigma_ts / E
delta_ts = eps_ts * L
n_load_steps = 25
delta_max = delta_ts * 1.2
delta_steps = np.linspace(0, delta_max, n_load_steps + 1)[1:]

print(f"  sigma_ts = {sigma_ts}, eps_ts = {eps_ts:.6e}, delta_ts = {delta_ts:.6f}")
print(f"  Load steps: {n_load_steps}")
print()

# ── BC: prescribe uniaxial strain on ALL boundary nodes ──────────────
# u_z = eps*z, u_x = -nu*eps*x, u_y = -nu*eps*y (Poisson contraction)
def make_bc(delta):
    def phys_bc_fn(nodes_phys):
        n = len(nodes_phys)
        u_bc = np.zeros((n, 3)); mask = np.zeros(n, dtype=bool)
        eps = delta / L
        # Constrain ALL boundary nodes
        mask[:] = True
        u_bc[:, 0] = -nu * eps * nodes_phys[:, 0]  # Poisson contraction
        u_bc[:, 1] = -nu * eps * nodes_phys[:, 1]
        u_bc[:, 2] = eps * nodes_phys[:, 2]
        return u_bc, mask
    return phys_bc_fn

# ── Incremental solve ─────────────────────────────────────────────────
strain_hist = []; stress_hist = []; F_dp_hist = []
nucleation_step = None
vtu_files = []
t_start = time.time()

for step, delta in enumerate(delta_steps):
    eps = delta / L
    S_expected = E * eps

    t0 = time.time()

    robin = RobinSchwarzSolver(
        chart_solvers=chart_solvers, seeds=seeds_t,
        decoders=chart_decoders, neighbors=neighbors,
        robin_delta=E * 0.5, parallel=True, n_workers=2,
    )

    u_charts = robin.solve(
        stress_fn, tangent_fn, make_bc(delta),
        max_iters=25, tol=1e-3,
    )

    dt = time.time() - t0

    # Extract stress
    all_sigma_zz = []
    all_nodes = []; all_u = []; all_cid = []
    for ci in range(n_charts):
        if u_charts[ci] is None: continue
        F_def = chart_solvers[ci].compute_F(u_charts[ci])
        sigma = stress_fn(F_def).detach().numpy()
        all_sigma_zz.extend(sigma[:, 2, 2].tolist())
        all_nodes.append(chart_solvers[ci].nodes_phys.detach().numpy())
        all_u.append(u_charts[ci].detach().numpy())
        all_cid.append(np.full(chart_solvers[ci].n_nodes, ci))

    szz = np.mean(all_sigma_zz) if all_sigma_zz else 0
    sigma_avg = np.zeros((1, 3, 3)); sigma_avg[0, 2, 2] = szz
    F_dp = drucker_prager_F(sigma_avg, sigma_ts, sigma_hs)[0]

    strain_hist.append(eps)
    stress_hist.append(szz if nucleation_step is None else 0.0)
    F_dp_hist.append(F_dp)

    nucleated = False
    if F_dp >= 0 and nucleation_step is None:
        nucleation_step = step; nucleated = True
        print(f"  *** NUCLEATION step {step}: eps={eps:.4e}, S={szz:.2f}, F_DP={F_dp:.2f}")

    # VTU
    if all_nodes:
        nc = np.concatenate(all_nodes); uc = np.concatenate(all_u)
        cids = np.concatenate(all_cid)
        fname = f"uniaxial_{step:04d}.vtu"
        write_vtu_points(os.path.join(OUT_VTU, fname), nc,
                          {"displacement": uc, "sigma_zz": np.full(len(nc), szz), "chart_id": cids})
        vtu_files.append(fname)

    err = abs(szz - S_expected) / S_expected * 100 if S_expected > 0 and nucleation_step is None else 0
    status = "NUC" if nucleated else ("post" if nucleation_step else "load")
    print(f"  Step {step:2d}/{n_load_steps-1} | eps={eps:.4e} | S_zz={szz:.2f} (exp {S_expected:.2f}, {err:.1f}%) | "
          f"F_DP={F_dp:.1f} | {status} | {dt:.1f}s")

    if nucleation_step is not None and step > nucleation_step + 1: break

# PVD
with open(os.path.join(OUT_VTU, "uniaxial.pvd"), "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n')
    for i, fn in enumerate(vtu_files):
        f.write(f'    <DataSet timestep="{float(i)}" file="{fn}"/>\n')
    f.write('  </Collection>\n</VTKFile>\n')

total_time = time.time() - t_start

# ── Plot ──────────────────────────────────────────────────────────────
strain_hist = np.array(strain_hist); stress_hist = np.array(stress_hist)
eps_ex = np.linspace(0, eps_ts*1.2, 200)
S_ex = np.where(eps_ex < eps_ts, E * eps_ex, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W*0.7, 2.5))
ax = axes[0]
ax.plot(eps_ex*1e3, S_ex, "k-", lw=2, label="Sharp")
ax.plot(strain_hist*1e3, stress_hist, "o-", color=PUB_COLORS[0], ms=3, lw=0.8, label="Robin DD")
ax.axhline(sigma_ts, color="gray", ls=":", lw=0.5)
if nucleation_step is not None:
    ax.axvline(strain_hist[nucleation_step]*1e3, color="red", ls=":", lw=0.8)
ax.set_xlabel("$\\varepsilon$ ($\\times 10^{-3}$)"); ax.set_ylabel("$\\sigma_{zz}$ (MPa)")
ax.set_title("(a) Stress-strain", fontsize=8); ax.legend(fontsize=5); ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

ax = axes[1]; ax.axis("off")
info = f"Charts: {n_charts} BoxDecoder\nNodes: {total_nodes}\nTime: {total_time:.0f}s\n"
if nucleation_step is not None:
    info += f"\nNucleation step {nucleation_step}\nsigma_zz = {stress_hist[nucleation_step]:.1f}\nsigma_ts = {sigma_ts:.0f}\nError: {abs(stress_hist[nucleation_step]-sigma_ts)/sigma_ts*100:.1f}%"
ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=6, va="center", family="monospace",
        bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray"))
ax.set_title("(b) Summary", fontsize=8)

fig.suptitle("Challenge 1: Uniaxial tension — Robin DD + BoxDecoder", fontsize=9, y=1.0)
plt.tight_layout()
path = os.path.join(OUT_FIG, "uniaxial_robin_dd.png")
fig.savefig(path, dpi=200, bbox_inches="tight"); print(f"Saved: {path}"); plt.close(fig)

print(f"\n=== Result ===")
print(f"  Total time: {total_time:.0f}s, {len(vtu_files)} steps")
if nucleation_step is not None:
    print(f"  sigma_zz = {stress_hist[nucleation_step]:.2f} vs sigma_ts = {sigma_ts}")
    print(f"  Error: {abs(stress_hist[nucleation_step]-sigma_ts)/sigma_ts*100:.1f}%")
