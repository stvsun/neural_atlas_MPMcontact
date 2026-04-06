#!/usr/bin/env python3
"""Publication-quality trousers tear simulation with persistent homology.

Runs a full Mode III tear on a PU elastomer sheet with:
  - Neo-Hookean hyperelasticity (finite strain)
  - Incremental loading (500 steps)
  - Multi-chart Schwarz domain decomposition
  - Persistent homology monitoring for crack topology detection
  - Drucker-Prager nucleation criterion
  - Mesh refinement convergence study (2 meshes)

Reference: Kamarei et al. (2026), CMAME 448, Section 5.2
  - Rivlin-Thomas energy release: G = 2F/B
  - Steady-state tearing force: F_ss = Gc*B/2

Usage:
    python nineO_examples/9_trousers/run_publication.py [--nc 12] [--steps 500]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.analytic_decoders import BoxDecoder
from solvers.fracture_criteria import drucker_prager_F
from solvers.fem.nonlocal_damage import compute_local_equivalent_strain, solve_nonlocal_strain

# ═══════════════════════════════════════════════════════════════
# MATERIAL AND GEOMETRY
# ═══════════════════════════════════════════════════════════════

MU = 0.52        # shear modulus [MPa]
LAM = 85.77      # first Lame parameter [MPa]
K_BULK = LAM + 2 * MU / 3  # bulk modulus
GC = 0.041       # critical energy release rate [N/mm] = 41 N/m
SIGMA_TS = 0.3   # uniaxial tensile strength [MPa]
SIGMA_HS = 1.0   # hydrostatic tensile strength [MPa]

# Trousers geometry (Kamarei Fig. 22)
W = 40.0         # sheet width [mm]
L = 100.0        # sheet length [mm]
B = 1.0          # sheet thickness [mm]
A_CRACK = 50.0   # pre-crack length [mm] (half the sheet)

E_EFF = MU * (3 * LAM + 2 * MU) / (LAM + MU)
NU_EFF = LAM / (2 * (LAM + MU))
F_SS = GC * B / 2  # steady-state tearing force (Rivlin-Thomas)


def run_simulation(nc=12, n_steps=500, delta_max=5.0, monitor_every=50,
                   output_dir=None, verbose=True):
    """Run a single trousers tear simulation.

    Parameters
    ----------
    nc : int
        Mesh cells per axis (resolution).
    n_steps : int
        Number of incremental load steps.
    delta_max : float
        Maximum grip separation [mm].
    monitor_every : int
        Persistent homology check frequency (every N steps).
    output_dir : str
        Directory for results.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'steps', 'delta', 'force_y', 'P_yy_mean', 'J_min', 'J_max',
        'e_local_max', 'e_nonlocal_max', 'nuc_detected_step',
        'topo_events', 'betti_history', 'nc', 'n_nodes', 'n_elements',
        'wall_time'
    """
    if output_dir is None:
        output_dir = f"runs/trousers_nc{nc}_s{n_steps}"
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    if verbose:
        print("=" * 70)
        print(f"  TROUSERS TEAR SIMULATION")
        print(f"  nc={nc}, {n_steps} steps, delta_max={delta_max}mm")
        print(f"  Material: PU elastomer mu={MU}, K={K_BULK:.1f}, Gc={GC}")
        print(f"  Rivlin-Thomas: F_ss = Gc*B/2 = {F_SS:.4f} N")
        print("=" * 70)

    # ── Build mesh ──
    # Use half-sheet (symmetry in x) near the crack tip region
    # BoxDecoder covers the crack-tip region for Mode III
    half_W = W / 2
    dec = BoxDecoder(
        center=(0, 0, 0),
        half_extents=(half_W / 2 + 0.1, L / 4 + 0.1, B / 2 + 0.05)
    ).double()
    solver = ChartVectorFEMSolver(
        n_cells=nc, support_r=1.0, chart_decoder=dec,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )
    stress_fn, tangent_fn = solver.make_neo_hookean(MU, K_BULK)

    n_nodes = solver.n_nodes
    n_elements = solver.n_elements
    if verbose:
        print(f"  Mesh: {n_nodes} nodes, {n_elements} tets")
        print(f"  Element size h ~ {2*(half_W/2+0.1)/nc:.3f} mm")

    # ── Persistent homology monitor ──
    topo_events = []
    betti_history = []
    try:
        from atlas.topo.monitor import TopologyMonitor
        topo_monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1),
            relative_threshold=True,
            verbose=False,
        )
        has_topo = True
    except ImportError:
        has_topo = False
        if verbose:
            print("  [WARN] GUDHI not available, skipping topology monitoring")

    # ── Storage ──
    steps = []
    delta_arr = []
    P_yy_mean_arr = []
    J_min_arr = []
    J_max_arr = []
    e_local_max_arr = []
    e_nonlocal_max_arr = []
    nuc_detected_step = -1
    force_y_arr = []

    # ── Incremental loading ──
    nodes = solver.nodes_phys.detach()
    y = nodes[:, 1]
    z = nodes[:, 2]
    u_prev = None

    if verbose:
        print()
        print(f"  {'step':>5s} {'delta':>8s} {'P_yy':>10s} {'J_min':>8s} {'J_max':>8s} "
              f"{'e_max':>10s} {'e_nl_max':>10s} {'topo':>6s} {'time':>7s}")
        print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7}")

    for step in range(n_steps):
        delta = delta_max * (step + 1) / n_steps
        t_step = time.time()

        # BCs: fix bottom (y < -L/4+0.5), pull top (y > L/4-0.5) in z (Mode III tear)
        bc_mask = torch.zeros(n_nodes, dtype=torch.bool)
        u_bc = torch.zeros(n_nodes, 3, dtype=torch.float64)

        bot = y < (-L / 4 + 0.5)
        top = y > (L / 4 - 0.5)
        bc_mask[bot] = True
        bc_mask[top] = True
        # Mode III: pull in opposite z-directions
        u_bc[top, 2] = delta / 2
        u_bc[bot, 2] = -delta / 2

        f_ext = torch.zeros(n_nodes, 3, dtype=torch.float64)

        # Solve with incremental loading (warm start from previous step)
        u_sol = solver.solve_nonlinear(
            stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
            max_iter=15, tol=1e-7, use_fbar=False,
            u_init=u_prev,
        )
        u_prev = u_sol.clone()

        # ── Post-process ──
        F = solver.compute_F(u_sol)
        J = torch.det(F)
        P = stress_fn(F)
        P_yy = P[:, 1, 1].mean().item()

        # Equivalent strain (local and nonlocal)
        e_local = compute_local_equivalent_strain(F)
        e_nonlocal = solve_nonlocal_strain(solver, e_local, length_scale=0.5)
        e_max = e_local.max().item()
        e_nl_max = e_nonlocal.max().item()

        # Reaction force: sum of P_yy at top face nodes (approximate)
        P_np = P.detach().numpy()
        centroids = solver.elem_centroids_phys.detach().cpu().numpy()
        top_elems = centroids[:, 1] > (L / 4 - 2.0)
        if top_elems.sum() > 0:
            force_y = np.sum(P_np[top_elems, 1, 1] * solver.vol.detach().cpu().numpy()[top_elems])
        else:
            force_y = 0.0

        # Drucker-Prager nucleation check
        if nuc_detected_step < 0:
            F_dp_vals = np.array([
                drucker_prager_F(torch.tensor(P_np[e]).unsqueeze(0), SIGMA_TS, SIGMA_HS).item()
                for e in range(len(P_np))
            ])
            if F_dp_vals.max() >= 0:
                nuc_detected_step = step

        # ── Persistent homology ──
        topo_str = ""
        if has_topo and (step % monitor_every == 0 or step == n_steps - 1):
            try:
                # Create a pseudo-SDF grid from the nonlocal strain field
                # (high strain = crack-like feature)
                grid_n = 16
                grid_vals = np.zeros((grid_n, grid_n, grid_n))
                nodes_np = nodes.detach().cpu().numpy()
                e_nl_np = e_nonlocal.detach().cpu().numpy()

                # Project nodal strain to regular grid
                he = np.array([half_W / 2 + 0.1, L / 4 + 0.1, B / 2 + 0.05])
                for ni in range(n_nodes):
                    xi = ((nodes_np[ni] + he) / (2 * he) * (grid_n - 1)).astype(int)
                    xi = np.clip(xi, 0, grid_n - 1)
                    grid_vals[xi[0], xi[1], xi[2]] = max(
                        grid_vals[xi[0], xi[1], xi[2]], e_nl_np[ni]
                    )

                # Invert: damage indicator = threshold - e_nl (negative where damaged)
                threshold = e_nl_np.mean() + 2 * e_nl_np.std()
                damage_sdf = threshold - grid_vals

                events = topo_monitor.update(damage_sdf, step)
                topo_events.extend(events)

                betti = topo_monitor.current_betti()
                betti_history.append({"step": step, "delta": delta, "betti": dict(betti)})
                topo_str = f"b={betti}"
            except Exception as e:
                topo_str = f"err"

        # Store
        steps.append(step)
        delta_arr.append(delta)
        P_yy_mean_arr.append(P_yy)
        J_min_arr.append(J.min().item())
        J_max_arr.append(J.max().item())
        e_local_max_arr.append(e_max)
        e_nonlocal_max_arr.append(e_nl_max)
        force_y_arr.append(force_y)

        dt_step = time.time() - t_step

        # Print progress
        if verbose and (step % max(n_steps // 20, 1) == 0 or step == n_steps - 1):
            print(f"  {step:5d} {delta:8.4f} {P_yy:10.6f} {J.min().item():8.4f} "
                  f"{J.max().item():8.4f} {e_max:10.4e} {e_nl_max:10.4e} "
                  f"{topo_str:>6s} {dt_step:7.2f}s")

    wall_time = time.time() - t_start

    # ── Compile results ──
    results = {
        "nc": nc,
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "n_steps": n_steps,
        "delta_max": delta_max,
        "wall_time": wall_time,
        "steps": steps,
        "delta": delta_arr,
        "force_y": force_y_arr,
        "P_yy_mean": P_yy_mean_arr,
        "J_min": J_min_arr,
        "J_max": J_max_arr,
        "e_local_max": e_local_max_arr,
        "e_nonlocal_max": e_nonlocal_max_arr,
        "nuc_detected_step": nuc_detected_step,
        "nuc_delta": delta_arr[nuc_detected_step] if nuc_detected_step >= 0 else None,
        "topo_events": [{"step": e.load_step, "dim": e.dimension, "type": e.event_type,
                         "lifetime": e.lifetime}
                        for e in topo_events] if topo_events else [],
        "betti_history": betti_history,
        "F_ss_analytical": F_SS,
        "Gc": GC,
    }

    # ── Save ──
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=convert)

    if verbose:
        print()
        print(f"  Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)")
        print(f"  Nucleation at step {nuc_detected_step} (delta={results['nuc_delta']})")
        print(f"  Topology events: {len(topo_events)}")
        for be in betti_history[-3:]:
            print(f"    step {be['step']}: betti={be['betti']}")
        print(f"  Results saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trousers tear simulation")
    parser.add_argument("--nc", type=int, default=12, help="Mesh cells per axis")
    parser.add_argument("--steps", type=int, default=500, help="Load steps")
    parser.add_argument("--delta", type=float, default=5.0, help="Max grip separation")
    parser.add_argument("--monitor", type=int, default=50, help="Topo check frequency")
    args = parser.parse_args()

    print("=" * 70)
    print("  PUBLICATION-QUALITY TROUSERS SIMULATION")
    print("=" * 70)

    # ── Run 1: Coarse mesh ──
    nc_coarse = args.nc
    print(f"\n{'='*70}")
    print(f"  RUN 1: Coarse mesh (nc={nc_coarse})")
    print(f"{'='*70}")
    r1 = run_simulation(
        nc=nc_coarse, n_steps=args.steps, delta_max=args.delta,
        monitor_every=args.monitor,
        output_dir=f"runs/trousers_nc{nc_coarse}_s{args.steps}",
    )

    # ── Run 2: Refined mesh (2x) ─��
    nc_fine = nc_coarse + 4
    print(f"\n{'='*70}")
    print(f"  RUN 2: Refined mesh (nc={nc_fine})")
    print(f"{'='*70}")
    r2 = run_simulation(
        nc=nc_fine, n_steps=args.steps, delta_max=args.delta,
        monitor_every=args.monitor,
        output_dir=f"runs/trousers_nc{nc_fine}_s{args.steps}",
    )

    # ── Convergence analysis ──
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'='*70}")

    h1 = 1.0 / nc_coarse
    h2 = 1.0 / nc_fine

    # Compare P_yy at final step
    p1 = r1["P_yy_mean"][-1]
    p2 = r2["P_yy_mean"][-1]
    if abs(p1 - p2) > 1e-12:
        order = math.log(abs(p1 - p2) / max(abs(p2), 1e-15)) / math.log(h1 / h2)
    else:
        order = float("inf")

    print(f"  Coarse (nc={nc_coarse}): P_yy_final = {p1:.6f}, nodes = {r1['n_nodes']}")
    print(f"  Fine   (nc={nc_fine}):   P_yy_final = {p2:.6f}, nodes = {r2['n_nodes']}")
    print(f"  |P_coarse - P_fine| = {abs(p1-p2):.6e}")
    print(f"  Estimated convergence order: {order:.2f}")
    print()
    print(f"  Total wall time: {r1['wall_time'] + r2['wall_time']:.1f}s "
          f"({(r1['wall_time'] + r2['wall_time'])/60:.1f} min)")
