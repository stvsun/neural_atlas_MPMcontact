#!/usr/bin/env python3
"""Publication-quality multi-chart trousers tear simulation.

6-chart Schwarz domain decomposition with:
  - Neo-Hookean hyperelasticity (finite strain, incremental loading)
  - Persistent homology monitoring (GUDHI)
  - Drucker-Prager + nonlocal nucleation detection
  - Checkpoint/restart for long runs (~10 hours)
  - Per-chart VTU output for ParaView

Reference: Kamarei et al. (2026), CMAME 448, Section 5.2

Usage:
    python nineO_examples/9_trousers/run_publication_multichart.py
    python nineO_examples/9_trousers/run_publication_multichart.py --restart runs/trousers_publication/checkpoint_00100.pt
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver
from solvers.fem.analytic_decoders import BoxDecoder
from solvers.fracture_criteria import drucker_prager_F
from solvers.fem.nonlocal_damage import compute_local_equivalent_strain, solve_nonlocal_strain

# Import VTU writer from the single-chart version
import importlib.util
_vtu_spec = importlib.util.spec_from_file_location(
    "run_publication",
    os.path.join(os.path.dirname(__file__), "run_publication.py"))
_vtu_mod = importlib.util.module_from_spec(_vtu_spec)
_vtu_spec.loader.exec_module(_vtu_mod)
write_vtu = _vtu_mod.write_vtu
write_pvd = _vtu_mod.write_pvd

# ═══════════════════════════════════════════════════════════════
# MATERIAL
# ═══════════════════════════════════════════════════════════════
MU = 0.52
LAM = 85.77
K_BULK = LAM + 2 * MU / 3
GC = 0.041
SIGMA_TS = 0.3
SIGMA_HS = 1.0


# ═══════════════════════════════════════════════════════════════
# 6-CHART LAYOUT
# ═══════════════════════════════════════════════════════════════

def build_charts(nc=16):
    """Build 6 overlapping BoxDecoder charts covering the trousers sheet.

    Layout (2 columns x 3 rows):
      Left column:  x in [-20, +2]  (covers left leg + overlap at crack)
      Right column: x in [-2, +20]  (covers right leg + overlap at crack)
      Row 0 (bottom legs): y in [0, 40]
      Row 1 (crack tip):   y in [30, 70]     (10mm overlap with rows 0,2)
      Row 2 (intact top):  y in [60, 100]

    Returns chart_solvers, decoders, seeds, neighbors, stress_fn, tangent_fn
    """
    # Chart parameters: (center_x, center_y, center_z, half_x, half_y, half_z)
    chart_params = [
        # Row 0: bottom legs (y=0..40)
        (-9.0, 20.0, 0.0,   11.5, 20.5, 0.55),   # Chart 0: left-bottom
        ( 9.0, 20.0, 0.0,   11.5, 20.5, 0.55),   # Chart 1: right-bottom
        # Row 1: crack tip region (y=30..70)
        (-9.0, 50.0, 0.0,   11.5, 20.5, 0.55),   # Chart 2: left-mid
        ( 9.0, 50.0, 0.0,   11.5, 20.5, 0.55),   # Chart 3: right-mid
        # Row 2: intact top (y=60..100)
        (-9.0, 80.0, 0.0,   11.5, 20.5, 0.55),   # Chart 4: left-top
        ( 9.0, 80.0, 0.0,   11.5, 20.5, 0.55),   # Chart 5: right-top
    ]

    neighbors = [
        [1, 2],       # Chart 0: overlaps with 1 (x-overlap) and 2 (y-overlap)
        [0, 3],       # Chart 1: overlaps with 0 and 3
        [0, 3, 4],    # Chart 2: overlaps with 0, 3, 4
        [1, 2, 5],    # Chart 3: overlaps with 1, 2, 5
        [2, 5],       # Chart 4: overlaps with 2 and 5
        [3, 4],       # Chart 5: overlaps with 3 and 4
    ]

    decoders = []
    solvers = []
    seeds = []

    for ci, (cx, cy, cz, hx, hy, hz) in enumerate(chart_params):
        dec = BoxDecoder(center=(cx, cy, cz), half_extents=(hx, hy, hz)).double()
        s = ChartVectorFEMSolver(
            n_cells=nc, support_r=1.0, chart_decoder=dec,
            decoder_kwargs={}, device="cpu", dtype=torch.float64,
        )
        decoders.append(dec)
        solvers.append(s)
        seeds.append([cx, cy, cz])

    seeds_t = torch.tensor(seeds, dtype=torch.float64)

    # Create material model from the first solver
    stress_fn, tangent_fn = solvers[0].make_neo_hookean(MU, K_BULK)

    total_nodes = sum(s.n_nodes for s in solvers)
    total_elems = sum(s.n_elements for s in solvers)
    print(f"  Charts: {len(solvers)}, total nodes: {total_nodes}, total tets: {total_elems}")
    for ci, s in enumerate(solvers):
        print(f"    Chart {ci}: {s.n_nodes:6d} nodes, {s.n_elements:6d} tets")

    return solvers, decoders, seeds_t, neighbors, stress_fn, tangent_fn


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE LOADING SCHEDULE
# ═══════════════════════════════════════════════════════════════

def make_loading_schedule(delta_max=65.0, n_total=500):
    """Create adaptive delta schedule: coarse early, fine near nucleation.

    Phase 1: 0-40mm in 100 steps  (Δδ=0.4mm)
    Phase 2: 40-55mm in 200 steps (Δδ=0.075mm) — nucleation zone
    Phase 3: 55-65mm in 200 steps (Δδ=0.05mm) — post-nucleation
    """
    phase1 = np.linspace(0, 40.0, 101)[1:]       # 100 steps
    phase2 = np.linspace(40.0, 55.0, 201)[1:]    # 200 steps
    phase3 = np.linspace(55.0, delta_max, 200)[1:]  # 199 steps
    schedule = np.concatenate([phase1, phase2, phase3])
    # Trim or pad to n_total
    if len(schedule) > n_total:
        schedule = schedule[:n_total]
    return schedule


# ═══════════════════════════════════════════════════════════════
# BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════

def make_bc_fn(delta):
    """Mode III tearing BCs: pull legs in opposite z-directions.

    Bottom leg (y < 2): fix u=0
    Top grip (y > 98): pull in +z by delta/2
    """
    def bc_fn(nodes_phys):
        n = len(nodes_phys)
        u = np.zeros((n, 3))
        mask = np.zeros(n, dtype=bool)

        y = nodes_phys[:, 1]
        bot = y < 2.0
        top = y > 98.0

        mask[bot] = True     # fix bottom leg
        mask[top] = True
        u[top, 2] = delta    # pull top in +z (Mode III)

        return u, mask
    return bc_fn


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(path, schwarz, step, delta, results_acc, nuc_step, topo_events, betti_hist):
    """Save full simulation state for restart."""
    ckpt = {
        "step": step,
        "delta": delta,
        "solver_state": schwarz.save_state(),
        "results": {k: list(v) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in results_acc.items()},
        "nuc_detected_step": nuc_step,
        "topo_events_count": len(topo_events),
        "betti_history": betti_hist,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, schwarz):
    """Restore simulation state from checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    schwarz.load_state(ckpt["solver_state"])
    return ckpt


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════

def run(nc=16, delta_max=65.0, n_steps=500, output_dir="runs/trousers_publication",
        checkpoint_every=20, vtu_every=5, topo_every=25, restart_from=None):

    os.makedirs(output_dir, exist_ok=True)
    vtu_dir = os.path.join(output_dir, "vtu")
    os.makedirs(vtu_dir, exist_ok=True)

    t_global = time.time()

    print("=" * 70)
    print(f"  TROUSERS MULTI-CHART SIMULATION")
    print(f"  nc={nc}, delta_max={delta_max}mm, {n_steps} steps")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ── Build charts ──
    solvers, decoders, seeds, neighbors, stress_fn, tangent_fn = build_charts(nc)

    schwarz = SchwarzVectorFEMSolver(
        chart_solvers=solvers, seeds=seeds, decoders=decoders,
        neighbors=neighbors, parallel=False,  # serial for stability
    )

    # ── Loading schedule ──
    schedule = make_loading_schedule(delta_max, n_steps)
    actual_steps = len(schedule)
    print(f"  Loading: {actual_steps} steps, delta range [{schedule[0]:.3f}, {schedule[-1]:.3f}]mm")

    # ── Topology monitor ──
    try:
        from atlas.topo.monitor import TopologyMonitor
        topo_monitor = TopologyMonitor(
            lifetime_threshold=0.05, bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1), relative_threshold=True, verbose=False,
        )
        has_topo = True
        print("  Topology: GUDHI enabled")
    except ImportError:
        has_topo = False
        print("  Topology: GUDHI not available")

    # ── Result accumulators ──
    results = {
        "delta": [], "P_yy_mean": [], "J_min": [], "J_max": [],
        "e_local_max": [], "e_nonlocal_max": [], "schwarz_iters": [],
        "schwarz_residual": [],
    }
    nuc_detected_step = -1
    topo_events = []
    betti_history = []
    vtu_files = []

    # ── Restart ──
    start_step = 0
    if restart_from:
        print(f"  Restarting from: {restart_from}")
        ckpt = load_checkpoint(restart_from, schwarz)
        start_step = ckpt["step"] + 1
        results = ckpt["results"]
        nuc_detected_step = ckpt["nuc_detected_step"]
        betti_history = ckpt.get("betti_history", [])
        print(f"  Resumed at step {start_step}, delta={schedule[start_step-1]:.3f}mm")

    # ── Main loop ──
    print()
    print(f"  {'step':>5s} {'delta':>8s} {'schwarz':>8s} {'P_yy':>10s} "
          f"{'J_min':>8s} {'J_max':>8s} {'e_max':>10s} {'nuc':>4s} {'topo':>6s} {'dt':>7s}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*4} {'-'*6} {'-'*7}")

    for step_idx in range(start_step, actual_steps):
        delta = schedule[step_idx]
        t_step = time.time()

        # ── Schwarz solve ──
        bc_fn = make_bc_fn(delta)
        u_charts = schwarz.solve(
            stress_fn, tangent_fn, bc_fn,
            max_schwarz_iters=15, tol=1e-2, relaxation=0.5,
        )

        # ── Post-process all charts ──
        all_P_yy = []
        all_J = []
        all_e_local = []
        all_e_nonlocal = []

        for ci, sol in enumerate(solvers):
            if u_charts[ci] is None or sol.n_elements == 0:
                continue
            F = sol.compute_F(u_charts[ci])
            J = torch.det(F)
            P = stress_fn(F)
            e_loc = compute_local_equivalent_strain(F)
            e_nl = solve_nonlocal_strain(sol, e_loc, length_scale=0.5)

            all_P_yy.append(P[:, 1, 1].mean().item())
            all_J.extend(J.detach().cpu().numpy().tolist())
            all_e_local.append(e_loc.max().item())
            all_e_nonlocal.append(e_nl.max().item())

        P_yy_mean = np.mean(all_P_yy) if all_P_yy else 0
        J_arr = np.array(all_J) if all_J else np.array([1.0])
        e_max = max(all_e_local) if all_e_local else 0
        e_nl_max = max(all_e_nonlocal) if all_e_nonlocal else 0

        # ── Nucleation check ──
        nuc_str = ""
        if nuc_detected_step < 0:
            for ci, sol in enumerate(solvers):
                if u_charts[ci] is None or sol.n_elements == 0:
                    continue
                F = sol.compute_F(u_charts[ci])
                P = stress_fn(F)
                P_np = P.detach().numpy()
                for e in range(len(P_np)):
                    fdp = drucker_prager_F(
                        torch.tensor(P_np[e]).unsqueeze(0), SIGMA_TS, SIGMA_HS
                    ).item()
                    if fdp >= 0:
                        nuc_detected_step = step_idx
                        nuc_str = "NUC!"
                        break
                if nuc_detected_step >= 0:
                    break

        # ── Topology monitoring ──
        topo_str = ""
        if has_topo and (step_idx % topo_every == 0 or step_idx == actual_steps - 1):
            try:
                grid_n = 24
                grid_vals = np.zeros((grid_n, grid_n, grid_n))
                # Global bounding box: x∈[-20,20], y∈[0,100], z∈[-0.55,0.55]
                bb_min = np.array([-20.0, 0.0, -0.55])
                bb_max = np.array([20.0, 100.0, 0.55])
                bb_range = bb_max - bb_min

                for ci, sol in enumerate(solvers):
                    if u_charts[ci] is None or sol.n_elements == 0:
                        continue
                    nodes_np = sol.nodes_phys.detach().cpu().numpy()
                    F = sol.compute_F(u_charts[ci])
                    e_loc = compute_local_equivalent_strain(F)
                    e_nl = solve_nonlocal_strain(sol, e_loc, length_scale=0.5)
                    e_nl_np = e_nl.detach().cpu().numpy()

                    for ni in range(sol.n_nodes):
                        xi = ((nodes_np[ni] - bb_min) / bb_range * (grid_n - 1)).astype(int)
                        xi = np.clip(xi, 0, grid_n - 1)
                        grid_vals[xi[0], xi[1], xi[2]] = max(
                            grid_vals[xi[0], xi[1], xi[2]], e_nl_np[ni]
                        )

                threshold = grid_vals[grid_vals > 0].mean() + 2 * grid_vals[grid_vals > 0].std() \
                    if (grid_vals > 0).any() else 0.01
                damage_sdf = threshold - grid_vals

                events = topo_monitor.update(damage_sdf, step_idx)
                topo_events.extend(events)
                betti = topo_monitor.current_betti()
                betti_history.append({
                    "step": step_idx, "delta": float(delta), "betti": dict(betti)
                })
                topo_str = f"b={betti}"
            except Exception as e:
                topo_str = "err"

        # ── Store results ──
        results["delta"].append(float(delta))
        results["P_yy_mean"].append(float(P_yy_mean))
        results["J_min"].append(float(J_arr.min()))
        results["J_max"].append(float(J_arr.max()))
        results["e_local_max"].append(float(e_max))
        results["e_nonlocal_max"].append(float(e_nl_max))

        # ── VTU output ──
        if step_idx % vtu_every == 0 or step_idx == actual_steps - 1 or nuc_str:
            for ci, sol in enumerate(solvers):
                if u_charts[ci] is None or sol.n_elements == 0:
                    continue
                F = sol.compute_F(u_charts[ci])
                J = torch.det(F)
                P = stress_fn(F)
                e_loc = compute_local_equivalent_strain(F)
                e_nl = solve_nonlocal_strain(sol, e_loc, length_scale=0.5)
                vtu_path = os.path.join(vtu_dir, f"chart{ci}_{step_idx:05d}.vtu")
                write_vtu(vtu_path, sol.nodes_phys, sol.elements,
                          u_charts[ci], P, e_loc, e_nl, J, step_idx, delta)
                vtu_files.append((step_idx, delta, vtu_path))

        # ── Checkpoint ──
        if step_idx % checkpoint_every == 0 and step_idx > start_step:
            ckpt_path = os.path.join(output_dir, f"checkpoint_{step_idx:05d}.pt")
            save_checkpoint(ckpt_path, schwarz, step_idx, delta,
                            results, nuc_detected_step, topo_events, betti_history)

        dt = time.time() - t_step

        # ── Progress ──
        if step_idx % max(actual_steps // 50, 1) == 0 or nuc_str or step_idx == actual_steps - 1:
            print(f"  {step_idx:5d} {delta:8.3f} {'':>8s} {P_yy_mean:10.6f} "
                  f"{J_arr.min():8.4f} {J_arr.max():8.4f} {e_max:10.4e} "
                  f"{nuc_str:>4s} {topo_str:>6s} {dt:7.1f}s")
            sys.stdout.flush()

    wall_time = time.time() - t_global

    # ── Final output ──
    print()
    print(f"  Wall time: {wall_time:.0f}s ({wall_time/3600:.1f} hours)")
    print(f"  Nucleation: step={nuc_detected_step}, delta={schedule[nuc_detected_step] if nuc_detected_step >= 0 else 'N/A'}")
    print(f"  Topology events: {len(topo_events)}")
    for bh in betti_history[-5:]:
        print(f"    step {bh['step']}: betti={bh['betti']}")

    # Save final results
    final = {
        "nc": nc, "n_charts": len(solvers),
        "total_nodes": sum(s.n_nodes for s in solvers),
        "total_elements": sum(s.n_elements for s in solvers),
        "n_steps": actual_steps, "delta_max": delta_max,
        "wall_time": wall_time,
        "nuc_detected_step": nuc_detected_step,
        "nuc_delta": float(schedule[nuc_detected_step]) if nuc_detected_step >= 0 else None,
        "topo_events": [{"step": e.load_step, "dim": e.dimension,
                         "type": e.event_type, "lifetime": e.lifetime}
                        for e in topo_events],
        "betti_history": betti_history,
        **{k: v for k, v in results.items()},
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(final, f, indent=2, default=convert)

    # PVD collection
    write_pvd(os.path.join(output_dir, "trousers.pvd"), vtu_files)

    # Final checkpoint
    ckpt_path = os.path.join(output_dir, f"checkpoint_final.pt")
    save_checkpoint(ckpt_path, schwarz, actual_steps - 1, schedule[-1],
                    results, nuc_detected_step, topo_events, betti_history)

    print(f"\n  Results: {output_dir}/results.json")
    print(f"  VTU: {len(vtu_files)} files in {vtu_dir}/")
    print(f"  PVD: {output_dir}/trousers.pvd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=16)
    parser.add_argument("--delta", type=float, default=65.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="runs/trousers_publication")
    parser.add_argument("--restart", type=str, default=None, help="Checkpoint file to restart from")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--vtu-every", type=int, default=5)
    parser.add_argument("--topo-every", type=int, default=25)
    args = parser.parse_args()

    run(nc=args.nc, delta_max=args.delta, n_steps=args.steps,
        output_dir=args.output, checkpoint_every=args.checkpoint_every,
        vtu_every=args.vtu_every, topo_every=args.topo_every,
        restart_from=args.restart)
