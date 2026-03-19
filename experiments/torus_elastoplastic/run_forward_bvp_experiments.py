#!/usr/bin/env python3
"""CMAME-quality experimental campaign for the 8-chart Schwarz forward
elastoplastic BVP on the torus.

Experiments:
  1. Mesh refinement study: n_cells in {4, 6, 8}, same loading protocol
  2. Schwarz convergence: track interface_jump per sweep at n_cells=6
  3. Load-step sensitivity: steps_per_half in {10, 20, 40}, n_cells=6

All results are saved to runs/torus_forward_bvp_experiments/ with JSON
logs and PyVista-rendered figures.

Usage:
    python experiments/torus_elastoplastic/run_forward_bvp_experiments.py
    python experiments/torus_elastoplastic/run_forward_bvp_experiments.py --exp mesh
    python experiments/torus_elastoplastic/run_forward_bvp_experiments.py --exp schwarz
    python experiments/torus_elastoplastic/run_forward_bvp_experiments.py --exp timestep
    python experiments/torus_elastoplastic/run_forward_bvp_experiments.py --exp all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO)

from experiments.torus_elastoplastic.run_forward_bvp_schwarz import (
    TorusSDF, TorusChartDecoder, SchwarzVectorSolver,
    make_ep_stress_fn, make_ep_tangent_fn, build_bc,
    R_MAJOR, R_MINOR, N_CHARTS,
)
from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.return_mapping import (
    ReturnMappingState, smooth_return_map,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Shared solver driver
# ═══════════════════════════════════════════════════════════════════════════

def run_bvp(
    n_cells: int = 6,
    delta_max_frac: float = 0.02,
    steps_per_half: int = 30,
    n_cycles: int = 1,
    n_schwarz: int = 8,
    max_newton: int = 25,
    newton_tol: float = 1e-7,
    eps_rm: float = 0.01,
    track_schwarz_convergence: bool = False,
    verbose: bool = True,
) -> Dict:
    """Run the full forward BVP and return diagnostics."""
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    E, nu = 200.0, 0.3
    mu = torch.tensor(E / (2 * (1 + nu)), device=device)
    K = torch.tensor(E / (3 * (1 - 2 * nu)), device=device)
    tau_y = torch.tensor(0.5, device=device)
    H_kin = torch.tensor(20.0, device=device)

    # Build atlas
    sdf = TorusSDF()
    phi_centers = [i * 2 * math.pi / N_CHARTS for i in range(N_CHARTS)]
    decoders = [TorusChartDecoder(phi_center=pc) for pc in phi_centers]
    solvers = []
    for dec in decoders:
        sol = ChartVectorFEMSolver(
            n_cells=n_cells, support_r=1.0, chart_decoder=dec,
            sdf_oracle=sdf, sdf_threshold=-0.005,
            device=device, dtype=torch.float64)
        solvers.append(sol)

    total_nodes = sum(s.n_nodes for s in solvers)
    total_elems = sum(s.n_elements for s in solvers)
    total_dof = total_nodes * 3

    if verbose:
        print(f"  Mesh: n_cells={n_cells}, {total_nodes} nodes, "
              f"{total_elems} elements, {total_dof} DOF")

    schwarz = SchwarzVectorSolver(solvers=solvers, chart_decoders=decoders)

    # Initialize plastic state
    states = [ReturnMappingState.zeros((s.n_elements,), device=device,
              dtype=torch.float64) for s in solvers]
    F_olds = [torch.eye(3, device=device, dtype=torch.float64
              ).unsqueeze(0).expand(s.n_elements, 3, 3).clone() for s in solvers]

    # Loading schedule
    delta_max = delta_max_frac * R_MINOR
    sph = steps_per_half
    deltas = []
    for _ in range(n_cycles):
        for s in range(sph):
            deltas.append(delta_max * (s + 1) / sph)
        for s in range(2 * sph):
            deltas.append(delta_max * (1.0 - (s + 1) / sph))
        for s in range(sph):
            deltas.append(delta_max * (-1.0 + (s + 1) / sph))
    n_steps = len(deltas)

    # Run
    history = {"step": [], "delta": [], "max_u": [], "max_ep_bar": [],
               "interface_jump": [], "wall_time": [], "newton_iters": []}
    schwarz_convergence = []  # per-step list of per-sweep jumps

    for step_idx, delta in enumerate(deltas):
        t0 = time.time()
        bc_masks, u_bcs = build_bc(solvers, decoders, delta)

        stress_fns = [make_ep_stress_fn(states[ci], F_olds[ci], mu, K, tau_y, H_kin, eps_rm)
                      for ci in range(N_CHARTS)]
        tangent_fns = [make_ep_tangent_fn(states[ci], F_olds[ci], mu, K, tau_y, H_kin, eps_rm)
                       for ci in range(N_CHARTS)]

        sweep_jumps = []
        total_newton = 0

        for sweep in range(n_schwarz):
            for group in schwarz.color_groups:
                for ci in group:
                    if solvers[ci].n_elements == 0:
                        continue
                    iface_mask, iface_vals = schwarz._interpolate_interface_bc(ci)
                    bc_mask = bc_masks[ci].clone()
                    bc_vals = u_bcs[ci].clone()
                    iface_only = iface_mask & ~bc_mask
                    bc_mask |= iface_only
                    bc_vals[iface_only] = iface_vals[iface_only]

                    f_ext = torch.zeros(solvers[ci].n_nodes, 3,
                                        device=device, dtype=torch.float64)
                    try:
                        u_new = solvers[ci].solve_nonlinear(
                            stress_fn=stress_fns[ci], tangent_fn=tangent_fns[ci],
                            f_ext=f_ext, u_bc=bc_vals, bc_mask=bc_mask,
                            u_init=schwarz.u_charts[ci],
                            max_iter=max_newton, tol=newton_tol)
                        schwarz.u_charts[ci] = u_new
                        total_newton += 1  # approximate
                    except Exception as e:
                        if verbose:
                            print(f"    [WARN] Chart {ci} sweep {sweep}: {e}")

            if track_schwarz_convergence:
                sweep_jumps.append(schwarz.interface_jump())

        if track_schwarz_convergence:
            schwarz_convergence.append(sweep_jumps)

        # Update plastic state
        with torch.no_grad():
            for ci in range(N_CHARTS):
                if solvers[ci].n_elements == 0:
                    continue
                F_cur = solvers[ci].compute_F(schwarz.u_charts[ci])
                F_old_inv = torch.linalg.inv(F_olds[ci])
                F_delta = torch.einsum("eij,ejk->eik", F_cur, F_old_inv)
                _, new_state = smooth_return_map(
                    F_delta, states[ci], mu, K, tau_y, H_kin, eps_rm)
                states[ci] = new_state
                F_olds[ci] = F_cur.detach().clone()

        max_u = max(torch.norm(schwarz.u_charts[ci], dim=1).max().item()
                    for ci in range(N_CHARTS) if solvers[ci].n_elements > 0)
        max_ep = max(states[ci].ep_bar.max().item()
                     for ci in range(N_CHARTS) if solvers[ci].n_elements > 0)
        jump = schwarz.interface_jump()
        dt = time.time() - t0

        history["step"].append(step_idx)
        history["delta"].append(delta)
        history["max_u"].append(max_u)
        history["max_ep_bar"].append(max_ep)
        history["interface_jump"].append(jump)
        history["wall_time"].append(dt)
        history["newton_iters"].append(total_newton)

        if verbose and (step_idx % 10 == 0 or step_idx == n_steps - 1):
            print(f"    Step {step_idx+1:4d}/{n_steps}: delta={delta:.5f}, "
                  f"max|u|={max_u:.4e}, ep_bar={max_ep:.4e}, "
                  f"jump={jump:.4e}, t={dt:.1f}s")

    result = {
        "n_cells": n_cells,
        "total_nodes": total_nodes,
        "total_elements": total_elems,
        "total_dof": total_dof,
        "n_steps": n_steps,
        "delta_max": delta_max,
        "steps_per_half": steps_per_half,
        "n_schwarz": n_schwarz,
        "history": history,
        "total_time": sum(history["wall_time"]),
    }
    if track_schwarz_convergence:
        result["schwarz_convergence"] = schwarz_convergence

    # Save final state for post-processing
    result["final_states"] = [{
        "ep_bar": states[ci].ep_bar.cpu().numpy().tolist(),
    } for ci in range(N_CHARTS)]

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 1: Mesh refinement study
# ═══════════════════════════════════════════════════════════════════════════

def experiment_mesh_refinement(out_dir: Path):
    """Run at n_cells in {4, 6, 8} with identical loading."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Mesh Refinement Study")
    print("="*70)

    results = {}
    for nc in [4, 6, 8]:
        print(f"\n--- n_cells = {nc} ---")
        res = run_bvp(
            n_cells=nc,
            delta_max_frac=0.02,
            steps_per_half=30,
            n_cycles=1,
            n_schwarz=8,
            verbose=True,
        )
        results[nc] = res
        print(f"  Completed: {res['total_time']:.1f}s, "
              f"final ep_bar={res['history']['max_ep_bar'][-1]:.6e}")

    # Save
    summary = {}
    for nc, res in results.items():
        summary[str(nc)] = {
            "n_cells": nc,
            "total_nodes": res["total_nodes"],
            "total_elements": res["total_elements"],
            "total_dof": res["total_dof"],
            "final_max_ep_bar": res["history"]["max_ep_bar"][-1],
            "final_interface_jump": res["history"]["interface_jump"][-1],
            "total_time": res["total_time"],
            "avg_time_per_step": res["total_time"] / res["n_steps"],
        }
    with open(out_dir / "mesh_refinement.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full histories
    for nc, res in results.items():
        with open(out_dir / f"history_ncells{nc}.json", "w") as f:
            json.dump(res["history"], f, indent=2)

    print("\n--- Mesh Refinement Summary ---")
    print(f"{'n_cells':>8} {'Nodes':>8} {'Elems':>8} {'DOF':>8} "
          f"{'ep_bar':>12} {'jump':>12} {'Time(s)':>10}")
    for nc in [4, 6, 8]:
        s = summary[str(nc)]
        print(f"{nc:8d} {s['total_nodes']:8d} {s['total_elements']:8d} "
              f"{s['total_dof']:8d} {s['final_max_ep_bar']:12.6e} "
              f"{s['final_interface_jump']:12.4e} {s['total_time']:10.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 2: Schwarz iteration convergence
# ═══════════════════════════════════════════════════════════════════════════

def experiment_schwarz_convergence(out_dir: Path):
    """Track interface_jump per Schwarz sweep at n_cells=6."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Schwarz Iteration Convergence")
    print("="*70)

    res = run_bvp(
        n_cells=6,
        delta_max_frac=0.02,
        steps_per_half=30,
        n_cycles=1,
        n_schwarz=10,  # more sweeps to track convergence
        track_schwarz_convergence=True,
        verbose=True,
    )

    # Save convergence data
    conv_data = res["schwarz_convergence"]
    with open(out_dir / "schwarz_convergence.json", "w") as f:
        json.dump({
            "n_steps": res["n_steps"],
            "n_schwarz": 10,
            "per_step_sweeps": conv_data,
        }, f, indent=2)

    # Print summary for a few representative steps
    print("\n--- Schwarz Convergence (selected steps) ---")
    print(f"{'Step':>5} " + " ".join(f"{'Sweep '+str(i+1):>12}" for i in range(10)))
    for step_idx in [0, 29, 59, 89, 119]:
        if step_idx < len(conv_data):
            sweeps = conv_data[step_idx]
            line = f"{step_idx+1:5d} " + " ".join(f"{s:12.4e}" for s in sweeps)
            print(line)


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 3: Load-step sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def experiment_timestep_sensitivity(out_dir: Path):
    """Vary steps_per_half in {10, 20, 40} at n_cells=6."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Load-Step Sensitivity")
    print("="*70)

    results = {}
    for sph in [10, 20, 40]:
        print(f"\n--- steps_per_half = {sph} ---")
        res = run_bvp(
            n_cells=6,
            delta_max_frac=0.02,
            steps_per_half=sph,
            n_cycles=1,
            n_schwarz=8,
            verbose=True,
        )
        results[sph] = res
        print(f"  Completed: {res['total_time']:.1f}s, "
              f"final ep_bar={res['history']['max_ep_bar'][-1]:.6e}")

    summary = {}
    for sph, res in results.items():
        summary[str(sph)] = {
            "steps_per_half": sph,
            "n_steps": res["n_steps"],
            "final_max_ep_bar": res["history"]["max_ep_bar"][-1],
            "peak_max_ep_bar": max(res["history"]["max_ep_bar"]),
            "final_interface_jump": res["history"]["interface_jump"][-1],
            "total_time": res["total_time"],
        }
    with open(out_dir / "timestep_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)

    for sph, res in results.items():
        with open(out_dir / f"history_sph{sph}.json", "w") as f:
            json.dump(res["history"], f, indent=2)

    print("\n--- Load-Step Sensitivity Summary ---")
    print(f"{'sph':>6} {'Steps':>6} {'peak_ep':>12} {'final_ep':>12} "
          f"{'jump':>12} {'Time(s)':>10}")
    for sph in [10, 20, 40]:
        s = summary[str(sph)]
        print(f"{sph:6d} {s['n_steps']:6d} {s['peak_max_ep_bar']:12.6e} "
              f"{s['final_max_ep_bar']:12.6e} {s['final_interface_jump']:12.4e} "
              f"{s['total_time']:10.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        choices=["mesh", "schwarz", "timestep", "all"])
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.n_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    out_dir = Path(_REPO) / "runs" / "torus_forward_bvp_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    if args.exp in ("mesh", "all"):
        experiment_mesh_refinement(out_dir)

    if args.exp in ("schwarz", "all"):
        experiment_schwarz_convergence(out_dir)

    if args.exp in ("timestep", "all"):
        experiment_timestep_sensitivity(out_dir)

    print(f"\n{'='*70}")
    print(f"All experiments completed in {time.time()-t_total:.0f}s")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
