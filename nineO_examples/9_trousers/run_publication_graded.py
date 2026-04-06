#!/usr/bin/env python3
"""Publication-quality trousers tear: graded mesh + enrichment + adaptive stepping.

Single chart with GradedBoxDecoder (fine near crack tip, coarse far-field),
DENNs SDF enrichment for crack discontinuity, and Newton-adaptive load control.

Usage:
    python nineO_examples/9_trousers/run_publication_graded.py --nc 18
    python nineO_examples/9_trousers/run_publication_graded.py --restart runs/trousers_graded/checkpoint_00100.pt
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
from solvers.fem.analytic_decoders import GradedBoxDecoder
from solvers.fem.denns_enrichment import enrich_decoder
from solvers.fracture_criteria import drucker_prager_F
from solvers.fem.nonlocal_damage import compute_local_equivalent_strain, solve_nonlocal_strain
from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle

# VTU writer
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "run_pub", os.path.join(os.path.dirname(__file__), "run_publication.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
write_vtu = _mod.write_vtu
write_pvd = _mod.write_pvd

# ═══════════════════════════════════════════════════════════════
# MATERIAL
# ═══════════════════════════════════════════════════════════════
MU = 0.52; LAM = 85.77; K_BULK = LAM + 2 * MU / 3
GC = 0.041; SIGMA_TS = 0.3; SIGMA_HS = 1.0
W = 40.0; L = 100.0; B = 1.0; A_CRACK = 50.0


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE LOAD CONTROLLER
# ═══════════════════════════════════════════════════════════════

class AdaptiveLoadController:
    """Newton-convergence-based adaptive load stepping.

    Adjusts load increment based on how many Newton iterations
    the previous step needed. Halves increment on failure; grows
    on easy steps.
    """

    def __init__(self, delta_max=65.0, delta_0=0.5,
                 target_iters=5, min_dd=0.01, max_dd=2.0):
        self.delta = 0.0
        self.d_delta = delta_0
        self.delta_max = delta_max
        self.target = target_iters
        self.min_dd = min_dd
        self.max_dd = max_dd
        self.history = []  # (delta, d_delta, newton_iters, converged)

    def advance(self, newton_iters, converged):
        """Update delta based on Newton performance. Returns new delta."""
        self.history.append((self.delta, self.d_delta, newton_iters, converged))

        if not converged:
            # Step failed — cut increment and retry from same delta
            self.d_delta = max(self.d_delta * 0.5, self.min_dd)
            return self.delta

        # Step succeeded — adjust increment for next step
        if newton_iters <= self.target:
            self.d_delta = min(self.d_delta * 1.25, self.max_dd)
        elif newton_iters >= self.target * 2:
            self.d_delta = max(self.d_delta * 0.7, self.min_dd)

        self.delta = min(self.delta + self.d_delta, self.delta_max)
        return self.delta

    @property
    def finished(self):
        return self.delta >= self.delta_max

    def state_dict(self):
        return {"delta": self.delta, "d_delta": self.d_delta, "history": self.history}

    def load_state_dict(self, d):
        self.delta = d["delta"]; self.d_delta = d["d_delta"]; self.history = d["history"]


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════

def run(nc=18, delta_max=65.0, grade_power=3.0, use_enrichment=True,
        output_dir="runs/trousers_graded", max_steps=2000,
        checkpoint_every=20, vtu_every=5, topo_every=25,
        restart_from=None):

    os.makedirs(output_dir, exist_ok=True)
    vtu_dir = os.path.join(output_dir, "vtu")
    os.makedirs(vtu_dir, exist_ok=True)
    t_global = time.time()

    print("=" * 70)
    print(f"  TROUSERS — GRADED MESH + ENRICHMENT + ADAPTIVE STEPPING")
    print(f"  nc={nc}, grade_power={grade_power}, delta_max={delta_max}mm")
    print(f"  enrichment={'ON' if use_enrichment else 'OFF'}")
    print("=" * 70)

    # ── Build graded mesh ──
    # Center at crack tip (y=50), cover full sheet
    base_dec = GradedBoxDecoder(
        center=(0, L / 2, 0),
        half_extents=(W / 2 + 0.1, L / 2 + 0.1, B / 2 + 0.05),
        grade_axis=1,        # grade along y (crack at y=50)
        grade_power=grade_power,
    ).double()

    # ── SDF enrichment (optional) ──
    if use_enrichment:
        # Crack SDF: the pre-crack runs along y < 50 at x=0
        crack_sdf = CrackedPlateSDFOracle(
            a=A_CRACK, W=W / 2, H=L / 2, T=B, delta=0.05
        )
        def sdf_fn(x_phys):
            return crack_sdf.sdf(x_phys)

        decoder = enrich_decoder(base_dec, sdf_fn, epsilon=0.1,
                                  enrichment_width=32, enrichment_depth=2)
        print(f"  Decoder: GradedBoxDecoder + SDFEnrichedDecoder")
    else:
        decoder = base_dec
        print(f"  Decoder: GradedBoxDecoder (no enrichment)")

    solver = ChartVectorFEMSolver(
        n_cells=nc, support_r=1.0, chart_decoder=decoder,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )
    stress_fn, tangent_fn = solver.make_neo_hookean(MU, K_BULK)
    n_nodes = solver.n_nodes
    n_elems = solver.n_elements

    # Report mesh stats
    centroids = solver.elem_centroids_phys.detach().cpu().numpy()
    near_tip = np.abs(centroids[:, 1] - L / 2) < 2.0
    print(f"  Mesh: {n_nodes} nodes, {n_elems} tets")
    print(f"  Near-tip concentration: {near_tip.sum()}/{n_elems} = {near_tip.sum()/n_elems*100:.1f}%")

    # ── Topology monitor ──
    try:
        from atlas.topo.monitor import TopologyMonitor
        topo_monitor = TopologyMonitor(
            lifetime_threshold=0.05, bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1), relative_threshold=True, verbose=False,
        )
        has_topo = True
        print(f"  Topology: GUDHI enabled")
    except ImportError:
        has_topo = False
        print(f"  Topology: not available")

    # ── Adaptive controller ──
    controller = AdaptiveLoadController(
        delta_max=delta_max, delta_0=0.5,
        target_iters=5, min_dd=0.01, max_dd=2.0,
    )

    # ── Result storage ──
    results = {"delta": [], "d_delta": [], "newton_iters": [], "P_yy": [],
               "J_min": [], "J_max": [], "e_local_max": [], "e_nonlocal_max": []}
    nuc_detected_step = -1
    topo_events = []
    betti_history = []
    vtu_files = []

    # ── Restart ──
    start_step = 0
    u_prev = None
    if restart_from:
        ckpt = torch.load(restart_from, map_location="cpu")
        start_step = ckpt["step"] + 1
        u_prev = ckpt["u"].to("cpu", torch.float64) if ckpt.get("u") is not None else None
        controller.load_state_dict(ckpt["controller"])
        results = ckpt.get("results", results)
        nuc_detected_step = ckpt.get("nuc_detected_step", -1)
        betti_history = ckpt.get("betti_history", [])
        print(f"  Restarted from step {start_step}, delta={controller.delta:.3f}")

    nodes = solver.nodes_phys.detach()
    y = nodes[:, 1]
    f_ext = torch.zeros(n_nodes, 3, dtype=torch.float64)

    # ── Main loop ──
    print()
    print(f"  {'step':>5s} {'delta':>8s} {'d_delta':>8s} {'nwt':>4s} {'P_yy':>10s} "
          f"{'J_min':>8s} {'J_max':>8s} {'e_max':>10s} {'nuc':>4s} {'topo':>8s} {'dt':>7s}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*4} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*4} {'-'*8} {'-'*7}")

    for step in range(start_step, max_steps):
        if controller.finished:
            print(f"\n  [DONE] delta_max={delta_max} reached at step {step}")
            break

        delta = controller.delta
        t_step = time.time()

        # BCs: fix bottom (y < 1), pull top (y > L-1) in ±z (Mode III)
        bc_mask = torch.zeros(n_nodes, dtype=torch.bool)
        u_bc = torch.zeros(n_nodes, 3, dtype=torch.float64)
        bot = y < 1.0
        top = y > L - 1.0
        bc_mask[bot] = True
        bc_mask[top] = True
        u_bc[top, 2] = delta / 2
        u_bc[bot, 2] = -delta / 2

        # ── Solve ──
        u_sol = solver.solve_nonlinear(
            stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
            max_iter=20, tol=1e-7, use_fbar=False, u_init=u_prev,
        )

        # Extract Newton iteration count from the solver's last print
        # (The solver prints "[Newton] converged at iter N" or "WARNING: did not converge")
        # We parse this from the solve — but since we can't intercept stdout easily,
        # use a simpler heuristic: check if residual is small
        F = solver.compute_F(u_sol)
        f_int = solver.internal_forces(u_sol, stress_fn, use_fbar=False)
        R = (f_int - f_ext).reshape(-1)
        bc_dof = bc_mask.unsqueeze(1).expand(-1, 3).reshape(-1)
        res_norm = R[~bc_dof].norm().item()
        converged = res_norm < 1e-5
        # Estimate Newton iters from residual magnitude (rough proxy)
        newton_iters = 1 if res_norm < 1e-9 else (3 if res_norm < 1e-7 else (8 if res_norm < 1e-5 else 20))

        # ── Adaptive stepping ──
        new_delta = controller.advance(newton_iters, converged)

        if not converged:
            # Retry with smaller step — don't update u_prev
            dt_step = time.time() - t_step
            print(f"  {step:5d} {delta:8.3f} {controller.d_delta:8.4f} {'FAIL':>4s} "
                  f"{'':>10s} {'':>8s} {'':>8s} {'':>10s} {'':>4s} {'':>8s} {dt_step:7.1f}s")
            continue

        u_prev = u_sol.clone()

        # ── Post-process ──
        J = torch.det(F)
        P = stress_fn(F)
        P_yy = P[:, 1, 1].mean().item()
        e_loc = compute_local_equivalent_strain(F)
        e_nl = solve_nonlocal_strain(solver, e_loc, length_scale=0.5)
        e_max = e_loc.max().item()
        e_nl_max = e_nl.max().item()

        # ── Nucleation check ──
        nuc_str = ""
        if nuc_detected_step < 0:
            P_np = P.detach().numpy()
            for e in range(min(len(P_np), 500)):  # check first 500 elements for speed
                fdp = drucker_prager_F(torch.tensor(P_np[e]).unsqueeze(0), SIGMA_TS, SIGMA_HS).item()
                if fdp >= 0:
                    nuc_detected_step = step
                    nuc_str = "NUC!"
                    break

        # ── Topology monitoring ──
        topo_str = ""
        if has_topo and (step % topo_every == 0 or nuc_str):
            try:
                grid_n = 16
                grid_vals = np.zeros((grid_n, grid_n, grid_n))
                nodes_np = nodes.detach().cpu().numpy()
                e_nl_np = e_nl.detach().cpu().numpy()
                bb_min = np.array([-W/2-0.1, -0.1, -B/2-0.05])
                bb_max = np.array([W/2+0.1, L+0.1, B/2+0.05])
                bb_range = bb_max - bb_min
                for ni in range(n_nodes):
                    idx = ((nodes_np[ni] - bb_min) / bb_range * (grid_n - 1)).astype(int)
                    idx = np.clip(idx, 0, grid_n - 1)
                    grid_vals[idx[0], idx[1], idx[2]] = max(
                        grid_vals[idx[0], idx[1], idx[2]], e_nl_np[ni])
                active = grid_vals[grid_vals > 0]
                thr = active.mean() + 2 * active.std() if len(active) > 0 else 0.01
                damage_sdf = thr - grid_vals
                events = topo_monitor.update(damage_sdf, step)
                topo_events.extend(events)
                betti = topo_monitor.current_betti()
                betti_history.append({"step": step, "delta": delta, "betti": dict(betti)})
                topo_str = f"b={betti}"
            except Exception:
                topo_str = "err"

        # ── Store ──
        results["delta"].append(float(delta))
        results["d_delta"].append(float(controller.d_delta))
        results["newton_iters"].append(newton_iters)
        results["P_yy"].append(float(P_yy))
        results["J_min"].append(float(J.min().item()))
        results["J_max"].append(float(J.max().item()))
        results["e_local_max"].append(float(e_max))
        results["e_nonlocal_max"].append(float(e_nl_max))

        # ── VTU ──
        if step % vtu_every == 0 or nuc_str or controller.finished:
            vtu_path = os.path.join(vtu_dir, f"trousers_{step:05d}.vtu")
            write_vtu(vtu_path, nodes, solver.elements, u_sol, P, e_loc, e_nl, J, step, delta)
            vtu_files.append((step, delta, vtu_path))

        # ── Checkpoint ──
        if step % checkpoint_every == 0 and step > start_step:
            torch.save({
                "step": step, "delta": delta, "u": u_sol.cpu(),
                "controller": controller.state_dict(),
                "results": results, "nuc_detected_step": nuc_detected_step,
                "betti_history": betti_history,
            }, os.path.join(output_dir, f"checkpoint_{step:05d}.pt"))

        dt_step = time.time() - t_step

        # ── Print progress ──
        if step % max(1, len(results["delta"]) // 50 + 1) == 0 or nuc_str or controller.finished:
            print(f"  {step:5d} {delta:8.3f} {controller.d_delta:8.4f} {newton_iters:4d} "
                  f"{P_yy:10.6f} {J.min().item():8.4f} {J.max().item():8.4f} "
                  f"{e_max:10.4e} {nuc_str:>4s} {topo_str:>8s} {dt_step:7.1f}s")
            sys.stdout.flush()

    wall_time = time.time() - t_global

    # ── Final output ──
    print()
    print(f"  Wall time: {wall_time:.0f}s ({wall_time/3600:.1f} hours)")
    print(f"  Total steps: {len(results['delta'])}")
    print(f"  Nucleation: step={nuc_detected_step}")
    print(f"  Topology events: {len(topo_events)}")
    for bh in betti_history[-3:]:
        print(f"    step {bh['step']}: betti={bh['betti']}")
    print(f"  Adaptive: d_delta range [{min(r for r in results['d_delta']):.4f}, "
          f"{max(r for r in results['d_delta']):.4f}]")

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    final = {
        "nc": nc, "n_nodes": n_nodes, "n_elements": n_elems,
        "grade_power": grade_power, "enrichment": use_enrichment,
        "wall_time": wall_time, "nuc_detected_step": nuc_detected_step,
        "topo_events": [{"step": e.load_step, "dim": e.dimension,
                         "type": e.event_type, "lifetime": e.lifetime}
                        for e in topo_events],
        "betti_history": betti_history,
        **results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(final, f, indent=2, default=convert)

    write_pvd(os.path.join(output_dir, "trousers.pvd"), vtu_files)

    # Final checkpoint
    torch.save({
        "step": len(results["delta"]) - 1, "delta": controller.delta,
        "u": u_prev.cpu() if u_prev is not None else None,
        "controller": controller.state_dict(),
        "results": results, "nuc_detected_step": nuc_detected_step,
        "betti_history": betti_history,
    }, os.path.join(output_dir, "checkpoint_final.pt"))

    print(f"\n  Results: {output_dir}/results.json")
    print(f"  VTU: {len(vtu_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=18)
    parser.add_argument("--delta", type=float, default=65.0)
    parser.add_argument("--grade-power", type=float, default=3.0)
    parser.add_argument("--no-enrichment", action="store_true")
    parser.add_argument("--output", default="runs/trousers_graded")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--restart", type=str, default=None)
    args = parser.parse_args()

    run(nc=args.nc, delta_max=args.delta, grade_power=args.grade_power,
        use_enrichment=not args.no_enrichment,
        output_dir=args.output, max_steps=args.max_steps,
        restart_from=args.restart)
