#!/usr/bin/env python3
"""
FEM Poisson solve on a rabbit atlas with multiplicative alternating Schwarz.

Uses the same frozen atlas (decoders, masks, chart geometry) as the PINN version,
but replaces each chart-local neural network with a P1 FEM discretization.

Inputs:
- atlas build output (.npz + meta.json)
- atlas training checkpoint (chart decoders + mask nets)
- SDF checkpoint (for mesh filtering and boundary classification)

Outputs:
- solution fields on canonical rabbit points
- metrics JSON + convergence history
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from chart_fem_solver import ChartFEMSolver

# Re-use infrastructure from the PINN Schwarz script
from run_poisson_rabbit_atlas_schwarz import (
    ChartDecoder,
    MaskNet,
    build_run_stem,
    chart_map_and_jacobian,
    choose_color_groups,
    forcing_f,
    load_atlas_models,
    local_coords,
    manufactured_grad_u,
    manufactured_u,
    metric_l2,
    normalize_rows_tensor,
    plot_history,
    resolve_device,
    resolve_dtype,
    set_seed,
    stabilized_jacobian_ops,
)

# SDF network from training script
from train_sdf_rabbit import SDFNet

torch.set_default_dtype(torch.float64)


# -------------------------------------------------------------------------
# SDF oracle wrapper
# -------------------------------------------------------------------------
class SDFOracle:
    """Wraps a trained SDFNet for use by ChartFEMSolver."""

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        kwargs = ckpt["model_kwargs"]
        self.model = SDFNet(width=kwargs["width"], depth=kwargs["depth"]).to(
            device=device, dtype=dtype
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.center = torch.tensor(ckpt["center"], device=device, dtype=dtype)
        self.scale = torch.tensor(float(ckpt["scale"]), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.center.unsqueeze(0)) / self.scale
        phi_norm = self.model(x_norm)
        return phi_norm * self.scale


# -------------------------------------------------------------------------
# NumPy versions of manufactured solution / forcing
# -------------------------------------------------------------------------
def manufactured_u_np(x: np.ndarray) -> np.ndarray:
    pi = math.pi
    return np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * np.sin(pi * x[:, 2])


def forcing_f_np(x: np.ndarray) -> np.ndarray:
    pi = math.pi
    return 3.0 * pi**2 * manufactured_u_np(x)


# -------------------------------------------------------------------------
# Schwarz boundary condition helpers
# -------------------------------------------------------------------------
def compute_phys_bc(
    solver: ChartFEMSolver,
) -> Dict[int, float]:
    """Physical boundary nodes get u = g(x) = manufactured_u(x)."""
    if len(solver.phys_bc_nodes) == 0:
        return {}
    xi = solver.nodes[solver.phys_bc_nodes]
    x_phys = solver._decode_points(xi)
    u_exact = manufactured_u_np(x_phys)
    return {int(idx): float(val) for idx, val in zip(solver.phys_bc_nodes, u_exact)}


def compute_schwarz_bc(
    chart_i: int,
    fem_solvers: List[ChartFEMSolver],
    decoders: List[ChartDecoder],
    masks: List[MaskNet],
    seeds_t: torch.Tensor,
    t1_t: torch.Tensor,
    t2_t: torch.Tensor,
    nvec_t: torch.Tensor,
    support_r_t: torch.Tensor,
    neighbors: List[List[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, float]:
    """For each artificial boundary node of chart i, interpolate neighbor solution values.

    Uses PoU-weighted (mask-logit softmax) blend of neighbor FEM solutions.
    """
    solver_i = fem_solvers[chart_i]
    if len(solver_i.art_bc_nodes) == 0:
        return {}

    art_bc = {}
    xi_art = solver_i.nodes[solver_i.art_bc_nodes]  # (N_art, 3)
    x_phys = solver_i._decode_points(xi_art)  # (N_art, 3)

    nbrs = neighbors[chart_i]
    if len(nbrs) == 0:
        # No neighbors: use manufactured solution as fallback
        u_exact = manufactured_u_np(x_phys)
        return {int(idx): float(val) for idx, val in zip(solver_i.art_bc_nodes, u_exact)}

    # Evaluate mask logits and neighbor solutions for all art BC nodes at once
    x_phys_t = torch.tensor(x_phys, device=device, dtype=dtype)

    # Collect logits and values from all charts that cover these points
    all_logits = []
    all_vals = []
    all_chart_ids = []

    for j in nbrs:
        xi_j_linear = local_coords(x_phys_t, seeds_t[j], t1_t[j], t2_t[j], nvec_t[j])
        xi_j_lin_np = xi_j_linear.cpu().numpy()

        # Check if points are within support radius (use linear coords for fast check)
        r_j = float(support_r_t[j].item())
        in_support = np.all(np.abs(xi_j_lin_np) <= 1.25 * r_j, axis=1)

        if not np.any(in_support):
            continue

        # Get mask logits (masks work in linear local_coords space)
        with torch.no_grad():
            logit_j = masks[j](xi_j_linear, chart_scale=support_r_t[j]).cpu().numpy()

        # Newton-invert the decoder for accurate FEM evaluation
        # Only invert for points in support to save time
        xi_j_np = xi_j_lin_np.copy()
        if np.any(in_support):
            with torch.enable_grad():
                xi_j_star = invert_decoder(
                    decoder=decoders[j],
                    x_target=x_phys_t[in_support],
                    seed=seeds_t[j],
                    t1=t1_t[j],
                    t2=t2_t[j],
                    n_vec=nvec_t[j],
                    chart_scale=support_r_t[j],
                    xi_init=xi_j_linear[in_support],
                    max_iter=10,
                    tol=1e-8,
                )
            xi_j_np[in_support] = xi_j_star.cpu().numpy()

        # Get FEM solution values (only for points in support AND in mesh)
        u_j = np.full(len(xi_j_np), np.nan)
        if fem_solvers[j].u is not None:
            r_j_mesh = fem_solvers[j].r
            in_mesh = in_support & np.all(np.abs(xi_j_np) <= r_j_mesh, axis=1)
            if np.any(in_mesh):
                u_j[in_mesh] = fem_solvers[j].evaluate_at(xi_j_np[in_mesh])

        all_logits.append(logit_j)
        all_vals.append(u_j)
        all_chart_ids.append(j)

    if len(all_logits) == 0:
        # No neighbors cover these points — use manufactured solution
        u_exact = manufactured_u_np(x_phys)
        return {int(idx): float(val) for idx, val in zip(solver_i.art_bc_nodes, u_exact)}

    logits_arr = np.array(all_logits)  # (n_nbrs, N_art)
    vals_arr = np.array(all_vals)  # (n_nbrs, N_art)

    # For each art BC node, compute PoU-weighted blend
    for k, node_idx in enumerate(solver_i.art_bc_nodes):
        valid = ~np.isnan(vals_arr[:, k])
        if not np.any(valid):
            art_bc[int(node_idx)] = float(manufactured_u_np(x_phys[k:k+1])[0])
            continue

        logits_k = logits_arr[valid, k]
        vals_k = vals_arr[valid, k]

        # Softmax weights
        logits_k = logits_k - logits_k.max()  # numerical stability
        weights = np.exp(logits_k)
        weights /= weights.sum()

        art_bc[int(node_idx)] = float(np.dot(weights, vals_k))

    return art_bc


# -------------------------------------------------------------------------
# Newton inversion of the decoder map
# -------------------------------------------------------------------------
def invert_decoder(
    decoder: ChartDecoder,
    x_target: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n_vec: torch.Tensor,
    chart_scale: torch.Tensor,
    xi_init: Optional[torch.Tensor] = None,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Find ξ* such that φ(ξ*) = x_target using Newton iteration.

    Args:
        decoder: chart decoder φ: ξ → x
        x_target: (N, 3) target physical coordinates
        xi_init: (N, 3) initial guess (default: linear local_coords)
        max_iter: max Newton iterations
        tol: convergence tolerance on |φ(ξ) - x|

    Returns:
        xi_star: (N, 3) inverted coordinates
    """
    if xi_init is None:
        xi_init = local_coords(x_target, seed, t1, t2, n_vec)

    xi = xi_init.clone().detach()
    N = xi.shape[0]

    for it in range(max_iter):
        xi_var = xi.clone().detach().requires_grad_(True)
        x_pred = decoder(xi_var, seed=seed, t1=t1, t2=t2, n=n_vec, chart_scale=chart_scale)

        residual = x_pred - x_target
        res_norm = torch.linalg.norm(residual, dim=1)

        if res_norm.max().item() < tol:
            break

        # Compute Jacobian J = ∂φ/∂ξ
        grads = []
        for i in range(3):
            gi = torch.autograd.grad(
                x_pred[:, i], xi_var,
                grad_outputs=torch.ones_like(x_pred[:, i]),
                create_graph=False, retain_graph=True,
            )[0]
            grads.append(gi)
        J = torch.stack(grads, dim=1)  # (N, 3, 3)

        # Newton update: ξ_new = ξ - J^{-1} (φ(ξ) - x_target)
        try:
            delta = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
        except Exception:
            # Fallback: pseudo-inverse
            delta = torch.bmm(torch.linalg.pinv(J), residual.unsqueeze(-1)).squeeze(-1)

        xi = xi - delta

    return xi.detach()


# -------------------------------------------------------------------------
# Global evaluation
# -------------------------------------------------------------------------
def evaluate_global_fem(
    fem_solvers: List[ChartFEMSolver],
    masks: List[MaskNet],
    decoders: List[ChartDecoder],
    eval_points: torch.Tensor,
    seeds_t: torch.Tensor,
    t1_t: torch.Tensor,
    t2_t: torch.Tensor,
    nvec_t: torch.Tensor,
    support_r_t: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute blended FEM field u_h(x) on evaluation points.

    Uses Newton inversion of the decoder map to find the correct ξ* for each
    evaluation point x, so that u_FEM(ξ*) = u_FEM(φ^{-1}(x)).

    Returns:
        u_pred: (N,) predicted solution
        u_true: (N,) exact solution
    """
    n_charts = len(fem_solvers)
    n_points = eval_points.shape[0]
    x_np = eval_points.cpu().numpy()

    # Collect logits and chart solutions for all points
    logits_all = np.full((n_charts, n_points), -1e10)  # default: very low logit
    vals_all = np.zeros((n_charts, n_points))

    # valid_all[i, k] = True if chart i's FEM mesh contains eval point k
    valid_all = np.zeros((n_charts, n_points), dtype=bool)

    for i in range(n_charts):
        # Linear projection for logits (masks work in this space)
        with torch.no_grad():
            xi_linear = local_coords(eval_points, seeds_t[i], t1_t[i], t2_t[i], nvec_t[i])
            logits_all[i] = masks[i](xi_linear, chart_scale=support_r_t[i]).cpu().numpy()

        if fem_solvers[i].u is not None and fem_solvers[i].n_nodes > 0:
            # Newton-invert the decoder to find ξ* = φ^{-1}(x)
            with torch.enable_grad():
                xi_star = invert_decoder(
                    decoder=decoders[i],
                    x_target=eval_points,
                    seed=seeds_t[i],
                    t1=t1_t[i],
                    t2=t2_t[i],
                    n_vec=nvec_t[i],
                    chart_scale=support_r_t[i],
                    xi_init=xi_linear,
                    max_iter=10,
                    tol=1e-8,
                )
            xi_star_np = xi_star.cpu().numpy()

            # Only evaluate at points within the FEM mesh domain
            r_i = fem_solvers[i].r
            in_mesh = np.all(np.abs(xi_star_np) <= r_i, axis=1)
            valid_all[i] = in_mesh

            if np.any(in_mesh):
                vals_all[i, in_mesh] = fem_solvers[i].evaluate_at(xi_star_np[in_mesh])

    # Softmax blend — only using valid chart contributions
    logits_all = logits_all.T  # (N_points, n_charts)
    vals_all = vals_all.T  # (N_points, n_charts)
    valid_all = valid_all.T  # (N_points, n_charts)

    # Mask out invalid charts by setting their logits to -inf
    masked_logits = np.where(valid_all, logits_all, -1e30)
    logits_shifted = masked_logits - masked_logits.max(axis=1, keepdims=True)
    weights = np.exp(logits_shifted)
    weight_sum = weights.sum(axis=1, keepdims=True)

    # For points with no valid charts, fall back to unmasked logits
    no_valid = weight_sum.ravel() < 1e-20
    if np.any(no_valid):
        logits_shifted_full = logits_all[no_valid] - logits_all[no_valid].max(axis=1, keepdims=True)
        weights[no_valid] = np.exp(logits_shifted_full)
        weight_sum[no_valid] = weights[no_valid].sum(axis=1, keepdims=True)

    weights /= np.maximum(weight_sum, 1e-20)
    u_pred = np.sum(weights * vals_all, axis=1)
    u_true = manufactured_u_np(x_np)

    return u_pred, u_true


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FEM Poisson on rabbit atlas with Schwarz")
    parser.add_argument("--atlas_data", required=True, help="Atlas .npz file")
    parser.add_argument("--atlas_checkpoint", required=True, help="Atlas decoder/mask checkpoint")
    parser.add_argument("--atlas_meta", default=None, help="Atlas meta.json with color groups")
    parser.add_argument("--sdf_checkpoint", default=None, help="SDF network checkpoint (optional; enables mesh filtering)")
    parser.add_argument("--n_cells", type=int, default=16, help="Hex cells per axis per chart")
    parser.add_argument("--max_schwarz_iters", type=int, default=50, help="Max Schwarz iterations")
    parser.add_argument("--target_rel_l2", type=float, default=0.01, help="Target relative L2 error")
    parser.add_argument("--run_tag", default="fem", help="Tag for output filenames")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--dtype", default="float64", help="Dtype (float32/float64/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sdf_threshold", type=float, default=-0.005, help="SDF threshold for mesh filtering")
    parser.add_argument("--allow_failed_gate", action="store_true", help="Allow failed atlas gate")
    parser.add_argument(
        "--n_cells_sweep", nargs="*", type=int, default=None,
        help="If provided, run h-refinement sweep over these n_cells values"
    )
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--eval_subsample", type=int, default=5000, help="Number of eval points per Schwarz iter (0=all)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)

    print(f"Device={device.type} dtype={dtype}")
    print(f"n_cells={args.n_cells}")

    # 1. Load atlas
    atlas_np = np.load(args.atlas_data)
    points = torch.tensor(atlas_np["points"], device=device, dtype=dtype)
    seeds_t = torch.tensor(atlas_np["seed_points"], device=device, dtype=dtype)
    t1_t = torch.tensor(atlas_np["frame_t1"], device=device, dtype=dtype)
    t2_t = torch.tensor(atlas_np["frame_t2"], device=device, dtype=dtype)
    nvec_t = torch.tensor(atlas_np["frame_n"], device=device, dtype=dtype)
    membership = atlas_np["membership"]
    support_r_t = torch.tensor(atlas_np["support_radii"], device=device, dtype=dtype)

    n_points, n_charts = membership.shape
    print(f"Atlas: {n_charts} charts, {n_points} points")

    decoders, masks, atlas_ckpt = load_atlas_models(
        atlas_checkpoint=args.atlas_checkpoint,
        device=device,
        dtype=dtype,
    )

    gate = atlas_ckpt.get("gate")
    if isinstance(gate, dict) and (not args.allow_failed_gate) and not bool(gate.get("passed", False)):
        raise RuntimeError(
            f"Atlas gate check failed. Checkpoint: {args.atlas_checkpoint}"
        )

    # 2. Load SDF (optional)
    sdf_oracle = None
    if args.sdf_checkpoint is not None:
        sdf_oracle = SDFOracle(args.sdf_checkpoint, device=device, dtype=dtype)

    # 3. Color groups
    color_groups = choose_color_groups(
        meta_json=args.atlas_meta,
        n_charts=n_charts,
        membership_np=membership,
    )
    print(f"Color groups: {[len(g) for g in color_groups]}")

    # 4. Neighbor lists
    mem_bool = membership.astype(bool)
    neighbors: List[List[int]] = [[] for _ in range(n_charts)]
    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            shared = np.sum(mem_bool[:, i] & mem_bool[:, j])
            if shared > 0:
                neighbors[i].append(j)
                neighbors[j].append(i)

    # Determine which n_cells values to run
    if args.n_cells_sweep is not None and len(args.n_cells_sweep) > 0:
        n_cells_list = sorted(args.n_cells_sweep)
    else:
        n_cells_list = [args.n_cells]

    sweep_results = []

    for n_cells in n_cells_list:
        print(f"\n{'='*60}")
        print(f"Running FEM Schwarz with n_cells = {n_cells}")
        print(f"{'='*60}")

        result = run_fem_schwarz(
            n_cells=n_cells,
            max_schwarz_iters=args.max_schwarz_iters,
            target_rel_l2=args.target_rel_l2,
            sdf_oracle=sdf_oracle,
            sdf_threshold=args.sdf_threshold,
            decoders=decoders,
            masks=masks,
            points=points,
            seeds_t=seeds_t,
            t1_t=t1_t,
            t2_t=t2_t,
            nvec_t=nvec_t,
            support_r_t=support_r_t,
            n_charts=n_charts,
            membership=membership,
            color_groups=color_groups,
            neighbors=neighbors,
            device=device,
            dtype=dtype,
            run_tag=f"{args.run_tag}_n{n_cells}",
            outdir=args.outdir,
            eval_subsample=args.eval_subsample,
        )
        sweep_results.append(result)

    # Print summary table
    if len(sweep_results) > 1:
        print(f"\n{'='*80}")
        print("h-REFINEMENT SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"{'n_cells':>8} {'h':>10} {'DOFs':>10} {'rel-L2':>12} {'max-err':>12} {'iters':>6} {'time(s)':>10}")
        print("-" * 80)
        for r in sweep_results:
            print(
                f"{r['n_cells']:>8d} "
                f"{r['h']:>10.4f} "
                f"{r['total_dofs']:>10d} "
                f"{r['rel_l2']:>12.4e} "
                f"{r['max_error']:>12.4e} "
                f"{r['n_iters']:>6d} "
                f"{r['time_s']:>10.1f}"
            )

        # Save sweep summary
        sweep_path = os.path.join(args.outdir, f"rabbit_poisson_schwarz_{args.run_tag}_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep summary saved to: {sweep_path}")


def run_fem_schwarz(
    n_cells: int,
    max_schwarz_iters: int,
    target_rel_l2: float,
    sdf_oracle: SDFOracle,
    sdf_threshold: float,
    decoders: List[ChartDecoder],
    masks: List[MaskNet],
    points: torch.Tensor,
    seeds_t: torch.Tensor,
    t1_t: torch.Tensor,
    t2_t: torch.Tensor,
    nvec_t: torch.Tensor,
    support_r_t: torch.Tensor,
    n_charts: int,
    membership: np.ndarray,
    color_groups: List[List[int]],
    neighbors: List[List[int]],
    device: torch.device,
    dtype: torch.dtype,
    run_tag: str = "fem",
    outdir: str = ".",
    eval_subsample: int = 5000,
) -> Dict:
    """Run full FEM Schwarz iteration for one mesh resolution."""

    start_time = time.time()

    # 1. Build FEM solvers for each chart
    print(f"\nBuilding {n_charts} chart FEM solvers (n_cells={n_cells})...")
    fem_solvers = []
    for i in range(n_charts):
        solver = ChartFEMSolver(
            chart_id=i,
            decoder=decoders[i],
            seed=seeds_t[i],
            t1=t1_t[i],
            t2=t2_t[i],
            n_vec=nvec_t[i],
            support_r=support_r_t[i],
            n_cells=n_cells,
            sdf_oracle=sdf_oracle,
            sdf_threshold=sdf_threshold,
            device=device,
            dtype=dtype,
        )
        fem_solvers.append(solver)

    total_dofs = sum(s.n_dofs for s in fem_solvers)
    total_elements = sum(s.n_elements for s in fem_solvers)
    print(f"Total DOFs: {total_dofs}, Total elements: {total_elements}")

    # 2. Pre-compute diffusion tensors
    print("Computing diffusion tensors...")
    for solver in fem_solvers:
        solver.compute_diffusion_tensors()

    # 3. Assemble stiffness matrices (constant — don't change during Schwarz)
    print("Assembling stiffness matrices...")
    for solver in fem_solvers:
        solver.assemble(forcing_fn=forcing_f_np)

    # 4. Initialize solution: use manufactured solution at all nodes
    print("Initializing solutions with manufactured solution...")
    for solver in fem_solvers:
        if solver.n_nodes > 0:
            x_phys = solver._decode_points(solver.nodes)
            solver.u = manufactured_u_np(x_phys) * 0.0  # start from zero
            # Better init: use manufactured solution
            solver.u = manufactured_u_np(x_phys)

    # 5. Schwarz iteration
    h = 2.0 * float(support_r_t[0].item()) / n_cells
    history = {
        "rel_l2_eval": [],
        "max_error": [],
        "schwarz_iter": [],
    }

    # Subsampled evaluation points for faster per-iteration monitoring
    n_eval_points = points.shape[0]
    if eval_subsample > 0 and eval_subsample < n_eval_points:
        eval_idx = np.random.choice(n_eval_points, eval_subsample, replace=False)
        eval_points_iter = points[eval_idx]
    else:
        eval_points_iter = points

    print(f"\nStarting Schwarz iterations (max={max_schwarz_iters}, eval_points={len(eval_points_iter)})...")
    for iteration in range(1, max_schwarz_iters + 1):
        for color_group in color_groups:
            for chart_i in color_group:
                if fem_solvers[chart_i].n_nodes == 0:
                    continue

                # Physical BC
                phys_bc = compute_phys_bc(fem_solvers[chart_i])

                # Schwarz BC from neighbors
                art_bc = compute_schwarz_bc(
                    chart_i=chart_i,
                    fem_solvers=fem_solvers,
                    decoders=decoders,
                    masks=masks,
                    seeds_t=seeds_t,
                    t1_t=t1_t,
                    t2_t=t2_t,
                    nvec_t=nvec_t,
                    support_r_t=support_r_t,
                    neighbors=neighbors,
                    device=device,
                    dtype=dtype,
                )

                # Solve
                fem_solvers[chart_i].solve(phys_bc, art_bc)

        # Evaluate (on subsampled points for speed)
        u_pred, u_true = evaluate_global_fem(
            fem_solvers=fem_solvers,
            masks=masks,
            decoders=decoders,
            eval_points=eval_points_iter,
            seeds_t=seeds_t,
            t1_t=t1_t,
            t2_t=t2_t,
            nvec_t=nvec_t,
            support_r_t=support_r_t,
            device=device,
            dtype=dtype,
        )

        stats = metric_l2(u_pred, u_true)
        rel_l2 = stats["relative_l2_error"]
        max_err = stats["max_error"]

        history["rel_l2_eval"].append(rel_l2)
        history["max_error"].append(max_err)
        history["schwarz_iter"].append(iteration)

        if iteration % 1 == 0:
            elapsed = time.time() - start_time
            print(
                f"  [Schwarz] iter={iteration}/{max_schwarz_iters} "
                f"rel-L²={rel_l2:.4e} max-err={max_err:.4e} "
                f"t={elapsed:.1f}s"
            )

        # Check convergence
        if iteration >= 3:
            # Check if rel_l2 has stabilized (change < 1% of current value)
            prev = history["rel_l2_eval"][-2]
            change = abs(rel_l2 - prev) / max(rel_l2, 1e-14)
            if change < 0.01 and iteration >= 5:
                print(f"  Schwarz converged (rel change {change:.2e}) at iter {iteration}")
                break

        if rel_l2 < target_rel_l2:
            print(f"  Target rel-L² = {target_rel_l2:.2e} reached at iter {iteration}")
            break

    elapsed = time.time() - start_time

    # Final evaluation
    u_pred, u_true = evaluate_global_fem(
        fem_solvers=fem_solvers,
        masks=masks,
        decoders=decoders,
        eval_points=points,
        seeds_t=seeds_t,
        t1_t=t1_t,
        t2_t=t2_t,
        nvec_t=nvec_t,
        support_r_t=support_r_t,
        device=device,
        dtype=dtype,
    )
    final_stats = metric_l2(u_pred, u_true)

    print(f"\n  FINAL (n_cells={n_cells}): rel-L²={final_stats['relative_l2_error']:.4e}, "
          f"max-err={final_stats['max_error']:.4e}, "
          f"time={elapsed:.1f}s")

    # Save outputs
    os.makedirs(outdir, exist_ok=True)
    run_stem = f"rabbit_poisson_schwarz_{run_tag}"

    # Solution NPZ
    sol_path = os.path.join(outdir, f"{run_stem}_solution.npz")
    x_np = points.cpu().numpy()
    np.savez(
        sol_path,
        points=x_np,
        u_pred=u_pred,
        u_true=u_true,
        u_error=u_pred - u_true,
        u_error_mag=np.abs(u_pred - u_true),
    )
    print(f"  Solution saved to: {sol_path}")

    # Metrics JSON
    metrics = {
        "n_cells": n_cells,
        "h": h,
        "total_dofs": total_dofs,
        "total_elements": total_elements,
        "n_charts": n_charts,
        "n_schwarz_iters": len(history["rel_l2_eval"]),
        "solver_type": "fem_p1",
        "runtime_seconds": elapsed,
        **final_stats,
    }
    metrics_path = os.path.join(outdir, f"{run_stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # History JSON
    history_path = os.path.join(outdir, f"{run_stem}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Convergence plot
    if len(history["rel_l2_eval"]) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        iters = np.array(history["schwarz_iter"])
        ax.semilogy(iters, history["rel_l2_eval"], "o-", label="rel-L²")
        ax.semilogy(iters, history["max_error"], "s-", label="max error")
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel("Error")
        ax.set_title(f"FEM Schwarz convergence (n_cells={n_cells})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(outdir, f"{run_stem}_convergence.png")
        plt.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"  Plot saved to: {fig_path}")

    return {
        "n_cells": n_cells,
        "h": h,
        "total_dofs": total_dofs,
        "total_elements": total_elements,
        "rel_l2": final_stats["relative_l2_error"],
        "l2_error": final_stats["l2_error"],
        "max_error": final_stats["max_error"],
        "n_iters": len(history["rel_l2_eval"]),
        "time_s": elapsed,
    }


if __name__ == "__main__":
    main()
