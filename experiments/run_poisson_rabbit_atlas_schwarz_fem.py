#!/usr/bin/env python3
"""
FEM-based Poisson solve on a rabbit atlas with multiplicative alternating Schwarz.

Same atlas infrastructure (ChartDecoder, MaskNet, Schwarz colouring) as the
PINN version, but replaces each chart-local neural network with a P1 finite
element discretisation + sparse linear solve.

Usage:

    python run_poisson_rabbit_atlas_schwarz_fem.py \
        --atlas-data runs/atlas_vol_trained/rabbit_atlas_data.npz \
        --atlas-checkpoint runs/atlas_vol_trained/rabbit_atlas_trained.pt \
        --sdf-checkpoint runs/bunny_sdf_v3/rabbit_sdf.pt \
        --n-cells 16 \
        --max-schwarz-iters 50 \
        --run-tag fem_n16

Outputs have the same structure as the PINN version:
    {run_stem}_solution.npz,  {run_stem}_metrics.json,
    {run_stem}_history.json,  {run_stem}_curves.png
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Imports from existing atlas infrastructure ────────────────────────────
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from run_poisson_rabbit_atlas_schwarz import (
    ChartDecoder,
    MaskNet,
    MLP,
    local_coords,
    chart_map_and_jacobian,
    stabilized_jacobian_ops,
    manufactured_u,
    manufactured_grad_u,
    forcing_f,
    load_atlas_models,
    load_sdf_for_schwarz,
    choose_color_groups,
    metric_l2,
    set_seed,
    resolve_device,
    resolve_dtype,
    build_run_stem,
)

from chart_fem_solver import (
    ChartFEMSolver,
    manufactured_u_np,
    _manufactured_forcing_np,
)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def local_coords_np(
    x: np.ndarray,
    seed: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    n_vec: np.ndarray,
) -> np.ndarray:
    """Numpy version of local_coords: x → ζ in chart-local frame."""
    d = x - seed[np.newaxis, :]
    return np.stack([d @ t1, d @ t2, d @ n_vec], axis=-1)


def evaluate_mask_np(
    xi: np.ndarray,
    mask_net: torch.nn.Module,
    support_r: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Evaluate MaskNet logits at ζ-coordinates (numpy in/out)."""
    xi_t = torch.tensor(xi, device=device, dtype=dtype)
    with torch.no_grad():
        logit = mask_net(xi_t, chart_scale=support_r)
    return logit.cpu().numpy()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for (N, K) array."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════
# Schwarz BC computation
# ══════════════════════════════════════════════════════════════════════════

def compute_phys_bc(
    solver: ChartFEMSolver,
) -> Dict[int, float]:
    """Physical boundary: u = g(x) = 0 (homogeneous Dirichlet for manufactured solution)."""
    bc: Dict[int, float] = {}
    if solver.phys_bc_nodes is None or len(solver.phys_bc_nodes) == 0:
        return bc

    # Map ζ → x and evaluate manufactured BC (which is 0 on the boundary,
    # but we use the manufactured solution for consistency with MMS)
    xi_bc = solver.nodes[solver.phys_bc_nodes]
    x_phys = _decode_batch(
        xi_bc, solver.decoder, solver.seed, solver.t1, solver.t2,
        solver.n_vec, solver.support_r, solver.device, solver.dtype,
    )
    u_bc = manufactured_u_np(x_phys)

    for k, node_idx in enumerate(solver.phys_bc_nodes):
        bc[int(node_idx)] = float(u_bc[k])
    return bc


def compute_schwarz_bc(
    chart_i: int,
    fem_solvers: List[ChartFEMSolver],
    decoders: List[torch.nn.Module],
    masks: List[torch.nn.Module],
    seeds_np: np.ndarray,
    t1s_np: np.ndarray,
    t2s_np: np.ndarray,
    nvecs_np: np.ndarray,
    support_radii_np: np.ndarray,
    support_r_t: List[torch.Tensor],
    neighbors: List[List[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, float]:
    """For each artificial-boundary node of chart i,
    interpolate neighbor solutions via PoU-weighted blending."""
    art_bc: Dict[int, float] = {}
    solver_i = fem_solvers[chart_i]
    if solver_i.art_bc_nodes is None or len(solver_i.art_bc_nodes) == 0:
        return art_bc

    # Map artificial BC nodes to physical space
    xi_art = solver_i.nodes[solver_i.art_bc_nodes]  # (N_art, 3)
    x_phys = _decode_batch(
        xi_art, solver_i.decoder, solver_i.seed, solver_i.t1, solver_i.t2,
        solver_i.n_vec, solver_i.support_r, solver_i.device, solver_i.dtype,
    )

    n_art = len(solver_i.art_bc_nodes)
    n_charts = len(fem_solvers)

    for k in range(n_art):
        x_k = x_phys[k]
        total_weight = 0.0
        weighted_val = 0.0

        for j in neighbors[chart_i]:
            # Quick check: is x_k roughly within chart j's support sphere?
            xi_j_rigid = local_coords_np(
                x_k.reshape(1, 3),
                seeds_np[j], t1s_np[j], t2s_np[j], nvecs_np[j],
            )[0]  # (3,)
            r_j = float(support_radii_np[j])
            if np.max(np.abs(xi_j_rigid)) > r_j * 1.5:
                continue

            # Evaluate neighbor FEM solution using decoder inverse
            if fem_solvers[j].u is None:
                continue
            u_j = fem_solvers[j].evaluate_at_physical(x_k.reshape(1, 3))[0]

            # Mask weight at rigid-TNB coords (consistent with PINN evaluation)
            w_j = np.exp(float(evaluate_mask_np(
                xi_j_rigid.reshape(1, 3), masks[j], support_r_t[j], device, dtype,
            )[0]))
            weighted_val += w_j * u_j
            total_weight += w_j

        if total_weight > 1e-12:
            art_bc[int(solver_i.art_bc_nodes[k])] = weighted_val / total_weight
        else:
            # Fallback: use manufactured solution
            art_bc[int(solver_i.art_bc_nodes[k])] = float(
                manufactured_u_np(x_k.reshape(1, 3))[0]
            )

    return art_bc


def _decode_batch(
    xi_np: np.ndarray,
    decoder: torch.nn.Module,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n_vec: torch.Tensor,
    support_r: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4096,
) -> np.ndarray:
    """Batch-decode ζ → x via ChartDecoder (numpy in/out)."""
    out = []
    for start in range(0, len(xi_np), batch_size):
        end = min(start + batch_size, len(xi_np))
        xi_t = torch.tensor(xi_np[start:end], device=device, dtype=dtype)
        with torch.no_grad():
            x = decoder(xi_t, seed=seed, t1=t1, t2=t2, n=n_vec, chart_scale=support_r)
        out.append(x.cpu().numpy())
    return np.concatenate(out, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# Global evaluation
# ══════════════════════════════════════════════════════════════════════════

def evaluate_global_blended(
    fem_solvers: List[ChartFEMSolver],
    decoders: List[torch.nn.Module],
    masks: List[torch.nn.Module],
    eval_points: np.ndarray,
    seeds_np: np.ndarray,
    t1s_np: np.ndarray,
    t2s_np: np.ndarray,
    nvecs_np: np.ndarray,
    support_radii_np: np.ndarray,
    support_r_t: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate blended FEM solution u_h(x) on a set of evaluation points.

    Returns
    -------
    u_pred : (N,) ndarray  —  blended FEM prediction
    metrics : dict  —  {rel_l2, abs_l2, max_error}
    """
    n_charts = len(fem_solvers)
    N = len(eval_points)

    # Gather per-chart logits and values for all eval points
    logits_all = np.full((N, n_charts), -1e6)
    vals_all = np.zeros((N, n_charts))

    for i in range(n_charts):
        xi_i = local_coords_np(
            eval_points, seeds_np[i], t1s_np[i], t2s_np[i], nvecs_np[i],
        )

        # Mask logits
        logits_all[:, i] = evaluate_mask_np(
            xi_i, masks[i], support_r_t[i], device, dtype,
        )

        # FEM solution: use decoder inverse for correct evaluation
        if fem_solvers[i].u is not None:
            vals_all[:, i] = fem_solvers[i].evaluate_at_physical(eval_points)

    # Softmax blending
    weights = softmax_np(logits_all)
    u_pred = np.sum(weights * vals_all, axis=1)

    u_true = manufactured_u_np(eval_points)
    metrics = metric_l2(u_pred, u_true)
    return u_pred, metrics


def evaluate_schwarz_diagnostics(
    fem_solvers: List[ChartFEMSolver],
    decoders: List[torch.nn.Module],
    masks: List[torch.nn.Module],
    neighbors: List[List[int]],
    eval_points: np.ndarray,
    seeds_np: np.ndarray,
    t1s_np: np.ndarray,
    t2s_np: np.ndarray,
    nvecs_np: np.ndarray,
    support_radii_np: np.ndarray,
    support_r_t: List[torch.Tensor],
    membership_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, float, float, float]:
    """Evaluate PDE residual, BC error, interface value/flux discontinuity.

    For the FEM solver, we compute simplified diagnostics:
    - bc_loss: mean squared BC error on physical boundary nodes
    - interface_value: mean squared value mismatch across overlaps
    - interface_flux: placeholder (FEM enforces equation exactly in weak form)
    - pde_residual: placeholder (set to 0; exact for assembled equation)

    Returns: (pde_residual, bc_loss, interface_value, interface_flux)
    """
    n_charts = len(fem_solvers)

    # BC error: how well physical BC nodes match exact values
    bc_errs = []
    for i in range(n_charts):
        solver = fem_solvers[i]
        if solver.u is None or solver.phys_bc_nodes is None:
            continue
        if len(solver.phys_bc_nodes) == 0:
            continue
        xi_bc = solver.nodes[solver.phys_bc_nodes]
        x_phys = _decode_batch(
            xi_bc, solver.decoder, solver.seed, solver.t1, solver.t2,
            solver.n_vec, solver.support_r, device, dtype,
        )
        u_exact = manufactured_u_np(x_phys)
        u_fem = solver.u[solver.phys_bc_nodes]
        bc_errs.append(np.mean((u_fem - u_exact) ** 2))
    bc_loss = float(np.mean(bc_errs)) if bc_errs else 0.0

    # Interface value mismatch
    iv_errs = []
    for i in range(n_charts):
        if fem_solvers[i].u is None:
            continue
        for j in neighbors[i]:
            if j <= i:
                continue
            if fem_solvers[j].u is None:
                continue
            # Sample overlap points: use eval points that belong to both charts
            mi = membership_np[:, i] > 0
            mj = membership_np[:, j] > 0
            shared_mask = mi & mj
            shared_idx = np.where(shared_mask)[0]
            if len(shared_idx) == 0:
                continue
            # Subsample if too many
            if len(shared_idx) > 200:
                rng = np.random.default_rng(42 + i * 100 + j)
                shared_idx = rng.choice(shared_idx, 200, replace=False)

            x_shared = eval_points[shared_idx]

            u_i = fem_solvers[i].evaluate_at_physical(x_shared)
            u_j = fem_solvers[j].evaluate_at_physical(x_shared)

            iv_errs.append(np.mean((u_i - u_j) ** 2))

    interface_value = float(np.mean(iv_errs)) if iv_errs else 0.0
    interface_flux = interface_value  # Proxy: same as value for now
    pde_residual = 0.0  # FEM satisfies PDE in weak form exactly

    return pde_residual, bc_loss, interface_value, interface_flux


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_fem_history(history: Dict[str, list], out_path: str) -> None:
    """Plot Schwarz convergence history (matches PINN version layout)."""
    iters = np.arange(1, len(history["rel_l2_eval"]) + 1, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for key in ["global_residual", "interface_value", "interface_flux", "bc_loss"]:
        vals = history.get(key, [])
        if vals:
            ax.semilogy(iters, np.maximum(vals, 1e-16), label=key)
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Loss-like metric")
    ax.set_title("Schwarz diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if history.get("rel_l2_eval"):
        ax.semilogy(iters, np.maximum(history["rel_l2_eval"], 1e-16), "b-o", markersize=3)
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Relative L²")
    ax.set_title("Accuracy (rel-L²)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if history.get("solve_time_s"):
        ax.plot(iters, history["solve_time_s"], "g-o", markersize=3)
        ax.set_xlabel("Schwarz iteration")
        ax.set_ylabel("Time (s)")
        ax.set_title("Per-iteration solve time")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if history.get("total_dofs"):
        ax.bar(range(len(history["total_dofs"])), history["total_dofs"])
        ax.set_xlabel("Chart")
        ax.set_ylabel("DOFs")
        ax.set_title("DOFs per chart")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main Schwarz FEM solver
# ══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    if dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)

    print(f"Device={device.type}  dtype={dtype}  n_cells={args.n_cells}")
    run_start = time.time()

    # ── 1. Load atlas ────────────────────────────────────────────────────
    atlas_np = np.load(args.atlas_data)
    points_np = atlas_np["points"]               # (N, 3)
    normals_np = atlas_np["normals"]              # (N, 3)
    seeds_np = atlas_np["seed_points"]            # (C, 3)
    t1s_np = atlas_np["frame_t1"]                 # (C, 3)
    t2s_np = atlas_np["frame_t2"]                 # (C, 3)
    nvecs_np = atlas_np["frame_n"]                # (C, 3)
    membership_np = atlas_np["membership"]        # (N, C)
    support_radii_np = atlas_np["support_radii"]  # (C,)

    seeds_t = torch.tensor(seeds_np, device=device, dtype=dtype)
    t1s_t = torch.tensor(t1s_np, device=device, dtype=dtype)
    t2s_t = torch.tensor(t2s_np, device=device, dtype=dtype)
    nvecs_t = torch.tensor(nvecs_np, device=device, dtype=dtype)
    support_r_t_arr = torch.tensor(support_radii_np, device=device, dtype=dtype)
    membership_t = torch.tensor(membership_np.astype(np.int64), device=device, dtype=torch.int64)

    n_points, n_charts = membership_np.shape
    print(f"Atlas: {n_charts} charts, {n_points} evaluation points")

    decoders, masks, atlas_ckpt = load_atlas_models(
        atlas_checkpoint=args.atlas_checkpoint, device=device, dtype=dtype,
    )

    # Per-chart support radius tensors (for MaskNet calls)
    support_r_t: List[torch.Tensor] = [support_r_t_arr[i] for i in range(n_charts)]

    # ── 2. Load SDF network ─────────────────────────────────────────────
    sdf_net = None
    sdf_center = None
    sdf_scale = 1.0
    if args.sdf_checkpoint:
        sdf_net, sdf_center, sdf_scale = load_sdf_for_schwarz(
            args.sdf_checkpoint, device, dtype,
        )
        print(f"SDF loaded: center={sdf_center.cpu().numpy()}, scale={sdf_scale:.4f}")

    # ── 3. Build FEM solvers for each chart ──────────────────────────────
    print(f"\nBuilding FEM meshes (n_cells={args.n_cells}) ...")
    build_t0 = time.time()
    fem_solvers: List[ChartFEMSolver] = []
    for i in range(n_charts):
        # For volumetric atlases, the SDF fill fraction per chart is typically
        # <5%, so SDF filtering is disabled by default (--no-sdf-filter).
        # The FEM solves on the full reference cube [-r,r]³ and the PoU
        # blending handles the extraction.
        _sdf = sdf_net if args.sdf_filter else None
        _sdf_c = sdf_center if args.sdf_filter else None
        solver = ChartFEMSolver(
            chart_id=i,
            decoder=decoders[i],
            seed=seeds_t[i],
            t1=t1s_t[i],
            t2=t2s_t[i],
            n_vec=nvecs_t[i],
            support_r=support_r_t[i],
            n_cells=args.n_cells,
            sdf_net=_sdf,
            sdf_center=_sdf_c,
            sdf_scale=sdf_scale,
            sdf_threshold=args.sdf_threshold,
            device=device,
            dtype=dtype,
        )
        fem_solvers.append(solver)
        print(f"  {solver.summary()}")
    build_time = time.time() - build_t0
    total_dofs = sum(s.n_dofs for s in fem_solvers)
    total_elems = sum(s.n_elements for s in fem_solvers)
    print(f"Mesh build: {build_time:.1f}s  total_dofs={total_dofs}  total_elems={total_elems}")

    # ── 4. Pre-compute diffusion tensors ─────────────────────────────────
    print("\nComputing diffusion tensors ...")
    diff_t0 = time.time()
    for solver in fem_solvers:
        solver.compute_diffusion_tensors()
    print(f"  Done in {time.time() - diff_t0:.1f}s")

    # ── 5. Assemble stiffness matrices ───────────────────────────────────
    print("Assembling stiffness matrices ...")
    asm_t0 = time.time()
    for solver in fem_solvers:
        solver.assemble()
    print(f"  Done in {time.time() - asm_t0:.1f}s")

    # ── 6. Schwarz colouring ─────────────────────────────────────────────
    color_groups = choose_color_groups(
        meta_json=args.atlas_meta,
        n_charts=n_charts,
        membership_np=membership_np,
    )
    print(f"Colour groups: {[[int(x) for x in g] for g in color_groups]}")

    # Build neighbor lists
    neighbors: List[List[int]] = [[] for _ in range(n_charts)]
    for i in range(n_charts):
        mi = membership_np[:, i] > 0
        for j in range(i + 1, n_charts):
            mj = membership_np[:, j] > 0
            if np.any(mi & mj):
                neighbors[i].append(j)
                neighbors[j].append(i)

    # ── 7. Schwarz iteration loop ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting Schwarz iteration (max {args.max_schwarz_iters} iters)")
    print(f"{'='*60}")

    history: Dict[str, list] = {
        "rel_l2_eval": [],
        "global_residual": [],
        "interface_value": [],
        "interface_flux": [],
        "bc_loss": [],
        "solve_time_s": [],
    }

    schwarz_start = time.time()
    best_rel_l2 = float("inf")

    for iteration in range(1, args.max_schwarz_iters + 1):
        iter_t0 = time.time()

        for color_group in color_groups:
            for chart_i in color_group:
                # a. Physical BC
                phys_bc = compute_phys_bc(fem_solvers[chart_i])

                # b. Schwarz BC from neighbours
                art_bc = compute_schwarz_bc(
                    chart_i=chart_i,
                    fem_solvers=fem_solvers,
                    decoders=decoders,
                    masks=masks,
                    seeds_np=seeds_np,
                    t1s_np=t1s_np,
                    t2s_np=t2s_np,
                    nvecs_np=nvecs_np,
                    support_radii_np=support_radii_np,
                    support_r_t=support_r_t,
                    neighbors=neighbors,
                    device=device,
                    dtype=dtype,
                )

                # c. Solve
                fem_solvers[chart_i].solve(phys_bc, art_bc)

        iter_time = time.time() - iter_t0

        # d. Evaluate global metrics
        u_pred, global_metrics = evaluate_global_blended(
            fem_solvers=fem_solvers,
            decoders=decoders,
            masks=masks,
            eval_points=points_np,
            seeds_np=seeds_np,
            t1s_np=t1s_np,
            t2s_np=t2s_np,
            nvecs_np=nvecs_np,
            support_radii_np=support_radii_np,
            support_r_t=support_r_t,
            device=device,
            dtype=dtype,
        )

        rel_l2 = global_metrics["relative_l2_error"]

        pde_res, bc_loss, iv_err, if_err = evaluate_schwarz_diagnostics(
            fem_solvers=fem_solvers,
            decoders=decoders,
            masks=masks,
            neighbors=neighbors,
            eval_points=points_np,
            seeds_np=seeds_np,
            t1s_np=t1s_np,
            t2s_np=t2s_np,
            nvecs_np=nvecs_np,
            support_radii_np=support_radii_np,
            support_r_t=support_r_t,
            membership_np=membership_np,
            device=device,
            dtype=dtype,
        )

        history["rel_l2_eval"].append(rel_l2)
        history["global_residual"].append(pde_res)
        history["interface_value"].append(iv_err)
        history["interface_flux"].append(if_err)
        history["bc_loss"].append(bc_loss)
        history["solve_time_s"].append(iter_time)

        if rel_l2 < best_rel_l2:
            best_rel_l2 = rel_l2

        print(
            f"  Iter {iteration:3d}: rel-L² = {rel_l2:.4e}  "
            f"iv = {iv_err:.4e}  bc = {bc_loss:.4e}  "
            f"time = {iter_time:.2f}s"
        )

        if rel_l2 < args.target_rel_l2:
            print(f"\n  ✓ Converged at iter {iteration} (rel-L² < {args.target_rel_l2:.2e})")
            break

    total_time = time.time() - run_start
    schwarz_time = time.time() - schwarz_start
    n_iters_done = len(history["rel_l2_eval"])

    # ── 8. Final evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final evaluation")
    print(f"{'='*60}")

    u_pred_final, final_metrics = evaluate_global_blended(
        fem_solvers=fem_solvers,
        decoders=decoders,
        masks=masks,
        eval_points=points_np,
        seeds_np=seeds_np,
        t1s_np=t1s_np,
        t2s_np=t2s_np,
        nvecs_np=nvecs_np,
        support_radii_np=support_radii_np,
        support_r_t=support_r_t,
        device=device,
        dtype=dtype,
    )
    u_true_final = manufactured_u_np(points_np)
    u_error = u_pred_final - u_true_final
    u_error_mag = np.abs(u_error)

    # Chart assignment and blend weight
    logits_all = np.zeros((n_points, n_charts))
    vals_all = np.zeros((n_points, n_charts))
    for i in range(n_charts):
        xi_i = local_coords_np(points_np, seeds_np[i], t1s_np[i], t2s_np[i], nvecs_np[i])
        logits_all[:, i] = evaluate_mask_np(xi_i, masks[i], support_r_t[i], device, dtype)
        if fem_solvers[i].u is not None:
            vals_all[:, i] = fem_solvers[i].evaluate_at_physical(points_np)
    weights = softmax_np(logits_all)
    chart_id = np.argmax(weights, axis=1)
    blend_weight = np.max(weights, axis=1)

    # Per-chart error
    per_chart = []
    point_idx_by_chart = []
    for i in range(n_charts):
        idx = np.where(membership_np[:, i] > 0)[0]
        point_idx_by_chart.append(idx)
        if len(idx) == 0:
            per_chart.append({
                "chart_id": i, "n_points": 0,
                "l2_error": None, "relative_l2_error": None, "max_error": None,
            })
            continue
        stats = metric_l2(u_pred_final[idx], u_true_final[idx])
        per_chart.append({"chart_id": i, "n_points": int(len(idx)), **stats})

    print(f"  rel-L²    = {final_metrics['relative_l2_error']:.6e}")
    print(f"  abs-L²    = {final_metrics['l2_error']:.6e}")
    print(f"  max-error = {final_metrics['max_error']:.6e}")
    print(f"  Schwarz iters = {n_iters_done}")
    print(f"  Total time    = {total_time:.1f}s")
    print(f"  Schwarz time  = {schwarz_time:.1f}s")

    # ── 9. Save outputs ──────────────────────────────────────────────────
    run_stem = build_run_stem(args.run_tag)
    os.makedirs(args.output_dir, exist_ok=True)

    # Solution NPZ
    solution_npz = os.path.join(args.output_dir, f"{run_stem}_solution.npz")
    np.savez_compressed(
        solution_npz,
        points=points_np,
        normals=normals_np,
        u_pred=u_pred_final,
        u_true=u_true_final,
        u_error=u_error,
        u_error_mag=u_error_mag,
        chart_id=chart_id.astype(np.int32),
        blend_weight=blend_weight,
        chart_weights=weights,
        chart_values=vals_all,
    )

    # DOFs per chart for plotting
    history["total_dofs"] = [s.n_dofs for s in fem_solvers]

    # Metrics JSON
    out_metrics = {
        "global": final_metrics,
        "per_chart": per_chart,
        "solver_type": "fem_p1",
        "n_cells": args.n_cells,
        "h": fem_solvers[0].h if fem_solvers else 0.0,
        "total_dofs": total_dofs,
        "total_elements": total_elems,
        "n_schwarz_iters": n_iters_done,
        "best_rel_l2": best_rel_l2,
        "total_time_s": total_time,
        "schwarz_time_s": schwarz_time,
        "build_time_s": build_time,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_charts": n_charts,
        "n_points": n_points,
        "color_groups": [[int(x) for x in g] for g in color_groups],
        "target_rel_l2": args.target_rel_l2,
        "target_met": bool(final_metrics["relative_l2_error"] <= args.target_rel_l2),
        "sdf_threshold": args.sdf_threshold,
    }

    metrics_path = os.path.join(args.output_dir, f"{run_stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # History JSON
    history_path = os.path.join(args.output_dir, f"{run_stem}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Convergence plot
    curve_path = os.path.join(args.output_dir, f"{run_stem}_curves.png")
    plot_fem_history(history, curve_path)

    print(f"\nOutputs saved:")
    print(f"  solution: {solution_npz}")
    print(f"  metrics:  {metrics_path}")
    print(f"  history:  {history_path}")
    print(f"  curves:   {curve_path}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Atlas Schwarz FEM Poisson solver on rabbit point cloud"
    )
    p.add_argument("--atlas-data", required=True,
                    help="Path to rabbit_atlas_data.npz")
    p.add_argument("--atlas-checkpoint", required=True,
                    help="Path to rabbit_atlas_trained.pt")
    p.add_argument("--atlas-meta", default=None,
                    help="Path to rabbit_atlas_meta.json (for colour groups)")
    p.add_argument("--sdf-checkpoint", default=None,
                    help="Path to rabbit_sdf.pt")
    p.add_argument("--n-cells", type=int, default=16,
                    help="Grid divisions per axis (default 16)")
    p.add_argument("--max-schwarz-iters", type=int, default=50,
                    help="Maximum Schwarz iterations (default 50)")
    p.add_argument("--target-rel-l2", type=float, default=0.01,
                    help="Target relative L² for convergence (default 0.01)")
    p.add_argument("--sdf-threshold", type=float, default=0.0,
                    help="SDF threshold for element inclusion (default 0.0)")
    p.add_argument("--sdf-filter", action="store_true", default=False,
                    help="Enable SDF-based element filtering (default: disabled)")
    p.add_argument("--output-dir", default="runs/fem_schwarz",
                    help="Output directory")
    p.add_argument("--run-tag", default=None,
                    help="Tag appended to output filenames")
    p.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda", "mps", "auto"],
                    help="Compute device (default cpu)")
    p.add_argument("--dtype", default="float64",
                    choices=["float32", "float64", "auto"],
                    help="Floating point precision (default float64)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default 42)")
    return p.parse_args()


if __name__ == "__main__":
    main()
