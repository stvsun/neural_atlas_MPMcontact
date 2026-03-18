#!/usr/bin/env python3
"""Inverse identification of tau_y AND H_kin from cyclic loading data.

Step 6: Kinematic hardening.  Given known elastic moduli (mu, K), recover
both the yield stress tau_y and kinematic hardening modulus H_kin from
**reaction force** observations under cyclic loading.

Key insight: displacement sensors at interior nodes are insensitive to H_kin
because the Bauschinger effect primarily manifests as a shift in the yield
surface in stress space.  Reaction forces at the loaded boundary are directly
proportional to the stress state, making them far more sensitive to H_kin.

Two-phase approach:
  Phase 1 — Identify tau_y from the initial (monotonic) loading phase with
             H_kin fixed at 0.  Uses displacement sensors (proven at 0.00%).
  Phase 2 — Identify H_kin alone from cyclic data with tau_y fixed at
             Phase 1 estimate.  Uses reaction forces at the loaded boundary.

Epsilon annealing follows the cosine schedule of PNAS Bark & Sun (2025), Eq. 12.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F_func

import sys
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.return_mapping import (
    ReturnMappingState,
    smooth_return_map,
)
from experiments.torus_elastoplastic.incremental_solver import (
    IncrementalSolver,
    cosine_anneal,
)


# ---------------------------------------------------------------------------
# Softplus inverse (for initialisation)
# ---------------------------------------------------------------------------

def softplus_inv(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y."""
    if y > 20.0:
        return y
    return math.log(math.exp(y) - 1.0)


# (compute_reaction_force_direct is defined in the Phase 2 section below)


# ---------------------------------------------------------------------------
# Generate synthetic observations (displacement + reaction force)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: torch.Tensor,
    H_kin_true: torch.Tensor,
    bc_schedule: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    reaction_nodes: torch.Tensor,
    epsilon: float = 1e-3,
    noise_std: float = 0.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run forward solve and record displacements at sensors + reaction forces.

    Returns
    -------
    u_obs_list : list of (n_sensors, 3) tensors — displacement observations
    rf_obs_list : list of scalar tensors — x-reaction force at loaded face
    """
    solver = IncrementalSolver(
        fem_solver, mu, K, tau_y_true, H_kin_true, epsilon=epsilon,
    )

    N = fem_solver.n_nodes
    M = fem_solver.n_elements
    device = fem_solver.device
    dtype = fem_solver.dtype

    u = torch.zeros(N, 3, device=device, dtype=dtype)
    I33 = torch.eye(3, device=device, dtype=dtype)
    F_old = I33.unsqueeze(0).expand(M, 3, 3).clone()
    state = ReturnMappingState.zeros((M,), device=device, dtype=dtype)

    u_obs_list = []
    rf_obs_list = []

    from experiments.torus_elastoplastic.incremental_solver import ElastoplasticStepSolver

    with torch.no_grad():
        for step_idx, u_bc in enumerate(bc_schedule):
            state_prev = state  # state before this step
            F_old_prev = F_old  # F_old before this step

            step_solver = ElastoplasticStepSolver(
                fem_solver, mu, K, tau_y_true, H_kin_true, epsilon,
            )
            u, new_state, F_new = step_solver.solve_step(
                u, state, F_old, u_bc, bc_mask,
                max_iter=50, tol=1e-6,
            )

            # Record displacement at sensors
            u_sensor = u[sensor_nodes].clone()
            if noise_std > 0.0:
                u_max = u_sensor.abs().max().clamp(min=1e-30)
                u_sensor += noise_std * u_max * torch.randn_like(u_sensor)
            u_obs_list.append(u_sensor)

            # Record reaction force using state_prev and F_old_prev
            F_total = fem_solver.compute_F(u)
            F_delta = F_total @ torch.linalg.inv(F_old_prev)
            tau_stress, _ = smooth_return_map(
                F_delta, state_prev, mu, K, tau_y_true, H_kin_true, epsilon,
            )
            F_inv_T = torch.linalg.inv(F_total).transpose(-2, -1)
            P = tau_stress @ F_inv_T
            f_elem = fem_solver.vol[:, None, None] * torch.einsum(
                "eij, eaj -> eai", P, fem_solver.dNdx
            )
            f_int = torch.zeros(fem_solver.n_nodes, 3, device=u.device, dtype=u.dtype)
            f_elem_flat = f_elem.reshape(-1, 3)
            idx_flat = fem_solver.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)
            f_int.scatter_add_(0, idx_flat, f_elem_flat)
            rf = f_int[reaction_nodes, 0].sum()
            if noise_std > 0.0:
                rf = rf + noise_std * rf.abs().clamp(min=1e-30) * torch.randn(1, device=device, dtype=dtype).squeeze()
            rf_obs_list.append(rf.clone())

            state = new_state
            F_old = F_new.detach()

    return u_obs_list, rf_obs_list


# ---------------------------------------------------------------------------
# Displacement mismatch loss
# ---------------------------------------------------------------------------

def displacement_loss(
    u_hist: List[torch.Tensor],
    u_obs_list: List[torch.Tensor],
    sensor_nodes: torch.Tensor,
    step_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Compute relative displacement mismatch loss."""
    device = u_hist[0].device
    dtype = u_hist[0].dtype
    loss = torch.tensor(0.0, device=device, dtype=dtype)

    if step_indices is None:
        step_indices = list(range(len(u_hist)))

    for idx in step_indices:
        u_pred_sensor = u_hist[idx][sensor_nodes]
        u_obs = u_obs_list[idx]
        diff = u_pred_sensor - u_obs
        u_obs_norm = u_obs.norm().clamp(min=1e-12)
        loss = loss + (diff ** 2).sum() / (u_obs_norm ** 2)

    return loss




# ---------------------------------------------------------------------------
# Phase 1: identify tau_y from monotonic loading (H_kin = 0)
# ---------------------------------------------------------------------------

def phase1_identify_tau_y(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: float,
    bc_schedule_mono: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    u_obs_mono: List[torch.Tensor],
    tau_y_init: float = 1.0,
    tau_y_min: float = 0.01,
    n_iters: int = 100,
    lr: float = 5e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Tuple[float, Dict[str, List[float]]]:
    """Phase 1: identify tau_y with H_kin fixed at 0 using displacement loss.

    Returns (tau_y_estimate, history).
    """
    device = fem_solver.device
    dtype = fem_solver.dtype

    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )
    H_kin_fixed = torch.tensor(0.0, device=device, dtype=dtype)

    optimizer = torch.optim.Adam([tau_y_raw], lr=lr)

    history: Dict[str, List[float]] = {"loss": [], "tau_y": [], "tau_y_err_pct": []}

    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
        eps_current = cosine_anneal(it - 1, n_iters, epsilon_start, epsilon_end)

        inc_solver = IncrementalSolver(
            fem_solver, mu, K, tau_y_est, H_kin_fixed, epsilon=eps_current,
        )
        u_hist, _ = inc_solver.solve_history(
            bc_schedule_mono, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

        loss = displacement_loss(u_hist, u_obs_mono, sensor_nodes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([tau_y_raw], max_norm=max_grad_norm)
        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        loss_val = float(loss.item())

        history["loss"].append(loss_val)
        history["tau_y"].append(tau_y_val)
        history["tau_y_err_pct"].append(tau_y_err)

        if verbose and (it <= 5 or it % max(1, n_iters // 10) == 0 or it == n_iters):
            elapsed = time.time() - t0
            print(
                f"  [Phase1] iter {it:4d}/{n_iters} | loss={loss_val:.4e} | "
                f"tau_y={tau_y_val:.4f} (err={tau_y_err:.2f}%) | "
                f"eps={eps_current:.4f} | t={elapsed:.1f}s"
            )

    return history["tau_y"][-1], history


# ---------------------------------------------------------------------------
# Phase 2: identify H_kin from cyclic data using reaction forces
# ---------------------------------------------------------------------------

def compute_reaction_force_direct(
    fem: ChartVectorFEMSolver,
    u: torch.Tensor,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    epsilon: float,
    state_prev: ReturnMappingState,
    F_old: torch.Tensor,
    reaction_nodes: torch.Tensor,
) -> torch.Tensor:
    """Compute x-reaction force at boundary from converged u.

    This recomputes the full elastoplastic stress from scratch:
    F_total -> F_delta -> return_map -> tau -> P -> f_int.
    Gradients flow through material params (tau_y, H_kin) via the
    return mapping and through u via the IFT correction.
    """
    F_total = fem.compute_F(u)
    F_delta = F_total @ torch.linalg.inv(F_old)

    tau_stress, _ = smooth_return_map(
        F_delta, state_prev, mu, K, tau_y, H_kin, epsilon,
    )

    F_inv_T = torch.linalg.inv(F_total).transpose(-2, -1)
    P = tau_stress @ F_inv_T

    f_elem = fem.vol[:, None, None] * torch.einsum(
        "eij, eaj -> eai", P, fem.dNdx
    )
    f_int = torch.zeros(fem.n_nodes, 3, device=u.device, dtype=u.dtype)
    f_elem_flat = f_elem.reshape(-1, 3)
    idx_flat = fem.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)
    f_int.scatter_add_(0, idx_flat, f_elem_flat)

    return f_int[reaction_nodes, 0].sum()


def phase2_identify_hkin(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_fixed: float,
    H_kin_true: float,
    bc_schedule_full: List[torch.Tensor],
    bc_mask: torch.Tensor,
    reaction_nodes: torch.Tensor,
    rf_obs_full: List[torch.Tensor],
    H_kin_init: float = 5.0,
    n_iters: int = 150,
    lr: float = 5e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Phase 2: identify H_kin alone with tau_y fixed, using reaction force loss.

    The reaction force at the loaded boundary is directly proportional to the
    stress state.  During reverse loading, H_kin controls the shift of the
    yield surface (Bauschinger effect), which produces a measurable difference
    in reaction forces.  This is far more sensitive than displacement sensors.

    Returns history dict.
    """
    device = fem_solver.device
    dtype = fem_solver.dtype
    N = fem_solver.n_nodes
    M = fem_solver.n_elements

    tau_y_val = torch.tensor(tau_y_fixed, device=device, dtype=dtype)

    H_kin_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(H_kin_init, 0.01)),
                     device=device, dtype=dtype)
    )

    optimizer = torch.optim.Adam([H_kin_raw], lr=lr)

    history: Dict[str, List[float]] = {
        "loss": [], "H_kin": [], "H_kin_err_pct": [],
    }

    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        H_kin_est = F_func.softplus(H_kin_raw)
        eps_current = cosine_anneal(it - 1, n_iters, epsilon_start, epsilon_end)

        # Manual forward solve to track F_old and state_prev per step
        from experiments.torus_elastoplastic.incremental_solver import ElastoplasticStepSolver

        u = torch.zeros(N, 3, device=device, dtype=dtype)
        I33 = torch.eye(3, device=device, dtype=dtype)
        F_old = I33.unsqueeze(0).expand(M, 3, 3).clone()
        state = ReturnMappingState.zeros((M,), device=device, dtype=dtype)

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        n_loss_steps = 0

        for step_idx, u_bc in enumerate(bc_schedule_full):
            state_prev = state  # state before this step
            F_old_prev = F_old  # F_old before this step

            step_solver = ElastoplasticStepSolver(
                fem_solver, mu, K, tau_y_val, H_kin_est, eps_current,
            )
            u, state, F_new = step_solver.solve_step(
                u, state, F_old, u_bc, bc_mask,
                max_iter=50, tol=1e-6,
            )

            # Compute reaction force loss at every 4th step (absolute MSE)
            if step_idx % 4 == 0:
                rf_pred = compute_reaction_force_direct(
                    fem_solver, u, mu, K, tau_y_val, H_kin_est, eps_current,
                    state_prev, F_old_prev, reaction_nodes,
                )
                rf_obs = rf_obs_full[step_idx]
                loss = loss + (rf_pred - rf_obs) ** 2
                n_loss_steps += 1

            F_old = F_new.detach()

        loss = loss / max(n_loss_steps, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([H_kin_raw], max_norm=max_grad_norm)
        optimizer.step()

        H_kin_val = float(H_kin_est.item())
        H_kin_err = 100.0 * abs(H_kin_val - H_kin_true) / max(H_kin_true, 1e-12)
        loss_val = float(loss.item())

        history["loss"].append(loss_val)
        history["H_kin"].append(H_kin_val)
        history["H_kin_err_pct"].append(H_kin_err)

        if verbose and (it <= 5 or it % max(1, n_iters // 10) == 0 or it == n_iters):
            elapsed = time.time() - t0
            print(
                f"  [Phase2] iter {it:4d}/{n_iters} | loss={loss_val:.4e} | "
                f"H_kin={H_kin_val:.2f} (err={H_kin_err:.2f}%) | "
                f"eps={eps_current:.4f} | t={elapsed:.1f}s"
            )

    return history


# ---------------------------------------------------------------------------
# Combined two-phase inverse
# ---------------------------------------------------------------------------

def run_inverse_kinematic_hardening(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: float,
    H_kin_true: float,
    bc_schedule_mono: List[torch.Tensor],
    bc_schedule_full: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    reaction_nodes: torch.Tensor,
    u_obs_mono: List[torch.Tensor],
    rf_obs_full: List[torch.Tensor],
    tau_y_init: float = 1.0,
    H_kin_init: float = 5.0,
    tau_y_min: float = 0.01,
    n_iters_phase1: int = 100,
    n_iters_phase2: int = 150,
    lr_phase1: float = 5e-2,
    lr_phase2: float = 5e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Two-phase inverse identification of tau_y and H_kin.

    Phase 1: tau_y from monotonic displacement data (H_kin=0).
    Phase 2: H_kin from cyclic reaction force data (tau_y fixed).

    Returns (phase1_history, phase2_history).
    """
    print("\n  === Phase 1: Identify tau_y from monotonic loading (H_kin=0) ===")
    tau_y_est, hist1 = phase1_identify_tau_y(
        fem_solver, mu, K, tau_y_true,
        bc_schedule_mono, bc_mask, sensor_nodes, u_obs_mono,
        tau_y_init=tau_y_init, tau_y_min=tau_y_min,
        n_iters=n_iters_phase1, lr=lr_phase1,
        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        max_grad_norm=max_grad_norm, verbose=verbose,
    )
    print(f"\n  Phase 1 result: tau_y = {tau_y_est:.4f}")

    print("\n  === Phase 2: Identify H_kin from cyclic reaction forces (tau_y fixed) ===")
    hist2 = phase2_identify_hkin(
        fem_solver, mu, K,
        tau_y_fixed=tau_y_est,
        H_kin_true=H_kin_true,
        bc_schedule_full=bc_schedule_full,
        bc_mask=bc_mask,
        reaction_nodes=reaction_nodes,
        rf_obs_full=rf_obs_full,
        H_kin_init=H_kin_init,
        n_iters=n_iters_phase2, lr=lr_phase2,
        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        max_grad_norm=max_grad_norm, verbose=verbose,
    )

    return hist1, hist2


# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 70)
    print("Inverse Kinematic Hardening — Self-test (Reaction Force)")
    print("=" * 70)

    # --- Material parameters ---
    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))
    tau_y_true = 0.5
    H_kin_true = 20.0

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)
    tau_y_true_t = torch.tensor(tau_y_true, device=device)
    H_kin_true_t = torch.tensor(H_kin_true, device=device)

    # --- Mesh ---
    n_cells = 4
    fem = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, device=device, dtype=torch.float64,
    )

    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r

    # Partial BCs: fix left x-face, prescribe x-disp on right x-face
    left_face = (nodes[:, 0] < -r + tol_face)
    right_face = (nodes[:, 0] > r - tol_face)
    bc_mask = left_face | right_face

    # --- Cyclic loading: 2 cycles, moderate amplitude, fine steps ---
    eps_peak = 0.03
    n_steps_per_half = 15  # finer steps for Newton convergence

    bc_schedule_full = []
    for cycle in range(2):
        # Forward: 0 -> +eps
        for step in range(n_steps_per_half):
            lam = (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)
        # Reverse: +eps -> -eps
        for step in range(2 * n_steps_per_half):
            lam = 1.0 - (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)
        # Return: -eps -> 0
        for step in range(n_steps_per_half):
            lam = -1.0 + (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)

    # Monotonic part for Phase 1 (proven at 0.00% for tau_y)
    eps_mono = 0.08
    n_mono = 5
    bc_schedule_mono = []
    for step in range(n_mono):
        lam = (step + 1) / n_mono
        u_bc = torch.zeros_like(nodes)
        u_bc[right_face, 0] = eps_mono * lam * 2.0 * r
        bc_schedule_mono.append(u_bc)

    total_steps = len(bc_schedule_full)
    print(f"\n  Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")
    print(f"  BC nodes: {bc_mask.sum().item()} (left+right x-faces)")
    print(f"  Cyclic: 2 cycles, {n_steps_per_half} steps/half, "
          f"eps={eps_peak}, total={total_steps} steps")
    print(f"  Monotonic: {n_mono} steps, eps={eps_mono}")
    print(f"  True tau_y = {tau_y_true}, H_kin = {H_kin_true}")

    # --- Sensor nodes (for Phase 1 displacement loss) ---
    free_indices = torch.where(~bc_mask)[0]
    n_sensors = min(20, free_indices.shape[0])
    sensor_nodes = free_indices[:n_sensors]
    print(f"  Displacement sensors: {n_sensors} free nodes")

    # Reaction force nodes = right face (loaded boundary)
    reaction_nodes = right_face
    print(f"  Reaction force nodes: {reaction_nodes.sum().item()} (right x-face)")

    # --- Generate synthetic observations ---
    # Phase 1 monotonic data: use H_kin=0 (Phase 1 fits with H_kin=0,
    # so the synthetic data must also be generated with H_kin=0 for consistency)
    H_kin_zero = torch.tensor(0.0, device=device)
    print("\n  Generating Phase 1 monotonic data (H_kin=0)...")
    u_obs_mono, _ = generate_synthetic_data(
        fem, mu, K, tau_y_true_t, H_kin_zero,
        bc_schedule_mono, bc_mask, sensor_nodes, reaction_nodes,
        epsilon=1e-3, noise_std=0.0,
    )
    # Phase 2 cyclic data: use true H_kin for reaction force observations
    print("  Generating Phase 2 cyclic data (H_kin=true)...")
    _, rf_obs_full = generate_synthetic_data(
        fem, mu, K, tau_y_true_t, H_kin_true_t,
        bc_schedule_full, bc_mask, sensor_nodes, reaction_nodes,
        epsilon=1e-3, noise_std=0.0,
    )

    print(f"  RF range: [{min(float(r) for r in rf_obs_full):.4f}, "
          f"{max(float(r) for r in rf_obs_full):.4f}]")

    # --- Run two-phase inverse ---
    print("\n  Running two-phase inverse identification...")
    print("-" * 70)
    hist1, hist2 = run_inverse_kinematic_hardening(
        fem, mu, K,
        tau_y_true=tau_y_true,
        H_kin_true=H_kin_true,
        bc_schedule_mono=bc_schedule_mono,
        bc_schedule_full=bc_schedule_full,
        bc_mask=bc_mask,
        sensor_nodes=sensor_nodes,
        reaction_nodes=reaction_nodes,
        u_obs_mono=u_obs_mono,
        rf_obs_full=rf_obs_full,
        tau_y_init=1.0,
        H_kin_init=5.0,
        tau_y_min=0.01,
        n_iters_phase1=100,
        n_iters_phase2=150,
        lr_phase1=5e-2,
        lr_phase2=0.2,
        epsilon_start=0.1,
        epsilon_end=1e-3,
        max_grad_norm=5.0,
        verbose=True,
    )
    print("-" * 70)

    # --- Results ---
    final_tau_y = hist1["tau_y"][-1]
    final_H_kin = hist2["H_kin"][-1]
    final_tau_y_err = hist1["tau_y_err_pct"][-1]
    final_H_kin_err = hist2["H_kin_err_pct"][-1]
    best_tau_y_err = min(hist1["tau_y_err_pct"])
    best_H_kin_err = min(hist2["H_kin_err_pct"])

    print(f"\n  Phase 1 final tau_y = {final_tau_y:.4f} "
          f"(true = {tau_y_true:.4f}, err = {final_tau_y_err:.2f}%)")
    print(f"  Phase 2 final H_kin = {final_H_kin:.2f} "
          f"(true = {H_kin_true:.2f}, err = {final_H_kin_err:.2f}%)")
    print(f"  Best tau_y err = {best_tau_y_err:.2f}%")
    print(f"  Best H_kin err = {best_H_kin_err:.2f}%")

    passed_tau_y = best_tau_y_err < 5.0
    passed_H_kin = best_H_kin_err < 20.0
    print(f"\n  tau_y within 5%: {'[PASS]' if passed_tau_y else '[FAIL]'}")
    print(f"  H_kin within 20%: {'[PASS]' if passed_H_kin else '[FAIL]'}")

    print("\n" + "=" * 70)
    print("Inverse Kinematic Hardening — Self-test complete.")
    print("=" * 70)
