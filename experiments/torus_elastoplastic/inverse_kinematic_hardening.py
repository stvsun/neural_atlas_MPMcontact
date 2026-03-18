#!/usr/bin/env python3
"""Inverse identification of tau_y AND H_kin from cyclic displacement data.

Step 6: Kinematic hardening.  Given known elastic moduli (mu, K), recover
both the yield stress tau_y and kinematic hardening modulus H_kin from
displacement observations under cyclic loading.

Two-phase approach:
  Phase 1 — Identify tau_y from the initial (monotonic) loading phase with
             H_kin fixed at 0.  This exploits the fact that tau_y controls the
             initial yield point independently of hardening.
  Phase 2 — Jointly optimise tau_y + H_kin from the full cyclic history
             (including reverse loading).  The Bauschinger effect on unloading
             / reverse loading is the key signature that pins down H_kin.

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


# ---------------------------------------------------------------------------
# Generate synthetic observations
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
    epsilon: float = 1e-3,
    noise_std: float = 0.0,
) -> List[torch.Tensor]:
    """Run forward solve with true parameters and record displacements at sensors.

    Parameters
    ----------
    sensor_nodes : 1-D long tensor
        Indices of nodes where displacements are "measured".
    noise_std : float
        Standard deviation of additive Gaussian noise (fraction of max |u|).

    Returns
    -------
    u_obs_list : list of tensors, each shape (n_sensors, 3)
    """
    solver = IncrementalSolver(
        fem_solver, mu, K, tau_y_true, H_kin_true, epsilon=epsilon,
    )
    with torch.no_grad():
        u_hist, _ = solver.solve_history(
            bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

    u_obs_list = []
    for u in u_hist:
        u_sensor = u[sensor_nodes].clone()
        if noise_std > 0.0:
            u_max = u_sensor.abs().max().clamp(min=1e-30)
            u_sensor = u_sensor + noise_std * u_max * torch.randn_like(u_sensor)
        u_obs_list.append(u_sensor)

    return u_obs_list


# ---------------------------------------------------------------------------
# Displacement mismatch loss
# ---------------------------------------------------------------------------

def displacement_loss(
    u_hist: List[torch.Tensor],
    u_obs_list: List[torch.Tensor],
    sensor_nodes: torch.Tensor,
    step_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Compute relative displacement mismatch loss.

    Parameters
    ----------
    step_indices : list of int, optional
        Which time steps to include.  If None, use all.
    """
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
    """Phase 1: identify tau_y with H_kin fixed at 0.

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
# Phase 2: jointly identify tau_y + H_kin from full cyclic data
# ---------------------------------------------------------------------------

def phase2_identify_joint(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: float,
    H_kin_true: float,
    bc_schedule_full: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    u_obs_full: List[torch.Tensor],
    tau_y_init: float,
    H_kin_init: float = 5.0,
    tau_y_min: float = 0.01,
    n_iters: int = 200,
    lr: float = 1e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Phase 2: jointly identify tau_y and H_kin from full cyclic data.

    Returns history dict.
    """
    device = fem_solver.device
    dtype = fem_solver.dtype

    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )
    H_kin_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(H_kin_init, 0.01)),
                     device=device, dtype=dtype)
    )

    # Phase 2 strategy: first optimize H_kin only (tau_y frozen),
    # then jointly optimize both.  This avoids the tau_y-H_kin trade-off.
    n_freeze_tau_y = n_iters // 3  # freeze tau_y for first 1/3 of iterations

    optimizer_hkin = torch.optim.Adam([H_kin_raw], lr=lr)
    optimizer_joint = torch.optim.Adam([tau_y_raw, H_kin_raw], lr=lr * 0.5)

    history: Dict[str, List[float]] = {
        "loss": [], "tau_y": [], "H_kin": [],
        "tau_y_err_pct": [], "H_kin_err_pct": [],
    }

    t0 = time.time()

    for it in range(1, n_iters + 1):
        # Select optimizer: freeze tau_y for first portion
        if it <= n_freeze_tau_y:
            optimizer = optimizer_hkin
            tau_y_raw.requires_grad_(False)
        else:
            optimizer = optimizer_joint
            tau_y_raw.requires_grad_(True)

        optimizer.zero_grad()

        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
        H_kin_est = F_func.softplus(H_kin_raw)

        eps_current = cosine_anneal(it - 1, n_iters, epsilon_start, epsilon_end)

        inc_solver = IncrementalSolver(
            fem_solver, mu, K, tau_y_est, H_kin_est, epsilon=eps_current,
        )
        u_hist, _ = inc_solver.solve_history(
            bc_schedule_full, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

        loss = displacement_loss(u_hist, u_obs_full, sensor_nodes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([tau_y_raw, H_kin_raw], max_norm=max_grad_norm)
        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        H_kin_val = float(H_kin_est.item())
        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        H_kin_err = 100.0 * abs(H_kin_val - H_kin_true) / max(H_kin_true, 1e-12)
        loss_val = float(loss.item())

        history["loss"].append(loss_val)
        history["tau_y"].append(tau_y_val)
        history["H_kin"].append(H_kin_val)
        history["tau_y_err_pct"].append(tau_y_err)
        history["H_kin_err_pct"].append(H_kin_err)

        if verbose and (it <= 5 or it % max(1, n_iters // 10) == 0 or it == n_iters):
            elapsed = time.time() - t0
            print(
                f"  [Phase2] iter {it:4d}/{n_iters} | loss={loss_val:.4e} | "
                f"tau_y={tau_y_val:.4f} (err={tau_y_err:.2f}%) | "
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
    u_obs_mono: List[torch.Tensor],
    u_obs_full: List[torch.Tensor],
    tau_y_init: float = 1.0,
    H_kin_init: float = 5.0,
    tau_y_min: float = 0.01,
    n_iters_phase1: int = 100,
    n_iters_phase2: int = 200,
    lr_phase1: float = 5e-2,
    lr_phase2: float = 1e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Two-phase inverse identification of tau_y and H_kin.

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

    print("\n  === Phase 2: Joint tau_y + H_kin from cyclic data ===")
    hist2 = phase2_identify_joint(
        fem_solver, mu, K, tau_y_true, H_kin_true,
        bc_schedule_full, bc_mask, sensor_nodes, u_obs_full,
        tau_y_init=tau_y_est,   # warm-start from Phase 1
        H_kin_init=H_kin_init, tau_y_min=tau_y_min,
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
    print("Inverse Kinematic Hardening — Self-test")
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

    # --- Cyclic loading schedule with increasing peak amplitude ---
    # Each cycle increases the peak strain to accumulate more plastic strain
    # and expose the Bauschinger effect (back-stress shift).
    eps_base = 0.04
    n_cycles = 3
    n_steps_per_half = 8  # steps per half-cycle (load or unload)
    amp_growth = 1.5      # peak amplitude multiplier per cycle

    bc_schedule_full = []
    for cycle in range(n_cycles):
        eps_peak = eps_base * (amp_growth ** cycle)
        # Forward half-cycle: 0 -> +eps_peak
        for step in range(n_steps_per_half):
            lam = (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)
        # Reverse half-cycle: +eps_peak -> -eps_peak
        for step in range(2 * n_steps_per_half):
            lam = 1.0 - (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)
        # Return to zero: -eps_peak -> 0
        for step in range(n_steps_per_half):
            lam = -1.0 + (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule_full.append(u_bc)

    # Monotonic part: use larger amplitude for better tau_y identification
    eps_mono = eps_base * amp_growth  # use 2nd cycle's peak for monotonic
    n_mono = 2 * n_steps_per_half  # more steps for monotonic
    bc_schedule_mono = []
    for step in range(n_mono):
        lam = (step + 1) / n_mono
        u_bc = torch.zeros_like(nodes)
        u_bc[right_face, 0] = eps_mono * lam * 2.0 * r
        bc_schedule_mono.append(u_bc)

    total_steps = len(bc_schedule_full)
    print(f"\n  Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")
    print(f"  BC nodes: {bc_mask.sum().item()} (left+right x-faces)")
    print(f"  Loading: {n_cycles} cycles, {n_steps_per_half} steps/half, "
          f"amp growth={amp_growth}x, total={total_steps} steps")
    print(f"  Peak strains: {[eps_base * amp_growth**c for c in range(n_cycles)]}")
    print(f"  True tau_y = {tau_y_true}, H_kin = {H_kin_true}")

    # --- Sensor nodes ---
    free_indices = torch.where(~bc_mask)[0]
    n_sensors = min(20, free_indices.shape[0])
    sensor_nodes = free_indices[:n_sensors]
    print(f"  Sensors: {n_sensors} free nodes")

    # --- Generate synthetic observations ---
    print("\n  Generating synthetic data (forward solve with true params)...")
    u_obs_full = generate_synthetic_data(
        fem, mu, K, tau_y_true_t, H_kin_true_t,
        bc_schedule_full, bc_mask, sensor_nodes,
        epsilon=1e-3, noise_std=0.0,
    )
    u_obs_mono = u_obs_full[:n_mono]
    print(f"  Observations: {len(u_obs_full)} steps total, "
          f"{len(u_obs_mono)} monotonic")

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
        u_obs_mono=u_obs_mono,
        u_obs_full=u_obs_full,
        tau_y_init=1.0,
        H_kin_init=5.0,
        tau_y_min=0.01,
        n_iters_phase1=150,
        n_iters_phase2=300,
        lr_phase1=5e-2,
        lr_phase2=2e-2,
        epsilon_start=0.1,
        epsilon_end=1e-3,
        max_grad_norm=5.0,
        verbose=True,
    )
    print("-" * 70)

    # --- Results ---
    final_tau_y = hist2["tau_y"][-1]
    final_H_kin = hist2["H_kin"][-1]
    final_tau_y_err = hist2["tau_y_err_pct"][-1]
    final_H_kin_err = hist2["H_kin_err_pct"][-1]
    best_tau_y_err = min(hist2["tau_y_err_pct"])
    best_H_kin_err = min(hist2["H_kin_err_pct"])

    print(f"\n  Phase 1 final tau_y = {hist1['tau_y'][-1]:.4f} "
          f"(err = {hist1['tau_y_err_pct'][-1]:.2f}%)")
    print(f"\n  Phase 2 final tau_y = {final_tau_y:.4f} "
          f"(true = {tau_y_true:.4f}, err = {final_tau_y_err:.2f}%)")
    print(f"  Phase 2 final H_kin = {final_H_kin:.2f} "
          f"(true = {H_kin_true:.2f}, err = {final_H_kin_err:.2f}%)")
    print(f"  Best tau_y err = {best_tau_y_err:.2f}%")
    print(f"  Best H_kin err = {best_H_kin_err:.2f}%")

    passed_tau_y = best_tau_y_err < 20.0
    passed_H_kin = best_H_kin_err < 20.0
    print(f"\n  tau_y within 20%: {'[PASS]' if passed_tau_y else '[FAIL]'}")
    print(f"  H_kin within 20%: {'[PASS]' if passed_H_kin else '[FAIL]'}")

    print("\n" + "=" * 70)
    print("Inverse Kinematic Hardening — Self-test complete.")
    print("=" * 70)
