#!/usr/bin/env python3
"""Inverse identification of yield stress tau_y from synthetic displacement data.

Step 5: Perfect plasticity (H_kin = 0).  Given known elastic moduli (mu, K),
recover the yield stress tau_y by minimising the displacement mismatch between
a forward elastoplastic simulation and synthetic "observed" data.

The entire pipeline — forward solve via incremental return mapping, loss
computation, and gradient-based optimisation — is differentiable via PyTorch
autograd.  The softplus reparameterisation ensures positivity of tau_y.

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
    H_kin: torch.Tensor,
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
        Observed displacements at sensor nodes for each load step.
    """
    solver = IncrementalSolver(
        fem_solver, mu, K, tau_y_true, H_kin, epsilon=epsilon,
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
# Inverse solver
# ---------------------------------------------------------------------------

def run_inverse_perfect_plasticity(
    fem_solver: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: float,
    bc_schedule: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    u_obs_list: List[torch.Tensor],
    tau_y_init: Optional[float] = None,
    tau_y_min: float = 0.01,
    n_iters: int = 200,
    lr: float = 1e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    max_grad_norm: float = 5.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Identify tau_y from displacement observations (perfect plasticity, H_kin=0).

    Parameters
    ----------
    tau_y_init : float or None
        Initial guess for tau_y.  If None, defaults to 2 * tau_y_true.
    tau_y_min : float
        Lower bound enforced via softplus offset.

    Returns
    -------
    history : dict with keys 'loss', 'tau_y', 'tau_y_err_pct'
    """
    device = fem_solver.device
    dtype = fem_solver.dtype

    if tau_y_init is None:
        tau_y_init = 2.0 * tau_y_true

    # Softplus parameterisation: tau_y = softplus(tau_y_raw) + tau_y_min
    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )

    H_kin_fixed = torch.tensor(0.0, device=device, dtype=dtype)
    optimizer = torch.optim.Adam([tau_y_raw], lr=lr)

    history: Dict[str, List[float]] = {
        "loss": [],
        "tau_y": [],
        "tau_y_err_pct": [],
    }

    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min

        # Cosine-anneal epsilon across optimisation iterations
        eps_current = cosine_anneal(it - 1, n_iters, epsilon_start, epsilon_end)

        inc_solver = IncrementalSolver(
            fem_solver, mu, K, tau_y_est, H_kin_fixed, epsilon=eps_current,
        )

        u_hist, _ = inc_solver.solve_history(
            bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

        # Loss: sum of relative displacement errors over time steps and sensors
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for step_idx, (u_pred, u_obs) in enumerate(zip(u_hist, u_obs_list)):
            u_pred_sensor = u_pred[sensor_nodes]
            diff = u_pred_sensor - u_obs
            # Relative error: normalise by observed displacement magnitude
            u_obs_norm = u_obs.norm().clamp(min=1e-12)
            loss = loss + (diff ** 2).sum() / (u_obs_norm ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([tau_y_raw], max_norm=max_grad_norm)
        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        loss_val = float(loss.item())

        history["loss"].append(loss_val)
        history["tau_y"].append(tau_y_val)
        history["tau_y_err_pct"].append(tau_y_err)

        if verbose and (it <= 10 or it % max(1, n_iters // 20) == 0 or it == n_iters):
            elapsed = time.time() - t0
            print(
                f"  iter {it:4d}/{n_iters} | loss={loss_val:.4e} | "
                f"tau_y={tau_y_val:.4f} (true={tau_y_true:.4f}, err={tau_y_err:.2f}%) | "
                f"eps={eps_current:.4f} | t={elapsed:.1f}s"
            )

    return history


# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 70)
    print("Inverse Perfect Plasticity — Self-test")
    print("=" * 70)

    # --- Material parameters ---
    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))
    tau_y_true = 0.5

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)
    tau_y_true_t = torch.tensor(tau_y_true, device=device)
    H_kin_zero = torch.tensor(0.0, device=device)

    # --- Mesh ---
    n_cells = 4
    fem = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, device=device, dtype=torch.float64,
    )

    # --- Partial BCs: fix left x-face, prescribe x-disp on right x-face ---
    # y/z DOFs free on loaded face → lateral contraction depends on plasticity
    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r
    left_face = (nodes[:, 0] < -r + tol_face)
    right_face = (nodes[:, 0] > r - tol_face)
    bc_mask = left_face | right_face  # only x-faces constrained

    eps_max = 0.08  # large enough to cause yielding
    n_steps = 5
    bc_schedule = []
    for step in range(n_steps):
        lam = (step + 1) / n_steps
        u_bc = torch.zeros_like(nodes)
        # Left face: fixed (u=0)
        # Right face: prescribed x-displacement
        u_bc[right_face, 0] = eps_max * lam * 2.0 * r
        bc_schedule.append(u_bc)

    # --- Sensor nodes: free (non-BC) nodes ---
    free_indices = torch.where(~bc_mask)[0]
    n_sensors = min(20, free_indices.shape[0])
    sensor_nodes = free_indices[:n_sensors]
    print(f"\n  Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")
    print(f"  BC nodes: {bc_mask.sum().item()} (left+right x-faces)")
    print(f"  Free nodes: {(~bc_mask).sum().item()}")
    print(f"  Sensors: {n_sensors} free nodes")
    print(f"  Load steps: {n_steps} monotonic, eps_max={eps_max}")
    print(f"  True tau_y = {tau_y_true}")

    # --- Generate synthetic observations ---
    print("\n  Generating synthetic data (forward solve with true params)...")
    u_obs_list = generate_synthetic_data(
        fem, mu, K, tau_y_true_t, H_kin_zero,
        bc_schedule, bc_mask, sensor_nodes,
        epsilon=1e-3, noise_std=0.0,
    )
    print(f"  Observations generated: {len(u_obs_list)} time steps, "
          f"{sensor_nodes.shape[0]} sensors each")

    # --- Run inverse ---
    print("\n  Running inverse identification (tau_y)...")
    print("-" * 70)
    history = run_inverse_perfect_plasticity(
        fem, mu, K,
        tau_y_true=tau_y_true,
        bc_schedule=bc_schedule,
        bc_mask=bc_mask,
        sensor_nodes=sensor_nodes,
        u_obs_list=u_obs_list,
        tau_y_init=1.0,       # start at 2x true value
        tau_y_min=0.01,
        n_iters=200,
        lr=5e-2,
        epsilon_start=0.1,
        epsilon_end=1e-3,
        max_grad_norm=5.0,
        verbose=True,
    )
    print("-" * 70)

    # --- Results ---
    final_tau_y = history["tau_y"][-1]
    final_err = history["tau_y_err_pct"][-1]
    best_err = min(history["tau_y_err_pct"])
    best_iter = history["tau_y_err_pct"].index(best_err) + 1

    print(f"\n  Final tau_y = {final_tau_y:.4f} (true = {tau_y_true:.4f})")
    print(f"  Final error = {final_err:.2f}%")
    print(f"  Best error  = {best_err:.2f}% at iter {best_iter}")

    passed = best_err < 15.0
    print(f"\n  Recovery within 15%: {'[PASS]' if passed else '[FAIL]'}")

    print("\n" + "=" * 70)
    print("Inverse Perfect Plasticity — Self-test complete.")
    print("=" * 70)
