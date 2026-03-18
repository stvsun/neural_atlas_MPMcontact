#!/usr/bin/env python3
"""Compare finite-difference vs autograd gradients for tau_y inverse identification.

Three experiments:
  (a) Smooth return mapping + autograd  (the standard differentiable approach)
  (b) Smooth return mapping + finite-difference gradient
  (c) Non-smooth return mapping (hard max) + autograd  (shows NaN gradients)

All use the same tau_y identification problem: E=200, nu=0.3, tau_y_true=0.5,
n_cells=4, eps=0.08, 5 load steps, 20 sensor nodes.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F_func

import sys
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.incremental_solver import (
    IncrementalSolver,
    cosine_anneal,
)
from experiments.torus_elastoplastic import return_mapping as rm_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softplus_inv(y: float) -> float:
    if y > 20.0:
        return y
    return math.log(math.exp(y) - 1.0)


def compute_loss(
    fem_solver, mu, K, tau_y_est, H_kin, bc_schedule, bc_mask,
    sensor_nodes, u_obs_list, epsilon,
):
    """Run forward solve and compute displacement-mismatch loss."""
    inc_solver = IncrementalSolver(
        fem_solver, mu, K, tau_y_est, H_kin, epsilon=epsilon,
    )
    u_hist, _ = inc_solver.solve_history(
        bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
    )
    loss = torch.tensor(0.0, device=fem_solver.device, dtype=fem_solver.dtype)
    for u_pred, u_obs in zip(u_hist, u_obs_list):
        u_pred_sensor = u_pred[sensor_nodes]
        diff = u_pred_sensor - u_obs
        u_obs_norm = u_obs.norm().clamp(min=1e-12)
        loss = loss + (diff ** 2).sum() / (u_obs_norm ** 2)
    return loss


# ---------------------------------------------------------------------------
# Monkey-patch for non-smooth (hard max) return mapping
# ---------------------------------------------------------------------------

_original_smooth_return_map = rm_module.smooth_return_map


def hard_max_return_map(F_delta, state, mu, K, tau_y, H_kin, epsilon=1e-2):
    """Return mapping with hard max(Phi, 0) instead of softplus."""
    I = rm_module._eye3(F_delta.device, F_delta.dtype)

    Be_n = state.Be
    ep_bar_n = state.ep_bar
    beta_n = state.beta

    Be_trial = F_delta @ Be_n @ F_delta.transpose(-2, -1)
    eps_e_trial = 0.5 * rm_module.sym_logm(Be_trial)
    tr_eps = torch.diagonal(eps_e_trial, dim1=-2, dim2=-1).sum(-1)
    dev_eps_trial = eps_e_trial - (tr_eps / 3.0)[..., None, None] * I
    tau_dev_trial = 2.0 * mu * dev_eps_trial

    s = tau_dev_trial - beta_n
    s_norm = rm_module.frobenius_norm(s)

    q = torch.sqrt(torch.tensor(1.5, device=s.device, dtype=s.dtype)) * s_norm
    Phi = q - tau_y

    # Hard max instead of softplus -- non-differentiable at Phi=0
    eta = torch.clamp(Phi, min=0.0)
    Delta_gamma = eta / (3.0 * mu + H_kin)

    _safe_norm = torch.where(s_norm > 1e-10, s_norm, torch.ones_like(s_norm))
    N = torch.where(
        (s_norm > 1e-10)[..., None, None],
        s / _safe_norm[..., None, None],
        torch.zeros_like(s),
    )

    sqrt_1p5 = torch.sqrt(torch.tensor(1.5, device=s.device, dtype=s.dtype))
    sqrt_2o3 = torch.sqrt(torch.tensor(2.0 / 3.0, device=s.device, dtype=s.dtype))

    eps_e = eps_e_trial - (Delta_gamma * sqrt_1p5)[..., None, None] * N
    Be_new = rm_module.sym_expm(2.0 * eps_e)
    ep_bar_new = ep_bar_n + sqrt_2o3 * Delta_gamma
    beta_new = beta_n + (2.0 / 3.0) * H_kin * Delta_gamma[..., None, None] * N

    tr_eps_e = torch.diagonal(eps_e, dim1=-2, dim2=-1).sum(-1)
    dev_eps_e = eps_e - (tr_eps_e / 3.0)[..., None, None] * I
    tau = 2.0 * mu * dev_eps_e + (K * tr_eps_e)[..., None, None] * I

    new_state = rm_module.ReturnMappingState(Be=Be_new, ep_bar=ep_bar_new, beta=beta_new)
    return tau, new_state


# ---------------------------------------------------------------------------
# Experiment (a): smooth + autograd
# ---------------------------------------------------------------------------

def run_autograd(
    fem, mu, K, tau_y_true, H_kin, bc_schedule, bc_mask,
    sensor_nodes, u_obs_list, n_iters, lr, eps_start, eps_end,
    tau_y_init, tau_y_min,
):
    device = fem.device
    dtype = fem.dtype
    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam([tau_y_raw], lr=lr)

    history = {"loss": [], "tau_y": [], "tau_y_err_pct": []}
    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()
        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
        eps_cur = cosine_anneal(it - 1, n_iters, eps_start, eps_end)

        loss = compute_loss(
            fem, mu, K, tau_y_est, H_kin, bc_schedule, bc_mask,
            sensor_nodes, u_obs_list, eps_cur,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([tau_y_raw], max_norm=5.0)
        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        history["loss"].append(float(loss.item()))
        history["tau_y"].append(tau_y_val)
        history["tau_y_err_pct"].append(tau_y_err)

        if it <= 5 or it % 20 == 0 or it == n_iters:
            print(f"  [autograd] iter {it:4d} | loss={float(loss.item()):.4e} | "
                  f"tau_y={tau_y_val:.4f} (err={tau_y_err:.2f}%) | eps={eps_cur:.4f}")

    wall_time = time.time() - t0
    return history, wall_time


# ---------------------------------------------------------------------------
# Experiment (b): smooth + finite difference
# ---------------------------------------------------------------------------

def run_fd(
    fem, mu, K, tau_y_true, H_kin, bc_schedule, bc_mask,
    sensor_nodes, u_obs_list, n_iters, lr, eps_start, eps_end,
    tau_y_init, tau_y_min, fd_delta=1e-4,
):
    device = fem.device
    dtype = fem.dtype
    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam([tau_y_raw], lr=lr)

    history = {"loss": [], "tau_y": [], "tau_y_err_pct": []}
    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
        eps_cur = cosine_anneal(it - 1, n_iters, eps_start, eps_end)

        # Central finite difference: perturb tau_y
        with torch.no_grad():
            tau_y_plus = tau_y_est.detach() + fd_delta
            tau_y_minus = tau_y_est.detach() - fd_delta

            loss_plus = compute_loss(
                fem, mu, K, tau_y_plus, H_kin, bc_schedule, bc_mask,
                sensor_nodes, u_obs_list, eps_cur,
            )
            loss_minus = compute_loss(
                fem, mu, K, tau_y_minus, H_kin, bc_schedule, bc_mask,
                sensor_nodes, u_obs_list, eps_cur,
            )
            dloss_dtau_y = (loss_plus - loss_minus) / (2.0 * fd_delta)

            # Also compute loss at current tau_y for logging
            loss_cur = compute_loss(
                fem, mu, K, tau_y_est.detach(), H_kin, bc_schedule, bc_mask,
                sensor_nodes, u_obs_list, eps_cur,
            )

        # Chain rule: dL/d(tau_y_raw) = dL/d(tau_y) * d(tau_y)/d(tau_y_raw)
        # d(tau_y)/d(tau_y_raw) = sigmoid(tau_y_raw)  (derivative of softplus)
        dtau_y_draw = torch.sigmoid(tau_y_raw)
        tau_y_raw.grad = (dloss_dtau_y * dtau_y_draw).detach()

        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        history["loss"].append(float(loss_cur.item()))
        history["tau_y"].append(tau_y_val)
        history["tau_y_err_pct"].append(tau_y_err)

        if it <= 5 or it % 20 == 0 or it == n_iters:
            print(f"  [FD]       iter {it:4d} | loss={float(loss_cur.item()):.4e} | "
                  f"tau_y={tau_y_val:.4f} (err={tau_y_err:.2f}%) | eps={eps_cur:.4f}")

    wall_time = time.time() - t0
    return history, wall_time


# ---------------------------------------------------------------------------
# Experiment (c): non-smooth (hard max) + autograd
# ---------------------------------------------------------------------------

def run_nonsmooth_autograd(
    fem, mu, K, tau_y_true, H_kin, bc_schedule, bc_mask,
    sensor_nodes, u_obs_list, n_iters, lr,
    tau_y_init, tau_y_min,
):
    """Attempt autograd with hard max return mapping. Expect NaN gradients."""
    device = fem.device
    dtype = fem.dtype
    tau_y_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                     device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam([tau_y_raw], lr=lr)

    # Monkey-patch the return mapping to use hard max
    rm_module.smooth_return_map = hard_max_return_map

    nan_count = 0
    grad_values = []
    t0 = time.time()

    try:
        for it in range(1, n_iters + 1):
            optimizer.zero_grad()
            tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
            eps_cur = 0.08  # fixed epsilon (no softplus to anneal since we use hard max)

            loss = compute_loss(
                fem, mu, K, tau_y_est, H_kin, bc_schedule, bc_mask,
                sensor_nodes, u_obs_list, eps_cur,
            )

            try:
                loss.backward()
            except RuntimeError as e:
                print(f"  [non-smooth] iter {it}: backward() raised: {e}")
                nan_count += 1
                grad_values.append(None)
                continue

            grad_val = tau_y_raw.grad
            if grad_val is None:
                nan_count += 1
                grad_values.append(None)
                status = "None"
            elif not torch.isfinite(grad_val):
                nan_count += 1
                grad_values.append(float(grad_val.item()))
                status = f"NaN/Inf ({float(grad_val.item())})"
            else:
                grad_values.append(float(grad_val.item()))
                status = f"{float(grad_val.item()):.4e}"

            if it <= 10 or it == n_iters:
                print(f"  [non-smooth] iter {it:4d} | loss={float(loss.item()):.4e} | "
                      f"grad={status} | tau_y={float(tau_y_est.item()):.4f}")

            optimizer.step()
    finally:
        # Restore original smooth return mapping
        rm_module.smooth_return_map = _original_smooth_return_map

    wall_time = time.time() - t0
    return nan_count, grad_values, wall_time


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 72)
    print("Finite-Difference vs Autograd Comparison for tau_y Identification")
    print("=" * 72)

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

    # --- BCs ---
    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r
    left_face = (nodes[:, 0] < -r + tol_face)
    right_face = (nodes[:, 0] > r - tol_face)
    bc_mask = left_face | right_face

    eps_max = 0.08
    n_steps = 5
    bc_schedule = []
    for step in range(n_steps):
        lam = (step + 1) / n_steps
        u_bc = torch.zeros_like(nodes)
        u_bc[right_face, 0] = eps_max * lam * 2.0 * r
        bc_schedule.append(u_bc)

    # --- Sensors ---
    free_indices = torch.where(~bc_mask)[0]
    n_sensors = min(20, free_indices.shape[0])
    sensor_nodes = free_indices[:n_sensors]

    print(f"\n  Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")
    print(f"  BC nodes: {bc_mask.sum().item()}, Sensors: {n_sensors}")
    print(f"  Load: {n_steps} steps, eps_max={eps_max}")
    print(f"  True tau_y = {tau_y_true}")

    # --- Synthetic data ---
    print("\n  Generating synthetic data...")
    from experiments.torus_elastoplastic.inverse_perfect_plasticity import (
        generate_synthetic_data,
    )
    u_obs_list = generate_synthetic_data(
        fem, mu, K, tau_y_true_t, H_kin_zero,
        bc_schedule, bc_mask, sensor_nodes,
        epsilon=1e-3, noise_std=0.0,
    )
    print(f"  Done: {len(u_obs_list)} steps x {sensor_nodes.shape[0]} sensors")

    # --- Common settings ---
    n_iters = 100
    lr = 5e-2
    eps_start = 0.08
    eps_end = 0.08   # fixed epsilon for fair comparison
    tau_y_init = 1.0
    tau_y_min = 0.01

    # ==================================================
    # (a) Smooth + autograd
    # ==================================================
    print("\n" + "=" * 72)
    print("(a) Smooth return mapping + AUTOGRAD")
    print("=" * 72)
    hist_ag, time_ag = run_autograd(
        fem, mu, K, tau_y_true, H_kin_zero, bc_schedule, bc_mask,
        sensor_nodes, u_obs_list, n_iters, lr, eps_start, eps_end,
        tau_y_init, tau_y_min,
    )

    # ==================================================
    # (b) Smooth + finite difference
    # ==================================================
    print("\n" + "=" * 72)
    print("(b) Smooth return mapping + FINITE DIFFERENCE")
    print("=" * 72)
    hist_fd, time_fd = run_fd(
        fem, mu, K, tau_y_true, H_kin_zero, bc_schedule, bc_mask,
        sensor_nodes, u_obs_list, n_iters, lr, eps_start, eps_end,
        tau_y_init, tau_y_min, fd_delta=1e-4,
    )

    # ==================================================
    # (c) Non-smooth (hard max) + autograd
    # ==================================================
    print("\n" + "=" * 72)
    print("(c) NON-SMOOTH return mapping (hard max) + autograd")
    print("=" * 72)
    nan_count, grad_vals, time_ns = run_nonsmooth_autograd(
        fem, mu, K, tau_y_true, H_kin_zero, bc_schedule, bc_mask,
        sensor_nodes, u_obs_list, min(n_iters, 10), lr,  # only 10 iters
        tau_y_init, tau_y_min,
    )

    # ==================================================
    # Summary
    # ==================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    final_ag = hist_ag["tau_y"][-1]
    err_ag = hist_ag["tau_y_err_pct"][-1]
    final_fd = hist_fd["tau_y"][-1]
    err_fd = hist_fd["tau_y_err_pct"][-1]

    print(f"\n  (a) Autograd:          tau_y={final_ag:.4f}  error={err_ag:.2f}%  "
          f"time={time_ag:.1f}s")
    print(f"  (b) Finite difference: tau_y={final_fd:.4f}  error={err_fd:.2f}%  "
          f"time={time_fd:.1f}s")
    print(f"  (c) Non-smooth+AG:     NaN/None grads in {nan_count}/{min(n_iters,10)} iters  "
          f"time={time_ns:.1f}s")
    print(f"\n  Autograd speedup over FD: {time_fd/max(time_ag,0.01):.1f}x")

    # --- Save results ---
    output_dir = Path(_REPO_ROOT) / "output_figures"
    output_dir.mkdir(exist_ok=True)
    results = {
        "problem": {
            "E": E_val, "nu": nu_val, "tau_y_true": tau_y_true,
            "n_cells": n_cells, "eps_max": eps_max, "n_steps": n_steps,
            "n_sensors": n_sensors, "n_iters": n_iters, "lr": lr,
            "eps_start": eps_start, "eps_end": eps_end,
            "tau_y_init": tau_y_init, "fd_delta": 1e-4,
        },
        "autograd": {
            "final_tau_y": final_ag,
            "error_pct": err_ag,
            "wall_time_s": round(time_ag, 2),
            "loss_history": hist_ag["loss"],
            "tau_y_history": hist_ag["tau_y"],
            "err_history": hist_ag["tau_y_err_pct"],
        },
        "finite_difference": {
            "final_tau_y": final_fd,
            "error_pct": err_fd,
            "wall_time_s": round(time_fd, 2),
            "loss_history": hist_fd["loss"],
            "tau_y_history": hist_fd["tau_y"],
            "err_history": hist_fd["tau_y_err_pct"],
        },
        "nonsmooth_autograd": {
            "nan_or_none_grad_count": nan_count,
            "total_iters": min(n_iters, 10),
            "wall_time_s": round(time_ns, 2),
            "grad_values": [
                g if g is not None and math.isfinite(g) else None
                for g in grad_vals
            ],
        },
    }

    out_path = output_dir / "fig_fd_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("=" * 72)
