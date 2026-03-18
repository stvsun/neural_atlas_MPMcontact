#!/usr/bin/env python3
"""Generate recovered-vs-true hysteresis overlay data for the torus elastoplastic
inverse problem.

Runs two forward cyclic simulations (true and recovered parameters) and records
the (strain, stress) hysteresis at element 0.  Saves to JSON for plotting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.return_mapping import smooth_return_map, ReturnMappingState
from experiments.torus_elastoplastic.incremental_solver import IncrementalSolver, ElastoplasticStepSolver


def run_forward_hysteresis(
    fem: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    bc_schedule: list,
    bc_mask: torch.Tensor,
    epsilon: float = 1e-3,
    elem_idx: int = 0,
) -> tuple:
    """Run forward cyclic simulation and record (strain_xx, stress_xx) at elem_idx.

    Returns (strain_list, stress_list) as plain Python lists of floats.
    """
    N = fem.n_nodes
    M = fem.n_elements
    device = fem.device
    dtype = fem.dtype

    u = torch.zeros(N, 3, device=device, dtype=dtype)
    I33 = torch.eye(3, device=device, dtype=dtype)
    F_old = I33.unsqueeze(0).expand(M, 3, 3).clone()
    state = ReturnMappingState.zeros((M,), device=device, dtype=dtype)

    strain_list = []
    stress_list = []

    with torch.no_grad():
        for step_idx, u_bc in enumerate(bc_schedule):
            step_solver = ElastoplasticStepSolver(
                fem, mu, K, tau_y, H_kin, epsilon,
            )
            u, new_state, F_new = step_solver.solve_step(
                u, state, F_old, u_bc, bc_mask,
                max_iter=50, tol=1e-6,
            )

            # Compute stress at element via return mapping
            F_total = fem.compute_F(u)
            F_delta = F_total @ torch.linalg.inv(F_old)
            tau_stress, _ = smooth_return_map(
                F_delta, state, mu, K, tau_y, H_kin, epsilon,
            )

            # Extract strain_xx and stress_xx (Kirchhoff) at element elem_idx
            # Strain: Green-Lagrange-like measure from F: E = 0.5*(F^T F - I)
            # For hysteresis plot, use engineering strain = F[0,0] - 1
            F_elem = F_total[elem_idx]
            strain_xx = float(F_elem[0, 0].item() - 1.0)
            stress_xx = float(tau_stress[elem_idx, 0, 0].item())

            strain_list.append(strain_xx)
            stress_list.append(stress_xx)

            state = new_state
            F_old = F_new.detach()

    return strain_list, stress_list


def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # --- Material parameters ---
    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)

    # True and recovered parameters
    tau_y_true = torch.tensor(0.5, device=device)
    H_kin_true = torch.tensor(20.0, device=device)
    tau_y_rec = torch.tensor(0.4988, device=device)
    H_kin_rec = torch.tensor(19.58, device=device)

    # --- Mesh ---
    n_cells = 4
    fem = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, device=device, dtype=torch.float64,
    )

    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r

    left_face = (nodes[:, 0] < -r + tol_face)
    right_face = (nodes[:, 0] > r - tol_face)
    bc_mask = left_face | right_face

    # --- Cyclic loading: 2 cycles, eps_peak=0.03, 15 steps per half-cycle ---
    eps_peak = 0.03
    n_steps_per_half = 15

    bc_schedule = []
    for cycle in range(2):
        # Forward: 0 -> +eps
        for step in range(n_steps_per_half):
            lam = (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)
        # Reverse: +eps -> -eps
        for step in range(2 * n_steps_per_half):
            lam = 1.0 - (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)
        # Return: -eps -> 0
        for step in range(n_steps_per_half):
            lam = -1.0 + (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)

    total_steps = len(bc_schedule)
    print(f"Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")
    print(f"Cyclic loading: 2 cycles, {n_steps_per_half} steps/half, "
          f"eps_peak={eps_peak}, total={total_steps} steps")

    # --- Run TRUE forward simulation ---
    print("\nRunning forward simulation with TRUE parameters "
          f"(tau_y={float(tau_y_true)}, H_kin={float(H_kin_true)})...")
    true_strain, true_stress = run_forward_hysteresis(
        fem, mu, K, tau_y_true, H_kin_true, bc_schedule, bc_mask,
        epsilon=1e-3, elem_idx=0,
    )
    print(f"  Done. {len(true_strain)} data points.")

    # --- Run RECOVERED forward simulation ---
    print(f"\nRunning forward simulation with RECOVERED parameters "
          f"(tau_y={float(tau_y_rec)}, H_kin={float(H_kin_rec)})...")
    rec_strain, rec_stress = run_forward_hysteresis(
        fem, mu, K, tau_y_rec, H_kin_rec, bc_schedule, bc_mask,
        epsilon=1e-3, elem_idx=0,
    )
    print(f"  Done. {len(rec_strain)} data points.")

    # --- Save to JSON ---
    out_dir = Path(_REPO_ROOT) / "output_figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_hysteresis_comparison.json"

    data = {
        "true_strain": true_strain,
        "true_stress": true_stress,
        "recovered_strain": rec_strain,
        "recovered_stress": rec_stress,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved hysteresis data to {out_path}")
    print(f"  True strain range: [{min(true_strain):.6f}, {max(true_strain):.6f}]")
    print(f"  True stress range: [{min(true_stress):.6f}, {max(true_stress):.6f}]")
    print(f"  Recovered strain range: [{min(rec_strain):.6f}, {max(rec_strain):.6f}]")
    print(f"  Recovered stress range: [{min(rec_stress):.6f}, {max(rec_stress):.6f}]")


if __name__ == "__main__":
    main()
