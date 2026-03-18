#!/usr/bin/env python3
"""Generate data for CMAME manuscript figures on elastoplasticity inverse problem.

Produces JSON data files in output_figures/ for:
1. Smoothing trick: stress-strain curves for different epsilon values
2. Initial guess sensitivity: tau_y convergence from multiple starting points
3. Mesh refinement: inverse accuracy vs n_cells
4. Newton convergence: residual norm per iteration
5. Cyclic hysteresis: stress-strain loops with kinematic hardening

Run:  python experiments/torus_elastoplastic/generate_manuscript_figures.py
"""

from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

import torch
torch.set_default_dtype(torch.float64)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.return_mapping import (
    ReturnMappingState, smooth_return_map,
)
from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.incremental_solver import IncrementalSolver, cosine_anneal

DEVICE = "cpu"
OUT_DIR = "output_figures"
os.makedirs(OUT_DIR, exist_ok=True)

E_VAL, NU_VAL = 200.0, 0.3
MU_VAL = E_VAL / (2 * (1 + NU_VAL))
K_VAL = E_VAL / (3 * (1 - 2 * NU_VAL))


def _uniaxial_F(eps):
    lam = 1.0 + eps
    lat = 1.0 / math.sqrt(lam)
    return torch.diag(torch.tensor([lam, lat, lat], device=DEVICE))


# =====================================================================
# Figure 1: Smoothing trick — stress-strain for different epsilon
# =====================================================================
def generate_smoothing_comparison():
    print("=== Figure 1: Smoothing trick comparison ===")
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y = torch.tensor(0.4)
    H_kin = torch.tensor(0.0)

    epsilons = [0.1, 0.01, 0.001]
    dep = 0.001
    n_steps = 120
    results = {}

    for eps_val in epsilons:
        state = ReturnMappingState.zeros((), device=DEVICE)
        strains, stresses_ax = [], []
        total_strain = 0.0
        for i in range(n_steps):
            F_inc = _uniaxial_F(dep)
            tau, state = smooth_return_map(F_inc, state, mu, K, tau_y, H_kin, epsilon=eps_val)
            total_strain += dep
            strains.append(total_strain)
            stresses_ax.append(tau[0, 0].item())

        results[f"eps={eps_val}"] = {"strain": strains, "stress": stresses_ax}
        print(f"  eps={eps_val}: final stress={stresses_ax[-1]:.4f}")

    with open(f"{OUT_DIR}/fig1_smoothing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig1_smoothing.json\n")


# =====================================================================
# Figure 2: Initial guess sensitivity for tau_y
# =====================================================================
def generate_initial_guess_sensitivity():
    print("=== Figure 2: Initial guess sensitivity ===")
    import torch.nn.functional as F_func

    fem = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device=DEVICE)
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y_true = 0.5

    tol_f = fem.h * 0.1
    r = fem.r
    left = fem.nodes[:, 0] < -r + tol_f
    right = fem.nodes[:, 0] > r - tol_f
    bc_mask = left | right

    bc_schedule = []
    for step in range(5):
        lam = (step + 1) / 5
        u_bc = torch.zeros_like(fem.nodes)
        u_bc[right, 0] = 0.08 * lam * 2.0 * r
        bc_schedule.append(u_bc)

    sensor = torch.where(~bc_mask)[0][:20]

    # Generate observations
    with torch.no_grad():
        u_obs, _ = IncrementalSolver(
            fem, mu, K, torch.tensor(tau_y_true), torch.tensor(0.0), epsilon=1e-3
        ).solve_history(bc_schedule, bc_mask, verbose=False)

    initial_guesses = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    results = {}

    for tau_y_init in initial_guesses:
        print(f"  Running tau_y_init={tau_y_init}...", end="", flush=True)
        raw = torch.nn.Parameter(torch.tensor(
            math.log(math.exp(max(tau_y_init - 0.01, 0.01)) - 1.0)
        ))
        opt = torch.optim.Adam([raw], lr=5e-2)
        trajectory = []

        for it in range(200):
            opt.zero_grad()
            tau_y_est = F_func.softplus(raw) + 0.01
            eps_c = cosine_anneal(it, 200, 0.1, 0.001)
            u_pred, _ = IncrementalSolver(
                fem, mu, K, tau_y_est, torch.tensor(0.0), epsilon=eps_c
            ).solve_history(bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8)

            loss = sum(torch.sum((up[sensor] - uo[sensor].detach())**2) for up, uo in zip(u_pred, u_obs))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 5.0)
            opt.step()
            trajectory.append(tau_y_est.item())

        results[f"init={tau_y_init}"] = trajectory
        print(f" final={trajectory[-1]:.4f}")

    with open(f"{OUT_DIR}/fig2_initial_guess.json", "w") as f:
        json.dump({"true_tau_y": tau_y_true, "trajectories": results}, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig2_initial_guess.json\n")


# =====================================================================
# Figure 3: Mesh refinement study
# =====================================================================
def generate_mesh_refinement():
    print("=== Figure 3: Mesh refinement ===")
    import torch.nn.functional as F_func

    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y_true = 0.5

    n_cells_list = [3, 4, 6, 8]
    results = {"n_cells": [], "n_nodes": [], "n_elements": [],
               "best_tau_y_err": [], "final_tau_y": []}

    for nc in n_cells_list:
        print(f"  n_cells={nc}...", end="", flush=True)
        fem = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, device=DEVICE)

        tol_f = fem.h * 0.1
        r = fem.r
        left = fem.nodes[:, 0] < -r + tol_f
        right = fem.nodes[:, 0] > r - tol_f
        bc_mask = left | right
        sensor = torch.where(~bc_mask)[0][:min(30, (~bc_mask).sum().item())]

        # Scale load steps with mesh size to keep strain increment constant
        n_steps = max(5, nc * 2)  # finer mesh → more steps
        bc_schedule = []
        for step in range(n_steps):
            lam = (step + 1) / n_steps
            u_bc = torch.zeros_like(fem.nodes)
            u_bc[right, 0] = 0.08 * lam * 2.0 * r
            bc_schedule.append(u_bc)

        with torch.no_grad():
            u_obs, _ = IncrementalSolver(
                fem, mu, K, torch.tensor(tau_y_true), torch.tensor(0.0), epsilon=1e-3
            ).solve_history(bc_schedule, bc_mask, verbose=False)

        raw = torch.nn.Parameter(torch.tensor(0.0))  # init ~0.69
        opt = torch.optim.Adam([raw], lr=5e-2)
        best_err = 100.0

        for it in range(150):
            opt.zero_grad()
            tau_y_est = F_func.softplus(raw) + 0.01
            eps_c = cosine_anneal(it, 150, 0.1, 0.001)
            u_pred, _ = IncrementalSolver(
                fem, mu, K, tau_y_est, torch.tensor(0.0), epsilon=eps_c
            ).solve_history(bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8)

            loss = sum(torch.sum((up[sensor] - uo[sensor].detach())**2) for up, uo in zip(u_pred, u_obs))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 5.0)
            opt.step()
            err = abs(tau_y_est.item() - tau_y_true) / tau_y_true * 100
            best_err = min(best_err, err)

        final = (F_func.softplus(raw) + 0.01).item()
        results["n_cells"].append(nc)
        results["n_nodes"].append(fem.n_nodes)
        results["n_elements"].append(fem.n_elements)
        results["best_tau_y_err"].append(best_err)
        results["final_tau_y"].append(final)
        print(f" nodes={fem.n_nodes}, best_err={best_err:.2f}%, final={final:.4f}")

    with open(f"{OUT_DIR}/fig3_refinement.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig3_refinement.json\n")


# =====================================================================
# Figure 4: Newton convergence rate
# =====================================================================
def generate_newton_convergence():
    print("=== Figure 4: Newton convergence ===")
    fem = ChartVectorFEMSolver(n_cells=6, support_r=1.0, device=DEVICE)
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y = torch.tensor(0.5)
    H_kin = torch.tensor(10.0)

    tol_f = fem.h * 0.1
    r = fem.r
    left = fem.nodes[:, 0] < -r + tol_f
    right = fem.nodes[:, 0] > r - tol_f
    bc_mask = left | right

    u_bc = torch.zeros_like(fem.nodes)
    u_bc[right, 0] = 0.08 * 2.0 * r

    from experiments.torus_elastoplastic.incremental_solver import (
        ElastoplasticStepSolver, _incremental_F,
    )

    I33 = torch.eye(3)
    F_old = I33.unsqueeze(0).expand(fem.n_elements, 3, 3).clone()
    state = ReturnMappingState.zeros((fem.n_elements,))

    step_solver = ElastoplasticStepSolver(fem, mu, K, tau_y, H_kin, epsilon=1e-3)

    # Manually run Newton and record residuals
    u = torch.zeros(fem.n_nodes, 3)
    u[bc_mask] = u_bc[bc_mask]
    bc_dof = bc_mask.unsqueeze(1).expand(-1, 3).reshape(-1)
    free = ~bc_dof
    f_ext = torch.zeros(fem.n_nodes, 3)

    residuals = []
    for it in range(15):
        R, P, trial = step_solver._compute_residual(u, F_old, state, f_ext)
        R[bc_dof] = 0.0
        res = torch.norm(R[free]).item()
        residuals.append(res)
        if res < 1e-12:
            break

        F_cur = fem.compute_F(u.detach())
        C_ep = step_solver._element_tangent_dPdF(F_cur, F_old, state)
        K_tan = fem.tangent_stiffness(u.detach(), lambda F: C_ep)
        K_tan[bc_dof, :] = 0.0
        K_tan[:, bc_dof] = 0.0
        K_tan[bc_dof, bc_dof] = 1.0
        R[bc_dof] = 0.0
        du = torch.linalg.solve(K_tan, -R.detach())
        u = u + du.reshape(-1, 3)

    # Compute convergence rates
    rates = []
    for i in range(2, len(residuals)):
        if residuals[i-1] > 1e-15 and residuals[i-2] > 1e-15:
            r = math.log(residuals[i] / residuals[i-1]) / math.log(residuals[i-1] / residuals[i-2])
            rates.append(r)
        else:
            rates.append(float('nan'))

    print(f"  Residuals: {[f'{r:.2e}' for r in residuals]}")
    print(f"  Rates: {[f'{r:.2f}' if not math.isnan(r) else 'nan' for r in rates]}")

    with open(f"{OUT_DIR}/fig4_newton.json", "w") as f:
        json.dump({"residuals": residuals, "rates": rates}, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig4_newton.json\n")


# =====================================================================
# Figure 5: Cyclic hysteresis loops
# =====================================================================
def generate_hysteresis():
    print("=== Figure 5: Cyclic hysteresis ===")
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y = torch.tensor(0.5)

    dep = 0.002
    n_half = 60

    for H_kin_val, label in [(0.0, "perfect"), (10.0, "H10"), (30.0, "H30")]:
        H_kin = torch.tensor(H_kin_val)
        state = ReturnMappingState.zeros(())
        strains, dev_stresses = [], []
        total = 0.0

        # Forward
        for _ in range(n_half):
            F_inc = _uniaxial_F(dep)
            tau, state = smooth_return_map(F_inc, state, mu, K, tau_y, H_kin, epsilon=1e-3)
            total += dep
            tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
            strains.append(total)
            dev_stresses.append(tau[0, 0].item() - tr / 3.0)

        # Reverse
        for _ in range(2 * n_half):
            F_inc = _uniaxial_F(-dep)
            tau, state = smooth_return_map(F_inc, state, mu, K, tau_y, H_kin, epsilon=1e-3)
            total -= dep
            tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
            strains.append(total)
            dev_stresses.append(tau[0, 0].item() - tr / 3.0)

        # Forward again
        for _ in range(n_half):
            F_inc = _uniaxial_F(dep)
            tau, state = smooth_return_map(F_inc, state, mu, K, tau_y, H_kin, epsilon=1e-3)
            total += dep
            tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
            strains.append(total)
            dev_stresses.append(tau[0, 0].item() - tr / 3.0)

        with open(f"{OUT_DIR}/fig5_hysteresis_{label}.json", "w") as f:
            json.dump({"H_kin": H_kin_val, "strain": strains, "dev_stress": dev_stresses}, f, indent=2)
        print(f"  {label}: {len(strains)} points")

    print(f"  Saved to {OUT_DIR}/fig5_hysteresis_*.json\n")


# =====================================================================
# Figure 6: Noise sensitivity for tau_y identification
# =====================================================================
def generate_noise_sensitivity():
    print("=== Figure 6: Noise sensitivity ===")
    import torch.nn.functional as F_func

    fem = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device=DEVICE)
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y_true = 0.5

    tol_f = fem.h * 0.1
    r = fem.r
    left = fem.nodes[:, 0] < -r + tol_f
    right = fem.nodes[:, 0] > r - tol_f
    bc_mask = left | right
    sensor = torch.where(~bc_mask)[0][:20]

    bc_schedule = []
    for step in range(5):
        lam = (step + 1) / 5
        u_bc = torch.zeros_like(fem.nodes)
        u_bc[right, 0] = 0.08 * lam * 2.0 * r
        bc_schedule.append(u_bc)

    # Clean observations
    with torch.no_grad():
        u_obs_clean, _ = IncrementalSolver(
            fem, mu, K, torch.tensor(tau_y_true), torch.tensor(0.0), epsilon=1e-3
        ).solve_history(bc_schedule, bc_mask, verbose=False)

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    results = {"noise_std": [], "best_tau_y_err": [], "final_tau_y": [],
               "n_trials": 3}

    for noise_std in noise_levels:
        trial_errs = []
        for trial in range(3):
            # Add noise
            torch.manual_seed(42 + trial)
            u_obs = []
            for u in u_obs_clean:
                u_s = u[sensor].clone()
                if noise_std > 0:
                    u_max = u_s.abs().max().clamp(min=1e-30)
                    u_s = u_s + noise_std * u_max * torch.randn_like(u_s)
                u_obs.append(u_s)

            raw = torch.nn.Parameter(torch.tensor(0.0))
            opt = torch.optim.Adam([raw], lr=5e-2)
            best_err = 100.0
            for it in range(200):
                opt.zero_grad()
                tau_y_est = F_func.softplus(raw) + 0.01
                eps_c = cosine_anneal(it, 200, 0.1, 0.001)
                u_pred, _ = IncrementalSolver(
                    fem, mu, K, tau_y_est, torch.tensor(0.0), epsilon=eps_c
                ).solve_history(bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8)
                loss = sum(torch.sum((up[sensor] - uo.detach())**2)
                           for up, uo in zip(u_pred, u_obs))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([raw], 5.0)
                opt.step()
                err = abs(tau_y_est.item() - tau_y_true) / tau_y_true * 100
                best_err = min(best_err, err)
            trial_errs.append(best_err)

        avg_err = sum(trial_errs) / len(trial_errs)
        final = (F_func.softplus(raw) + 0.01).item()
        results["noise_std"].append(noise_std)
        results["best_tau_y_err"].append(trial_errs)
        results["final_tau_y"].append(final)
        print(f"  noise={noise_std}: errs={[f'{e:.2f}%' for e in trial_errs]}, avg={avg_err:.2f}%")

    with open(f"{OUT_DIR}/fig6_noise.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig6_noise.json\n")


# =====================================================================
# Figure 7: Epsilon sensitivity for inverse
# =====================================================================
def generate_epsilon_sensitivity():
    print("=== Figure 7: Epsilon sensitivity ===")
    import torch.nn.functional as F_func

    fem = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device=DEVICE)
    mu = torch.tensor(MU_VAL)
    K = torch.tensor(K_VAL)
    tau_y_true = 0.5

    tol_f = fem.h * 0.1
    r = fem.r
    left = fem.nodes[:, 0] < -r + tol_f
    right = fem.nodes[:, 0] > r - tol_f
    bc_mask = left | right
    sensor = torch.where(~bc_mask)[0][:20]

    bc_schedule = []
    for step in range(5):
        lam = (step + 1) / 5
        u_bc = torch.zeros_like(fem.nodes)
        u_bc[right, 0] = 0.08 * lam * 2.0 * r
        bc_schedule.append(u_bc)

    with torch.no_grad():
        u_obs, _ = IncrementalSolver(
            fem, mu, K, torch.tensor(tau_y_true), torch.tensor(0.0), epsilon=1e-3
        ).solve_history(bc_schedule, bc_mask, verbose=False)

    eps_starts = [1.0, 0.5, 0.1, 0.05, 0.01]
    results = {}
    for eps_s in eps_starts:
        raw = torch.nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam([raw], lr=5e-2)
        trajectory = []
        for it in range(200):
            opt.zero_grad()
            tau_y_est = F_func.softplus(raw) + 0.01
            eps_c = cosine_anneal(it, 200, eps_s, 0.001)
            u_pred, _ = IncrementalSolver(
                fem, mu, K, tau_y_est, torch.tensor(0.0), epsilon=eps_c
            ).solve_history(bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8)
            loss = sum(torch.sum((up[sensor] - uo[sensor].detach())**2)
                       for up, uo in zip(u_pred, u_obs))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 5.0)
            opt.step()
            trajectory.append(tau_y_est.item())
        results[f"eps_start={eps_s}"] = trajectory
        print(f"  eps_start={eps_s}: final={trajectory[-1]:.4f}")

    with open(f"{OUT_DIR}/fig7_epsilon.json", "w") as f:
        json.dump({"true_tau_y": tau_y_true, "trajectories": results}, f, indent=2)
    print(f"  Saved to {OUT_DIR}/fig7_epsilon.json\n")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific figure (e.g. '3' or '6')")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Manuscript Figure Data Generation")
    print("=" * 60)
    print()

    figs = {
        "1": ("Smoothing comparison", generate_smoothing_comparison),
        "4": ("Newton convergence", generate_newton_convergence),
        "5": ("Hysteresis loops", generate_hysteresis),
        "2": ("Initial guess sensitivity", generate_initial_guess_sensitivity),
        "3": ("Mesh refinement", generate_mesh_refinement),
        "6": ("Noise sensitivity", generate_noise_sensitivity),
        "7": ("Epsilon sensitivity", generate_epsilon_sensitivity),
    }

    if args.only:
        if args.only in figs:
            figs[args.only][1]()
        else:
            print(f"Unknown figure: {args.only}. Available: {list(figs.keys())}")
    else:
        for key, (name, fn) in figs.items():
            fn()

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s")
    print(f"Output directory: {OUT_DIR}/")
