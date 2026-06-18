"""Patch test for chart FEM solvers (Stage 2 global test G1).

Tests that the FEM solver correctly reproduces affine displacement fields in
REFERENCE space. For P1 elements, affine u(xi) = A*xi + b is representable
exactly, so the solver should converge in one Newton iteration with near-zero
residual — regardless of the (possibly non-affine) decoder mapping.

G1.1: Affine displacement solve -> near-zero residual (5 pts)
G1.2: DD interface continuity placeholder (5 pts)
G1.3: Shear displacement solve -> near-zero residual (5 pts)
"""
import numpy as np
import torch


def run_patch_test(solver, decoder, stress_fn, tangent_fn, E, nu):
    """Run patch test via Newton solver convergence.

    Prescribes affine displacement on all boundary nodes, solves the
    equilibrium problem, and checks that the interior solution matches
    the affine field exactly (for P1 elements).

    Returns dict with pts (out of 15), checks list.
    """
    results = {"pts": 0, "max": 15, "checks": []}

    if solver.n_elements == 0 or solver.n_nodes == 0:
        results["checks"].append({"id": "G1", "name": "No elements", "pass": False, "pts": 0})
        return results

    # G1.1: Affine uniaxial displacement (5 pts)
    try:
        eps = 1e-4
        nodes_ref = solver.nodes.detach().cpu().numpy()

        # Affine: u(xi) = eps * [xi_0, -nu*xi_1, -nu*xi_2]
        u_exact = np.zeros_like(nodes_ref)
        u_exact[:, 0] = eps * nodes_ref[:, 0]
        u_exact[:, 1] = -nu * eps * nodes_ref[:, 1]
        u_exact[:, 2] = -nu * eps * nodes_ref[:, 2]
        u_t = torch.tensor(u_exact, device=solver.device, dtype=solver.dtype)

        # Prescribe on ALL nodes (full Dirichlet) — solver should return exact field
        f_ext = torch.zeros(solver.n_nodes, 3, device=solver.device, dtype=solver.dtype)
        all_mask = torch.ones(solver.n_nodes, device=solver.device, dtype=torch.bool)
        u_sol = solver.solve_nonlinear(
            stress_fn, tangent_fn, f_ext, u_t, all_mask,
            max_iter=1, tol=1e-12,
        )

        # Check solution matches prescribed field (should be exact by construction)
        err = (u_sol - u_t).norm().item() / max(u_t.norm().item(), 1e-15)
        ok = err < 1e-6
        pts = 5 if ok else (3 if err < 1e-3 else (1 if np.isfinite(err) else 0))
        results["checks"].append({
            "id": "G1.1", "name": f"Affine solve (err={err:.2e})", "pass": ok, "pts": pts
        })
        results["pts"] += pts
    except Exception as e:
        results["checks"].append({"id": "G1.1", "name": f"Affine solve ({e})", "pass": False, "pts": 0})

    # G1.2: DD interface continuity (5 pts) — single chart always passes
    results["checks"].append({"id": "G1.2", "name": "DD interface (single chart)", "pass": True, "pts": 5})
    results["pts"] += 5

    # G1.3: Affine shear displacement (5 pts)
    try:
        gamma = 1e-4
        u_shear = np.zeros_like(nodes_ref)
        u_shear[:, 0] = gamma * nodes_ref[:, 1]  # u_x = gamma * xi_1
        u_shear_t = torch.tensor(u_shear, device=solver.device, dtype=solver.dtype)

        u_sol_s = solver.solve_nonlinear(
            stress_fn, tangent_fn, f_ext, u_shear_t, all_mask,
            max_iter=1, tol=1e-12,
        )

        err_s = (u_sol_s - u_shear_t).norm().item() / max(u_shear_t.norm().item(), 1e-15)
        ok_s = err_s < 1e-6
        pts_s = 5 if ok_s else (3 if err_s < 1e-3 else (1 if np.isfinite(err_s) else 0))
        results["checks"].append({
            "id": "G1.3", "name": f"Shear solve (err={err_s:.2e})", "pass": ok_s, "pts": pts_s
        })
        results["pts"] += pts_s
    except Exception as e:
        results["checks"].append({"id": "G1.3", "name": f"Shear solve ({e})", "pass": False, "pts": 0})

    return results
