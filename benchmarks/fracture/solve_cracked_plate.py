"""Single-chart elasticity solve on a cracked plate.

Uses ChartVectorFEMSolver with the cracked plate SDF oracle to solve
linear elasticity. Williams displacement BCs are imposed on the boundary
so the numerical solution should recover the analytical K_I.

This is the Step 2 building block: prove the FEM solver can produce
correct near-tip fields on SDF-filtered crack geometry.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.linear_elastic import make_linear_elastic
from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
from benchmarks.fracture.lefm_reference import (
    stress_intensity_factor,
    williams_displacement,
    extract_K_I_from_displacement,
)


def solve_cracked_plate_single_chart(
    a: float = 0.3,
    W: float = 1.0,
    H: float = 4.0,
    T: float = 0.5,
    sigma_inf: float = 1.0,
    E: float = 200.0,
    nu: float = 0.3,
    n_cells: int = 16,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict:
    """Solve linear elasticity on a cracked plate and extract K_I.

    Parameters
    ----------
    a : float
        Crack length.
    W : float
        Plate half-width.
    n_cells : int
        Mesh cells per axis.

    Returns
    -------
    result : dict
        K_I_numerical, K_I_analytical, relative_error, u (displacement).
    """
    # SDF oracle
    sdf_oracle = CrackedPlateSDFOracle(a=a, W=W, H=H, T=T, delta=0.02)
    crack_tip = np.array([-W + a, 0.0, 0.0])

    # Analytical K_I
    K_I_exact = stress_intensity_factor(sigma_inf, a, W)

    # Build FEM solver — use the plate extent as support radius
    extent = max(W, H / 2, T / 2) * 1.2
    solver = ChartVectorFEMSolver(
        n_cells=n_cells,
        support_r=extent,
        sdf_oracle=sdf_oracle,
        sdf_threshold=-0.005,
        mesh_extent=1.0,
        device=device,
        dtype=dtype,
    )

    if solver.n_nodes == 0:
        return {"K_I_numerical": float("nan"), "K_I_analytical": K_I_exact,
                "relative_error": float("nan"), "n_nodes": 0}

    # Material
    stress_fn, tangent_fn = make_linear_elastic(E, nu, device=device, dtype=dtype)

    # Boundary conditions: Williams displacement on all boundary nodes
    bc_mask = solver.boundary_mask.clone()
    u_bc = torch.zeros(solver.n_nodes, 3, device=device, dtype=dtype)

    if bc_mask.any():
        bc_nodes_xi = solver.nodes[bc_mask].detach().cpu().numpy()

        # Map to physical space (identity decoder: xi = x_phys for this test)
        bc_nodes_phys = bc_nodes_xi

        # Distance and angle from crack tip
        dx = bc_nodes_phys[:, 0] - crack_tip[0]
        dy = bc_nodes_phys[:, 1] - crack_tip[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        u_x, u_y = williams_displacement(r, theta, K_I_exact, E, nu, plane_strain=True)
        u_bc_np = np.zeros_like(bc_nodes_phys)
        u_bc_np[:, 0] = u_x
        u_bc_np[:, 1] = u_y
        u_bc[bc_mask] = torch.tensor(u_bc_np, device=device, dtype=dtype)

    # External force = 0 (all loading via BCs)
    f_ext = torch.zeros(solver.n_nodes, 3, device=device, dtype=dtype)

    # Solve
    u = solver.solve_nonlinear(
        stress_fn=stress_fn,
        tangent_fn=tangent_fn,
        f_ext=f_ext,
        u_bc=u_bc,
        bc_mask=bc_mask,
        max_iter=20,
        tol=1e-8,
    )

    # Extract K_I from numerical displacement
    u_np = u.detach().cpu().numpy()
    nodes_np = solver.nodes.detach().cpu().numpy()

    dx = nodes_np[:, 0] - crack_tip[0]
    dy = nodes_np[:, 1] - crack_tip[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Use interior nodes in annular region
    r_min = 0.05 * a if a > 0 else 0.01
    r_max = 0.5 * a if a > 0 else 0.5
    mask = (r >= r_min) & (r <= r_max) & ~bc_mask.cpu().numpy()

    if mask.sum() < 3:
        mask = (r >= r_min) & (r <= r_max)

    K_I_num = float("nan")
    if mask.sum() >= 3:
        K_I_num = extract_K_I_from_displacement(
            u_np[mask, 1], r[mask], theta[mask], E, nu, plane_strain=True,
        )

    rel_err = abs(K_I_num - K_I_exact) / K_I_exact if K_I_exact > 0 else float("nan")

    return {
        "K_I_numerical": K_I_num,
        "K_I_analytical": K_I_exact,
        "relative_error": rel_err,
        "n_nodes": solver.n_nodes,
        "n_elements": solver.n_elements,
        "u": u_np,
        "nodes": nodes_np,
    }
