"""K_I extraction from FEM displacement fields near crack tips.

Supports two methods:
  - 'displacement': Williams asymptotic expansion fitting (legacy)
  - 'j_integral': Domain J-integral method (recommended, more robust on coarse meshes)

Bridges the FEM solver displacement solution with analytical LEFM to extract
stress intensity factors K_I (and K_II) from a chart-based FEM solution.
"""

import numpy as np
from typing import Optional, Tuple


def extract_K_from_fem(
    solver,
    u,
    crack_tip,
    crack_direction,
    opening_direction,
    E: float,
    nu: float,
    plane_strain: bool = True,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
) -> float:
    """Extract Mode-I stress intensity factor K_I from FEM displacement.

    Projects the nodal displacement onto the crack-tip polar coordinate
    system and fits the opening displacement to the Williams asymptotic
    expansion u_y ~ K_I/(2mu) * sqrt(r/(2pi)) * shape(theta).

    Parameters
    ----------
    solver : ChartVectorFEMSolver
        The solver that produced the displacement field.
    u : torch.Tensor, shape (N, 3)
        Nodal displacement vector.
    crack_tip : array-like, shape (3,)
        Crack tip position in physical space.
    crack_direction : array-like, shape (3,)
        Unit vector along crack face (from tip toward crack interior).
    opening_direction : array-like, shape (3,)
        Unit vector perpendicular to crack face (Mode I opening).
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_strain : bool
        True for plane-strain, False for plane-stress.
    r_min, r_max : float or None
        Annular region for fitting. Defaults to 0.05 * r_char and 0.5 * r_char
        where r_char is the max distance from tip to any node.

    Returns
    -------
    K_I : float
        Extracted Mode-I stress intensity factor.
    """
    nodes = solver.nodes_phys.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()

    crack_tip = np.asarray(crack_tip, dtype=float)
    cd = np.asarray(crack_direction, dtype=float)
    od = np.asarray(opening_direction, dtype=float)
    cd = cd / np.linalg.norm(cd)
    od = od / np.linalg.norm(od)

    # Compute polar coordinates relative to crack tip in the crack frame
    dx = nodes - crack_tip
    x1 = dx @ cd    # along crack face
    x2 = dx @ od    # perpendicular (opening direction)

    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)

    # Extract opening displacement component
    u_opening = u_np @ od

    # Set annular region — tight near tip to reduce far-field contamination
    r_char = r.max() if r.max() > 0 else 1.0
    if r_min is None:
        r_min = 0.02 * r_char
    if r_max is None:
        r_max = 0.15 * r_char

    # Filter to annular region
    mask = (r >= r_min) & (r <= r_max) & (r > 1e-10)
    if np.sum(mask) < 5:
        r_max = 0.3 * r_char
        mask = (r >= r_min) & (r <= r_max) & (r > 1e-10)
    if not np.any(mask):
        mask = r > 1e-10
    if not np.any(mask):
        return float("nan")

    # Subtract far-field linear component to isolate Williams singularity.
    # Far-field u_opening ≈ a * x2 + b (affine strain field).
    u_op_mask = u_opening[mask]
    x2_mask = x2[mask]
    if len(u_op_mask) > 3:
        A_mat = np.column_stack([x2_mask, np.ones_like(x2_mask)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, u_op_mask, rcond=None)
        u_op_corrected = u_op_mask - A_mat @ coeffs
    else:
        u_op_corrected = u_op_mask

    from benchmarks.fracture.lefm_reference import extract_K_I_from_displacement
    return extract_K_I_from_displacement(
        u_op_corrected, r[mask], theta[mask], E, nu, plane_strain
    )


def extract_K_from_charts(
    chart_solvers,
    u_charts,
    crack_tip,
    crack_direction,
    opening_direction,
    E: float,
    nu: float,
    plane_strain: bool = True,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    method: str = "displacement",
    stress_fn=None,
) -> float:
    """Extract K_I from multi-chart FEM solution by pooling all charts.

    Parameters
    ----------
    chart_solvers : list of ChartVectorFEMSolver
    u_charts : list of torch.Tensor (or None for inactive charts)
    crack_tip, crack_direction, opening_direction : array-like (3,)
    E, nu : float
    plane_strain : bool
    r_min, r_max : float or None
    method : str
        'displacement' (legacy Williams fitting) or 'j_integral' (domain integral).
    stress_fn : callable, optional
        Required when method='j_integral'. Maps F -> P (1st Piola-Kirchhoff stress).

    Returns
    -------
    K_I : float
    """
    if method == "j_integral":
        from solvers.fem.j_integral import extract_K_via_J_integral
        if stress_fn is None:
            raise ValueError("stress_fn is required for method='j_integral'")
        return extract_K_via_J_integral(
            chart_solvers, u_charts,
            crack_tip, crack_direction, opening_direction,
            stress_fn, E, nu, plane_strain,
            r_inner=r_min, r_outer=r_max,
            n_contours=3,
        )
    all_r = []
    all_theta = []
    all_u_opening = []

    crack_tip = np.asarray(crack_tip, dtype=float)
    cd = np.asarray(crack_direction, dtype=float)
    od = np.asarray(opening_direction, dtype=float)
    cd = cd / np.linalg.norm(cd)
    od = od / np.linalg.norm(od)

    for ci, solver in enumerate(chart_solvers):
        if u_charts[ci] is None:
            continue
        nodes = solver.nodes_phys.detach().cpu().numpy()
        u_np = u_charts[ci].detach().cpu().numpy()

        dx = nodes - crack_tip
        x1 = dx @ cd
        x2 = dx @ od

        r = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)
        u_opening = u_np @ od

        all_r.append(r)
        all_theta.append(theta)
        all_u_opening.append(u_opening)

    if not all_r:
        return float("nan")

    r_all = np.concatenate(all_r)
    theta_all = np.concatenate(all_theta)
    u_all = np.concatenate(all_u_opening)

    # Also gather x2 (opening direction coordinate) for far-field subtraction
    all_x2 = []
    for ci, solver in enumerate(chart_solvers):
        if u_charts[ci] is None:
            continue
        nodes = solver.nodes_phys.detach().cpu().numpy()
        dx = nodes - crack_tip
        all_x2.append(dx @ od)
    x2_all = np.concatenate(all_x2)

    r_char = r_all.max() if r_all.max() > 0 else 1.0
    if r_min is None:
        r_min = 0.02 * r_char
    if r_max is None:
        r_max = 0.15 * r_char

    mask = (r_all >= r_min) & (r_all <= r_max) & (r_all > 1e-10)
    if np.sum(mask) < 5:
        r_max = 0.3 * r_char
        mask = (r_all >= r_min) & (r_all <= r_max) & (r_all > 1e-10)
    if not np.any(mask):
        mask = r_all > 1e-10
    if not np.any(mask):
        return float("nan")

    # Subtract far-field linear component
    u_op_mask = u_all[mask]
    x2_mask = x2_all[mask]
    if len(u_op_mask) > 3:
        A_mat = np.column_stack([x2_mask, np.ones_like(x2_mask)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, u_op_mask, rcond=None)
        u_op_corrected = u_op_mask - A_mat @ coeffs
    else:
        u_op_corrected = u_op_mask

    from benchmarks.fracture.lefm_reference import extract_K_I_from_displacement
    return extract_K_I_from_displacement(
        u_op_corrected, r_all[mask], theta_all[mask], E, nu, plane_strain
    )
