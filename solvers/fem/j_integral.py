"""Domain J-integral for stress intensity factor extraction.

Implements the domain-integral formulation of the J-integral (also known as
the equivalent domain integral or EDI method) for extracting stress intensity
factors from FEM displacement solutions near crack tips.

The J-integral is path-independent and converges much faster on coarse meshes
than displacement-correlation methods because it integrates over an area
rather than fitting individual nodal point values.

Mathematical formulation (2D equivalent, applied element-by-element in 3D):

    J = integral_A [ sigma_ij * du_j/dx_1 * dq/dx_i - W * dq/dx_1 ] dA

where:
  - sigma_ij is the Cauchy stress tensor
  - u_j is the displacement field
  - x_1 is the crack-direction coordinate
  - W = 0.5 * sigma : epsilon is the strain energy density
  - q is a smooth virtual crack extension field:
      q = 1 near the crack tip (r < r_inner)
      q = 0 far from tip (r > r_outer)
      smooth transition in between

Then K_I = sqrt(J * E') where E' = E/(1-nu^2) for plane strain.

Reference:
    Shih, Moran & Nakamura (1986), "Energy release rate along a
    three-dimensional crack front in a thermally stressed body",
    Int J Fracture, 30, 79-102.
"""

import numpy as np
import math
from typing import Optional, List, Tuple


def _smooth_q(r: np.ndarray, r_inner: float, r_outer: float) -> np.ndarray:
    """Smooth plateau function: 1 for r < r_inner, 0 for r > r_outer.

    Uses a C^inf bump function based on the standard smooth step:
        q(r) = 1 - smooth_step((r - r_inner) / (r_outer - r_inner))

    where smooth_step(t) = 3t^2 - 2t^3 (Hermite interpolation).
    """
    q = np.ones_like(r)
    transition = (r >= r_inner) & (r <= r_outer)
    if r_outer > r_inner:
        t = (r[transition] - r_inner) / (r_outer - r_inner)
        t = np.clip(t, 0.0, 1.0)
        q[transition] = 1.0 - (3.0 * t**2 - 2.0 * t**3)
    q[r > r_outer] = 0.0
    return q


def _grad_q(
    r: np.ndarray,
    dx: np.ndarray,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    """Gradient of the smooth q-function in physical coordinates.

    dq/dx_i = dq/dr * dr/dx_i = dq/dr * x_i / r

    Returns
    -------
    grad_q : ndarray (..., 3)
    """
    grad_q = np.zeros_like(dx)
    transition = (r >= r_inner) & (r <= r_outer) & (r > 1e-15)

    if r_outer > r_inner:
        t = (r[transition] - r_inner) / (r_outer - r_inner)
        t = np.clip(t, 0.0, 1.0)
        # dq/dr = -(6t - 6t^2) / (r_outer - r_inner)
        dq_dr = -(6.0 * t - 6.0 * t**2) / (r_outer - r_inner)
        # dr/dx_i = x_i / r
        r_trans = r[transition]
        for d in range(dx.shape[-1]):
            grad_q[transition, d] = dq_dr * dx[transition, d] / r_trans

    return grad_q


def compute_J_integral(
    chart_solvers: list,
    u_charts: list,
    crack_tip: np.ndarray,
    crack_direction: np.ndarray,
    opening_direction: np.ndarray,
    stress_fn,
    E: float,
    nu: float,
    plane_strain: bool = True,
    r_inner: Optional[float] = None,
    r_outer: Optional[float] = None,
    n_contours: int = 3,
) -> float:
    """Compute the J-integral using the domain integral method.

    Collects elements from all charts within the integration domain,
    computes stress, strain energy, and displacement gradients at each
    element, then evaluates the domain integral.

    Parameters
    ----------
    chart_solvers : list of ChartVectorFEMSolver
        FEM solvers for each chart.
    u_charts : list of torch.Tensor (or None)
        Nodal displacements per chart.
    crack_tip : array-like (3,)
        Crack tip position in physical space.
    crack_direction : array-like (3,)
        Unit vector along crack face (from tip into crack).
    opening_direction : array-like (3,)
        Unit vector perpendicular to crack face (Mode I opening).
    stress_fn : callable
        stress_fn(F) -> P (1st Piola-Kirchhoff stress), where F is (M,3,3).
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_strain : bool
        True for plane-strain, False for plane-stress.
    r_inner : float, optional
        Inner radius of q-function (q=1 inside). Default: 0.1 * r_char.
    r_outer : float, optional
        Outer radius of q-function (q=0 outside). Default: 0.5 * r_char.
    n_contours : int
        Number of domain contours to average (for robustness).

    Returns
    -------
    J : float
        J-integral value. Related to K_I by: K_I = sqrt(J * E').
    """
    import torch

    crack_tip = np.asarray(crack_tip, dtype=np.float64)
    cd = np.asarray(crack_direction, dtype=np.float64)
    od = np.asarray(opening_direction, dtype=np.float64)
    cd = cd / np.linalg.norm(cd)
    od = od / np.linalg.norm(od)
    # Out-of-plane direction
    e3 = np.cross(cd, od)
    e3 = e3 / np.linalg.norm(e3)

    # Collect element-level data from all charts
    all_centroids = []      # physical centroid positions
    all_grad_u = []         # displacement gradient at centroid
    all_stress = []         # Cauchy stress at centroid
    all_vol = []            # element physical volume

    for ci, solver in enumerate(chart_solvers):
        if u_charts[ci] is None:
            continue
        if solver.n_elements == 0:
            continue

        u = u_charts[ci]
        centroids = solver.elem_centroids_phys.detach().cpu().numpy()

        # Compute deformation gradient
        F_elem = solver.compute_F(u)  # (M, 3, 3)

        # Compute stress: P = stress_fn(F)
        P_elem = stress_fn(F_elem)  # (M, 3, 3) first Piola-Kirchhoff

        # For small-strain linear elastic, Cauchy stress = P (approx)
        # For finite strain: sigma = (1/J) P F^T
        J_det = torch.det(F_elem)
        J_det = torch.clamp(J_det, min=1e-10)

        # Cauchy stress: sigma = (1/J) P F^T
        sigma = torch.einsum("eij,ekj->eik", P_elem, F_elem) / J_det.unsqueeze(-1).unsqueeze(-1)

        # Displacement gradient: grad_u = F - I
        I3 = torch.eye(3, device=F_elem.device, dtype=F_elem.dtype).unsqueeze(0)
        grad_u = F_elem - I3

        # Physical volumes
        vol = solver.vol.detach().cpu().numpy()

        all_centroids.append(centroids)
        all_grad_u.append(grad_u.detach().cpu().numpy())
        all_stress.append(sigma.detach().cpu().numpy())
        all_vol.append(vol)

    if not all_centroids:
        return float("nan")

    centroids = np.concatenate(all_centroids, axis=0)
    grad_u = np.concatenate(all_grad_u, axis=0)
    stress = np.concatenate(all_stress, axis=0)
    vol = np.concatenate(all_vol, axis=0)

    # Distance from crack tip
    dx = centroids - crack_tip
    r = np.linalg.norm(dx, axis=1)

    # Characteristic radius: use MEDIAN of near-tip elements, NOT global 90th percentile.
    # This avoids inflated r_char when distant bulk charts are included.
    near_tip = r[r > 1e-10]
    if len(near_tip) > 0:
        # Use elements within the closest 30% as the "near-tip" population
        r_sorted = np.sort(near_tip)
        cutoff_idx = max(1, len(r_sorted) // 3)
        r_char = r_sorted[cutoff_idx]
    else:
        r_char = 1.0

    # Compute J for multiple contour radii and average
    J_values = []

    for k in range(n_contours):
        # Scale factor for this contour: vary from 0.6x to 1.4x of base radii
        scale = 0.6 + 0.8 * k / max(n_contours - 1, 1)

        ri = (r_inner if r_inner is not None else 0.15 * r_char) * scale
        ro = (r_outer if r_outer is not None else 0.6 * r_char) * scale

        if ro <= ri:
            ro = ri * 3.0

        # Compute q and grad_q at element centroids
        q = _smooth_q(r, ri, ro)
        gq = _grad_q(r, dx, ri, ro)

        # Only process elements in the transition region (where grad_q != 0)
        active = (r >= ri * 0.9) & (r <= ro * 1.1)
        if not np.any(active):
            continue

        # J = sum_elements [ (sigma_ij * du_j/dx_1 * dq/dx_i - W * dq/dx_1) * vol ]
        # where x_1 is the crack direction

        # du_j/dx_1 = grad_u[j, :] . crack_direction
        # For each element: du_dx1[j] = sum_k grad_u[j,k] * cd[k]
        du_dx1 = np.einsum("eij,j->ei", grad_u[active], cd)  # (M_active, 3)

        # sigma_ij * du_j/dx_1 * dq/dx_i
        # = sum_i sum_j sigma[i,j] * du_dx1[j] * gq[i]
        # = sum_i gq[i] * (sum_j sigma[i,j] * du_dx1[j])
        sigma_du = np.einsum("eij,ej->ei", stress[active], du_dx1)  # (M_active, 3)
        term1 = np.einsum("ei,ei->e", sigma_du, gq[active])  # (M_active,)

        # Strain energy density W = 0.5 * sigma : epsilon
        # For small strain: epsilon = 0.5 * (grad_u + grad_u^T)
        epsilon = 0.5 * (grad_u[active] + np.swapaxes(grad_u[active], -2, -1))
        W = 0.5 * np.einsum("eij,eij->e", stress[active], epsilon)  # (M_active,)

        # W * dq/dx_1 = W * (gq . cd)
        dq_dx1 = np.einsum("ei,i->e", gq[active], cd)  # (M_active,)
        term2 = W * dq_dx1

        # J contribution from this contour
        J_k = np.sum((term1 - term2) * vol[active])
        J_values.append(J_k)

    if not J_values:
        return float("nan")

    # Average over contours for robustness
    J = float(np.median(J_values))
    return J


def extract_K_via_J_integral(
    chart_solvers: list,
    u_charts: list,
    crack_tip,
    crack_direction,
    opening_direction,
    stress_fn,
    E: float,
    nu: float,
    plane_strain: bool = True,
    r_inner: Optional[float] = None,
    r_outer: Optional[float] = None,
    n_contours: int = 3,
) -> float:
    """Extract K_I from J-integral: K_I = sqrt(J * E').

    Convenience wrapper around compute_J_integral.

    Parameters
    ----------
    (same as compute_J_integral)

    Returns
    -------
    K_I : float
        Mode-I stress intensity factor.
    """
    J = compute_J_integral(
        chart_solvers, u_charts,
        crack_tip, crack_direction, opening_direction,
        stress_fn, E, nu, plane_strain,
        r_inner, r_outer, n_contours,
    )

    if math.isnan(J) or J < 0:
        return float("nan")

    # Irwin relation: J = K_I^2 / E'
    if plane_strain:
        E_prime = E / (1.0 - nu**2)
    else:
        E_prime = E

    K_I = math.sqrt(J * E_prime)
    return K_I
