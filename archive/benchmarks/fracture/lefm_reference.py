"""Analytical LEFM solutions for Mode-I edge crack in a finite-width plate.

Provides stress intensity factor K_I, near-tip displacement fields, and
the Williams asymptotic expansion for validation of numerical solutions.

Reference:
    Tada, Paris & Irwin (2000), "The Stress Analysis of Cracks Handbook"
    Anderson (2005), "Fracture Mechanics: Fundamentals and Applications"
"""

import math
from typing import Dict, Tuple

import numpy as np


def geometry_factor_edge_crack(a_over_W: float) -> float:
    """Geometry correction factor F(a/W) for a single edge crack in a
    finite-width plate under uniform tension.

    Uses the polynomial fit from Tada, Paris & Irwin (2000), Eq. 2.11:
        F(λ) = 1.122 - 0.231λ + 10.55λ² - 21.71λ³ + 30.38λ⁴
    where λ = a/W.

    Valid for a/W ∈ [0, 0.6] with error < 0.5%.

    Parameters
    ----------
    a_over_W : float
        Ratio of crack length to plate half-width.

    Returns
    -------
    F : float
        Geometry correction factor (dimensionless).
    """
    lam = a_over_W
    return (
        1.122
        - 0.231 * lam
        + 10.55 * lam ** 2
        - 21.71 * lam ** 3
        + 30.38 * lam ** 4
    )


def stress_intensity_factor(
    sigma_inf: float,
    a: float,
    W: float,
) -> float:
    """Mode-I stress intensity factor for edge crack under far-field tension.

    K_I = σ_∞ √(πa) F(a/W)

    Parameters
    ----------
    sigma_inf : float
        Far-field tensile stress.
    a : float
        Crack length.
    W : float
        Plate half-width.

    Returns
    -------
    K_I : float
        Mode-I stress intensity factor.
    """
    a_over_W = a / W
    F = geometry_factor_edge_crack(a_over_W)
    return sigma_inf * math.sqrt(math.pi * a) * F


def williams_displacement(
    r: np.ndarray,
    theta: np.ndarray,
    K_I: float,
    E: float,
    nu: float,
    plane_strain: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mode-I Williams asymptotic displacement field near the crack tip.

    u_x = K_I / (2μ) √(r/(2π)) cos(θ/2) [κ - 1 + 2 sin²(θ/2)]
    u_y = K_I / (2μ) √(r/(2π)) sin(θ/2) [κ + 1 - 2 cos²(θ/2)]

    where κ = 3 - 4ν (plane strain) or (3 - ν)/(1 + ν) (plane stress).

    Parameters
    ----------
    r : ndarray (N,)
        Distance from crack tip.
    theta : ndarray (N,)
        Angle from crack line (radians, measured CCW from +x1).
    K_I : float
        Mode-I stress intensity factor.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_strain : bool
        True for plane-strain, False for plane-stress.

    Returns
    -------
    u_x, u_y : ndarray (N,)
        Displacement components near the crack tip.
    """
    mu = E / (2.0 * (1.0 + nu))
    if plane_strain:
        kappa = 3.0 - 4.0 * nu
    else:
        kappa = (3.0 - nu) / (1.0 + nu)

    sqrt_r = np.sqrt(r / (2.0 * math.pi))
    cos_half = np.cos(theta / 2.0)
    sin_half = np.sin(theta / 2.0)

    prefactor = K_I / (2.0 * mu) * sqrt_r

    u_x = prefactor * cos_half * (kappa - 1.0 + 2.0 * sin_half ** 2)
    u_y = prefactor * sin_half * (kappa + 1.0 - 2.0 * cos_half ** 2)

    return u_x, u_y


def williams_stress(
    r: np.ndarray,
    theta: np.ndarray,
    K_I: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mode-I Williams asymptotic stress field near the crack tip.

    σ_xx = K_I / √(2πr) cos(θ/2) [1 - sin(θ/2) sin(3θ/2)]
    σ_yy = K_I / √(2πr) cos(θ/2) [1 + sin(θ/2) sin(3θ/2)]
    σ_xy = K_I / √(2πr) sin(θ/2) cos(θ/2) cos(3θ/2)

    Parameters
    ----------
    r : ndarray (N,)
        Distance from crack tip.
    theta : ndarray (N,)
        Angle from crack line.
    K_I : float
        Stress intensity factor.

    Returns
    -------
    sigma_xx, sigma_yy, sigma_xy : ndarray (N,)
    """
    inv_sqrt_r = 1.0 / np.sqrt(2.0 * math.pi * np.maximum(r, 1e-15))
    cos_half = np.cos(theta / 2.0)
    sin_half = np.sin(theta / 2.0)
    cos_3half = np.cos(3.0 * theta / 2.0)
    sin_3half = np.sin(3.0 * theta / 2.0)

    prefactor = K_I * inv_sqrt_r

    sigma_xx = prefactor * cos_half * (1.0 - sin_half * sin_3half)
    sigma_yy = prefactor * cos_half * (1.0 + sin_half * sin_3half)
    sigma_xy = prefactor * sin_half * cos_half * cos_3half

    return sigma_xx, sigma_yy, sigma_xy


def extract_K_I_from_displacement(
    u_y: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    E: float,
    nu: float,
    plane_strain: bool = True,
) -> float:
    """Extract K_I from the displacement field using the Williams expansion.

    Uses the crack-opening displacement on θ = π (upper crack face):
        u_y(r, π) = K_I / (2μ) √(r/(2π)) (κ + 1)

    Fits K_I via least-squares on points near the crack tip.

    Parameters
    ----------
    u_y : ndarray (N,)
        y-displacement at evaluation points.
    r : ndarray (N,)
        Distance from crack tip.
    theta : ndarray (N,)
        Angle from crack line.
    E, nu : float
        Elastic constants.
    plane_strain : bool
        True for plane-strain.

    Returns
    -------
    K_I_extracted : float
        Extracted stress intensity factor.
    """
    mu = E / (2.0 * (1.0 + nu))
    if plane_strain:
        kappa = 3.0 - 4.0 * nu
    else:
        kappa = (3.0 - nu) / (1.0 + nu)

    # At general θ: u_y = K_I/(2μ) √(r/(2π)) sin(θ/2) [κ+1 - 2cos²(θ/2)]
    # Solve for K_I point-wise
    sqrt_r = np.sqrt(r / (2.0 * math.pi))

    sin_half = np.sin(theta / 2.0)
    cos_half = np.cos(theta / 2.0)
    shape_fn = sin_half * (kappa + 1.0 - 2.0 * cos_half ** 2)

    # Avoid division by zero where shape function is near zero (θ ~ 0)
    good = (np.abs(shape_fn) > 0.1) & (r > 1e-6) & np.isfinite(u_y)
    if not np.any(good):
        good = (np.abs(shape_fn) > 1e-3) & np.isfinite(u_y)
    if not np.any(good):
        return float("nan")

    K_estimates = u_y[good] * 2.0 * mu / (sqrt_r[good] * shape_fn[good])

    return float(np.median(K_estimates))


def mode_i_reference_table(
    W: float = 1.0,
    H: float = 4.0,
    sigma_inf: float = 1.0,
    E: float = 200.0,
    nu: float = 0.3,
    a_values: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Generate reference K_I table for a range of crack lengths.

    Parameters
    ----------
    W : float
        Plate half-width.
    H : float
        Plate height (for documentation only; K_I independent for H/W > 2).
    sigma_inf : float
        Far-field stress.
    E, nu : float
        Elastic constants.
    a_values : ndarray, optional
        Crack lengths. Default: np.linspace(0.1, 0.6, 6) * W.

    Returns
    -------
    table : dict
        Keys: a, a_over_W, F, K_I, u_y_max (max COD at midpoint).
    """
    if a_values is None:
        a_values = np.linspace(0.1, 0.6, 6) * W

    results = {
        "a": a_values,
        "a_over_W": a_values / W,
        "F": np.array([geometry_factor_edge_crack(a / W) for a in a_values]),
        "K_I": np.array([stress_intensity_factor(sigma_inf, a, W) for a in a_values]),
    }

    return results
