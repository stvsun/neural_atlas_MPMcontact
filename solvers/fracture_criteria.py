"""Fracture nucleation and propagation criteria.

Implements the Drucker-Prager strength surface from Kamarei et al. (2026)
and Griffith propagation criterion for the topo_atlas fracture pipeline.

Drucker-Prager strength surface (Eq. 2 of Kamarei et al.):
    F(sigma) = sqrt(J2) + alpha * I1 - k = 0
where:
    alpha = sigma_ts / (sqrt(3) * (3*sigma_hs - sigma_ts))
    k     = sqrt(3) * sigma_hs * sigma_ts / (3*sigma_hs - sigma_ts)
    I1    = sig1 + sig2 + sig3             (first stress invariant)
    J2    = (1/3)(sig1^2+sig2^2+sig3^2-sig1*sig2-sig1*sig3-sig2*sig3)

Derived strengths:
    sigma_bs = 3*sigma_hs*sigma_ts / (3*sigma_hs + sigma_ts)
    sigma_ss = sqrt(3)*sigma_hs*sigma_ts / (3*sigma_hs - sigma_ts) = k
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Drucker-Prager strength surface
# ---------------------------------------------------------------------------

def drucker_prager_coefficients(sigma_ts: float, sigma_hs: float) -> Tuple[float, float]:
    """Compute Drucker-Prager coefficients alpha and k.

    Parameters
    ----------
    sigma_ts : float
        Uniaxial tensile strength.
    sigma_hs : float
        Hydrostatic tensile strength.

    Returns
    -------
    alpha, k : float
        Drucker-Prager friction angle and cohesion.
    """
    denom = math.sqrt(3) * (3 * sigma_hs - sigma_ts)
    alpha = sigma_ts / denom
    k = math.sqrt(3) * sigma_hs * sigma_ts / (3 * sigma_hs - sigma_ts)
    return alpha, k


def drucker_prager_F(sigma: np.ndarray, sigma_ts: float, sigma_hs: float) -> np.ndarray:
    """Evaluate the Drucker-Prager strength surface at stress states.

    Parameters
    ----------
    sigma : ndarray (..., 3, 3)
        Cauchy stress tensors.
    sigma_ts : float
        Uniaxial tensile strength.
    sigma_hs : float
        Hydrostatic tensile strength.

    Returns
    -------
    F : ndarray (...)
        F < 0: elastic (inside strength surface).
        F >= 0: nucleation (on or outside strength surface).
    """
    alpha, k = drucker_prager_coefficients(sigma_ts, sigma_hs)

    # Principal stresses
    eigvals = np.linalg.eigvalsh(sigma)  # (..., 3), sorted ascending
    s1 = eigvals[..., 0]
    s2 = eigvals[..., 1]
    s3 = eigvals[..., 2]

    I1 = s1 + s2 + s3
    J2 = (1.0 / 3.0) * (s1**2 + s2**2 + s3**2 - s1 * s2 - s1 * s3 - s2 * s3)
    J2 = np.maximum(J2, 0.0)  # numerical safety

    return np.sqrt(J2) + alpha * I1 - k


def derived_biaxial_strength(sigma_ts: float, sigma_hs: float) -> float:
    """Compute biaxial tensile strength from Drucker-Prager.

    sigma_bs = 3 * sigma_hs * sigma_ts / (3 * sigma_hs + sigma_ts)
    """
    return 3 * sigma_hs * sigma_ts / (3 * sigma_hs + sigma_ts)


def derived_shear_strength(sigma_ts: float, sigma_hs: float) -> float:
    """Compute shear strength from Drucker-Prager.

    sigma_ss = sqrt(3) * sigma_hs * sigma_ts / (3 * sigma_hs - sigma_ts)
    """
    return math.sqrt(3) * sigma_hs * sigma_ts / (3 * sigma_hs - sigma_ts)


# ---------------------------------------------------------------------------
# Crack direction from stress
# ---------------------------------------------------------------------------

def crack_normal_from_stress(sigma: np.ndarray) -> np.ndarray:
    """Determine crack normal as the eigenvector of the largest principal stress.

    The crack nucleates perpendicular to the direction of maximum tension.

    Parameters
    ----------
    sigma : ndarray (3, 3)
        Cauchy stress tensor at a single point.

    Returns
    -------
    normal : ndarray (3,)
        Unit normal of the crack plane (eigenvector of max principal stress).
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    max_idx = np.argmax(eigvals)
    normal = eigvecs[:, max_idx]
    return normal / np.linalg.norm(normal)


def max_hoop_stress_angle(K_I: float, K_II: float) -> float:
    """Maximum tangential stress angle for mixed-mode propagation.

    theta_c = 2 * arctan( (K_I - sqrt(K_I^2 + 8*K_II^2)) / (4*K_II) )

    For pure Mode I (K_II = 0): theta_c = 0 (straight ahead).

    Parameters
    ----------
    K_I : float
        Mode-I stress intensity factor.
    K_II : float
        Mode-II stress intensity factor.

    Returns
    -------
    theta_c : float
        Propagation angle in radians (0 = straight ahead).
    """
    if abs(K_II) < 1e-12:
        return 0.0
    disc = math.sqrt(K_I**2 + 8 * K_II**2)
    return 2.0 * math.atan2(K_I - disc, 4 * K_II)


# ---------------------------------------------------------------------------
# Griffith propagation criterion
# ---------------------------------------------------------------------------

def griffith_K_Ic(E: float, G_c: float, nu: float = 0.0, plane_strain: bool = True) -> float:
    """Critical stress intensity factor from Griffith energy balance.

    Plane strain:  K_Ic = sqrt(E * G_c / (1 - nu^2))
    Plane stress:  K_Ic = sqrt(E * G_c)

    Parameters
    ----------
    E : float
        Young's modulus.
    G_c : float
        Critical energy release rate.
    nu : float
        Poisson's ratio (only used for plane strain).
    plane_strain : bool
        If True, use plane-strain formula.

    Returns
    -------
    K_Ic : float
        Fracture toughness.
    """
    if plane_strain:
        return math.sqrt(E * G_c / (1 - nu**2))
    return math.sqrt(E * G_c)


# ---------------------------------------------------------------------------
# Nucleation check on a solved FEM domain
# ---------------------------------------------------------------------------

def check_nucleation_pointwise(
    stress_field: np.ndarray,
    element_centroids: np.ndarray,
    sigma_ts: float,
    sigma_hs: float,
) -> List[Dict]:
    """Check Drucker-Prager nucleation criterion at all elements.

    Parameters
    ----------
    stress_field : ndarray (M, 3, 3)
        Cauchy stress at each element centroid.
    element_centroids : ndarray (M, 3)
        Physical coordinates of element centroids.
    sigma_ts : float
        Uniaxial tensile strength.
    sigma_hs : float
        Hydrostatic tensile strength.

    Returns
    -------
    nucleation_sites : list of dict
        Each dict has: element_id, center, F_value, principal_stress,
        crack_normal. Empty list if no nucleation detected.
    """
    F_vals = drucker_prager_F(stress_field, sigma_ts, sigma_hs)

    nucleation_sites = []
    violated = np.where(F_vals >= 0)[0]

    if len(violated) == 0:
        return nucleation_sites

    # Find the element with the maximum F value (most critical)
    max_idx = violated[np.argmax(F_vals[violated])]

    sigma_at_max = stress_field[max_idx]
    eigvals = np.linalg.eigvalsh(sigma_at_max)
    normal = crack_normal_from_stress(sigma_at_max)

    nucleation_sites.append({
        "element_id": int(max_idx),
        "center": element_centroids[max_idx].copy(),
        "F_value": float(F_vals[max_idx]),
        "principal_stresses": eigvals.tolist(),
        "crack_normal": normal.tolist(),
        "n_violated": len(violated),
    })

    return nucleation_sites


def cauchy_from_first_piola(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Convert first Piola-Kirchhoff stress to Cauchy stress.

    sigma = (1/J) * P @ F^T

    Parameters
    ----------
    P : ndarray (M, 3, 3)
        First Piola-Kirchhoff stress.
    F : ndarray (M, 3, 3)
        Deformation gradient.

    Returns
    -------
    sigma : ndarray (M, 3, 3)
        Cauchy stress.
    """
    J = np.linalg.det(F)  # (M,)
    PFt = np.einsum("mij,mkj->mik", P, F)  # (M, 3, 3)
    return PFt / J[:, None, None]
