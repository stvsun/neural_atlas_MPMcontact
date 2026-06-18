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
    """Maximum tangential stress angle for in-plane mixed-mode propagation.

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
        In-plane kinking angle in radians (0 = straight ahead).
    """
    if abs(K_II) < 1e-12:
        return 0.0
    disc = math.sqrt(K_I**2 + 8 * K_II**2)
    return 2.0 * math.atan2(K_I - disc, 4 * K_II)


def mode_III_twist_angle(K_I: float, K_III: float, nu: float = 0.3) -> float:
    """Out-of-plane twist angle for Mode I + III mixed-mode propagation.

    Implements the Schöllmann et al. (2002) criterion for the twist angle
    ψ caused by the anti-plane shear component K_III:

        ψ = -arctan( 2·K_III / (K_I + √(K_I² + (2/(2-ν))² · K_III²)) )  (simplified)

    For engineering purposes, the Pook (1985) approximation is often used:
        ψ ≈ -arctan( K_III / (2·K_I) ) · (1 - 2·ν) / (1 - ν)

    For pure Mode I (K_III = 0): ψ = 0 (no twist).
    For pure Mode III (K_I → 0): ψ → ±π/2 (maximum twist, crack becomes a kink).

    References
    ----------
    Schöllmann, Richard, Kullmer & Fulland (2002), "A new criterion for the
    prediction of crack development in multiaxially loaded structures",
    Int J Fracture, 117(2), 129-141.

    Pook (1985), "The fatigue crack direction and threshold behaviour of mild steel
    under mixed mode I and III loading", Int J Fatigue, 7(1), 21-30.

    Parameters
    ----------
    K_I : float
        Mode-I stress intensity factor.
    K_III : float
        Mode-III stress intensity factor.
    nu : float
        Poisson's ratio.

    Returns
    -------
    psi : float
        Out-of-plane twist angle in radians.
    """
    if abs(K_III) < 1e-12:
        return 0.0
    if abs(K_I) < 1e-12:
        # Pure Mode III: maximum twist
        return -math.copysign(math.pi / 4, K_III)

    # Pook approximation (simple, robust)
    psi = -math.atan(K_III / (2.0 * K_I)) * (1.0 - 2.0 * nu) / max(1.0 - nu, 1e-10)
    return psi


def propagation_direction_3d(
    K_I: float, K_II: float, K_III: float,
    crack_direction: np.ndarray,
    opening_direction: np.ndarray,
    nu: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D crack propagation direction from K_I, K_II, K_III.

    The propagation direction is determined by two angles:
      θ (in-plane kink): from max hoop stress (Erdogan-Sih), K_I + K_II
      ψ (out-of-plane twist): from Pook/Schöllmann criterion, K_I + K_III

    The new crack tip triad (e_crack, e_opening, e_front) is obtained by
    applying two rotations to the current triad:
      1. Rotate by θ around e_front (in-plane kink)
      2. Rotate by ψ around the new e_crack (out-of-plane twist)

    Parameters
    ----------
    K_I, K_II, K_III : float
        Stress intensity factors (Mode I, II, III).
    crack_direction : ndarray (3,)
        Current crack propagation direction (e_crack).
    opening_direction : ndarray (3,)
        Current crack opening direction (e_opening).
    nu : float
        Poisson's ratio.

    Returns
    -------
    new_crack_dir : ndarray (3,)
        Updated propagation direction.
    new_opening_dir : ndarray (3,)
        Updated opening direction.
    new_front_dir : ndarray (3,)
        Updated crack front direction.
    """
    cd = np.array(crack_direction, dtype=float)
    od = np.array(opening_direction, dtype=float)
    cd = cd / np.linalg.norm(cd)
    od = od / np.linalg.norm(od)
    e_front = np.cross(cd, od)
    e_front = e_front / np.linalg.norm(e_front)

    def rodrigues(v, axis, angle):
        """Rotate vector v around axis by angle (Rodrigues formula)."""
        c, s = math.cos(angle), math.sin(angle)
        return c * v + s * np.cross(axis, v) + (1 - c) * np.dot(axis, v) * axis

    # Step 1: In-plane kink (θ) — rotate around e_front
    theta = max_hoop_stress_angle(K_I, K_II)
    if abs(theta) > 1e-10:
        cd = rodrigues(cd, e_front, theta)
        od = rodrigues(od, e_front, theta)
        cd = cd / np.linalg.norm(cd)
        od = od / np.linalg.norm(od)

    # Step 2: Out-of-plane twist (ψ) — rotate around new e_crack
    psi = mode_III_twist_angle(K_I, K_III, nu)
    if abs(psi) > 1e-10:
        od = rodrigues(od, cd, psi)
        e_front = rodrigues(e_front, cd, psi)
        od = od / np.linalg.norm(od)
        e_front = e_front / np.linalg.norm(e_front)

    return cd, od, e_front


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
