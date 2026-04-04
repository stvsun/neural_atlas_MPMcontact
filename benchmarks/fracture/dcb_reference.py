"""Double Cantilever Beam (DCB) analytical reference solution.

Uses standard beam theory with shear correction (Kanninen 1973 / ASTM D5528):

    Compliance: C = 2a^3 / (3EI) + 2a / (kAG)  [Timoshenko beam]
    G = F^2 / (2B) * dC/da                       [Irwin-Kies]
    At Griffith: G = Gc => F_crit, delta_crit

Simplified (Euler-Bernoulli, no shear):
    C = 2a^3 / (3EI)
    G = 12 F^2 a^2 / (E B^2 h^3)
    F_crit = B * sqrt(E * Gc * h^3 / (12 * a^2))
    delta_crit = F_crit * C

where h = H/2 (half-height = arm height), I = B*h^3/12.

Geometry (Kamarei et al. 2026, Challenge Problem 8):
    L=55mm, H=20mm, B=2.5mm, A=25mm (initial crack)

Material (soda-lime glass):
    E=70 GPa, nu=0.22, G_c=10 N/m = 0.01 N/mm
"""

import math
from typing import Dict

import numpy as np
from scipy.optimize import brentq


# ── Geometry ──────────────────────────────────────────────────────────
DCB_L = 55.0     # mm, total length
DCB_H = 20.0     # mm, full height
DCB_B = 2.5      # mm, thickness (out-of-plane)
DCB_A = 25.0     # mm, initial crack length
DCB_h = DCB_H / 2  # mm, arm height

# ── Material (soda-lime glass) ────────────────────────────────────────
DCB_E = 70e3     # MPa
DCB_nu = 0.22
DCB_Gc = 0.01    # N/mm (= 10 N/m = 10 J/m^2)


def dcb_compliance(a, E=DCB_E, B=DCB_B, h=DCB_h):
    """Compliance C(a) = delta/F = 2a^3 / (3*E*I) where I = B*h^3/12."""
    I = B * h**3 / 12.0
    return 2.0 * a**3 / (3.0 * E * I)


def dcb_energy_release_rate(F, a, E=DCB_E, B=DCB_B, h=DCB_h):
    """G = 12 * F^2 * a^2 / (E * B^2 * h^3) [Irwin-Kies for DCB]."""
    return 12.0 * F**2 * a**2 / (E * B**2 * h**3)


def dcb_force_from_delta(delta, a, E=DCB_E, B=DCB_B, h=DCB_h):
    """F = delta / C(a)."""
    C = dcb_compliance(a, E, B, h)
    return delta / C if C > 0 else 0.0


def dcb_critical_force(a, E=DCB_E, B=DCB_B, h=DCB_h, Gc=DCB_Gc):
    """Critical force at Griffith: G = Gc.

    F_crit = B * sqrt(E * Gc * h^3 / (12 * a^2))
    """
    return B * math.sqrt(E * Gc * h**3 / (12.0 * a**2))


def dcb_critical_displacement(a=DCB_A, E=DCB_E, B=DCB_B, h=DCB_h, Gc=DCB_Gc):
    """Critical displacement at which crack starts growing.

    delta_crit = F_crit * C(a)
    """
    F_crit = dcb_critical_force(a, E, B, h, Gc)
    C = dcb_compliance(a, E, B, h)
    return F_crit * C


def dcb_crack_length_at_delta(delta, A=DCB_A, E=DCB_E, B=DCB_B, h=DCB_h, Gc=DCB_Gc):
    """Crack length a(delta) from the Griffith condition.

    During propagation: G(F(delta, a), a) = Gc
    => 12 * (delta/C(a))^2 * a^2 / (E * B^2 * h^3) = Gc
    => 12 * delta^2 * a^2 / (C(a)^2 * E * B^2 * h^3) = Gc

    Substituting C(a) = 2a^3/(3EI):
    => 12 * delta^2 * a^2 * (3EI)^2 / (4 * a^6 * E * B^2 * h^3) = Gc
    => 12 * 9 * E^2 * I^2 * delta^2 / (4 * a^4 * E * B^2 * h^3) = Gc
    => 27 * E * I^2 * delta^2 / (a^4 * B^2 * h^3) = Gc

    With I = B*h^3/12:
    => 27 * E * B^2 * h^6 * delta^2 / (144 * a^4 * B^2 * h^3) = Gc
    => 27 * E * h^3 * delta^2 / (144 * a^4) = Gc
    => 3 * E * h^3 * delta^2 / (16 * a^4) = Gc
    => a^4 = 3 * E * h^3 * delta^2 / (16 * Gc)
    => a = (3 * E * h^3 * delta^2 / (16 * Gc))^(1/4)
    """
    delta_crit = dcb_critical_displacement(A, E, B, h, Gc)

    if delta <= delta_crit:
        return A

    a_propagating = (3.0 * E * h**3 * delta**2 / (16.0 * Gc))**0.25
    return max(a_propagating, A)


def dcb_response(delta_max=None, n_points=300, A=DCB_A, E=DCB_E, B=DCB_B,
                 h=DCB_h, Gc=DCB_Gc):
    """Complete DCB force-displacement and crack-length response.

    Returns
    -------
    result : dict with keys: delta, force, crack_length, delta_crit, F_crit.
    """
    delta_crit = dcb_critical_displacement(A, E, B, h, Gc)
    F_crit = dcb_critical_force(A, E, B, h, Gc)

    if delta_max is None:
        delta_max = delta_crit * 4.0

    delta_vals = np.linspace(0, delta_max, n_points)
    forces = []
    crack_lengths = []

    for d in delta_vals:
        a = dcb_crack_length_at_delta(d, A, E, B, h, Gc)
        F = dcb_force_from_delta(d, a, E, B, h)
        forces.append(F)
        crack_lengths.append(a)

    return {
        "delta": delta_vals,
        "force": np.array(forces),
        "crack_length": np.array(crack_lengths),
        "delta_crit": delta_crit,
        "F_crit": F_crit,
        "A": A,
    }
