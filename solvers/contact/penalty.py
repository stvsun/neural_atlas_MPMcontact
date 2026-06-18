"""Penalty contact force computation and time-step restriction.

Implements the penalty normal contact force

    f_p = epsilon_n * max(0, -g_N) * n * V_p

and the CFL-like contact stability bound for explicit MPM

    dt <= safety * sqrt(m_min / epsilon_n).

See Section 11.1-11.2 of 03_mathematical_theory.md.
"""

import math

import torch


def compute_contact_force(
    gap: torch.Tensor,
    normal: torch.Tensor,
    volume: torch.Tensor,
    epsilon_n: float,
) -> torch.Tensor:
    """Compute penalty contact force per particle.

    Parameters
    ----------
    gap : torch.Tensor
        (N,) signed distance.  Negative means penetration.
    normal : torch.Tensor
        (N, 3) unit outward normal of the obstacle body.
    volume : torch.Tensor
        (N,) current particle volumes ``V_p``.
    epsilon_n : float
        Penalty stiffness (typical: ``E / h``).

    Returns
    -------
    f_contact : torch.Tensor
        (N, 3) contact force per particle in physical space.
        Zero for particles with ``gap >= 0``.
    """
    penetration = torch.clamp(-gap, min=0.0)       # max(0, -g)
    f_mag = epsilon_n * penetration * volume        # (N,)
    return f_mag.unsqueeze(1) * normal              # (N, 3)


def contact_stable_dt(
    epsilon_n: float,
    mass_min: float,
    safety_factor: float = 0.5,
) -> float:
    """Maximum stable time step for explicit MPM with penalty contact.

    Parameters
    ----------
    epsilon_n : float
        Penalty stiffness.
    mass_min : float
        Minimum particle mass in the system.
    safety_factor : float
        Multiplier (default 0.5).

    Returns
    -------
    dt_contact : float
    """
    if epsilon_n <= 0.0:
        return float("inf")
    return safety_factor * math.sqrt(mass_min / epsilon_n)
