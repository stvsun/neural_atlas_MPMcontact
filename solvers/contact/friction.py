"""Regularized Coulomb friction for MPM contact.

Implements per-particle tangential friction in physical space, suitable
for explicit MPM where the friction force is computed each step from the
current particle velocity and the magnitude of the normal contact force.

Mathematical formulation (see Section 6.2 of
``contact_atlas/03_mathematical_theory.md``):

    v_T = v - (v . n) * n                              # tangential velocity
    f_T = -mu * |f_N| * v_T / sqrt(|v_T|^2 + eps_T^2)  # regularized Coulomb

The regularization gives a smooth tangent-space restoring force whose
magnitude approaches ``mu * |f_N|`` in the slip limit
(``|v_T| >> eps_T``) and goes linearly through zero in the stick limit
(``|v_T| << eps_T``).  Friction always opposes the tangential motion.

The returned force has the same shape and units as
``compute_contact_force`` in ``solvers/contact/penalty.py`` (newtons),
so it can be added directly to the normal contact force before passing
the sum to ``particle_to_grid(..., contact_force=...)``.
"""

from typing import Tuple

import torch


def decompose_velocity(
    velocity: torch.Tensor,
    normal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a velocity field into normal and tangential components.

    ``v = v_n_scalar.unsqueeze(1) * normal + v_t_vec``

    Parameters
    ----------
    velocity : torch.Tensor
        (N, 3) velocities in physical space.
    normal : torch.Tensor
        (N, 3) unit outward normals.

    Returns
    -------
    v_n_scalar : torch.Tensor
        (N,) signed projection ``v . n``.
    v_t_vec : torch.Tensor
        (N, 3) tangential component perpendicular to ``normal``.
    """
    v_n_scalar = (velocity * normal).sum(dim=1)              # (N,)
    v_t_vec = velocity - v_n_scalar.unsqueeze(1) * normal    # (N, 3)
    return v_n_scalar, v_t_vec


def compute_friction_force(
    velocity: torch.Tensor,
    normal: torch.Tensor,
    normal_force_magnitude: torch.Tensor,
    mu: float,
    epsilon_t: float = 1e-6,
) -> torch.Tensor:
    """Regularized Coulomb friction force per particle.

    ``f_T = -mu * |f_N| * v_T / sqrt(|v_T|^2 + eps_T^2)``

    Parameters
    ----------
    velocity : torch.Tensor
        (N, 3) particle velocity in physical space.
    normal : torch.Tensor
        (N, 3) unit outward contact normal.
    normal_force_magnitude : torch.Tensor
        (N,) magnitude ``|f_N|`` of the normal contact force on each
        particle.  Particles not in contact should have ``|f_N| = 0``,
        which automatically zeroes their friction.
    mu : float
        Coulomb friction coefficient (>= 0).
    epsilon_t : float
        Regularization parameter (smooths the stick/slip transition over
        a velocity scale of order ``epsilon_t``).

    Returns
    -------
    f_T : torch.Tensor
        (N, 3) friction force per particle in physical space.  Same
        units and convention as ``compute_contact_force``.
    """
    if mu <= 0.0:
        return torch.zeros_like(velocity)

    # Tangential velocity component
    _, v_t = decompose_velocity(velocity, normal)            # (N, 3)
    v_t_sq = (v_t * v_t).sum(dim=1, keepdim=True)            # (N, 1)

    # Smooth direction:  v_T / sqrt(|v_T|^2 + eps^2)
    smooth_dir = v_t / torch.sqrt(v_t_sq + epsilon_t * epsilon_t)

    # Friction opposes tangential velocity, scaled by mu * |f_N|
    return -mu * normal_force_magnitude.unsqueeze(1) * smooth_dir
