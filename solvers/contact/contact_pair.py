"""Data structures for contact bodies and detected contact pairs."""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ContactBody:
    """One body participating in contact.

    Wraps a neural SDF together with the chart seed geometry needed for
    broad-phase distance culling.

    Attributes
    ----------
    body_id : int
        Unique identifier.
    sdf_net : torch.nn.Module
        Neural SDF for this body (``phi: R^3 -> R``).
    seeds : torch.Tensor
        (M, 3) chart seed points in physical space.
    support_radii : torch.Tensor
        (M,) chart support radii.
    """

    body_id: int
    sdf_net: torch.nn.Module
    seeds: torch.Tensor
    support_radii: torch.Tensor


@dataclass
class ContactPair:
    """Result of narrow-phase contact detection for one chart.

    Stores per-particle contact data for particles on body B evaluated
    against the SDF of body A.

    Attributes
    ----------
    body_id_A : int
        Body whose SDF defines the obstacle surface.
    body_id_B : int
        Body whose particles are tested for penetration.
    chart_id_B : int
        Chart index on body B containing the candidate particles.
    particle_indices : torch.Tensor
        (N_active,) indices into the particle cloud of ``chart_id_B``.
    gap : torch.Tensor
        (N_active,) signed distance ``phi_A(x_p)``.
    normal : torch.Tensor
        (N_active, 3) unit outward normal of body A at each particle.
    x_phys : torch.Tensor
        (N_active, 3) physical positions where gap was evaluated.
    """

    body_id_A: int
    body_id_B: int
    chart_id_B: int
    particle_indices: torch.Tensor
    gap: torch.Tensor
    normal: torch.Tensor
    x_phys: torch.Tensor
