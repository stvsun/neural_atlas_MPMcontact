"""Data structures for contact bodies and detected contact pairs."""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ContactBody:
    """One body participating in contact.

    Wraps a contact-detection source (a neural SDF **or** a level-set-free
    radial boundary chart) together with the chart seed geometry needed for
    broad-phase distance culling.

    The ``detector`` field selects which source the narrow phase uses:

    - ``"sdf"``   : evaluate ``sdf_net`` via :func:`solvers.contact.gap.evaluate_gap`
      (the SDF benchmark path — must supply ``sdf_net``).
    - ``"chart"`` : evaluate ``chart`` via
      :func:`solvers.contact.chart_gap.evaluate_gap_chart` (the level-set-free
      path — must supply ``chart``).

    Both produce the identical ``(gap, normal)`` contract, so penalty / friction /
    augmented-Lagrangian forces and the broad-phase culler are detector-agnostic.

    Attributes
    ----------
    body_id : int
        Unique identifier.
    sdf_net : torch.nn.Module, optional
        Neural SDF for this body (``phi: R^3 -> R``).  Required for ``detector="sdf"``.
    seeds : torch.Tensor
        (M, 3) chart seed points in physical space.
    support_radii : torch.Tensor
        (M,) chart support radii.
    chart : RadialChart, optional
        Radial boundary chart.  Required for ``detector="chart"``.
    detector : str
        ``"sdf"`` (default) or ``"chart"``.
    """

    body_id: int
    sdf_net: Optional[torch.nn.Module] = None
    seeds: Optional[torch.Tensor] = None
    support_radii: Optional[torch.Tensor] = None
    chart: Optional[object] = None          # RadialChart (avoid import cycle)
    detector: str = "sdf"

    def __post_init__(self):
        if self.detector not in ("sdf", "chart"):
            raise ValueError(
                f"detector must be 'sdf' or 'chart', got {self.detector!r}"
            )
        if self.detector == "sdf" and self.sdf_net is None:
            raise ValueError("detector='sdf' requires sdf_net")
        if self.detector == "chart" and self.chart is None:
            raise ValueError("detector='chart' requires chart")

    @classmethod
    def from_chart(cls, body_id: int, chart, margin: float = 0.0) -> "ContactBody":
        """Build a chart-detector body with broad-phase seeds from the bounding sphere.

        A radial chart is star-shaped about its center, so a single seed at
        ``chart.center`` with support radius ``chart.bounding_radius() + margin``
        is an exact broad-phase bound.  Use this when registering a chart body
        with ``SchwarzMPMSolver`` (whose broad-phase needs ``seeds`` /
        ``support_radii``); ``ContactManager.detect_mpm`` works without them.
        """
        center = torch.as_tensor(chart.center).reshape(1, 3)
        radius = torch.tensor(
            [float(chart.bounding_radius()) + float(margin)], dtype=center.dtype
        )
        return cls(
            body_id=body_id, chart=chart, seeds=center,
            support_radii=radius, detector="chart",
        )


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
