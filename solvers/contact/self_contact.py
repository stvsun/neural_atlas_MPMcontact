"""Self-contact for MPM particles on a single deformable body.

Self-contact is harder than inter-body contact because a body's own
particles are *expected* to be inside their own SDF (``phi < 0``
everywhere in the interior).  A naive "particle penetrates its own SDF"
detector would fire on every bulk particle, all the time.

This module uses an **initial-gap delta filter**: at construction time
the manager snapshots ``gap_0 = phi_A(X_0)`` for every particle.  A
particle is treated as self-contacting only when its *current* gap is
significantly more negative than its initial gap — that is, the
particle has moved notably deeper into its own body than where it
started.  Combined with a *surface filter* that excludes deep-interior
particles entirely, this distinguishes two physically different cases:

    * A bulk particle sitting at ``phi = -0.3`` for all time          → ignored
    * A surface particle whose gap was ``0`` and is now ``-0.1``      → active

The second case is the signature of a folding body whose surface has
moved inward onto another part of itself.

Scope & limitations
-------------------
* Uses the **reference** neural SDF, not a deformed one.  For small-to-
  moderate deformations this is accurate; for large deformations
  (large ``|det(F)| - 1``) the reference SDF is no longer a good model
  of the deformed surface and the detector degrades.
* The force direction is the current SDF gradient ``grad phi_A`` at the
  particle's position.  This pushes the particle back along the
  steepest outward direction of the reference SDF, which is the correct
  physics for the small-deformation regime.
* Only detects self-penetration along the SDF normal.  Tangential
  self-sliding (one surface of the body rubbing along another part of
  its own surface) still works through the standard friction module
  once self-contact has been identified.

See ``contact_atlas/03_mathematical_theory.md`` Section 8.3 for the
topological discussion of self-contact (H_1 birth events).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from solvers.contact.contact_pair import ContactBody
from solvers.contact.gap import evaluate_gap


@dataclass
class SelfContactState:
    """Per-particle baseline state captured at manager construction.

    Attributes
    ----------
    initial_gap : torch.Tensor
        (N,) SDF value at each particle's initial position.
    is_surface : torch.Tensor
        (N,) bool mask — True if the particle was on the body surface
        at initialization (``|initial_gap| < surface_tol``).
    """

    initial_gap: torch.Tensor
    is_surface: torch.Tensor


class SelfContactManager:
    """Detect and resolve self-contact for one deformable body.

    Parameters
    ----------
    body : ContactBody
        The body whose SDF is used for detection.
    initial_positions : torch.Tensor
        ``(N, 3)`` physical-space positions of every particle at t=0.
        These are used to snapshot the baseline ``phi_A(X_0)``.
    surface_tol : float
        ``|initial_gap| < surface_tol`` defines "surface particles".
        Particles with a deeper initial gap are considered bulk and
        never participate in self-contact.  Default ``1e-2``.
    penetration_delta : float
        Minimum additional penetration (relative to the initial gap)
        that triggers a self-contact event.  A particle is active when
        ``current_gap < initial_gap - penetration_delta``.  Default
        ``1e-2``.
    """

    def __init__(
        self,
        body: ContactBody,
        initial_positions: torch.Tensor,
        surface_tol: float = 1e-2,
        penetration_delta: float = 1e-2,
    ):
        if initial_positions.dim() != 2 or initial_positions.shape[1] != 3:
            raise ValueError(
                f"initial_positions must have shape (N, 3), got "
                f"{tuple(initial_positions.shape)}"
            )

        self.body = body
        self.surface_tol = float(surface_tol)
        self.penetration_delta = float(penetration_delta)

        # Snapshot baseline gap at t=0
        initial_gap, _ = evaluate_gap(initial_positions, body.sdf_net)
        is_surface = initial_gap.abs() < self.surface_tol
        self.state = SelfContactState(
            initial_gap=initial_gap,
            is_surface=is_surface,
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        x_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect self-contact at the particles' current positions.

        Parameters
        ----------
        x_current : torch.Tensor
            ``(N, 3)`` current physical-space positions.  Must be the
            same size as the ``initial_positions`` passed at
            construction time.

        Returns
        -------
        gap : torch.Tensor
            ``(N,)`` current SDF values.
        normal : torch.Tensor
            ``(N, 3)`` current outward unit normals.
        active : torch.Tensor
            ``(N,)`` bool mask indicating which particles are flagged
            as self-contacting.  A particle is active when it (a) was a
            surface particle at t=0, and (b) its current gap is at
            least ``penetration_delta`` more negative than its initial
            gap.
        """
        if x_current.shape != (self.state.initial_gap.shape[0], 3):
            raise ValueError(
                f"x_current shape {tuple(x_current.shape)} does not "
                f"match manager's particle count "
                f"{self.state.initial_gap.shape[0]}"
            )
        gap, normal = evaluate_gap(x_current, self.body.sdf_net)

        # Delta relative to initial gap — negative means "deeper than
        # where I started" — below -penetration_delta is the trigger.
        delta = gap - self.state.initial_gap
        penetrated = delta < -self.penetration_delta
        active = self.state.is_surface & penetrated
        return gap, normal, active

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_force(
        self,
        x_current: torch.Tensor,
        volume: torch.Tensor,
        epsilon_n: float,
    ) -> torch.Tensor:
        """Compute the self-contact penalty force per particle.

        Uses the same penalty functional as inter-body contact, but the
        force is zero for particles outside the active self-contact set.
        The force magnitude is driven by the **delta** between the
        current and initial gap (how much deeper the particle moved),
        not the absolute current gap — otherwise bulk particles that
        stay at their home depth would contribute large spurious forces.

        ``f_p = epsilon_n * max(0, -(current_gap - initial_gap)) * V_p * n``
        for active particles, zero otherwise.

        Parameters
        ----------
        x_current : torch.Tensor
            ``(N, 3)`` current positions.
        volume : torch.Tensor
            ``(N,)`` current particle volumes.
        epsilon_n : float
            Penalty stiffness.

        Returns
        -------
        f : torch.Tensor
            ``(N, 3)`` self-contact force per particle, in the same
            physical-space convention as ``compute_contact_force`` in
            ``solvers/contact/penalty.py``.
        """
        gap, normal, active = self.detect(x_current)
        delta = gap - self.state.initial_gap               # (N,)
        penetration = torch.clamp(-delta, min=0.0)          # (N,)
        # Zero out inactive (non-surface) particles
        penetration = torch.where(
            active, penetration, torch.zeros_like(penetration),
        )
        f_mag = epsilon_n * penetration * volume            # (N,)
        return f_mag.unsqueeze(1) * normal                  # (N, 3)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def n_surface_particles(self) -> int:
        return int(self.state.is_surface.sum().item())

    def n_active(self, x_current: torch.Tensor) -> int:
        _, _, active = self.detect(x_current)
        return int(active.sum().item())

    def max_delta_penetration(self, x_current: torch.Tensor) -> float:
        """Largest ``max(0, initial_gap - current_gap)`` over all surface
        particles — i.e., the biggest additional penetration since t=0.
        """
        gap, _, _ = self.detect(x_current)
        delta = gap - self.state.initial_gap
        pen = torch.clamp(-delta, min=0.0)
        pen = torch.where(
            self.state.is_surface, pen, torch.zeros_like(pen),
        )
        if pen.numel() == 0:
            return 0.0
        return float(pen.max().item())
