"""Contact detection manager with broad-phase and narrow-phase.

Broad-phase culls chart pairs by seed-to-seed distance (mirrors the
neighbour graph construction in ``schwarz_mpm.py``).  Narrow-phase
evaluates the neural SDF at particle positions via ``evaluate_gap``.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch

from solvers.contact.contact_pair import ContactBody, ContactPair
from solvers.contact.gap import evaluate_gap


class ContactManager:
    """Detect contacts between multiple bodies.

    Parameters
    ----------
    bodies : list of ContactBody
        Bodies that may come into contact.
    margin : float
        Extra distance added to support-radius sum during broad-phase.
    """

    def __init__(
        self,
        bodies: List[ContactBody],
        margin: float = 0.1,
    ):
        self.bodies = {b.body_id: b for b in bodies}
        self.margin = margin

    # ------------------------------------------------------------------
    # Broad phase
    # ------------------------------------------------------------------

    def broad_phase(
        self,
        body_A: ContactBody,
        body_B: ContactBody,
    ) -> List[Tuple[int, int]]:
        """Return chart-index pairs ``(i_A, j_B)`` that may be in contact.

        A pair is kept when the seed-to-seed distance is less than the
        sum of their support radii plus ``self.margin``.

        Implementation note: fully vectorized (no Python double loop),
        so the runtime is ``O(M_A * M_B)`` tensor ops rather than
        ``M_A * M_B`` Python-side ``.item()`` synchronisations — this
        matters a lot for atlases with hundreds of charts.
        """
        seeds_A = body_A.seeds                              # (M_A, 3)
        seeds_B = body_B.seeds                              # (M_B, 3)
        radii_A = body_A.support_radii                      # (M_A,)
        radii_B = body_B.support_radii                      # (M_B,)

        if seeds_A.numel() == 0 or seeds_B.numel() == 0:
            return []

        # Pairwise distances via broadcasting
        diff = seeds_A.unsqueeze(1) - seeds_B.unsqueeze(0)  # (M_A, M_B, 3)
        dist = diff.norm(dim=-1)                            # (M_A, M_B)

        # Sum of support radii per pair
        r_sum = radii_A.unsqueeze(1) + radii_B.unsqueeze(0)  # (M_A, M_B)

        mask = dist < (r_sum + self.margin)                  # (M_A, M_B) bool
        idx = mask.nonzero(as_tuple=False)                   # (K, 2)
        return [(int(i), int(j)) for i, j in idx.tolist()]

    # ------------------------------------------------------------------
    # Narrow phase (MPM particles)
    # ------------------------------------------------------------------

    def detect_mpm(
        self,
        body_A: ContactBody,
        body_B_id: int,
        chart_id_B: int,
        x_phys: torch.Tensor,
    ) -> Optional[ContactPair]:
        """Narrow-phase detection for MPM particles on one chart.

        Evaluates ``phi_A`` at the physical positions of body B's
        particles and returns a :class:`ContactPair` if any particle
        penetrates.

        Parameters
        ----------
        body_A : ContactBody
            The obstacle body whose SDF is evaluated.
        body_B_id : int
            Identifier of the body that owns the particles.
        chart_id_B : int
            Chart index on body B.
        x_phys : torch.Tensor
            (N, 3) physical-space particle positions.

        Returns
        -------
        ContactPair or None
            ``None`` when no penetration is detected.
        """
        gap, normal = evaluate_gap(x_phys, body_A.sdf_net)
        active = gap < 0
        if not active.any():
            return None
        idx = torch.where(active)[0]
        return ContactPair(
            body_id_A=body_A.body_id,
            body_id_B=body_B_id,
            chart_id_B=chart_id_B,
            particle_indices=idx,
            gap=gap[idx],
            normal=normal[idx],
            x_phys=x_phys[idx],
        )
