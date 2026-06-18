"""Augmented Lagrangian (Uzawa) contact for MPM.

Maintains a persistent contact-pressure multiplier that accumulates
restraint across time steps so that the penalty parameter ``epsilon_n``
need not be driven to infinity to enforce non-penetration.

Mathematical formulation (see Section 4.2 of
``contact_atlas/03_mathematical_theory.md``).  Signed-distance convention:
``g < 0`` means penetration.

    Augmented pressure:    p_aug(g, lambda)  = max(0, lambda - eps_n * g)
    Multiplier update:     lambda^{k+1}      = max(0, lambda^k - eps_n * g(u^{k+1}))

When the constraint is violated (``g < 0``), ``-eps_n * g > 0`` and both
``p_aug`` and the new multiplier grow.  When the constraint is satisfied
(``g >= 0``), the ``max(0, .)`` clamp drops both terms back to zero.  At
convergence, ``lambda`` equals the true Lagrange multiplier (the contact
pressure) and the penetration vanishes.

For explicit MPM there is no Newton sub-iteration within a time step, so
``uzawa_update`` is called once per step after the gap has been
re-evaluated post-G2P.  The multiplier persists across time steps, which
lets it build up restraint when penetration is sustained and decay when
contact separates.

**Stable shape requirement**
    The persistent multiplier ``self.lam`` is indexed by position in the
    input ``gap`` tensor.  If the caller supplies a gap of a different
    shape (e.g., a dynamically-sized "active set" subset), the multiplier
    can no longer be matched particle-by-particle and is reset to zero
    with a warning.  The correct usage pattern is to **always pass the
    full per-particle gap** (zeros where particles are not in contact)
    so the multiplier preserves its per-particle state across calls.
    When the particle set genuinely changes (e.g., after
    ``_transfer_particles``), call :meth:`reset` explicitly.
"""

import warnings
from typing import Optional

import torch


class AugmentedLagrangianContact:
    """Augmented-Lagrangian (Uzawa) contact pressure tracker.

    Parameters
    ----------
    epsilon_n : float
        Penalty parameter.  Can be moderate (~ ``E / h``) because the
        multiplier absorbs the long-term restraint rather than relying
        on a stiff penalty.
    tol : float
        Penetration tolerance for ``converged``.
    """

    def __init__(self, epsilon_n: float, tol: float = 1e-6):
        self.epsilon_n = epsilon_n
        self.tol = tol
        self.lam: Optional[torch.Tensor] = None  # (N,) per particle

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the current multiplier (e.g., new problem instance)."""
        self.lam = None

    def _ensure_multiplier(self, gap: torch.Tensor) -> None:
        """Allocate or resize ``self.lam`` to match the gap shape.

        If a previous multiplier exists but the gap shape has changed,
        the accumulated state is lost — we warn loudly because this is
        almost always a caller bug (see the docstring at the top of the
        module).  Callers who deliberately want a fresh state should
        use :meth:`reset` instead.
        """
        if self.lam is None:
            self.lam = torch.zeros_like(gap)
            return
        if self.lam.shape != gap.shape:
            warnings.warn(
                f"AugmentedLagrangianContact: gap shape changed from "
                f"{tuple(self.lam.shape)} to {tuple(gap.shape)}; the "
                f"accumulated multiplier is being reset to zero.  "
                f"Pass the full per-particle gap with zeros in the "
                f"inactive set, or call reset() explicitly before "
                f"changing particle counts.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.lam = torch.zeros_like(gap)

    # ------------------------------------------------------------------
    # Core AL operations
    # ------------------------------------------------------------------

    def augmented_pressure(self, gap: torch.Tensor) -> torch.Tensor:
        """Effective contact pressure ``max(0, lambda - eps_n * g)``.

        Parameters
        ----------
        gap : torch.Tensor
            (N,) signed distance.  ``gap < 0`` is penetration.

        Returns
        -------
        p : torch.Tensor
            (N,) non-negative augmented pressure.
        """
        self._ensure_multiplier(gap)
        return torch.clamp(self.lam - self.epsilon_n * gap, min=0.0)

    def uzawa_update(self, gap: torch.Tensor) -> torch.Tensor:
        """Update the multiplier from the latest gap.

        ``lambda <- max(0, lambda - eps_n * g)``

        Should be called once per time step after the gap has been
        recomputed at the new particle positions.

        Returns the updated multiplier.
        """
        self._ensure_multiplier(gap)
        self.lam = torch.clamp(self.lam - self.epsilon_n * gap, min=0.0)
        return self.lam

    def compute_force(
        self,
        gap: torch.Tensor,
        normal: torch.Tensor,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        """Augmented-Lagrangian contact force per particle.

        ``f_p = p_aug(gap, lambda) * V_p * n``

        The shape and units match ``compute_contact_force`` in
        ``solvers/contact/penalty.py``, so the result plugs directly into
        ``particle_to_grid(..., contact_force=...)``.

        Parameters
        ----------
        gap : torch.Tensor
            (N,) signed distance.
        normal : torch.Tensor
            (N, 3) unit outward normal.
        volume : torch.Tensor
            (N,) particle volumes.

        Returns
        -------
        f : torch.Tensor
            (N, 3) contact force per particle in physical space.
        """
        p = self.augmented_pressure(gap)
        f_mag = p * volume  # (N,)
        return f_mag.unsqueeze(1) * normal  # (N, 3)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def max_penetration(self, gap: torch.Tensor) -> float:
        """Largest penetration depth ``max(0, -g).max()``.

        Returns 0 for an empty input.
        """
        if gap.numel() == 0:
            return 0.0
        return torch.clamp(-gap, min=0.0).max().item()

    def converged(self, gap: torch.Tensor) -> bool:
        """True iff the active-set penetration is below ``self.tol``.

        ``active`` is defined as ``self.lam > 0``.  When no constraint is
        active, the convergence test falls back to the global maximum
        penetration so the function is well-defined on the first call.
        """
        if self.lam is None:
            return self.max_penetration(gap) < self.tol
        active = self.lam > 0
        if not active.any():
            return self.max_penetration(gap) < self.tol
        pen_active = torch.clamp(-gap[active], min=0.0)
        return pen_active.max().item() < self.tol
