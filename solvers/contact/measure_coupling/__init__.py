"""Surface-to-surface contact as a measure coupling — gap & traction FIELDS.

This package replaces node-to-surface, tributary-lumped contact (each slave node projects
independently to its closest master point; diagonal penalty tangent) with a measure-coupling
formulation that produces a continuous **gap field** ``g_N(xi)`` on the slave surface, a pointwise
**traction field** ``p_N n + t_T``, and **consistent Galerkin nodal forces**

    f_I = sum_q w_q J_s N_I(xi_q) [ p_N n_s + t_T ]

with a 2-node-coupled (mortar-structured) tangent ``K_c = sum_q w_q J_s N_I N_J eps_n (n (x) n)``.
The marginal/mass constraint of the coupling is what makes the traction a true field and lets the
assembly pass the contact patch test (a uniform pressure across non-matching meshes), which the
lumped node-to-surface scheme fails.

Modules
-------
- :mod:`quadrature`  — 1-D Gauss-Legendre + P1 boundary shape functions (surface integration).
- :mod:`coupling`    — :class:`MonotoneCoupling1D` (exact 1-D optimal transport); Sinkhorn (later).
- :mod:`gap_field`   — :class:`GapField`: gap/slip fields at quadrature points from a coupling.
- :mod:`traction`    — :class:`TractionField`: pointwise normal/tangential law (wraps the existing
  ``penalty`` / ``friction`` kernels).
- :mod:`assembly`    — :func:`assemble_contact`: consistent Galerkin force + tangent.

See ``docs/contact_verification_manual.md`` (CV-1b) and the Bandeira-2004 audit
(``memory/bandeira-2004-contact-audit.md``) for the contact-mechanics context.
"""
from __future__ import annotations

from .quadrature import gauss_legendre_1d, lagrange_p1, segment_quadrature
from .coupling import MonotoneCoupling1D
from .gap_field import GapField
from .traction import TractionField
from .assembly import assemble_contact
from .two_body import assemble_two_body_contact

__all__ = [
    "gauss_legendre_1d",
    "lagrange_p1",
    "segment_quadrature",
    "MonotoneCoupling1D",
    "GapField",
    "TractionField",
    "assemble_contact",
    "assemble_two_body_contact",
]
