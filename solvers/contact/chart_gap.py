"""Level-set-free contact detection via radial boundary charts (3-D).

This is the 3-D lift of the 2-D ``supershape.py`` oracle, packaged behind the
*exact* same ``(gap, normal)`` contract as ``solvers/contact/gap.py::evaluate_gap``
so it is a drop-in replacement for the neural-SDF detector — penalty / friction /
augmented-Lagrangian forces and the broad-phase culler are untouched.

Idea (no ambient signed distance field anywhere)
------------------------------------------------
A star-shaped body is represented level-set-free by a **radial boundary chart**

    rho : S^2 -> R+        (a height field on the sphere of directions),

together with a center ``c`` and an orientation ``Q`` (body axes as columns, in
world coordinates).  In the body frame ``d = Q^T (x - c)`` with ``r = |d|`` and
``dhat = d/r``, the implicit boundary is

    F(x) = r - rho(dhat)            ( F < 0 inside, F > 0 outside ),

so the **gap** is ``gap = r - rho(dhat)`` and the (matched / conservative)
**normal** is the unit ``grad_x F``:

    grad_d F = dhat - grad_S rho / r,     n_world = normalize( Q @ grad_d F ),

where ``grad_S rho`` is the **tangential** (projected) gradient of ``rho`` on
``S^2``: ``grad_S rho = (I - dhat dhat^T) grad_dhat rho``.  The projection is the
load-bearing correctness step for any non-spherical chart (a raw autograd
gradient of ``rho`` w.r.t. a unit vector carries a spurious radial component);
``evaluate_gap_chart`` performs it and the test-suite asserts ``grad_S rho . dhat ~ 0``.

This is the *conservative* normal: the penalty force ``-eps_n <F> n V`` is the
gradient of the potential ``0.5 eps_n <F>^2 V``.  It is exact on the sphere at all
depths, and equals the true surface normal as the query approaches the surface.

Honest caveats (mirrors supershape.py — read before trusting the gap as a distance)
----------------------------------------------------------------------------------
- The radial gap is **not** a Euclidean distance: ``gap_rad = gap_perp / cos(alpha)
  + O(gap^2)`` where ``alpha`` is the angle between the ray ``dhat`` and the true
  surface normal.  Since ``1/cos(alpha) >= 1`` and the sign is preserved, its
  *magnitude* is conservative-large: ``|gap_rad| >= |gap_perp|`` **always** (the
  along-ray distance to the boundary is an upper bound on the true minimum
  distance).  Equivalently: an exterior point reads a slightly *larger* gap, a
  penetrating point a slightly *deeper* penetration (a marginally stiffer
  penalty) — it is NOT uniformly "soft".  Crucially the sign is exact, so the
  **active set ``gap < 0`` is identical to the true SDF's** for star-shaped
  bodies.  ``closest_point_refine_chart`` recovers the true perpendicular gap +
  surface normal (verification only; never in the integrator — the foot jumps
  across the medial axis, which is non-smooth and would leak energy).
- Valid only for **star-shaped** bodies (every ray from ``c`` hits the boundary
  once).  A plane / half-space is not star-shaped about any finite center — keep
  floors on the SDF path.

Stage scope: this module ships **analytic** ``rho`` (``SphereRho``,
``SuperquadricRho``).  A trained ``NeuralRho(net)`` (the neural chart that replaces
the neural SDF) is the deferred Stage-1 work; it would plug into the identical
``evaluate_gap_chart`` path (the projected-gradient step becomes load-bearing there).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _dtype_eps(dtype: torch.dtype) -> float:
    """dtype-aware epsilon for norm clamps (matches gap.py)."""
    return max(torch.finfo(dtype).eps, 1e-12)


# ---------------------------------------------------------------------------
# Radial charts  rho : S^2 -> R+
# ---------------------------------------------------------------------------

class RadialChart:
    """Star-shaped body as (center, orientation, radial function ``rho``).

    Parameters
    ----------
    center : sequence or torch.Tensor
        (3,) body center ``c`` in world coordinates.
    orientation : torch.Tensor, optional
        (3, 3) rotation ``Q`` whose **columns** are the body axes in world
        coordinates (world->body is ``Q^T``).  ``None`` means identity.

    Subclasses implement :meth:`radius` (a torch-differentiable map from unit
    directions ``dhat`` (N, 3) to radii (N,)) and :meth:`bounding_radius`.
    """

    def __init__(self, center, orientation: Optional[torch.Tensor] = None):
        self.center = torch.as_tensor(center, dtype=torch.float64)
        if orientation is None:
            self.Q = torch.eye(3, dtype=torch.float64)
        else:
            self.Q = torch.as_tensor(orientation, dtype=torch.float64)

    # -- to be implemented by subclasses ----------------------------------
    def radius(self, dhat: torch.Tensor) -> torch.Tensor:
        """Surface radius ``rho(dhat)`` for unit directions ``dhat`` (N, 3) -> (N,).

        Must be differentiable w.r.t. ``dhat`` (autograd) so the normal can be
        formed.  A constant ``rho`` (sphere) is allowed — ``evaluate_gap_chart``
        uses ``allow_unused`` so a zero gradient is handled.
        """
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """Max ``rho`` over ``S^2`` (for the broad-phase seed support radius)."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    def _frame_to(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (center, Q) cast to ``x``'s dtype/device."""
        return self.center.to(x), self.Q.to(x)


class SphereRho(RadialChart):
    """Sphere of radius ``R``: ``rho(dhat) = R`` (gradient identically zero).

    With this chart ``evaluate_gap_chart`` reproduces the analytic sphere SDF
    ``phi = |x - c| - R`` and its radial normal to machine precision, at all
    depths — the headline equivalence benchmark.
    """

    def __init__(self, R: float, center=(0.0, 0.0, 0.0),
                 orientation: Optional[torch.Tensor] = None):
        super().__init__(center, orientation)
        self.R = float(R)

    def radius(self, dhat: torch.Tensor) -> torch.Tensor:
        # Constant; shape (N,).  Does not depend on dhat (zero gradient).
        return torch.full((dhat.shape[0],), self.R, dtype=dhat.dtype,
                          device=dhat.device)

    def bounding_radius(self) -> float:
        return self.R


class SuperquadricRho(RadialChart):
    """Superellipsoid ``sum_i |x_i / S_i|^e = 1`` as a radial chart.

    Along a unit direction ``dhat`` the radius is the closed form

        rho(dhat) = ( sum_i |dhat_i / S_i|^e )^{-1/e}.

    ``exponent == 2`` is a smooth ellipsoid (``C^inf``; use it for the
    finite-difference conservativeness check).  ``e > 2`` is a rounded box,
    ``0 < e < 2`` a pinched star — all star-shaped about the center.

    Parameters
    ----------
    semi_axes : sequence of 3 floats
        ``(A, B, C)`` semi-axis lengths.
    exponent : float
        Superquadric exponent ``e`` (default 2 = ellipsoid).
    """

    def __init__(self, semi_axes=(1.0, 1.0, 1.0), exponent: float = 2.0,
                 center=(0.0, 0.0, 0.0), orientation: Optional[torch.Tensor] = None):
        super().__init__(center, orientation)
        self.semi_axes = torch.as_tensor(semi_axes, dtype=torch.float64)
        self.exponent = float(exponent)

    def radius(self, dhat: torch.Tensor) -> torch.Tensor:
        S = self.semi_axes.to(dhat)
        e = self.exponent
        eps = _dtype_eps(dhat.dtype)
        if e == 2.0:
            # Smooth form (squares avoid the |.| kink at axis planes).
            base = ((dhat / S) ** 2).sum(dim=1)
        else:
            base = (dhat.abs() / S).pow(e).sum(dim=1)
        return base.clamp_min(eps) ** (-1.0 / e)

    def bounding_radius(self) -> float:
        return float(self.semi_axes.max())


# ---------------------------------------------------------------------------
# Detector  (matches solvers/contact/gap.py::evaluate_gap)
# ---------------------------------------------------------------------------

def evaluate_gap_chart(
    x_candidates: torch.Tensor,
    chart: RadialChart,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate gap and contact normal from a radial boundary chart.

    Drop-in twin of :func:`solvers.contact.gap.evaluate_gap`: same shapes, same
    sign convention (``gap < 0`` => penetration), same ``torch.enable_grad()``
    behaviour (safe to call inside an outer ``torch.no_grad()`` block), returns
    **detached** tensors.

    Parameters
    ----------
    x_candidates : torch.Tensor
        (N, 3) candidate contact points in physical (world) space.
    chart : RadialChart
        The body's radial boundary chart.

    Returns
    -------
    gap : torch.Tensor
        (N,) signed radial gap ``|d| - rho(dhat)``.  ``gap < 0`` => penetration.
    normal : torch.Tensor
        (N, 3) unit outward (matched/conservative) normals ``grad_x F / |grad_x F|``.
    """
    if x_candidates.numel() == 0:
        return x_candidates.new_zeros(0), x_candidates.new_zeros(0, 3)

    x = x_candidates
    eps = _dtype_eps(x.dtype)
    c, Q = chart._frame_to(x)

    with torch.enable_grad():
        # Body-frame coordinates (no grad needed through the rigid transform).
        d = (x.detach() - c) @ Q                         # (N, 3)
        r = d.norm(dim=1, keepdim=True).clamp(min=eps)   # (N, 1)
        dhat = d / r                                     # (N, 3)

        # rho and its ambient gradient w.r.t. the unit direction.
        dhat_g = dhat.detach().requires_grad_(True)
        rho = chart.radius(dhat_g)
        if rho.dim() > 1:                                # accept (N, 1) -> (N,)
            rho = rho.squeeze(-1)
        # A fully constant rho (e.g. a sphere) is disconnected from the graph;
        # autograd.grad would raise, so short-circuit to a zero gradient.
        if rho.requires_grad:
            g_amb = torch.autograd.grad(
                rho, dhat_g,
                grad_outputs=torch.ones_like(rho),
                create_graph=False, retain_graph=False,
                allow_unused=True,
            )[0]
        else:
            g_amb = None
    if g_amb is None:                                    # constant rho (sphere)
        g_amb = torch.zeros_like(dhat)

    # Tangential (projected) surface gradient: remove the radial component.
    g_S = g_amb - (g_amb * dhat).sum(dim=1, keepdim=True) * dhat

    # grad_d F = dhat - grad_S rho / r  ;  rotate body -> world ; normalize.
    grad_d = dhat - g_S / r                              # (N, 3)
    grad_x = grad_d @ Q.t()                              # (N, 3)
    normal = grad_x / grad_x.norm(dim=1, keepdim=True).clamp(min=eps)

    gap = (r.squeeze(-1) - rho)
    return gap.detach(), normal.detach()


# ---------------------------------------------------------------------------
# Verification-only refine: true perpendicular gap + surface normal
# ---------------------------------------------------------------------------

def _make_tangent_frame(dhat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors spanning the tangent plane of ``S^2`` at each ``dhat``."""
    # Pick a helper axis not parallel to dhat, then Gram-Schmidt.
    helper = torch.zeros_like(dhat)
    # use +x unless dhat is ~+/-x, in which case use +y
    near_x = dhat[:, 0].abs() > 0.9
    helper[:, 0] = (~near_x).to(dhat.dtype)
    helper[:, 1] = near_x.to(dhat.dtype)
    e1 = helper - (helper * dhat).sum(1, keepdim=True) * dhat
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp(min=1e-12)
    e2 = torch.cross(dhat, e1, dim=1)
    return e1, e2


def _surface_point_and_normal(chart: RadialChart, u: torch.Tensor,
                              x_ref: torch.Tensor):
    """For body-frame unit directions ``u`` (N, 3): world surface point and the
    true outward unit surface normal at the foot ``rho(u) u``.

    The on-surface normal is ``normalize(u - grad_S rho(u) / rho(u))`` rotated to
    world (same formula as the matched normal evaluated *at* the surface, where it
    equals the exact surface normal).
    """
    eps = _dtype_eps(u.dtype)
    c, Q = chart._frame_to(x_ref)
    with torch.enable_grad():
        u_g = u.detach().requires_grad_(True)
        rho = chart.radius(u_g)
        if rho.dim() > 1:
            rho = rho.squeeze(-1)
        if rho.requires_grad:
            g_amb = torch.autograd.grad(
                rho, u_g, grad_outputs=torch.ones_like(rho),
                allow_unused=True,
            )[0]
        else:
            g_amb = None
    if g_amb is None:
        g_amb = torch.zeros_like(u)
    rho_d = rho.detach()
    g_S = g_amb - (g_amb * u).sum(1, keepdim=True) * u
    foot_body = rho_d.unsqueeze(1) * u                          # (N, 3)
    foot_world = foot_body @ Q.t() + c
    nb = u - g_S / rho_d.clamp(min=eps).unsqueeze(1)            # body normal
    n_world = nb @ Q.t()
    n_world = n_world / n_world.norm(dim=1, keepdim=True).clamp(min=eps)
    return foot_world, n_world, rho_d


def closest_point_refine_chart(
    x_candidates: torch.Tensor,
    chart: RadialChart,
    cap: float = 0.6,
    n_gamma: int = 9,
    n_beta: int = 16,
    n_refine: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """True (perpendicular) signed gap + surface normal — VERIFICATION ONLY.

    Minimises ``|x - surface(u)|`` over directions ``u`` in a geodesic cap of
    angular radius ``cap`` around the radial direction ``dhat``, via a coarse
    spherical-cap scan refined by ``n_refine`` shrinking re-scans (no Gauss-Newton
    needed at the few-percent tolerance this is checked to).

    NOT used by the time integrator: the perpendicular foot jumps across the
    medial axis, which is non-smooth and would leak energy.  The radial mode
    (:func:`evaluate_gap_chart`) is the integrator path.

    Returns
    -------
    gap_perp : torch.Tensor
        (N,) signed perpendicular gap (sign from the star-shaped inside test).
    n_surf : torch.Tensor
        (N, 3) true outward unit surface normal at the foot.
    """
    if x_candidates.numel() == 0:
        return x_candidates.new_zeros(0), x_candidates.new_zeros(0, 3)

    x = x_candidates
    eps = _dtype_eps(x.dtype)
    c, Q = chart._frame_to(x)
    d = (x - c) @ Q
    r = d.norm(dim=1, keepdim=True).clamp(min=eps)
    dhat = d / r                                              # (N, 3)

    N = x.shape[0]
    best_d2 = x.new_full((N,), float("inf"))
    best_u = dhat.clone()
    half = cap
    for _ in range(n_refine + 1):
        gammas = torch.linspace(0.0, half, n_gamma, dtype=x.dtype, device=x.device)
        betas = torch.linspace(0.0, 2.0 * math.pi, n_beta + 1,
                               dtype=x.dtype, device=x.device)[:-1]
        # Build candidate directions as a geodesic cap around the *current* best.
        cb, sb = torch.cos(betas), torch.sin(betas)
        cg, sg = torch.cos(gammas), torch.sin(gammas)
        # local frame around best_u
        e1, e2 = _make_tangent_frame(best_u)
        for ig in range(n_gamma):
            # tangent direction sweep at this geodesic radius
            tang = cb.unsqueeze(1) * e1.unsqueeze(1) + sb.unsqueeze(1) * e2.unsqueeze(1)
            #   tang: (N, n_beta, 3)
            u = (cg[ig] * best_u.unsqueeze(1) + sg[ig] * tang)   # (N, n_beta, 3)
            u = u / u.norm(dim=2, keepdim=True).clamp(min=eps)
            rho = chart.radius(u.reshape(-1, 3)).reshape(N, -1)  # (N, n_beta)
            foot_body = rho.unsqueeze(2) * u                     # (N, n_beta, 3)
            foot_world = foot_body @ Q.t() + c                   # (N, n_beta, 3)
            d2 = ((x.unsqueeze(1) - foot_world) ** 2).sum(dim=2)  # (N, n_beta)
            d2min, j = d2.min(dim=1)                             # (N,)
            improve = d2min < best_d2
            if improve.any():
                best_d2 = torch.where(improve, d2min, best_d2)
                u_best_here = u[torch.arange(N), j]             # (N, 3)
                best_u = torch.where(improve.unsqueeze(1), u_best_here, best_u)
        half *= 0.35                                            # shrink the cap

    foot_world, n_surf, _ = _surface_point_and_normal(chart, best_u, x)
    dist = (x - foot_world).norm(dim=1)
    gap_now, _ = evaluate_gap_chart(x, chart)
    sign = torch.where(gap_now < 0, -torch.ones_like(dist), torch.ones_like(dist))
    return (sign * dist).detach(), n_surf.detach()
