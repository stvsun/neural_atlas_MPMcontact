"""Gielis superformula boundary chart for nonconvex star-shaped particles (2D).

A supershape boundary is the analytic chart

    phi(theta) = c + R(alpha) @ ( rho(theta) * [cos theta, sin theta] ),

    rho(theta) = ( |cos(m theta/4)/a|^{n2} + |sin(m theta/4)/b|^{n3} )^{-1/n1}   (Gielis 2003).

rho is a single-valued function of theta, so the body is STAR-SHAPED about its center
even when nonconvex -- every ray from c hits the boundary once. This module provides:

- the boundary chart phi and its analytic tangent / outward normal;
- the INVERSE radial chart as a contact gap: g_B(p) = |p - c| - rho(psi), psi = angle of
  R(-alpha)(p - c), returned as a *matched pair* (g, grad_p g) so the penalty force
  f = -eps_n <g> grad_p g is conservative (radial mode);
- a bounded 1D closest-point refine giving the true perpendicular gap + surface normal
  (refined mode), which removes the radial-vs-perpendicular bias at the cost of
  re-introducing medial-axis nonsmoothness for points whose true foot leaves the bracket;
- area and polar moment of inertia by theta-quadrature.

Honest framing (see docs/contact_verification_manual.md, CV-5): the *gap* is a single-body
inverse radial chart; the *transition map* is the boundary-to-boundary correspondence
theta_A -> psi_B = (phi_B^{-1} o phi_A)(theta_A). The radial gap is a valid, sign-correct,
C^1-smooth penalty gap but it is NOT a distance/Eikonal function: it overestimates the
perpendicular gap as g_rad = g_perp/cos(alpha) + O(gap^2) at the radial foot, and |grad g| can be
large (~1e4) in deep concave valleys. The bounded closest_point_refine recovers the perpendicular
distance (verification-only; not used by the time integrator).

Pure numpy (mirrors the style of postprocessing/contact_fields.py); the penalty/friction
*formulas* used by the driver mirror solvers/contact/{penalty,friction}.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class SuperParams:
    """Gielis superformula parameters."""
    m: float = 6.0
    n1: float = 0.7
    n2: float = 0.7
    n3: float = 0.7
    a: float = 1.0
    b: float = 1.0
    scale: float = 1.0     # uniform size multiplier on rho


def _rot(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def radius(theta, p: SuperParams):
    """Superformula radius rho(theta) (vectorized)."""
    theta = np.asarray(theta, dtype=float)
    u = p.m * theta / 4.0
    ca = np.cos(u) / p.a
    sb = np.sin(u) / p.b
    S = np.abs(ca) ** p.n2 + np.abs(sb) ** p.n3
    return p.scale * np.clip(S, _EPS, None) ** (-1.0 / p.n1)


def dradius_dtheta(theta, p: SuperParams):
    """Analytic d rho / d theta (vectorized; clamped bases for finite cusps)."""
    theta = np.asarray(theta, dtype=float)
    u = p.m * theta / 4.0
    du = p.m / 4.0
    ca = np.cos(u) / p.a
    sb = np.sin(u) / p.b
    S = np.clip(np.abs(ca) ** p.n2 + np.abs(sb) ** p.n3, _EPS, None)
    # Each branch's derivative is singular at its own cusp (base -> 0 with exponent < 1).
    # Zero out the singular branch AT the cusp (its true contribution there is a one-sided
    # limit; the smooth branch dominates rho'); just off-cusp the large value is physical.
    tol = 1e-9
    dA = p.n2 * np.clip(np.abs(ca), tol, None) ** (p.n2 - 1.0) * np.sign(ca) * (-np.sin(u) / p.a)
    dB = p.n3 * np.clip(np.abs(sb), tol, None) ** (p.n3 - 1.0) * np.sign(sb) * (np.cos(u) / p.b)
    dA = np.where(np.abs(ca) < tol, 0.0, dA)
    dB = np.where(np.abs(sb) < tol, 0.0, dB)
    dS_dtheta = (dA + dB) * du
    return p.scale * (-1.0 / p.n1) * S ** (-1.0 / p.n1 - 1.0) * dS_dtheta


def boundary(theta, c, alpha, p: SuperParams):
    """Physical boundary points phi(theta), shape (N, 2)."""
    theta = np.asarray(theta, dtype=float)
    rho = radius(theta, p)
    lx = rho * np.cos(theta)
    ly = rho * np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    x = c[0] + lx * ca - ly * sa
    y = c[1] + lx * sa + ly * ca
    return np.stack([x, y], axis=-1)


def tangent(theta, alpha, p: SuperParams):
    """d phi / d theta in world frame, shape (N, 2)."""
    theta = np.asarray(theta, dtype=float)
    rho = radius(theta, p)
    drho = dradius_dtheta(theta, p)
    lx = drho * np.cos(theta) - rho * np.sin(theta)
    ly = drho * np.sin(theta) + rho * np.cos(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.stack([lx * ca - ly * sa, lx * sa + ly * ca], axis=-1)


def outward_normal(theta, c, alpha, p: SuperParams):
    """Unit outward normal at phi(theta) (tangent rotated -90, sign-fixed), (N, 2)."""
    t = tangent(theta, alpha, p)
    n = np.stack([t[..., 1], -t[..., 0]], axis=-1)        # rotate -90 deg
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), _EPS, None)
    pts = boundary(theta, c, alpha, p)
    out = pts - np.asarray(c)
    flip = np.sum(n * out, axis=-1) < 0.0
    n[flip] *= -1.0
    return n


def _to_body(p_pts, c, alpha):
    """World points -> body frame coords d = R(-alpha)(p - c)."""
    d = np.asarray(p_pts, dtype=float) - np.asarray(c)
    ca, sa = np.cos(alpha), np.sin(alpha)
    dx = d[..., 0] * ca + d[..., 1] * sa
    dy = -d[..., 0] * sa + d[..., 1] * ca
    return dx, dy


def radial_gap(p_pts, c, alpha, p: SuperParams):
    """Inverse radial chart gap g = |p-c| - rho(psi) and its matched world gradient.

    Returns (g, grad_g) with grad_g shape (N, 2). g < 0 => inside the body.
    The penalty force -eps_n <g> grad_g is conservative (radial mode normal = grad_g/|grad_g|).
    """
    dx, dy = _to_body(p_pts, c, alpha)
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.clip(r, _EPS, None)
    psi = np.arctan2(dy, dx)
    rho = radius(psi, p)
    drho = dradius_dtheta(psi, p)
    g = r - rho
    # body-frame gradient: d/|d| (radial) - drho * grad psi ;  grad psi = (-dy, dx)/r^2
    gd_x = dx / r_safe - drho * (-dy) / (r_safe ** 2)
    gd_y = dy / r_safe - drho * (dx) / (r_safe ** 2)
    ca, sa = np.cos(alpha), np.sin(alpha)
    gw_x = gd_x * ca - gd_y * sa
    gw_y = gd_x * sa + gd_y * ca
    return g, np.stack([gw_x, gw_y], axis=-1)


def inside(p_pts, c, alpha, p: SuperParams):
    """Star-shaped inside test: True where g < 0 (valid for star-shaped bodies)."""
    g, _ = radial_gap(p_pts, c, alpha, p)
    return g < 0.0


def closest_point_refine(p_pt, c, alpha, p: SuperParams, bracket_frac=0.15, n_scan=201):
    """Bounded 1D closest-point on the boundary near the radial angle (refined mode).

    Returns (psi_star, foot, signed_perp_gap, surf_normal) for a single point p_pt (2,).
    Bracket is psi0 +/- bracket_frac*2pi to suppress medial-axis foot-jumps.

    NB: verification-only -- the time integrator (supershape_cam_drive.py) uses the radial
    mode (radial-gap gradient normal), not this refine.
    """
    p_pt = np.asarray(p_pt, dtype=float)
    dx, dy = _to_body(p_pt[None, :], c, alpha)
    psi0 = float(np.arctan2(dy[0], dx[0]))
    half = bracket_frac * 2.0 * np.pi
    psis = np.linspace(psi0 - half, psi0 + half, n_scan)
    feet = boundary(psis, c, alpha, p)
    d2 = np.sum((feet - p_pt) ** 2, axis=1)
    k = int(np.argmin(d2))
    # parabolic refine around the scan minimum (equally spaced: x* = b - 0.5 h (fc-fa)/(fa-2fb+fc))
    lo = max(k - 1, 0)
    hi = min(k + 1, n_scan - 1)
    a, b3, cc = psis[lo], psis[k], psis[hi]
    fa, fb, fc = d2[lo], d2[k], d2[hi]
    h = 0.5 * (cc - a)
    denom = (fa - 2 * fb + fc)
    psi_star = b3 if abs(denom) < _EPS else b3 - 0.5 * h * (fc - fa) / denom
    psi_star = float(np.clip(psi_star, a, cc))
    foot = boundary(np.array([psi_star]), c, alpha, p)[0]
    dist = float(np.linalg.norm(p_pt - foot))
    sign = -1.0 if inside(p_pt[None, :], c, alpha, p)[0] else 1.0
    n = outward_normal(np.array([psi_star]), c, alpha, p)[0]
    return psi_star, foot, sign * dist, n


def area(p: SuperParams, n_quad=4000):
    """Polar area A = 1/2 integral rho^2 dtheta."""
    th = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    rho = radius(th, p)
    return 0.5 * np.trapz(np.concatenate([rho ** 2, rho[:1] ** 2]),
                          np.concatenate([th, [2.0 * np.pi]]))


def polar_inertia(p: SuperParams, density=1.0, n_quad=4000):
    """Mass and polar second moment about the center: I = density * 1/4 integral rho^4 dtheta."""
    th = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    rho = radius(th, p)
    thc = np.concatenate([th, [2.0 * np.pi]])
    A = 0.5 * np.trapz(np.concatenate([rho ** 2, rho[:1] ** 2]), thc)
    J = 0.25 * np.trapz(np.concatenate([rho ** 4, rho[:1] ** 4]), thc)
    return density * A, density * J


def centroid(p: SuperParams, n_quad=4000):
    """Numeric centroid (should be ~0 for symmetric supershapes)."""
    th = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    rho = radius(th, p)
    # centroid of polar region: (1/A) integral over area of (x,y)
    # x-moment = integral_theta integral_0^rho (r cos) r dr dtheta = integral rho^3/3 cos dtheta
    A = 0.5 * np.trapz(np.concatenate([rho ** 2, rho[:1] ** 2]), np.concatenate([th, [2 * np.pi]]))
    mx = np.trapz(np.concatenate([rho ** 3 / 3 * np.cos(th), [rho[0] ** 3 / 3]]),
                  np.concatenate([th, [2 * np.pi]]))
    my = np.trapz(np.concatenate([rho ** 3 / 3 * np.sin(th), [rho[0] ** 3 / 3 * np.sin(0)]]),
                  np.concatenate([th, [2 * np.pi]]))
    return np.array([mx / A, my / A])
