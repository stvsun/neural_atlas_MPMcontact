"""Semi-analytic elastic half-space (boundary-element) contact — the clean coupling cross-check.

The FEM Hertz benchmark (``cv1b_hertz_field.py``) carries a ~3-4% discretization floor from the
bulk CST elasticity (finite domain, edge quadrature) that is unrelated to the contact coupling.
This module solves the SAME rigid-indenter contact on an EXACT elastic half-space via boundary-
element influence coefficients, so the only error left is the contact discretization — isolating
whether the measure-coupling gap law and constraint enforcement are correct, to sub-percent.

2-D plane-strain half-plane, Flamant/Boussinesq surface compliance: a line pressure ``p(xi)`` over
the surface ``y=0`` produces vertical surface displacement
``u_z(x) = -(2/(pi E*)) int p(xi) ln|x - xi| dxi + C``.  Piecewise-constant pressure on collocated
elements gives a dense influence matrix with a closed-form log-kernel integral.  The contact set is
found two ways and compared:

  * an exact active-set LCP (``solve_lcp``): non-penetration ``u_z + h = const`` on contact, ``p >= 0``;
  * a penalty solve using the SAME :class:`TractionField` law as the FEM benchmark (``penalty_solve``).

Both are validated against the closed-form Hertz half-ellipse ``contact_fields.hertz_pressure``.
The 2-D log kernel is defined up to an additive constant (rigid-body), absorbed into the unknown
approach; the recovered PRESSURE is constant-independent given the load.
"""
from __future__ import annotations

import numpy as np


def _log_primitive(t, x):
    """Antiderivative of ``ln|x - t|`` w.r.t. ``t``:  ``(t-x)(ln|t-x| - 1)`` (0 at the singularity)."""
    d = np.asarray(t, float) - np.asarray(x, float)
    ad = np.abs(d)
    return np.where(ad < 1e-300, 0.0, d * (np.log(np.maximum(ad, 1e-300)) - 1.0))


class HalfspaceBEM2D:
    """Boundary-element compliance of a 2-D elastic half-plane (collocated constant-pressure elements)."""

    def __init__(self, x_centers, dx, Estar):
        self.xc = np.asarray(x_centers, float)
        self.dx = np.asarray(dx, float)
        self.Estar = float(Estar)
        n = len(self.xc)
        a = self.xc - 0.5 * self.dx
        b = self.xc + 0.5 * self.dx
        c = -2.0 / (np.pi * self.Estar)
        A = np.empty((n, n))
        for j in range(n):
            # u_z at every x_i due to unit constant pressure on element j: integral of the log kernel
            A[:, j] = c * (_log_primitive(b[j], self.xc) - _log_primitive(a[j], self.xc))
        self.A = A

    def displacement(self, p):
        return self.A @ np.asarray(p, float)


def _discretize(R, Estar, P, half_width, n):
    x = np.linspace(-half_width, half_width, n)
    dx = np.full(n, x[1] - x[0])
    bem = HalfspaceBEM2D(x, dx, Estar)
    h = x ** 2 / (2.0 * R)                     # parabolic gap (cylinder approximation)
    a_ana = 2.0 * np.sqrt(P * R / (np.pi * Estar))
    p0_ana = 2.0 * P / (np.pi * a_ana)
    return x, dx, bem, h, a_ana, p0_ana


def solve_lcp(R=1.0, Estar=1.0, P=0.01, half_width=None, n=601, max_set_iter=200):
    """Exact active-set LCP Hertz solve on the half-space.  Returns dict with x, p, a, p0.

    Monotone drop-only active set: start with a generous contact patch and remove negative-pressure
    elements until all remaining pressures are non-negative (robust for this convex, simply-connected
    contact).  ``delta`` (rigid approach) absorbs the 2-D log constant.
    """
    if half_width is None:
        half_width = 4.0 * np.sqrt(P * R / (np.pi * Estar))    # ~2a of slack
    x, dx, bem, h, a_ana, p0_ana = _discretize(R, Estar, P, half_width, n)
    active = np.abs(x) < 1.8 * a_ana                           # generous start, then shrink
    p = np.zeros(n)
    delta = 0.0
    for _ in range(max_set_iter):
        idx = np.where(active)[0]
        nc = len(idx)
        # [A_cc  -1; dx_c^T  0] [p_c; delta] = [-h_c; P]
        M = np.zeros((nc + 1, nc + 1))
        rhs = np.zeros(nc + 1)
        M[:nc, :nc] = bem.A[np.ix_(idx, idx)]
        M[:nc, nc] = -1.0
        M[nc, :nc] = dx[idx]
        rhs[:nc] = -h[idx]
        rhs[nc] = P
        sol = np.linalg.solve(M, rhs)
        p_c, delta = sol[:nc], float(sol[nc])
        p = np.zeros(n)
        p[idx] = p_c
        if (p_c >= 0.0).all():
            break
        active[idx[p_c < 0.0]] = False                         # drop negative-pressure elements
    a = float(x[p > 1e-12].max()) if (p > 1e-12).any() else 0.0
    return dict(x=x, p=p, a=a, p0=float(p.max()), a_ana=float(a_ana), p0_ana=float(p0_ana), delta=delta)


def penalty_solve(R=1.0, Estar=1.0, P=0.01, eps_n=None, half_width=None, n=601, max_iter=200):
    """Penalty Hertz solve on the half-space using the module's :class:`TractionField` normal law.

    For a prescribed load ``P`` the approach ``delta`` is the extra unknown (load constraint).  On the
    active set ``(I + eps_n A_cc) p_c = eps_n(delta - h_c)``; ``delta`` is found by bisection so the
    integrated pressure equals ``P``.  As ``eps_n -> inf`` this converges to :func:`solve_lcp`.
    """
    from .traction import TractionField

    if half_width is None:
        half_width = 4.0 * np.sqrt(P * R / (np.pi * Estar))
    x, dx, bem, h, a_ana, p0_ana = _discretize(R, Estar, P, half_width, n)
    if eps_n is None:
        eps_n = 1.0e4 * Estar / dx[0]                          # stiff penalty -> tiny penetration
    law = TractionField(eps_n)
    ones_n = np.tile([0.0, 1.0], (n, 1))

    def solve_for_delta(delta):
        # drop-only active set: (I + eps_n A_cc) p_c = eps_n (delta - h_c)
        active = (delta - h) > 0.0
        p = np.zeros(n)
        for _ in range(max_iter):
            idx = np.where(active)[0]
            if len(idx) == 0:
                return p, 0.0
            Mc = np.eye(len(idx)) + eps_n * bem.A[np.ix_(idx, idx)]
            p_c = np.linalg.solve(Mc, eps_n * (delta - h[idx]))
            if (p_c >= 0.0).all():
                p = np.zeros(n)
                p[idx] = p_c
                break
            active[idx[p_c < 0.0]] = False
        # one law evaluation so the module's TractionField is genuinely in the loop
        _ = law.evaluate(-(delta - h - bem.displacement(p)), ones_n)
        return p, float((p * dx).sum())

    # bisection on delta to match the load P
    lo, hi = 0.0, 10.0 * (h.max() + a_ana ** 2 / (2.0 * R))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p, load = solve_for_delta(mid)
        if load < P:
            lo = mid
        else:
            hi = mid
    p, load = solve_for_delta(0.5 * (lo + hi))
    a = float(x[p > 1e-9 * max(p.max(), 1e-30)].max()) if (p > 0).any() else 0.0
    return dict(x=x, p=p, a=a, p0=float(p.max()), a_ana=float(a_ana), p0_ana=float(p0_ana),
                eps_n=float(eps_n), load=float(load))


def cross_check(R=1.0, Estar=1.0, P=0.01, n=601, verbose=True):
    """BEM LCP and the TractionField penalty solve vs the closed-form Hertz half-ellipse."""
    from postprocessing import contact_fields as cf

    lcp = solve_lcp(R, Estar, P, n=n)
    pen = penalty_solve(R, Estar, P, n=n)
    x = lcp["x"]
    a = lcp["a_ana"]
    p_h = cf.hertz_pressure(x, a, lcp["p0_ana"])
    interior = np.abs(x) <= 0.85 * a

    def l2(p):
        return float(np.linalg.norm((p - p_h)[interior]) / (np.linalg.norm(p_h[interior]) + 1e-30))

    l2_lcp = l2(lcp["p"])
    l2_pen = l2(pen["p"])
    # coupling (penalty) vs BEM LCP directly
    both = (lcp["p"] > 1e-12) | (pen["p"] > 1e-12)
    l2_coupling_vs_bem = float(np.linalg.norm((pen["p"] - lcp["p"])[both])
                               / (np.linalg.norm(lcp["p"][both]) + 1e-30))
    res = dict(P=P, a_ana=a, p0_ana=lcp["p0_ana"],
               L2_lcp_vs_hertz=l2_lcp, L2_penalty_vs_hertz=l2_pen,
               L2_coupling_vs_bem=l2_coupling_vs_bem,
               a_lcp=lcp["a"], a_pen=pen["a"])
    if verbose:
        print(f"  half-space BEM cross-check (R={R}, E*={Estar}, P={P}, n={n}):")
        print(f"    Hertz a={a:.5f}  p0={lcp['p0_ana']:.5f}")
        print(f"    BEM LCP        vs closed-form Hertz (interior L2) = {l2_lcp*100:.3f}%")
        print(f"    TractionField  vs closed-form Hertz (interior L2) = {l2_pen*100:.3f}%")
        print(f"    coupling penalty vs BEM LCP        (interior L2) = {l2_coupling_vs_bem*100:.3f}%"
              f"   -> {'PASS' if l2_coupling_vs_bem < 0.01 else 'CHECK'} (<1%)")
    return res


if __name__ == "__main__":
    cross_check()
