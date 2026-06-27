"""Surface-to-surface measure couplings.

A *coupling* replaces the per-node closest-point projection with a mass-balanced correspondence
between two surfaces, carried by their arclength measures.  This is the optimal-transport view of
contact: the marginal (mass) constraint forbids the many-slaves-to-one-master collapse and makes
the resulting gap a continuous, patch-test-consistent field.

:class:`MonotoneCoupling1D` is the exact closed-form 1-D optimal transport (monotone rearrangement)
between two height-chart profiles ``z = h(x)``.  The transport map ``T : x_s -> x_m`` matches equal
cumulative-arclength quantiles, ``F_m(T(x_s)) = F_s(x_s)`` so ``T = F_m^{-1} o F_s``.  As an entropic
Sinkhorn coupling's regularization ``eps -> 0`` it converges to this map; the monotone map is the
bias-free verification anchor (see :class:`Sinkhorn2DCoupling`, added in M4).

All couplings expose the same contract::

    x_m, X_m, mass = coupling.map(xi)

where ``xi`` are slave parameters (the quadrature x-coordinates), ``x_m`` the corresponding master
parameter, ``X_m (n,2)`` the master surface point, and ``mass (n,)`` the coupled mass in ``[0,1]``
(0 where an unbalanced/partial coupling transports nothing, so the active set emerges as
``supp(mass)`` rather than being prescribed).
"""
from __future__ import annotations

import numpy as np


def _arclength_cdf(x: np.ndarray, hp: np.ndarray) -> np.ndarray:
    """Cumulative arclength measure ``F(x) = int sqrt(1+h'^2) dx'`` (trapezoid, strictly increasing).

    ``sqrt(1+h'^2) >= 1 > 0`` guarantees strict monotonicity, so ``F^{-1}`` is well defined and the
    monotone map is stable (no division by a vanishing density).
    """
    x = np.asarray(x, float)
    ds = np.sqrt(1.0 + np.asarray(hp, float) ** 2)
    incr = 0.5 * (ds[1:] + ds[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(incr)])


class MonotoneCoupling1D:
    """Exact 1-D optimal-transport coupling of two height profiles by monotone rearrangement.

    Parameters
    ----------
    slave, master : dict
        Baked height profiles ``{"x": (n,), "h": (n,), "hp": (n,)}`` (numpy, ``x`` strictly
        ascending), e.g. from ``rock_joint_shear.bake_height`` or
        ``profile_chart_2d.height_and_grad``.
    unbalanced : bool
        If True (default) and ``contact_band`` is given, slave points whose vertical separation
        from their mapped master point exceeds ``contact_band`` receive ``mass=0`` (the active set
        then emerges as the support of transported mass).  If ``contact_band`` is None the coupling
        is balanced (``mass==1`` everywhere) and the active set is decided downstream by the
        penalty/AL clamp (``g_N < 0``).
    contact_band : float, optional
        Vertical band (height units) for the unbalanced pre-screen.
    """

    def __init__(self, slave: dict, master: dict, unbalanced: bool = True,
                 contact_band: float | None = None):
        self.slave = {k: np.asarray(v, float) for k, v in slave.items()}
        self.master = {k: np.asarray(v, float) for k, v in master.items()}
        self.unbalanced = bool(unbalanced)
        self.contact_band = contact_band
        Fs = _arclength_cdf(self.slave["x"], self.slave["hp"])
        Fm = _arclength_cdf(self.master["x"], self.master["hp"])
        if Fs[-1] <= 0 or Fm[-1] <= 0:
            raise ValueError("degenerate surface (zero arclength)")
        self._Fs1 = Fs / Fs[-1]                 # normalized cumulative measures in [0,1]
        self._Fm1 = Fm / Fm[-1]

    def map(self, xi: np.ndarray):
        """Evaluate the correspondence at slave parameters ``xi`` (the quadrature x-coords).

        Returns ``(x_m (n,), X_m (n,2), mass (n,))``.
        """
        xi = np.asarray(xi, float)
        q = np.interp(xi, self.slave["x"], self._Fs1)            # slave quantile  F_s(xi)
        x_m = np.interp(q, self._Fm1, self.master["x"])          # T = F_m^{-1}(q)
        h_m = np.interp(x_m, self.master["x"], self.master["h"])
        Xm = np.column_stack([x_m, h_m])
        mass = np.ones_like(xi)
        if self.unbalanced and self.contact_band is not None:
            h_s = np.interp(xi, self.slave["x"], self.slave["h"])
            mass = np.where(np.abs(h_s - h_m) <= float(self.contact_band), 1.0, 0.0)
        return x_m, Xm, mass


def _logsumexp(M, axis):
    m = np.max(M, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class SinkhornCoupling1D:
    """Entropic (Sinkhorn) optimal-transport coupling of two 1-D height profiles.

    The regularized analogue of :class:`MonotoneCoupling1D`: it solves
    ``min_pi <C, pi> + eps KL(pi | a (x) b)`` for the squared-distance cost
    ``C_ij = 0.5 |X_s(x_i) - X_m(x_j)|^2`` on the arclength measures ``a, b``, in the log domain
    (numerically stable for small ``eps``).  The barycentric map ``T(x) = E_pi[X_m | x]`` is smooth,
    and as ``eps -> 0`` it converges to the exact monotone map — the consistency that
    :func:`measure_coupling_compare` checks.  This coupling generalizes to genuine 2-D surfaces
    (3-D contact); the 1-D case is the bias-free-anchored verification vehicle.

    Parameters
    ----------
    slave, master : dict ``{"x","h","hp"}`` baked profiles.
    eps : entropic regularization (in cost units; smaller = sharper = closer to monotone).
    n_iter : max Sinkhorn iterations.
    tol : convergence tolerance on the potentials.
    """

    def __init__(self, slave, master, eps: float = 1e-3, n_iter: int = 4000, tol: float = 1e-9):
        self.slave = {k: np.asarray(v, float) for k, v in slave.items()}
        self.master = {k: np.asarray(v, float) for k, v in master.items()}
        self.eps = float(eps)
        self.Xs = np.column_stack([self.slave["x"], self.slave["h"]])
        self.Xm = np.column_stack([self.master["x"], self.master["h"]])
        a = np.sqrt(1.0 + self.slave["hp"] ** 2)
        b = np.sqrt(1.0 + self.master["hp"] ** 2)
        self.loga = np.log(a / a.sum())
        self.logb = np.log(b / b.sum())
        C = 0.5 * ((self.Xs[:, None, :] - self.Xm[None, :, :]) ** 2).sum(-1)   # (ns, nm)
        f = np.zeros(len(self.Xs))
        g = np.zeros(len(self.Xm))
        for it in range(n_iter):
            f_new = -self.eps * _logsumexp((g[None, :] - C) / self.eps + self.logb[None, :], axis=1)
            g = -self.eps * _logsumexp((f_new[:, None] - C) / self.eps + self.loga[:, None], axis=0)
            if np.max(np.abs(f_new - f)) < tol:
                f = f_new
                break
            f = f_new
        self.f, self.g, self.n_iter_used = f, g, it + 1

    def map(self, xi):
        xi = np.asarray(xi, float)
        h_s = np.interp(xi, self.slave["x"], self.slave["h"])
        Xq = np.column_stack([xi, h_s])
        C = 0.5 * ((Xq[:, None, :] - self.Xm[None, :, :]) ** 2).sum(-1)        # (n, nm)
        logw = (self.g[None, :] - C) / self.eps + self.logb[None, :]
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        wsum = w.sum(axis=1, keepdims=True)
        Xm_bary = (w[:, :, None] * self.Xm[None, :, :]).sum(axis=1) / wsum
        mass = np.ones_like(xi)
        return Xm_bary[:, 0], Xm_bary, mass


def measure_coupling_compare(slave, master, xi, eps_list=(0.1, 0.03, 0.01, 0.003, 0.001)):
    """Compare the Sinkhorn map to the exact monotone map at several ``eps`` (consistency table).

    Returns a list of ``(eps, n_iter, relerr)`` where ``relerr`` is the relative L2 difference of
    the barycentric master x-coordinate from the monotone reference.  As ``eps -> 0`` it -> 0.
    """
    mono = MonotoneCoupling1D(slave, master)
    xm_ref, _, _ = mono.map(xi)
    rows = []
    for eps in eps_list:
        sk = SinkhornCoupling1D(slave, master, eps=eps)
        xm, _, _ = sk.map(xi)
        relerr = float(np.linalg.norm(xm - xm_ref) / (np.linalg.norm(xm_ref - xm_ref.mean()) + 1e-30))
        rows.append((float(eps), sk.n_iter_used, relerr))
    return rows


if __name__ == "__main__":
    # self-test 1: identical surfaces -> identity map (T(x)=x), zero gap proxy.
    x = np.linspace(0.0, 10.0, 401)
    h = 0.3 * np.sin(0.7 * x)
    hp = 0.3 * 0.7 * np.cos(0.7 * x)
    prof = dict(x=x, h=h, hp=hp)
    c = MonotoneCoupling1D(prof, prof)
    xq = np.linspace(0.5, 9.5, 23)
    x_m, Xm, mass = c.map(xq)
    assert np.allclose(x_m, xq, atol=1e-6), "identity coupling must map x->x"
    assert np.allclose(Xm[:, 1], np.interp(xq, x, h), atol=1e-6)

    # self-test 2: mass conservation — the map pushes the slave measure onto the master measure.
    # equal-total-arclength surfaces: cumulative quantiles must align endpoints.
    assert abs(c._Fs1[0]) < 1e-15 and abs(c._Fs1[-1] - 1.0) < 1e-15
    print("  MonotoneCoupling1D self-test OK (identity map + normalized CDFs)")
