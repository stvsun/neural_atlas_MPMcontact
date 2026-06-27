"""Continuous gap and slip FIELDS on a slave surface, sampled at quadrature points.

Given a baked slave height profile and a :class:`~solvers.contact.measure_coupling.coupling`
correspondence to the master, :class:`GapField` evaluates, at any slave parameters ``xi`` (the
quadrature x-coordinates):

    Xs   slave surface point (xi, h_s(xi))
    n_s  slave unit normal  (-h_s', 1)/sqrt(1+h_s'^2)     (profile_chart_2d.surface_normal)
    t_s  slave unit tangent ( 1, h_s')/sqrt(1+h_s'^2)
    J_s  arclength Jacobian sqrt(1+h_s'^2)
    g_N  signed normal gap (X_s - X_m).n_s   (g_N < 0 => penetration, repo convention)
    g_T  tangential slip-correspondence (X_s - X_m).t_s   (feeds friction)

As the entropic coupling ``eps -> 0`` the master point ``X_m = T(xi)`` is the closest-point
projection and ``g_N`` is the classical node-to-surface normal gap; equivalently ``g_N = d_n f``,
the normal derivative of the Kantorovich potential.  :meth:`eval_gap` adapts this to the
``(g_N, n)`` callback signature the assembler consumes (parameterizing the slave surface by the
x-coordinate of each quadrature point).
"""
from __future__ import annotations

import numpy as np


class GapField:
    """Gap/slip fields from a slave height profile + a measure coupling to the master."""

    def __init__(self, slave_baked: dict, coupling):
        self.slave = {k: np.asarray(v, float) for k, v in slave_baked.items()}
        self.coupling = coupling

    def _slave_at(self, x: np.ndarray):
        x = np.asarray(x, float)
        h = np.interp(x, self.slave["x"], self.slave["h"])
        hp = np.interp(x, self.slave["x"], self.slave["hp"])
        sec = np.sqrt(1.0 + hp ** 2)
        ns = np.column_stack([-hp / sec, 1.0 / sec])       # upward unit normal
        ts = np.column_stack([1.0 / sec, hp / sec])        # forward unit tangent
        Xs = np.column_stack([x, h])
        return Xs, ns, ts, sec

    def sample(self, xi: np.ndarray) -> dict:
        """Full gap/slip record at slave parameters ``xi`` (numpy arrays, all detached)."""
        Xs, ns, ts, Js = self._slave_at(xi)
        _, Xm, mass = self.coupling.map(np.asarray(xi, float))
        d = Xs - Xm
        gN = (d * ns).sum(axis=1)
        gT = (d * ts).sum(axis=1)
        return dict(x=np.asarray(xi, float), Xs=Xs, ns=ns, ts=ts, Js=Js,
                    Xm=Xm, gN=gN, gT=gT, mass=mass)

    def eval_gap(self, Xq: np.ndarray):
        """``(g_N, n_s)`` at quadrature points ``Xq (Q,2)`` — the assembler's gap callback.

        The slave surface is single-valued, so the x-coordinate of each quadrature point is its
        slave parameter.
        """
        s = self.sample(np.asarray(Xq, float)[:, 0])
        return s["gN"], s["ns"]

    def eval_slip(self, Xq: np.ndarray):
        """Tangential slip-correspondence vector ``g_T t_s`` at ``Xq`` (for friction)."""
        s = self.sample(np.asarray(Xq, float)[:, 0])
        return s["gT"][:, None] * s["ts"]


if __name__ == "__main__":
    from solvers.contact.measure_coupling.coupling import MonotoneCoupling1D
    # two flat surfaces offset by a known vertical gap -> constant gap field.
    x = np.linspace(0.0, 4.0, 201)
    slave = dict(x=x, h=np.full_like(x, 0.10), hp=np.zeros_like(x))    # upper at z=0.10
    master = dict(x=x, h=np.full_like(x, 0.00), hp=np.zeros_like(x))   # lower at z=0.00
    gf = GapField(slave, MonotoneCoupling1D(slave, master))
    s = gf.sample(np.linspace(0.5, 3.5, 17))
    assert np.allclose(s["ns"], np.array([0.0, 1.0]), atol=1e-12)
    assert np.allclose(s["gN"], 0.10, atol=1e-9), f"flat gap should be 0.10, got {s['gN'][:3]}"
    # push the slave below the master -> penetration (gN<0)
    slave2 = dict(x=x, h=np.full_like(x, -0.05), hp=np.zeros_like(x))
    gf2 = GapField(slave2, MonotoneCoupling1D(slave2, master))
    s2 = gf2.sample(np.linspace(0.5, 3.5, 17))
    assert np.allclose(s2["gN"], -0.05, atol=1e-9)
    print("  GapField self-test OK (flat separation +0.10, penetration -0.05)")
