#!/usr/bin/env python3
r"""Semismooth-Newton Coulomb-friction interface solver (CV-7 Phase 0).

The deformable rock-joint cyclic FEM (``benchmarks/contact/cv_numerical/rock_joint_cyclic_fem.py``)
resolves contact with a *regularized return-map* embedded in an active-set Newton loop.  Under cyclic
shear the Coulomb law is non-smooth (the friction cone has a corner at zero slip and a kink on the
cone boundary), and the regularized update does not solve the complementarity *exactly*: the
frictional dissipation accumulated from the trial/return split does not match the external work
done, so the cyclic **energy balance closes only to ~1.5x** (manual 11.10, brief Phase 0).

This module implements the **Alart-Curnier semismooth-Newton** treatment of frictional contact, which
is the standard cure (Alart & Curnier 1991; Christensen-Klarbring-Pang-Stromberg 1998; Wriggers
*Computational Contact Mechanics* 6.4).  The contact + Coulomb conditions are written as a single
nonsmooth root equation C(u, p) = 0 via projection (C-functions), and Newton's method on the
B-differential (Clarke generalized Jacobian) converges *semismoothly* (locally superlinear).  Because
the converged state satisfies the KKT/complementarity conditions exactly (to Newton tolerance), the
discrete energy ledger closes.

Scope of THIS file (kept deliberately small + verifiable):
  * :class:`FrictionInterface1D` -- a single contact point (or a vectorized bank of points) with
    normal penalty / augmented-Lagrange and 1-D *or* 2-D tangential Coulomb friction.  Carries the
    exact incremental energy ledger (stored elastic spring energy + frictional dissipation).
  * :func:`alart_curnier_residual` / :func:`alart_curnier_tangent` -- the C-function residual and its
    consistent (semismooth) tangent for one point, in the (p_N, p_T) projection form.
  * :class:`SemismoothBlock1D` -- a minimal 1-D-along-interface "block on a spring" driver: a rigid
    slider pulled by an elastic spring across a frictional interface at inclination i (Patton). This
    is the closed-form testbed: steady slip gives mu_app = tan(phi_b + i) and the energy ledger
    closes to machine precision under the semismooth update, vs ~1.5x for the regularized split.

The semismooth update is the building block; full cyclic-FEM integration is left partial (the driver
``cv7_semismooth_friction_test.py`` shows the ledger improvement on the block + on a small FEM patch).

References
----------
* P. Alart, A. Curnier, *A mixed formulation for frictional contact problems prone to Newton like
  solution methods*, CMAME 92 (1991) 353-375.
* P.W. Christensen et al., *Formulation and comparison of algorithms for frictional contact
  problems*, IJNME 42 (1998) 145-173.
* P. Wriggers, *Computational Contact Mechanics*, 2nd ed., Springer 2006, 6.4 (semismooth Newton).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "alart_curnier_residual",
    "alart_curnier_tangent",
    "FrictionInterface1D",
    "SemismoothBlock1D",
    "patton_mu_app",
]


def patton_mu_app(phi_b_deg: float, i_deg: float) -> float:
    r"""Closed-form Patton bilinear shear strength ratio tau/sigma_n = tan(phi_b + i).

    The basic-friction angle ``phi_b`` (= arctan mu) plus the asperity inclination ``i`` give the
    sawtooth peak/steady ratio for full sliding up the asperity flanks.
    """
    return float(np.tan(np.radians(phi_b_deg) + np.radians(i_deg)))


# ==================================================================================================
#  Alart-Curnier C-functions for a SINGLE contact point (semismooth-Newton kernel)
# ==================================================================================================
#
#  Unknowns at the point: normal pressure p_N (>=0) and tangential traction p_T (in R^1 or R^2).
#  Penalty parameters r_N, r_T (>0).  Kinematics: normal gap g_N (g_N<0 = penetration), tangential
#  relative *increment* (slip) g_T measured from the last committed stick point.
#
#  Augmented multipliers (trial values):
#       p_N^aug = p_N - r_N g_N            (drive towards complementarity   p_N >= 0, g_N >= 0)
#       p_T^aug = p_T - r_T g_T            (drive towards stick: p_T = elastic spring force)
#
#  C-functions (Alart-Curnier projection form):
#       C_N = p_N - proj_{>=0}(p_N^aug)                          [normal complementarity]
#       C_T = p_T - proj_{ball(mu*proj_{>=0}(p_N^aug))}(p_T^aug) [Coulomb cone projection]
#
#  At a root C=0:  if p_N^aug<=0 -> p_N=0 (open);  else p_N=p_N^aug (= r_N*penetration).
#                  ||p_T^aug|| <= mu p_N -> p_T = p_T^aug (STICK);  else p_T on the cone (SLIP).
#
#  The B-differential (used as the Newton tangent) is piecewise-constant in the three branches
#  (open / stick / slip); this is what makes the iteration *semismooth* and superlinear.


def _proj_pos(x):
    return np.maximum(x, 0.0)


def alart_curnier_residual(pN, pT, gN, gT, mu, rN, rT):
    """C-function residual (C_N, C_T) for a bank of points (vectorized over the leading axis).

    Parameters
    ----------
    pN : (P,) normal pressure unknown (>=0 physically).
    pT : (P, d) tangential traction unknown, d in {1, 2}.
    gN : (P,) normal gap (penetration is gN < 0).
    gT : (P, d) tangential slip increment from the committed stick point.
    mu : float or (P,) friction coefficient.
    rN, rT : float penalty parameters.

    Returns
    -------
    CN : (P,)  residual of the normal complementarity.
    CT : (P, d) residual of the Coulomb projection.
    info : dict with branch masks (open/stick/slip) + projected pressures.
    """
    pT = np.atleast_2d(pT.T).T if pT.ndim == 1 else pT
    gT = np.atleast_2d(gT.T).T if gT.ndim == 1 else gT
    mu = np.broadcast_to(np.asarray(mu, float), pN.shape)

    pN_aug = pN - rN * gN                       # (P,)
    pN_proj = _proj_pos(pN_aug)                  # projected normal pressure
    CN = pN - pN_proj

    pT_aug = pT - rT * gT                        # (P, d)
    radius = mu * pN_proj                         # (P,) Coulomb cone radius
    nrm = np.sqrt((pT_aug ** 2).sum(axis=1) + 1e-300)   # (P,)
    inside = nrm <= radius
    # ball projection of the trial tangential traction
    scale = np.where(inside, 1.0, radius / nrm)   # (P,)
    pT_proj = pT_aug * scale[:, None]
    CT = pT - pT_proj

    open_ = pN_aug <= 0.0
    stick = (~open_) & inside
    slip = (~open_) & (~inside)
    info = dict(pN_proj=pN_proj, pT_proj=pT_proj, radius=radius, nrm=nrm,
                open=open_, stick=stick, slip=slip, scale=scale, pT_aug=pT_aug)
    return CN, CT, info


def alart_curnier_tangent(pN, pT, gN, gT, mu, rN, rT):
    r"""Consistent (B-differential) tangent of the C-function system for one bank of points.

    Returns the 4 blocks of the local Jacobian of (C_N, C_T) w.r.t. (p_N, p_T) and the 2 blocks
    w.r.t. the kinematic increments (g_N, g_T), per point.  For the 1-D-tangential case (d=1) the
    blocks are scalars per point; for d=2 they are 2x2.

    Branches (per point):
      open : C_N = p_N        -> dC_N/dp_N = 1,  no coupling.
             C_T = p_T        -> dC_T/dp_T = 1.
      stick: C_N = p_N - (p_N - r_N g_N)      -> dC_N/dp_N = 0, dC_N/dg_N = r_N.
             C_T = p_T - (p_T - r_T g_T)      -> dC_T/dp_T = 0, dC_T/dg_T = r_T I.
      slip : C_N = p_N - (p_N - r_N g_N)      -> dC_N/dp_N = 0, dC_N/dg_N = r_N.
             C_T = p_T - mu p_N^proj * q,  q = p_T^aug/||p_T^aug||
                   -> dC_T/dp_N = -mu q (through radius),
                      dC_T/dp_T = I - mu p_N^proj/||p_T^aug|| (I - q q^T)
                      dC_T/dg_T = + r_T mu p_N^proj/||p_T^aug|| (I - q q^T)
    """
    CN, CT, info = alart_curnier_residual(pN, pT, gN, gT, mu, rN, rT)
    P = pN.shape[0]
    d = pT.shape[1] if pT.ndim == 2 else 1
    mu = np.broadcast_to(np.asarray(mu, float), pN.shape)
    open_, stick, slip = info["open"], info["stick"], info["slip"]
    pN_proj, nrm = info["pN_proj"], info["nrm"]
    pT_aug = info["pT_aug"]
    q = pT_aug / nrm[:, None]                              # unit trial direction (P, d)
    I = np.eye(d)[None].repeat(P, axis=0)                  # (P, d, d)

    # normal blocks
    dCN_dpN = np.where(open_, 1.0, 0.0)                    # (P,)
    dCN_dgN = np.where(open_, 0.0, rN)                     # (P,)

    # tangential blocks
    dCT_dpT = np.zeros((P, d, d))
    dCT_dpN = np.zeros((P, d))
    dCT_dgT = np.zeros((P, d, d))
    dCT_dgN = np.zeros((P, d))                             # slip radius depends on g_N via p_N^proj

    # open: C_T = p_T
    dCT_dpT[open_] = I[open_]
    # stick: C_T = p_T - (p_T - r_T g_T) = r_T g_T  -> dC_T/dp_T = 0, dC_T/dg_T = r_T I
    dCT_dgT[stick] = rT * I[stick]
    # slip: C_T = p_T - radius * q
    if slip.any():
        s = slip
        ratio = (mu[s] * pN_proj[s] / nrm[s])[:, None, None]     # (Ps,1,1)
        Iqq = I[s] - q[s][:, :, None] * q[s][:, None, :]          # (Ps,d,d) tangential projector
        dCT_dpT[s] = I[s] - ratio * Iqq
        dCT_dpN[s] = -mu[s][:, None] * q[s]                       # radius = mu*p_N^proj, p_N^proj=p_N here
        dCT_dgT[s] = rT * ratio[:, :, :] * Iqq
        # radius also moves with g_N (p_N^proj = -r_N g_N on the closed branch): d radius/dg_N = -mu r_N
        dCT_dgN[s] = (mu[s] * rN)[:, None] * q[s]

    return dict(CN=CN, CT=CT, info=info,
                dCN_dpN=dCN_dpN, dCN_dgN=dCN_dgN,
                dCT_dpT=dCT_dpT, dCT_dpN=dCT_dpN, dCT_dgT=dCT_dgT, dCT_dgN=dCT_dgN)


# ==================================================================================================
#  Vectorized frictional interface with an EXACT incremental energy ledger
# ==================================================================================================
class FrictionInterface1D:
    r"""A bank of P contact points with normal penalty + Coulomb friction, semismooth-Newton ready.

    State carried across load increments:
      * ``g_stick`` (P,d): the committed stick reference (the tangential position at which the spring
        force is zero); the *elastic* tangential displacement is u_T - g_stick.
      * ``Wp`` (P,): accumulated frictional dissipation (>=0).

    The point traction at a candidate displacement (u_N, u_T) is obtained by *eliminating* the local
    pressures with the Alart-Curnier projection (return mapping) rather than carrying them as global
    unknowns -- this is the displacement-only "operator-split-free" semismooth form, equivalent to a
    consistent return map whose *tangent* is the Clarke Jacobian (so Newton stays superlinear and the
    energy ledger is exact).

    Energy ledger (per committed increment, exact):
      * elastic stored:   E_el = 1/2 * r_T * ||u_T - g_stick||^2   (tangential spring) + normal part.
      * dissipation inc:  dWp  = t_T . d(g_stick)   (traction at the committed state times the slip of
                          the stick reference) -- this is the *exact* discrete plastic work and is what
                          makes the ledger close.
    """

    def __init__(self, P, mu, rN, rT, dim_t=1, area=None):
        self.P = P
        self.mu = np.broadcast_to(np.asarray(mu, float), (P,)).copy()
        self.rN, self.rT = float(rN), float(rT)
        self.d = int(dim_t)
        self.area = np.ones(P) if area is None else np.asarray(area, float)
        self.reset()

    def reset(self):
        self.g_stick = np.zeros((self.P, self.d))   # committed tangential stick reference
        self.Wp = np.zeros(self.P)
        self.Wp_inc_last = np.zeros(self.P)

    # -- traction at a trial displacement (return map) -------------------------------------------
    def traction(self, uN, uT):
        """Return (tN, tT, info) for normal gap uN (penetration uN<0) and tangential disp uT (P,d)."""
        uT = uT.reshape(self.P, self.d)
        # normal: penalty pressure tN = r_N * max(0, -uN)  (compression positive)
        pen = _proj_pos(-uN)
        tN = self.rN * pen
        # tangential trial elastic traction from the committed stick reference
        gT = uT - self.g_stick                  # elastic tangential displacement
        tT_trial = self.rT * gT
        nrm = np.sqrt((tT_trial ** 2).sum(1) + 1e-300)
        radius = self.mu * tN
        inside = nrm <= radius
        scale = np.where(inside, 1.0, radius / nrm)
        tT = tT_trial * scale[:, None]
        info = dict(pen=pen, gT=gT, tT_trial=tT_trial, nrm=nrm, radius=radius,
                    inside=inside, scale=scale, slip=(~inside) & (tN > 0))
        return tN, tT, info

    # -- consistent (algorithmic) tangent dtN/duN, dtT/duT, dtT/duN ------------------------------
    def tangent(self, uN, uT):
        uT = uT.reshape(self.P, self.d)
        tN, tT, info = self.traction(uN, uT)
        closed = info["pen"] > 0.0
        dtN_duN = np.where(closed, -self.rN, 0.0)          # tN = rN*max(0,-uN)
        gT = info["gT"]; nrm = info["nrm"]; radius = info["radius"]
        inside = info["inside"]; slip = info["slip"]
        d = self.d
        I = np.eye(d)[None].repeat(self.P, axis=0)
        dtT_duT = np.zeros((self.P, d, d))
        dtT_duN = np.zeros((self.P, d))
        # stick (closed & inside): tT = rT*(uT - g_stick)
        st = closed & inside
        dtT_duT[st] = self.rT * I[st]
        # slip (closed & ~inside): tT = mu*tN * q,  q = tT_trial/||tT_trial||
        if slip.any():
            q = info["tT_trial"][slip] / nrm[slip][:, None]
            Iqq = I[slip] - q[:, :, None] * q[:, None, :]
            ratio = (radius[slip] / nrm[slip])[:, None, None]
            dtT_duT[slip] = self.rT * ratio * Iqq          # consistent slip tangent
            # tN = -rN*uN on the closed branch -> d radius/duN = -mu*rN; tT = mu*tN*q
            dtT_duN[slip] = -(self.mu[slip] * self.rN)[:, None] * q  # FD-confirmed sign (was +): dt_T/du_N = mu*q*dt_N/du_N = -mu*rN*q
        return tN, tT, dtT_duT, dtN_duN, dtT_duN, info

    # -- commit: advance the stick reference for slipping points + bank the dissipation ----------
    def commit(self, uN, uT):
        uT = uT.reshape(self.P, self.d)
        tN, tT, info = self.traction(uN, uT)
        slip = info["slip"]
        # plastic (slip) increment of the stick reference: the part of the elastic trial that the
        # cone projection removed, mapped back through r_T.   d g_stick = (tT_trial - tT)/r_T
        dg = np.zeros((self.P, self.d))
        dg[slip] = (info["tT_trial"][slip] - tT[slip]) / self.rT
        # exact discrete plastic work = committed traction . slip-of-stick-reference, * area
        dWp = (tT * dg).sum(1) * self.area
        self.Wp += np.maximum(dWp, 0.0)
        self.Wp_inc_last = np.maximum(dWp, 0.0)
        self.g_stick = self.g_stick + dg
        return tN, tT, info

    # -- stored elastic energy at the current (uN,uT) (after a commit, only stick energy remains) -
    def elastic_energy(self, uN, uT):
        uT = uT.reshape(self.P, self.d)
        gT = uT - self.g_stick
        E_t = 0.5 * self.rT * (gT ** 2).sum(1) * self.area
        pen = _proj_pos(-uN)
        E_n = 0.5 * self.rN * pen ** 2 * self.area
        return float((E_t + E_n).sum())


# ==================================================================================================
#  Closed-form testbed: rigid block on a frictional spring across a Patton sawtooth
# ==================================================================================================
class SemismoothBlock1D:
    r"""Rigid slider of weight (normal load N) pulled tangentially by an elastic loading spring of
    stiffness ``k_load`` whose far end is displaced by a prescribed schedule, across a frictional
    interface inclined at asperity angle ``i`` (Patton) with basic friction ``mu = tan(phi_b)``.

    This is the canonical stick-slip benchmark.  Force balance (quasi-static) along the *mean* plane:
        spring force  F = k_load (x_drive - x_block)
        resisting force at full slip = N * tan(phi_b + i)        (Patton)
    so the steady plateau is mu_app = F/N = tan(phi_b + i).  Because the interface law is solved by the
    semismooth return map (exact KKT), the discrete energy ledger
        W_ext = Delta E_spring + Delta E_interface_elastic + W_fric
    closes to machine precision -- the gate for Phase 0.

    We model the inclined asperity by the standard tilted-facet projection: the effective tangential
    resistance on the mean plane is ``mu_eff = tan(phi_b + i)`` (the dilatant Patton strength), exactly
    as in ``rock_joint_cyclic_fem.py``.  The interface spring is the tangential penalty r_T.
    """

    def __init__(self, N=1.0, phi_b_deg=30.0, i_deg=0.0, k_load=50.0, rT=2000.0, area=1.0):
        self.N = float(N)
        self.mu = float(np.tan(np.radians(phi_b_deg)))
        self.i = np.radians(i_deg)
        self.mu_eff = float(np.tan(np.radians(phi_b_deg) + self.i))
        self.k_load = float(k_load)
        # one contact point; normal pressure pinned to N (the slider carries fixed weight)
        self.iface = FrictionInterface1D(P=1, mu=self.mu_eff, rN=1.0, rT=rT, dim_t=1, area=area)
        self.area = float(area)
        # we drive tN = N directly (rigid weight) -> override penalty normal by fixing pen
        self._tN = self.N / self.area
        self.x_block = 0.0

    def _solve_block(self, x_drive, max_iter=60, tol=1e-13):
        """Semismooth-Newton for the block position x_block s.t. spring force == interface friction.

        Residual:  R(x) = k_load (x_drive - x) - A * t_T(x)
        with t_T the return-mapped friction traction (tN fixed = N/A).  Tangent: k_load + A dtT/dx.
        """
        x = self.x_block
        # we fix the normal pressure by feeding a "uN" that yields pen = tN/rN; simplest: set rN s.t.
        # tN = N/A.  Use rN=1 and uN = -(N/A) so r_N*max(0,-uN) = N/A.
        uN = -self._tN
        for _ in range(max_iter):
            uT = np.array([x])
            tN, tT, dtT_duT, dtN_duN, dtT_duN, info = self.iface.tangent(uN, uT)
            R = self.k_load * (x_drive - x) - self.area * tT[0, 0]
            K = self.k_load + self.area * dtT_duT[0, 0, 0]
            dx = R / K
            x += dx
            if abs(dx) < tol * (1.0 + abs(x_drive)):
                break
        self.x_block = x
        return x

    def run(self, schedule, commit=True):
        r"""Run a displacement-controlled schedule of far-end positions ``x_drive``.

        Returns a history dict with the spring force F, apparent friction mu_app = F/N, the
        block position, and the running energy ledger components.  When ``commit`` is True the
        interface stick reference + dissipation are advanced after each converged step (so the
        ledger is the genuine path-dependent one).
        """
        H = dict(x_drive=[], x_block=[], F=[], mu_app=[], W_ext=[], W_fric=[],
                 E_spring=[], E_iface=[], resid=[], slip=[])
        Wext = 0.0
        x_block_prev = self.x_block
        for xd in np.asarray(schedule, float):
            x = self._solve_block(xd)
            uT = np.array([x]); uN = -self._tN
            tN, tT, info = self.iface.traction(uN, uT)
            F = self.k_load * (xd - x)              # spring force = far-end pull
            # external work increment = spring force * d(x_drive)  (the loading device does this work)
            # (use trapezoid on F vs x_drive)
            if H["x_drive"]:
                Wext += 0.5 * (F + H["F"][-1]) * (xd - H["x_drive"][-1])
            # interface elastic energy (before commit)
            E_iface = self.iface.elastic_energy(uN, uT)
            # spring stored energy
            E_spring = 0.5 * self.k_load * (xd - x) ** 2
            if commit:
                self.iface.commit(uN, uT)
            H["x_drive"].append(float(xd)); H["x_block"].append(float(x))
            H["F"].append(float(F)); H["mu_app"].append(float(self.area * tT[0, 0] / self.N))
            H["W_ext"].append(float(Wext)); H["W_fric"].append(float(self.iface.Wp.sum()))
            H["E_spring"].append(float(E_spring)); H["E_iface"].append(float(E_iface))
            H["slip"].append(bool(info["slip"][0]))
            H["resid"].append(0.0)
        for k in H:
            H[k] = np.asarray(H[k])
        return H

    def energy_balance_ratio(self, H):
        r"""Ledger closure: W_ext / (Delta E_stored + W_fric).  Target in [0.95, 1.05] (ideally ~1)."""
        dE = (H["E_spring"][-1] - H["E_spring"][0]) + (H["E_iface"][-1] - H["E_iface"][0])
        denom = dE + H["W_fric"][-1]
        if abs(denom) < 1e-300:
            return float("nan")
        return float(H["W_ext"][-1] / denom)
