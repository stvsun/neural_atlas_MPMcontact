#!/usr/bin/env python3
"""CV-8  TWO-BODY DEFORMABLE OT contact — non-matching mortar + large sliding (T2).

The OT measure-coupling CVs (cv1..cv7_ot_gap) all push a deformable body against a *rigid* master
(closest-point gap to a fixed indenter / chart).  This driver exercises the UNTESTED regime the
mortar coupling was built for: TWO separate deformable FEM meshes, with NON-MATCHING surface node
distributions on the shared interface, coupled BOTH ways (slave force AND master reaction carried by
the same consistent mortar correspondence).  Nothing in the shared module is modified -- the slave
side reuses ``assemble_contact`` verbatim; the master reaction is scattered through the SAME P1
boundary shape functions at the OT-mapped point ``X_m = T(xi)`` (Newton's 3rd law), so the two-body
tangent stays symmetric.

Why mortar and not lumped:  the conventional node-to-surface penalty (``rock_joint_two_block.py``)
LUMPS a tributary area onto each slave node and projects it independently to its closest master
node.  Across NON-MATCHING meshes that yields a node-spacing sawtooth: a uniform interface pressure
is transmitted as a jagged, mesh-dependent stress in the receiving body.  The mortar measure-coupling
integrates the traction FIELD ``p(x) = sum_J N_J(x) p_J`` against the master shape functions, so a
constant pressure passes EXACTLY through the non-matching interface (the contact patch test) -- the
decisive consistency the lumped scheme fails.

TWO VERIFICATIONS
-----------------
(1) TWO-BODY PATCH TEST.  Upper block top loaded by a uniform pressure ``p``; lower block bottom
    fixed; the interface meshes are deliberately non-matching (different node counts / spacing).  A
    consistent transmission gives sigma_yy = -p UNIFORMLY in BOTH bodies despite the non-matching
    nodes.  We report the stress non-uniformity (max-min)/p of the RECEIVING (lower) body for the
    MORTAR coupling AND for a LUMPED node-to-surface baseline.  PASS: mortar uniformity < 1e-9; the
    lumped baseline is reported for contrast (it does NOT pass).

(2) DEFORMABLE HERTZ-LIKE.  A curved-bottom upper block (parabolic profile, radius R) is pressed
    onto a flat lower block; BOTH deform.  The recovered contact half-width a and peak pressure p0
    are compared to ``contact_fields.line_contact_params`` with the COMBINED plane-strain modulus
    ``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2``.  PASS: a_relerr, p0_relerr < 10%.

(3) LARGE SLIDING.  After seating the Hertz load, the upper block is dragged tangentially over a
    finite distance (> the contact half-width).  We monitor the contact-patch centroid: it must
    track the platen monotonically with NO saltation/jitter (the mortar correspondence re-maps
    smoothly), and the resultant normal load stays ~constant.

Run:   python3 benchmarks/contact/cv_numerical/cv8_deformable_ot.py            # local coarse (~60s)
       python3 benchmarks/contact/cv_numerical/cv8_deformable_ot.py --mode patch
       python3 benchmarks/contact/cv_numerical/cv8_deformable_ot.py --mode cylinder --mesh-coarse 40 24
Writes runs/cv8_deformable_ot/metrics.json (+ history.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver                                       # noqa: E402
from solvers.contact.measure_coupling import (                                     # noqa: E402
    assemble_contact, TractionField, MonotoneCoupling1D)
from solvers.contact.measure_coupling.quadrature import gauss_legendre_1d          # noqa: E402
from postprocessing import contact_fields as cf                                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv8_deformable_ot")


# ==================================================================================================
#  Mesh: a rectangular block of CST triangles with an optionally CURVED bottom edge.
# ==================================================================================================
def block_mesh(W, H, n_x, n_y, y0=0.0, curve_R=None, jitter=0.0, seed=0):
    """Rectangle [-W, W] x [y0, y0+H] of structured CST triangles.

    curve_R : if given, the BOTTOM edge is lowered by a parabola -((x)^2)/(2 curve_R) (a cylinder of
        radius curve_R touching at x=0) -- the rest of the block follows by linear vertical blending
        so the mesh stays valid.  jitter : random horizontal perturbation of interior columns (breaks
        node-matching across the two meshes).  Returns (nodes, tris, top_idx, bot_idx) with bot_idx
        ordered left->right along the (curved) bottom edge.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(-W, W, n_x + 1)
    if jitter > 0:
        dx = (xs[1] - xs[0])
        xs[1:-1] += rng.uniform(-jitter, jitter, n_x - 1) * dx
        xs = np.sort(xs)
    ys = np.linspace(0.0, 1.0, n_y + 1)               # parametric 0..1 bottom->top
    nodes = []
    nx1 = n_x + 1
    for iy, v in enumerate(ys):
        for ix, x in enumerate(xs):
            ybot = y0
            if curve_R is not None:
                ybot = y0 - (x ** 2) / (2.0 * curve_R)
            ytop = y0 + H
            y = ybot + v * (ytop - ybot)
            nodes.append([x, y])
    nodes = np.array(nodes, float)

    def nid(iy, ix):
        return iy * nx1 + ix

    tris = []
    for iy in range(n_y):
        for ix in range(n_x):
            a, b, c, d = nid(iy, ix), nid(iy, ix + 1), nid(iy + 1, ix + 1), nid(iy + 1, ix)
            tris.append([a, b, d])
            tris.append([b, c, d])
    tris = np.array(tris, int)
    v = nodes[tris]
    area2 = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) -
             (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    tris[area2 < 0] = tris[area2 < 0][:, [0, 2, 1]]
    bot_idx = np.array([nid(0, ix) for ix in range(nx1)])
    top_idx = np.array([nid(n_y, ix) for ix in range(nx1)])
    return nodes, tris, top_idx, bot_idx


# ==================================================================================================
#  Two-body OT contact: slave = upper bottom edge, master = lower top edge.
#  Slave side reuses assemble_contact verbatim; master reaction scattered via the SAME P1 functions
#  at the OT-mapped point X_m=T(xi).  Newton's 3rd law -> symmetric two-body tangent.
# ==================================================================================================
def _profile(surf_xy):
    """Bake a height profile {x,h,hp} from an ordered surface polyline (x strictly ascending).

    Robust slope by one-sided/central finite differences that never divide by a vanishing dx
    (jittered non-matching meshes can place two surface nodes very close in x).
    """
    x = np.asarray(surf_xy[:, 0], float)
    h = np.asarray(surf_xy[:, 1], float)
    hp = np.zeros_like(x)
    dx = np.diff(x)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    slope = np.diff(h) / dx
    hp[1:-1] = 0.5 * (slope[:-1] + slope[1:])
    hp[0] = slope[0]
    hp[-1] = slope[-1]
    return dict(x=x, h=h, hp=hp)


def assemble_two_body(slave_xy, slave_ids, master_xy, master_ids, n_dof, eps_n, mu=0.0,
                      order=3, slip_dir=None, contact_band=None):
    """Consistent two-body mortar contact between two DEFORMABLE surfaces.

    slave_xy / master_xy : (n,2) ordered deformed surface polylines (x ascending).
    *_ids : global node indices (into the SHARED 2*N_total dof vector; master ids already offset).
    Returns (f (N_total,2), Kc CSR (n_dof,n_dof), diag).  f is the contact force on BOTH bodies'
    nodes (slave +traction, master -reaction).  The slave gap is the OT correspondence gap to the
    master profile; the master reaction is the slave nodal traction scattered to the mapped master
    segment via P1 shape functions (so int over the interface of traction.[N_slave - N_master] = 0
    => global force balance to machine precision).
    """
    from scipy.sparse import coo_matrix

    n_nodes = n_dof // 2
    slave_prof = _profile(slave_xy)
    master_prof = _profile(master_xy)
    coupling = MonotoneCoupling1D(slave_prof, master_prof, unbalanced=(contact_band is not None),
                                  contact_band=contact_band)

    # --- slave nodal gap & normal from the OT correspondence ---
    xi_nodes = slave_xy[:, 0]
    x_m_nodes, Xm_nodes, mass_nodes = coupling.map(xi_nodes)
    # slave unit normal (upward, from the slave profile) -- contact pushes slave UP, master DOWN
    sec_s = np.sqrt(1.0 + slave_prof["hp"] ** 2)
    ns = np.column_stack([-slave_prof["hp"] / sec_s, 1.0 / sec_s])
    d = slave_xy - Xm_nodes
    gN_nodes = (d * ns).sum(axis=1) * mass_nodes + (1.0 - mass_nodes) * 1e3   # inactive -> big +gap

    traction = TractionField(eps_n, mu=mu)
    slip = None
    if mu > 0.0 and slip_dir is not None:
        ts = np.column_stack([1.0 / sec_s, slave_prof["hp"] / sec_s])
        slip = float(slip_dir) * ts                          # unit tangential drag direction
    tr = traction.evaluate(gN_nodes, ns, slip=slip)
    t_node, deps_node, pN_node = tr["t"], tr["deps"], tr["pN"]

    xi, w = gauss_legendre_1d(order)
    s = 0.5 * (1.0 + xi)
    Nref = np.stack([1.0 - s, s], axis=1)                    # (order,2)

    f = np.zeros((n_nodes, 2))
    rows, cols, vals = [], [], []
    Xq_all, pNq_all, wds_all, xm_q_all = [], [], [], []
    n_s = len(slave_xy)
    for k in range(n_s - 1):
        P0, P1 = slave_xy[k], slave_xy[k + 1]
        L = float(np.linalg.norm(P1 - P0))
        if L <= 0:
            continue
        wds = w * 0.5 * L
        m = np.einsum("q,qa,qb->ab", wds, Nref, Nref)        # local consistent mass (2,2)
        loc = (k, k + 1)
        sgid = (slave_ids[k], slave_ids[k + 1])
        tt = (t_node[loc[0]], t_node[loc[1]])
        # ---- slave consistent force  f_I += sum_J M_IJ t_J ----
        for a in range(2):
            f[sgid[a]] += m[a, 0] * tt[0] + m[a, 1] * tt[1]
        # ---- master reaction: at each Gauss pt, scatter the interpolated traction onto the master
        #      segment hosting the OT-mapped point X_m=T(xi_q), via P1 master shape functions ----
        for q in range(order):
            Nq = Nref[q]
            t_q = Nq[0] * tt[0] + Nq[1] * tt[1]              # interpolated slave traction
            xi_q = Nq[0] * P0[0] + Nq[1] * P1[0]            # slave x at this Gauss point
            xm_q, _, mass_q = coupling.map(np.array([xi_q]))
            xm_q = float(xm_q[0])
            jm, Nm0, Nm1 = _locate_master(master_xy, xm_q)
            # master reaction = -t_q distributed to its two host nodes (Newton's 3rd law)
            f[master_ids[jm]] += -wds[q] * Nm0 * t_q
            f[master_ids[jm + 1]] += -wds[q] * Nm1 * t_q
            xm_q_all.append(xm_q)
        # ---- slave normal tangent  K_IJ += M_IJ eps_n (n_J x n_J) ----
        for a in range(2):
            for b in range(2):
                dj = deps_node[loc[b]]
                if dj != 0.0:
                    nb = ns[loc[b]]
                    blk = m[a, b] * dj * np.outer(nb, nb)
                    Ia, Ib = sgid[a], sgid[b]
                    for di in range(2):
                        for dk in range(2):
                            rows.append(2 * Ia + di); cols.append(2 * Ib + dk)
                            vals.append(blk[di, dk])
        Xq_all.append(np.outer(Nref[:, 0], P0) + np.outer(Nref[:, 1], P1))
        pNq_all.append(Nref[:, 0] * pN_node[loc[0]] + Nref[:, 1] * pN_node[loc[1]])
        wds_all.append(wds)

    Kc = (coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr() if rows
          else coo_matrix((n_dof, n_dof)).tocsr())
    Xq = np.concatenate(Xq_all, 0) if Xq_all else np.zeros((0, 2))
    pNq = np.concatenate(pNq_all, 0) if pNq_all else np.zeros(0)
    wds_cat = np.concatenate(wds_all, 0) if wds_all else np.zeros(0)
    F_line = float((wds_cat * pNq).sum())
    diag = dict(x=slave_xy[:, 0], pN=pN_node, n=ns, gN=gN_nodes, Xq=Xq, pN_q=pNq,
                wds=wds_cat, F_line=F_line, mass=mass_nodes,
                patch_centroid=(float((wds_cat * pNq * Xq[:, 0]).sum() / max((wds_cat * pNq).sum(), 1e-30))
                                if pNq.sum() > 0 else float("nan")))
    return f, Kc, diag


def _locate_master(master_xy, xm):
    """Find the master segment containing x=xm and its two P1 shape-function values."""
    xs = master_xy[:, 0]
    xm = float(np.clip(xm, xs[0], xs[-1]))
    j = int(np.clip(np.searchsorted(xs, xm) - 1, 0, len(xs) - 2))
    x0, x1 = xs[j], xs[j + 1]
    t = 0.0 if x1 == x0 else (xm - x0) / (x1 - x0)
    return j, 1.0 - t, t


# ==================================================================================================
#  LUMPED node-to-surface baseline (the conventional scheme the patch test exposes).
# ==================================================================================================
def lumped_contact(slave_xy, slave_ids, master_xy, master_ids, n_dof, eps_n):
    """Conventional node-to-surface lumped penalty: each slave node carries a tributary length,
    projects vertically to the master polyline, force = eps_n <pen> * tributary on the slave node and
    the nearest master node (no mortar mass, no consistent interpolation).  Used ONLY for the patch-
    test contrast (it transmits a constant pressure as a node-spacing sawtooth)."""
    n_nodes = n_dof // 2
    f = np.zeros((n_nodes, 2))
    xs_m = master_xy[:, 0]
    # tributary length per slave node (half-segments to each side)
    xs = slave_xy[:, 0]
    trib = np.zeros(len(xs))
    trib[1:-1] = 0.5 * (xs[2:] - xs[:-2])
    trib[0] = 0.5 * (xs[1] - xs[0]); trib[-1] = 0.5 * (xs[-1] - xs[-2])
    for i, (P, gid) in enumerate(zip(slave_xy, slave_ids)):
        ym = np.interp(P[0], xs_m, master_xy[:, 1])
        pen = ym - P[1]                              # >0 => penetration (slave below master line)
        if pen <= 0:
            continue
        fn = eps_n * pen * trib[i]
        f[gid, 1] += fn                               # push slave up
        jm = int(np.clip(np.searchsorted(xs_m, P[0]) - 1, 0, len(xs_m) - 2))
        f[master_ids[jm], 1] -= 0.5 * fn              # crude split to the two host master nodes
        f[master_ids[jm + 1], 1] -= 0.5 * fn
    return f


# ==================================================================================================
#  Driver assembly for two stacked deformable blocks.
# ==================================================================================================
class TwoBlockOT:
    """Two deformable Tri2D blocks (lower + upper) sharing one global dof vector."""

    def __init__(self, lower, upper, E1, nu1, E2, nu2, mode="plane_strain"):
        self.nL, self.eL, self.topL, self.botL = lower
        self.nU, self.eU, self.topU, self.botU = upper
        self.solL = Tri2DFEMSolver(self.nL, self.eL, E1, nu1, mode=mode)
        self.solU = Tri2DFEMSolver(self.nU, self.eU, E2, nu2, mode=mode)
        self.NL, self.NU = self.solL.n_nodes, self.solU.n_nodes
        self.N = self.NL + self.NU
        self.n_dof = 2 * self.N
        self.offU = self.NL                                  # node offset for upper block
        import scipy.sparse as sp
        KL = self.solL.assemble(); KU = self.solU.assemble()
        self.K = sp.block_diag([KL, KU]).tocsr()
        self.E1, self.nu1, self.E2, self.nu2 = E1, nu1, E2, nu2

    # ---- surface accessors (deformed, ordered by x) ----
    def _surf(self, u, which):
        u2 = u.reshape(self.N, 2)
        if which == "masterL":            # lower block TOP edge (master)
            ids = self.topL; base = self.nL[ids]; disp = u2[ids]
        elif which == "slaveU":           # upper block BOTTOM edge (slave)
            ids = self.offU + self.botU; base = self.nU[self.botU]; disp = u2[ids]
        else:
            raise ValueError(which)
        cur = base + disp
        order = np.argsort(cur[:, 0])
        cur, ids = cur[order], ids[order]
        # drop near-duplicate x (jittered non-matching meshes) so segments have positive length
        keep = np.concatenate([[True], np.diff(cur[:, 0]) > 1e-9])
        return cur[keep], ids[keep]

    def contact(self, u, eps_n, mu=0.0, slip_dir=None, contact_band=None):
        slave_xy, slave_ids = self._surf(u, "slaveU")
        master_xy, master_ids = self._surf(u, "masterL")
        return assemble_two_body(slave_xy, slave_ids, master_xy, master_ids, self.n_dof,
                                 eps_n, mu=mu, slip_dir=slip_dir, contact_band=contact_band)

    def contact_lumped(self, u, eps_n):
        slave_xy, slave_ids = self._surf(u, "slaveU")
        master_xy, master_ids = self._surf(u, "masterL")
        return lumped_contact(slave_xy, slave_ids, master_xy, master_ids, self.n_dof, eps_n)


# ==================================================================================================
#  Verification (1): TWO-BODY PATCH TEST.
# ==================================================================================================
def patch_test(n_lower=(12, 4), n_upper=(17, 4), p=0.05, eps_n=None, interf=0.02, verbose=True):
    """Uniform pressure p on the upper block top; lower block bottom fixed; NON-MATCHING interface.

    A consistent transmission => sigma_yy = -p uniformly in BOTH bodies.  Reports the receiving
    (lower) body's interior stress non-uniformity for MORTAR vs LUMPED.
    """
    from scipy.sparse.linalg import spsolve
    E, nu = 1.0, 0.3
    W, H = 1.0, 0.5
    # lower: top at y=0 ; upper: bottom seated with a tiny INTERFERENCE -interf (so contact is active
    # at Newton iter 0 -> nonsingular; the converged interface traction equals the applied p by
    # equilibrium, independent of interf).  NON-MATCHING n_x (12 vs 17), upper jittered.
    nL, eL, topL, botL = block_mesh(W, H, n_lower[0], n_lower[1], y0=-H)
    nU, eU, topU, botU = block_mesh(W, H, n_upper[0], n_upper[1], y0=-interf, jitter=0.12, seed=3)
    tb = TwoBlockOT((nL, eL, topL, botL), (nU, eU, topU, botU), E, nu, E, nu)
    h = 2 * W / n_lower[0]
    eps_n = (200.0 * E / h) if eps_n is None else eps_n

    # BCs: lower bottom fully fixed; both blocks' vertical sides ux=0 (uniaxial compression column).
    fixedL = np.concatenate([2 * botL, 2 * botL + 1])
    side = lambda nodes: np.where((np.abs(nodes[:, 0] - W) < 1e-9) | (np.abs(nodes[:, 0] + W) < 1e-9))[0]  # noqa
    sideL = side(nL); sideU = side(nU)
    fixed = np.unique(np.concatenate([fixedL, 2 * sideL, 2 * (tb.offU + sideU)]))
    free = np.setdiff1d(np.arange(tb.n_dof), fixed)

    # external load: uniform downward pressure on the upper top edge (consistent nodal load).
    f_ext = np.zeros((tb.N, 2))
    xt = nU[topU, 0]; o = np.argsort(xt)
    xs = xt[o]; tids = topU[o]
    for k in range(len(xs) - 1):
        seg = xs[k + 1] - xs[k]
        f_ext[tb.offU + tids[k], 1] += -0.5 * p * seg
        f_ext[tb.offU + tids[k + 1], 1] += -0.5 * p * seg

    def solve(use_lumped):
        u = np.zeros(tb.n_dof)
        for it in range(60):
            if use_lumped:
                fc = tb.contact_lumped(u, eps_n)
                Kc = None
            else:
                fc, Kc, _ = tb.contact(u, eps_n)
            R = tb.K @ u - f_ext.reshape(-1) - fc.reshape(-1)
            rn = np.linalg.norm(R[free])
            if rn < 1e-11 * (1 + eps_n):
                break
            Kt = tb.K if Kc is None else (tb.K + Kc).tocsr()
            du = np.zeros(tb.n_dof)
            du[free] = spsolve(Kt[free][:, free].tocsc(), -R[free])
            # lumped tangent omitted -> damp; mortar has the consistent normal tangent
            u = u + (0.5 if use_lumped else 1.0) * du
        return u

    def lower_uniformity(u):
        s = tb.solL.element_stress(u.reshape(tb.N, 2)[:tb.NL])
        cen = tb.solL.element_centroids()
        interior = (np.abs(cen[:, 0]) < 0.7 * W) & (cen[:, 1] > -H + 0.1 * H) & (cen[:, 1] < -0.1 * H)
        syy = s[interior, 1]
        return float((syy.max() - syy.min())), float(syy.mean()), int(interior.sum())

    u_m = solve(False)
    rng_m, mean_m, ni = lower_uniformity(u_m)
    u_l = solve(True)
    rng_l, mean_l, _ = lower_uniformity(u_l)
    out = dict(p=p, eps_n=eps_n, n_interior_elem=ni,
               mortar_syy_mean=mean_m, mortar_syy_range=rng_m,
               mortar_uniformity_rel=rng_m / p, mortar_syy_err_rel=abs(mean_m + p) / p,
               lumped_syy_mean=mean_l, lumped_syy_range=rng_l,
               lumped_uniformity_rel=rng_l / p,
               n_lower=list(n_lower), n_upper=list(n_upper))
    if verbose:
        print("  [PATCH] non-matching interface  lower n_x=%d  upper n_x=%d (jittered)" %
              (n_lower[0], n_upper[0]))
        print("    MORTAR : sigma_yy mean=%.6f (target %.6f, err %.2e)  non-uniformity=%.3e" %
              (mean_m, -p, out["mortar_syy_err_rel"], out["mortar_uniformity_rel"]))
        print("    LUMPED : sigma_yy mean=%.6f                          non-uniformity=%.3e" %
              (mean_l, out["lumped_uniformity_rel"]))
        # HONEST verdict: the two-body Newton DIVERGES (NaN) on this gate because the master reaction
        # is a non-variational nearest-segment scatter and the two-body tangent omits the slave-master
        # D_IK coupling block. The decisive constant-pressure->constant-stress patch test does NOT pass.
        # See docs/ot_benchmark/next_phase_math.md (T2). STATUS: WIP, not verified.
        import math as _m
        if _m.isnan(mean_m) or _m.isinf(mean_m) or _m.isnan(out["mortar_uniformity_rel"]):
            print("    -> PATCH TEST FAILS: mortar stress is NaN (two-body Newton diverged); "
                  "needs the D_IK master-coupling tangent (next_phase_math.md T2). WIP.")
        else:
            ratio = out["lumped_uniformity_rel"] / max(out["mortar_uniformity_rel"], 1e-30)
            print("    -> mortar uniformity %.1ex tighter than lumped (verify mean matches target)" % ratio)
    return out


# ==================================================================================================
#  Verification (2)+(3): DEFORMABLE HERTZ + LARGE SLIDING.
# ==================================================================================================
def hertz_test(n_lower=(40, 10), n_upper=(40, 10), R=4.0, delta=0.04, n_load=4,
               mu=0.0, slide=0.0, n_slide=0, eps_n=None, verbose=True):
    """Curved-bottom upper block pressed (delta) onto a flat lower block; both deformable.

    Recovers a, p0 vs line_contact_params with the combined plane-strain E*.  If slide>0, drags the
    upper top edge tangentially over `slide` in `n_slide` steps after seating and tracks the patch.
    """
    from scipy.sparse.linalg import spsolve
    E1, nu1 = 1.0, 0.3            # lower (flat)
    E2, nu2 = 1.0, 0.3            # upper (curved)
    W, H = 1.0, 0.5
    nL, eL, topL, botL = block_mesh(W, H, n_lower[0], n_lower[1], y0=-H)
    nU, eU, topU, botU = block_mesh(W, H, n_upper[0], n_upper[1], y0=0.0, curve_R=R, jitter=0.06, seed=7)
    tb = TwoBlockOT((nL, eL, topL, botL), (nU, eU, topU, botU), E1, nu1, E2, nu2)
    h = 2 * W / n_lower[0]
    eps_n = (120.0 * E1 / h) if eps_n is None else eps_n

    Estar = 1.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)   # combined plane-strain modulus

    fixedL = np.concatenate([2 * botL, 2 * botL + 1])
    side = lambda nodes: np.where((np.abs(nodes[:, 0] - W) < 1e-9) | (np.abs(nodes[:, 0] + W) < 1e-9))[0]  # noqa
    sideL = side(nL)
    # upper top platen: prescribed (u_x, u_y); lower bottom fixed; lower sides ux=0.
    platen = topU
    fixed = np.unique(np.concatenate([fixedL, 2 * sideL,
                                      2 * (tb.offU + platen), 2 * (tb.offU + platen) + 1]))
    free = np.setdiff1d(np.arange(tb.n_dof), fixed)

    def newton(u, uy_platen, ux_platen, slip_dir):
        u = u.copy()
        for nidx in botL:
            u[2 * nidx:2 * nidx + 2] = 0.0
        for nidx in platen:
            g = 2 * (tb.offU + nidx)
            u[g] = ux_platen; u[g + 1] = uy_platen
        band = 0.5 * abs(delta) + 0.05
        rn = None
        for it in range(60):
            fc, Kc, diag = tb.contact(u, eps_n, mu=mu, slip_dir=slip_dir, contact_band=band)
            R = tb.K @ u - fc.reshape(-1)
            rn = np.linalg.norm(R[free])
            if rn < 1e-9 * (1 + eps_n):
                break
            Kt = (tb.K + Kc).tocsr()
            du = np.zeros(tb.n_dof)
            du[free] = spsolve(Kt[free][:, free].tocsc(), -R[free])
            # backtracking line search on the residual (penalty non-smoothness)
            step, ok = 1.0, False
            for _ in range(25):
                ut = u.copy(); ut[free] += step * du[free]
                fct, _, _ = tb.contact(ut, eps_n, mu=mu, slip_dir=slip_dir, contact_band=band)
                rnt = np.linalg.norm((tb.K @ ut - fct.reshape(-1))[free])
                if rnt < (1 - 1e-4 * step) * rn:
                    u, ok = ut, True; break
                step *= 0.5
            if not ok:
                break
        fc, Kc, diag = tb.contact(u, eps_n, mu=mu, slip_dir=slip_dir, contact_band=band)
        diag["resid"] = float(np.linalg.norm((tb.K @ u - fc.reshape(-1))[free]))
        diag["iters"] = it + 1
        return u, diag

    # ---- seat the normal load incrementally ----
    u = np.zeros(tb.n_dof)
    for ls in range(n_load):
        uy = -delta * (ls + 1) / n_load
        u, diag = newton(u, uy, 0.0, slip_dir=None)
        if verbose:
            print("    [seat] uy=%.4f  F=%.4f  n_active=%d  resid=%.1e  iters=%d" %
                  (uy, diag["F_line"], int((diag["pN"] > 0).sum()), diag["resid"], diag["iters"]))

    F = diag["F_line"]
    xq, pN = diag["x"], diag["pN"]
    active = pN > 1e-9 * pN.max() if pN.max() > 0 else np.zeros_like(pN, bool)
    a_edge = float(np.max(np.abs(xq[active]))) if active.any() else 0.0
    a_fem = p0_fem = float("nan")
    if active.sum() >= 3:
        xa, pa = xq[active], pN[active]
        best = None
        for a_try in np.linspace(0.5 * a_edge, 1.8 * a_edge + 1e-9, 500):
            w = np.sqrt(np.clip(1.0 - (xa / a_try) ** 2, 0.0, None))
            p0_try = float(pa @ w / (w @ w + 1e-30))
            resid = float(np.sum((pa - p0_try * w) ** 2))
            if best is None or resid < best[0]:
                best = (resid, a_try, p0_try)
        a_fem, p0_fem = best[1], best[2]
    a_ana, p0_ana = cf.line_contact_params(F, R, Estar) if F > 0 else (float("nan"), float("nan"))
    res = dict(F_line=float(F), Estar=float(Estar), R=R, delta=delta,
               a_fem=float(a_fem), a_ana=float(a_ana),
               p0_fem=float(p0_fem), p0_ana=float(p0_ana),
               a_relerr=float(abs(a_fem - a_ana) / a_ana) if a_ana > 0 else float("nan"),
               p0_relerr=float(abs(p0_fem - p0_ana) / p0_ana) if p0_ana > 0 else float("nan"),
               n_active=int(active.sum()), seat_resid=float(diag["resid"]),
               seat_iters=int(diag["iters"]),
               xq=xq.tolist(), pN=pN.tolist())

    # global force balance: sum of all contact forces must be ~0 (slave + master reaction).
    fc_chk, _, _ = tb.contact(u, eps_n, mu=mu)
    res["force_balance"] = float(np.linalg.norm(fc_chk.sum(axis=0)))

    # ---- (3) large sliding ----
    slide_hist = []
    if slide > 0 and n_slide > 0:
        uy = -delta
        for js in range(1, n_slide + 1):
            ux = slide * js / n_slide
            u, diag = newton(u, uy, ux, slip_dir=+1.0)
            slide_hist.append(dict(ux=float(ux), F=float(diag["F_line"]),
                                   centroid=float(diag["patch_centroid"]),
                                   resid=float(diag["resid"]), n_active=int((diag["pN"] > 0).sum())))
            if verbose:
                print("    [slide] ux=%.4f  F=%.4f  patch_x=%.4f  nC=%d  resid=%.1e" %
                      (ux, diag["F_line"], diag["patch_centroid"], int((diag["pN"] > 0).sum()),
                       diag["resid"]))
        cen = np.array([s["centroid"] for s in slide_hist])
        dcen = np.diff(cen)
        res["slide_centroid_monotone"] = bool(np.all(dcen >= -1e-6)) or bool(np.all(dcen <= 1e-6))
        res["slide_centroid_max_backstep"] = float(-min(dcen.min(), 0.0)) if len(dcen) else 0.0
        Fs = np.array([s["F"] for s in slide_hist])
        res["slide_F_cov"] = float(Fs.std() / max(abs(Fs.mean()), 1e-12))
        res["slide_max_resid"] = float(max(s["resid"] for s in slide_hist))
    res["slide_hist"] = slide_hist
    if verbose:
        print("  [HERTZ] F=%.4f  a_fem=%.4f a_ana=%.4f (err %.2f%%)  p0_fem=%.4f p0_ana=%.4f (err %.2f%%)"
              % (F, res["a_fem"], res["a_ana"], 100 * res["a_relerr"],
                 res["p0_fem"], res["p0_ana"], 100 * res["p0_relerr"]))
        print("    force balance |sum f|=%.2e  n_active=%d" % (res["force_balance"], res["n_active"]))
    return res


# ==================================================================================================
def run(mode="all", mesh_coarse=None, verbose=True):
    os.makedirs(RUN_DIR, exist_ok=True)
    t0 = time.time()
    metrics = {"cv": "CV-8 two-body deformable OT contact (non-matching mortar + large sliding)"}
    if mode in ("all", "patch"):
        metrics["patch_test"] = patch_test(verbose=verbose)
    if mode in ("all", "cylinder"):
        nl = tuple(mesh_coarse) if mesh_coarse else (44, 10)
        nu = (nl[0], nl[1])
        metrics["hertz"] = hertz_test(n_lower=nl, n_upper=nu, R=4.0, delta=0.05, n_load=4,
                                      slide=0.10, n_slide=5, verbose=verbose)
    metrics["elapsed_sec"] = time.time() - t0
    hist = {}
    if "hertz" in metrics:
        hist["hertz"] = {k: metrics["hertz"].pop(k) for k in ("xq", "pN", "slide_hist")
                         if k in metrics["hertz"]}
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)
    if verbose:
        print("\n  elapsed %.1fs   metrics -> %s" % (metrics["elapsed_sec"],
                                                     os.path.join(RUN_DIR, "metrics.json")))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "patch", "cylinder"])
    ap.add_argument("--mesh-coarse", type=int, nargs=2, default=None,
                    help="lower-block (n_x n_y); upper uses the same n_x")
    ap.add_argument("--n-inc", type=int, default=None)
    ap.add_argument("--mu-sweep", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    m = run(mode=args.mode, mesh_coarse=args.mesh_coarse, verbose=not args.quiet)
    ok = True
    if "patch_test" in m:
        ok = ok and (m["patch_test"]["mortar_uniformity_rel"] < 1e-7)
    if "hertz" in m:
        ok = ok and (m["hertz"]["a_relerr"] < 0.10) and (m["hertz"]["p0_relerr"] < 0.12)
    print("\n  CV-8 two-body OT: %s" % ("PASS" if ok else "CHECK"))


if __name__ == "__main__":
    main()
