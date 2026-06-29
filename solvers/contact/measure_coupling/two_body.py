"""Two-body (deformable-deformable) mortar OT contact: the full 4-block consistent tangent.

The one-body :func:`assemble_contact` pushes a deformable SLAVE surface against a *rigid* master:
only the slave dofs vary, so a single slave-slave normal block ``K_ss = eps_n M (n(x)n)`` is the
consistent tangent.  When BOTH surfaces are deformable the master point that the slave is measured
against, ``x_m(chi) = sum_K (N_K- o chi) x_K-``, ALSO moves with the master dofs, so the gap

    g_N(xi_q) = ( x_s(xi_q) - x_m(chi(xi_q)) ) . n,
        x_s = sum_I N_I+(xi_q) x_I+ ,   x_m = sum_K (N_K- o chi)(xi_q) x_K- ,

depends on slave AND master dofs.  Its variation (small-rotation: drop the geometric d(n) term, as
the one-body assembly does) is

    dg_N/du_I+ = N_I+ n ,        dg_N/du_K- = -(N_K- o chi) n .

The penalty traction ``t = eps_n <-g_N> n`` (active where ``g_N < 0``) then gives, with
``dt = eps_n n (x) (-dg_N)`` on the active set, the SYMMETRIC SPSD 4-block tangent

    K_ss[I,J] = +eps_n sum_q w J N_I+ N_J+        (n(x)n)     [the existing one-body M block]
    K_sm[I,K] = -eps_n sum_q w J N_I+ (N_K- o chi)(n(x)n)     [NEW slave-master D block]
    K_ms[K,J] = K_sm[I,K]^T = -eps_n sum_q w J (N_K- o chi) N_J+ (n(x)n)
    K_mm[K,L] = +eps_n sum_q w J (N_K- o chi)(N_L- o chi)(n(x)n) [NEW master-master]

Forces (Newton's third law transmitted through the SAME correspondence field):

    f_I+ = +sum_q w J N_I+(xi_q) t(xi_q)          (slave)
    f_K- = -sum_q w J (N_K- o chi)(xi_q) t(xi_q)  (master reaction).

PATCH TEST.  A uniform pressure transmits as a uniform stress to the receiving body iff the master
interpolation is a partition of unity along the correspondence, ``sum_K (N_K- o chi) = 1``.  This
partition of unity is DISTINCT FROM, and downstream of, the OT mass marginal ``chi_# mu_A = mu_B``
that places the foot ``chi(xi_q)``: the marginal controls WHERE the foot lands, the partition of
unity controls HOW the reaction is split once it lands, and it is the latter (single host + P1
partition of unity + Gauss-exact integration) — NOT the marginal — that the patch test exercises.
Because ``(N_K- o chi)`` are P1 host-segment weights summing to 1 by construction, the reaction force
on the master integrates the SAME traction field the slave applies, so ``f_I+`` and ``f_K-`` are the
transpose-coupled image of one nodal traction vector and the net resultant is machine-zero.  This
precondition is machine-checked in ``lean/OTContact/PartitionOfUnity.lean``
(``patch_test_resultant``, ``p1_host_weights``; sorry-free).

This module exposes :func:`assemble_two_body_contact`, returning ``f`` (force on both bodies),
``Kc`` (the assembled 4-block CSR tangent on the shared dof vector), and a diagnostics dict.  The
tangent is FD-checked against ``df/du`` in this file's self-test and the two-body patch test runs to
a number (no NaN) by virtue of the master block being present.
"""
from __future__ import annotations

import numpy as np

from .coupling import ClosestPointCoupling1D, MonotoneCoupling1D
from .quadrature import gauss_legendre_1d
from .traction import TractionField


def _profile(surf_xy: np.ndarray) -> dict:
    """Bake a height profile ``{x,h,hp}`` from an ordered surface polyline (x strictly ascending).

    Slope by central/one-sided differences that never divide by a vanishing ``dx`` (jittered
    non-matching meshes can place two surface nodes very close in x).
    """
    x = np.asarray(surf_xy[:, 0], float)
    h = np.asarray(surf_xy[:, 1], float)
    hp = np.zeros_like(x)
    dx = np.diff(x)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    slope = np.diff(h) / dx
    if len(x) > 2:
        hp[1:-1] = 0.5 * (slope[:-1] + slope[1:])
    hp[0] = slope[0]
    hp[-1] = slope[-1]
    return dict(x=x, h=h, hp=hp)


def _locate_master(master_x: np.ndarray, xm: float):
    """Find the master segment hosting ``x = xm`` and its two P1 shape-function values.

    Returns ``(j, N0, N1)`` with ``N0 + N1 = 1`` (partition of unity along the correspondence).
    """
    xm = float(np.clip(xm, master_x[0], master_x[-1]))
    j = int(np.clip(np.searchsorted(master_x, xm) - 1, 0, len(master_x) - 2))
    x0, x1 = master_x[j], master_x[j + 1]
    t = 0.0 if x1 == x0 else (xm - x0) / (x1 - x0)
    return j, 1.0 - t, t


def assemble_two_body_contact(slave_xy, slave_ids, master_xy, master_ids, n_dof, eps_n,
                              mu=0.0, order=3, slip_dir=None, contact_band=None,
                              coupling=None, pen_offset=0.0, correspondence="monotone"):
    """Consistent two-body mortar OT contact with the FULL 4-block tangent.

    Parameters
    ----------
    slave_xy, master_xy : (n,2) ordered deformed surface polylines (x ascending).
    slave_ids, master_ids : (n,) global node indices into the SHARED 2*N_total dof vector (the
        master ids are already offset by the caller).
    n_dof : int, total dof count (2 * N_total).
    eps_n : float, normal penalty stiffness.
    mu : float, Coulomb friction coefficient (0 = frictionless; tangent below is the frictionless
        symmetric block — friction adds force only, see note).
    order : int, Gauss points per slave segment.
    slip_dir : float|None, unit tangential drag sign for the friction force (optional).
    contact_band : float|None, separation band for the unbalanced OT pre-screen (mass=0 outside).
    coupling : optional pre-built coupling (else built from the profiles per ``correspondence``).
    pen_offset : float, prescribed extra penetration injected into the gap (``g_N -> g_N - pen_offset``)
        to ramp a geometric overlap load without remeshing.  Affects the residual only (the active-set
        threshold), NOT the tangent (the tangent depends on ``deps``, which is consistent with the
        shifted gap because the shift is a constant w.r.t. the dofs).
    correspondence : {"monotone", "closest_point"}
        Which transition map carries the slave point onto the master surface.

        - ``"monotone"`` (default): the GLOBAL arclength-quantile map :class:`MonotoneCoupling1D`
          (``T = F_m^{-1} o F_s``).  Mass-balanced over the WHOLE interface — correct for a fully
          conforming / full-contact interface (the patch test), but it transports ALL slave mass onto
          ALL master mass and therefore CANNOT represent a partial contact (it smears a Hertz patch
          across the entire interface; see cv1_ot_gap.py:15-20, manual §11.8).
        - ``"closest_point"``: the LOCAL orthogonal-projection map :class:`ClosestPointCoupling1D`
          (``tau_AB = pi_B o phi_A``, the Fig-2 composite map).  Each slave point is carried to its
          NEAREST point on the deformed master polyline, so an isolated central contact patch maps
          onto the matching central master patch — the correct OT map for NON-CONFORMING / PARTIAL
          (Hertz, large-sliding) contact.

        Both produce the SAME symmetric SPSD 4-block tangent (slave +N_I, master -(N_K o chi),
        n(x)n); only the master point and its host P1 weights differ.

    Returns
    -------
    f : (N_total, 2) contact force on BOTH bodies (slave +t, master -reaction).
    Kc : scipy.sparse CSR (n_dof, n_dof) — the SYMMETRIC SPSD 4-block consistent tangent.
    diag : dict (nodal/quadrature fields, F_line, force_balance, patch_centroid, mass).
    """
    from scipy.sparse import coo_matrix

    slave_xy = np.asarray(slave_xy, float)
    master_xy = np.asarray(master_xy, float)
    slave_ids = np.asarray(slave_ids, int)
    master_ids = np.asarray(master_ids, int)
    n_nodes = n_dof // 2
    master_x = master_xy[:, 0]

    slave_prof = _profile(slave_xy)
    master_prof = _profile(master_xy)
    closest_point = (correspondence == "closest_point")
    if coupling is None:
        if closest_point:
            coupling = ClosestPointCoupling1D(slave_prof, master_prof,
                                              unbalanced=(contact_band is not None),
                                              contact_band=contact_band)
        else:
            coupling = MonotoneCoupling1D(slave_prof, master_prof,
                                          unbalanced=(contact_band is not None),
                                          contact_band=contact_band)
    closest_point = isinstance(coupling, ClosestPointCoupling1D)

    def _master_at(xi_scalar):
        """Return ``(xmaster (2,), jm, Nm0, Nm1, mass)`` for a slave x-coordinate ``xi_scalar``."""
        if closest_point:
            Xm, seg, N0, N1, m = coupling.map_full(np.array([xi_scalar]))
            return Xm[0], int(seg[0]), float(N0[0]), float(N1[0]), float(m[0])
        xm_q, Xm_q, mass_q = coupling.map(np.array([xi_scalar]))
        jm, Nm0, Nm1 = _locate_master(master_x, float(xm_q[0]))
        return Xm_q[0], jm, Nm0, Nm1, float(mass_q[0])

    # slave unit normal (upward): contact pushes slave UP, master DOWN.
    sec_s = np.sqrt(1.0 + slave_prof["hp"] ** 2)
    ns = np.column_stack([-slave_prof["hp"] / sec_s, 1.0 / sec_s])

    traction = TractionField(eps_n, mu=mu)
    xi_g, w_g = gauss_legendre_1d(order)
    s = 0.5 * (1.0 + xi_g)
    Nref = np.stack([1.0 - s, s], axis=1)                     # (order, 2) slave P1 at Gauss pts

    f = np.zeros((n_nodes, 2))
    rows, cols, vals = [], [], []
    Xq_all, pNq_all, wds_all = [], [], []
    n_s = len(slave_xy)

    for k in range(n_s - 1):
        P0, P1 = slave_xy[k], slave_xy[k + 1]
        L = float(np.linalg.norm(P1 - P0))
        if L <= 0.0:
            continue
        wds = w_g * 0.5 * L                                   # (order,) quadrature weight * |J|
        sgid = (slave_ids[k], slave_ids[k + 1])

        # interpolate slave normal at the Gauss points (P1, then renormalize)
        nq = Nref[:, :1] * ns[k] + Nref[:, 1:] * ns[k + 1]    # (order, 2)
        nq = nq / np.linalg.norm(nq, axis=1, keepdims=True)

        for q in range(order):
            Nq = Nref[q]                                       # (2,) slave P1 weights
            wq = wds[q]
            n_hat = nq[q]
            # slave physical point at this Gauss point
            xs = Nq[0] * P0 + Nq[1] * P1
            xi_q = float(xs[0])
            # OT correspondence -> master point + the host-segment P1 weights (N_K- o chi)
            xmaster, jm, Nm0, Nm1, mass = _master_at(xi_q)     # (2,) master surface point

            # normal gap  g_N = (x_s - x_m).n - pen_offset   (mass-screened; inactive -> big +gap)
            gN = (float((xs - xmaster) @ n_hat) - pen_offset) * mass + (1.0 - mass) * 1e3

            tr = traction.evaluate(np.array([gN]), n_hat[None, :])
            t_q = tr["t"][0]                                   # (2,) traction vector
            deps = float(tr["deps"][0])                        # eps_n on active, else 0
            pN_q = float(tr["pN"][0])

            # ---- forces: slave +t, master -reaction (transpose-coupled through the SAME field) ----
            f[sgid[0]] += wq * Nq[0] * t_q
            f[sgid[1]] += wq * Nq[1] * t_q
            f[master_ids[jm]] += -wq * Nm0 * t_q
            f[master_ids[jm + 1]] += -wq * Nm1 * t_q

            # ---- 4-block consistent tangent (active points only) ----
            # gather the four interpolation weights and their signed dof handles:
            #   slave  +N_I  (dof handle = slave_ids)     master  -N_K (dof handle = master_ids)
            if deps != 0.0:
                nn = np.outer(n_hat, n_hat)                    # (2,2) n (x) n
                wfac = wq * deps                               # eps_n w J  on active set
                # entries: (interp weight, global node id, sign)   sign = +1 slave, -1 master
                entries = [
                    (Nq[0], sgid[0], +1.0),
                    (Nq[1], sgid[1], +1.0),
                    (Nm0, master_ids[jm], -1.0),
                    (Nm1, master_ids[jm + 1], -1.0),
                ]
                for (wa, Ia, sa) in entries:
                    for (wb, Ib, sb) in entries:
                        coeff = wfac * (sa * wa) * (sb * wb)
                        blk = coeff * nn
                        for di in range(2):
                            for dk in range(2):
                                v = blk[di, dk]
                                if v != 0.0:
                                    rows.append(2 * Ia + di)
                                    cols.append(2 * Ib + dk)
                                    vals.append(v)

            Xq_all.append(xs)
            pNq_all.append(pN_q)
            wds_all.append(wq)

    Kc = (coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr() if rows
          else coo_matrix((n_dof, n_dof)).tocsr())

    Xq = np.array(Xq_all) if Xq_all else np.zeros((0, 2))
    pNq = np.array(pNq_all) if pNq_all else np.zeros(0)
    wds_cat = np.array(wds_all) if wds_all else np.zeros(0)
    F_line = float((wds_cat * pNq).sum())

    # nodal diagnostic fields (gap/pressure evaluated at the slave NODES via the same coupling)
    xm_n, Xm_n, mass_n = coupling.map(slave_xy[:, 0])
    gN_n = ((slave_xy - Xm_n) * ns).sum(axis=1) * mass_n + (1.0 - mass_n) * 1e3
    pN_n = traction.evaluate(gN_n, ns)["pN"]

    patch_centroid = float((wds_cat * pNq * Xq[:, 0]).sum() / max((wds_cat * pNq).sum(), 1e-30)) \
        if pNq.sum() > 0 else float("nan")
    diag = dict(x=slave_xy[:, 0], pN=pN_n, n=ns, gN=gN_n, Xq=Xq, pN_q=pNq,
                wds=wds_cat, F_line=F_line, mass=mass_n,
                force_balance=float(np.linalg.norm(f.sum(axis=0))),
                patch_centroid=patch_centroid)
    return f, Kc, diag


# ==================================================================================================
if __name__ == "__main__":
    # ---- self-test 1: two flat non-matching facets under uniform penetration ----
    # Slave at y=0, master at y=+pen (overlap), both flat -> gap = -pen everywhere, uniform p.
    pen, eps_n = 0.02, 500.0
    slave_xy = np.array([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0]])
    slave_ids = np.array([0, 1, 2])
    master_xy = np.array([[0.0, pen], [0.55, pen], [1.0, pen]])   # NON-MATCHING node spacing
    master_ids = np.array([3, 4, 5])
    n_dof = 2 * 6
    f, Kc, diag = assemble_two_body_contact(slave_xy, slave_ids, master_xy, master_ids,
                                            n_dof, eps_n, order=3)
    pN = eps_n * pen
    # total slave force = p * length ; total master reaction = -that ; net ~ 0
    assert abs(diag["force_balance"]) < 1e-10, diag["force_balance"]
    assert abs(diag["F_line"] - pN * 1.0) < 1e-10, diag["F_line"]
    fy_slave = f[[0, 1, 2], 1].sum()
    fy_master = f[[3, 4, 5], 1].sum()
    assert abs(fy_slave - pN) < 1e-10, fy_slave
    assert abs(fy_master + pN) < 1e-10, fy_master
    Kd = Kc.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12), "two-body Kc must be symmetric (frictionless)"
    assert np.linalg.eigvalsh(Kd).min() > -1e-9, "two-body Kc must be SPSD"
    print("  two_body self-test 1 OK: force balance %.1e, F_line exact, symmetric SPSD 4-block"
          % diag["force_balance"])

    # ---- self-test 2: tangent == finite-difference Jacobian of the residual (the hard gate) ----
    # The 4-block tangent is the EXACT Jacobian of the contact residual under the documented
    # small-rotation approximation (drop d(n) and d(chi), exactly as the one-body assemble_contact).
    # To check the gate honestly we FD-differentiate the residual that the Newton loop actually
    # uses: the normal n_hat and the host-segment weights (N_K o chi) are FROZEN at the linearization
    # point (recomputed each Newton iteration, frozen WITHIN the linear solve), so dt = eps_n n(x)(-dg).
    from .quadrature import gauss_legendre_1d as _gl
    rng = np.random.default_rng(1)
    sx = np.array([[0.0, 0.0], [0.37, 0.0], [0.71, 0.0], [1.0, 0.0]])
    sid = np.array([0, 1, 2, 3])
    mx = np.array([[0.0, 0.015], [0.42, 0.015], [0.83, 0.015], [1.0, 0.015]])
    mid = np.array([4, 5, 6, 7])
    N = 8
    ndof = 2 * N
    u0 = np.zeros(ndof)
    base = np.vstack([sx, mx])                                  # (8,2) reference node positions

    # freeze normal + correspondence at u=0 (the linearization point)
    order = 3
    xi_g, w_g = _gl(order); ss = 0.5 * (1 + xi_g); Nref = np.stack([1 - ss, ss], 1)
    sprof = _profile(sx); sec = np.sqrt(1 + sprof["hp"] ** 2)
    ns0 = np.column_stack([-sprof["hp"] / sec, 1 / sec])
    coup = MonotoneCoupling1D(sprof, _profile(mx))
    frozen = []
    for k in range(3):
        for q in range(order):
            Nq = Nref[q]; P0, P1 = sx[k], sx[k + 1]; L = np.linalg.norm(P1 - P0)
            wq = w_g[q] * 0.5 * L
            nq = Nq[0] * ns0[k] + Nq[1] * ns0[k + 1]; nq = nq / np.linalg.norm(nq)
            xs0 = Nq[0] * P0 + Nq[1] * P1
            xm_q, _, _ = coup.map(np.array([xs0[0]]))
            jm, Nm0, Nm1 = _locate_master(mx[:, 0], float(xm_q[0]))
            frozen.append((k, q, wq, nq, Nq, jm, Nm0, Nm1))

    def resid_frozen(u):
        cur = base + u.reshape(N, 2); f = np.zeros((N, 2)); tr = TractionField(eps_n)
        for (k, q, wq, nhat, Nq, jm, Nm0, Nm1) in frozen:
            xs = Nq[0] * cur[k] + Nq[1] * cur[k + 1]
            xm = Nm0 * cur[4 + jm] + Nm1 * cur[4 + jm + 1]
            gN = (xs - xm) @ nhat
            t = tr.evaluate(np.array([gN]), nhat[None, :])["t"][0]
            f[k] += wq * Nq[0] * t; f[k + 1] += wq * Nq[1] * t
            f[4 + jm] += -wq * Nm0 * t; f[4 + jm + 1] += -wq * Nm1 * t
        return -f.reshape(-1)                                  # R = -f_c (bulk/ext are u-linear/const)

    _, Kc0, _ = assemble_two_body_contact(base[:4], sid, base[4:], mid, ndof, eps_n, order=3)
    K_ana = Kc0.toarray()                                       # dR/du = -df_c/du = +Kc
    h = 1e-7
    K_fd = np.zeros((ndof, ndof))
    for j in range(ndof):
        up = u0.copy(); up[j] += h
        um = u0.copy(); um[j] -= h
        K_fd[:, j] = (resid_frozen(up) - resid_frozen(um)) / (2 * h)
    scale = max(np.abs(K_ana).max(), np.abs(K_fd).max(), 1e-30)
    rel = np.abs(K_ana - K_fd).max() / scale
    print("  two_body self-test 2: max|K_ana - K_fd| / scale = %.3e" % rel)
    assert rel < 1e-6, "tangent does NOT match the finite-difference Jacobian (rel=%.2e)" % rel
    print("  two_body self-test 2 OK: 4-block tangent == df/du (FD, frozen-geom) to %.1e" % rel)

    # ---- self-test 3: CLOSEST-POINT correspondence — same guarantees as the monotone path ----
    #   (a) exact force balance (Newton's 3rd law) ;
    #   (b) symmetric SPSD 4-block tangent ;
    #   (c) tangent == finite-difference df/du under FROZEN geometry (normal + host weights frozen at
    #       the linearization point, exactly as the Newton loop uses them).
    from .coupling import ClosestPointCoupling1D as _CPC
    pen3, eps3 = 0.02, 500.0
    sxy3 = np.array([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0]])
    sid3 = np.array([0, 1, 2])
    mxy3 = np.array([[0.0, pen3], [0.55, pen3], [1.0, pen3]])      # NON-MATCHING node spacing
    mid3 = np.array([3, 4, 5])
    ndof3 = 2 * 6
    f3, Kc3, diag3 = assemble_two_body_contact(sxy3, sid3, mxy3, mid3, ndof3, eps3, order=3,
                                               correspondence="closest_point")
    pN3 = eps3 * pen3
    assert abs(diag3["force_balance"]) < 1e-10, diag3["force_balance"]
    assert abs(diag3["F_line"] - pN3 * 1.0) < 1e-10, diag3["F_line"]
    assert abs(f3[[0, 1, 2], 1].sum() - pN3) < 1e-10, f3[[0, 1, 2], 1].sum()
    assert abs(f3[[3, 4, 5], 1].sum() + pN3) < 1e-10, f3[[3, 4, 5], 1].sum()
    Kd3 = Kc3.toarray()
    assert np.allclose(Kd3, Kd3.T, atol=1e-12), "closest-point Kc must be symmetric"
    assert np.linalg.eigvalsh(Kd3).min() > -1e-9, "closest-point Kc must be SPSD"
    print("  two_body self-test 3 OK: closest-point force balance %.1e, F_line exact, symmetric SPSD"
          % diag3["force_balance"])

    # FD-tangent gate for the closest-point path (curved master so the foot is non-trivially placed).
    sx4 = np.array([[0.0, 0.0], [0.37, 0.0], [0.71, 0.0], [1.0, 0.0]])
    sid4 = np.array([0, 1, 2, 3])
    mx4 = np.column_stack([np.array([0.0, 0.42, 0.83, 1.0]),
                           0.015 + 0.01 * np.array([0.0, 0.42, 0.83, 1.0]) ** 2])   # curved master
    mid4 = np.array([4, 5, 6, 7])
    N4 = 8; ndof4 = 2 * N4; u04 = np.zeros(ndof4); base4 = np.vstack([sx4, mx4])
    sprof4 = _profile(sx4); sec4 = np.sqrt(1 + sprof4["hp"] ** 2)
    ns4 = np.column_stack([-sprof4["hp"] / sec4, 1 / sec4])
    cpc4 = _CPC(sprof4, _profile(mx4))
    frozen4 = []
    for k in range(3):
        for q in range(order):
            Nq = Nref[q]; P0, P1 = sx4[k], sx4[k + 1]; L = np.linalg.norm(P1 - P0)
            wq = w_g[q] * 0.5 * L
            nq = Nq[0] * ns4[k] + Nq[1] * ns4[k + 1]; nq = nq / np.linalg.norm(nq)
            xs0 = Nq[0] * P0 + Nq[1] * P1
            _Xm, seg, Nm0, Nm1, _m = cpc4.map_full(np.array([xs0[0]]))
            frozen4.append((k, q, wq, nq, Nq, int(seg[0]), float(Nm0[0]), float(Nm1[0])))

    def resid_frozen4(u):
        cur = base4 + u.reshape(N4, 2); f = np.zeros((N4, 2)); tr = TractionField(eps_n)
        for (k, q, wq, nhat, Nq, jm, Nm0, Nm1) in frozen4:
            xs = Nq[0] * cur[k] + Nq[1] * cur[k + 1]
            xm = Nm0 * cur[4 + jm] + Nm1 * cur[4 + jm + 1]
            gN = (xs - xm) @ nhat
            t = tr.evaluate(np.array([gN]), nhat[None, :])["t"][0]
            f[k] += wq * Nq[0] * t; f[k + 1] += wq * Nq[1] * t
            f[4 + jm] += -wq * Nm0 * t; f[4 + jm + 1] += -wq * Nm1 * t
        return -f.reshape(-1)

    _, Kc4, _ = assemble_two_body_contact(base4[:4], sid4, base4[4:], mid4, ndof4, eps_n, order=3,
                                          correspondence="closest_point")
    K_ana4 = Kc4.toarray(); K_fd4 = np.zeros((ndof4, ndof4))
    for j in range(ndof4):
        up = u04.copy(); up[j] += h
        um = u04.copy(); um[j] -= h
        K_fd4[:, j] = (resid_frozen4(up) - resid_frozen4(um)) / (2 * h)
    scale4 = max(np.abs(K_ana4).max(), np.abs(K_fd4).max(), 1e-30)
    rel4 = np.abs(K_ana4 - K_fd4).max() / scale4
    print("  two_body self-test 3: closest-point max|K_ana - K_fd| / scale = %.3e" % rel4)
    assert rel4 < 1e-6, "closest-point tangent != FD Jacobian (rel=%.2e)" % rel4
    print("  two_body self-test 3 OK: closest-point 4-block tangent == df/du (FD, frozen) to %.1e"
          % rel4)
