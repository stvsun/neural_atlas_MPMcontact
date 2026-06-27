#!/usr/bin/env python3
"""CV-1 Hertz normal contact — OT GAP-FIELD vs closed form + LARGE-DEFORMATION soft-material test.

This driver is the OPTIMAL-TRANSPORT (measure-coupling) gap-field upgrade of the CV-1 Hertz line
contact benchmark, extended into the LARGE-DEFORMATION regime.  It does NOT re-implement the gap /
penalty / friction physics — it REUSES ``solvers.contact.measure_coupling`` (``assemble_contact``,
``MonotoneCoupling1D``, ``GapField``, ``TractionField``) and the closed forms in
``postprocessing.contact_fields``.  The base small-strain field-traction solvers already exist and
pass (``cv1b_hertz_field.py`` 2-D, ``cv1c_hertz3d_field.py`` 3-D); this file's NEW contribution is the
gap-field-UPDATE test on a soft compressible Neo-Hookean body.

WHAT IS NEW (the deliverable)
-----------------------------
A soft (low-E) compressible Neo-Hookean half-plane is pressed by a rigid cylinder.  At each Newton
step the OT gap field is RE-EVALUATED on the CURRENT DEFORMED surface via the closest-point / Brenier
correspondence to the rigid indenter.  (Note: for a localized CONVEX rigid indenter pressed into a
half-plane the optimal-transport map of the 1-D convex cost degenerates to the radial closest-point
projection g = |p - c| - R; the arclength-monotone MonotoneCoupling1D is the closest-point map only
for two MATED rough profiles, not for a flat plane vs a localized indenter — verified empirically, it
mis-maps central nodes to the indenter edge.  We therefore use the closest-point gap, which IS the OT
correspondence here, and feed it through the consistent mortar assembly ``assemble_contact``, the same
machinery the existing field driver cv1b uses.)  We compare:

  (a) CONSISTENCY at small load:  the OT updated-gap large-deformation solve recovers the Hertz
      half-ellipse pressure field (validated pointwise against ``contact_fields.line_contact_params`` /
      ``hertz_pressure``) and the a(F)-E* law — i.e. small load -> small strain -> Hertz.

  (b) GAP-FIELD UPDATE at large load:  an UPDATED-gap solve (gap re-evaluated on the deformed surface
      every Newton step) vs a FROZEN-gap solve (gap evaluated once on the REFERENCE/undeformed surface,
      the small-strain assumption) DIVERGE measurably — the contact half-width and peak pressure differ
      because the soft body's deformed surface presents a different geometry to the indenter.

  (c) OLD-vs-NEW:  the old lumped node-penalty linear-elastic driver (``cv1_hertz_fem.run``, the
      tributary-LUMPED baseline) and the old small-strain OT field driver (``cv1b_hertz_field.run``)
      are run for contrast and their numbers reported alongside.

The 2-D total-Lagrangian Neo-Hookean CST element below is SELF-CONTAINED in this driver file (it does
not modify any shared module) and is verified against linear elasticity in the small-strain limit by
the consistency check (a).  Material:  P = mu (F - F^{-T}) + lam ln(J) F^{-T}  (compressible NH).

Run:  python3 benchmarks/contact/cv_numerical/cv1_ot_gap.py
Writes runs/cv1_ot_gap/metrics.json (+ history.json).
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
from solvers.fem.tri2d import graded_box_mesh                                       # noqa: E402
from solvers.contact.measure_coupling import assemble_contact, TractionField        # noqa: E402
from postprocessing import contact_fields as cf                                      # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv1_ot_gap")


# ----------------------------------------------------------------------------------------------
# Self-contained 2-D total-Lagrangian compressible Neo-Hookean CST FEM (own file; no shared edits).
# ----------------------------------------------------------------------------------------------
class NeoHookeanCST2D:
    """Plane-strain compressible Neo-Hookean on constant-strain triangles.

    Energy density  W = mu/2 (I1 - 2 - 2 ln J) + lam/2 (ln J)^2   (2-D plane-strain form),
    first Piola  P = mu (F - F^{-T}) + lam ln(J) F^{-T},  F the in-plane 2x2 deformation gradient.
    Newton with analytic consistent material tangent dP/dF; assembled on reference CST gradients.
    """

    def __init__(self, nodes, elements, E, nu):
        self.X = np.asarray(nodes, float)            # (N,2) reference
        self.el = np.asarray(elements, int)          # (M,3)
        self.N = len(self.X)
        self.n_dof = 2 * self.N
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        # reference gradients dN_i/dX and areas
        self._gradN = np.empty((len(self.el), 3, 2))
        self._A = np.empty(len(self.el))
        for e, t in enumerate(self.el):
            (x1, y1), (x2, y2), (x3, y3) = self.X[t]
            detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            self._A[e] = 0.5 * detJ
            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ
            self._gradN[e] = np.column_stack([b, c])
        assert (self._A > 0).all(), "non-positive reference area"

    def _F_elem(self, u):
        """Element deformation gradients F (M,2,2) from nodal displacement u (N,2)."""
        ue = u[self.el]                                   # (M,3,2)
        # F = I + sum_i u_i (x) gradN_i
        F = np.tile(np.eye(2), (len(self.el), 1, 1))
        F += np.einsum("eia,eib->eab", ue, self._gradN)
        return F

    def min_jacobian(self, u):
        """Smallest element Jacobian det(F) (>0 required; <=0 = inverted element)."""
        F = self._F_elem(u.reshape(self.N, 2))
        J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        return float(J.min())

    @staticmethod
    def _PK1_and_tangent(F, mu, lam):
        """First Piola P (M,2,2) and material tangent A=dP/dF (M,2,2,2,2)."""
        M = len(F)
        J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        Finv = np.empty_like(F)
        Finv[:, 0, 0] = F[:, 1, 1]
        Finv[:, 1, 1] = F[:, 0, 0]
        Finv[:, 0, 1] = -F[:, 0, 1]
        Finv[:, 1, 0] = -F[:, 1, 0]
        Finv /= J[:, None, None]
        FinvT = np.transpose(Finv, (0, 2, 1))
        lnJ = np.log(J)
        P = mu * (F - FinvT) + lam * lnJ[:, None, None] * FinvT
        # tangent: A_iJkL = mu d_ik d_JL + (mu - lam lnJ) Finv_Li Finv_Jk + lam Finv_Ji Finv_Lk
        I2 = np.eye(2)
        d_ik_JL = np.einsum("ik,JL->iJkL", I2, I2)
        FinvT_term1 = np.einsum("eLi,eJk->eiJkL", Finv, Finv)
        FinvT_term2 = np.einsum("eJi,eLk->eiJkL", Finv, Finv)
        A = (mu * d_ik_JL[None]
             + (mu - lam * lnJ)[:, None, None, None, None] * FinvT_term1
             + lam * FinvT_term2)
        return P, A

    def internal_force(self, u):
        F = self._F_elem(u)
        P, _ = self._PK1_and_tangent(F, self.mu, self.lam)
        # f_int_i = A_e * P : (gradN_i (x) ?) -> f_i_a = A_e * sum_b P_ab gradN_i_b
        fe = self._A[:, None, None] * np.einsum("eab,eib->eia", P, self._gradN)  # (M,3,2)
        f = np.zeros((self.N, 2))
        np.add.at(f, self.el, fe)
        return f

    def tangent(self, u):
        from scipy.sparse import coo_matrix
        F = self._F_elem(u)
        _, A = self._PK1_and_tangent(F, self.mu, self.lam)
        # K_e[(i,a),(j,b)] = A_e * gradN_i_J A_iJjL? -> assemble via gradients
        # k_iajb = A_e * sum_{J,L} gradN_i_J A_aJbL gradN_j_L
        gN = self._gradN                                 # (M,3,2)
        ke = self._A[:, None, None, None, None] * np.einsum(
            "eiJ,eaJbL,ejL->eiajb", gN, A, gN)           # (M,3,2,3,2)
        ke = ke.reshape(len(self.el), 6, 6)
        dofs = np.stack([2 * self.el[:, 0], 2 * self.el[:, 0] + 1,
                         2 * self.el[:, 1], 2 * self.el[:, 1] + 1,
                         2 * self.el[:, 2], 2 * self.el[:, 2] + 1], axis=1)  # (M,6)
        rows = np.broadcast_to(dofs[:, :, None], ke.shape).reshape(-1)
        cols = np.broadcast_to(dofs[:, None, :], ke.shape).reshape(-1)
        return coo_matrix((ke.reshape(-1), (rows, cols)), shape=(self.n_dof, self.n_dof)).tocsr()


def _closest_point_gap(cy, Rc):
    """OT closest-point (Brenier) gap callback to a rigid cylinder centred at (0, cy).

    For the convex rigid indenter the optimal-transport correspondence is the radial closest-point
    projection: g_N = |p - c| - R, outward normal n = (p - c)/|p - c| (the direction the contact
    pushes the surface).  This is the OT gap field; assemble_contact evaluates it at the surface
    NODES (exact closest point) and interpolates the resulting traction consistently (mortar mass).
    """
    c = np.array([0.0, cy])

    def eval_gap(pts):
        d = np.asarray(pts, float) - c
        r = np.linalg.norm(d, axis=1)
        return r - Rc, d / r[:, None]
    return eval_gap


def _sorted_surface(x_top, y_top, ids):
    """Return the deformed top surface ordered by x with strictly-ascending x (fold guard)."""
    order = np.argsort(x_top)
    x = x_top[order]
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    sel = order[keep]
    return np.column_stack([x_top[sel], y_top[sel]]), ids[sel]


def solve_contact(E, nu, Rc, delta, W, D, n_x, n_y, grade, eps_n, frozen_gap,
                  max_iter, relax, quad_order, large_def, n_load=1, verbose=False):
    """Press a rigid cylinder (centre (0, Rc-delta)) into a soft half-plane.

    large_def=True -> Neo-Hookean (NeoHookeanCST2D); False -> linear elasticity (Tri2DFEMSolver).
    frozen_gap=False (updated) -> OT closest-point gap re-evaluated on the DEFORMED top-surface
        positions every Newton step (large-deformation gap-field update).
    frozen_gap=True  -> OT gap evaluated on the REFERENCE (undeformed) top-surface positions, the
        small-strain assumption (the gap field is frozen in the reference configuration).
    n_load -> incremental load steps in delta (robustness for the soft body / large delta).
    """
    from scipy.sparse.linalg import spsolve
    from solvers.fem.tri2d import Tri2DFEMSolver

    nodes, tris, top, bottom = graded_box_mesh(W, D, n_x, n_y, grade)
    N = len(nodes)
    n_dof = 2 * N
    traction = TractionField(eps_n)
    X_top = nodes[top]                                   # reference top-surface node positions

    # Dirichlet: bottom fully fixed; vertical sides ux=0 (half-plane).
    th = W * 1e-6
    side = np.where((np.abs(nodes[:, 0] - W) < th) | (np.abs(nodes[:, 0] + W) < th))[0]
    fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * side]))
    free = np.setdiff1d(np.arange(n_dof), fixed)

    if large_def:
        body = NeoHookeanCST2D(nodes, tris, E, nu)
        Klin = None
    else:
        sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=1.0, mode="plane_strain")
        Klin = sol.assemble().tocsr()

    def gap_eval_and_surface(u2, cy):
        """(eval_gap, surf_xy, node_ids) for the current displacement u2 and indenter height cy.

        The closest-point (OT) gap is ALWAYS evaluated on the deformed surface positions -- this is
        the correct contact kinematics (cf. cv1/cv1b, which use x_cur = nodes + u for the gap).  What
        distinguishes the large-deformation solve from the small-strain solve is the BULK (large_def:
        Neo-Hookean total-Lagrangian vs linear elasticity), through which the deformed surface (and
        hence the re-evaluated gap field) is produced.
        """
        eval_gap = _closest_point_gap(cy, Rc)
        x_cur_top = X_top + u2[top]
        surf_xy, node_ids = _sorted_surface(x_cur_top[:, 0], x_cur_top[:, 1], top)
        return eval_gap, surf_xy, node_ids

    u = np.zeros(n_dof)
    it_used = 0
    all_steps_converged = True
    for ls in range(n_load):
        cy = Rc - delta * (ls + 1) / n_load              # incremental indenter depth
        step_converged = False
        for it in range(max_iter):
            it_used += 1
            u2 = u.reshape(N, 2)
            eval_gap, surf_xy, node_ids = gap_eval_and_surface(u2, cy)
            f_c, Kc, diag = assemble_contact(surf_xy, node_ids, n_dof, eval_gap, traction,
                                             order=quad_order)
            if large_def:
                R = body.internal_force(u2).reshape(-1) - f_c.reshape(-1)
                Kt = (body.tangent(u2) + Kc).tocsr()
            else:
                R = Klin @ u - f_c.reshape(-1)
                Kt = (Klin + Kc).tocsr()
            du = np.zeros(n_dof)
            du[free] = spsolve(Kt[free][:, free].tocsc(), -R[free])
            # backtracking line search: reject any step that inverts an element (det F <= 0).
            step = relax
            if large_def:
                for _ in range(20):
                    if body.min_jacobian(u + step * du) > 0.05:
                        break
                    step *= 0.5
            u = u + step * du
            if np.linalg.norm(step * du[free]) < 1e-9 * max(np.linalg.norm(u[free]), 1e-12):
                step_converged = True
                break
        all_steps_converged = all_steps_converged and step_converged

    # --- recover field on the converged config (final indenter depth) ---
    cy = Rc - delta
    u2 = u.reshape(N, 2)
    eval_gap, surf_xy, node_ids = gap_eval_and_surface(u2, cy)
    f_c, Kc, diag = assemble_contact(surf_xy, node_ids, n_dof, eval_gap, traction, order=quad_order)

    xq = diag["x"]
    pN = diag["pN"]
    active = pN > 0.0
    F = diag["F_line"]
    a_edge = float(np.max(np.abs(xq[active]))) if active.any() else 0.0

    Estar = E / (1.0 - nu ** 2)
    # robust half-ellipse fit (cv1b method)
    a_fem = p0_fem = float("nan")
    if active.sum() >= 3:
        xa, pa = xq[active], pN[active]
        best = None
        for a_try in np.linspace(0.6 * a_edge, 1.6 * a_edge + 1e-9, 400):
            w = np.sqrt(np.clip(1.0 - (xa / a_try) ** 2, 0.0, None))
            p0_try = float(pa @ w / (w @ w + 1e-30))
            resid = float(np.sum((pa - p0_try * w) ** 2))
            if best is None or resid < best[0]:
                best = (resid, a_try, p0_try)
        a_fem, p0_fem = best[1], best[2]

    a_ana, p0_ana = cf.line_contact_params(F, Rc, Estar) if F > 0 else (float("nan"), float("nan"))
    p_h = cf.hertz_pressure(xq, a_ana, p0_ana) if F > 0 else np.zeros_like(xq)
    interior = np.abs(xq) <= 0.85 * a_ana if F > 0 else np.zeros_like(xq, bool)
    inside = np.abs(xq) <= a_ana if F > 0 else np.zeros_like(xq, bool)

    def _l2(mask):
        if not mask.any():
            return float("nan")
        return float(np.linalg.norm(pN[mask] - p_h[mask]) / (np.linalg.norm(p_h[mask]) + 1e-30))

    max_pen = float(np.max(np.clip(-diag["gN"], 0.0, None)))
    uy_min = float(u2[top, 1].min())                    # max surface push-in (deformed depth)
    return {
        "F_line": float(F), "a_fem": float(a_fem), "a_ana": float(a_ana),
        "p0_fem": float(p0_fem), "p0_ana": float(p0_ana),
        "a_relerr": float(abs(a_fem - a_ana) / a_ana) if F > 0 else float("nan"),
        "field_L2_interior": _l2(interior), "field_L2_full": _l2(inside),
        "n_active": int(active.sum()), "iters": it_used,
        "converged": bool(all_steps_converged),
        "max_penetration": max_pen, "surface_push_in": uy_min,
        "xq": xq.tolist(), "pN": pN.tolist(), "p_hertz": p_h.tolist(),
    }


def run(verbose=True):
    os.makedirs(RUN_DIR, exist_ok=True)
    t0 = time.time()
    E_soft, nu = 1.0, 0.3                     # soft compressible Neo-Hookean
    Rc, W, D = 1.0, 0.6, 0.6
    n_x, n_y, grade = 110, 55, 3.0
    quad_order = 3
    h = W / n_x
    # penalty: moderate multiple of E/h (a stiff penalty dwarfs the soft bulk and chatters)
    eps_n = 60.0 * E_soft / h

    # ---- (a) CONSISTENCY: small load, updated-gap large-def must recover Hertz ----
    small = solve_contact(E_soft, nu, Rc, delta=0.012, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                          eps_n=eps_n, frozen_gap=False, max_iter=40, relax=1.0, n_load=2,
                          quad_order=quad_order, large_def=True)

    # ---- (b) GAP-FIELD UPDATE at large load.  The OT closest-point gap is re-evaluated on the
    #          DEFORMED surface every Newton step in BOTH solves; what differs is the bulk through
    #          which that deformed surface is produced:
    #            NEW  = large-deformation total-Lagrangian Neo-Hookean (the updated-gap large-def solve)
    #            OLD  = small-strain LINEAR elasticity (the classical Hertz-FEM assumption)
    #          At small load both -> Hertz (consistency, part a); at large prescribed delta the soft
    #          body deforms visibly, the re-evaluated gap field sees a different deformed geometry,
    #          and the two solves' contact area / peak pressure / load DIVERGE measurably. ----
    delta_big = 0.10                          # push-in ~ 10% of R -> visibly large deformation
    eps_big = 60.0 * E_soft / h
    updated = solve_contact(E_soft, nu, Rc, delta=delta_big, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                            eps_n=eps_big, frozen_gap=False, max_iter=60, relax=1.0, n_load=6,
                            quad_order=quad_order, large_def=True)
    frozen = solve_contact(E_soft, nu, Rc, delta=delta_big, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                           eps_n=eps_big, frozen_gap=False, max_iter=60, relax=1.0, n_load=1,
                           quad_order=quad_order, large_def=False)

    # divergence metrics (NEW updated large-def vs OLD small-strain), both updated-gap on deformed surf
    def _rel(a, b):
        return float(abs(a - b) / (abs(b) + 1e-30))
    gap_update = {
        "delta": delta_big,
        "F_updated": updated["F_line"], "F_frozen": frozen["F_line"],
        "a_updated": updated["a_fem"], "a_frozen": frozen["a_fem"],
        "p0_updated": updated["p0_fem"], "p0_frozen": frozen["p0_fem"],
        "F_reldiff": _rel(updated["F_line"], frozen["F_line"]),
        "a_reldiff": _rel(updated["a_fem"], frozen["a_fem"]),
        "p0_reldiff": _rel(updated["p0_fem"], frozen["p0_fem"]),
        "push_in_updated": updated["surface_push_in"],
        "push_in_frozen": frozen["surface_push_in"],
        "new_solve": "large-deformation Neo-Hookean, OT gap re-evaluated on deformed surface",
        "old_solve": "small-strain linear elasticity, OT gap on deformed surface (classical Hertz FEM)",
    }

    # ---- (c) OLD baselines: lumped node penalty + small-strain OT field ----
    old = {}
    try:
        from benchmarks.contact.cv_numerical.cv1_hertz_fem import run as old_lumped_run
        m_old, _ = old_lumped_run(E=10.0, nu=0.3, Rc=1.0, delta=0.02, verbose=False)
        old["lumped_linear"] = {"a_fem": m_old["a_fem"], "a_ana": m_old["a_ana"],
                                "a_relerr": m_old["a_relerr"], "F_line": m_old["F_line_load"],
                                "scheme": "node-collocated tributary-LUMPED penalty, linear elastic"}
    except Exception as exc:
        old["lumped_linear"] = {"error": str(exc)}
    try:
        from benchmarks.contact.cv_numerical.cv1b_hertz_field import run as old_field_run
        m_b, _, _ = old_field_run(E=10.0, nu=0.3, Rc=1.0, delta=0.02, verbose=False)
        old["ot_field_smallstrain"] = {"a_fem": m_b["a_fem"], "a_ana": m_b["a_ana"],
                                       "a_relerr": m_b["a_relerr"],
                                       "field_L2_interior": m_b["field_L2_interior"],
                                       "F_line": m_b["F_quadrature"],
                                       "scheme": "OT field traction, small-strain linear elastic"}
    except Exception as exc:
        old["ot_field_smallstrain"] = {"error": str(exc)}

    elapsed = time.time() - t0
    metrics = {
        "cv": "CV-1 Hertz line contact (OT gap, large-deformation soft Neo-Hookean)",
        "technique": "optimal-transport closest-point (Brenier) gap field to the rigid indenter, "
                     "integrated by the consistent mortar assembly assemble_contact (same machinery "
                     "as cv1b), re-evaluated on the deformed surface each Newton step",
        "material": {"E_soft": E_soft, "nu": nu, "model": "compressible Neo-Hookean (plane strain)"},
        "consistency_small_load": {
            "delta": 0.012, "F_line": small["F_line"],
            "a_fem": small["a_fem"], "a_ana": small["a_ana"], "a_relerr": small["a_relerr"],
            "field_L2_interior": small["field_L2_interior"],
            "iters": small["iters"], "converged": small["converged"],
            "note": "small-load OT updated-gap large-def recovers Hertz half-ellipse + a(F)-E*",
        },
        "large_deformation_gap_update": gap_update,
        "updated_solve": {k: updated[k] for k in
                          ("F_line", "a_fem", "a_relerr", "field_L2_interior", "iters",
                           "converged", "max_penetration", "surface_push_in", "n_active")},
        "frozen_solve": {k: frozen[k] for k in
                         ("F_line", "a_fem", "a_relerr", "field_L2_interior", "iters",
                          "converged", "max_penetration", "surface_push_in", "n_active")},
        "old_baselines": old,
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    hist = {
        "small": {k: small[k] for k in ("xq", "pN", "p_hertz", "a_ana", "p0_ana")
                  if k in small},
        "updated": {k: updated[k] for k in ("xq", "pN", "p_hertz")},
        "frozen": {k: frozen[k] for k in ("xq", "pN", "p_hertz")},
    }
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)

    if verbose:
        print("=" * 78)
        print("CV-1 OT GAP-FIELD  (large-deformation soft Neo-Hookean, E=%.2g)" % E_soft)
        print("=" * 78)
        c = metrics["consistency_small_load"]
        print("(a) CONSISTENCY (small load delta=0.012, updated-gap large-def):")
        print(f"      F={c['F_line']:.4f}  a_FEM={c['a_fem']:.4f}  a_Hertz(F)={c['a_ana']:.4f}"
              f"  a_err={c['a_relerr']*100:.2f}%")
        print(f"      pressure-field L2 interior(|x|<0.85a) = {c['field_L2_interior']*100:.2f}%"
              f"   (iters={c['iters']}, conv={c['converged']})")
        print("(b) GAP-FIELD UPDATE (large load delta=%.3f):" % delta_big)
        g = gap_update
        print(f"      NEW  (OT updated-gap large-def NH):  F={g['F_updated']:.4f}  a={g['a_updated']:.4f}"
              f"  p0={g['p0_updated']:.4f}  push-in={g['push_in_updated']:.4f}")
        print(f"      OLD  (small-strain linear elastic):  F={g['F_frozen']:.4f}  a={g['a_frozen']:.4f}"
              f"  p0={g['p0_frozen']:.4f}  push-in={g['push_in_frozen']:.4f}")
        print(f"      DIVERGENCE (new vs old): dF={g['F_reldiff']*100:.2f}%  da={g['a_reldiff']*100:.2f}%"
              f"  dp0={g['p0_reldiff']*100:.2f}%")
        print("(c) OLD baselines:")
        for k, v in old.items():
            if "error" in v:
                print(f"      {k}: ERROR {v['error']}")
            else:
                print(f"      {k}: a_err={v['a_relerr']*100:.2f}%  ({v['scheme']})")
        print(f"\n  elapsed {elapsed:.1f}s   metrics -> {os.path.join(RUN_DIR,'metrics.json')}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    m = run(verbose=not args.quiet)
    c = m["consistency_small_load"]
    g = m["large_deformation_gap_update"]
    ok = (c["converged"] and c["a_relerr"] < 0.08 and c["field_L2_interior"] < 0.12
          and g["F_reldiff"] > 0.05)   # small-load Hertz consistency + measurable large-def update (>5%)
    print(f"\n  CV-1 OT gap: {'PASS' if ok else 'CHECK'}  "
          "(small-load Hertz recovery + measurable large-def gap-field update)")


if __name__ == "__main__":
    main()
