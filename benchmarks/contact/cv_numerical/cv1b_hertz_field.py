#!/usr/bin/env python3
"""CV-1b Hertz line contact — CONSISTENT FIELD TRACTION (measure-coupling) vs the analytical solution.

The field-traction upgrade of CV-1 (``cv1_hertz_fem.py``).  Same rigid cylinder pressed into the same
elastic half-plane (``Tri2DFEMSolver`` + ``graded_box_mesh``), but the node-collocated, tributary-LUMPED
penalty contact (``cv1_hertz_fem.py:129-144``: diagonal tangent, per-node pressure) is replaced by the
CONSISTENT Galerkin surface integral of a traction field
(``solvers/contact/measure_coupling/assembly.assemble_contact``):

    f_I = sum_q w_q N_I(xi_q) p_N(xi_q) n,    K_c = sum_q w_q N_I N_J eps_n (n (x) n)

so the recovered quantity is the pressure DISTRIBUTION p_N(x) at Gauss points — validated pointwise
against the Hertz half-ellipse ``contact_fields.hertz_pressure`` — not just the scalar half-width a.

What IS verified (honest scope):
  * CONTACT PATCH TEST: a uniform pressure is transmitted EXACTLY across a non-uniform surface
    discretization (total force to ~1e-12; pressure uniform to ~1e-12).  The consistent assembly
    integrates the field; the lumped node penalty produces a node-spacing sawtooth here.
  * The recovered pressure FIELD matches the Hertz half-ellipse pointwise in the contact INTERIOR
    (|x| < 0.85 a) to a few percent.  This is the field check the scalar a(F) misses.
  * The consistent quadrature integrates the field to the line load F WITHOUT the ~3.5% tributary
    over-count cv1 had to patch with trapz (``cv1_hertz_fem.py:162-167``).
  * The a(F)-E* law (a = 2 sqrt(F R / pi E*)) — the one genuinely independent physics check, inherited.

What is NOT verified:
  * The contact EDGE (|x| -> a): the half-ellipse has an infinite slope there; any penalty FEM rounds
    it over ~1 element, so the full-interval L2 is edge-dominated (interior L2 is the tight metric).
  * p0 = 2F/pi a is the half-ellipse load identity, not independent (carried from cv1).
  * The rigid analytic cylinder exercises the SLAVE-side coupling/integration only; the master-side
    interpolation half of a true mortar coupling is exercised by the two-chart rock-joint case (M6).

Run:  python3 benchmarks/contact/cv_numerical/cv1b_hertz_field.py [--patch] [--sweep]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, graded_box_mesh                       # noqa: E402
from solvers.contact.measure_coupling import assemble_contact, TractionField         # noqa: E402
from postprocessing import contact_fields as cf                                      # noqa: E402
from benchmarks.contact.cv_numerical.cv1_hertz_fem import _analytic_cylinder, _neural_cylinder  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv1b_hertz_field")


def run(E=10.0, nu=0.3, Rc=1.0, delta=0.02, W=0.6, D=0.6, n_x=140, n_y=70, grade=3.0,
        eps_n=None, max_iter=60, relax=1.0, quad_order=3, indenter="analytic", verbose=True):
    nodes, tris, top, bottom = graded_box_mesh(W, D, n_x, n_y, grade)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=1.0, mode="plane_strain")
    Kcsr = sol.assemble().tocsr()
    Estar = E / (1.0 - nu ** 2)
    if eps_n is None:
        eps_n = (500.0 if indenter == "analytic" else 80.0) * E / (W / n_x)
    if indenter == "neural":
        from atlas.sdf.train_analytical_sdf import load_trained_sdf
        disc = load_trained_sdf("disc")
        if disc is None:
            raise RuntimeError("no neural disc SDF — run atlas/sdf/train_analytical_sdf.py --shape disc")
        gap_normal = _neural_cylinder([0.0, Rc - delta], Rc, disc)
        relax = min(relax, 0.5)
    else:
        gap_normal = _analytic_cylinder([0.0, Rc - delta], Rc)

    th = W * 1e-6
    side = np.where((np.abs(nodes[:, 0] - W) < th) | (np.abs(nodes[:, 0] + W) < th))[0]
    fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * side]))
    free = np.setdiff1d(np.arange(sol.n_dof), fixed)

    from scipy.sparse.linalg import spsolve

    traction = TractionField(eps_n)
    u = np.zeros(sol.n_dof)
    for it in range(max_iter):
        x_cur = nodes + u.reshape(sol.n_nodes, 2)
        f, Kc, _ = assemble_contact(x_cur[top], top, sol.n_dof, gap_normal, traction, order=quad_order)
        R = Kcsr @ u - f.reshape(-1)
        J = (Kcsr + Kc).tocsr()
        du = np.zeros(sol.n_dof)
        du[free] = spsolve(J[free][:, free].tocsc(), -R[free])
        u = u + relax * du
        if np.linalg.norm(du[free]) < 1e-8 * max(np.linalg.norm(u[free]), 1e-12):
            break
    u2 = u.reshape(sol.n_nodes, 2)

    # --- recover the traction FIELD (nodal pressures, the conforming field) ---
    x_cur = nodes + u2
    f, Kc, diag = assemble_contact(x_cur[top], top, sol.n_dof, gap_normal, traction, order=quad_order)
    xq = diag["x"]                                             # nodal surface coordinate
    pN = diag["pN"]                                            # nodal contact pressure
    active = pN > 0.0
    F = diag["F_line"]                                         # int p ds (consistent, no tributary over-count)
    a_edge = float(np.max(np.abs(xq[active]))) if active.any() else 0.0

    # robust half-ellipse fit p(x)=p0 sqrt(1-(x/a)^2) to the Gauss-point pressures (cv1 method)
    xa, pa = xq[active], pN[active]
    best = None
    for a_try in np.linspace(0.6 * a_edge, 1.5 * a_edge + 1e-9, 400):
        w = np.sqrt(np.clip(1.0 - (xa / a_try) ** 2, 0.0, None))
        p0_try = float(pa @ w / (w @ w + 1e-30))
        resid = float(np.sum((pa - p0_try * w) ** 2))
        if best is None or resid < best[0]:
            best = (resid, a_try, p0_try)
    a_fem, p0_fem = best[1], best[2]
    a_ana, p0_ana = cf.line_contact_params(F, Rc, Estar)

    # pointwise pressure-FIELD L2 vs the Hertz half-ellipse (the new, stronger check)
    p_h = cf.hertz_pressure(xq, a_ana, p0_ana)                  # p0 sqrt(1-(x/a)^2), 0 outside
    inside = np.abs(xq) <= a_ana
    interior = np.abs(xq) <= 0.85 * a_ana
    def _l2(mask):
        if not mask.any():
            return float("nan")
        return float(np.linalg.norm(pN[mask] - p_h[mask]) / (np.linalg.norm(p_h[mask]) + 1e-30))
    L2_full, L2_int = _l2(inside), _l2(interior)

    m = {
        "indenter": indenter, "E": E, "nu": nu, "Rc": Rc, "delta": delta, "Estar": Estar,
        "eps_n": eps_n, "quad_order": quad_order, "iters": it + 1,
        "converged": bool(it + 1 < max_iter),
        "n_gauss": int(len(xq)), "n_active_gauss": int(active.sum()),
        "F_quadrature": F, "a_fem": a_fem, "a_ana": float(a_ana),
        "p0_fem": p0_fem, "p0_ana": float(p0_ana), "p0_independent": False,
        "a_relerr": float(abs(a_fem - a_ana) / a_ana),
        "field_L2_full": L2_full, "field_L2_interior": L2_int,
        "n_nodes": int(sol.n_nodes), "n_elements": int(len(tris)),
    }
    # arrays for the plot script
    hist = {"x_q": xq.tolist(), "pN_q": pN.tolist(),
            "p_hertz_q": p_h.tolist(), "a_ana": float(a_ana), "p0_ana": float(p0_ana)}
    if verbose:
        print(f"  CV-1b Hertz FIELD traction  [{indenter}]  ({m['n_nodes']} nodes, {m['iters']} iters"
              f"{'' if m['converged'] else ' (HIT max_iter!)'}, {m['n_active_gauss']} active Gauss pts)")
        print(f"    line load F (consistent quadrature) = {F:.4f}")
        print(f"    half-width a: FEM={a_fem:.4f}  Hertz(F)={a_ana:.4f}  err={m['a_relerr']*100:.2f}%"
              f"   <- a(F)-E* independent check")
        print(f"    pressure-FIELD L2: interior(|x|<0.85a)={L2_int*100:.2f}%   full(|x|<a)={L2_full*100:.2f}%"
              f"   <- the field check")
    return m, hist, (sol, u2, top, diag)


def patch_test(eps_n=1.0e4, delta=0.01, W=1.0, n=21, irregular=True, quad_order=3, verbose=True):
    """Contact patch test: a rigid FLAT platen presses a flat surface with NON-UNIFORM node spacing.
    The recovered pressure must be uniform and the total force exact, independent of node spacing —
    the marginal/partition-of-unity consistency the lumped node penalty lacks."""
    if irregular:
        s = np.linspace(-1.0, 1.0, n)
        xs = W * np.sign(s) * np.abs(s) ** 1.7                  # deliberately non-uniform
    else:
        xs = np.linspace(-W, W, n)
    surf = np.column_stack([xs, np.zeros(n)])                   # flat top at y=0
    node_ids = np.arange(n)
    y_f = -delta                                                # platen face pressed delta below

    def gap_flat(pts):
        g = y_f - pts[:, 1]                                     # <0 when the surface pokes up into platen
        nn = np.tile([0.0, -1.0], (len(pts), 1))               # eject downward, out of the platen
        return g, nn

    f, Kc, diag = assemble_contact(surf, node_ids, 2 * n, gap_flat, TractionField(eps_n), order=quad_order)
    pN = diag["pN"]
    pN_uniform = eps_n * delta
    L = float(xs[-1] - xs[0])
    res = {
        "uniformity": float((pN.max() - pN.min()) / pN.mean()),
        "F_quadrature": float(diag["F_line"]),
        "F_exact": float(pN_uniform * L),
        "force_relerr": float(abs(diag["F_line"] - pN_uniform * L) / (pN_uniform * L)),
        "total_force_abserr": float(abs(f.sum(axis=0)[1] - (-pN_uniform * L))),
        "n_nodes": int(n), "irregular": bool(irregular),
    }
    res["PASS"] = bool(res["uniformity"] < 1e-3 and res["force_relerr"] < 1e-10
                       and res["total_force_abserr"] < 1e-9)
    if verbose:
        print(f"  CONTACT PATCH TEST ({'non-uniform' if irregular else 'uniform'} nodes, n={n}):")
        print(f"    pressure uniformity (max-min)/mean = {res['uniformity']:.2e}")
        print(f"    line load: quadrature={res['F_quadrature']:.10f}  exact={res['F_exact']:.10f}"
              f"   rel.err={res['force_relerr']:.2e}")
        print(f"    total nodal force error = {res['total_force_abserr']:.2e}   -> "
              f"{'PASS' if res['PASS'] else 'FAIL'}")
    return res


def estar_sweep(E_list=(5.0, 10.0, 20.0, 40.0), nu=0.3, Rc=1.0, delta=0.02):
    rows = []
    for E in E_list:
        m, _, _ = run(E=E, nu=nu, Rc=Rc, delta=delta, verbose=False)
        Estar = E / (1.0 - nu ** 2)
        ratio = (m["a_fem"] / np.sqrt(m["F_quadrature"])) / (2.0 * np.sqrt(Rc / (np.pi * Estar)))
        rows.append((E, Estar, m["F_quadrature"], m["a_fem"], ratio))
    ratios = np.array([r[4] for r in rows])
    return rows, float(ratios.mean()), float(ratios.std() / ratios.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true", help="use the trained neural disc SDF as indenter")
    ap.add_argument("--patch", action="store_true", help="run the contact patch test")
    ap.add_argument("--sweep", action="store_true", help="run the E* sweep (a(F)-E* physics)")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)

    if args.patch:
        patch_test()
        print()
    m, hist, _ = run(indenter="neural" if args.neural else "analytic")
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)
    ok = (m["field_L2_interior"] < 0.06 and m["a_relerr"] < 0.05 and m["converged"])
    print(f"\n  CV-1b field-traction: {'PASS' if ok else 'CHECK'} "
          f"(interior field L2 < 6%, a(F) < 5%)")
    if args.sweep:
        rows, rmean, rcov = estar_sweep()
        print("\n  E* sweep (a/sqrt(F) / 2sqrt(R/piE*) should be CONSTANT):")
        for E, Es, F, a, ratio in rows:
            print(f"    E={E:5.1f} (E*={Es:7.2f}): F={F:.4f} a={a:.4f}  ratio={ratio:.4f}")
        print(f"    mean ratio={rmean:.4f}  CoV={rcov*100:.2f}%")


if __name__ == "__main__":
    main()
