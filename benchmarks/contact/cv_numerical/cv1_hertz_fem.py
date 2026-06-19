#!/usr/bin/env python3
"""CV-1 Hertz line contact — NUMERICAL (2-D FEM + penalty contact) vs the analytical solution.

The headline contact benchmark (plan M4): a rigid cylinder (the counter-body, supplied as an
analytic OR a trained NEURAL disc SDF) is pressed into an elastic half-plane; the 2-D FEM
(`solvers/fem/tri2d.py`) with static penalty contact recovers the contact half-width a, the peak
pressure p0 and the line load F, compared to `contact_fields.line_contact_params`.

Static penalty contact (the static analog of the MPM contact channel, reusing the same
(gap, normal) -> force law): at each active-set iteration the surface nodes inside the indenter
get a penalty force f = eps_n <-g> n * ds and a tangent eps_n (n (x) n) ds; the elastic stiffness
is constant (linear elasticity), so only the contact active set is iterated to convergence.

Hertz line contact is scale-free in P/E*, so a SOFT material + a GRADED mesh give a resolvable
contact patch (a ~ 0.1 R); we check the internal Hertz relations for the MEASURED load F
(sidestepping the log-ambiguous 2-D approach-load relation):
    a = 2 sqrt(F R / pi E*),   p0 = 2 F / pi a,   E* = E / (1 - nu^2)  (rigid-on-elastic).

What is and is NOT verified (honest scope, after adversarial review):
  * The ONE independent physics check is a = 2 sqrt(F R / pi E*) — it ties the contact half-width
    to the elastic modulus E*.  It holds: over an 8x E* sweep, a/sqrt(F) tracks 2 sqrt(R/pi E*)
    with a CONSTANT ratio (~0.97 => a steady ~3% low bias from mesh + finite-domain + edge
    quadrature).  Run with --sweep to see it.
  * p0 = 2 F / pi a is the half-ellipse LOAD IDENTITY (line_contact_params DEFINES p0 := 2F/pi a),
    so it is NOT an independent check — it just re-expresses a.
  * The EXACT-SDF (analytic) cylinder is the FAITHFUL solve at ~2-3%.  The trained NEURAL disc SDF
    MATCHES this baseline; it does not beat it (a single-config sub-percent number is geometry-
    specific error cancellation — non-monotonic in delta and degrading as the domain grows).

Run:  python3 benchmarks/contact/cv_numerical/cv1_hertz_fem.py [--neural] [--sweep]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, graded_box_mesh         # noqa: E402
from postprocessing import contact_fields as cf                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv1_hertz_fem")


def _analytic_cylinder(center, Rc):
    """gap/normal of a rigid cylinder: g=|p-c|-Rc (<0 inside), n=(p-c)/|p-c| (points into the
    half-plane = the direction the contact pushes the surface)."""
    c = np.asarray(center, float)

    def gap_normal(pts):
        d = pts - c
        r = np.linalg.norm(d, axis=1)
        return r - Rc, d / r[:, None]
    return gap_normal


def _neural_cylinder(center, Rc, disc_sdf):
    """Rigid cylinder from the trained NEURAL disc SDF (radius-1 disc at origin), scaled to Rc and
    shifted to `center` via the SDF scale property g(p)=Rc*phi_disc((p-c)/Rc), embedded at z=0."""
    import torch
    c = np.asarray(center, float)
    from solvers.contact.gap import evaluate_gap

    def gap_normal(pts):
        q = (pts - c) / Rc
        q3 = np.column_stack([q, np.zeros(len(q))])
        g, n = evaluate_gap(torch.tensor(q3, dtype=torch.float64), disc_sdf)
        n = n.numpy()[:, :2]
        n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
        return Rc * g.numpy(), n
    return gap_normal


def _tributary(x_top):
    """Surface tributary length per top node (sorted by x): half the neighbour spacing."""
    x = np.sort(x_top)
    ds = np.zeros_like(x)
    ds[1:-1] = 0.5 * (x[2:] - x[:-2])
    ds[0] = 0.5 * (x[1] - x[0])
    ds[-1] = 0.5 * (x[-1] - x[-2])
    order = np.argsort(np.argsort(x_top))          # map back to original order
    return ds[order]


def run(E=10.0, nu=0.3, Rc=1.0, delta=0.02, W=0.6, D=0.6, n_x=140, n_y=70, grade=3.0,
        eps_n=None, max_iter=60, relax=None, indenter="analytic", verbose=True):
    nodes, tris, top, bottom = graded_box_mesh(W, D, n_x, n_y, grade)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=1.0, mode="plane_strain")
    K = sol.assemble().tolil()
    Estar = E / (1.0 - nu ** 2)
    # a perfect analytic SDF supports full Newton (relax=1); an imperfect NEURAL SDF (n != dg/du
    # exactly) needs under-relaxation + a softer penalty to avoid active-set chatter.
    if eps_n is None:
        eps_n = (500.0 if indenter == "analytic" else 80.0) * E / (W / n_x)
    if relax is None:
        relax = 1.0 if indenter == "analytic" else 0.5

    cy = Rc - delta                                    # cylinder centre; lowest point at y=-delta
    if indenter == "neural":
        from atlas.sdf.train_analytical_sdf import load_trained_sdf
        disc = load_trained_sdf("disc")
        if disc is None:
            raise RuntimeError("no neural disc SDF — run atlas/sdf/train_analytical_sdf.py --shape disc")
        gap_normal = _neural_cylinder([0.0, cy], Rc, disc)
    else:
        gap_normal = _analytic_cylinder([0.0, cy], Rc)

    ds_top = _tributary(nodes[top, 0])

    # Dirichlet: bottom fully fixed; vertical sides rollered (ux=0) to mimic a half-plane.
    th = W * 1e-6
    side = np.where((np.abs(nodes[:, 0] - W) < th) | (np.abs(nodes[:, 0] + W) < th))[0]
    fixed = np.concatenate([2 * bottom, 2 * bottom + 1, 2 * side])
    fixed = np.unique(fixed)
    free = np.setdiff1d(np.arange(sol.n_dof), fixed)

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    Kcsr = K.tocsr()
    u = np.zeros(sol.n_dof)                              # flat (2N,)
    for it in range(max_iter):
        x_cur = nodes + u.reshape(sol.n_nodes, 2)
        g, ncon = gap_normal(x_cur[top])
        active = g < 0.0
        pen = np.clip(-g, 0.0, None)
        # contact force f_c (N,2) and tangent Kc (= -d f_c/du = eps_n n(x)n ds over active)
        f_c = np.zeros((sol.n_nodes, 2))
        rows, cols, vals = [], [], []
        ak = np.where(active)[0]
        for k in ak:
            nd = top[k]
            f_c[nd] += eps_n * pen[k] * ds_top[k] * ncon[k]
            w = eps_n * ds_top[k] * np.outer(ncon[k], ncon[k])
            for a in range(2):
                for b in range(2):
                    rows.append(2 * nd + a); cols.append(2 * nd + b); vals.append(w[a, b])
        Kc = coo_matrix((vals, (rows, cols)), shape=(sol.n_dof, sol.n_dof)).tocsr() \
            if rows else coo_matrix((sol.n_dof, sol.n_dof)).tocsr()
        # Newton on R(u) = K u - f_c(u) = 0 (f_ext = 0); J = K + Kc
        R = Kcsr @ u - f_c.reshape(-1)
        J = (Kcsr + Kc).tocsr()
        du = np.zeros(sol.n_dof)
        du[free] = spsolve(J[free][:, free].tocsc(), -R[free])
        u = u + relax * du
        if np.linalg.norm(du[free]) < 1e-8 * max(np.linalg.norm(u[free]), 1e-12):
            break
    u = u.reshape(sol.n_nodes, 2)

    # --- recover contact quantities ---
    x_cur = nodes + u
    g, ncon = gap_normal(x_cur[top])
    active = g < 0.0
    pressure = eps_n * np.clip(-g, 0.0, None)          # contact pressure (force / length in 2-D)
    xt = x_cur[top, 0]
    a_edge = float(np.max(np.abs(xt[active]))) if active.any() else 0.0
    # line load by TRAPEZOIDAL integration of the pressure profile (the midpoint/tributary sum
    # over-counts F by ~3.5% because edge nodes get full half-tributaries where the pressure
    # tapers to zero — review finding; trapz tapers correctly at the contact edge).
    xa, pa = xt[active], pressure[active]
    order = np.argsort(xa)
    F = float(np.trapz(pa[order], xa[order]))

    # robust a, p0 by fitting the HERTZIAN half-ellipse p(x)=p0 sqrt(1-(x/a)^2) to ALL contact
    # pressures (the edge node alone is +/- one element noisy); p0 is linear given a -> scan a.
    best = None
    for a_try in np.linspace(0.6 * a_edge, 1.5 * a_edge + 1e-9, 400):
        w = np.sqrt(np.clip(1.0 - (xa / a_try) ** 2, 0.0, None))
        p0_try = float(pa @ w / (w @ w + 1e-30))
        resid = float(np.sum((pa - p0_try * w) ** 2))
        if best is None or resid < best[0]:
            best = (resid, a_try, p0_try)
    a_fem, p0_fem = best[1], best[2]

    a_ana, p0_ana = cf.line_contact_params(F, Rc, Estar)

    m = {
        "indenter": indenter, "E": E, "nu": nu, "Rc": Rc, "delta": delta, "Estar": Estar,
        "eps_n": eps_n, "relax": relax, "n_active": int(active.sum()), "iters": it + 1,
        "converged": bool(it + 1 < max_iter),
        "F_line_load": F, "a_fem": a_fem, "a_ana": float(a_ana), "a_edge_node": a_edge,
        "p0_fem": p0_fem, "p0_ana": float(p0_ana), "p0_peak_node": float(pressure.max()),
        # a(F) tying the half-width to E* is the ONE genuinely independent physics check; p0=2F/pi a
        # is the half-ellipse load identity (p0_ana := 2F/pi a_ana BY DEFINITION) -> NOT independent.
        "a_relerr": float(abs(a_fem - a_ana) / a_ana),
        "p0_relerr": float(abs(p0_fem - p0_ana) / p0_ana),
        "p0_independent": False,
        "n_nodes": int(sol.n_nodes), "n_elements": int(len(tris)),
    }
    if verbose:
        print(f"  CV-1 Hertz line contact  [{indenter} indenter]  "
              f"({m['n_nodes']} nodes, {m['iters']} iters{'' if m['converged'] else ' (HIT max_iter!)'}, "
              f"{m['n_active']} contact nodes)")
        print(f"    line load F = {F:.4f}")
        print(f"    half-width a: FEM={a_fem:.4f}  Hertz(F)={a_ana:.4f}  err={m['a_relerr']*100:.2f}%"
              f"   <- the independent a(F)-E* physics check")
        print(f"    peak press p0: FEM={p0_fem:.4f}  Hertz(F)={p0_ana:.4f}  err={m['p0_relerr']*100:.2f}%"
              f"   (NOT independent: half-ellipse identity p0=2F/pi a)")
    return m, (sol, u, top, pressure, active)


def estar_sweep(indenter="analytic", E_list=(5.0, 10.0, 20.0, 40.0), nu=0.3, Rc=1.0, delta=0.02):
    """The GENUINE, reproducible Hertz physics check: the contact half-width must encode the
    elastic modulus as a = 2 sqrt(F R / pi E*).  Vary E by 8x and confirm a/sqrt(F) tracks
    2 sqrt(R/pi E*) with a CONSTANT ratio (the ~3% bias is mesh/finite-domain/edge-quadrature,
    not E*-dependent).  This is the one degree of freedom that is NOT self-consistency."""
    rows = []
    for E in E_list:
        m, _ = run(E=E, nu=nu, Rc=Rc, delta=delta, indenter=indenter, verbose=False)
        Estar = E / (1.0 - nu ** 2)
        ratio = (m["a_fem"] / np.sqrt(m["F_line_load"])) / (2.0 * np.sqrt(Rc / (np.pi * Estar)))
        rows.append((E, Estar, m["F_line_load"], m["a_fem"], ratio))
    ratios = np.array([r[4] for r in rows])
    return rows, float(ratios.mean()), float(ratios.std() / ratios.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true", help="use the trained neural disc SDF as the indenter")
    ap.add_argument("--sweep", action="store_true", help="run the E* and delta sweeps (honest accuracy)")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    ind = "neural" if args.neural else "analytic"
    m, _ = run(indenter=ind)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    # PASS on the independent a(F)-E* check only (p0 is the half-ellipse identity, not independent)
    ok = m["a_relerr"] < 0.05 and m["converged"]
    print(f"\n  CV-1 numerical L1 (a(F)-E* check): {'PASS' if ok else 'CHECK'} (a within 5%)")
    print("  NOTE: the exact-SDF (analytic) solve is the FAITHFUL baseline at ~2-3% (mesh + finite-"
          "domain + edge); the neural disc SDF MATCHES it (it does not beat it — the occasional "
          "sub-percent point is geometry-specific error cancellation, see --sweep).")
    if args.sweep:
        print("\n  delta sweep (geometry-dependence of the single-point a-error):")
        for d in (0.01, 0.02, 0.04):
            md, _ = run(delta=d, indenter=ind, verbose=False)
            print(f"    delta={d:.3f}: a_err={md['a_relerr']*100:5.2f}%  (F={md['F_line_load']:.4f})")
        rows, rmean, rcov = estar_sweep(indenter=ind)
        print(f"\n  E* sweep (the genuine physics: a/sqrt(F) / 2sqrt(R/piE*) should be CONSTANT):")
        for E, Es, F, a, ratio in rows:
            print(f"    E={E:5.1f} (E*={Es:7.2f}): F={F:.4f} a={a:.4f}  ratio={ratio:.4f}")
        print(f"    mean ratio={rmean:.4f}  coeff-of-variation={rcov*100:.2f}%  "
              f"(~{(1-rmean)*100:.1f}% constant low bias = the discretization floor)")


if __name__ == "__main__":
    main()
