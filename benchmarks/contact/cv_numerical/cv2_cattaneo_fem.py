#!/usr/bin/env python3
"""CV-2 Cattaneo–Mindlin partial slip — NUMERICAL (2-D FEM) vs the analytical stick law.

A pre-formed Hertz line contact (CV-1) under a tangential load Q < muP develops a central STICK
zone surrounded by a SLIP annulus.  For 2-D LINE contact the Cattaneo–Mindlin stick half-width is

    c / a = sqrt( 1 - Q/(mu P) )                            (Q = mu P (1 - (c/a)^2))

(the 2-D analogue of the manual's 3-D c/a=(1-Q/muP)^{1/3}; `contact_fields.cattaneo_stick_radius`
is the 3-D form, so the 2-D law is used here), with tangential traction

    q(x) = mu p0 [ sqrt(1 - x^2/a^2)  -  (c/a) sqrt(1 - x^2/c^2) ]   (|x| < c, stick)
    q(x) = mu p0   sqrt(1 - x^2/a^2)                                 (c < |x| < a, slip).

Method (Goodman / uncoupled tangential problem — exact for the stick-radius law, robust): solve the
normal Hertz contact (CV-1) to get the per-node normal force f_n and half-width a; then, holding the
normal state fixed, drive a rigid tangential indenter displacement and solve the tangential field
with per-node Coulomb stick/slip (a tangential spring tying each contact node to the indenter,
return-mapped to the Coulomb cap mu f_n).  Sweep the tangential drive -> a (Q/muP, c/a) curve, and
verify it tracks c/a = sqrt(1 - Q/muP).

Run:  python3 benchmarks/contact/cv_numerical/cv2_cattaneo_fem.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from benchmarks.contact.cv_numerical.cv1_hertz_fem import run as hertz_run, _tributary   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv2_cattaneo_fem")


def _tangential_solve(sol, K_free, free, contact_nodes, fn, xc, dxt, eps_t, mu,
                      max_iter=60, relax=0.5):
    """Decoupled tangential stick/slip solve for one rigid-indenter tangential displacement dxt.

    Returns (u_t flat, q_force per contact node, stuck mask).  Per contact node: a tangential
    spring of nodal stiffness k_t = eps_t * tributary ties u_x to dxt, return-mapped to |q| <= mu f_n
    (stick if under the cap, else slip at the cap)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    n_dof = sol.n_dof
    u = np.zeros(n_dof)
    kt = eps_t                                                      # nodal tangential stiffness
    for _ in range(max_iter):
        ux = u.reshape(sol.n_nodes, 2)[contact_nodes, 0]
        trial = kt * (dxt - ux)                                    # trial tangential nodal force
        cap = mu * fn
        stuck = np.abs(trial) <= cap
        q = np.where(stuck, trial, cap * np.sign(trial))           # Coulomb return-map
        f = np.zeros((sol.n_nodes, 2))
        f[contact_nodes, 0] = q
        # tangential stiffness only on STICK nodes' x-dof
        sidx = contact_nodes[stuck]
        rows = 2 * sidx; vals = np.full(len(sidx), kt)
        Kc = coo_matrix((vals, (rows, rows)), shape=(n_dof, n_dof)).tocsr()
        R = K_free.dot(u) - f.reshape(-1)
        J = (K_free + Kc).tocsr()
        du = np.zeros(n_dof)
        du[free] = spsolve(J[free][:, free].tocsc(), -R[free])
        u = u + relax * du
        if np.linalg.norm(du[free]) < 1e-9 * max(np.linalg.norm(u[free]), 1e-12):
            break
    ux = u.reshape(sol.n_nodes, 2)[contact_nodes, 0]
    trial = kt * (dxt - ux)
    stuck = np.abs(trial) <= mu * fn
    q = np.where(stuck, trial, mu * fn * np.sign(trial))
    return u, q, stuck


def run(mu=0.3, E=10.0, nu=0.3, Rc=1.0, delta=0.025, verbose=True):
    # 1) normal Hertz contact (CV-1) on a DEEP/WIDE domain so the TANGENTIAL response is half-plane
    #    (edge-singular traction -> Cattaneo edge slip), not shallow-block shear (uniform traction).
    m1, (sol, u_n, top, pressure, active) = hertz_run(E=E, nu=nu, Rc=Rc, delta=delta,
                                                      W=1.6, D=1.6, n_x=160, n_y=90, verbose=False)
    ds_top = _tributary(sol.nodes[top, 0])
    contact_nodes = top[active]
    fn = pressure[active] * ds_top[active]                          # nodal normal force (>=0)
    P = float(fn.sum())                                            # total normal line load
    xc = (sol.nodes + u_n)[contact_nodes, 0]                       # contact-node x positions
    x_center0 = xc[int(fn.argmax())]
    a = float(np.max(np.abs(xc - x_center0)))                      # contact half-width from the ACTIVE nodes
    K = sol.assemble()
    # bottom + sides fixed (same as the normal problem)
    th = 1e-6
    side = np.where((np.abs(sol.nodes[:, 0] - sol.nodes[:, 0].max()) < th) |
                    (np.abs(sol.nodes[:, 0] - sol.nodes[:, 0].min()) < th))[0]
    bottom = np.where(np.abs(sol.nodes[:, 1] - sol.nodes[:, 1].min()) < th)[0]
    fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * side]))
    free = np.setdiff1d(np.arange(sol.n_dof), fixed)
    eps_t = float(m1["eps_n"])                                     # match the normal penalty scale

    # 2) sweep the tangential drive geometrically -> (Q/muP, c/a); keep the PARTIAL-slip points
    #    (edge nodes slip first, the centre last, so the stick zone shrinks from a to 0 as Q->muP).
    x_center = xc[int(fn.argmax())]                               # contact-patch centre (peak f_n)
    dxt_scale = mu * float(np.median(fn)) / eps_t                 # node at the cap when dxt~this
    rows = []
    for dxt in dxt_scale * np.geomspace(0.05, 80.0, 18):
        _, q, stuck = _tangential_solve(sol, K, free, contact_nodes, fn, xc, dxt, eps_t, mu)
        Q = float(q.sum())
        ratio = Q / (mu * P)
        if not (0.1 < ratio < 0.92) or stuck.all() or (~stuck).all():
            continue                                              # keep only partial slip
        c_fem = float(np.max(np.abs(xc[stuck] - x_center)))       # outermost stuck node = stick radius
        c_ana = a * np.sqrt(max(1.0 - ratio, 0.0))
        rows.append({"Q_over_muP": ratio, "c_over_a_fem": c_fem / a, "c_over_a_ana": c_ana / a,
                     "c_relerr": abs(c_fem - c_ana) / max(c_ana, 1e-9)})
    # de-duplicate near-identical Q points (geom sweep can repeat once slip saturates)
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda z: z["Q_over_muP"]):
        key = round(r["Q_over_muP"], 2)
        if key not in seen:
            seen.add(key); uniq.append(r)
    rows = uniq

    relerrs = [r["c_relerr"] for r in rows]
    m = {"mu": mu, "a": a, "P": P, "n_contact": int(len(contact_nodes)),
         "stick_law": "c/a = sqrt(1 - Q/muP)  (2-D line contact)",
         "sweep": rows, "max_c_relerr": float(max(relerrs)) if relerrs else float("nan"),
         "mean_c_relerr": float(np.mean(relerrs)) if relerrs else float("nan")}
    if verbose:
        print(f"  CV-2 Cattaneo-Mindlin (2-D line, mu={mu})  a={a:.4f}  P={P:.4f}  "
              f"{len(contact_nodes)} contact nodes")
        print(f"    {'Q/muP':>7} {'c/a FEM':>9} {'c/a ana':>9} {'err':>7}")
        for r in rows:
            print(f"    {r['Q_over_muP']:7.3f} {r['c_over_a_fem']:9.3f} {r['c_over_a_ana']:9.3f} "
                  f"{r['c_relerr']*100:6.1f}%")
        print(f"    stick-radius law c/a=sqrt(1-Q/muP): mean err {m['mean_c_relerr']*100:.1f}%, "
              f"max {m['max_c_relerr']*100:.1f}%")
    return m, (sol, u_n)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    m, _ = run()
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    ok = np.isfinite(m["mean_c_relerr"]) and m["mean_c_relerr"] < 0.15
    print(f"\n  CV-2 numerical vs analytical stick law: {'PASS' if ok else 'CHECK'} "
          f"(mean c/a error < 15%)")


if __name__ == "__main__":
    main()
