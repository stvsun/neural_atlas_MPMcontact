#!/usr/bin/env python3
"""CV-2b Cattaneo-Mindlin — TANGENTIAL traction FIELD under partial slip (3-D axisymmetric).

The frictional companion of CV-1c.  A sphere held under normal load F (Hertz pressure p(r) from the
axisymmetric BEM of ``cv1c_hertz3d_field``) is sheared by a tangential force Q < mu F.  The interface
partial-slips: an inner STICK zone r < c (uniform tangential surface displacement) and an outer SLIP
annulus c < r < a where the Coulomb cone is saturated, q = mu p(r).  The stick radius and the
tangential traction FIELD are recovered by a stick/slip active set on the same elastic compliance and
validated against the closed forms ``contact_fields.cattaneo_stick_radius`` and ``cattaneo_traction``.

Mindlin's result: the tangential traction *distribution* that holds the stick zone rigid has the same
compliance structure as the normal Hertz problem, so the recovered ``q(r)`` and ``c`` are exact in
shape (the material factor enters only the tangential displacement magnitude, not the traction).  The
Coulomb cone ``|q| <= mu p`` is the same friction law the measure-coupling ``TractionField`` embodies
(here enforced as an exact stick/slip return rather than its MPM velocity-regularized form).

What IS verified: stick radius c/a = (1 - Q/mu F)^(1/3) to ~1%; the tangential traction FIELD q(r)
pointwise vs Cattaneo to a few percent in the interior.  NOT: the contact/stick edges (sqrt-singular
slopes, ~1-ring rounding); a deformable FEM (the half-space is exact).

Run:  python3 benchmarks/contact/cv_numerical/cv2b_cattaneo_field.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from benchmarks.contact.cv_numerical.cv1c_hertz3d_field import axi_compliance, solve_hertz_axi  # noqa: E402
from postprocessing import contact_fields as cf                                                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv2b_cattaneo_field")


def solve_cattaneo_axi(R=1.0, Estar=1.0, F=0.02, mu=0.5, Q_over_muF=0.5, n=500, max_set_iter=200):
    """Axisymmetric Cattaneo partial slip.  Returns r, q(r), normal p(r), stick radius c."""
    Q = Q_over_muF * mu * F
    norm = solve_hertz_axi(R, Estar, F, n=n, mode="lcp")
    r, p, a = norm["r"], norm["p"], norm["a_ana"]
    dr = np.diff(np.linspace(0.0, 3.0 * a, n + 1))
    A = axi_compliance(r, dr, Estar)
    ring = 2.0 * np.pi * r * dr
    cone = mu * p                                                  # Coulomb cone |q| <= mu p

    in_contact = p > 1e-12
    stick = in_contact.copy()                                     # start fully stuck, shrink from outside
    q = np.zeros(n)
    for _ in range(max_set_iter):
        slip = in_contact & ~stick
        q[slip] = cone[slip]                                      # saturated annulus
        idx = np.where(stick)[0]
        ns = len(idx)
        # [A_ss  -1; ring_s^T  0][q_s; delta_t] = [-A_{s,slip} q_slip; Q - sum_slip q ring]
        M = np.zeros((ns + 1, ns + 1))
        rhs = np.zeros(ns + 1)
        M[:ns, :ns] = A[np.ix_(idx, idx)]
        M[:ns, ns] = -1.0
        M[ns, :ns] = ring[idx]
        sidx = np.where(slip)[0]
        rhs[:ns] = -(A[np.ix_(idx, sidx)] @ q[sidx])
        rhs[ns] = Q - float((q[sidx] * ring[sidx]).sum())
        sol = np.linalg.solve(M, rhs)
        q_s = sol[:ns]
        q[idx] = q_s
        viol = np.abs(q_s) > cone[idx] + 1e-15
        if not viol.any():
            break
        stick[idx[viol]] = False                                  # outer violators go to slip
    c = float(r[stick].max()) if stick.any() else 0.0
    return dict(r=r, q=q, p=p, a=float(a), c=c, p0=float(p.max()),
                mu=mu, Q=Q, F=F, Q_over_muF=float(Q_over_muF))


def run(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=500, verbose=True):
    # tangential traction field at one load ratio + a stick-radius sweep
    sol = solve_cattaneo_axi(R, Estar, F, mu, Q_over_muF=0.5, n=n)
    r, q, a, c, p0 = sol["r"], sol["q"], sol["a"], sol["c"], sol["p0"]
    c_ana = float(cf.cattaneo_stick_radius(sol["Q"], mu, F, a))
    q_ana = cf.cattaneo_traction(r, a, c_ana, mu, p0)
    interior = r <= 0.85 * a
    field_L2 = float(np.linalg.norm((q - q_ana)[interior]) / (np.linalg.norm(q_ana[interior]) + 1e-30))

    sweep = []
    for ratio in (0.2, 0.4, 0.6, 0.8):
        s = solve_cattaneo_axi(R, Estar, F, mu, Q_over_muF=ratio, n=n)
        c_a_fem = s["c"] / s["a"]
        c_a_ana = float(cf.cattaneo_stick_radius(s["Q"], mu, F, s["a"]) / s["a"])
        sweep.append(dict(Q_over_muF=ratio, c_a_fem=c_a_fem, c_a_ana=c_a_ana,
                          relerr=abs(c_a_fem - c_a_ana) / max(c_a_ana, 1e-9)))
    mean_c_relerr = float(np.mean([s["relerr"] for s in sweep]))

    m = {"R": R, "Estar": Estar, "F": F, "mu": mu, "n": n, "a": a,
         "c_fem": c, "c_ana": c_ana, "c_relerr": float(abs(c - c_ana) / max(c_ana, 1e-9)),
         "field_L2_interior": field_L2, "mean_c_relerr": mean_c_relerr, "sweep": sweep}
    hist = {"r": r.tolist(), "q": q.tolist(), "q_cattaneo": q_ana.tolist(),
            "p": sol["p"].tolist(), "a": a, "c": c_ana, "p0": p0, "mu": mu}
    if verbose:
        print(f"  CV-2b Cattaneo TANGENTIAL field (axisymmetric, n={n}, mu={mu}):")
        print(f"    stick radius c/a: FEM={c/a:.4f}  Cattaneo={c_ana/a:.4f}  err={m['c_relerr']*100:.2f}%")
        print(f"    tangential-traction FIELD L2 interior = {field_L2*100:.2f}%   <- the friction field")
        print(f"    stick-radius law c/a=(1-Q/muF)^(1/3) over Q/muF sweep: mean err={mean_c_relerr*100:.2f}%")
    return m, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mu", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    m, hist = run(mu=args.mu, n=args.n)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)
    ok = m["field_L2_interior"] < 0.10 and m["mean_c_relerr"] < 0.05
    print(f"\n  CV-2b tangential field: {'PASS' if ok else 'CHECK'} (field L2 < 10%, c/a law < 5%)")


if __name__ == "__main__":
    main()
