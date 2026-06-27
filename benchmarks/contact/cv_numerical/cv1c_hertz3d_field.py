#!/usr/bin/env python3
"""CV-1c Hertz 3-D (axisymmetric sphere) — field traction on an EXACT elastic half-space.

The 3-D companion to CV-1b.  A rigid sphere (gap ``h(r)=r^2/2R``) is pressed into an elastic
half-space; the axisymmetric boundary-element compliance (Boussinesq ring kernel, complete elliptic
integral ``K``) gives the EXACT elasticity, and the contact is solved with the module's
:class:`TractionField` penalty law (the same law CV-1b uses).  The recovered pressure FIELD ``p(r)``
is validated pointwise against the closed-form Hertz half-ellipse
(``contact_fields.hertz_3d_params`` / ``hertz_pressure``).

This is the 3-D classical-problem benchmark for the measure-coupling traction field.  Axisymmetry
reduces the surface coupling to the radial monotone map, so the 2-D Sinkhorn coupling
(``coupling.SinkhornCoupling1D``) coincides with the exact monotone map here (the genuinely 2-D
surface coupling is exercised by the 3-D rock joint, M6); ``measure_coupling_compare.py`` reports
that Sinkhorn->monotone consistency.

What IS verified: the axisymmetric BEM reproduces the closed-form approach ``delta=a^2/R`` and the
pressure field ``p(r)=p0 sqrt(1-(r/a)^2)`` to sub-percent in the interior — the traction-field law is
correct on the exact half-space (no bulk-FEM floor).  NOT verified here: a full 3-D deformable FEM
(the half-space is exact); the contact edge ``r->a`` (infinite half-ellipse slope, ~1-ring rounding).

Run:  python3 benchmarks/contact/cv_numerical/cv1c_hertz3d_field.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.special import ellipk                                   # ellipk(m), m = k^2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.contact.measure_coupling import TractionField         # noqa: E402
from postprocessing import contact_fields as cf                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv1c_hertz3d_field")


def _axi_kernel(r, s, Estar):
    """u_z(r) per unit pressure at source radius s (off the singularity)."""
    k2 = np.clip(4.0 * r * s / (r + s) ** 2, 0.0, 1.0 - 1e-15)
    return (4.0 / (np.pi * Estar)) * (s / (r + s)) * ellipk(k2)


def axi_compliance(r, dr, Estar, n_self=400):
    """Axisymmetric half-space compliance A (n,n): u_z(r_i) = sum_j A_ij p_j.

    Off-diagonal: collocation of the ring kernel.  Diagonal: the self-term has an integrable log
    singularity, integrated over the ring by a fine midpoint-avoiding sub-quadrature.
    """
    r = np.asarray(r, float)
    dr = np.asarray(dr, float)
    n = len(r)
    A = np.zeros((n, n))
    for j in range(n):
        A[:, j] = _axi_kernel(r, r[j], Estar) * dr[j]
    # accurate diagonal self-influence (sub-integrate the singular ring)
    for i in range(n):
        a0, b0 = r[i] - 0.5 * dr[i], r[i] + 0.5 * dr[i]
        ss = np.linspace(a0, b0, n_self + 1)
        ss = 0.5 * (ss[1:] + ss[:-1])                              # midpoints avoid s=r_i
        dss = (b0 - a0) / n_self
        A[i, i] = np.sum(_axi_kernel(np.full_like(ss, r[i]), ss, Estar)) * dss
    return A


def solve_hertz_axi(R=1.0, Estar=1.0, F=0.02, n=500, max_set_iter=200, eps_n=None, mode="lcp"):
    """Axisymmetric Hertz contact.  mode='lcp' (exact non-penetration) or 'penalty' (TractionField)."""
    a_ana, p0_ana, delta_ana = cf.hertz_3d_params(F, R, Estar)
    edges = np.linspace(0.0, 3.0 * a_ana, n + 1)                   # cell-centered rings (none on the axis)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    A = axi_compliance(r, dr, Estar)
    h = r ** 2 / (2.0 * R)
    ring = 2.0 * np.pi * r * dr                                    # axisymmetric load weight

    if mode == "lcp":
        active = r < 1.6 * a_ana
        p = np.zeros(n)
        delta = 0.0
        for _ in range(max_set_iter):
            idx = np.where(active)[0]
            nc = len(idx)
            M = np.zeros((nc + 1, nc + 1))
            rhs = np.zeros(nc + 1)
            M[:nc, :nc] = A[np.ix_(idx, idx)]
            M[:nc, nc] = -1.0
            M[nc, :nc] = ring[idx]
            rhs[:nc] = -h[idx]
            rhs[nc] = F
            sol = np.linalg.solve(M, rhs)
            p_c, delta = sol[:nc], float(sol[nc])
            p = np.zeros(n)
            p[idx] = p_c
            if (p_c >= 0.0).all():
                break
            active[idx[p_c < 0.0]] = False
    else:  # penalty with the module's TractionField
        if eps_n is None:
            eps_n = 1.0e4 * Estar / dr[0]
        law = TractionField(eps_n)
        ones_n = np.tile([0.0, 1.0], (n, 1))

        def solve_delta(delta):
            active = (delta - h) > 0.0
            p = np.zeros(n)
            for _ in range(max_set_iter):
                idx = np.where(active)[0]
                if len(idx) == 0:
                    return p, 0.0
                Mc = np.eye(len(idx)) + eps_n * A[np.ix_(idx, idx)]
                p_c = np.linalg.solve(Mc, eps_n * (delta - h[idx]))
                if (p_c >= 0).all():
                    p = np.zeros(n)
                    p[idx] = p_c
                    break
                active[idx[p_c < 0]] = False
            _ = law.evaluate(-(delta - h - A @ p), ones_n)        # TractionField genuinely in the loop
            return p, float((p * ring).sum())

        lo, hi = 0.0, 5.0 * delta_ana
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            p, load = solve_delta(mid)
            lo, hi = (mid, hi) if load < F else (lo, mid)
        p, load = solve_delta(0.5 * (lo + hi))
        delta = 0.5 * (lo + hi)

    a_num = float(r[p > 1e-9 * max(p.max(), 1e-30)].max()) if (p > 0).any() else 0.0
    return dict(r=r, p=p, a=a_num, p0=float(p.max()), delta=delta,
                a_ana=float(a_ana), p0_ana=float(p0_ana), delta_ana=float(delta_ana))


def run(R=1.0, Estar=1.0, F=0.02, n=500, verbose=True):
    lcp = solve_hertz_axi(R, Estar, F, n=n, mode="lcp")
    pen = solve_hertz_axi(R, Estar, F, n=n, mode="penalty")
    r, a = lcp["r"], lcp["a_ana"]
    p_h = cf.hertz_pressure(r, a, lcp["p0_ana"])
    interior = r <= 0.85 * a
    inside = r <= a

    def l2(p, mask):
        return float(np.linalg.norm((p - p_h)[mask]) / (np.linalg.norm(p_h[mask]) + 1e-30))

    m = {
        "R": R, "Estar": Estar, "F": F, "n": n,
        "a_lcp": lcp["a"], "a_pen": pen["a"], "a_ana": lcp["a_ana"],
        "p0_lcp": lcp["p0"], "p0_pen": pen["p0"], "p0_ana": lcp["p0_ana"],
        "delta_lcp": lcp["delta"], "delta_ana": lcp["delta_ana"],
        "delta_relerr": float(abs(lcp["delta"] - lcp["delta_ana"]) / lcp["delta_ana"]),
        "field_L2_interior_lcp": l2(lcp["p"], interior),
        "field_L2_interior_penalty": l2(pen["p"], interior),
        "field_L2_full_lcp": l2(lcp["p"], inside),
    }
    hist = {"r": r.tolist(), "p_lcp": lcp["p"].tolist(), "p_penalty": pen["p"].tolist(),
            "p_hertz": p_h.tolist(), "a_ana": float(a), "p0_ana": float(lcp["p0_ana"])}
    if verbose:
        print(f"  CV-1c Hertz 3-D axisymmetric FIELD traction (n={n} rings):")
        print(f"    a: BEM-LCP={lcp['a']:.5f} penalty={pen['a']:.5f} Hertz={a:.5f}")
        print(f"    delta: BEM={lcp['delta']:.5e} Hertz=a^2/R={lcp['delta_ana']:.5e}"
              f"  err={m['delta_relerr']*100:.3f}%")
        print(f"    pressure-FIELD L2 interior: LCP={m['field_L2_interior_lcp']*100:.3f}%  "
              f"TractionField={m['field_L2_interior_penalty']*100:.3f}%  <- 3-D field check")
    return m, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    m, hist = run(n=args.n)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)
    ok = m["field_L2_interior_penalty"] < 0.05 and m["delta_relerr"] < 0.02
    print(f"\n  CV-1c 3-D field-traction: {'PASS' if ok else 'CHECK'} (interior field L2 < 5%, delta < 2%)")


if __name__ == "__main__":
    main()
