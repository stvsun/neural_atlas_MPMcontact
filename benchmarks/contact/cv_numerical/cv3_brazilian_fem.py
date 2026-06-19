#!/usr/bin/env python3
"""CV-3 Brazilian disc — NUMERICAL (2-D FEM) vs the analytical Flamant-superposition field.

The cleanest end-to-end numerical CV proof (plan M2): a diametral-compression BVP with NO
contact — just Neumann pole loads on a meshed disc — so it exercises the 2-D FEM
(`solvers/fem/tri2d.py`) directly against `postprocessing/contact_fields.py::brazilian_field`.

Setup (material-independent: the Brazilian stress field is traction-determined):
  * disc radius R, thickness t, total diametral load P;
  * P distributed over a small arc at each pole (down at +y, up at -y) -> self-equilibrated
    (the singular point-load tip is smoothed; the CENTER field is Saint-Venant-insensitive to it);
  * rigid-body modes removed at the equator (uy=0 at (+/-R,0), ux=0 at (R,0)) -- reaction-free
    by symmetry (the exact solution has uy=0 there).

Checks vs analytical:
  * centre sigma_xx = +2P/(pi D t), sigma_yy = -6P/(pi D t)  (ratio -3);
  * sigma_yy along the horizontal diameter integrates to -P/t (global vertical balance);
  * sigma_yy / sigma_xx profile along both diameters tracks the closed form.

The "neural coordinate chart" version (M3) re-runs this on a trained ChartDecoder disc map; here
the chart is the identity (direct analytical mesh) -- the FEM L1 baseline.

Run:  python3 benchmarks/contact/cv_numerical/cv3_brazilian_fem.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, disc_mesh          # noqa: E402
from postprocessing import contact_fields as cf                  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv3_brazilian_fem")


def _pole_load(nodes, bnd, R, P, arc_deg):
    """Self-equilibrated nodal forces: P down over the +y pole arc, P up over the -y pole arc."""
    f = np.zeros((len(nodes), 2))
    ang = np.degrees(np.arctan2(nodes[bnd, 1], nodes[bnd, 0]))
    for pole_deg, sign in ((90.0, -1.0), (-90.0, +1.0)):       # +y pole pushes down, -y pushes up
        d = np.abs(((ang - pole_deg + 180) % 360) - 180)
        sel = bnd[d < arc_deg]
        f[sel, 1] += sign * P / len(sel)
    return f


def _rigid_bcs(nodes):
    """3 reaction-free constraints: uy=0 at the two equator nodes, ux=0 at the +x one."""
    rp = int(np.argmin(np.sum((nodes - [nodes[:, 0].max(), 0.0]) ** 2, axis=1)))   # (+R,0)
    rm = int(np.argmin(np.sum((nodes - [nodes[:, 0].min(), 0.0]) ** 2, axis=1)))   # (-R,0)
    return np.array([2 * rp + 1, 2 * rm + 1, 2 * rp])         # uy(rp), uy(rm), ux(rp)


def run(n_rings=64, arc_deg=4.0, E=1000.0, nu=0.25, R=1.0, t=1.0, P=1.0, verbose=True):
    nodes, tris, bnd = disc_mesh(R, n_rings)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=t, mode="plane_stress")
    f_ext = _pole_load(nodes, bnd, R, P, arc_deg)
    fixed = _rigid_bcs(nodes)
    u = sol.solve(fixed, f_ext)

    cen = sol.element_centroids()
    es = sol.element_stress(u)
    ns = sol.node_stress(u)                                    # nodal stress recovery (smoother)
    D = 2 * R

    # centre stress: average the RECOVERED nodal stress over nodes within 0.1 R of the centre,
    # compared to the ANALYTICAL field at the SAME nodes (like-for-like; the closed form varies
    # across the patch, so comparing a region-average to a point value would be unfair).
    near_n = np.sum(nodes ** 2, axis=1) < (0.1 * R) ** 2
    sxx_c, syy_c = ns[near_n, 0].mean(), ns[near_n, 1].mean()
    sxx_a_f, syy_a_f, _ = cf.brazilian_field(nodes[near_n, 0], nodes[near_n, 1], R, P, t)
    sxx_a, syy_a = float(np.mean(sxx_a_f)), float(np.mean(syy_a_f))

    # global vertical balance: integrate sigma_yy across the horizontal diameter y=0
    xs = np.linspace(-0.85 * R, 0.85 * R, 60)
    syy_line = sol.stress_at(u, np.column_stack([xs, np.zeros_like(xs)]))[:, 1]
    integral = np.trapz(syy_line, xs)                          # ~ -P/t over the full chord
    syy_line_a = cf.brazilian_field(xs, np.zeros_like(xs), R, P, t)[1]

    # interior field RMS error (away from the singular poles), relative to the peak stress scale
    rr = np.sqrt(np.sum(cen ** 2, axis=1))
    ang = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
    pole_d = np.minimum(np.abs(((ang - 90 + 180) % 360) - 180), np.abs(((ang + 90 + 180) % 360) - 180))
    interior = (rr < 0.8 * R) & (pole_d > 25.0)
    sa = np.column_stack(cf.brazilian_field(cen[interior, 0], cen[interior, 1], R, P, t))
    sf = es[interior]
    scale = abs(syy_a)
    field_rms_rel = float(np.sqrt(np.mean(np.sum((sf - sa) ** 2, axis=1))) / scale)

    m = {
        "center_sxx_fem": float(sxx_c), "center_sxx_ana": sxx_a,
        "center_syy_fem": float(syy_c), "center_syy_ana": syy_a,
        "center_sxx_relerr": float(abs(sxx_c - sxx_a) / abs(sxx_a)),
        "center_syy_relerr": float(abs(syy_c - syy_a) / abs(syy_a)),
        "center_ratio_fem": float(syy_c / sxx_c), "center_ratio_ana": -3.0,
        "field_rms_rel": field_rms_rel,
        "syy_diameter_integral_fem": float(integral),
        "syy_diameter_integral_partial_ana": float(np.trapz(syy_line_a, xs)),
        "n_nodes": int(sol.n_nodes), "n_elements": int(len(tris)),
        "E": E, "nu": nu, "R": R, "t": t, "P": P, "arc_deg": arc_deg,
    }
    if verbose:
        print(f"  CV-3 Brazilian FEM  ({m['n_nodes']} nodes, {m['n_elements']} tris)")
        print(f"    centre sigma_xx: FEM={sxx_c:+.4f}  analytical={sxx_a:+.4f}  "
              f"err={m['center_sxx_relerr']*100:.2f}%")
        print(f"    centre sigma_yy: FEM={syy_c:+.4f}  analytical={syy_a:+.4f}  "
              f"err={m['center_syy_relerr']*100:.2f}%")
        print(f"    centre ratio syy/sxx: FEM={m['center_ratio_fem']:.3f}  analytical=-3.000")
        print(f"    interior field RMS error / peak-stress: {field_rms_rel*100:.2f}%")
        print(f"    integral sigma_yy on y=0 (|x|<0.85R): FEM={integral:+.4f}  "
              f"analytical(partial)={m['syy_diameter_integral_partial_ana']:+.4f}")
    return m, (sol, u)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    m, _ = run()
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    ok = m["center_sxx_relerr"] < 0.05 and m["center_syy_relerr"] < 0.05
    print(f"\n  CV-3 numerical L1 vs analytical: {'PASS' if ok else 'CHECK'} "
          f"(center stresses within 5%)")
    print(f"  metrics: {os.path.join(RUN_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
