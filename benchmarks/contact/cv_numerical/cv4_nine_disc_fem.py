#!/usr/bin/env python3
"""CV-4 nine-disc packing — NUMERICAL (2-D FEM) vs the analytical equibiaxial centre stress.

By full D4 symmetry of a 3x3 isotropically-confined packing, every disc carries the SAME normal
contact force N at its four N/S/E/W neighbours, so the per-disc UNIT CELL is a single disc under
four equal diametral loads.  That load set is the superposition of two orthogonal Brazilian
(diametral-compression) states, giving an EQUIBIAXIAL centre stress

    sigma_xx = sigma_yy = -2 N / (pi R t),   sigma_xy = 0

(the analytical CV-4 result, ``contact_fields.nine_disc_unit_cell_field``).  We solve that unit
cell with the 2-D FEM (``solvers/fem/tri2d.py``) and verify the equibiaxial centre stress — a
clean, contact-free Neumann verification that reuses the CV-3 Brazilian machinery (the neighbour
contacts enter as the four prescribed diametral tractions; the disc geometry is the same neural
disc SDF L0-verified for CV-3).  Narrow load arcs approximate the point-load reference (Hondros).

Scope: this verifies the per-disc equibiaxial CENTRE STRESS (the physics).  The full N-body solve
with explicit inter-disc CONTACT (each disc's force found by the contact solver rather than
prescribed) is the heavier multi-body extension, noted but not built here.

Run:  python3 benchmarks/contact/cv_numerical/cv4_nine_disc_fem.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, disc_mesh              # noqa: E402
from postprocessing import contact_fields as cf                     # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv4_nine_disc_fem")


def _four_pole_load(nodes, bnd, R, N, arc_deg):
    """Self-equilibrated INWARD diametral loads of magnitude N at the four poles
    (+y pushes -y, -y pushes +y, +x pushes -x, -x pushes +x)."""
    f = np.zeros((len(nodes), 2))
    ang = np.degrees(np.arctan2(nodes[bnd, 1], nodes[bnd, 0]))
    poles = [(90.0, np.array([0.0, -1.0])), (-90.0, np.array([0.0, 1.0])),
             (0.0, np.array([-1.0, 0.0])), (180.0, np.array([1.0, 0.0]))]
    for pole_deg, direction in poles:
        d = np.abs(((ang - pole_deg + 180) % 360) - 180)
        sel = bnd[d < arc_deg]
        f[sel] += (N / len(sel)) * direction
    return f


def _rigid_bcs(nodes):
    """3 reaction-free constraints for the self-equilibrated 4-fold load: centre node pinned
    (ux,uy)=0 removes translation; a +x node uy=0 removes rotation.  By D4 symmetry the centre
    does not move, so the reaction is ~0 (verified)."""
    ci = int(np.argmin(np.sum(nodes ** 2, axis=1)))                 # centre (0,0)
    rp = int(np.argmin(np.sum((nodes - [nodes[:, 0].max(), 0.0]) ** 2, axis=1)))   # (+R,0)
    return np.array([2 * ci, 2 * ci + 1, 2 * rp + 1]), ci


def run(n_rings=64, arc_deg=4.0, E=1000.0, nu=0.25, R=1.0, t=1.0, N=1.0, verbose=True):
    nodes, tris, bnd = disc_mesh(R, n_rings)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=t, mode="plane_stress")
    f_ext = _four_pole_load(nodes, bnd, R, N, arc_deg)
    fixed, ci = _rigid_bcs(nodes)
    u = sol.solve(fixed, f_ext)

    # reaction-free check (self-equilibrated 4-fold load => pin reactions ~ 0)
    K = sol.assemble()
    react = (K @ u.reshape(-1) - f_ext.reshape(-1))[fixed]
    reaction_max = float(np.max(np.abs(react)))

    ns = sol.node_stress(u)
    near = np.sum(nodes ** 2, axis=1) < (0.1 * R) ** 2
    sxx_c, syy_c, sxy_c = float(ns[near, 0].mean()), float(ns[near, 1].mean()), float(ns[near, 2].mean())
    exact = -2 * N / (np.pi * R * t)                                # equibiaxial closed form
    sxx_a, syy_a, _ = cf.nine_disc_unit_cell_field(np.array([0.0]), np.array([0.0]), R, N, t)
    scale = abs(exact)

    m = {
        "center_sxx_fem": sxx_c, "center_syy_fem": syy_c, "center_sxy_fem": sxy_c,
        "center_exact": float(exact), "center_ana_field": float(sxx_a[0]),
        "sxx_relerr": float(abs(sxx_c - exact) / scale), "syy_relerr": float(abs(syy_c - exact) / scale),
        "equibiaxial_anisotropy": float(abs(sxx_c - syy_c) / scale),   # |sxx-syy|/|exact| (should be ~0)
        "shear_rel": float(abs(sxy_c) / scale),                        # |sxy|/|exact| (should be ~0)
        "reaction_max": reaction_max,
        "n_nodes": int(sol.n_nodes), "n_elements": int(len(tris)),
        "E": E, "nu": nu, "R": R, "t": t, "N": N, "arc_deg": arc_deg,
    }
    if verbose:
        print(f"  CV-4 nine-disc unit cell  ({m['n_nodes']} nodes, {m['n_elements']} tris)")
        print(f"    centre sigma_xx: FEM={sxx_c:+.4f}  closed-form(-2N/piRt)={exact:+.4f}  err={m['sxx_relerr']*100:.2f}%")
        print(f"    centre sigma_yy: FEM={syy_c:+.4f}  closed-form(-2N/piRt)={exact:+.4f}  err={m['syy_relerr']*100:.2f}%")
        print(f"    equibiaxial anisotropy |sxx-syy|/|s|: {m['equibiaxial_anisotropy']*100:.2f}%   "
              f"shear |sxy|/|s|: {m['shear_rel']*100:.2f}%")
        print(f"    rigid-body pin reaction (should be ~0): {reaction_max:.2e}")
    return m, (sol, u)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    m, _ = run()
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    ok = (m["sxx_relerr"] < 0.05 and m["syy_relerr"] < 0.05
          and m["equibiaxial_anisotropy"] < 0.02 and m["shear_rel"] < 0.02)
    print(f"\n  CV-4 numerical L1 vs analytical: {'PASS' if ok else 'CHECK'} "
          f"(equibiaxial centre within 5%, isotropic)")


if __name__ == "__main__":
    main()
