#!/usr/bin/env python3
"""CV-4 nine-disc unit cell — OPTIMAL-TRANSPORT GAP FIELD vs the equibiaxial closed form.

The OT-gap upgrade of CV-4 (``cv4_nine_disc_fem.py``).  Where the old driver *prescribes* the four
inward diametral loads N as Neumann tractions and reads back the centre stress, this driver makes the
confinement EMERGE from contact: a single elastic disc (``Tri2DFEMSolver`` + ``disc_mesh``) is
squeezed equibiaxially by four RIGID FLAT WALLS (N/S/E/W) pressed inward by a prescribed offset
``delta``.  The per-wall contact force N is found by the contact solver, then validated against the
equibiaxial closed form

    sigma_xx = sigma_yy = -2 N / (pi R t),   sigma_xy = 0           (nine_disc_unit_cell_field)

GAP via measure coupling.  Each wall contact is a 1-D optimal-transport (monotone-rearrangement)
correspondence between two height profiles, in the WALL-LOCAL frame (rotate so the wall normal is +y,
the along-wall axis is x):

  * SLAVE  = the disc rim band near that pole, ordered by the along-wall coordinate, baked as
    ``{x, h(x), h'(x)}`` on the CURRENT deformed configuration.
  * MASTER = the rigid flat wall, a constant-height profile h_m = const pressed ``delta`` past the
    initial rim apex.

``MonotoneCoupling1D(slave, master)`` gives the mass-balanced arclength correspondence
``T = F_m^{-1} o F_s``; ``GapField`` turns it into the ``(g_N, n_s)`` callback; ``assemble_contact``
integrates the interpolated traction field consistently (mortar mass coupling), and ``TractionField``
applies the penalty law ``p_N = eps_n <-g_N>_+``.  The CRITICAL lesson is honoured: the gap is
evaluated at the rim NODES (exact closest-point), then interpolated to Gauss points.

OLD vs NEW.  We also run the old ``cv4_nine_disc_fem.run`` (prescribed-load Neumann) at the SAME
emergent N and contrast: the old centre stress is exact-by-construction (it integrates a prescribed
load); the OT result must REPRODUCE that centre stress with N now an OUTPUT of contact, not an input.

Scope / honesty: this is the per-disc UNIT CELL (one disc, four rigid walls) — the genuine N-body
9-disc array with disc-disc contact is the heavier multi-body extension, noted but not built.  D4
symmetry is checked (four wall forces equal); a residual anisotropy comes from the unstructured disc
mesh not being exactly D4-symmetric.

Run:  python3 benchmarks/contact/cv_numerical/cv4_ot_gap.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, disc_mesh                     # noqa: E402
from solvers.contact.measure_coupling import assemble_contact, TractionField, MonotoneCoupling1D  # noqa: E402
from solvers.contact.measure_coupling.gap_field import GapField             # noqa: E402
from postprocessing import contact_fields as cf                            # noqa: E402
from benchmarks.contact.cv_numerical import cv4_nine_disc_fem as cv4_old    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv4_ot_gap")


# Four cardinal walls.  Each is a rotation that takes the wall-local frame
# (along-wall = +x_local, inward-normal = +y_local) into global coordinates.
# pole_dir = outward global direction of the pole; wall pushes inward (-pole_dir).
_WALLS = {
    "E": np.array([1.0, 0.0]),    # +x pole, wall pushes -x
    "W": np.array([-1.0, 0.0]),   # -x pole, wall pushes +x
    "N": np.array([0.0, 1.0]),    # +y pole, wall pushes -y
    "S": np.array([0.0, -1.0]),   # -y pole, wall pushes +y
}


def _wall_frame(pole_dir):
    """Return (R_lg, R_gl): rotation local->global and global->local for a wall.

    Wall-local frame: inward normal (toward disc centre) = +y_local = -pole_dir;
    along-wall = +x_local = 90deg-rotation of the local normal.
    """
    n_in = -np.asarray(pole_dir, float)          # local +y (points into the disc)
    t = np.array([n_in[1], -n_in[0]])            # local +x  (90deg clockwise of n_in)
    R_lg = np.column_stack([t, n_in])            # columns are local axes in global coords
    return R_lg, R_lg.T


def _bake_wall_slave(rim_xy_local, arc_half):
    """Bake the rim band near a pole as a single-valued height profile in the wall-local frame.

    rim_xy_local : (k,2) rim node positions already rotated into the wall frame (y_local = inward
        depth).  Keep |x_local| < arc_half (the contact band), sort by x_local, drop duplicates.
    Returns (baked dict {x,h,hp}, order indices into rim_xy_local subset, mask of selected nodes).
    """
    x = rim_xy_local[:, 0]
    h = rim_xy_local[:, 1]
    # Select ONLY the pole-near hemisphere (inward depth h<0, i.e. on this wall's side) and the
    # along-wall band |x_local|<arc_half.  Without the h<0 guard the OPPOSITE pole (h~+R) is also
    # within the band and corrupts the single-valued profile.
    sel = (np.abs(x) < arc_half) & (h < 0.0)
    idx = np.where(sel)[0]
    if len(idx) < 3:
        return None, idx
    order = idx[np.argsort(x[idx])]
    xo, ho = x[order], h[order]
    # de-duplicate near-equal x (monotone CDF needs strictly ascending x)
    keep = np.concatenate([[True], np.diff(xo) > 1e-9])
    order = order[keep]
    xo, ho = xo[keep], ho[keep]
    hp = np.gradient(ho, xo)
    return dict(x=xo, h=ho, hp=hp), order


def run(n_rings=64, arc_half=0.30, E=1000.0, nu=0.25, R=1.0, t=1.0, delta=0.01,
        eps_n=None, max_iter=80, relax=0.7, quad_order=3, verbose=True):
    nodes, tris, bnd = disc_mesh(R, n_rings)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=t, mode="plane_stress")
    Kcsr = sol.assemble().tocsr()
    h_elem = R / n_rings
    if eps_n is None:
        eps_n = 300.0 * E / h_elem
    traction = TractionField(eps_n)

    # rim node order along the boundary (by angle) — used to extract per-wall bands
    rim = bnd

    # 3 constraints to kill rigid-body modes (self-equilibrated 4-fold load):
    # centre node pinned, a +x rim node uy=0.
    ci = int(np.argmin(np.sum(nodes ** 2, axis=1)))
    rp = int(np.argmin(np.sum((nodes - [nodes[:, 0].max(), 0.0]) ** 2, axis=1)))
    fixed = np.array([2 * ci, 2 * ci + 1, 2 * rp + 1])
    free = np.setdiff1d(np.arange(sol.n_dof), fixed)

    from scipy.sparse.linalg import spsolve

    # Precompute per-wall frames; the master height (wall face) is placed at the initial rim apex
    # along the inward normal, then pressed an extra `delta` inward.
    frames = {name: _wall_frame(d) for name, d in _WALLS.items()}
    # initial inward depth of the apex node for each wall: y_local of the pole-most rim node = -R
    # (in the local frame the inward normal points into the disc, so the apex sits at y_local=-R).
    # The wall face is at y_local = -R + delta  (pressed delta past the apex => penetration delta).
    wall_face = -R + delta

    def build_wall_contacts(u2):
        """Return list of (eval_gap, node_ids_global, order_local) per wall on the current config."""
        contacts = []
        for name, d in _WALLS.items():
            R_lg, R_gl = frames[name]
            x_cur = nodes[rim] + u2[rim]                       # global deformed rim positions
            loc = x_cur @ R_gl.T                               # rotate into wall-local frame (n,2)
            baked, order = _bake_wall_slave(loc, arc_half)
            if baked is None or len(order) < 3:
                continue
            node_ids_global = rim[order]
            # master: flat wall at y_local = wall_face, spanning the slave x range
            mx = baked["x"]
            master = dict(x=mx.copy(), h=np.full_like(mx, wall_face), hp=np.zeros_like(mx))
            coupling = MonotoneCoupling1D(baked, master)
            gf = GapField(baked, coupling)

            # gap callback in GLOBAL coords: rotate to local, eval OT gap, rotate normal back.
            def eval_gap(X_global, R_gl=R_gl, R_lg=R_lg, gf=gf):
                Xl = X_global @ R_gl.T
                gN, n_loc = gf.eval_gap(Xl)
                n_glob = n_loc @ R_lg.T
                return gN, n_glob

            contacts.append((eval_gap, node_ids_global, order))
        return contacts

    u = np.zeros(sol.n_dof)
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        u2 = u.reshape(sol.n_nodes, 2)
        contacts = build_wall_contacts(u2)
        f_tot = np.zeros((sol.n_nodes, 2))
        from scipy.sparse import csr_matrix
        Kc_tot = csr_matrix((sol.n_dof, sol.n_dof))
        x_curr_global = nodes + u2
        for eval_gap, node_ids_global, order in contacts:
            surf = x_curr_global[node_ids_global]
            f, Kc, _ = assemble_contact(surf, node_ids_global, sol.n_dof, eval_gap,
                                        traction, order=quad_order)
            f_tot += f
            Kc_tot = Kc_tot + Kc
        res = Kcsr @ u - f_tot.reshape(-1)
        J = (Kcsr + Kc_tot).tocsr()
        du = np.zeros(sol.n_dof)
        du[free] = spsolve(J[free][:, free].tocsc(), -res[free])
        u = u + relax * du
        if np.linalg.norm(du[free]) < 1e-9 * max(np.linalg.norm(u[free]), 1e-12):
            break
    u2 = u.reshape(sol.n_nodes, 2)

    # --- recover per-wall contact force N (integrated OT field) ---
    contacts = build_wall_contacts(u2)
    x_curr_global = nodes + u2
    wall_force = {}
    wall_names = list(_WALLS.keys())
    for (eval_gap, node_ids_global, order), name in zip(contacts, wall_names):
        surf = x_curr_global[node_ids_global]
        f, Kc, diag = assemble_contact(surf, node_ids_global, sol.n_dof, eval_gap,
                                       traction, order=quad_order)
        # net inward force magnitude on this wall = |sum of nodal forces| projected on inward normal
        d = _WALLS[name]
        n_in = -d
        Fvec = f[node_ids_global].sum(axis=0)
        wall_force[name] = float(Fvec @ n_in)
    N_emergent = float(np.mean(list(wall_force.values())))
    forces = np.array(list(wall_force.values()))
    force_imbalance = float((forces.max() - forces.min()) / (abs(N_emergent) + 1e-30))

    # --- centre stress ---
    ns = sol.node_stress(u2)
    near = np.sum(nodes ** 2, axis=1) < (0.1 * R) ** 2
    sxx_c = float(ns[near, 0].mean())
    syy_c = float(ns[near, 1].mean())
    sxy_c = float(ns[near, 2].mean())

    exact = -2.0 * N_emergent / (np.pi * R * t)               # equibiaxial closed form at emergent N
    sxx_a, syy_a, _ = cf.nine_disc_unit_cell_field(np.array([0.0]), np.array([0.0]), R, N_emergent, t)
    scale = abs(exact) + 1e-30

    m = {
        "method": "OT-gap (measure coupling, 4 rigid walls, emergent N)",
        "center_sxx_fem": sxx_c, "center_syy_fem": syy_c, "center_sxy_fem": sxy_c,
        "center_exact": float(exact), "center_ana_field": float(sxx_a[0]),
        "N_emergent": N_emergent, "wall_force": wall_force,
        "force_imbalance": force_imbalance,
        "sxx_relerr": float(abs(sxx_c - exact) / scale),
        "syy_relerr": float(abs(syy_c - exact) / scale),
        "equibiaxial_anisotropy": float(abs(sxx_c - syy_c) / scale),
        "shear_rel": float(abs(sxy_c) / scale),
        "iters": iters, "converged": bool(iters < max_iter),
        "n_nodes": int(sol.n_nodes), "n_elements": int(len(tris)),
        "E": E, "nu": nu, "R": R, "t": t, "delta": delta, "eps_n": eps_n,
        "arc_half": arc_half, "quad_order": quad_order,
    }
    if verbose:
        print(f"  CV-4 OT-gap unit cell  ({m['n_nodes']} nodes, {m['n_elements']} tris, "
              f"{iters} iters{'' if m['converged'] else ' (HIT max_iter!)'})")
        print(f"    emergent wall force N (OT field integral) = {N_emergent:.4f}  "
              f"(walls: {', '.join(f'{k}={v:.4f}' for k,v in wall_force.items())})")
        print(f"    force imbalance (D4): {force_imbalance*100:.2f}%")
        print(f"    centre sigma_xx: FEM={sxx_c:+.4f}  closed-form(-2N/piRt)={exact:+.4f}  "
              f"err={m['sxx_relerr']*100:.2f}%")
        print(f"    centre sigma_yy: FEM={syy_c:+.4f}  closed-form(-2N/piRt)={exact:+.4f}  "
              f"err={m['syy_relerr']*100:.2f}%")
        print(f"    equibiaxial anisotropy |sxx-syy|/|s|: {m['equibiaxial_anisotropy']*100:.2f}%   "
              f"shear |sxy|/|s|: {m['shear_rel']*100:.2f}%")
    return m, (sol, u2, wall_force)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CV-4 nine-disc unit cell, OT-gap field")
    ap.add_argument("--n-rings", type=int, default=64,
                    help="radial mesh refinement (round-0 default 64; --fine sweet spot 96).")
    ap.add_argument("--quad-order", type=int, default=3,
                    help="Gauss order for the mortar contact integral (3 is exact for the smooth "
                         "wall pressure; 5 measured identical).")
    ap.add_argument("--fine", action="store_true",
                    help="shorthand for the measured-best ACCURACY mesh (--n-rings 96). Drives the "
                         "centre sigma_xx error from the round-0 0.12%% (n_rings=64, mesh-anisotropy "
                         "inflated) down to the symmetric discretization floor ~0.077%%.")
    args = ap.parse_args()
    n_rings = 96 if args.fine else args.n_rings

    os.makedirs(RUN_DIR, exist_ok=True)
    print("=" * 78)
    print("CV-4 nine-disc unit cell:  OT-gap field (NEW)  vs  prescribed-load (OLD)")
    print(f"  (n_rings={n_rings}, quad_order={args.quad_order})")
    print("=" * 78)
    m, _ = run(n_rings=n_rings, quad_order=args.quad_order)

    # --- OLD driver at the SAME emergent N (prescribed-load Neumann) ---
    print("\n  --- OLD driver (cv4_nine_disc_fem, prescribed-load Neumann) at the same N ---")
    m_old, _ = cv4_old.run(N=m["N_emergent"], n_rings=n_rings, verbose=True)

    m["old_center_sxx"] = m_old["center_sxx_fem"]
    m["old_center_syy"] = m_old["center_syy_fem"]
    m["old_sxx_relerr"] = m_old["sxx_relerr"]
    m["old_anisotropy"] = m_old["equibiaxial_anisotropy"]

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)

    ok = (m["sxx_relerr"] < 0.05 and m["syy_relerr"] < 0.05
          and m["equibiaxial_anisotropy"] < 0.05 and m["shear_rel"] < 0.05
          and m["force_imbalance"] < 0.10 and m["converged"])
    print("\n  OLD vs NEW centre sigma_xx:  "
          f"OLD(prescribed N)={m_old['center_sxx_fem']:+.4f} ({m_old['sxx_relerr']*100:.2f}%)   "
          f"NEW(OT emergent N)={m['center_sxx_fem']:+.4f} ({m['sxx_relerr']*100:.2f}%)")
    print(f"\n  CV-4 OT-gap vs analytical: {'PASS' if ok else 'CHECK'} "
          f"(equibiaxial centre within 5%, isotropic, D4-balanced)")
    print(f"  metrics -> {os.path.join(RUN_DIR, 'metrics.json')}")
    return m, ok


if __name__ == "__main__":
    main()
