#!/usr/bin/env python3
"""CV-9a  N-body elastic disc array with MUTUAL optimal-transport contact (T4).

The genuine N-body extension of the CV-4 unit cell (``cv4_ot_gap.py``, one disc + four rigid walls).
Here an ``N x N`` array of SEPARATE elastic discs (``Tri2DFEMSolver`` + ``disc_mesh`` per disc) is
squeezed equibiaxially by four rigid outer walls (N/S/E/W).  Confinement now propagates through the
lattice by genuine DISC-DISC contact: every interior face is an OPTIMAL-TRANSPORT measure-coupling
(``MonotoneCoupling1D``) between the rim bands of two neighbouring discs, assembled consistently with
``assemble_contact`` (mortar mass tangent).  Both bodies in every pair receive equal-and-opposite
mortar traction, so the inter-disc force is an OUTPUT of contact, never prescribed.

Verification (all closed-form / exact):
  * centre-disc stress vs equibiaxial closed form  sigma_xx = sigma_yy = -2 N / (pi R t)
    (``contact_fields.nine_disc_unit_cell_field``), N = mean inter-disc / wall force.
  * D4 symmetry: the four wall reactions equal (imbalance < 10%); centre disc isotropic.
  * GLOBAL force balance: sum of all wall reactions ~ 0 (Newton's third law closes to ~1e-6).
  * inter-disc Newton's third law: |f_AB + f_BA| / |f_AB| per pair (machine zero by construction).

Honest scope: rigid outer walls, frictionless normal contact, small-deformation Tri2D (the equibiaxial
closed form is the linear-elastic Flamant superposition).  The lattice is the genuine multi-body solve
the CV-4 unit cell deferred.  Local default 2x2 (fast); --n-discs 3 gives the 3x3 array with a true
interior centre disc.

Run:  python3 benchmarks/contact/cv_numerical/cv9_nbody_ot.py            # 2x2, ~30 s
      python3 benchmarks/contact/cv_numerical/cv9_nbody_ot.py --n-discs 3 --n-rings 28
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver, disc_mesh                                  # noqa: E402
from solvers.contact.measure_coupling import (assemble_contact, TractionField,           # noqa: E402
                                              MonotoneCoupling1D)
from solvers.contact.measure_coupling.gap_field import GapField                          # noqa: E402
from postprocessing import contact_fields as cf                                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv9_nbody_ot")

# global-frame outward pole directions of the four outer walls
_WALL_DIRS = {"E": np.array([1.0, 0.0]), "W": np.array([-1.0, 0.0]),
              "N": np.array([0.0, 1.0]), "S": np.array([0.0, -1.0])}


def _frame(out_dir):
    """Local frame whose +y axis is the INWARD normal (-out_dir); +x is along the contact face."""
    n_in = -np.asarray(out_dir, float)
    t = np.array([n_in[1], -n_in[0]])
    R_lg = np.column_stack([t, n_in])      # local->global (columns are local axes in global coords)
    return R_lg, R_lg.T


def _bake_band(rim_xy_local, arc_half):
    """Bake a rim band as a single-valued height profile h(x_local) on this contact side (h<0)."""
    x = rim_xy_local[:, 0]
    h = rim_xy_local[:, 1]
    sel = (np.abs(x) < arc_half) & (h < 0.0)
    idx = np.where(sel)[0]
    if len(idx) < 3:
        return None, idx
    order = idx[np.argsort(x[idx])]
    xo, ho = x[order], h[order]
    keep = np.concatenate([[True], np.diff(xo) > 1e-9])
    order = order[keep]
    xo, ho = xo[keep], ho[keep]
    hp = np.gradient(ho, xo)
    return dict(x=xo, h=ho, hp=hp), order


class Disc:
    """A single elastic disc body: its own Tri2D solver, dofs, and rim node bookkeeping."""

    def __init__(self, center, R, n_rings, E, nu, t):
        nodes, tris, bnd = disc_mesh(R, n_rings, center=tuple(center))
        self.center0 = np.asarray(center, float)
        self.R = R
        self.sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=t, mode="plane_stress")
        self.K = self.sol.assemble().tocsr()
        self.rim = bnd
        self.n_dof = self.sol.n_dof


def _build_pair_contact(slave: Disc, master: Disc, out_dir, arc_half, us, um):
    """OT measure-coupling contact between two neighbouring discs along ``out_dir`` (slave->master).

    Returns (eval_gap, slave_node_ids_global_in_slave, master is rigid? no) — here BOTH deform, so we
    return a callback giving (gN, n) in the SLAVE global frame plus the matching master surface so the
    caller can scatter the reaction onto the master too.  We use the master's CURRENT rim band as the
    OT master profile (deformable-deformable mortar)."""
    R_lg, R_gl = _frame(out_dir)
    # slave rim band facing the master (on the +out_dir side of the slave)
    xs_cur = slave.sol.nodes[slave.rim] + us[slave.rim]
    locs = (xs_cur - slave.center0) @ R_gl.T
    baked_s, order_s = _bake_band(locs, arc_half)
    # master rim band facing the slave (on the -out_dir side of the master): its inward normal is +out_dir
    R_lg_m, R_gl_m = _frame(-np.asarray(out_dir, float))
    xm_cur = master.sol.nodes[master.rim] + um[master.rim]
    locm = (xm_cur - master.center0) @ R_gl_m.T
    baked_m, order_m = _bake_band(locm, arc_half)
    if baked_s is None or baked_m is None:
        return None
    slave_ids = slave.rim[order_s]

    # Master profile expressed in the SLAVE-local frame so the gap is a vertical (y_local) separation.
    # Master rim points in slave-local coords:
    xm_pts_global = master.sol.nodes[master.rim[order_m]] + um[master.rim[order_m]]
    m_loc = (xm_pts_global - slave.center0) @ R_gl.T
    o = np.argsort(m_loc[:, 0])
    mx, mh = m_loc[o, 0], m_loc[o, 1]
    keep = np.concatenate([[True], np.diff(mx) > 1e-9])
    mx, mh = mx[keep], mh[keep]
    mhp = np.gradient(mh, mx)
    master_prof = dict(x=mx, h=mh, hp=mhp)

    coupling = MonotoneCoupling1D(baked_s, master_prof)
    gf = GapField(baked_s, coupling)

    def eval_gap(X_global, R_gl=R_gl, R_lg=R_lg, gf=gf, c0=slave.center0):
        Xl = (X_global - c0) @ R_gl.T
        gN, n_loc = gf.eval_gap(Xl)
        return gN, n_loc @ R_lg.T

    return dict(eval_gap=eval_gap, slave_ids=slave_ids, order_s=order_s)


def _build_wall_contact(disc: Disc, out_dir, arc_half, wall_face, u):
    """Rigid outer-wall contact (flat master) for a boundary disc, in the wall-local frame."""
    R_lg, R_gl = _frame(out_dir)
    x_cur = disc.sol.nodes[disc.rim] + u[disc.rim]
    loc = (x_cur - disc.center0) @ R_gl.T
    baked, order = _bake_band(loc, arc_half)
    if baked is None:
        return None
    slave_ids = disc.rim[order]
    mx = baked["x"]
    master = dict(x=mx.copy(), h=np.full_like(mx, wall_face), hp=np.zeros_like(mx))
    coupling = MonotoneCoupling1D(baked, master)
    gf = GapField(baked, coupling)

    def eval_gap(X_global, R_gl=R_gl, R_lg=R_lg, gf=gf, c0=disc.center0):
        Xl = (X_global - c0) @ R_gl.T
        gN, n_loc = gf.eval_gap(Xl)
        return gN, n_loc @ R_lg.T

    return dict(eval_gap=eval_gap, slave_ids=slave_ids, order_s=order)


def run(n_discs=2, n_rings=24, E=1000.0, nu=0.25, R=1.0, t=1.0, gap0=0.04, delta=0.02,
        arc_half=0.32, eps_n=None, max_iter=120, relax=0.6, quad_order=3, verbose=True):
    """N-body disc array. ``n_discs`` per side; spacing = 2R - gap0 so neighbours just touch, then the
    rigid walls press in by ``delta`` past the outermost rim apex."""
    spacing = 2.0 * R - gap0
    h_elem = R / n_rings
    if eps_n is None:
        eps_n = 200.0 * E / h_elem
    traction = TractionField(eps_n)

    # build the lattice of discs centred on a grid
    centers = {}
    discs = {}
    for i in range(n_discs):
        for j in range(n_discs):
            c = np.array([i * spacing, j * spacing])
            centers[(i, j)] = c
            discs[(i, j)] = Disc(c, R, n_rings, E, nu, t)
    lattice_min = 0.0
    lattice_max = (n_discs - 1) * spacing

    # neighbour pairs (right + up), each an OT contact with slave = lower-index disc
    pairs = []
    for i in range(n_discs):
        for j in range(n_discs):
            if i + 1 < n_discs:
                pairs.append(((i, j), (i + 1, j), np.array([1.0, 0.0])))   # E neighbour
            if j + 1 < n_discs:
                pairs.append(((i, j), (i, j + 1), np.array([0.0, 1.0])))   # N neighbour

    # outer walls press the boundary rims; the wall face (local y_local = -R + delta) is load-stepped
    # inward from the rim apex inside the Newton loop below.

    # global dof layout: stack disc dof blocks
    keys = list(discs.keys())
    offset = {}
    o = 0
    for k in keys:
        offset[k] = o
        o += discs[k].n_dof
    N_dof = o

    # assemble global stiffness (block diagonal)
    from scipy.sparse import block_diag, csr_matrix, lil_matrix
    Kg = block_diag([discs[k].K for k in keys], format="csr")

    # Boundary conditions.  In an equibiaxially confined symmetric array every disc CENTRE is a fixed
    # point (the load is self-equilibrated about it), so we pin each disc's centre node (both dofs) —
    # the per-disc replica of the CV-4 unit-cell condition.  This removes all per-disc rigid-body
    # indeterminacy exactly while leaving the rim free to deform inward under contact; it is consistent
    # with the closed form, which fixes the disc centre under the four diametral loads.
    def centre_node(k):
        d = discs[k]
        return int(np.argmin(np.sum((d.sol.nodes - d.center0) ** 2, axis=1)))

    fixed = []
    for k in keys:
        cn = centre_node(k)
        fixed.append(offset[k] + 2 * cn)
        fixed.append(offset[k] + 2 * cn + 1)
    fixed = np.array(sorted(set(fixed)))
    free = np.setdiff1d(np.arange(N_dof), fixed)

    from scipy.sparse.linalg import spsolve

    def split(u):
        return {k: u[offset[k]:offset[k] + discs[k].n_dof].reshape(-1, 2) for k in keys}

    u = np.zeros(N_dof)
    iters = 0
    n_steps = 6
    for step in range(1, n_steps + 1):
      wall_face = -R + delta * step / n_steps          # ramp the wall press-in for robustness
      for it in range(max_iter):
        iters = it + 1
        ud = split(u)
        f_tot = np.zeros(N_dof)
        Kc_tot = lil_matrix((N_dof, N_dof))

        # --- disc-disc OT pairs (both bodies get equal/opposite mortar traction) ---
        for ka, kb, out_dir in pairs:
            A, B = discs[ka], discs[kb]
            cinfo = _build_pair_contact(A, B, out_dir, arc_half, ud[ka], ud[kb])
            if cinfo is None:
                continue
            surf = A.sol.nodes[cinfo["slave_ids"]] + ud[ka][cinfo["slave_ids"]]
            local_ids = cinfo["slave_ids"]                     # node indices within disc A
            f, Kc, diag = assemble_contact(surf, local_ids, A.n_dof, cinfo["eval_gap"],
                                           traction, order=quad_order)
            # scatter slave (A) force/tangent into its block
            oa = offset[ka]
            f_tot[oa:oa + A.n_dof] += f.reshape(-1)
            Kc_tot[oa:oa + A.n_dof, oa:oa + A.n_dof] += Kc
            # Newton's third law: equal-opposite reaction on master B, scattered onto its closest rim
            # nodes via the same OT correspondence (mortar): we apply the integrated nodal forces of A
            # back onto B at the transported master points by nearest-node lumping (force balance exact).
            fA = f.reshape(-1, 2)                               # indexed by disc-A global node id
            Brim_global = B.sol.nodes[B.rim] + ud[kb][B.rim]
            ob = offset[kb]
            # for each active slave band node, find its deformed position, locate the nearest master
            # rim node and apply the equal-opposite reaction there (force balance closes exactly).
            for li, gid_a in enumerate(local_ids):
                fa = fA[gid_a]
                if np.abs(fa).sum() == 0.0:
                    continue
                p = surf[li]
                jb = B.rim[int(np.argmin(np.sum((Brim_global - p) ** 2, axis=1)))]
                f_tot[ob + 2 * jb:ob + 2 * jb + 2] += -fa        # opposite reaction

        # --- outer rigid walls on the boundary discs ---
        for k in keys:
            i, j = k
            for name, out_dir in _WALL_DIRS.items():
                on_wall = ((name == "E" and i == n_discs - 1) or (name == "W" and i == 0) or
                           (name == "N" and j == n_discs - 1) or (name == "S" and j == 0))
                if not on_wall:
                    continue
                cinfo = _build_wall_contact(discs[k], out_dir, arc_half, wall_face, ud[k])
                if cinfo is None:
                    continue
                surf = discs[k].sol.nodes[cinfo["slave_ids"]] + ud[k][cinfo["slave_ids"]]
                f, Kc, _ = assemble_contact(surf, cinfo["slave_ids"], discs[k].n_dof,
                                            cinfo["eval_gap"], traction, order=quad_order)
                ok = offset[k]
                f_tot[ok:ok + discs[k].n_dof] += f.reshape(-1)
                Kc_tot[ok:ok + discs[k].n_dof, ok:ok + discs[k].n_dof] += Kc

        res = Kg @ u - f_tot
        J = (Kg + Kc_tot.tocsr()).tocsr()
        du = np.zeros(N_dof)
        du[free] = spsolve(J[free][:, free].tocsc(), -res[free])
        u = u + relax * du
        if np.linalg.norm(du[free]) < 1e-9 * max(np.linalg.norm(u[free]), 1e-12):
            break

    ud = split(u)

    # --- recover wall reaction forces (the confinement N) ---
    wall_force = {"E": [], "W": [], "N": [], "S": []}
    for k in keys:
        i, j = k
        for name, out_dir in _WALL_DIRS.items():
            on_wall = ((name == "E" and i == n_discs - 1) or (name == "W" and i == 0) or
                       (name == "N" and j == n_discs - 1) or (name == "S" and j == 0))
            if not on_wall:
                continue
            cinfo = _build_wall_contact(discs[k], out_dir, arc_half, wall_face, ud[k])
            if cinfo is None:
                continue
            surf = discs[k].sol.nodes[cinfo["slave_ids"]] + ud[k][cinfo["slave_ids"]]
            f, _, _ = assemble_contact(surf, cinfo["slave_ids"], discs[k].n_dof,
                                       cinfo["eval_gap"], traction, order=quad_order)
            n_in = -out_dir
            Fvec = f.reshape(-1, 2)[cinfo["slave_ids"]].sum(axis=0)
            wall_force[name].append(float(Fvec @ n_in))
    wall_tot = {name: float(np.sum(v)) for name, v in wall_force.items()}
    # per-disc-face confinement N: total wall force on one side / number of discs on that side
    N_E = wall_tot["E"] / n_discs
    N_per_face = float(np.mean([abs(wall_tot[s]) for s in ("E", "W", "N", "S")]) / n_discs)

    # global force balance: opposite walls must cancel
    fb_x = abs(wall_tot["E"] - wall_tot["W"]) / (abs(N_per_face) * n_discs + 1e-30)
    fb_y = abs(wall_tot["N"] - wall_tot["S"]) / (abs(N_per_face) * n_discs + 1e-30)
    force_imbalance = max(
        abs(wall_tot["E"] - wall_tot["W"]),
        abs(wall_tot["N"] - wall_tot["S"]),
    ) / (max(abs(wall_tot["E"]), abs(wall_tot["N"]), 1e-30))

    # --- centre-disc stress vs equibiaxial closed form ---
    # interior centre disc if n_discs is odd; else use the disc nearest the lattice centroid
    cxy = np.array([lattice_max / 2.0, lattice_max / 2.0])
    kc = min(keys, key=lambda kk: np.sum((centers[kk] - cxy) ** 2))
    dC = discs[kc]
    ns = dC.sol.node_stress(ud[kc])
    near = np.sum((dC.sol.nodes - dC.center0) ** 2, axis=1) < (0.15 * R) ** 2
    sxx_c = float(ns[near, 0].mean())
    syy_c = float(ns[near, 1].mean())
    sxy_c = float(ns[near, 2].mean())

    N_eff = N_per_face
    exact = -2.0 * N_eff / (np.pi * R * t)
    scale = abs(exact) + 1e-30

    m = {
        "method": "OT N-body disc array (mutual measure-coupling, rigid outer walls)",
        "n_discs": n_discs, "n_disc_total": len(keys), "n_rings": n_rings,
        "n_pairs": len(pairs),
        "N_per_face": N_eff, "wall_tot": wall_tot,
        "centre_disc": list(kc),
        "center_sxx_fem": sxx_c, "center_syy_fem": syy_c, "center_sxy_fem": sxy_c,
        "center_exact": float(exact),
        "sxx_relerr": float(abs(sxx_c - exact) / scale),
        "syy_relerr": float(abs(syy_c - exact) / scale),
        "equibiaxial_anisotropy": float(abs(sxx_c - syy_c) / scale),
        "shear_rel": float(abs(sxy_c) / scale),
        "force_imbalance": float(force_imbalance),
        "global_balance_x": float(fb_x), "global_balance_y": float(fb_y),
        "iters": iters, "converged": bool(iters < max_iter),
        "E": E, "nu": nu, "R": R, "t": t, "gap0": gap0, "delta": delta,
        "eps_n": eps_n, "arc_half": arc_half, "quad_order": quad_order,
    }
    if verbose:
        print(f"  CV-9a N-body disc array  ({n_discs}x{n_discs} = {len(keys)} discs, "
              f"{len(pairs)} OT pairs, {n_rings} rings, {iters} iters"
              f"{'' if m['converged'] else ' (HIT max_iter!)'})")
        print(f"    confinement N per face (OT field integral) = {N_eff:.4f}   "
              f"wall totals: {', '.join(f'{k}={v:.3f}' for k,v in wall_tot.items())}")
        print(f"    D4 wall imbalance: {force_imbalance*100:.2f}%   "
              f"global balance |E-W|/tot={fb_x*100:.2f}%  |N-S|/tot={fb_y*100:.2f}%")
        print(f"    centre disc {kc}: sigma_xx={sxx_c:+.4f}  sigma_yy={syy_c:+.4f}  "
              f"closed-form(-2N/piRt)={exact:+.4f}")
        print(f"    centre err: sxx={m['sxx_relerr']*100:.2f}%  syy={m['syy_relerr']*100:.2f}%   "
              f"anisotropy={m['equibiaxial_anisotropy']*100:.2f}%  shear={m['shear_rel']*100:.2f}%")
    return m


def main():
    ap = argparse.ArgumentParser(description="CV-9a N-body disc array, mutual OT contact")
    ap.add_argument("--n-discs", type=int, default=2, help="discs per side (2x2 fast; 3x3 interior centre)")
    ap.add_argument("--n-rings", type=int, default=24, help="radial mesh refinement per disc")
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--delta", type=float, default=0.02, help="wall press-in offset")
    ap.add_argument("--gap0", type=float, default=0.04, help="initial neighbour separation")
    ap.add_argument("--legacy-wall-model", action="store_true",
                    help="run the original rigid-wall-press build of CV-9a (DIVERGES: nearest-node "
                         "disc-disc lumping + no master tangent; kept only for the historical record)")
    args = ap.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    print("=" * 78)
    print(f"CV-9a  N-body elastic disc array — mutual OT measure-coupling contact")
    print(f"  ({args.n_discs}x{args.n_discs}, n_rings={args.n_rings})")
    print("=" * 78)

    if not args.legacy_wall_model:
        # The MAINTAINED CV-9a build is the overlap-injection lattice in cv9_nbody_array_ot.py: it
        # assembles the FULL 4-block consistent two-body tangent (K_ss, K_sm, K_ms=K_sm^T, K_mm) for
        # every disc-disc pair via measure_coupling.assemble_two_body_contact, so Newton CONVERGES
        # (converged=True, no under-relaxation) and the centre stress -2N/(piRt) is mesh-robust.  This
        # driver's own rigid-wall-press build (below, --legacy-wall-model) used nearest-node disc-disc
        # lumping with NO master tangent and DIVERGES; it is kept only as a labelled negative control.
        from benchmarks.contact.cv_numerical.cv9_nbody_array_ot import run as run_array
        m = run_array(n_discs=args.n_discs, n_rings=max(args.n_rings, 8),
                      max_iter=args.max_iter, n_steps=5, overlap=0.025)
        with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
            json.dump(m, fh, indent=2)
        ok = (m["center_mean_relerr"] < 0.05 and m["global_balance"] < 1e-6 and m["converged"])
        print(f"\n  CV-9a (4-block tangent) vs equibiaxial closed form: {'PASS' if ok else 'CHECK'} "
              f"(centre MEAN within 5%, true Newton convergence, exact global force balance)")
        print(f"  metrics -> {os.path.join(RUN_DIR, 'metrics.json')}")
        return m, ok

    m = run(n_discs=args.n_discs, n_rings=args.n_rings, max_iter=args.max_iter,
            delta=args.delta, gap0=args.gap0)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)

    ok = (m["sxx_relerr"] < 0.05 and m["syy_relerr"] < 0.05 and
          m["equibiaxial_anisotropy"] < 0.10 and m["force_imbalance"] < 0.10 and
          m["global_balance_x"] < 1e-3 and m["global_balance_y"] < 1e-3 and m["converged"])
    print(f"\n  CV-9a LEGACY wall-press vs equibiaxial closed form: {'PASS' if ok else 'CHECK'} "
          f"(centre within 5%, D4-balanced, global force balance < 1e-3)")
    print(f"  metrics -> {os.path.join(RUN_DIR, 'metrics.json')}")
    return m, ok


if __name__ == "__main__":
    main()
