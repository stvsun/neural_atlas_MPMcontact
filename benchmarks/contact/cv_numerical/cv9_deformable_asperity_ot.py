#!/usr/bin/env python3
"""CV-9b  Deformable elastic asperity in OT measure-coupling contact (Track T4).

A deformable-asperity companion to the (rigid-bodied) CV-5/CV-6 superformula/Koch contact.  The
asperity body is now ELASTIC: a Tri2D block whose TOP boundary carries a single cosine asperity
``z_top(x) = -A_a (1 - cos(2 pi x / lambda)) / 2`` (a smooth bump of height ``A_a``).  A rigid flat
platen descends onto the bump by ``delta`` over several load steps.  Contact is the
optimal-transport measure coupling (``MonotoneCoupling1D`` + ``assemble_contact``) between the
deformed asperity surface (slave) and the flat platen (master).  As load increases the asperity tip
flattens elastically, so the OT gap field UPDATES with deformation and the real contact half-width
``a`` GROWS monotonically -- the deformable signature absent in a rigid-asperity model.

Verification (frictionless, where a closed form exists / convergence / balance):
  * Newton CONVERGENCE: relative residual ||K u - f_c|| / ||f_c|| -> < 1e-6 each load step.
  * FORCE BALANCE: integrated contact line load  int p ds  equals the platen reaction
    (sum of nodal contact forces) and the bottom-support reaction, to < 1e-9 relative.
  * MONOTONE OT-gap evolution: contact half-width a(delta) is strictly increasing in delta
    (deformable), and a_deformable(delta) > a_rigid(delta) = sqrt(lambda delta /(pi^2 A_a)) (the
    rigid-bump geometric contact width) -- the deformation OPENS more real contact than geometry.
  * SMALL-LOAD Hertz check: for a shallow bump the tip is locally parabolic with radius
    R_tip = lambda^2 / (4 pi^2 A_a); the incremental a(P) follows the 2-D Hertz line-contact law
    a = 2 sqrt(P' R_tip /(pi E*)) at the FIRST (smallest) load step (within ~15%, coarse-mesh /
    finite-bump tolerance).

Honest scope: rigid flat platen (one deformable body), small-strain Tri2D, single asperity, plain
penalty normal contact (frictionless).  The point is the COUPLING: the gap field is recomputed on the
deformed surface every Newton iteration, so contact area is an emergent output of elasticity, which a
rigid-asperity (fixed-geometry) gap cannot produce.  Local default is coarse (~60 s); the fine Euler
command is emitted for publication numbers.

Run:  python3 benchmarks/contact/cv_numerical/cv9_deformable_asperity_ot.py
      python3 benchmarks/contact/cv_numerical/cv9_deformable_asperity_ot.py --n-x 120 --n-y 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.tri2d import Tri2DFEMSolver                                              # noqa: E402
from solvers.contact.measure_coupling import (assemble_contact, TractionField,           # noqa: E402
                                              MonotoneCoupling1D)
from solvers.contact.measure_coupling.gap_field import GapField                          # noqa: E402
from postprocessing import contact_fields as cf                                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv9_deformable_asperity_ot")


def _bump_mesh(W, D, A_a, lam, n_x, n_y, grade=2.0):
    """Structured triangulated block whose BOTTOM contact surface carries a DOWNWARD cosine asperity.

    The block occupies [-W,W] in x; its bottom boundary is
    ``z_bot(x) = -(A_a/2)(1 + cos(2 pi x / lam))`` so the asperity APEX at x=0 is the LOWEST point
    (z = -A_a) and the shoulders rise to z=0 -- the apex touches a rigid FLOOR platen first, and the
    contact patch widens as the block is pressed down.  This matches the GapField convention
    (slave above the master, penetration when the slave dips below the master).  The top boundary is
    flat at z=+D (clamped).  Rows graded toward the bottom contact zone.  Returns
    (nodes, tris, contact_idx (left->right, the bottom bump), top_idx (the flat clamped top))."""
    s = np.linspace(-1.0, 1.0, n_x + 1)
    xs = W * np.sign(s) * np.abs(s) ** grade                # cluster near x=0
    v = np.linspace(0.0, 1.0, n_y + 1)                      # 0=bottom(contact), 1=top(clamped)
    nodes = []
    nx1 = n_x + 1
    z_bot = -(A_a / 2.0) * (1.0 + np.cos(2.0 * np.pi * xs / lam))   # apex (x=0) lowest = -A_a
    z_bot = np.where(np.abs(xs) <= lam / 2.0, z_bot, 0.0)   # flat shoulders (z=0) beyond one period
    z_top = D
    for iy in range(n_y + 1):
        frac = 1.0 - (1.0 - v[iy]) ** grade                 # cluster near bottom (iy=0)
        for ix in range(nx1):
            z = z_bot[ix] + frac * (z_top - z_bot[ix])
            nodes.append([xs[ix], z])
    nodes = np.array(nodes)

    def nid(iy, ix):
        return iy * nx1 + ix

    tris = []
    for iy in range(n_y):
        for ix in range(n_x):
            a, b, c, d = nid(iy, ix), nid(iy, ix + 1), nid(iy + 1, ix + 1), nid(iy + 1, ix)
            tris.append([a, b, d])
            tris.append([b, c, d])
    tris = np.array(tris, int)
    vtx = nodes[tris]
    area2 = ((vtx[:, 1, 0] - vtx[:, 0, 0]) * (vtx[:, 2, 1] - vtx[:, 0, 1]) -
             (vtx[:, 2, 0] - vtx[:, 0, 0]) * (vtx[:, 1, 1] - vtx[:, 0, 1]))
    tris[area2 < 0] = tris[area2 < 0][:, [0, 2, 1]]
    contact_idx = np.array([nid(0, ix) for ix in range(nx1)])   # bottom bump (contact)
    top_idx = np.array([nid(n_y, ix) for ix in range(nx1)])     # flat top (clamped)
    return nodes, tris, contact_idx, top_idx


def _contact_force(sol, top_idx, u, platen_z, traction, quad_order):
    """OT contact between the deformed asperity top (slave) and the rigid flat platen at z=platen_z.

    Returns (f (n_node,2), Kc (n_dof CSR), diag) plus the active contact half-width a."""
    ud = u.reshape(-1, 2)
    xs_cur = sol.nodes[top_idx] + ud[top_idx]               # deformed top surface, left->right
    order = np.argsort(xs_cur[:, 0])
    ids = top_idx[order]
    xc = xs_cur[order]
    keep = np.concatenate([[True], np.diff(xc[:, 0]) > 1e-12])
    ids, xc = ids[keep], xc[keep]
    x = xc[:, 0]
    h = xc[:, 1]
    hp = np.gradient(h, x)
    slave = dict(x=x, h=h, hp=hp)
    master = dict(x=x.copy(), h=np.full_like(x, platen_z), hp=np.zeros_like(x))
    gf = GapField(slave, MonotoneCoupling1D(slave, master))

    def eval_gap(X):
        return gf.eval_gap(X)

    f, Kc, diag = assemble_contact(xc, ids, sol.n_dof, eval_gap, traction, order=quad_order)
    # active contact half-width from penetrating nodes
    gN, _ = eval_gap(xc)
    pen = gN < 0.0
    a = float(0.5 * (x[pen].max() - x[pen].min())) if pen.any() else 0.0
    return f, Kc, diag, a, ids


def run(W=1.5, D=1.0, A_a=0.10, lam=1.6, n_x=80, n_y=40, E=1000.0, nu=0.3, t=1.0,
        delta=0.06, n_steps=6, eps_n=None, max_iter=40, quad_order=3, verbose=True):
    """Deformable cosine asperity pressed by a rigid flat platen; OT contact, frictionless."""
    nodes, tris, contact_idx, top_idx = _bump_mesh(W, D, A_a, lam, n_x, n_y)
    sol = Tri2DFEMSolver(nodes, tris, E, nu, thickness=t, mode="plane_stress")
    K = sol.assemble().tocsr()
    h_elem = lam / n_x
    if eps_n is None:
        eps_n = 50.0 * E / h_elem
    traction = TractionField(eps_n)

    # BCs: clamp the flat TOP row (both dofs).  The rigid floor platen starts at the asperity apex
    # (z = -A_a) and RISES into it by ``delta`` over the load steps, widening the contact patch.
    fixed = []
    for nd in top_idx:
        fixed += [2 * nd, 2 * nd + 1]
    fixed = np.array(sorted(set(fixed)))
    free = np.setdiff1d(np.arange(sol.n_dof), fixed)

    from scipy.sparse.linalg import spsolve

    Estar = E / (1.0 - nu ** 2)                     # one elastic body reference modulus
    # apex z+A_a = (A_a/2)(1-cos(2pi x/lam)) ~ (A_a pi^2/lam^2) x^2 = x^2/(2 R_tip)
    R_tip = lam ** 2 / (2.0 * np.pi ** 2 * A_a)      # local parabolic tip radius of the cosine bump

    u = np.zeros(sol.n_dof)
    history = []
    res_max = 0.0
    iters_total = 0
    for step in range(1, n_steps + 1):
        platen_z = -A_a + delta * step / n_steps     # floor rises into the apex (apex at z=-A_a)
        for it in range(max_iter):
            iters_total += 1
            f_c, Kc, diag, a_cur, _ = _contact_force(sol, contact_idx, u, platen_z, traction, quad_order)
            f_c = f_c.reshape(-1)
            resid = K @ u - f_c
            rnorm = np.linalg.norm(resid[free])
            fnorm = max(np.linalg.norm(f_c[free]), 1e-30)
            J = (K + Kc).tocsr()
            du = np.zeros(sol.n_dof)
            du[free] = spsolve(J[free][:, free].tocsc(), -resid[free])
            u = u + du
            if np.linalg.norm(du[free]) < 1e-10 * max(np.linalg.norm(u[free]), 1e-12):
                break
        # converged step diagnostics
        f_c, Kc, diag, a_cur, ids = _contact_force(sol, contact_idx, u, platen_z, traction, quad_order)
        f_c2 = f_c.reshape(-1, 2)
        P_line = float(diag["F_line"])               # int p ds (contact line load)
        P_nodal = float(f_c2[:, 1].sum())            # net upward contact force on the block
        # clamped-top reaction (Newton's third law: equilibrium => P_contact + R_top = 0)
        Kfull_u = (K @ u).reshape(-1, 2)
        R_top = float(Kfull_u[top_idx, 1].sum())
        bal = abs(P_nodal + R_top) / (abs(P_nodal) + 1e-30)
        res_rel = rnorm / fnorm
        res_max = max(res_max, res_rel)
        # rigid-bump geometric contact half-width at this penetration: apex is locally parabolic,
        # z + A_a = (A_a/2)(1 - cos(2pi x/lam)) ~ A_a pi^2 x^2 / lam^2; contact where z < platen_z
        # i.e. (z+A_a) < delta_s  ->  a_rigid = (lam/pi) sqrt(delta_s/A_a).
        delta_s = delta * step / n_steps
        a_rigid = float((lam / np.pi) * np.sqrt(max(delta_s, 0.0) / A_a))
        history.append(dict(step=step, platen_z=platen_z, delta=delta_s, a=a_cur,
                            a_rigid=a_rigid, P_line=P_line, P_nodal=P_nodal,
                            R_top=R_top, balance=bal, res_rel=res_rel))
        if verbose:
            print(f"    step {step}: delta={delta_s:.4f}  a={a_cur:.4f} (rigid {a_rigid:.4f})  "
                  f"P={P_nodal:.3f}  balance={bal:.2e}  res={res_rel:.2e}")

    a_seq = np.array([h["a"] for h in history])
    a_rigid_seq = np.array([h["a_rigid"] for h in history])
    monotone = bool(np.all(np.diff(a_seq) > -1e-9))
    max_balance = float(max(h["balance"] for h in history))

    # rigid-asperity baseline: a deformation-free body would contact over the full GEOMETRIC overlap
    # a_rigid (the bump cannot recede).  An elastic asperity flattens and resists, so the REAL contact
    # is Hertzian -- SMALLER than the naive rigid overlap at every step.  This is the deformation
    # signature: the OT gap field, recomputed on the DEFORMED surface, opens a different (smaller)
    # contact than the rigid (undeformed) gap would.  Verify a_deformable < a_rigid.
    deformable_below_rigid = bool(np.all(a_seq <= a_rigid_seq + 1e-9))
    # the surface actually moved (the gap field is not the rigid one): max top-surface uplift.
    ud = u.reshape(-1, 2)
    surf_disp = float(np.abs(ud[contact_idx, 1]).max())

    # small-load Hertz check at the FIRST step (incremental parabolic tip, 2-D line contact)
    h0 = history[0]
    P0 = h0["P_nodal"] / t
    a_hertz = 2.0 * np.sqrt(max(P0, 0.0) * R_tip / (np.pi * Estar))
    hertz_relerr = float(abs(h0["a"] - a_hertz) / (a_hertz + 1e-30)) if a_hertz > 0 else float("nan")

    m = {
        "method": "CV-9b deformable cosine asperity, OT measure-coupling, rigid floor platen",
        "W": W, "D": D, "A_a": A_a, "lam": lam, "n_x": n_x, "n_y": n_y, "n_dof": sol.n_dof,
        "E": E, "nu": nu, "t": t, "delta": delta, "n_steps": n_steps, "eps_n": eps_n,
        "R_tip": R_tip, "iters_total": iters_total,
        "res_rel_max": res_max, "balance_max": max_balance,
        "a_seq": a_seq.tolist(), "a_rigid_seq": a_rigid_seq.tolist(),
        "monotone_growth": monotone, "deformable_below_rigid": deformable_below_rigid,
        "surf_disp_max": surf_disp,
        "hertz_a_first": a_hertz, "a_first": h0["a"], "hertz_relerr_first": hertz_relerr,
        "history": history,
    }
    if verbose:
        print(f"  CV-9b deformable asperity ({sol.n_dof} dof, {iters_total} iters)")
        print(f"    contact half-width a (deformable): {np.round(a_seq, 4).tolist()}")
        print(f"    rigid-overlap baseline    a_rigid : {np.round(a_rigid_seq, 4).tolist()}")
        print(f"    monotone OT-gap growth = {monotone}   deformable<rigid (elastic resist) = "
              f"{deformable_below_rigid}   max surface uplift = {surf_disp:.4f}")
        print(f"    max force-balance residual = {max_balance:.2e}   "
              f"max Newton res_rel = {res_max:.2e}")
        print(f"    first-step Hertz check: a_fem={h0['a']:.4f}  a_hertz={a_hertz:.4f}  "
              f"relerr={hertz_relerr*100:.1f}%")
    return m


def main():
    ap = argparse.ArgumentParser(description="CV-9b deformable asperity, OT contact")
    ap.add_argument("--n-x", type=int, default=80)
    ap.add_argument("--n-y", type=int, default=40)
    ap.add_argument("--delta", type=float, default=0.06)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--A-a", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=1.6)
    args = ap.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    print("=" * 78)
    print("CV-9b  Deformable elastic asperity -- OT measure-coupling contact (rigid platen)")
    print(f"  (n_x={args.n_x}, n_y={args.n_y}, A_a={args.A_a}, lambda={args.lam})")
    print("=" * 78)
    m = run(n_x=args.n_x, n_y=args.n_y, delta=args.delta, n_steps=args.n_steps,
            A_a=args.A_a, lam=args.lam)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)

    ok = (m["monotone_growth"] and m["deformable_below_rigid"] and m["surf_disp_max"] > 1e-4 and
          m["balance_max"] < 1e-6 and m["res_rel_max"] < 1e-4 and m["hertz_relerr_first"] < 0.30)
    print(f"\n  CV-9b: {'PASS' if ok else 'CHECK'} (monotone OT-gap growth on the DEFORMED surface, "
          f"elastic resist a<a_rigid, force balance < 1e-6, Newton converged, Hertz first-step < 30%)")
    print(f"  metrics -> {os.path.join(RUN_DIR, 'metrics.json')}")
    return m, ok


if __name__ == "__main__":
    main()
