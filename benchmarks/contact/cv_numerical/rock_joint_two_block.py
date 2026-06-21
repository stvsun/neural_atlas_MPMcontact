#!/usr/bin/env python3
"""GENUINE TWO-BLOCK rough-joint shear — both faces are DEFORMABLE chart-FEM decoder blocks (Phase 1).

The §11.11 capstone solved ONE deformable rough block on a RIGID mating rough surface.  This is the
honest two-deformable-block extension (manual §11.12 Phase 1): a lower block (rough TOP face) and an
upper block (rough BOTTOM face) are each a trained Fourier ``RoughBlockDecoder`` chart-FEM body, mated
into a tensile-fracture joint and sheared.  BOTH blocks deform; the contact tractions are mutual.

Geometry (a mated tensile fracture):
  * lower block: chart-FEM on decoder_L (rough_face="top"),  bottom face z=-L fixed,
    top face physical z = +L + h(x,y).
  * upper block: chart-FEM on decoder_U (rough_face="bottom"), seed shifted +2L so its bottom face
    physical z = +L + h(x,y) too -> the two rough faces COINCIDE at zero offset (a perfectly mated
    joint, zero initial aperture).  Top face z=+3L is the platen.
  * the platen prescribes shear (u_x,u_y) + a normal control (CNL: free dilation, solve platen u_z for
    target normal load; CNV: pinned u_z).

Contact = NODE-TO-SURFACE (not node-to-node): each lower-top "slave" node contacts the upper block's
*deformed* bottom surface ("master"), found by an approximate inverse map on the upper bottom-face grid.
Because a slave node that slides tangentially physically RIDES UP the master asperity, dilation EMERGES
from the resolved geometry — there is NO effective dilation angle (that flat shortcut is the labelled
§11.10 benchmark).  The contact force is applied to the slave AND distributed to the 4 surrounding
master nodes (Newton's 3rd law) via bilinear weights, so the two FE systems are genuinely coupled.

Consistent penalty tangent (small-sliding, frozen geometry per Newton iter):
    K_c = eps_n * A * ( G G^T  +  mu * T G^T ),
    G = gap gradient  [ +n on slave ; -w_k n on the 4 master nodes ],
    T = friction dir  [ +t on slave ; -w_k t on the 4 master nodes ],
which reduces to the one-block eps_n A (n n^T + mu t n^T) when the master is rigid (w->0).

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_two_block.py --verify
      python3 benchmarks/contact/cv_numerical/rock_joint_two_block.py --mode in_plane --protocol CNL
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver          # noqa: E402
from solvers.fem.linear_elastic import make_linear_elastic_small_strain  # noqa: E402
from solvers.fem.rough_block_decoder import (                          # noqa: E402
    band_limited_rough_surface, train_rough_decoder)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DT = torch.float64
torch.set_default_dtype(DT)

MODES = {"in_plane": (1.0, 0.0), "out_of_plane": (0.0, 1.0),
         "mixed": (math.cos(math.radians(45)), math.sin(math.radians(45)))}


class TwoBlockJointShear:
    """Two mutually-deformable rough decoder blocks in node-to-surface frictional contact."""

    def __init__(self, dec_L, dk_L, dec_U, dk_U, L=1.0, E=2.0e3, nu=0.25, mu=0.4, n_cells=10,
                 surf_amp=0.08, eps_n=None, build_K=True):
        self.L, self.mu, self.surf_amp = L, mu, surf_amp
        self.solL = ChartVectorFEMSolver(n_cells=n_cells, support_r=L, chart_decoder=dec_L,
                                         decoder_kwargs=dk_L, dtype=DT)
        self.solU = ChartVectorFEMSolver(n_cells=n_cells, support_r=L, chart_decoder=dec_U,
                                         decoder_kwargs=dk_U, dtype=DT)
        self.NL, self.NU = self.solL.n_nodes, self.solU.n_nodes
        self.ndof = 3 * (self.NL + self.NU)
        self.offU = 3 * self.NL
        self.pL = self.solL.nodes_phys.numpy()                        # lower physical reference coords
        self.pU = self.solU.nodes_phys.numpy()                        # upper physical reference coords
        sfn, tfn = make_linear_elastic_small_strain(E, nu)
        self.stress_fn, self.tangent_fn = sfn, tfn
        import scipy.sparse as sp
        if build_K:                                                   # skip the (dense) stiffness for viz-only rebuilds
            KL = self.solL.tangent_stiffness(torch.zeros(self.NL, 3), tfn).numpy()
            KU = self.solU.tangent_stiffness(torch.zeros(self.NU, 3), tfn).numpy()
            self.K = sp.block_diag([sp.csr_matrix(KL), sp.csr_matrix(KU)]).tocsr()
        else:
            self.K = None
        # face node sets (reference coords)
        refL = self.solL.nodes.numpy(); refU = self.solU.nodes.numpy()
        self.slave = np.where(np.abs(refL[:, 2] - L) < 1e-9)[0]       # lower TOP face (slave)
        self.botU = np.where(np.abs(refU[:, 2] + L) < 1e-9)[0]        # upper BOTTOM face (master)
        self.botL = np.where(np.abs(refL[:, 2] + L) < 1e-9)[0]        # lower bottom (fixed)
        self.topU = np.where(np.abs(refU[:, 2] - L) < 1e-9)[0]        # upper top (platen)
        # master surface as a regular (a,b) grid on the upper bottom face (reference x,y)
        ab = refU[self.botU, :2]
        self.a_lin = np.unique(np.round(ab[:, 0], 9)); self.b_lin = np.unique(np.round(ab[:, 1], 9))
        self.na, self.nb = len(self.a_lin), len(self.b_lin)
        self.h_cell = (self.a_lin[1] - self.a_lin[0]) if self.na > 1 else 2 * L
        # grid_id[ia,ib] -> GLOBAL upper node index of that bottom-face node
        self.grid_id = np.full((self.na, self.nb), -1, np.int64)
        for loc in self.botU:
            ia = int(round((refU[loc, 0] - self.a_lin[0]) / self.h_cell))
            ib = int(round((refU[loc, 1] - self.b_lin[0]) / self.h_cell))
            self.grid_id[ia, ib] = loc
        assert (self.grid_id >= 0).all(), "upper bottom face is not a full regular grid"
        self.zU0 = self.pU[self.botU]                                 # reference physical of bottom nodes
        # tributary area per slave node (nominal footprint / slave count)
        self.A_slave = (2 * L) ** 2 / max(len(self.slave), 1)
        h = 2 * L / n_cells
        self.eps_n = (20.0 * E / h) if eps_n is None else eps_n

    # ---- master surface helpers (VECTORIZED over all slave nodes) --------------------------------
    def _bilinear_vec(self, qx, qy):
        """Bilinear stencil on the upper bottom (a,b) grid for arrays (qx,qy)->(ids (S,4), w (S,4))."""
        fa = (qx - self.a_lin[0]) / self.h_cell; fb = (qy - self.b_lin[0]) / self.h_cell
        ia = np.clip(np.floor(fa).astype(int), 0, self.na - 2)
        ib = np.clip(np.floor(fb).astype(int), 0, self.nb - 2)
        ta = np.clip(fa - ia, 0.0, 1.0); tb = np.clip(fb - ib, 0.0, 1.0)
        g = self.grid_id
        ids = np.stack([g[ia, ib], g[ia + 1, ib], g[ia, ib + 1], g[ia + 1, ib + 1]], axis=1)
        w = np.stack([(1 - ta) * (1 - tb), ta * (1 - tb), (1 - ta) * tb, ta * tb], axis=1)
        return ids, w

    def _surf_normal_vec(self, a, b):
        """Outward (downward) normals of the upper bottom rough face at material (a,b) arrays -> (S,3)."""
        e = 1e-4
        h0 = band_limited_rough_surface(a, b, amp=self.surf_amp)
        hx = (band_limited_rough_surface(a + e, b, amp=self.surf_amp) - h0) / e
        hy = (band_limited_rough_surface(a, b + e, amp=self.surf_amp) - h0) / e
        sec = np.sqrt(1 + hx ** 2 + hy ** 2)
        return np.stack([hx / sec, hy / sec, -1.0 / sec], 1)

    def _master_vec(self, Xs, Ys, uU):
        """Approx inverse map for all slaves: material (a,b) on the upper bottom whose deformed (x,y) ==
        (Xs,Ys).  Returns (a, b, ids (S,4), w (S,4))."""
        a, b = Xs.copy(), Ys.copy()
        for _ in range(3):
            ids, w = self._bilinear_vec(a, b)
            uin = (uU[ids] * w[:, :, None]).sum(1)                   # (S,3) interpolated upper disp
            a = np.clip(Xs - uin[:, 0], self.a_lin[0], self.a_lin[-1])
            b = np.clip(Ys - uin[:, 1], self.b_lin[0], self.b_lin[-1])
        ids, w = self._bilinear_vec(a, b)
        return a, b, ids, w

    def _contact_geom(self, u, d):
        """Shared geometry for contact force + tangent (all vectorized over slaves)."""
        U = u.reshape(-1, 3); uL = U[:self.NL]; uU = U[self.NL:]
        ps = self.pL[self.slave] + uL[self.slave]                    # (S,3) deformed slave positions
        Xs, Ys, Zs = ps[:, 0], ps[:, 1], ps[:, 2]
        a, b, ids, w = self._master_vec(Xs, Ys, uU)
        Zm = ((uU[ids, 2] + self.pU[ids, 2]) * w).sum(1)            # (S,) master surface height
        nmat = self._surf_normal_vec(a, b)                          # (S,3) downward normal (nz<0)
        gN = (Zm - Zs) * (-nmat[:, 2])                             # <0 => penetration
        active = gN < 0.0
        dvec = np.array([d[0], d[1], 0.0])
        dn = nmat @ dvec
        tproj = dvec[None, :] - dn[:, None] * nmat
        tn = np.linalg.norm(tproj, axis=1, keepdims=True)
        tproj = np.where(tn > 1e-12, tproj / np.clip(tn, 1e-12, None), 0.0)
        return uU, ps, ids, w, nmat, gN, active, tproj

    # ---- contact ---------------------------------------------------------------------------------
    def contact(self, u, d):
        """Node-to-surface penalty + Coulomb friction between lower-top slaves and the upper bottom
        master surface (VECTORIZED).  ``d`` = (dx,dy) shear direction (friction drag direction)."""
        uU, ps, ids, w, nmat, gN, active, tproj = self._contact_geom(u, d)
        f = np.zeros((self.NL + self.NU, 3))
        fn = np.where(active, self.eps_n * (-gN) * self.A_slave, 0.0)     # (S,) >=0
        fs = fn[:, None] * nmat + self.mu * fn[:, None] * tproj          # (S,3) force on each slave node
        np.add.at(f, self.slave, fs)                                     # scatter to lower slave nodes
        for k in range(4):                                               # Newton's 3rd law on masters
            np.add.at(f, self.NL + ids[:, k], -w[:, k, None] * fs)
        Fc = fs.sum(0)
        diag = dict(n_active=int(active.sum()),
                    pen_max=float((-gN[active]).max()) if active.any() else 0.0,
                    Fn=float(-Fc[2]), Fx=float(Fc[0]), Fy=float(Fc[1]))
        return f, diag

    def _contact_tangent(self, u, d):
        from scipy.sparse import coo_matrix
        uU, ps, ids, w, nmat, gN, active, tproj = self._contact_geom(u, d)
        kA = self.eps_n * self.A_slave
        S = len(self.slave)
        # DOF index list per slave: [slave xyz, master0 xyz, ..., master3 xyz]  (S,15)
        dof = np.empty((S, 15), np.int64)
        dof[:, 0:3] = 3 * self.slave[:, None] + np.array([0, 1, 2])
        for k in range(4):
            dof[:, 3 + 3 * k:6 + 3 * k] = self.offU + 3 * ids[:, k][:, None] + np.array([0, 1, 2])
        # gap-gradient G and friction-direction T over the 15 dofs: +n/+t on slave, -w_k n/-w_k t on masters
        G = np.zeros((S, 15)); T = np.zeros((S, 15))
        G[:, 0:3] = nmat; T[:, 0:3] = tproj
        for k in range(4):
            G[:, 3 + 3 * k:6 + 3 * k] = -w[:, k, None] * nmat
            T[:, 3 + 3 * k:6 + 3 * k] = -w[:, k, None] * tproj
        blk = kA * (G[:, :, None] * G[:, None, :] + self.mu * T[:, :, None] * G[:, None, :])  # (S,15,15)
        m = active
        rows = np.broadcast_to(dof[m][:, :, None], (m.sum(), 15, 15)).reshape(-1)
        cols = np.broadcast_to(dof[m][:, None, :], (m.sum(), 15, 15)).reshape(-1)
        vals = blk[m].reshape(-1)
        return coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()

    # ---- solve -----------------------------------------------------------------------------------
    def _free_mask(self, u_top):
        free = np.ones(self.ndof, bool)
        for nidx in self.botL:                                        # lower bottom fixed
            free[3 * nidx:3 * nidx + 3] = False
        for nidx in self.topU:                                        # upper top platen prescribed
            g = self.offU + 3 * nidx
            free[g:g + 3] = False
        return free

    def _apply_bc(self, u, u_top):
        for nidx in self.botL:
            u[3 * nidx:3 * nidx + 3] = 0.0
        for nidx in self.topU:
            g = self.offU + 3 * nidx
            u[g:g + 3] = u_top
        return u

    def solve_fixed(self, u_top, d, u0, max_iter=80, tol=1e-9):
        """Newton (block-diag elastic K + node-to-surface contact + friction), platen at u_top."""
        from scipy.sparse.linalg import spsolve
        free = self._free_mask(u_top)
        u = self._apply_bc(u0.copy().reshape(-1), u_top)
        scaleR = tol * (1 + self.eps_n * self.A_slave)
        rn = None
        for it in range(max_iter):
            f, diag = self.contact(u, d)
            R = self.K @ u - f.reshape(-1)
            rn = np.linalg.norm(R[free])
            if rn < scaleR:
                break
            Kc = self._contact_tangent(u, d)
            Kff = (self.K + Kc)[free][:, free].tocsc()
            du = spsolve(Kff, -R[free])
            step, ok = 1.0, False
            for _ in range(30):
                ut = u.copy(); ut[free] += step * du
                ft, _ = self.contact(ut, d)
                rnt = np.linalg.norm((self.K @ ut - ft.reshape(-1))[free])
                if rnt < (1 - 1e-4 * step) * rn:
                    u, ok = ut, True; break
                step *= 0.5
            if not ok:
                break
        f, diag = self.contact(u, d)
        diag["resid"] = float(np.linalg.norm((self.K @ u - f.reshape(-1))[free]))
        diag["resid_rel"] = diag["resid"] / max(self.eps_n * self.A_slave, 1e-12)
        return u, diag

    def elastic_energy(self, u):
        return 0.5 * float(u @ (self.K @ u))


def _make_blocks(amp=0.08, n_cells=10, L=1.0, mu=0.4, E=2.0e3, iters=4000, eps_n=None, verbose=False):
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=amp)      # noqa: E731
    decL, rmseL, dkL = train_rough_decoder(tgt, rough_face="top", iters=iters)
    decU, rmseU, dkU = train_rough_decoder(tgt, rough_face="bottom", iters=iters)
    dkU = dict(dkU); dkU["seed"] = torch.tensor([0.0, 0.0, 2.0 * L], dtype=DT)   # stack upper above lower
    js = TwoBlockJointShear(decL, dkL, decU, dkU, L=L, E=E, mu=mu, n_cells=n_cells, surf_amp=amp,
                            eps_n=eps_n)
    if verbose:
        print(f"  decoder recon RMSE: lower {rmseL:.3e}  upper {rmseU:.3e}")
    return js, dict(rmse_L=rmseL, rmse_U=rmseU, decL=decL, dkL=dkL, decU=decU, dkU=dkU)


def run_shear(mode="in_plane", protocol="CNL", amp=0.08, n_cells=10, mu=0.4, E=2.0e3,
              shear_total=0.16, n_inc=13, W_over_A=2.0, compress=0.5, iters=4000, js=None,
              info=None, verbose=True):
    """Monotonic genuine TWO-BLOCK shear, 3 modes.  CNL: solve platen u_z for mean normal traction
    W_over_A (platen rises = emergent dilation).  CNV: platen u_z fixed (normal stress evolves)."""
    L = 1.0
    if js is None:
        js, info = _make_blocks(amp=amp, n_cells=n_cells, L=L, mu=mu, E=E, iters=iters, verbose=verbose)
    d = np.asarray(MODES[mode], float)
    dperp = np.array([-d[1], d[0]])
    A_nom = (2 * L) ** 2
    u = np.zeros(js.ndof)
    uz = -compress * amp                                              # platen normal displacement
    rec = {k: [] for k in ("u", "z", "dilation", "Tx", "Ty", "T_par", "T_perp", "sigma_n",
                           "mu_app", "n_active", "pen_max", "resid", "resid_rel")}
    uz0 = None
    snaps = []                                                          # (u_x, displacement) per increment
    for j in range(n_inc):
        umag = shear_total * j / (n_inc - 1)
        ux, uy = umag * d[0], umag * d[1]
        if protocol == "CNL":                                        # damped-Newton platen u_z -> mean tN=W/A
            for _ in range(14):
                u, diag = js.solve_fixed((ux, uy, uz), d, u)
                err = diag["Fn"] / A_nom - W_over_A
                if abs(err) < 1e-4 * W_over_A:
                    break
                frac = max(diag["n_active"], 1) / max(len(js.slave), 1)
                uz += err / (js.eps_n * frac)                        # raise platen if over-compressed
        u, diag = js.solve_fixed((ux, uy, uz), d, u, max_iter=120)
        Fx, Fy, Fn = diag["Fx"], diag["Fy"], diag["Fn"]
        sigma_n = Fn / A_nom
        T_par = (Fx * d[0] + Fy * d[1]) / A_nom
        T_perp = (Fx * dperp[0] + Fy * dperp[1]) / A_nom
        if uz0 is None:
            uz0 = uz
        rec["u"].append(umag); rec["z"].append(uz); rec["dilation"].append(uz - uz0)
        rec["Tx"].append(Fx / A_nom); rec["Ty"].append(Fy / A_nom)
        rec["T_par"].append(T_par); rec["T_perp"].append(T_perp); rec["sigma_n"].append(sigma_n)
        rec["mu_app"].append(T_par / max(sigma_n, 1e-9))
        rec["n_active"].append(diag["n_active"]); rec["pen_max"].append(diag["pen_max"])
        rec["resid"].append(diag["resid"]); rec["resid_rel"].append(diag["resid_rel"])
        snaps.append((float(umag), u.copy()))                           # snapshot for time-evolution viz
        if verbose:
            print(f"    u={umag:.3f}  uz={uz:+.4f}  dil={uz-uz0:+.4f}  tau={T_par:+.4f}  "
                  f"sig={sigma_n:.4f}  mu_app={T_par/max(sigma_n,1e-9):+.4f}  Tperp={T_perp:+.4f}  "
                  f"nC={diag['n_active']}  resid={diag['resid']:.1e}")
    for k in rec:
        rec[k] = np.asarray(rec[k])
    js.u_final = u.copy(); js.uz_final = float(uz)                    # expose final state (for field viz)
    js.u_snaps = snaps                                                # per-increment (u_x, u) for time evolution
    summary = dict(mode=mode, protocol=protocol, amp=amp, mu=mu, E=E, n_cells=n_cells,
                   peak_mu_app=float(np.abs(rec["mu_app"]).max()),
                   total_dilation=float(rec["dilation"][-1]),
                   recon_rmse_L=info["rmse_L"], recon_rmse_U=info["rmse_U"],
                   max_resid=float(rec["resid"].max()))
    return js, info, dict(history=rec, summary=summary,
                          params=dict(direction=d.tolist(), W_over_A=W_over_A, mu=mu, amp=amp,
                                      shear_total=shear_total, n_inc=n_inc, protocol=protocol))


def _save(name, payload):
    out_dir = os.path.join(_ROOT, "runs", name); os.makedirs(out_dir, exist_ok=True)
    ser = {}
    for k, v in payload.items():
        ser[k] = {kk: np.asarray(vv).tolist() for kk, vv in v.items()} if k == "history" else v
    json.dump(ser, open(os.path.join(out_dir, "history.json"), "w"))
    json.dump(payload.get("summary", {}), open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    return out_dir


def verify(amp=0.06, n_cells=8, iters=2500):
    """Phase-1 verification BEFORE any contact claim: det J on both blocks, MMS O(h^2) on the
    bottom-rough decoder, and a frictionless monotonic sanity (emergent dilation > 0, residual ~1e-9)."""
    from benchmarks.contact.cv_numerical.cv7_decoder_verify import mms_convergence
    from solvers.fem.rough_block_decoder import verify_decoder
    print("=== Phase-1 two-block verification ===")
    js, info = _make_blocks(amp=amp, n_cells=n_cells, iters=iters, verbose=True)
    vL = verify_decoder(info["decL"], info["dkL"], n_cells=n_cells)
    dkU = dict(info["dkU"])
    vU = verify_decoder(info["decU"], dkU, n_cells=n_cells)
    print(f"  det J lower in [{vL['detJ_min']:.3f},{vL['detJ_max']:.3f}] valid={vL['all_valid']}")
    print(f"  det J upper in [{vU['detJ_min']:.3f},{vU['detJ_max']:.3f}] valid={vU['all_valid']}")
    errs, rates = mms_convergence(info["decU"], info["dkU"])
    print(f"  MMS (bottom-rough decoder) L2 {['%.2e' % e for e in errs]}  rates {['%.2f' % r for r in rates]}")
    # frictionless emergent dilation sanity (CNL)
    js.mu = 0.0
    _, _, pay = run_shear(mode="in_plane", protocol="CNL", amp=amp, n_cells=n_cells,
                          shear_total=0.12, n_inc=5, js=js, info=info, verbose=False)
    h = pay["history"]
    print(f"  frictionless emergent dilation (in-plane): {h['dilation'][-1]:+.4e}  "
          f"max resid_rel {h['resid_rel'].max():.2e}")
    ok = (vL["all_valid"] and vU["all_valid"] and min(rates) > 1.5
          and h["dilation"][-1] > 0 and h["resid_rel"].max() < 1e-7)
    print("  PHASE-1 VERIFY:", "PASS" if ok else "CHECK")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mode", default="in_plane", choices=list(MODES) + ["all"])
    ap.add_argument("--protocol", default="CNL", choices=["CNL", "CNV"])
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--n_cells", type=int, default=10)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    if args.verify:
        verify(); return
    modes = list(MODES) if args.mode == "all" else [args.mode]
    js = info = None
    for m in modes:
        print(f"\n=== TWO-BLOCK genuine shear: mode={m} protocol={args.protocol} ===")
        js, info, pay = run_shear(mode=m, protocol=args.protocol, amp=args.amp, n_cells=args.n_cells,
                                  mu=args.mu, iters=args.iters, js=(js if m != modes[0] else None),
                                  info=(info if m != modes[0] else None))
        out = _save(f"rock_joint_3d_twoblock_{m}", pay)
        print(f"  peak |mu_app|={pay['summary']['peak_mu_app']:.4f}  "
              f"dilation={pay['summary']['total_dilation']:+.4f}  -> {out}")


if __name__ == "__main__":
    main()
