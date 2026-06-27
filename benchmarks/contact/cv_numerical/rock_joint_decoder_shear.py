#!/usr/bin/env python3
"""GENUINE rough-geometry friction shear — chart-FEM on a trained ChartDecoder (the neural atlas).

The honest version of the rock-joint capstone (PI mandate, manual 11.11): a DEFORMABLE block whose
rough top face is a *trained boundary-fitted ChartDecoder* (`solvers/fem/rough_block_decoder.py`,
verified by `cv7_decoder_verify.py`) is sheared against the MATING rigid rough surface under Coulomb
friction.  Dilation and the apparent friction EMERGE from the resolved asperities — there is NO
effective dilation angle (the flat-interface model `rock_joint_cyclic_fem.py` that imposed one is kept
as a labeled benchmark).

  * lower block: chart-FEM on the decoder (rough top face z=1+h(x,y)); bottom face fixed.
  * upper body : rigid platen carrying the MATING rough surface z_up(X,Y)=z_p+h(X-u_x,Y); prescribed
    shear u_x; height z_p set by the protocol (CNL: solve z_p so the total normal force = W -> the
    platen rises = emergent dilation; CNV: z_p fixed -> normal stress rises).
  * contact: node-to-surface penalty + Coulomb friction on the lower block's TOP nodes vs the upper
    rough surface; injected into the FEM residual (the CV-1 static-contact pattern), Newton + line
    search.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_decoder_shear.py --protocol CNL
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


class DecoderJointShear:
    def __init__(self, decoder, dk, L=1.0, E=2.0e3, nu=0.25, mu=0.4, n_cells=12,
                 surf_amp=0.12, eps_n=None):
        self.L, self.mu = L, mu
        self.solver = ChartVectorFEMSolver(n_cells=n_cells, support_r=L, chart_decoder=decoder,
                                           decoder_kwargs=dk, dtype=DT)
        self.N = self.solver.n_nodes
        self.xyz0 = self.solver.nodes_phys.numpy()                    # rough physical reference coords
        self.stress_fn, self.tangent_fn = make_linear_elastic_small_strain(E, nu)
        self.K = self.solver.tangent_stiffness(torch.zeros(self.N, 3, dtype=DT), self.tangent_fn)
        self.K = self.K.numpy()
        ref = self.solver.nodes.numpy()
        self.top = np.where(np.abs(ref[:, 2] - L) < 1e-9)[0]          # rough top-face nodes
        self.bot = np.where(np.abs(ref[:, 2] + L) < 1e-9)[0]          # fixed bottom
        # tributary area per top node (nominal LxL footprint over the top-face node count)
        self.A_top = (2 * L) ** 2 / max(len(self.top), 1)
        h = 2 * L / n_cells
        self.eps_n = (20.0 * E / h) if eps_n is None else eps_n
        self.surf_amp = surf_amp
        self.A_nom = (2 * L) ** 2
        self._surf_fn = lambda x, y: band_limited_rough_surface(x, y, amp=surf_amp)  # mating surface
        # Opt-in App.-E consistent tangent (resolved-force Jacobian = the 2-block tangent plus the
        # Weingarten curvature and friction-direction blocks).  Default False keeps the smoother
        # 2-block tangent, which is more robust on the non-smooth rough-Coulomb interface; the
        # consistent tangent restores quadratic Newton convergence on smooth geometry (App. E study).
        self.consistent_tangent = False

    def _upper_surface(self, X, Y, u_x, z_p):
        """Rigid mating rough surface height + downward-normal gradient at world (X,Y)."""
        eps = 1e-4
        h0 = self._surf_fn(X - u_x, Y)
        hx = (self._surf_fn(X - u_x + eps, Y) - h0) / eps
        hy = (self._surf_fn(X - u_x, Y + eps) - h0) / eps
        return z_p + h0, hx, hy

    def contact(self, u, u_x, z_p):
        """Contact force on all nodes (N,3) + diagnostics; top nodes vs the upper rough surface."""
        f = np.zeros((self.N, 3))
        p = self.xyz0[self.top] + u[self.top]                         # deformed top-node positions
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        z_up, hx, hy = self._upper_surface(X, Y, u_x, z_p)
        sec = np.sqrt(1 + hx ** 2 + hy ** 2)
        nx, ny, nz = hx / sec, hy / sec, -1.0 / sec                   # upper-body outward (downward) normal
        gap = (z_up - Z)                                              # >0 separation, <0 penetration
        gN = gap * (-nz)                                             # normal gap (project onto -n=up)
        active = gN < 0.0
        fn = np.where(active, self.eps_n * (-gN) * self.A_top, 0.0)   # >=0 normal magnitude
        # normal force on the lower node = fn * n  (n is the upper body's outward/DOWNward normal,
        # nz<0) -> pushes the penetrating lower node OUT of the upper body (down).
        Fn = fn[:, None] * np.stack([nx, ny, nz], 1)
        # Coulomb friction: the upper surface slides +x over the node -> friction drags the node +x
        tvec = np.stack([np.ones_like(X), np.zeros_like(X), np.zeros_like(X)], 1)
        tproj = tvec - (tvec * np.stack([nx, ny, nz], 1)).sum(1, keepdims=True) * np.stack([nx, ny, nz], 1)
        tproj = tproj / np.clip(np.linalg.norm(tproj, axis=1, keepdims=True), 1e-12, None)
        Ft = self.mu * fn[:, None] * tproj                           # drags node along upper motion (+x)
        f[self.top] = Fn + Ft
        diag = dict(n_active=int(active.sum()),
                    Fz=float(f[self.top, 2].sum()), Fx=float(f[self.top, 0].sum()),
                    pen_max=float((-gN[active]).max()) if active.any() else 0.0)
        return f, diag

    def _contact_tangent(self, U, u_x, z_p):
        """3x3 contact tangent per ACTIVE top node.

        With ``self.consistent_tangent`` False, only the two analytic blocks of eq:contact-block are
        assembled, eps_n*A*(n n^T + mu t_proj n^T): the tilted-normal penalty stiffness and the
        non-symmetric friction-normal coupling.  With it True (default) the CONSISTENT tangent of
        Appendix~E is assembled: the exact per-node Jacobian of the resolved contact force, which adds
        the curvature (Weingarten) block t_N*A*(dn/du), eq:weingarten-chart, and the friction-direction
        block mu*t_N*A*(dt_proj/du), eq:friction-dir-deriv.  Differencing the resolved force avoids the
        third surface derivatives the closed form would need and stays consistent with the eps-smoothed
        normal the force evaluates, so the Newton iteration recovers its asymptotic quadratic rate on
        smooth geometry (the rough-Coulomb residual is then limited only by the slip non-smoothness)."""
        from scipy.sparse import csr_matrix
        ndof = 3 * self.N
        p = self.xyz0[self.top] + U[self.top]
        z_up, hx, hy = self._upper_surface(p[:, 0], p[:, 1], u_x, z_p)
        sec = np.sqrt(1 + hx ** 2 + hy ** 2)
        gN = (z_up - p[:, 2]) / sec                                      # = gap * (-nz)
        act = gN < 0.0
        if self.consistent_tangent:
            e = 1e-7                                                     # per-node force Jacobian (top nodes independent)
            f0 = self.contact(U, u_x, z_p)[0][self.top]
            blocks = np.zeros((len(self.top), 3, 3))
            for d in range(3):
                Ue = U.copy(); Ue[self.top, d] += e
                blocks[:, :, d] = -(self.contact(Ue, u_x, z_p)[0][self.top] - f0) / e
        else:
            nx, ny, nz = hx / sec, hy / sec, -1.0 / sec
            tvec = np.stack([np.ones_like(hx), np.zeros_like(hx), np.zeros_like(hx)], 1)
            nmat = np.stack([nx, ny, nz], 1)
            tp = tvec - (tvec * nmat).sum(1, keepdims=True) * nmat
            tp = tp / np.clip(np.linalg.norm(tp, axis=1, keepdims=True), 1e-12, None)
            kA = self.eps_n * self.A_top
            blocks = np.stack([kA * (np.outer(nmat[k], nmat[k]) + self.mu * np.outer(tp[k], nmat[k]))
                               for k in range(len(self.top))])
        rows, cols, vals = [], [], []
        for k, nidx in enumerate(self.top):
            if not act[k]:
                continue
            blk = blocks[k]
            for a in range(3):
                for b in range(3):
                    rows.append(3 * nidx + a); cols.append(3 * nidx + b); vals.append(blk[a, b])
        return csr_matrix((vals, (rows, cols)), shape=(ndof, ndof)), int(act.sum())

    def solve_fixed(self, u_x, z_p, u0, max_iter=80, tol=1e-8):
        """Newton (FEM + penalty contact + friction) for the lower block; bottom fixed.
        Consistent contact tangent (tilted n n^T + friction coupling) + Armijo line search."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        ndof = 3 * self.N
        free = np.ones(ndof, bool)
        for nidx in self.bot:
            for c in range(3):
                free[3 * nidx + c] = False
        Ksp = csr_matrix(self.K)
        scaleR = tol * (1 + self.eps_n * self.A_top)
        u = u0.copy().reshape(-1)
        rn = None
        res_hist = []
        for it in range(max_iter):
            U = u.reshape(self.N, 3)
            fext, diag = self.contact(U, u_x, z_p)
            R = self.K @ u - fext.reshape(-1)
            rn = np.linalg.norm(R[free])
            res_hist.append(rn)
            if rn < scaleR:
                break
            Kc, _ = self._contact_tangent(U, u_x, z_p)
            Kff = (Ksp + Kc)[free][:, free].tocsc()
            du = spsolve(Kff, -R[free])
            step, ok = 1.0, False
            for _ in range(30):
                ut = u.copy(); ut[free] += step * du
                Ut = ut.reshape(self.N, 3); ft, _ = self.contact(Ut, u_x, z_p)
                rnt = np.linalg.norm((self.K @ ut - ft.reshape(-1))[free])
                if rnt < (1 - 1e-4 * step) * rn:
                    u, ok = ut, True; break
                step *= 0.5
            if not ok:
                break
        U = u.reshape(self.N, 3)
        fext, diag = self.contact(U, u_x, z_p)
        diag["resid"] = float(np.linalg.norm((self.K @ u - fext.reshape(-1))[free]))
        diag["resid_rel"] = diag["resid"] / max(self.eps_n * self.A_top, 1e-12)
        diag["res_hist"] = res_hist
        diag["n_iter"] = len(res_hist)
        return u, diag

    def normal_force(self, u, u_x, z_p):
        U = u.reshape(self.N, 3); f, _ = self.contact(U, u_x, z_p)
        return -float(f[self.top, 2].sum())                          # upward reaction the platen feels


def run_shear(decoder, dk, protocol="CNL", sigma_n=2.0, shear_total=0.18, n_inc=13, n_cells=12,
              mu=0.4, surf_amp=0.06, eps_n=1.2e4, compress=0.5, W=None, surf_fn=None, verbose=True):
    """Monotonic genuine rough-geometry shear.  CNV: fixed platen at z_p=1-compress*amp (normal stress
    evolves -> emergent strengthening).  CNL: solve z_p so the normal force = W each step (platen rises
    = emergent DILATION).  ``surf_fn`` overrides the upper rough surface (e.g. an SDF-smoothed one)."""
    js = DecoderJointShear(decoder, dk, mu=mu, n_cells=n_cells, surf_amp=surf_amp, eps_n=eps_n)
    if surf_fn is not None:
        js._surf_fn = surf_fn
    u = np.zeros(3 * js.N)
    z_p = 1.0 - compress * surf_amp                                  # CNV platen height
    hist = {k: [] for k in ("u_x", "z_p", "dilation", "tau", "sigma_n", "mu_app", "n_active",
                            "pen_max", "resid")}
    z_p0 = None
    traj = []                                                        # per-increment (u_x, z_p, u) for field viz
    for j in range(n_inc):
        u_x = shear_total * j / (n_inc - 1)
        if protocol == "CNL":                                        # warm-started wide bracket
            zlo, zhi = z_p - 0.6 * surf_amp, z_p + 1.2 * surf_amp
            for _ in range(20):
                zm = 0.5 * (zlo + zhi); u, diag = js.solve_fixed(u_x, zm, u)
                if js.normal_force(u, u_x, zm) > W:
                    zlo = zm
                else:
                    zhi = zm
            z_p = 0.5 * (zlo + zhi)
        u, diag = js.solve_fixed(u_x, z_p, u, max_iter=120)
        traj.append((float(u_x), float(z_p), u.copy()))
        Fn = js.normal_force(u, u_x, z_p)
        U = u.reshape(js.N, 3); fc, _ = js.contact(U, u_x, z_p)
        Fx = float(fc[js.top, 0].sum())
        if z_p0 is None:
            z_p0 = z_p
        tau = Fx / js.A_nom; sig = Fn / js.A_nom
        hist["u_x"].append(u_x); hist["z_p"].append(z_p); hist["dilation"].append(z_p - z_p0)
        hist["tau"].append(tau); hist["sigma_n"].append(sig)
        hist["mu_app"].append(tau / max(sig, 1e-9)); hist["n_active"].append(diag["n_active"])
        hist["pen_max"].append(diag["pen_max"]); hist["resid"].append(diag["resid"])
        if verbose:
            print(f"    u_x={u_x:.3f}  z_p={z_p:.4f}  dil={z_p-z_p0:+.4f}  tau={tau:+.4f}  "
                  f"sig={sig:.4f}  mu_app={tau/max(sig,1e-9):+.4f}  nC={diag['n_active']}  "
                  f"pen={diag['pen_max']:.2e}  resid={diag['resid']:.1e}")
    for k in hist:
        hist[k] = np.asarray(hist[k])
    js.u_final = u.copy()                                            # expose final displacement (for field viz)
    js.z_p_final = float(z_p)
    js.traj = traj                                                   # full (u_x, z_p, u) trajectory
    return js, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="CNL", choices=["CNL", "CNV"])
    ap.add_argument("--sigma_n", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--n_cells", type=int, default=12)
    args = ap.parse_args()
    print("=== training the rough-block decoder (verified separately by cv7_decoder_verify) ===")
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.12)     # noqa: E731
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=4000)
    print(f"  decoder reconstruction RMSE {rmse:.3e}")
    print(f"=== GENUINE rough-geometry shear (protocol={args.protocol}) ===")
    js, hist = run_shear(dec, dk, protocol=args.protocol, sigma_n=args.sigma_n, mu=args.mu,
                         n_cells=args.n_cells)
    out = os.path.join(_ROOT, "runs", "rock_joint_decoder", args.protocol)
    os.makedirs(out, exist_ok=True)
    json.dump({k: v.tolist() for k, v in hist.items()}, open(os.path.join(out, "history.json"), "w"))
    summary = dict(protocol=args.protocol, mu=args.mu, sigma_n=args.sigma_n,
                   decoder_rmse=rmse, peak_mu_app=float(np.abs(hist["mu_app"]).max()),
                   total_dilation=float(hist["dilation"][-1]),
                   mu_base=args.mu, phi_b_deg=math.degrees(math.atan(args.mu)))
    json.dump(summary, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"\n  peak mu_app={summary['peak_mu_app']:.4f}  total dilation={summary['total_dilation']:+.4f} "
          f"(EMERGENT from the rough geometry)  -> {out}")


if __name__ == "__main__":
    main()
