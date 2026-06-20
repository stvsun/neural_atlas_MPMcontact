#!/usr/bin/env python3
"""GENUINE rough-geometry CYCLIC shear with a CLOSED energy ledger (Phase 4).

The flat-interface cyclic benchmark (`rock_joint_cyclic_fem.py`, §11.10) put the roughness in an
effective dilation angle and its cyclic energy balance was honestly stuck at ~1.5x: (i) Coulomb
friction is non-smooth and the full-slip stateless model does not reproduce stick/slip hysteresis,
and (ii) the ledger was incomplete (only W_ext vs W_fric).  This driver solves the GENUINE rough
geometry (chart-FEM on the trained Fourier decoder, the same as §11.11) under CYCLIC shear with:

  * STATEFUL friction — a penalty-regularized Coulomb RETURN MAP carrying per-contact plastic slip
    s_p across increments (elastic predictor t_tr = k_t A (s_rel - s_p); stick if |t_tr| <= mu f_N,
    else radial return to the cone + plastic-slip update).  This gives genuine stick/slip hysteresis;
  * a COMPLETE energy ledger   W_ext = dU_el + W_fric + W_pen   computed incrementally, where
        W_ext  = sum  F_contact . d(platen displacement)        (machine work),
        dU_el  = 1/2 u^T K u                                     (recoverable block strain energy),
        W_fric = sum |t_T| . |ds_p|   (>=0)                      (frictional dissipation, plastic slip),
        W_pen  = 1/2 sum eps_n A g_N^2                           (recoverable normal penalty energy).
    Over a CLOSED cycle dU_el -> 0 and W_pen -> 0, so the gate is  W_ext / W_fric -> 1  ([0.98,1.02]).

We REPORT the measured closure ratio honestly (per the user's directive) — no tuning to hit a target.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_decoder_cyclic.py --mode in_plane --cycles 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import DecoderJointShear         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DT = torch.float64
torch.set_default_dtype(DT)

MODES = {"in_plane": (1.0, 0.0), "out_of_plane": (0.0, 1.0),
         "mixed": (math.cos(math.radians(45)), math.sin(math.radians(45)))}


class CyclicJoint:
    """One deformable rough decoder block sheared cyclically against the rigid mating rough surface,
    with a STATEFUL friction return map (stick/slip) and a complete energy ledger."""

    def __init__(self, dec, dk, L=1.0, E=2.0e3, nu=0.25, mu=0.4, n_cells=10, surf_amp=0.08,
                 eps_n=None, kt_ratio=1.0, c_deg=0.0):
        self.js = DecoderJointShear(dec, dk, L=L, E=E, nu=nu, mu=mu, n_cells=n_cells,
                                    surf_amp=surf_amp, eps_n=eps_n)
        self.L, self.mu, self.amp, self.c_deg = L, mu, surf_amp, c_deg
        self.eps_n, self.A = self.js.eps_n, self.js.A_top
        self.kt = kt_ratio * self.eps_n
        self.top, self.bot, self.N = self.js.top, self.js.bot, self.js.N
        self.xyz0 = self.js.xyz0
        self.K = self.js.K
        self.A_nom = self.js.A_nom
        self.reset_state()

    def reset_state(self):
        self.sp = np.zeros((len(self.top), 3))         # stored plastic tangential slip (tangent-plane)
        self.Wp = np.zeros(len(self.top))              # accumulated plastic work per node (degradation)
        self.Ft_prev = np.zeros((len(self.top), 3))    # committed friction force (for trapezoidal work)

    def _surface(self, X, Y, u_x, z_p, d):
        """Rigid mating rough surface height + downward normal at world (X,Y); sheared by u_x along d."""
        e = 1e-4
        sx, sy = u_x * d[0], u_x * d[1]
        h0 = band_limited_rough_surface(X - sx, Y - sy, amp=self.amp)
        hx = (band_limited_rough_surface(X - sx + e, Y - sy, amp=self.amp) - h0) / e
        hy = (band_limited_rough_surface(X - sx, Y - sy + e, amp=self.amp) - h0) / e
        sec = np.sqrt(1 + hx ** 2 + hy ** 2)
        return z_p + h0, hx / sec, hy / sec, -1.0 / sec

    def contact(self, u, u_x, z_p, d, commit=False):
        """Stateful penalty + Coulomb-return-map contact.  Returns (force (N,3), diag).
        commit=True updates plastic slip s_p and accumulated work, and reports dissipation increment."""
        f = np.zeros((self.N, 3))
        U = u.reshape(self.N, 3)
        p = self.xyz0[self.top] + U[self.top]
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        z_up, hx, hy, nz = self._surface(X, Y, u_x, z_p, d)
        gN = (z_up - Z) * (-nz)
        active = gN < 0.0
        fn = np.where(active, self.eps_n * (-gN) * self.A, 0.0)
        nvec = np.stack([hx, hy, nz], 1)
        d3 = np.array([d[0], d[1], 0.0])
        # tangential relative slip of block w.r.t. the rigid upper surface (which moves u_x*d), projected
        s_rel = U[self.top] - u_x * d3[None, :]
        s_rel = s_rel - (s_rel * nvec).sum(1, keepdims=True) * nvec          # tangent-plane component
        ktA = self.kt * self.A
        # friction force on the block OPPOSES its elastic relative slip (return-map / elastoplastic slider)
        ft_tr = -ktA * (s_rel - self.sp)                                     # trial friction force
        ft_tr = ft_tr - (ft_tr * nvec).sum(1, keepdims=True) * nvec          # keep in tangent plane
        ft_mag = np.sqrt((ft_tr ** 2).sum(1) + 1e-300)
        cone = self.mu * np.exp(-self.c_deg * self.Wp) * fn                  # Coulomb cone (Plesha wear)
        slip = (ft_mag > cone) & active
        scale = np.where(slip, cone / ft_mag, 1.0)
        Ft = ft_tr * scale[:, None]
        Ft[~active] = 0.0
        Fn = fn[:, None] * nvec
        f[self.top] = Fn + Ft
        diss = 0.0
        if commit:
            sp_new = self.sp.copy()                                          # sp s.t. Ft = -ktA(s_rel-sp)
            sp_new[active] = s_rel[active] + Ft[active] / ktA                # active stick -> ~same; slip -> flows
            sp_new[~active] = s_rel[~active]                                 # SEPARATED -> reset friction memory
            dsp = np.zeros_like(self.sp)
            dsp[active] = sp_new[active] - self.sp[active]                   # plastic slip only where in contact
            # frictional dissipation = |F_T| . |plastic slip|  (perfectly-plastic: |F_T|=cone during flow)
            dWp = np.abs((Ft * dsp).sum(1))
            self.sp = sp_new
            self.Wp = self.Wp + dWp
            self.Ft_prev = Ft.copy()
            diss = float(dWp.sum())
        Fc = f[self.top].sum(0)
        Wpen = 0.5 * float((self.eps_n * self.A * np.maximum(-gN, 0.0) ** 2).sum())
        s_el = (s_rel - self.sp)                                             # recoverable (stick) slip
        Wstick = 0.5 * ktA * float(((s_el ** 2).sum(1) * active).sum())      # tangential spring energy
        diag = dict(n_active=int(active.sum()), n_slip=int(slip.sum()),
                    Fx=float(Fc[0]), Fy=float(Fc[1]), Fz=float(Fc[2]), Fn=float(-Fc[2]),
                    pen_max=float((-gN[active]).max()) if active.any() else 0.0,
                    Wpen=Wpen, Wstick=Wstick, diss=diss)
        return f, diag

    def _contact_tangent(self, u, u_x, z_p, d):
        from scipy.sparse import csr_matrix
        U = u.reshape(self.N, 3)
        p = self.xyz0[self.top] + U[self.top]
        z_up, hx, hy, nz = self._surface(p[:, 0], p[:, 1], u_x, z_p, d)
        gN = (z_up - p[:, 2]) * (-nz)
        active = gN < 0.0
        nvec = np.stack([hx, hy, nz], 1)
        d3 = np.array([d[0], d[1], 0.0])
        s_rel = U[self.top] - u_x * d3[None, :]
        s_rel = s_rel - (s_rel * nvec).sum(1, keepdims=True) * nvec
        ft_tr = self.kt * self.A * (s_rel - self.sp)
        ft_tr = ft_tr - (ft_tr * nvec).sum(1, keepdims=True) * nvec
        ft_mag = np.sqrt((ft_tr ** 2).sum(1) + 1e-300)
        cone = self.mu * np.exp(-self.c_deg * self.Wp) * np.where(active, self.eps_n * (-gN) * self.A, 0.0)
        slip = (ft_mag > cone) & active
        kAn = self.eps_n * self.A
        kAt = self.kt * self.A
        I3 = np.eye(3)
        rows, cols, vals = [], [], []
        for k, nidx in enumerate(self.top):
            if not active[k]:
                continue
            nv = nvec[k]; P = I3 - np.outer(nv, nv)                          # tangent-plane projector
            kt_eff = 0.1 * kAt if slip[k] else kAt                           # reduced stiffness if slipping
            blk = kAn * np.outer(nv, nv) + kt_eff * P
            base = 3 * nidx
            for a in range(3):
                for b in range(3):
                    rows.append(base + a); cols.append(base + b); vals.append(blk[a, b])
        return csr_matrix((vals, (rows, cols)), shape=(3 * self.N, 3 * self.N))

    def solve_fixed(self, u_x, z_p, d, u0, max_iter=120, tol=1e-9):
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        ndof = 3 * self.N
        free = np.ones(ndof, bool)
        for nidx in self.bot:
            free[3 * nidx:3 * nidx + 3] = False
        Ksp = csr_matrix(self.K)
        scaleR = tol * (1 + self.eps_n * self.A)
        u = u0.copy().reshape(-1)
        rn = None
        for it in range(max_iter):
            f, diag = self.contact(u, u_x, z_p, d)
            R = self.K @ u - f.reshape(-1)
            rn = np.linalg.norm(R[free])
            if rn < scaleR:
                break
            Kc = self._contact_tangent(u, u_x, z_p, d)
            Kff = (Ksp + Kc)[free][:, free].tocsc()
            du = spsolve(Kff, -R[free])
            step, ok = 1.0, False
            for _ in range(30):
                ut = u.copy(); ut[free] += step * du
                ft, _ = self.contact(ut, u_x, z_p, d)
                rnt = np.linalg.norm((self.K @ ut - ft.reshape(-1))[free])
                if rnt < (1 - 1e-4 * step) * rn:
                    u, ok = ut, True; break
                step *= 0.5
            if not ok:
                break
        f, diag = self.contact(u, u_x, z_p, d)
        diag["resid"] = float(np.linalg.norm((self.K @ u - f.reshape(-1))[free]))
        diag["resid_rel"] = diag["resid"] / max(self.eps_n * self.A, 1e-12)
        return u, diag

    def elastic_energy(self, u):
        return 0.5 * float(u @ (self.K @ u))


def run_cyclic(dec, dk, mode="in_plane", n_cycles=3, amplitude=0.06, n_per_quarter=8, compress=0.6,
               amp=0.08, n_cells=10, mu=0.4, E=2.0e3, c_deg=0.0, cj=None, verbose=True):
    """Cyclic CNV shear (platen height fixed; normal stress evolves) with the full energy ledger."""
    if cj is None:
        cj = CyclicJoint(dec, dk, L=1.0, E=E, mu=mu, n_cells=n_cells, surf_amp=amp, c_deg=c_deg)
    else:
        cj.reset_state(); cj.mu = mu; cj.c_deg = c_deg
    d = np.asarray(MODES[mode], float)
    z_p = 1.0 - compress * amp                                              # CNV platen height (fixed)
    # triangle-wave shear schedule 0 -> +A -> -A -> 0 per cycle
    segs = []
    for _ in range(n_cycles):
        segs += [np.linspace(0, amplitude, n_per_quarter, endpoint=False),
                 np.linspace(amplitude, -amplitude, 2 * n_per_quarter, endpoint=False),
                 np.linspace(-amplitude, 0, n_per_quarter, endpoint=False)]
    sched = np.concatenate(segs + [np.array([0.0])])
    cyc_idx = np.concatenate([np.repeat(np.arange(n_cycles), 4 * n_per_quarter), [n_cycles - 1]])
    u = np.zeros(3 * cj.N)
    hist = {k: [] for k in ("u_par", "ux", "uy", "u_n", "t_N", "t_x", "t_y", "tau", "mu_app",
                            "W_ext", "W_fric", "W_el", "W_pen", "W_stick", "cycle", "n_active",
                            "n_slip", "resid")}
    W_ext = 0.0; W_fric = 0.0; Fc_prev = np.zeros(3)
    u_prev = u.copy(); ux_prev = 0.0
    for j, umag in enumerate(sched):
        u, sdiag = cj.solve_fixed(umag, z_p, d, u, max_iter=120)
        f, diag = cj.contact(u, umag, z_p, d, commit=True)                  # commit plastic slip
        diag["resid"] = sdiag["resid"]; diag["resid_rel"] = sdiag["resid_rel"]
        Fc = np.array([diag["Fx"], diag["Fy"], diag["Fz"]])
        tau = (Fc[0] * d[0] + Fc[1] * d[1]) / cj.A_nom
        t_N = diag["Fn"] / cj.A_nom
        # machine work: contact force on block . platen displacement, TRAPEZOIDAL (rigid upper moves d*du)
        d_uplaten = np.array([(umag - ux_prev) * d[0], (umag - ux_prev) * d[1], 0.0])
        W_ext += float(0.5 * (Fc_prev + Fc) @ d_uplaten)
        W_fric += diag["diss"]
        Fc_prev = Fc.copy()
        W_el = cj.elastic_energy(u); W_pen = diag["Wpen"]; W_stick = diag["Wstick"]
        hist["u_par"].append(float(umag)); hist["ux"].append(float(umag * d[0]))
        hist["uy"].append(float(umag * d[1])); hist["u_n"].append(float(diag["pen_max"]))
        hist["t_N"].append(float(t_N)); hist["t_x"].append(float(Fc[0] / cj.A_nom))
        hist["t_y"].append(float(Fc[1] / cj.A_nom)); hist["tau"].append(float(tau))
        hist["mu_app"].append(float(tau / max(t_N, 1e-9)))
        hist["W_ext"].append(W_ext); hist["W_fric"].append(W_fric)
        hist["W_el"].append(W_el); hist["W_pen"].append(W_pen); hist["W_stick"].append(W_stick)
        hist["cycle"].append(int(cyc_idx[j])); hist["n_active"].append(diag["n_active"])
        hist["n_slip"].append(diag["n_slip"]); hist["resid"].append(diag["resid"])
        u_prev = u.copy(); ux_prev = umag
        if verbose and j % max(1, len(sched) // 16) == 0:
            print(f"    j={j:3d} cyc={cyc_idx[j]} u={umag:+.4f} tau={tau:+.4f} tN={t_N:.3f} "
                  f"mu={tau/max(t_N,1e-9):+.3f} Wext={W_ext:.3e} Wfric={W_fric:.3e} "
                  f"nC={diag['n_active']} nS={diag['n_slip']} r={diag['resid']:.1e}")
    for k in hist:
        hist[k] = np.asarray(hist[k])
    return cj, hist


def energy_report(hist):
    """Per-cycle and global energy closure W_ext / (W_fric + dU_el + W_pen).  Returns a dict."""
    W = hist
    rep = {"per_cycle": []}
    for c in np.unique(W["cycle"]):
        idx = np.where(W["cycle"] == c)[0]
        i0, i1 = idx[0], idx[-1]
        d_ext = W["W_ext"][i1] - W["W_ext"][i0]
        d_fric = W["W_fric"][i1] - W["W_fric"][i0]
        d_el = W["W_el"][i1] - W["W_el"][i0]
        d_pen = W["W_pen"][i1] - W["W_pen"][i0]
        d_stk = W["W_stick"][i1] - W["W_stick"][i0]
        denom = d_fric + d_el + d_pen + d_stk
        rep["per_cycle"].append(dict(cycle=int(c), W_ext=float(d_ext), W_fric=float(d_fric),
                                     dU_el=float(d_el), dW_pen=float(d_pen), dW_stick=float(d_stk),
                                     ratio_full=float(d_ext / denom) if abs(denom) > 1e-30 else float("nan"),
                                     ratio_ext_fric=float(d_ext / d_fric) if abs(d_fric) > 1e-30 else float("nan")))
    recover = (W["W_el"][-1] - W["W_el"][0]) + (W["W_pen"][-1] - W["W_pen"][0]) + (W["W_stick"][-1] - W["W_stick"][0])
    rep["global_ratio_full"] = float(W["W_ext"][-1] / (W["W_fric"][-1] + recover + 1e-30))
    rep["global_ratio_ext_fric"] = float(W["W_ext"][-1] / (W["W_fric"][-1] + 1e-30))
    return rep


def _save(name, hist, params, meta=None):
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from joint_data_io import save_joint_history
    out = os.path.join(_ROOT, "runs", "rock_joint_3d", name)
    save_joint_history(out, {k: v for k, v in hist.items()}, params, meta=meta or {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="in_plane", choices=list(MODES) + ["all"])
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--n_cells", type=int, default=10)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--degrade", type=float, default=0.0, help="Plesha wear c_deg (0=off)")
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    print("=== training the rough-block decoder ===")
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=args.amp)     # noqa: E731
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=args.iters)
    print(f"  decoder recon RMSE {rmse:.3e}")
    modes = list(MODES) if args.mode == "all" else [args.mode]
    cj = None
    for m in modes:
        print(f"\n=== GENUINE cyclic shear: mode={m} cycles={args.cycles} (CNV) ===")
        cj, hist = run_cyclic(dec, dk, mode=m, n_cycles=args.cycles, amp=args.amp, n_cells=args.n_cells,
                              mu=args.mu, c_deg=args.degrade, cj=cj)
        rep = energy_report(hist)
        print(f"  peak |mu_app|={np.abs(hist['mu_app']).max():.4f}  "
              f"max resid_rel={ (hist['resid']/(cj.eps_n*cj.A)).max():.2e}")
        print(f"  ENERGY CLOSURE  global W_ext/(W_fric+dU_el+W_pen) = {rep['global_ratio_full']:.4f}  "
              f"| W_ext/W_fric = {rep['global_ratio_ext_fric']:.4f}")
        for pc in rep["per_cycle"]:
            print(f"    cycle {pc['cycle']}: W_ext={pc['W_ext']:.3e} W_fric={pc['W_fric']:.3e} "
                  f"dU_el={pc['dU_el']:+.2e} dW_pen={pc['dW_pen']:+.2e} ratio_full={pc['ratio_full']:.3f}")
        out = _save(f"cyclic_genuine_{m}_CNV", hist,
                    dict(mode=m, cycles=args.cycles, mu=args.mu, amp=args.amp, c_deg=args.degrade,
                         protocol="CNV", recon_rmse=rmse, energy=rep),
                    meta={"per_cycle": rep["per_cycle"], "energy_global": dict(
                        ratio_full=rep["global_ratio_full"], ratio_ext_fric=rep["global_ratio_ext_fric"])})
        print(f"  saved -> {out}")
    # degradation run for the decay panel (in_plane)
    if args.degrade == 0.0 and "in_plane" in modes:
        print("\n=== degradation run (Plesha wear c_deg=6) in_plane ===")
        cjd, histd = run_cyclic(dec, dk, mode="in_plane", n_cycles=4, amp=args.amp, n_cells=args.n_cells,
                                mu=args.mu, c_deg=6.0, verbose=False)
        repd = energy_report(histd)
        _save("cyclic_genuine_in_plane_degrade_CNV", histd,
              dict(mode="in_plane", cycles=4, mu=args.mu, amp=args.amp, c_deg=6.0, protocol="CNV"),
              meta={"per_cycle": repd["per_cycle"]})
        peaks = [np.abs(histd["mu_app"][histd["cycle"] == c]).max() for c in np.unique(histd["cycle"])]
        print(f"  peak |mu_app| per cycle: {['%.3f' % p for p in peaks]}")


if __name__ == "__main__":
    main()
