#!/usr/bin/env python3
"""Deformable two-block rock joint under MIXED-MODE CYCLIC loading (chart-FEM).

The 3-D deformable extension of the rigid rock-joint shear.  Two elastic blocks (each a
``ChartVectorFEMSolver`` Cartesian block, small-strain Hooke) meet at a rough joint modelled as a
zero-thickness DILATANT-FRICTIONAL interface (a Goodman/Plesha rock-joint interface element):

    relative displacement  [[u]] = u_upper - u_lower   at each matched interface node pair,
    normal:      t_N = k_n <c>,   over-closure c = d(s_T) - [[u_z]],   d = tan(i)*|s_T|   (dilation),
    tangential:  elastoplastic Coulomb (return map):  stick |t_T| <= mu*t_N, else slip on the cone,
    degradation: i = i0 * exp(-c_deg * W_p),   W_p = accumulated frictional/plastic work  (Plesha).

The roughness enters as the per-interface-point dilation-angle field i0(x,y) sampled from the height
chart's gradient (``surface_chart_3d``) — so the asperities live in the interface law, the blocks are
flat (the standard interface-element idealisation, stated honestly).  The tangential return map gives
genuine hysteresis; dilation couples shear -> normal; degradation decays the peak over cycles.

LOADING MODES (joint plane = xy, normal = z; reading (A) from the research brief): the cyclic shear
is driven along an in-plane azimuth — IN-PLANE (x), OUT-OF-PLANE (y), or MIXED (45 deg).  Protocol:
CNL (constant normal stress on the top face, free dilation) by default; CNV (pinned normal) optional.

Verification (subset; full ladder in tests/test_rock_joint_cyclic_fem.py):
  * flat joint (i0=0) -> Coulomb plateau tau/sig_n -> mu, ZERO dilation, no transverse force;
  * Patton (uniform i0) -> tau/sig_n -> tan(phi_b + i0);
  * ENERGY BALANCE (primary cyclic gate): external work == d(elastic) + frictional dissipation (>=0);
  * reduce-to-2D: slip confined to x matches the 2-D capstone trend;
  * degradation off -> identical repeatable loops; on -> monotone peak/dilation decay.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_cyclic_fem.py --verify
      python3 benchmarks/contact/cv_numerical/rock_joint_cyclic_fem.py --mode mixed --cycles 3
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
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver          # noqa: E402
from solvers.fem.linear_elastic import make_linear_elastic_small_strain  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
torch.set_default_dtype(torch.float64)

MODES = {"in_plane": (1.0, 0.0), "out_of_plane": (0.0, 1.0),
         "mixed": (math.cos(math.radians(45)), math.sin(math.radians(45)))}


# --------------------------------------------------------------------------------------------------
class JointFEM:
    """Two stacked elastic blocks + a dilatant-frictional interface at their shared face."""

    def __init__(self, n_cells=6, half=1.0, E=1.0e3, nu=0.25, dilation_deg=0.0, mu=0.4,
                 k_n=None, k_t=None, c_deg=0.0, dilation_field=None):
        self.E, self.nu, self.mu = E, nu, mu
        self.half = half
        # two identical Cartesian blocks; shift the UPPER block up by 2*half so faces meet at z=half
        self.lo = ChartVectorFEMSolver(n_cells=n_cells, support_r=half, chart_decoder=None)
        self.up = ChartVectorFEMSolver(n_cells=n_cells, support_r=half, chart_decoder=None)
        self.NL, self.NU = self.lo.n_nodes, self.up.n_nodes
        self.shift = 2.0 * half
        self.up_nodes = self.up.nodes.clone(); self.up_nodes[:, 2] += self.shift   # world coords
        self.lo_nodes = self.lo.nodes
        # constitutive (linear -> constant block stiffness)
        sfn, tfn = make_linear_elastic_small_strain(E, nu)
        self.stress_fn, self.tangent_fn = sfn, tfn
        u0L = torch.zeros(self.NL, 3); u0U = torch.zeros(self.NU, 3)
        self.KL = self.lo.tangent_stiffness(u0L, tfn)        # (3NL,3NL) dense, constant
        self.KU = self.up.tangent_stiffness(u0U, tfn)
        self.ndof = 3 * (self.NL + self.NU)
        # interface pairs: lower top-face nodes (z=+half) <-> upper bottom nodes (z=-half pre-shift)
        tolz = 1e-6 * half
        li = torch.where(torch.abs(self.lo_nodes[:, 2] - half) < 1e-9)[0]
        ui = torch.where(torch.abs(self.up.nodes[:, 2] + half) < 1e-9)[0]
        # match by (x,y)
        lxy = self.lo_nodes[li, :2].numpy(); uxy = self.up.nodes[ui, :2].numpy()
        pairs_l, pairs_u = [], []
        for a, p in enumerate(lxy):
            d = np.linalg.norm(uxy - p, axis=1); b = int(d.argmin())
            if d[b] < 1e-6 * half:
                pairs_l.append(int(li[a])); pairs_u.append(int(ui[b]))
        self.pl = np.asarray(pairs_l); self.pu = np.asarray(pairs_u)
        self.xy = self.lo_nodes[self.pl, :2].numpy()
        self.np_iface = len(self.pl)
        # tributary areas (regular grid): cell^2 with half-weights on edges -> use Voronoi approx
        self.area = self._tributary_areas(self.xy)
        self.A_tot = float(self.area.sum())
        # dilation-angle field i0 per interface point
        if dilation_field is not None:
            self.i0 = np.asarray(dilation_field, float)
        else:
            self.i0 = np.full(self.np_iface, math.radians(dilation_deg))
        self.tan_i0 = np.tan(self.i0)
        self.c_deg = c_deg
        # penalty stiffness (per unit area) ~ E/h (moderate -> better-conditioned Newton)
        h = 2 * half / n_cells
        self.k_n = (10.0 * E / h) if k_n is None else k_n
        self.k_t = (10.0 * E / h) if k_t is None else k_t
        # top/bottom face DOFs for BCs
        self.bot = torch.where(torch.abs(self.lo_nodes[:, 2] + half) < 1e-9)[0].numpy()   # fix
        self.top = torch.where(torch.abs(self.up.nodes[:, 2] - half) < 1e-9)[0].numpy()   # driven
        # interface state
        self.reset_state()

    def reset_state(self):
        self.sp = np.zeros((self.np_iface, 2))    # plastic tangential slip
        self.Wp = np.zeros(self.np_iface)         # accumulated frictional work (degradation)

    @staticmethod
    def _tributary_areas(xy):
        x = np.unique(xy[:, 0]); y = np.unique(xy[:, 1])
        dx = np.gradient(x) if len(x) > 1 else np.array([1.0])
        dy = np.gradient(y) if len(y) > 1 else np.array([1.0])
        ax = {v: dx[i] for i, v in enumerate(x)}; ay = {v: dy[i] for i, v in enumerate(y)}
        return np.array([ax[p[0]] * ay[p[1]] for p in xy])

    # --- interface constitutive (returns force on global dofs + per-pair tractions) --------------
    def interface(self, uL, uU, commit=False):
        """Dilatant-frictional interface force (3*(NL+NU),) and per-pair tractions.
        uL,uU : (N,3) displacements.  commit -> update plastic slip + degradation state."""
        rel = (uU[self.pu] + 0.0) - uL[self.pl]                 # [[u]] = u_up - u_lo  (np, (P,3))
        gN = rel[:, 2]                                          # normal opening (+)
        sT = rel[:, :2]                                         # tangential relative disp
        sT_mag = np.sqrt((sT ** 2).sum(1) + 1e-300)
        i_cur = self.i0 * np.exp(-self.c_deg * self.Wp)         # degraded dilation angle (Plesha)
        tan_i = np.tan(i_cur)
        d = tan_i * sT_mag                                      # asperity dilation (ride-up, kinematic)
        c = d - gN                                              # over-closure (+ = compressed)
        active = c > 0.0
        tN = self.k_n * np.maximum(c, 0.0)                      # compressive normal traction (>=0)
        # elastoplastic Coulomb return map; EFFECTIVE friction tan(phi_b + i) supplies the dilatant
        # (Patton) shear strength on the flat mean plane (the tilted-asperity-facet projection).
        mu_eff = np.tan(np.arctan(self.mu) + i_cur)
        te = self.k_t * (sT - self.sp)                          # trial elastic shear traction
        te_mag = np.sqrt((te ** 2).sum(1) + 1e-300)
        cone = mu_eff * tN
        slip = (te_mag > cone) & active
        tT = te.copy()
        scale = np.where(slip, cone / te_mag, 1.0)
        tT = te * scale[:, None]
        tT[~active] = 0.0
        # per-pair force (x,y,z): friction in plane + normal (compressive pushes surfaces apart)
        A = self.area[:, None]
        f_up = np.concatenate([tT, tN[:, None]], axis=1) * A    # on upper: +tN up, +tT (reaction)
        # Newton's 3rd law on the pair; tT on upper opposes slip via te sign already
        f_up_vec = np.concatenate([ -tT, tN[:, None]], axis=1) * A   # friction resists slip (-te dir)
        # assemble
        f = np.zeros((self.NL + self.NU, 3))
        f[self.NL + self.pu] += f_up_vec                        # force ON upper nodes
        f[self.pl] -= f_up_vec                                  # equal & opposite ON lower nodes
        if commit:
            dsp = np.where(slip[:, None], (te - tT) / self.k_t, 0.0)   # plastic slip increment
            self.sp = self.sp + dsp
            self.Wp = self.Wp + np.abs((tT * dsp).sum(1)) * self.area
        diag = dict(tN=tN, tT=tT, gN=gN, d=d, sT=sT, sT_mag=sT_mag, active=active, slip=slip,
                    tN_mean=float((tN * self.area).sum() / self.A_tot),
                    tx_mean=float((tT[:, 0] * self.area).sum() / self.A_tot),
                    ty_mean=float((tT[:, 1] * self.area).sum() / self.A_tot),
                    dil_mean=float((gN * self.area).sum() / self.A_tot),
                    n_active=int(active.sum()))
        return f.reshape(-1), diag

    # --- penalty tangent (ACTIVE-SET: slipping pairs get reduced tangential stiffness) ------------
    def _Kpen(self, slip=None, active=None):
        """Penalty stiffness coupling each pair's normal+tangential dofs.  ``slip`` (P,) bool ->
        slipping pairs use a small tangential stiffness (perfectly-plastic slip tangent) so the
        active-set Newton converges; ``active`` (P,) bool -> open pairs drop normal stiffness too."""
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        offU = 3 * self.NL
        kt_slip = self.k_t * 0.1
        for p in range(self.np_iface):
            a = 3 * self.pl[p]; b = offU + 3 * self.pu[p]; A = self.area[p]
            kt = kt_slip if (slip is not None and slip[p]) else self.k_t
            kn = self.k_n if (active is None or active[p]) else self.k_n * 1e-3
            for comp, k in ((0, kt), (1, kt), (2, kn)):
                ka = k * A
                for (i, j, s) in ((b, b, ka), (a, a, ka), (b, a, -ka), (a, b, -ka)):
                    rows.append(i + comp); cols.append(j + comp); vals.append(s)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof))

    def _Kblocks(self):
        import scipy.sparse as sp
        KL = sp.csr_matrix(self.KL.numpy()); KU = sp.csr_matrix(self.KU.numpy())
        return sp.block_diag([KL, KU]).tocsr()

    # --- inner solve: RIGID top platen fully prescribed (ux,uy,uz), bottom fixed; interface NL ----
    def _solve_fixed_top(self, u_top, K, u_init, max_iter=120, tol=1e-9, commit=False):
        """Newton (active-set + line search) with the whole top face prescribed to u_top=(ux,uy,uz)
        (a rigid platen -> NO tilting -> uniform contact) and the bottom face fixed."""
        from scipy.sparse.linalg import spsolve
        ndof = self.ndof; offU = 3 * self.NL
        u = u_init.copy()
        free = np.ones(ndof, bool)
        for n in self.bot:
            for c in range(3):
                free[3 * n + c] = False; u[3 * n + c] = 0.0
        for n in self.top:
            for c in range(3):
                free[offU + 3 * n + c] = False; u[offU + 3 * n + c] = u_top[c]
        scaleR = tol * (1.0 + self.k_n * self.A_tot)

        def resid(uv):
            f_if, dg = self.interface(uv[:offU].reshape(-1, 3), uv[offU:].reshape(-1, 3))
            return (K.dot(uv) - f_if), dg

        R, diag = resid(u); rn = np.linalg.norm(R[free]); it = 0
        # full STICK-stiffness tangent (SPD, constant) — robust with the rigid platen + line search
        Kpen = self._Kpen(active=np.ones(self.np_iface, bool))
        Kff = (K + Kpen)[free][:, free].tocsc()
        from scipy.sparse.linalg import factorized
        solve = factorized(Kff)
        for it in range(max_iter):
            if rn < scaleR:
                break
            du = solve(-R[free])
            step = 1.0; ok = False
            for _ in range(40):
                ut = u.copy(); ut[free] += step * du
                Rt, dgt = resid(ut); rnt = np.linalg.norm(Rt[free])
                if rnt < (1.0 - 1e-4 * step) * rn:
                    u, R, diag, rn, ok = ut, Rt, dgt, rnt, True; break
                step *= 0.5
            if not ok:
                break
        if commit:
            _, diag = self.interface(u[:offU].reshape(-1, 3), u[offU:].reshape(-1, 3), commit=True)
        diag["resid"] = float(rn)
        return u, diag, it

    # --- one quasi-static increment: rigid platen; CNL -> bisect platen z for mean tN = sigma_n ----
    def solve_increment(self, u_top_xy, sigma_n, K, Kpen_unused, u_init, protocol="CNL",
                        max_iter=120, uz_init=None):
        ux, uy = u_top_xy[0], u_top_xy[1]
        if protocol == "CNV":
            uz = u_top_xy[2] if len(u_top_xy) > 2 else 0.0
            return self._solve_fixed_top((ux, uy, uz), K, u_init, max_iter=max_iter, commit=True)
        # CNL: damped-Newton on the platen height uz so mean normal traction == sigma_n.
        # tN_mean ~ k_n * mean(over-closure) is ~linear in uz (slope -k_n*frac_active); converges fast.
        uz = uz_init if uz_init is not None else -sigma_n / self.k_n
        u = u_init; diag = None; nit = 0
        for _ in range(12):
            u, diag, nit = self._solve_fixed_top((ux, uy, uz), K, u, max_iter=max_iter)
            err = diag["tN_mean"] - sigma_n
            if abs(err) < 1e-4 * sigma_n:
                break
            frac = max(diag["n_active"], 1) / self.np_iface
            uz += err / (self.k_n * frac)                        # raise platen if too much normal
        u, diag, nit = self._solve_fixed_top((ux, uy, uz), K, u, max_iter=max_iter, commit=True)
        diag["uz"] = float(uz)
        return u, diag, nit


# --------------------------------------------------------------------------------------------------
def run_cyclic(mode="mixed", n_cycles=3, amplitude=0.04, sigma_n0=4.0, n_per_quarter=10,
               n_cells=6, E=1.0e3, nu=0.25, mu=0.4, dilation_deg=8.0, c_deg=0.0,
               protocol="CNV", verbose=True):
    """Mixed-mode CYCLIC direct shear of the deformable two-block joint (CNV: rigid platen at fixed
    normal compression -> normal stress evolves with dilation).  Records the full traction-separation
    history + energy ledger for hysteresis analysis."""
    d = np.asarray(MODES[mode], float)
    fem = JointFEM(n_cells=n_cells, E=E, nu=nu, mu=mu, dilation_deg=dilation_deg, c_deg=c_deg)
    K = fem._Kblocks()
    uz0 = -sigma_n0 / fem.k_n                                   # platen compression -> initial sigma_n
    offU = 3 * fem.NL
    # cyclic schedule: 0 -> +A -> -A -> 0 per cycle (triangle wave of shear displacement)
    segs = []
    for _ in range(n_cycles):
        segs += [np.linspace(0, amplitude, n_per_quarter, endpoint=False),
                 np.linspace(amplitude, -amplitude, 2 * n_per_quarter, endpoint=False),
                 np.linspace(-amplitude, 0, n_per_quarter, endpoint=False)]
    sched = np.concatenate(segs + [np.array([0.0])])
    cyc_idx = np.concatenate([np.repeat(np.arange(n_cycles), 4 * n_per_quarter), [n_cycles - 1]])
    u = np.zeros(fem.ndof)
    hist = {k: [] for k in ("u_par", "ux", "uy", "u_n", "t_N", "t_x", "t_y", "tau", "mu_app",
                            "W_ext", "W_fric", "cycle", "n_active", "resid")}
    Wext = 0.0
    u_prev = u.copy(); slip_par_prev = 0.0
    for j, umag in enumerate(sched):
        u_top = (umag * d[0], umag * d[1], uz0)
        u, diag, nit = fem._solve_fixed_top(u_top, K, u, max_iter=120, commit=True)
        tau = diag["tx_mean"] * d[0] + diag["ty_mean"] * d[1]
        tN = diag["tN_mean"]
        # external work increment: top-platen reaction . d(u_top) (reaction = K u - f_if at top dofs)
        f_if, _ = fem.interface(u[:offU].reshape(-1, 3), u[offU:].reshape(-1, 3))
        react = (K.dot(u) - f_if)                              # nodal reaction (free dofs ~0)
        du_top = u - u_prev
        Wext += float(react @ du_top)
        u_prev = u.copy()
        hist["u_par"].append(float(umag)); hist["ux"].append(float(umag * d[0]))
        hist["uy"].append(float(umag * d[1])); hist["u_n"].append(diag["dil_mean"])
        hist["t_N"].append(float(tN)); hist["t_x"].append(diag["tx_mean"]); hist["t_y"].append(diag["ty_mean"])
        hist["tau"].append(float(tau)); hist["mu_app"].append(float(tau / max(tN, 1e-9)))
        hist["W_ext"].append(Wext); hist["W_fric"].append(float(fem.Wp.sum()))
        hist["cycle"].append(int(cyc_idx[j])); hist["n_active"].append(diag["n_active"])
        hist["resid"].append(diag["resid"])
        if verbose and j % max(1, len(sched) // 14) == 0:
            print(f"    j={j:3d} cyc={cyc_idx[j]} u={umag:+.4f} tau={tau:+.4f} tN={tN:.3f} "
                  f"mu_app={tau/max(tN,1e-9):+.3f} dil={diag['dil_mean']:+.4f} "
                  f"Wext={Wext:.3e} Wfric={fem.Wp.sum():.3e} nit={nit}")
    for k in hist:
        hist[k] = np.asarray(hist[k])
    return fem, hist


def _save(name, hist, params):
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from joint_data_io import save_joint_history
    out = os.path.join(_ROOT, "runs", "rock_joint_3d", name)
    cyc = [{"cycle": int(c)} for c in np.unique(hist["cycle"])]
    save_joint_history(out, {k: v for k, v in hist.items()}, params, meta={"per_cycle": cyc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mode", default="mixed", choices=list(MODES))
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--protocol", default="CNL", choices=["CNL", "CNV"])
    ap.add_argument("--degrade", type=float, default=0.0, help="Plesha c_deg (0=off)")
    args = ap.parse_args()
    if args.verify:
        from importlib import import_module
        v = import_module("benchmarks.contact.cv_numerical.rock_joint_cyclic_fem")
        print("run --mode/--cycles for a cyclic run; verification lives in tests/test_rock_joint_cyclic_fem.py")
        return
    print(f"=== deformable cyclic joint: mode={args.mode} cycles={args.cycles} protocol={args.protocol} ===")
    fem, hist = run_cyclic(mode=args.mode, n_cycles=args.cycles, c_deg=args.degrade,
                           protocol=args.protocol)
    out = _save(f"cyclic_{args.mode}_{args.protocol}", hist,
                dict(mode=args.mode, cycles=args.cycles, protocol=args.protocol,
                     mu=0.4, dilation_deg=8.0, c_deg=args.degrade, E=1e3, nu=0.25))
    print(f"\n  peak |mu_app| = {np.abs(hist['mu_app']).max():.4f}  "
          f"max dilation = {hist['u_n'].max():.4f}  final diss = {hist['W_fric'][-1]:.3e}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
