#!/usr/bin/env python3
"""Direct shear of a rough rock joint — the neural-atlas capstone (CMAME).

Two single-valued rough surfaces (footwall fixed, hangingwall driven) are held in contact under a
constant normal STRESS and sheared.  Each surface is carried by a learned 1-D height chart
``h_theta(x)`` (``solvers/contact/profile_chart_2d.py`` — random-Fourier-feature MLP), NOT an ambient
level set.  Friction is **plain Coulomb** (constant mu = tan(phi_b)); there is no dilatancy law.  The
dilation (normal opening) and the apparent friction angle are therefore EMERGENT outputs of the
resolved asperity geometry — which is exactly why representing the geometry faithfully (a chart that
resolves every roughness scale) versus smoothing it (a neural SDF, spectral-bias-limited) changes the
engineering answer.

Mechanics (quasi-static incremental direct shear, per unit out-of-plane thickness):
  * The hangingwall is rigid, sheared horizontally by a prescribed u_x and free to move vertically
    (the dilation DOF y).  Its bottom surface in world coords is z(X) = y + h_U(X - u_x).
  * Contact = node-to-surface penalty with the FOOTWALL as master.  At each upper node the normal
    gap is the vertical offset projected on the footwall normal; penalty pressure eps_n*<-gap_n>.
  * Friction: fully-mobilized Coulomb opposing the (forward) slip, |f_t| = mu*f_n along the footwall
    tangent — standard direct-shear assumption (monotonic shear, every contact sliding).
  * Each shear increment: solve the vertical DOF y so the net vertical contact force balances the
    applied normal load W = sigma_n * L_overlap (normal equilibrium), then read the shear resistance
    tau = -F_x / L_overlap and mu_app = tau / sigma_n.

VERIFICATION ANCHOR (Patton).  For two mating sawtooth faces of asperity angle i, this scheme
emergently yields the Patton law  mu_app = tan(phi_b + i)  (derived: (tan i + mu)/(1 - mu tan i) with
mu = tan phi_b).  ``--sawtooth`` runs that check against the closed form — a real L1 reference, not an
imposed law.  The real Inada surface (``--surface rough``) has no closed form; its curves are the
showcase, validated by the sawtooth anchor + mesh/penalty convergence.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_shear.py --sawtooth
      python3 benchmarks/contact/cv_numerical/rock_joint_shear.py --surface rough
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(_ROOT, "data", "inada_joint")


# --------------------------------------------------------------------------------------------------
# surface baking: evaluate any height representation to numpy grids (h, h') once, then contact is
# pure-numpy interpolation (fast inner loop, the chart is "baked").
# --------------------------------------------------------------------------------------------------
def bake_height(height_module, x_grid: np.ndarray) -> dict:
    """Bake a torch height module h(x) to numpy (x, h, h') via one autograd pass."""
    import torch
    from solvers.contact.profile_chart_2d import height_and_grad
    xt = torch.from_numpy(np.asarray(x_grid, float))
    h, dh = height_and_grad(xt, height_module)
    return dict(x=np.asarray(x_grid, float), h=h.numpy(), hp=dh.numpy())


def bake_samples(x: np.ndarray, z: np.ndarray) -> dict:
    """Bake raw sampled topography (x, z) — gradient by central differences (no chart)."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    hp = np.gradient(z, x)
    return dict(x=x, h=z, hp=hp)


def _interp(grid: dict, X: np.ndarray):
    """Footwall height + slope at world positions X (outside-range nodes flagged invalid)."""
    h = np.interp(X, grid["x"], grid["h"])
    hp = np.interp(X, grid["x"], grid["hp"])
    valid = (X >= grid["x"][0]) & (X <= grid["x"][-1])
    return h, hp, valid


# --------------------------------------------------------------------------------------------------
# contact: footwall = master.  Upper nodes are the hangingwall grid points (material xi).
# --------------------------------------------------------------------------------------------------
def contact_forces(yU: float, ux: float, lower: dict, upper: dict, eps_n: float, mu: float):
    """Net (F_x, F_z) on the hangingwall + diagnostics, at vertical position yU and shear offset ux.

    Footwall is master: normal/tangent from the footwall.  Fully-mobilized forward-slip Coulomb."""
    xi = upper["x"]                                   # hangingwall material coords
    dxi = np.gradient(xi)                             # tributary width per node
    X = xi + ux                                       # world x of each upper node
    Z = yU + upper["h"]                               # world z of each upper node (bottom surface)
    hL, hpL, valid = _interp(lower, X)
    # footwall unit normal (up) and tangent (forward)
    sec = np.sqrt(1.0 + hpL ** 2)
    nz = 1.0 / sec; nx = -hpL / sec                   # n = (-h', 1)/sec
    tx = 1.0 / sec; tz = hpL / sec                    # t = (1,  h')/sec
    gap_n = (Z - hL) * nz                             # normal gap (vertical offset projected on n)
    active = valid & (gap_n < 0.0)
    fn = np.where(active, eps_n * (-gap_n) * dxi, 0.0)   # >=0 normal magnitude per node
    # node forces: normal along n, friction -mu*fn along t (opposing forward slip)
    Fx = np.sum(fn * nx - mu * fn * tx)
    Fz = np.sum(fn * nz - mu * fn * tz)
    diag = dict(n_active=int(active.sum()), fn_sum=float(fn.sum()),
                pen_max=float((-gap_n[active]).max()) if active.any() else 0.0)
    return Fx, Fz, diag


def solve_y_equilibrium(ux: float, lower: dict, upper: dict, eps_n: float, mu: float, W: float,
                        y_bracket) -> float:
    """Bisection: find yU such that the vertical contact force F_z(yU) == W (normal equilibrium).
    F_z is monotone decreasing in yU (higher => less penetration => less force)."""
    ylo, yhi = y_bracket                              # ylo: deep penetration (F_z>W); yhi: free (F_z<W)
    for _ in range(60):
        ym = 0.5 * (ylo + yhi)
        _, Fz, _ = contact_forces(ym, ux, lower, upper, eps_n, mu)
        if Fz > W:                                    # too much force -> raise the block
            ylo = ym
        else:
            yhi = ym
    return 0.5 * (ylo + yhi)


def run_shear(lower: dict, upper: dict, sigma_n: float = 1.0, mu: float = 0.3,
              shear_total: float = 8.0, n_inc: int = 160, eps_n: float = 5.0e3,
              verbose: bool = False) -> dict:
    """Quasi-static incremental direct shear.  Returns history arrays + scalar summaries."""
    x0, x1 = lower["x"][0], lower["x"][-1]
    span = x1 - x0
    amp = max(upper["h"].max() - upper["h"].min(), lower["h"].max() - lower["h"].min())
    y_lo = float(lower["h"].max() - upper["h"].min() - 2.0 * amp)   # deep-penetration bracket
    y_hi = float(lower["h"].max() - upper["h"].min() + 2.0 * amp)   # separated bracket

    uxs = np.linspace(0.0, shear_total, n_inc)
    rec = {k: [] for k in ("ux", "y", "dilation", "tau", "sigma_n", "mu_app",
                            "n_active", "pen_max", "L_overlap")}
    y0 = None
    for j, ux in enumerate(uxs):
        L_ov = max(span - abs(ux), 0.25 * span)        # current overlap length
        W = sigma_n * L_ov                             # constant normal STRESS -> load over overlap
        y = solve_y_equilibrium(ux, lower, upper, eps_n, mu, W, (y_lo, y_hi))
        Fx, Fz, d = contact_forces(y, ux, lower, upper, eps_n, mu)
        if y0 is None:
            y0 = y
        tau = -Fx / L_ov
        rec["ux"].append(float(ux)); rec["y"].append(float(y)); rec["dilation"].append(float(y - y0))
        rec["tau"].append(float(tau)); rec["sigma_n"].append(float(sigma_n))
        rec["mu_app"].append(float(tau / sigma_n)); rec["n_active"].append(d["n_active"])
        rec["pen_max"].append(d["pen_max"]); rec["L_overlap"].append(float(L_ov))
        if verbose and (j % 20 == 0 or j == n_inc - 1):
            print(f"    ux={ux:6.3f}  dil={y-y0:+.4f}  mu_app={tau/sigma_n:6.4f}  "
                  f"nC={d['n_active']:4d}  pen={d['pen_max']:.2e}")
    for k in rec:
        rec[k] = np.asarray(rec[k])
    # steady-state (last 40%) and peak summaries
    ss = slice(int(0.6 * n_inc), n_inc)
    rec_summary = dict(
        mu_app_peak=float(rec["mu_app"].max()),
        mu_app_steady=float(np.mean(rec["mu_app"][ss])),
        phi_app_peak_deg=float(math.degrees(math.atan(rec["mu_app"].max()))),
        dilation_total=float(rec["dilation"][-1]),
        dilation_peak=float(rec["dilation"].max()),
        pen_max=float(rec["pen_max"].max()),
    )
    return dict(history=rec, summary=rec_summary,
                params=dict(sigma_n=sigma_n, mu=mu, shear_total=shear_total, n_inc=n_inc,
                            eps_n=eps_n))


# --------------------------------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------------------------------
def _save(name: str, payload: dict):
    out_dir = os.path.join(_ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    ser = {}
    for k, v in payload.items():
        if k == "history":
            ser[k] = {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
        else:
            ser[k] = v
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(ser, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload.get("summary", {}), f, indent=2)
    return out_dir


def run_sawtooth_patton(angle_deg=20.0, mu=0.3, wavelength=5.0):
    """L1 verification: mating sawtooth shear -> emergent mu_app vs Patton tan(phi_b + i)."""
    import torch  # noqa
    from solvers.contact.profile_chart_2d import AnalyticSawtooth1D
    x = np.linspace(0.0, 50.0, 6000)
    saw = AnalyticSawtooth1D(wavelength=wavelength, angle_deg=angle_deg, x_lo=0.0, x_hi=50.0).double()
    g = bake_height(saw, x)
    lower = g
    upper = dict(x=g["x"].copy(), h=g["h"].copy(), hp=g["hp"].copy())   # perfectly mated
    # shear by half a wavelength (a full ride-up flank) to reach the Patton plateau
    res = run_shear(lower, upper, sigma_n=1.0, mu=mu, shear_total=0.45 * wavelength,
                    n_inc=120, eps_n=2.0e4, verbose=True)
    phi_b = math.degrees(math.atan(mu))
    patton = math.tan(math.radians(phi_b + angle_deg))
    res["summary"]["patton_mu_pred"] = float(patton)
    res["summary"]["patton_phi_b_deg"] = float(phi_b)
    res["summary"]["asperity_angle_deg"] = float(angle_deg)
    res["summary"]["mu_app_relerr_vs_patton"] = float(
        abs(res["summary"]["mu_app_peak"] - patton) / patton)
    out = _save("rock_joint_sawtooth", res)
    s = res["summary"]
    print(f"\n  Patton check (i={angle_deg} deg, phi_b={phi_b:.2f} deg):")
    print(f"    mu_app peak (emergent) = {s['mu_app_peak']:.4f}")
    print(f"    tan(phi_b + i)  (Patton) = {patton:.4f}")
    print(f"    rel. error = {s['mu_app_relerr_vs_patton']*100:.2f}%")
    print(f"  saved -> {out}")
    return res


def _load_inada_profile(tag: str):
    f = os.path.join(DATA, f"inada_{tag}_profile.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} — run postprocessing/characterize_inada_joint.py first")
    d = np.load(f, allow_pickle=True)
    return d["x_mm"].astype(float), d["footwall_mm"].astype(float), d["hangingwall_mm"].astype(float)


def run_real_surface(tag="rough", mated=False, mu=0.3, sigma_n=1.0, shear_total=8.0, use_chart=True):
    """Direct shear of the real Inada profile (footwall + hangingwall, or perfectly-mated)."""
    x, foot, hang = _load_inada_profile(tag)
    if use_chart:
        from solvers.contact.profile_chart_2d import fit_height_chart
        print(f"  fitting footwall height chart ({len(x)} pts)...")
        cL, rL = fit_height_chart(x, foot, f_max=1200.0, n_freq=192, iters=3500)
        lower = bake_height(cL, x)
        if mated:
            upper = dict(x=x.copy(), h=foot.copy(), hp=lower["hp"].copy())
        else:
            print(f"  fitting hangingwall height chart...")
            cU, rU = fit_height_chart(x, hang, f_max=1200.0, n_freq=192, iters=3500)
            upper = bake_height(cU, x)
        chart_rmse = float(rL)
    else:
        lower = bake_samples(x, foot)
        upper = bake_samples(x, foot if mated else hang)
        chart_rmse = 0.0
    res = run_shear(lower, upper, sigma_n=sigma_n, mu=mu, shear_total=shear_total,
                    n_inc=160, eps_n=5.0e3, verbose=True)
    res["summary"]["surface"] = tag
    res["summary"]["mated"] = bool(mated)
    res["summary"]["chart_rmse_mm"] = chart_rmse
    res["summary"]["mu_base"] = float(mu)
    res["summary"]["phi_b_deg"] = float(math.degrees(math.atan(mu)))
    name = f"rock_joint_{tag}" + ("_mated" if mated else "_real")
    out = _save(name, res)
    s = res["summary"]
    print(f"\n  {tag} joint ({'mated' if mated else 'real foot/hang'}):")
    print(f"    peak mu_app   = {s['mu_app_peak']:.4f}  (phi_app = {s['phi_app_peak_deg']:.2f} deg)")
    print(f"    steady mu_app = {s['mu_app_steady']:.4f}   base mu = {mu} (phi_b={s['phi_b_deg']:.2f})")
    print(f"    total dilation= {s['dilation_total']:+.4f} mm")
    print(f"  saved -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sawtooth", action="store_true", help="Patton sawtooth L1 verification")
    ap.add_argument("--angle", type=float, default=20.0, help="sawtooth asperity angle (deg)")
    ap.add_argument("--surface", choices=["rough", "smooth"], help="run the real Inada surface")
    ap.add_argument("--mated", action="store_true", help="perfectly-mated (upper=footwall)")
    ap.add_argument("--no-chart", action="store_true", help="use raw samples, not the neural chart")
    ap.add_argument("--mu", type=float, default=0.3)
    args = ap.parse_args()

    if args.sawtooth:
        run_sawtooth_patton(angle_deg=args.angle, mu=args.mu)
    if args.surface:
        run_real_surface(tag=args.surface, mated=args.mated, mu=args.mu,
                         use_chart=not args.no_chart)
    if not args.sawtooth and not args.surface:
        print("nothing to do — pass --sawtooth and/or --surface rough|smooth")


if __name__ == "__main__":
    main()
