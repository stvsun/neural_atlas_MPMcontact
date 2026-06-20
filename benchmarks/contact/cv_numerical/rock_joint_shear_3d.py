#!/usr/bin/env python3
"""3-D direct shear of a rough rock joint — in-plane / out-of-plane / mixed-mode (rigid blocks).

The 3-D extension of ``rock_joint_shear.py``.  Two single-valued rough surfaces z=h(x,y)
(footwall fixed, hangingwall driven) are held in contact under constant normal stress and sheared
by an in-plane displacement VECTOR u=(u_x,u_y).  Each surface is a learned 2-D height chart
(``solvers/contact/surface_chart_3d.py``) baked to a grid; contact is node-to-surface penalty with
the footwall as master; friction is fully-mobilized Coulomb opposing the in-plane slip (now a 2-D
tangential vector).  Dilation and the apparent friction angle are EMERGENT from the resolved 3-D
asperity geometry — no dilatancy law.

LOADING MODES (shear-direction unit vector d in the joint plane; joint plane = xy, normal = z):
  * IN-PLANE shear   — d = (1,0): slide along x, within the x-z observation plane.
  * OUT-OF-PLANE     — d = (0,1): slide along y, perpendicular to the x-z plane (anti-plane, Mode-III-like).
  * MIXED-MODE       — d = (cos a, sin a): a combined in-plane direction (default 45 deg).
For an anisotropic (ridged) surface these differ; for the real Inada surface they probe roughness
anisotropy.  Output: traction vector (T_x,T_y), its components parallel/perpendicular to d, dilation,
and mu_app vs shear displacement — stored as runs/<name>/history.json.

VERIFICATION ANCHOR.  On a parallel-RIDGED sawtooth (``RidgedSawtooth3D``): shearing ACROSS the
ridges (in-plane, x) emergently gives Patton mu_app = tan(phi_b+i) with dilation rate tan(i);
shearing ALONG the ridges (out-of-plane, y) gives pure Coulomb mu with ZERO dilation.  A clean,
closed-form in-plane-vs-out-of-plane anisotropy check (``--ridged``).

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_shear_3d.py --ridged
      python3 benchmarks/contact/cv_numerical/rock_joint_shear_3d.py --surface rough --mode all
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
RAW = os.path.join(_ROOT, "downloads", "inada_granite")
DX_MM = 0.0234

MODES = {"in_plane": (1.0, 0.0), "out_of_plane": (0.0, 1.0),
         "mixed": (math.cos(math.radians(45)), math.sin(math.radians(45)))}


# --------------------------------------------------------------------------------------------------
# baking: a surface -> regular grid (h, h_x, h_y) for fast bilinear contact queries
# --------------------------------------------------------------------------------------------------
def bake_module(height_module, x: np.ndarray, y: np.ndarray) -> dict:
    """Bake a torch height module h(x,y) on the lattice (x,y) -> grids (h, hx, hy) via autograd."""
    import torch
    from solvers.contact.surface_chart_3d import height_and_grad
    XX, YY = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([XX.ravel(), YY.ravel()], 1)
    h, gh = height_and_grad(torch.from_numpy(xy), height_module)
    h = h.numpy().reshape(XX.shape)
    hx = gh[:, 0].numpy().reshape(XX.shape); hy = gh[:, 1].numpy().reshape(XX.shape)
    return _grid(x, y, h, hx, hy)


def bake_samples(x: np.ndarray, y: np.ndarray, H: np.ndarray) -> dict:
    """Bake a raw 2-D height map H[i,j]=z(x_i,y_j); gradient by finite differences."""
    hx, hy = np.gradient(H, x, y)
    return _grid(x, y, H, hx, hy)


def _grid(x, y, h, hx, hy) -> dict:
    from scipy.interpolate import RegularGridInterpolator
    o = dict(x=x, y=y, h=h, hx=hx, hy=hy)
    o["fh"] = RegularGridInterpolator((x, y), h, bounds_error=False, fill_value=None)
    o["fhx"] = RegularGridInterpolator((x, y), hx, bounds_error=False, fill_value=None)
    o["fhy"] = RegularGridInterpolator((x, y), hy, bounds_error=False, fill_value=None)
    return o


def _query(grid, X, Y):
    P = np.stack([X, Y], -1)
    valid = (X >= grid["x"][0]) & (X <= grid["x"][-1]) & (Y >= grid["y"][0]) & (Y <= grid["y"][-1])
    return grid["fh"](P), grid["fhx"](P), grid["fhy"](P), valid


# --------------------------------------------------------------------------------------------------
# contact: footwall = master.  Upper nodes = hangingwall grid points.  d = slip direction (2,).
# --------------------------------------------------------------------------------------------------
def contact_forces(z0, u, lower, upper, eps_n, mu, d):
    xi, eta = np.meshgrid(upper["x"], upper["y"], indexing="ij")
    dA = (upper["x"][1] - upper["x"][0]) * (upper["y"][1] - upper["y"][0])
    X = (xi + u[0]).ravel(); Y = (eta + u[1]).ravel()
    Z = (z0 + upper["h"]).ravel()
    hL, hLx, hLy, valid = _query(lower, X, Y)
    sec = np.sqrt(1.0 + hLx ** 2 + hLy ** 2)
    nx, ny, nz = -hLx / sec, -hLy / sec, 1.0 / sec
    gap_n = (Z - hL) * nz
    active = valid & (gap_n < 0.0)
    fn = np.where(active, eps_n * (-gap_n) * dA, 0.0)
    # in-plane slip direction d -> tangential component on each node's tangent plane
    dz_slip = 0.0  # dilation contributes negligibly to slip direction at the increment scale
    sx, sy, sz = d[0], d[1], dz_slip
    sdotn = sx * nx + sy * ny + sz * nz
    tx, ty, tz = sx - sdotn * nx, sy - sdotn * ny, sz - sdotn * nz
    tnorm = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    tnorm = np.where(tnorm > 1e-12, tnorm, 1.0)
    tx, ty, tz = tx / tnorm, ty / tnorm, tz / tnorm
    Fx = np.sum(fn * nx - mu * fn * tx)
    Fy = np.sum(fn * ny - mu * fn * ty)
    Fz = np.sum(fn * nz - mu * fn * tz)
    diag = dict(n_active=int(active.sum()), fn_sum=float(fn.sum()),
                pen_max=float((-gap_n[active]).max()) if active.any() else 0.0)
    return Fx, Fy, Fz, diag


def solve_z_equilibrium(u, lower, upper, eps_n, mu, W, d, z_bracket):
    zlo, zhi = z_bracket
    for _ in range(60):
        zm = 0.5 * (zlo + zhi)
        _, _, Fz, _ = contact_forces(zm, u, lower, upper, eps_n, mu, d)
        if Fz > W:
            zlo = zm
        else:
            zhi = zm
    return 0.5 * (zlo + zhi)


def run_shear_3d(lower, upper, direction, sigma_n=1.0, mu=0.3, shear_total=8.0, n_inc=80,
                 eps_n=5.0e4, verbose=False) -> dict:
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    Ax = lower["x"][-1] - lower["x"][0]; Ay = lower["y"][-1] - lower["y"][0]
    amp = max(upper["h"].max() - upper["h"].min(), lower["h"].max() - lower["h"].min())
    z_lo = float(lower["h"].max() - upper["h"].min() - 2 * amp)
    z_hi = float(lower["h"].max() - upper["h"].min() + 2 * amp)
    us = np.linspace(0.0, shear_total, n_inc)
    rec = {k: [] for k in ("u", "z", "dilation", "Tx", "Ty", "T_par", "T_perp",
                           "mu_app", "n_active", "pen_max")}
    z0 = None
    A = Ax * Ay
    for j, umag in enumerate(us):
        u = umag * d
        # overlap area shrinks with shear; keep normal STRESS constant
        A_ov = max((Ax - abs(u[0])) * (Ay - abs(u[1])), 0.25 * A)
        W = sigma_n * A_ov
        z = solve_z_equilibrium(u, lower, upper, eps_n, mu, W, d, (z_lo, z_hi))
        Fx, Fy, Fz, dg = contact_forces(z, u, lower, upper, eps_n, mu, d)
        if z0 is None:
            z0 = z
        Tx, Ty = -Fx / A_ov, -Fy / A_ov
        T_par = Tx * d[0] + Ty * d[1]
        T_perp = -Tx * d[1] + Ty * d[0]
        rec["u"].append(float(umag)); rec["z"].append(float(z)); rec["dilation"].append(float(z - z0))
        rec["Tx"].append(float(Tx)); rec["Ty"].append(float(Ty))
        rec["T_par"].append(float(T_par)); rec["T_perp"].append(float(T_perp))
        rec["mu_app"].append(float(T_par / sigma_n))
        rec["n_active"].append(dg["n_active"]); rec["pen_max"].append(dg["pen_max"])
        if verbose and (j % 10 == 0 or j == n_inc - 1):
            print(f"    u={umag:6.3f}  dil={z-z0:+.4f}  mu_par={T_par/sigma_n:6.4f}  "
                  f"T_perp/sig={T_perp/sigma_n:+.4f}  nC={dg['n_active']:5d}  pen={dg['pen_max']:.2e}")
    for k in rec:
        rec[k] = np.asarray(rec[k])
    ss = slice(int(0.6 * n_inc), n_inc)
    summary = dict(mu_app_peak=float(rec["mu_app"].max()),
                   mu_app_steady=float(np.mean(rec["mu_app"][ss])),
                   phi_app_peak_deg=float(math.degrees(math.atan(max(rec["mu_app"].max(), 0)))),
                   dilation_total=float(rec["dilation"][-1]),
                   dilation_peak=float(rec["dilation"].max()),
                   T_perp_steady=float(np.mean(rec["T_perp"][ss]) / sigma_n),
                   pen_max=float(rec["pen_max"].max()))
    return dict(history=rec, summary=summary,
                params=dict(direction=d.tolist(), sigma_n=sigma_n, mu=mu,
                            shear_total=shear_total, n_inc=n_inc, eps_n=eps_n))


# --------------------------------------------------------------------------------------------------
def _save(name, payload):
    out_dir = os.path.join(_ROOT, "runs", name); os.makedirs(out_dir, exist_ok=True)
    ser = {}
    for k, v in payload.items():
        ser[k] = {kk: np.asarray(vv).tolist() for kk, vv in v.items()} if k == "history" else v
    json.dump(ser, open(os.path.join(out_dir, "history.json"), "w"))
    json.dump(payload.get("summary", {}), open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    return out_dir


def run_ridged_anisotropy(angle_deg=20.0, mu=0.3, wavelength=5.0):
    """Ridged sawtooth: across-ridge (in-plane) -> Patton tan(phi_b+i); along-ridge -> pure mu."""
    from solvers.contact.surface_chart_3d import RidgedSawtooth3D
    x = np.linspace(0, 40, 400); y = np.linspace(0, 20, 120)
    saw = RidgedSawtooth3D(wavelength=wavelength, angle_deg=angle_deg).double()
    lo = bake_module(saw, x, y)
    up = dict(lo); up = _copy_grid(lo)
    phi_b = math.degrees(math.atan(mu)); patton = math.tan(math.radians(phi_b + angle_deg))
    out = {}
    for mode, d in (("in_plane", (1.0, 0.0)), ("out_of_plane", (0.0, 1.0))):
        r = run_shear_3d(lo, up, d, sigma_n=1.0, mu=mu, shear_total=0.45 * wavelength, n_inc=60,
                         eps_n=2.0e4, verbose=True)
        out[mode] = r["summary"]
        print(f"  [{mode}] peak mu_app={r['summary']['mu_app_peak']:.4f}  "
              f"dilation_total={r['summary']['dilation_total']:+.4f}")
    out["patton_pred"] = patton; out["phi_b_deg"] = phi_b; out["asperity_angle_deg"] = angle_deg
    out["in_plane_relerr_vs_patton"] = abs(out["in_plane"]["mu_app_peak"] - patton) / patton
    _save("rock_joint_3d_ridged", {"summary": out})
    print(f"\n  in-plane peak mu_app = {out['in_plane']['mu_app_peak']:.4f}  vs Patton {patton:.4f}  "
          f"({out['in_plane_relerr_vs_patton']*100:.2f}% err)")
    print(f"  out-of-plane: mu_app steady = {out['out_of_plane']['mu_app_steady']:.4f} (~mu={mu}), "
          f"dilation = {out['out_of_plane']['dilation_total']:+.4f} (~0)")
    return out


def _copy_grid(g):
    return _grid(g["x"].copy(), g["y"].copy(), g["h"].copy(), g["hx"].copy(), g["hy"].copy())


def _load_inada_map(tag="rough", crop_mm=20.0, every=4):
    import pandas as pd
    Z = pd.read_csv(os.path.join(RAW, f"{tag}_footwall.csv"), header=None).to_numpy(float)
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, 1.0)[::every, ::every]
    dx = DX_MM * every
    nx = min(int(crop_mm / dx), Z.shape[0]); ny = min(int(crop_mm / dx), Z.shape[1])
    Z = Z[:nx, :ny]
    x = np.arange(Z.shape[0]) * dx; y = np.arange(Z.shape[1]) * dx
    return x, y, Z


def run_real_surface(tag="rough", mode="all", mu=0.3, sigma_n=1.0, shear_total=6.0):
    if not os.path.isdir(RAW):
        print("  raw Inada CSVs absent — run characterize_inada_joint.py / downloader first"); return
    x, y, Z = _load_inada_map(tag)
    print(f"  {tag} patch: {Z.shape} grid, {x[-1]:.1f} x {y[-1]:.1f} mm, RMS={Z.std():.3f} mm")
    lo = bake_samples(x, y, Z); up = bake_samples(x, y, Z.copy())   # perfectly mated
    modes = MODES if mode == "all" else {mode: MODES[mode]}
    out = {}
    for nm, d in modes.items():
        r = run_shear_3d(lo, up, d, sigma_n=sigma_n, mu=mu, shear_total=shear_total, n_inc=60,
                         eps_n=5.0e4, verbose=True)
        _save(f"rock_joint_3d_{tag}_{nm}", r)
        out[nm] = r["summary"]
        print(f"  [{nm:12s}] peak mu_app={r['summary']['mu_app_peak']:.3f}  "
              f"steady={r['summary']['mu_app_steady']:.3f}  dilation={r['summary']['dilation_total']:+.3f} mm  "
              f"T_perp/sig={r['summary']['T_perp_steady']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridged", action="store_true", help="ridged-sawtooth anisotropy V&V")
    ap.add_argument("--surface", choices=["rough", "smooth"])
    ap.add_argument("--mode", default="all", choices=["all", "in_plane", "out_of_plane", "mixed"])
    ap.add_argument("--mu", type=float, default=0.3)
    args = ap.parse_args()
    if args.ridged:
        run_ridged_anisotropy(mu=args.mu)
    if args.surface:
        run_real_surface(tag=args.surface, mode=args.mode, mu=args.mu)
    if not args.ridged and not args.surface:
        print("pass --ridged and/or --surface rough|smooth")


if __name__ == "__main__":
    main()
