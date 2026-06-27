#!/usr/bin/env python3
"""CV-7 rock-joint direct shear via the OPTIMAL-TRANSPORT gap field.

This driver solves the Patton sawtooth direct-shear problem (CV-7, manual §11.9) using the
*optimal-transport* (measure-coupling) gap, then validates the emergent apparent friction against
the closed-form Patton law and contrasts it with the OLD vertical-ray (tributary-lumped) scheme.

Formulation
-----------
Two mating sawtooth faces of asperity angle ``i`` are sheared under plain Coulomb ``mu = tan(phi_b)``.
The dilation and apparent friction are EMERGENT outputs of the resolved geometry (no dilatancy law).

Two gap schemes are run on the IDENTICAL incremental quasi-static loop:

  * OLD (vertical-ray lumped): each upper node projects vertically to the footwall; the gap is the
    vertical offset projected onto the footwall normal.  Tributary-lumped penalty.

  * NEW (OT coupling): the footwall contact point for each upper node is found by the measure-coupling
    correspondence ``MonotoneCoupling1D`` (exact 1-D optimal transport, the monotone/Brenier map that
    matches equal cumulative-arclength quantiles), and the gap is measured in the footwall normal frame
    at that mass-balanced contact point:  ``g_N = (X_s - X_m).n_master``.

Both share the SAME penalty (``eps_n <-g_N>_+``) and fully-mobilized Coulomb friction (the audited
``TractionField`` law).  For a RIGID block the NET force is integration-scheme-invariant — the
consistent/OT and tributary-lumped sums coincide — so the OT changes the rigid rock-joint answer ONLY
through the correspondence (the point-wise gap field, friction direction, dilatancy distribution).
This driver makes that invariance an explicit, measured deliverable.

Closed form (Patton): for mated sawtooths the emergent apparent friction is
    mu_app = tan(phi_b + i)  =  (tan i + mu) / (1 - mu tan i).
Pass criterion (matching the lumped driver):  |mu_app_peak - tan(phi_b+i)| / tan(phi_b+i) < 0.02.

Run:  python3 benchmarks/contact/cv_numerical/cv7_ot_gap.py
      python3 benchmarks/contact/cv_numerical/cv7_ot_gap.py --angle 20 --mu 0.3
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

from solvers.contact.measure_coupling.coupling import MonotoneCoupling1D
from solvers.contact.measure_coupling.traction import TractionField


# --------------------------------------------------------------------------------------------------
# surface baking (analytic sawtooth — no neural training; runs in <1 s)
# --------------------------------------------------------------------------------------------------
def bake_sawtooth(wavelength: float, angle_deg: float, x_lo: float, x_hi: float,
                  n: int = 6000) -> dict:
    """Mated triangular sawtooth height profile h(x) of asperity angle ``angle_deg``.

    Symmetric triangle wave of slope ``tan(i)`` (rise) / ``-tan(i)`` (fall); single-valued so the
    1-D OT coupling is exact.  Returns the baked ``{x, h, hp}`` dict (numpy, x ascending)."""
    x = np.linspace(x_lo, x_hi, n)
    slope = math.tan(math.radians(angle_deg))
    amp = 0.5 * slope * (0.5 * wavelength)               # peak-to-mean of the triangle
    # symmetric triangle wave in [0,1) phase: up on [0,0.5), down on [0.5,1)
    phase = (x / wavelength) % 1.0
    tri = np.where(phase < 0.5, phase, 1.0 - phase)      # 0..0.5..0 sawtooth, period 1
    h = (slope * wavelength) * tri                       # height; slope magnitude = tan(i)
    h = h - h.mean()
    hp = np.gradient(h, x)
    return dict(x=x, h=h, hp=hp)


# --------------------------------------------------------------------------------------------------
# contact: footwall = master.  field=False -> vertical ray (OLD).  field=True -> OT coupling (NEW).
# Uses TractionField for the penalty+Coulomb law (the audited measure_coupling kernel).
# --------------------------------------------------------------------------------------------------
def contact_forces(yU: float, ux: float, lower: dict, upper: dict, eps_n: float, mu: float,
                   field: bool):
    """Net (F_x, F_z) on the hangingwall + diagnostics, at vertical position yU and shear ux."""
    xi = upper["x"]
    dxi = np.gradient(xi)                                # tributary width per node
    X = xi + ux                                          # world x of each upper node
    Z = yU + upper["h"]                                  # world z of each upper node (bottom surface)

    if field:
        # NEW: optimal-transport correspondence (monotone rearrangement of arclength measures).
        slave = dict(x=X, h=Z, hp=np.gradient(Z, X))
        master = dict(x=lower["x"], h=lower["h"], hp=lower["hp"])
        _, Xm, _ = MonotoneCoupling1D(slave, master).map(X)
        xm = Xm[:, 0]
        hpL = np.interp(xm, lower["x"], lower["hp"])
        valid = (xm >= lower["x"][0]) & (xm <= lower["x"][-1])
        sec = np.sqrt(1.0 + hpL ** 2)
        nz = 1.0 / sec; nx = -hpL / sec
        tx = 1.0 / sec; tz = hpL / sec
        gap_n = (Z - Xm[:, 1]) * nz + (X - xm) * nx      # (X_s - X_m).n_master  (OT gap)
        Xm_x = xm
    else:
        # OLD: vertical-ray closest point on the footwall.
        hL = np.interp(X, lower["x"], lower["h"])
        hpL = np.interp(X, lower["x"], lower["hp"])
        valid = (X >= lower["x"][0]) & (X <= lower["x"][-1])
        sec = np.sqrt(1.0 + hpL ** 2)
        nz = 1.0 / sec; nx = -hpL / sec
        tx = 1.0 / sec; tz = hpL / sec
        gap_n = (Z - hL) * nz                            # vertical offset projected on n
        Xm_x = X

    # Build per-point unit normals (Q,2) and feed the audited TractionField penalty law.
    n_vec = np.column_stack([nx, nz])
    n_vec = np.where(valid[:, None], n_vec, 0.0)
    gN = np.where(valid, gap_n, 1.0)                     # invalid -> open (no force)
    tr = TractionField(eps_n=eps_n, mu=0.0).evaluate(gN, n_vec)
    pN = tr["pN"]                                        # = eps_n <-gN>_+  (pressure)
    fn = pN * dxi                                        # nodal normal magnitude (tributary width)
    active = valid & (gap_n < 0.0)
    fn = np.where(active, fn, 0.0)

    # node forces: normal along n, fully-mobilized Coulomb -mu*fn along the footwall tangent t.
    Fx = float(np.sum(fn * nx - mu * fn * tx))
    Fz = float(np.sum(fn * nz - mu * fn * tz))
    diag = dict(n_active=int(active.sum()), fn_sum=float(fn.sum()),
                pen_max=float((-gap_n[active]).max()) if active.any() else 0.0)
    return Fx, Fz, diag


def solve_y_equilibrium(ux, lower, upper, eps_n, mu, W, y_bracket, field):
    """Bisection: find yU s.t. F_z(yU) == W (normal equilibrium; F_z monotone-decreasing in yU)."""
    ylo, yhi = y_bracket
    for _ in range(60):
        ym = 0.5 * (ylo + yhi)
        _, Fz, _ = contact_forces(ym, ux, lower, upper, eps_n, mu, field=field)
        if Fz > W:
            ylo = ym
        else:
            yhi = ym
    return 0.5 * (ylo + yhi)


def run_shear(lower, upper, sigma_n, mu, shear_total, n_inc, eps_n, field, verbose=False):
    """Quasi-static incremental direct shear.  Returns history + scalar summary."""
    x0, x1 = lower["x"][0], lower["x"][-1]
    span = x1 - x0
    amp = max(upper["h"].max() - upper["h"].min(), lower["h"].max() - lower["h"].min())
    y_lo = float(lower["h"].max() - upper["h"].min() - 2.0 * amp)
    y_hi = float(lower["h"].max() - upper["h"].min() + 2.0 * amp)

    uxs = np.linspace(0.0, shear_total, n_inc)
    rec = {k: [] for k in ("ux", "y", "dilation", "tau", "mu_app", "n_active", "pen_max")}
    y0 = None
    for j, ux in enumerate(uxs):
        L_ov = max(span - abs(ux), 0.25 * span)
        W = sigma_n * L_ov
        y = solve_y_equilibrium(ux, lower, upper, eps_n, mu, W, (y_lo, y_hi), field=field)
        Fx, Fz, d = contact_forces(y, ux, lower, upper, eps_n, mu, field=field)
        if y0 is None:
            y0 = y
        tau = -Fx / L_ov
        rec["ux"].append(float(ux)); rec["y"].append(float(y))
        rec["dilation"].append(float(y - y0)); rec["tau"].append(float(tau))
        rec["mu_app"].append(float(tau / sigma_n)); rec["n_active"].append(d["n_active"])
        rec["pen_max"].append(d["pen_max"])
        if verbose and (j % 20 == 0 or j == n_inc - 1):
            print(f"    [{'OT ' if field else 'ray'}] ux={ux:6.3f}  dil={y-y0:+.4f}  "
                  f"mu_app={tau/sigma_n:6.4f}  nC={d['n_active']:4d}  pen={d['pen_max']:.2e}")
    for k in rec:
        rec[k] = np.asarray(rec[k])
    ss = slice(int(0.6 * n_inc), n_inc)
    # Plateau-median emergent-friction estimator.  For a rigid mated sawtooth the apparent friction
    # is a CONSTANT plateau (= tan(phi_b+i)) once full contact is seated; across increments the only
    # variation is float64 round-off in the force closure.  The plateau MEDIAN is therefore the
    # variance-reduced, unbiased estimator of the emergent friction, whereas ``max()`` is biased
    # toward the single most positively-rounded sample (it picks the worst round-off).  We report
    # BOTH: ``mu_app_peak`` (max, kept for the head-to-head invariance) and ``mu_app_robust``
    # (steady-plateau median, the faithful emergent-friction scalar).
    mu_seated = rec["mu_app"][rec["mu_app"] > 0.5]   # drop the unseated first increment(s)
    mu_robust = float(np.median(mu_seated)) if mu_seated.size else float(rec["mu_app"].max())
    summary = dict(
        mu_app_peak=float(rec["mu_app"].max()),
        mu_app_robust=mu_robust,
        mu_app_steady=float(np.mean(rec["mu_app"][ss])),
        phi_app_peak_deg=float(math.degrees(math.atan(rec["mu_app"].max()))),
        dilation_total=float(rec["dilation"][-1]),
        dilation_peak=float(rec["dilation"].max()),
        pen_max=float(rec["pen_max"].max()),
    )
    return dict(history=rec, summary=summary)


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def run(angle_deg=25.0, mu=0.3, wavelength=5.0):
    saw = bake_sawtooth(wavelength=wavelength, angle_deg=angle_deg, x_lo=0.0, x_hi=50.0)
    lower = saw
    upper = dict(x=saw["x"].copy(), h=saw["h"].copy(), hp=saw["hp"].copy())   # mated

    phi_b = math.degrees(math.atan(mu))
    patton = math.tan(math.radians(phi_b + angle_deg))

    common = dict(sigma_n=1.0, mu=mu, shear_total=0.45 * wavelength, n_inc=120, eps_n=2.0e4)
    print("== CV-7 OT-gap direct shear (Patton sawtooth) ==")
    print(f"   i={angle_deg} deg  mu=tan(phi_b)={mu}  phi_b={phi_b:.3f} deg  "
          f"tan(phi_b+i)={patton:.6f}\n")

    print("  NEW: optimal-transport coupling gap (MonotoneCoupling1D)")
    res_ot = run_shear(lower, upper, field=True, verbose=True, **common)
    print("\n  OLD: vertical-ray tributary-lumped gap")
    res_ray = run_shear(lower, upper, field=False, verbose=True, **common)

    def _err(s):
        return abs(s["mu_app_peak"] - patton) / patton

    def _err_robust(s):
        return abs(s["mu_app_robust"] - patton) / patton

    err_ot = _err(res_ot["summary"])
    err_ray = _err(res_ray["summary"])
    # Variance-reduced plateau-median emergent-friction error (the faithful model-floor measurement).
    err_ot_robust = _err_robust(res_ot["summary"])
    err_ray_robust = _err_robust(res_ray["summary"])
    dil_rate_ot = res_ot["summary"]["dilation_total"] / common["shear_total"]
    dil_rate_ray = res_ray["summary"]["dilation_total"] / common["shear_total"]
    tan_i = math.tan(math.radians(angle_deg))

    # rigid-block net-force invariance: the two schemes' NET peak shear should coincide.
    net_force_rel_diff = abs(res_ot["summary"]["mu_app_peak"]
                             - res_ray["summary"]["mu_app_peak"]) / res_ray["summary"]["mu_app_peak"]

    passed = (err_ot < 0.02) and (err_ray < 0.02)

    metrics = dict(
        cv="CV-7",
        problem="rock_joint_direct_shear_patton_sawtooth",
        asperity_angle_deg=float(angle_deg),
        mu_base=float(mu),
        phi_b_deg=float(phi_b),
        patton_mu_pred=float(patton),
        tan_i=float(tan_i),
        # NEW (OT coupling)
        ot_mu_app_peak=res_ot["summary"]["mu_app_peak"],
        ot_mu_app_robust=res_ot["summary"]["mu_app_robust"],
        ot_mu_app_steady=res_ot["summary"]["mu_app_steady"],
        ot_relerr_vs_patton=float(err_ot),
        ot_relerr_vs_patton_robust=float(err_ot_robust),
        ot_dilation_total=res_ot["summary"]["dilation_total"],
        ot_dilation_rate=float(dil_rate_ot),
        ot_dilation_rate_relerr_vs_tan_i=float(abs(dil_rate_ot - tan_i) / tan_i),
        ot_pen_max=res_ot["summary"]["pen_max"],
        # OLD (vertical-ray lumped)
        ray_mu_app_peak=res_ray["summary"]["mu_app_peak"],
        ray_mu_app_robust=res_ray["summary"]["mu_app_robust"],
        ray_mu_app_steady=res_ray["summary"]["mu_app_steady"],
        ray_relerr_vs_patton=float(err_ray),
        ray_relerr_vs_patton_robust=float(err_ray_robust),
        ray_dilation_total=res_ray["summary"]["dilation_total"],
        ray_dilation_rate=float(dil_rate_ray),
        ray_pen_max=res_ray["summary"]["pen_max"],
        # comparison / invariance
        net_force_rel_diff_ot_vs_ray=float(net_force_rel_diff),
        pass_tol=0.02,
        passed=bool(passed),
    )

    out_dir = os.path.join(_ROOT, "runs", "cv7_ot_gap")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    hist = {scheme: {k: np.asarray(v).tolist() for k, v in res["history"].items()}
            for scheme, res in (("ot", res_ot), ("ray", res_ray))}
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)

    print("\n== RESULTS ==")
    print(f"  closed form  tan(phi_b+i)        = {patton:.6f}")
    print(f"  NEW (OT)     mu_app_peak (max)   = {metrics['ot_mu_app_peak']:.6f}  "
          f"(relerr {err_ot:.3e})")
    print(f"  NEW (OT)     mu_app_robust(med)  = {metrics['ot_mu_app_robust']:.12f}  "
          f"(relerr {err_ot_robust:.3e})  <- variance-reduced model floor")
    print(f"  OLD (ray)    mu_app_peak (max)   = {metrics['ray_mu_app_peak']:.6f}  "
          f"(relerr {err_ray:.3e})")
    print(f"  OLD (ray)    mu_app_robust(med)  = {metrics['ray_mu_app_robust']:.6f}  "
          f"(relerr {err_ray_robust:.3e})")
    print(f"  OT  dilation rate dy/dux         = {dil_rate_ot:.4f}  (tan i = {tan_i:.4f}, "
          f"relerr {metrics['ot_dilation_rate_relerr_vs_tan_i']*100:.2f}%)")
    print(f"  ray dilation rate dy/dux         = {dil_rate_ray:.4f}")
    print(f"  rigid net-force invariance |OT-ray|/ray = {net_force_rel_diff*100:.4f}%")
    print(f"  PASS = {passed}")
    print(f"  saved -> {out_dir}/metrics.json")
    return metrics


def run_real_surface(mu=0.3, n_inc=80, shear_frac=0.10, offset_frac=0.5, profile="rough"):
    """SECOND VALIDATION (data-grounded demonstration): emergent dilatancy on the REAL Inada granite
    tensile-fracture profile (Digital Rocks #273).  This is a DEMONSTRATION, not a closed-form pass:
    a self-affine fractal joint has no Patton angle.  We offset the upper face by half a wavelength-
    scale (``offset_frac`` of the mean asperity spacing) so the mated faces are NOT degenerately
    coincident, shear, and confirm that (i) dilation EMERGES (dy/dux > 0, geometry-driven, no
    dilatancy law) and (ii) the OT-emergent dilation rate is bounded by the surface's asperity-slope
    statistics (mean uphill slope <= rate <= max slope) — an independent geometric envelope the
    contact solver must respect.  We report it; the quantitative pass lives in the Patton checks."""
    npz = os.path.join(_ROOT, "data", "inada_joint", f"inada_{profile}_profile.npz")
    if not os.path.exists(npz):
        print(f"  [real-surface] data not found ({npz}); skipping second validation.")
        return None
    d = np.load(npz)
    x = np.asarray(d["x_mm"], float)
    h = np.asarray(d["footwall_mm"], float)
    h = h - h.mean()
    hp = np.gradient(h, x)
    # offset the hangingwall horizontally so the two mated faces are not degenerately coincident
    # (a perfectly seated identical pair has zero gap everywhere -> ill-posed seating).
    dx = float(np.median(np.diff(x)))
    shift = max(1, int(offset_frac * (d["rms_mm"] / dx)))
    xu = x.copy()
    hu = np.roll(h, shift)
    lower = dict(x=x, h=h, hp=hp)
    upper = dict(x=xu, h=hu, hp=np.gradient(hu, xu))

    span = x[-1] - x[0]
    shear_total = shear_frac * span
    common = dict(sigma_n=1.0, mu=mu, shear_total=shear_total, n_inc=n_inc, eps_n=2.0e4)
    res_ot = run_shear(lower, upper, field=True, **common)

    # geometric envelope: mean UPHILL slope and max slope (the rate must sit in [mean_up, max]).
    up = hp[hp > 0.0]
    mean_uphill = float(up.mean()) if up.size else 0.0
    max_slope = float(np.abs(hp).max())
    # OT emergent dilation rate over the early climbing phase.
    hist = res_ot["history"]
    k = max(2, int(0.4 * n_inc))
    dil_rate_ot = float((hist["dilation"][k] - hist["dilation"][0]) /
                        (hist["ux"][k] - hist["ux"][0] + 1e-30))
    dilation_emerges = bool(hist["dilation"][-1] > 0.0)
    rate_in_envelope = bool(mean_uphill <= dil_rate_ot <= max_slope)
    out = dict(
        problem="rock_joint_real_inada_emergent_dilatancy_demo",
        profile=profile, rms_mm=float(d["rms_mm"]), fractal_D=float(d["fractal_D"]),
        mean_uphill_slope=mean_uphill, max_slope=max_slope,
        ot_early_dilation_rate=dil_rate_ot,
        ot_total_dilation=float(hist["dilation"][-1]),
        dilation_emerges=dilation_emerges,
        rate_in_geometric_envelope=rate_in_envelope,
    )
    print("\n== SECOND VALIDATION (demo): real Inada granite surface — emergent dilatancy ==")
    print(f"   profile={profile}  RMS={out['rms_mm']:.3f} mm  D={out['fractal_D']:.3f}  "
          f"(no Patton angle for a fractal — DEMONSTRATION, not a closed-form pass)")
    print(f"   geometric slope envelope         = [{mean_uphill:.4f}, {max_slope:.4f}]")
    print(f"   OT early dilation rate dy/dux    = {dil_rate_ot:.4f}  "
          f"-> in envelope: {rate_in_envelope}")
    print(f"   dilation emerges (no law)        = {dilation_emerges}  "
          f"(total {out['ot_total_dilation']:+.3f} mm)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=25.0, help="sawtooth asperity angle (deg)")
    ap.add_argument("--mu", type=float, default=0.3, help="base Coulomb mu = tan(phi_b)")
    ap.add_argument("--wavelength", type=float, default=5.0)
    ap.add_argument("--real-surface", action="store_true",
                    help="also run the real-Inada-granite emergent-dilatancy second validation")
    args = ap.parse_args()
    run(angle_deg=args.angle, mu=args.mu, wavelength=args.wavelength)
    if args.real_surface:
        run_real_surface(mu=args.mu)


if __name__ == "__main__":
    main()
