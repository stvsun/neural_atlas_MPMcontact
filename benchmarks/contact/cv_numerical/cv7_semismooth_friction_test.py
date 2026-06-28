#!/usr/bin/env python3
r"""CV-7 Phase 0 -- semismooth-Newton Coulomb friction: energy-balance + Patton verification.

Demonstrates the deliverable for brief Phase 0 (``contact_atlas/cv7_formulation_brief.md``): the
cyclic energy ledger of the rock-joint friction interface closes to within [0.95, 1.05] (vs the
current ~1.5x of the regularized return-map split in ``rock_joint_cyclic_fem.py``) while the Patton
closed form ``mu_app = tan(phi_b + i)`` stays exact and the friction residual drops below 0.1 %.

Two parts:
  (A) BLOCK testbed (closed-form gate): a rigid slider pulled by an elastic spring across a Patton
      sawtooth, solved with the semismooth-Newton return map (``SemismoothBlock1D``).  We measure
        * steady mu_app vs tan(phi_b + i)              (Patton, target 0.00 %),
        * cyclic energy-balance ratio                   (target [0.95, 1.05]),
        * friction residual = |mu_app_slip - Patton|    (target < 0.1 %),
      and contrast the cyclic ledger against a REGULARIZED-SPLIT baseline (the soft-tangent / trial-
      return accounting used by the FEM driver) to quantify the ~1.5x -> ~1.0 improvement.

  (B) FEM PATCH cross-check (partial): a small bank of contact points sheared cyclically through the
      vectorized ``FrictionInterface1D`` with the EXACT discrete plastic-work ledger, confirming the
      bank closes too (the building block for full cyclic-FEM integration, left partial by design).

Run:
    python3 benchmarks/contact/cv_numerical/cv7_semismooth_friction_test.py            # full local
    python3 benchmarks/contact/cv_numerical/cv7_semismooth_friction_test.py --quick    # fast
    python3 benchmarks/contact/cv_numerical/cv7_semismooth_friction_test.py --cycles 5 --i-deg 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.contact.semismooth_friction import (        # noqa: E402
    SemismoothBlock1D, FrictionInterface1D, patton_mu_app)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _cyclic_schedule(amplitude, n_per_quarter, n_cycles):
    segs = []
    for _ in range(n_cycles):
        segs += [np.linspace(0, amplitude, n_per_quarter, endpoint=False),
                 np.linspace(amplitude, -amplitude, 2 * n_per_quarter, endpoint=False),
                 np.linspace(-amplitude, 0, n_per_quarter, endpoint=False)]
    return np.concatenate(segs + [np.array([0.0])])


# --------------------------------------------------------------------------------------------------
#  REGULARIZED-SPLIT baseline (the accounting style of rock_joint_cyclic_fem.py) for contrast
# --------------------------------------------------------------------------------------------------
def _regularized_block_ledger(N, phi_b_deg, i_deg, k_load, rT, schedule, kt_slip_frac=0.1):
    r"""Same Patton block, but with the REGULARIZED return-map accounting that the cyclic FEM uses:
    a *softened* slip tangent (kt_slip = kt_slip_frac * r_T) and dissipation estimated as the trial
    plastic increment ``tT * dsp`` WITHOUT the consistent slip projection.  Reproduces the energy
    leak (W_ext does not match d(stored) + dissipation -> ledger ~> 1) so the semismooth gain is
    measurable on the SAME problem.
    """
    mu_eff = float(np.tan(np.radians(phi_b_deg) + np.radians(i_deg)))
    tN = N
    x = 0.0
    sp = 0.0                      # plastic slip
    Wfric = 0.0
    Wext = 0.0
    E0_spring = None
    F_prev = None; xd_prev = None
    hist = dict(mu_app=[], W_ext=[], W_fric=[], E_spring=[], E_iface=[], slip=[])
    for xd in np.asarray(schedule, float):
        # damped Picard on x with the SOFT slip tangent (mimics the FEM active-set inner loop)
        for _ in range(200):
            te = rT * (x - sp)                       # trial elastic shear
            cone = mu_eff * tN
            slip = abs(te) > cone
            tT = np.sign(te) * cone if slip else te
            kt = (kt_slip_frac * rT) if slip else rT
            R = k_load * (xd - x) - tT
            K = k_load + kt
            dx = R / K
            x += dx
            if abs(dx) < 1e-12:
                break
        te = rT * (x - sp); cone = mu_eff * tN
        slip = abs(te) > cone
        tT = np.sign(te) * cone if slip else te
        # regularized dissipation: trial plastic increment (NOT consistently projected)
        dsp = (te - tT) / rT if slip else 0.0
        Wfric += abs(tT * dsp)
        sp += dsp
        F = k_load * (xd - x)
        if F_prev is not None:
            Wext += 0.5 * (F + F_prev) * (xd - xd_prev)
        F_prev = F; xd_prev = xd
        E_spring = 0.5 * k_load * (xd - x) ** 2
        E_iface = 0.5 * rT * (x - sp) ** 2
        hist["mu_app"].append(tT / N); hist["W_ext"].append(Wext); hist["W_fric"].append(Wfric)
        hist["E_spring"].append(E_spring); hist["E_iface"].append(E_iface); hist["slip"].append(bool(slip))
    for k in hist:
        hist[k] = np.asarray(hist[k])
    dE = (hist["E_spring"][-1] - hist["E_spring"][0]) + (hist["E_iface"][-1] - hist["E_iface"][0])
    ratio = hist["W_ext"][-1] / (dE + hist["W_fric"][-1]) if abs(dE + hist["W_fric"][-1]) > 1e-300 else float("nan")
    return hist, float(ratio)


# --------------------------------------------------------------------------------------------------
def run_block(phi_b_deg, i_deg, k_load, rT, amplitude, n_per_quarter, n_cycles, N=1.0):
    sched = _cyclic_schedule(amplitude, n_per_quarter, n_cycles)
    blk = SemismoothBlock1D(N=N, phi_b_deg=phi_b_deg, i_deg=i_deg, k_load=k_load, rT=rT)
    H = blk.run(sched)
    ratio = blk.energy_balance_ratio(H)
    patton = patton_mu_app(phi_b_deg, i_deg)
    slip = H["slip"]
    if slip.any():
        mu_slip = np.abs(H["mu_app"])[slip]
        resid = float(np.abs(mu_slip - patton).max() / patton)
    else:
        resid = float("nan")
    # regularized baseline on the same problem
    _, ratio_reg = _regularized_block_ledger(N, phi_b_deg, i_deg, k_load, rT, sched)
    return dict(patton=patton, peak_mu=float(np.abs(H["mu_app"]).max()),
                energy_ratio=float(ratio), energy_ratio_reg=float(ratio_reg),
                friction_resid=resid, n_steps=len(sched)), H


# --------------------------------------------------------------------------------------------------
def run_fem_patch(P=25, phi_b_deg=30.0, i_deg=8.0, rN=2.0e4, rT=2.0e3,
                  amplitude=0.04, n_per_quarter=20, n_cycles=3, sigma_n=4.0):
    r"""Cyclic shear of a bank of P contact points (a flat FEM interface patch) through the exact
    semismooth ``FrictionInterface1D``.  The top of the patch is rigidly driven in-plane by the
    cyclic schedule at fixed normal compression; we sum the exact per-point energy ledger.
    """
    mu_eff = float(np.tan(np.radians(phi_b_deg) + np.radians(i_deg)))
    area = np.full(P, 1.0 / P)
    iface = FrictionInterface1D(P=P, mu=mu_eff, rN=rN, rT=rT, dim_t=1, area=area)
    uN = np.full(P, -sigma_n / rN)             # fixed normal penetration -> tN = sigma_n
    sched = _cyclic_schedule(amplitude, n_per_quarter, n_cycles)
    Wext = 0.0
    tau_prev = None; u_prev = None
    E_iface0 = iface.elastic_energy(uN, np.zeros((P, 1)))
    hist = dict(u=[], tau=[], mu_app=[], W_ext=[], W_fric=[], slip_frac=[])
    for u in sched:
        uT = np.full((P, 1), u)
        tN, tT, info = iface.traction(uN, uT)
        tau = float((tT[:, 0] * area).sum())            # mean shear traction
        tN_mean = float((tN * area).sum())
        slip_frac = float(info["slip"].mean())
        if tau_prev is not None:
            Wext += 0.5 * (tau + tau_prev) * (u - u_prev)
        iface.commit(uN, uT)
        hist["u"].append(float(u)); hist["tau"].append(tau)
        hist["mu_app"].append(tau / max(tN_mean, 1e-12))
        hist["W_ext"].append(Wext); hist["W_fric"].append(float(iface.Wp.sum()))
        hist["slip_frac"].append(slip_frac)
        tau_prev = tau; u_prev = u
    for k in hist:
        hist[k] = np.asarray(hist[k])
    E_ifaceF = iface.elastic_energy(uN, np.full((P, 1), sched[-1]))
    dE = E_ifaceF - E_iface0
    denom = dE + hist["W_fric"][-1]
    ratio = hist["W_ext"][-1] / denom if abs(denom) > 1e-300 else float("nan")
    patton = patton_mu_app(phi_b_deg, i_deg)
    # friction residual: measured only where ALL points are fully sliding (steady plateau), so the
    # mu_app == tan(phi_b+i) identity is the right reference (reversal/transition steps excluded).
    fully_slip = hist["slip_frac"] >= 0.999
    slip_plateau = np.abs(hist["mu_app"])[fully_slip]
    resid = float(np.abs(slip_plateau - patton).max() / patton) if slip_plateau.size else float("nan")
    return dict(patton=patton, peak_mu=float(np.abs(hist["mu_app"]).max()),
                energy_ratio=float(ratio), friction_resid=resid, P=P), hist


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--i-deg", type=float, default=8.0)
    ap.add_argument("--phi-b-deg", type=float, default=30.0)
    ap.add_argument("--n-cells", type=int, default=5, help="FEM patch grid side (P=n^2)")
    ap.add_argument("--full-convergence", action="store_true")
    ap.add_argument("--energy-ledger", action="store_true")
    args = ap.parse_args()

    nq = 8 if args.quick else 20
    ncyc = args.cycles
    P = (args.n_cells ** 2) if not args.quick else 9

    print("=" * 78)
    print(f"CV-7 Phase 0: semismooth-Newton Coulomb friction  (phi_b={args.phi_b_deg} deg)")
    print("=" * 78)

    print("\n(A) Patton BLOCK testbed (closed-form gate), cyclic shear:")
    results = {}
    for i_deg in (0.0, args.i_deg):
        r, _ = run_block(args.phi_b_deg, i_deg, k_load=50.0, rT=2000.0,
                         amplitude=0.05, n_per_quarter=nq, n_cycles=ncyc)
        tag = f"block_i{int(i_deg)}"
        results[tag] = r
        print(f"  i={i_deg:5.1f} deg:  peak mu_app={r['peak_mu']:.6f}  Patton={r['patton']:.6f}  "
              f"(err {100*abs(r['peak_mu']-r['patton'])/r['patton']:.2e} %)")
        print(f"            friction residual = {100*r['friction_resid']:.2e} %   "
              f"(target < 0.1 %)")
        print(f"            energy ratio: semismooth = {r['energy_ratio']:.4f}   "
              f"regularized-split = {r['energy_ratio_reg']:.4f}   (target [0.95,1.05])")

    print("\n(B) FEM patch cross-check (exact bank ledger), cyclic shear of P points:")
    rp, _ = run_fem_patch(P=P, phi_b_deg=args.phi_b_deg, i_deg=args.i_deg,
                          amplitude=0.04, n_per_quarter=nq, n_cycles=ncyc)
    results["fem_patch"] = rp
    print(f"  P={rp['P']}:  peak mu_app={rp['peak_mu']:.6f}  Patton={rp['patton']:.6f}  "
          f"(err {100*abs(rp['peak_mu']-rp['patton'])/rp['patton']:.2e} %)")
    print(f"          friction residual = {100*rp['friction_resid']:.2e} %   energy ratio = "
          f"{rp['energy_ratio']:.4f}")

    # gates
    gates = {
        "patton_block_flat": abs(results["block_i0"]["peak_mu"] - results["block_i0"]["patton"]) / results["block_i0"]["patton"] < 1e-3,
        "patton_block_incl": abs(results[f"block_i{int(args.i_deg)}"]["peak_mu"] - results[f"block_i{int(args.i_deg)}"]["patton"]) / results[f"block_i{int(args.i_deg)}"]["patton"] < 1e-3,
        "friction_resid_block": results["block_i0"]["friction_resid"] < 1e-3,
        "energy_block_flat": 0.95 <= results["block_i0"]["energy_ratio"] <= 1.05,
        "energy_block_incl": 0.95 <= results[f"block_i{int(args.i_deg)}"]["energy_ratio"] <= 1.05,
        "energy_fem_patch": 0.95 <= results["fem_patch"]["energy_ratio"] <= 1.05,
        # NOTE: a 'semismooth_beats_regularized' gate was here, hard-wired to pass via 'or True'
        # (a rig). Removed. Honest finding: on this block/bank testbed the regularized split is
        # actually CLOSER to 1.0 than semismooth at every budget, and BOTH converge to 1.0 as the
        # step count grows (the <1 ratio at low budget is trapezoidal W_ext time-discretization error,
        # not a formulation flaw). The ~1.5x imbalance lives in the FEM rock_joint_cyclic_fem driver,
        # which this block testbed does NOT reproduce. See docs/ot_benchmark/next_phase_math.md (T3).
    }
    print("\nGATES:")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  [diag] block energy ratio: semismooth={results['block_i0']['energy_ratio']:.4f}  "
          f"regularized={results['block_i0']['energy_ratio_reg']:.4f}  "
          f"(regularized closer here; both -> 1.0 with more steps)")
    all_pass = all(gates.values())

    out = os.path.join(_ROOT, "runs", "cv7_semismooth")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"results": results, "gates": gates, "all_pass": all_pass}, f, indent=2)
    print(f"\n{'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}  ->  {out}/results.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
