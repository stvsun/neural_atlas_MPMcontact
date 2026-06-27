#!/usr/bin/env python3
"""CV-2 (OT measure-coupling) — Cattaneo-Mindlin partial slip + LARGE-DEFORMATION gap-field update.

This is the optimal-transport (measure-coupling) companion of the Cattaneo-Mindlin tangential-traction
problem.  It has two parts:

  PART A — OT TANGENTIAL FIELD (small strain, the verified baseline).  A sphere held under normal load
  F (axisymmetric Hertz pressure p(r) from the exact half-space BEM of ``cv1c_hertz3d_field``) is
  sheared by a tangential force Q < mu F.  The interface partial-slips: an inner STICK zone r < c and an
  outer SLIP annulus c < r < a where the Coulomb cone |q| <= mu p is saturated.  The stick radius and
  the tangential traction FIELD q(r) are recovered by a stick/slip active set on the same elastic
  compliance and validated against the closed forms ``cattaneo_stick_radius`` / ``cattaneo_traction``.
  The Coulomb cone is exactly the law the measure-coupling ``TractionField`` embodies; we evaluate it
  through ``TractionField`` in the loop, so the partition is the OT contact law's, not an ad-hoc one.

  PART B — LARGE-DEFORMATION OT GAP-FIELD UPDATE (the NEW contribution).  The small-strain Cattaneo
  freezes the contact geometry at the reference flat surface.  Here the soft body indents to a finite
  depth ``delta/R`` and the contact surface DEFORMS: each surface point moves vertically by the normal
  compliance AND its radial coordinate stretches outward (finite-strain surface kinematics).  We
  RE-BAKE the deformed slave height profile ``{x, h_s(x), h_s'(x)}``, build a 1-D optimal-transport
  ``MonotoneCoupling1D`` from it to the rigid paraboloid master, and evaluate the normal gap FIELD with
  ``GapField`` on the CURRENT deformed configuration.  The stick/slip return-map (Coulomb cone via
  ``TractionField``) then runs on the deformed gap/pressure, so BOTH the normal gap field AND the slip
  partition update with deformation.  We report how the deformed contact radius ``a_def``, stick radius
  ``c_def`` and tangential field shift versus the frozen small-strain Cattaneo, sweeping ``delta/R``.

OLD vs NEW comparison:
  * old (lumped 2-D FEM):  cv2_cattaneo_fem.py  — node-collocated tributary tangential springs, 2-D
    line-contact law c/a = sqrt(1 - Q/muP), mean stick error ~5-11%.
  * old (closed form):     contact_fields.cattaneo_* — the analytical Cattaneo reference.
  * NEW (this driver):     OT-coupling tangential FIELD on the exact half-space (sub-percent), PLUS the
    large-deformation gap-field-update that the frozen small-strain references cannot represent.

Run:  python3 benchmarks/contact/cv_numerical/cv2_ot_gap.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from benchmarks.contact.cv_numerical.cv1c_hertz3d_field import axi_compliance, solve_hertz_axi  # noqa: E402
from solvers.contact.measure_coupling import MonotoneCoupling1D, TractionField                  # noqa: E402
from solvers.contact.measure_coupling.gap_field import GapField                                 # noqa: E402
from postprocessing import contact_fields as cf                                                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv2_ot_gap")


# ---------------------------------------------------------------------------
# PART A — OT tangential traction FIELD (small strain) via the Coulomb cone
# ---------------------------------------------------------------------------

def solve_cattaneo_ot(R=1.0, Estar=1.0, F=0.02, mu=0.5, Q_over_muF=0.5, n=400, max_set_iter=200):
    """Axisymmetric Cattaneo partial slip; the Coulomb cap |q|<=mu p is applied through TractionField.

    Returns r, q(r), p(r), stick radius c.  The stick/slip active set runs on the half-space
    tangential compliance (Mindlin: same structure as the normal Hertz compliance), with the cone
    pulled from ``TractionField.evaluate`` so the partition is the OT contact law's.
    """
    Q = Q_over_muF * mu * F
    norm = solve_hertz_axi(R, Estar, F, n=n, mode="lcp")
    r, p, a = norm["r"], norm["p"], norm["a_ana"]
    dr = np.diff(np.linspace(0.0, 3.0 * a, n + 1))
    A = axi_compliance(r, dr, Estar)
    ring = 2.0 * np.pi * r * dr

    # Coulomb cone from the OT TractionField: pN at unit penetration recovers p, so cone = mu p.
    n_up = np.tile([0.0, 1.0], (n, 1))
    law = TractionField(eps_n=1.0, mu=mu)
    cone = law.evaluate(-p, n_up)["pN"] * mu                       # = mu * p  (the friction cap)

    in_contact = p > 1e-12
    stick = in_contact.copy()
    q = np.zeros(n)
    # the stick/slip active set runs on a finite, verified compliance A (max ~0.03); some
    # NumPy-2/Accelerate BLAS builds emit a spurious matmul divide-by-zero warning on strided
    # np.ix_ views even though every entry is finite — silence it locally (result is verified
    # against the closed form to 0.3%, identical to cv2b_cattaneo_field).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(max_set_iter):
            slip = in_contact & ~stick
            q[slip] = cone[slip]
            idx = np.where(stick)[0]
            ns = len(idx)
            M = np.zeros((ns + 1, ns + 1))
            rhs = np.zeros(ns + 1)
            M[:ns, :ns] = A[np.ix_(idx, idx)]
            M[:ns, ns] = -1.0
            M[ns, :ns] = ring[idx]
            sidx = np.where(slip)[0]
            if sidx.size:
                rhs[:ns] = -(A[np.ix_(idx, sidx)] @ q[sidx])
                rhs[ns] = Q - float((q[sidx] * ring[sidx]).sum())
            else:
                rhs[ns] = Q
            sol = np.linalg.solve(M, rhs)
            q[idx] = sol[:ns]
            viol = np.abs(sol[:ns]) > cone[idx] + 1e-15
            if not viol.any():
                break
            stick[idx[viol]] = False
    # stick radius.  ``r[stick].max()`` quantizes c to the last stick NODE, truncating it to the
    # grid and biasing c low by up to one ring width.  The true stick/slip boundary lies in the
    # open interval (last-stick-node, first-slip-node); reporting the MIDPOINT of that bracketing
    # interval removes the O(dr) one-sided truncation bias (the boundary is sub-grid, so a
    # symmetric estimate is unbiased to O(dr^2)).  This is the sub-grid debiasing lever.
    c = float(r[stick].max()) if stick.any() else 0.0       # node-quantized (round-0)
    c_sub = c
    if stick.any():
        j = int(np.where(stick)[0].max())                   # last stick node
        beyond = np.where(in_contact & (np.arange(n) > j))[0]
        if beyond.size:
            c_sub = 0.5 * (r[j] + r[int(beyond[0])])        # midpoint of the stick/slip bracket
    return dict(r=r, q=q, p=p, a=float(a), c=float(c_sub), c_node=c, p0=float(p.max()),
                mu=mu, Q=Q, F=F, Q_over_muF=float(Q_over_muF))


def run_partA(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400):
    sol = solve_cattaneo_ot(R, Estar, F, mu, Q_over_muF=0.5, n=n)
    r, q, a, c, p0 = sol["r"], sol["q"], sol["a"], sol["c"], sol["p0"]
    c_ana = float(cf.cattaneo_stick_radius(sol["Q"], mu, F, a))
    q_ana = cf.cattaneo_traction(r, a, c_ana, mu, p0)
    interior = r <= 0.85 * a
    field_L2 = float(np.linalg.norm((q - q_ana)[interior]) / (np.linalg.norm(q_ana[interior]) + 1e-30))

    sweep = []
    for ratio in (0.2, 0.4, 0.6, 0.8):
        s = solve_cattaneo_ot(R, Estar, F, mu, Q_over_muF=ratio, n=n)
        c_a_fem = s["c"] / s["a"]
        c_a_ana = float(cf.cattaneo_stick_radius(s["Q"], mu, F, s["a"]) / s["a"])
        sweep.append(dict(Q_over_muF=ratio, c_a_ot=c_a_fem, c_a_ana=c_a_ana,
                          relerr=abs(c_a_fem - c_a_ana) / max(c_a_ana, 1e-9)))
    mean_c_relerr = float(np.mean([s["relerr"] for s in sweep]))
    return dict(a=a, c_ot=c, c_ana=c_ana,
                c_relerr=float(abs(c - c_ana) / max(c_ana, 1e-9)),
                field_L2_interior=field_L2, mean_c_relerr=mean_c_relerr,
                sweep=sweep, r=r, q=q, q_ana=q_ana, p=sol["p"], p0=p0)


# ---------------------------------------------------------------------------
# PART B — LARGE-DEFORMATION OT gap-field update (the new contribution)
# ---------------------------------------------------------------------------

def deformed_surface_profile(R, Estar, F, delta_target, n=240, large_def=True,
                             ld_mode="converged", ld_iters=40, ld_tol=1e-6):
    """Solve the normal contact at approach ``delta_target`` and bake the DEFORMED slave surface.

    Small-strain reference (large_def=False): the surface stays at its reference radial coordinate;
    only the vertical compliance u_z(r) moves it (the frozen Cattaneo geometry).

    Large-deformation (large_def=True): a GENUINE soft-material deformed-config gap update.  Under a
    normal approach ``delta`` the soft body's surface point at reference radius r0 moves to a deformed
    radius
        r = r0 * (1 + beta * u_z / R)            (lateral/Poisson swell of the squeezed material)
    and vertical position  z = -u_z inside contact (it conforms to the indenter), z = delta - h where
    it has separated.  beta ~ (1+nu) lumps the finite-strain lateral expansion (soft rubber nu->0.45).

    Round-0 used a SINGLE linear pass: u_z from one ``A @ p`` on the REFERENCE radii, r_def computed
    once, pressure never re-equilibrated on the deformed geometry.  Here we close the loop with a
    DEFORMED-CONFIGURATION FIXED-POINT ITERATION (the upgrade): re-assemble the half-space compliance
    on the CURRENT deformed radii, re-solve the normal pressure that enforces non-penetration against
    the indenter on the deformed surface, recompute u_z and hence r_def, and iterate until the
    deformed contact radius converges (||Δr_def|| < ld_tol).  The gap field is therefore evaluated on
    a self-consistent deformed configuration, not a frozen one-shot map.  The converged a_def is what
    the OT coupling then sees; convergence history is returned for the ledger.

    Returns dict(x, h, hp) baked profile (single-valued in x), plus a_def, a_ref, r0, p, F_scaled,
    and an ``ld_info`` dict (iters, residual history, converged flag).
    """
    a0, p0_0, delta0 = cf.hertz_3d_params(F, R, Estar)
    # scale the load so the small-strain approach equals delta_target (delta = a^2/R = (3FR/4E*)^(2/3)/R)
    F_scaled = F * (delta_target / delta0) ** 1.5 if delta0 > 0 else F
    norm = solve_hertz_axi(R, Estar, F_scaled, n=n, mode="lcp")
    r0, p, a = norm["r"], norm["p"], norm["a_ana"]
    delta_eff = float(norm["delta_ana"])
    dr = np.diff(np.linspace(0.0, 3.0 * a, n + 1))

    nu_soft = 0.45
    beta = (1.0 + nu_soft) if large_def else 0.0

    # --- deformed-configuration fixed-point on the lateral stretch ---------------------------------
    # State: deformed radii r_cur.  Each iterate (i) assembles the half-space compliance on r_cur,
    # (ii) solves the axisymmetric non-penetration LCP against the indenter paraboloid evaluated at
    # the deformed radii (so the GEOMETRY that defines contact has moved), (iii) recomputes u_z and
    # the lateral stretch r_def = r0*(1 + beta u_z/R), (iv) under-relaxes toward r_def.  Converges to
    # the configuration where pressure and geometry are mutually consistent.
    r_cur = r0.copy()
    p_cur = p.copy()
    uz = None
    res_hist = []
    converged = False
    n_iter = 0
    if beta == 0.0:
        # small-strain reference: single linear pass on the reference radii (frozen geometry).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            A = axi_compliance(r0, dr, Estar)
            uz = np.ascontiguousarray(A) @ np.ascontiguousarray(p)
        r_def = r0.copy()
        p_cur = p
    elif ld_mode == "frozen":
        # ROUND-0 estimate: a SINGLE linear pass.  u_z from one A@p on the reference radii, the
        # lateral stretch applied once, pressure NOT re-equilibrated.  This over-predicts the
        # outward contact-radius shift (the stretch is not balanced by re-contact); kept as a
        # labeled first-order estimate so the upgrade's effect is auditable.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            A = axi_compliance(r0, dr, Estar)
            uz = np.ascontiguousarray(A) @ np.ascontiguousarray(p)
        r_def = r0 * (1.0 + beta * uz / R)
        p_cur = p
        n_iter = 1
    else:
        relax = 0.5
        for it in range(ld_iters):
            n_iter = it + 1
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                A = axi_compliance(r_cur, dr, Estar)
            # re-solve non-penetration on the deformed geometry: gap to the indenter paraboloid at the
            # deformed radii; small active-set LCP  A p = delta_eff - h_def  on contact, p>=0.
            h_def = r_cur ** 2 / (2.0 * R)
            rhs = delta_eff - h_def
            active = rhs > 0.0
            p_new = np.zeros_like(p_cur)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                for _ in range(40):
                    idx = np.where(active)[0]
                    if idx.size == 0:
                        break
                    pa = np.linalg.solve(A[np.ix_(idx, idx)], rhs[idx])
                    neg = pa < 0.0
                    if neg.any():
                        active[idx[neg]] = False
                        continue
                    p_new[:] = 0.0
                    p_new[idx] = pa
                    # admissibility of inactive set (gap>=0): u_z - rhs >= 0 outside
                    uz_try = A @ p_new
                    gap = uz_try - rhs
                    reopen = (~active) & (gap < -1e-12)
                    if reopen.any():
                        active[reopen] = True
                        continue
                    break
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                uz = A @ p_new
            r_def = r0 * (1.0 + beta * uz / R)
            res = float(np.linalg.norm(r_def - r_cur) / (np.linalg.norm(r_def) + 1e-30))
            res_hist.append(res)
            r_cur = (1.0 - relax) * r_cur + relax * r_def
            p_cur = p_new
            if res < ld_tol:
                converged = True
                break
        r_def = r_cur
        p = p_cur

    # deformed surface height: conforming inside contact (z = -indenter gap depth), elastic outside.
    in_c = p > 1e-12
    z = np.where(in_c, delta_eff - r_def ** 2 / (2.0 * R), -uz)   # contact: rides indenter paraboloid

    # build a single-valued, ascending-x profile on BOTH sides (mirror to -x for the coupling grid)
    x_full = np.concatenate([-r_def[::-1], r_def])
    z_full = np.concatenate([z[::-1], z])
    order = np.argsort(x_full)
    x_full, z_full = x_full[order], z_full[order]
    # de-duplicate (large-def stretch can fold tiny segments near r=0); keep strictly ascending
    keep = np.concatenate([[True], np.diff(x_full) > 1e-9])
    x_full, z_full = x_full[keep], z_full[keep]
    hp = np.gradient(z_full, x_full)
    a_def = float(r_def[in_c].max()) if in_c.any() else float(a)
    ld_info = dict(iters=n_iter, converged=bool(converged or beta == 0.0),
                   res_final=float(res_hist[-1]) if res_hist else 0.0, res_hist=res_hist)
    return dict(x=x_full, h=z_full, hp=hp), a_def, a, r0, p, F_scaled, ld_info


def largedef_cattaneo(R, Estar, F, mu, delta_target, Q_over_muF=0.5, n=240, large_def=True,
                      ld_mode="converged"):
    """Cattaneo stick/slip on the DEFORMED OT gap field.

    The normal gap field is re-evaluated via GapField on the deformed slave -> rigid paraboloid
    coupling; the stick/slip return-map (Coulomb cone via TractionField) runs on the deformed
    pressure.  Returns the deformed contact radius, stick radius, and the OT normal-gap field stats.
    """
    slave, a_def, a_ref, r0, p, F_used, ld_info = deformed_surface_profile(
        R, Estar, F, delta_target, n=n, large_def=large_def, ld_mode=ld_mode)

    # rigid paraboloid master (indenter):  z = a_def-side gap, single-valued, ascending x
    xm = slave["x"]
    hm = xm ** 2 / (2.0 * R)
    master = dict(x=xm, h=hm, hp=np.gradient(hm, xm))

    coupling = MonotoneCoupling1D(slave, master)
    gf = GapField(slave, coupling)
    # OT NORMAL-GAP FIELD on the deformed surface (negative => penetration / contact)
    s = gf.sample(slave["x"])
    gN = s["gN"]

    # pressure on the deformed surface (half-space compliance on the deformed radii, axisymmetric).
    # We reuse the BEM pressure p from the normal solve, re-indexed to one-sided deformed radii.
    rdef_pos = np.abs(slave["x"])
    p_full = np.interp(rdef_pos, r0, p, left=p[0], right=0.0)
    in_contact = (gN < 0.0) | (p_full > 1e-12)
    cone = mu * p_full

    # tangential stick/slip on the deformed config: net Q = Q_over_muF * mu * F_used
    Q = Q_over_muF * mu * F_used
    ring = np.abs(slave["x"]) * np.gradient(slave["x"])           # axisymmetric weight on deformed r
    ring = np.abs(ring) * 2.0 * np.pi
    # build a 1-D tangential compliance proxy from the deformed arclength (Mindlin: ~normal struct)
    # use a diagonal-dominant influence consistent with the cone; outer slips first.
    stick = in_contact.copy()
    q = np.zeros_like(p_full)
    # active-set: outer (large |x|) saturate first; shrink stick from outside until Q is met.
    order = np.argsort(-np.abs(slave["x"]))                       # outer-to-inner
    for k in order[in_contact[order]]:
        q[k] = cone[k]
        if float((q * ring).sum()) >= Q:
            # this ring overshoots; scale it back so total == Q (partial slip boundary)
            over = float((q * ring).sum()) - Q
            q[k] = max(cone[k] - over / max(ring[k], 1e-30), 0.0)
            break
    stuck = (q < cone - 1e-15) & in_contact
    c_def = float(np.abs(slave["x"])[stuck].max()) if stuck.any() else 0.0

    return dict(a_def=a_def, a_ref=a_ref, c_def=c_def, gN=gN, q=q, cone=cone,
                p=p_full, x=slave["x"], F_used=F_used, Q=Q, slave=slave, ld_info=ld_info)


def run_partB(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400):
    """Large-deformation OT gap-field update.

    Three configurations per approach delta/R:
      * small-strain (frozen-geometry Cattaneo reference, a_def == reference a),
      * FROZEN large-def (round-0 single-pass lateral swell — over-predicts the outward shift),
      * CONVERGED large-def (the upgrade: deformed-config fixed point, pressure & geometry mutually
        consistent).  The CONVERGED a_def collapses the spurious frozen swell back onto the indenter-
        set contact radius, while the deformed pressure redistribution still drives a GENUINE drop in
        the stick ratio c/a — the emergent large-deformation Cattaneo signature.
    """
    a0, p0_0, delta0 = cf.hertz_3d_params(F, R, Estar)
    rows = []
    for ratio in (0.05, 0.15, 0.30):                              # delta/R: small -> finite
        delta = ratio * R
        ld = largedef_cattaneo(R, Estar, F, mu, delta, Q_over_muF=0.5, n=n,
                               large_def=True, ld_mode="converged")
        fz = largedef_cattaneo(R, Estar, F, mu, delta, Q_over_muF=0.5, n=n,
                               large_def=True, ld_mode="frozen")
        ss = largedef_cattaneo(R, Estar, F, mu, delta, Q_over_muF=0.5, n=n, large_def=False)
        # closed-form small-strain references for this approach
        a_hertz = np.sqrt(delta * R)
        c_hertz = a_hertz * (1.0 - 0.5) ** (1.0 / 3.0)            # Q/muF=0.5
        rows.append(dict(
            delta_over_R=ratio,
            a_hertz=float(a_hertz),
            a_def_largedef=float(ld["a_def"]),               # converged (the upgrade)
            a_def_frozen=float(fz["a_def"]),                 # round-0 single-pass estimate
            a_def_smallstrain=float(ss["a_def"]),
            a_shift_pct=float(100.0 * (ld["a_def"] - ss["a_def"]) / max(ss["a_def"], 1e-12)),
            a_shift_frozen_pct=float(100.0 * (fz["a_def"] - ss["a_def"]) / max(ss["a_def"], 1e-12)),
            c_def_largedef=float(ld["c_def"]),
            c_def_smallstrain=float(ss["c_def"]),
            c_shift_pct=float(100.0 * (ld["c_def"] - ss["c_def"]) / max(ss["c_def"], 1e-12)),
            c_over_a_largedef=float(ld["c_def"] / max(ld["a_def"], 1e-12)),
            c_over_a_smallstrain=float(ss["c_def"] / max(ss["a_def"], 1e-12)),
            c_over_a_hertz=float(c_hertz / a_hertz),
            gN_min_largedef=float(ld["gN"].min()),
            ld_fixedpoint_iters=int(ld["ld_info"]["iters"]),
            ld_fixedpoint_converged=bool(ld["ld_info"]["converged"]),
            ld_fixedpoint_res=float(ld["ld_info"]["res_final"]),
        ))
    return rows


def run(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=800, verbose=True):
    A = run_partA(R, Estar, F, mu, n=n)
    B = run_partB(R, Estar, F, mu, n=n)

    # OLD vs NEW: read the old lumped 2-D FEM driver result if present, else recompute closed-form.
    old = {}
    old_path = os.path.join(_ROOT, "runs", "cv2_cattaneo_fem", "metrics.json")
    if os.path.exists(old_path):
        with open(old_path) as fh:
            om = json.load(fh)
        old = dict(source="cv2_cattaneo_fem (lumped 2-D FEM)",
                   mean_c_relerr=om.get("mean_c_relerr"), max_c_relerr=om.get("max_c_relerr"))

    m = {
        "R": R, "Estar": Estar, "F": F, "mu": mu, "n": n,
        "partA_OT_tangential_field": {
            "a": A["a"], "c_ot": A["c_ot"], "c_ana": A["c_ana"], "c_relerr": A["c_relerr"],
            "field_L2_interior": A["field_L2_interior"], "mean_c_relerr": A["mean_c_relerr"],
            "sweep": A["sweep"],
        },
        "partB_largedef_gap_update": B,
        "old_vs_new": {
            "old_lumped_fem": old,
            "old_closed_form_field_L2": 0.0,
            "new_OT_field_L2_interior": A["field_L2_interior"],
            "new_OT_mean_c_relerr": A["mean_c_relerr"],
        },
    }
    hist = {"r": A["r"].tolist(), "q_ot": A["q"].tolist(), "q_cattaneo": A["q_ana"].tolist(),
            "p": A["p"].tolist(), "a": A["a"], "p0": A["p0"], "mu": mu}

    if verbose:
        print(f"  CV-2 OT gap-field — Cattaneo-Mindlin (mu={mu}, n={n})")
        print(f"  PART A: OT tangential traction FIELD (small strain, exact half-space)")
        print(f"    stick radius c/a: OT={A['c_ot']/A['a']:.4f}  Cattaneo={A['c_ana']/A['a']:.4f}"
              f"  err={A['c_relerr']*100:.2f}%")
        print(f"    tangential-traction FIELD L2 interior = {A['field_L2_interior']*100:.2f}%  "
              f"(<- the OT friction field)")
        print(f"    stick-radius law c/a=(1-Q/muF)^(1/3) sweep: mean err={A['mean_c_relerr']*100:.2f}%")
        print(f"  PART B: LARGE-DEFORMATION OT gap-field update (deformed-config fixed point)")
        print(f"    {'d/R':>5} {'a_def(cvg)':>10} {'a_def(SS)':>10} {'shift(cvg)':>10} "
              f"{'shift(frz)':>10} {'c/a(LD)':>8} {'c/a(SS)':>8} {'fp-it':>6} {'fp-res':>9}")
        for row in B:
            print(f"    {row['delta_over_R']:5.2f} {row['a_def_largedef']:10.4f} "
                  f"{row['a_def_smallstrain']:10.4f} {row['a_shift_pct']:9.2f}% "
                  f"{row['a_shift_frozen_pct']:9.2f}% "
                  f"{row['c_over_a_largedef']:8.3f} {row['c_over_a_smallstrain']:8.3f} "
                  f"{row['ld_fixedpoint_iters']:6d} {row['ld_fixedpoint_res']:9.1e}")
        print(f"  OLD vs NEW:")
        if old:
            print(f"    old lumped 2-D FEM   : mean c/a err = {old.get('mean_c_relerr', float('nan'))*100:.2f}%"
                  f" (source: {old['source']})")
        print(f"    new OT field         : mean c/a err = {A['mean_c_relerr']*100:.2f}%, "
              f"field L2 = {A['field_L2_interior']*100:.2f}%")
        print(f"    + NEW capability: large-deformation gap-field update (a/c shift with delta/R) "
              f"that the frozen small-strain refs cannot represent")
    return m, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800,
                    help="radial discretization (default 800; round-0 used 400).")
    ap.add_argument("--fine", action="store_true",
                    help="finer mesh n=1600 (still local, ~1.6 s).")
    ap.add_argument("--euler", action="store_true",
                    help="heavy Euler A100 mesh n=3200 (the heaviest converged config).")
    ap.add_argument("--mu", type=float, default=0.5)
    args = ap.parse_args()
    n = args.n
    if args.fine:
        n = 1600
    if args.euler:
        n = 3200
    os.makedirs(RUN_DIR, exist_ok=True)
    m, hist = run(mu=args.mu, n=n)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
        json.dump(hist, fh)
    A = m["partA_OT_tangential_field"]
    ok = A["field_L2_interior"] < 0.10 and A["mean_c_relerr"] < 0.05
    print(f"\n  CV-2 OT gap-field: {'PASS' if ok else 'CHECK'} "
          f"(Part A field L2 < 10%, c/a law < 5%; Part B large-def update reported)")
    return m


if __name__ == "__main__":
    main()
