"""Head-to-head: level-set-free radial-chart detector vs the neural-SDF detector.

Runs BOTH detectors on shared scenes and prints one comparison table.  The SDF
model (``solvers/contact/gap.py::evaluate_gap``, here with the analytic
``SphereSDF``) is the acceptance oracle; the chart detector
(``solvers/contact/chart_gap.py::evaluate_gap_chart``) is the candidate.

Scenes
------
1. Analytic sphere probe cloud   — chart == SDF to machine precision (the
   falsifiable headline: phi = |x-c|-R and rho = R are both exact).
2. 3D superquadric static cloud   — active-set agreement (sign exactness),
   conservative magnitude |gap_rad| >= |gap_perp|, and normal-angle error vs the
   true surface normal -> 0 as penetration -> 0.
3. Two-sphere collision MPM       — swap the detector in the live MPM contact
   path; momentum drift, contact impulse, and penetration must match the SDF run.
4. CV-1 Hertz geometry            — detected contact-circle radius is geometrically
   exact and identical for both detectors; cross-referenced to the closed-form
   Hertz contact radius (``postprocessing/contact_fields.py::hertz_3d_params``).

Run::

    python3 benchmarks/contact/chart_vs_sdf_detection.py

Writes ``runs/chart_vs_sdf_detection/summary.json``.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from postprocessing.contact_fields import hertz_3d_params
from solvers.contact.chart_gap import (
    SphereRho,
    SuperquadricRho,
    evaluate_gap_chart,
    closest_point_refine_chart,
)
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.mpm.particles import MaterialPointCloud

torch.set_default_dtype(torch.float64)


# ── Analytic SDF helpers (kept local to avoid importing the test module) ──

class SphereSDF(torch.nn.Module):
    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        super().__init__()
        self.register_buffer("center", torch.tensor(center, dtype=torch.float64))
        self.radius = radius

    def update_center(self, c):
        self.center.copy_(c)

    def forward(self, x):
        return (x - self.center.unsqueeze(0)).norm(dim=1) - self.radius


def _angle(n_a, n_b):
    cos = (n_a * n_b).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos)


# ── Scene 1: analytic sphere ──────────────────────────────────────────

def scene_sphere():
    center = (1.0, 2.0, 3.0)
    R = 0.5
    chart = SphereRho(R, center=center)
    sdf = SphereSDF(center=center, radius=R)
    c = torch.tensor(center)
    torch.manual_seed(0)
    dirs = torch.randn(2000, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    rad = 0.1 + 1.5 * torch.rand(2000)
    x = c.unsqueeze(0) + rad.unsqueeze(1) * dirs

    g_c, n_c = evaluate_gap_chart(x, chart)
    g_s, n_s = evaluate_gap(x, sdf)
    # 1 - cos avoids the sqrt-sensitivity of acos near cos = 1 (the angle floors
    # at ~2e-8 rad by conditioning even when the normals are machine-equal).
    one_minus_cos = (1.0 - (n_c * n_s).sum(dim=1)).abs().max().item()
    return {
        "scene": "1. analytic sphere",
        "max_gap_diff": (g_c - g_s).abs().max().item(),
        "max_normal_1mcos": one_minus_cos,
        "max_normal_angle_rad": _angle(n_c, n_s).max().item(),
        "active_set_agree": float((g_c < 0).eq(g_s < 0).float().mean()),
    }


# ── Scene 2: 3D superquadric ──────────────────────────────────────────

def scene_superquadric():
    S = torch.tensor([0.6, 0.9, 0.4])
    ctr = torch.tensor([0.1, -0.2, 0.05])
    ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                          center=tuple(ctr.tolist()))
    torch.manual_seed(1)
    x = ctr + 0.8 * torch.randn(5000, 3)

    g_rad, _ = evaluate_gap_chart(x, ell)
    f_ana = (((x - ctr) / S) ** 2).sum(dim=1) - 1.0       # true ellipsoid sign
    g_ref, _ = closest_point_refine_chart(x, ell)

    # normal-angle vs true surface normal at two penetration depths
    u = torch.randn(400, 3)
    u = u / u.norm(dim=1, keepdim=True)
    rho = ell.radius(u)

    def normal_angle(frac):
        xp = ctr + (rho * (1.0 + frac)).unsqueeze(1) * u
        _, n_rad = evaluate_gap_chart(xp, ell)
        _, n_surf = closest_point_refine_chart(xp, ell)
        return _angle(n_rad, n_surf).max().item()

    return {
        "scene": "2. superquadric",
        "active_set_vs_true_sdf": float((g_rad < 0).eq(f_ana < 0).float().mean()),
        "conservative_mag_holds": float(
            (g_rad.abs() >= g_ref.abs() - 1e-12).float().mean()
        ),
        "normal_angle_deg_at_10pct": math.degrees(normal_angle(0.10)),
        "normal_angle_deg_at_contact": math.degrees(normal_angle(0.001)),
    }


# ── Scene 3: two-sphere collision MPM, detector swapped ──────────────

def _run_two_sphere(detector, n_steps):
    density, eps_n, n_per_axis = 1000.0, 5e7, 3
    R, v_app = 0.07, 0.2
    s = R
    cA0 = torch.tensor([-s, 0.0, 0.0])
    cB0 = torch.tensor([+s, 0.0, 0.0])

    if detector == "sdf":
        obs_A, obs_B = SphereSDF(cA0.tolist(), R), SphereSDF(cB0.tolist(), R)
        detect = lambda x, o: evaluate_gap(x, o)
        set_c = lambda o, c: o.update_center(c)
    else:
        obs_A, obs_B = SphereRho(R, cA0.tolist()), SphereRho(R, cB0.tolist())
        detect = lambda x, o: evaluate_gap_chart(x, o)
        set_c = lambda o, c: setattr(o, "center", c.clone())

    solver_A = ChartMPMSolver(n_cells=16, extent=0.3, gravity=None, bc_type="free")
    solver_B = ChartMPMSolver(n_cells=16, extent=0.3, gravity=None, bc_type="free")
    pA = MaterialPointCloud.create_uniform(n_per_axis=n_per_axis, extent=R, density=density)
    pA.xi += cA0.unsqueeze(0); pA.v[:, 0] = +v_app
    pB = MaterialPointCloud.create_uniform(n_per_axis=n_per_axis, extent=R, density=density)
    pB.xi += cB0.unsqueeze(0); pB.v[:, 0] = -v_app

    m_min = min(pA.mass.min().item(), pB.mass.min().item())
    dt = min(1e-5, contact_stable_dt(eps_n, m_min))

    def com(p):
        m = p.mass.unsqueeze(1)
        return (m * p.xi).sum(0) / p.mass.sum()

    def vcom(p):
        m = p.mass.unsqueeze(1)
        return (m * p.v).sum(0) / p.mass.sum()

    impulse = 0.0
    min_gap = 0.0
    for _ in range(n_steps):
        set_c(obs_A, com(pA)); set_c(obs_B, com(pB))
        gA, nA = detect(pA.xi.detach(), obs_B)
        gB, nB = detect(pB.xi.detach(), obs_A)
        min_gap = min(min_gap, float(gA.min()), float(gB.min()))
        cfA = compute_contact_force(gA, nA, pA.current_volume, eps_n)
        cfB = compute_contact_force(gB, nB, pB.current_volume, eps_n)
        impulse += float(cfA.sum(0).norm()) * dt
        solver_A.step(pA, dt, contact_force=cfA)
        solver_B.step(pB, dt, contact_force=cfB)
        mom = pA.mass.sum() * vcom(pA) + pB.mass.sum() * vcom(pB)

    return {
        "vA_x": vcom(pA)[0].item(),
        "vB_x": vcom(pB)[0].item(),
        # |total P_x|: ~0 here only because the setup is symmetric (P_init = 0).
        # NOT a general conservation proof (the two balls use separate solvers);
        # the point of this scene is sdf-vs-chart transparency, below.
        "total_Px_symmetric": abs(mom[0].item()),
        "contact_impulse": impulse,
        "max_penetration": -min_gap,
    }


def scene_two_sphere(n_steps=2000):
    sdf = _run_two_sphere("sdf", n_steps)
    chart = _run_two_sphere("chart", n_steps)
    return {
        "scene": "3. two-sphere MPM",
        "n_steps": n_steps,
        "vA_diff": abs(sdf["vA_x"] - chart["vA_x"]),
        "vB_diff": abs(sdf["vB_x"] - chart["vB_x"]),
        "impulse_rel_diff": abs(sdf["contact_impulse"] - chart["contact_impulse"])
        / (abs(sdf["contact_impulse"]) + 1e-30),
        "penetration_diff": abs(sdf["max_penetration"] - chart["max_penetration"]),
        "sdf_total_Px_symmetric": sdf["total_Px_symmetric"],
        "chart_total_Px_symmetric": chart["total_Px_symmetric"],
        "sdf_vA_x": sdf["vA_x"], "chart_vA_x": chart["vA_x"],
    }


# ── Scene 4: CV-1 Hertz contact geometry ─────────────────────────────

def scene_hertz():
    # Hertz reference (two equal spheres of radius Rb): R_eff = Rb/2.
    Rb = 1.0
    F, R_eff, Estar = 5.0, Rb / 2.0, 100.0
    a_hertz, p0, delta = hertz_3d_params(F, R_eff, Estar)

    # Two spheres pressed to the Hertz approach delta (centers 2Rb - delta apart).
    sep = Rb - delta / 2.0
    sdf_A = SphereSDF((-sep, 0.0, 0.0), Rb)
    sdf_B = SphereSDF((+sep, 0.0, 0.0), Rb)
    ch_A = SphereRho(Rb, (-sep, 0.0, 0.0))
    ch_B = SphereRho(Rb, (+sep, 0.0, 0.0))

    # Sample the contact mid-plane (x = 0) and find the lens {gap_A<0 & gap_B<0}.
    n = 400
    rr = torch.linspace(0.0, 0.5, n)
    pts = torch.stack([torch.zeros(n), rr, torch.zeros(n)], dim=1)

    def lens_radius(detect, A, B):
        gA, _ = detect(pts, A)
        gB, _ = detect(pts, B)
        inside = (gA < 0) & (gB < 0)
        idx = torch.where(inside)[0]
        return rr[idx.max()].item() if idx.numel() else 0.0

    a_sdf = lens_radius(lambda x, o: evaluate_gap(x, o), sdf_A, sdf_B)
    a_chart = lens_radius(lambda x, o: evaluate_gap_chart(x, o), ch_A, ch_B)
    a_geom = math.sqrt(max(Rb * delta - (delta / 2.0) ** 2, 0.0))  # exact lens radius

    return {
        "scene": "4. Hertz CV-1 geometry",
        "hertz_a": a_hertz, "hertz_p0": p0, "hertz_delta": delta,
        "detected_a_sdf": a_sdf, "detected_a_chart": a_chart,
        "chart_vs_sdf_radius_diff": abs(a_sdf - a_chart),
        "detected_vs_geom_lens_diff": abs(a_chart - a_geom),
    }


# ── Driver ───────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join("runs", "chart_vs_sdf_detection")
    os.makedirs(out_dir, exist_ok=True)

    results = [scene_sphere(), scene_superquadric(), scene_two_sphere(), scene_hertz()]

    print("\n" + "=" * 74)
    print("CHART (level-set-free) vs SDF detector — head to head")
    print("=" * 74)
    for r in results:
        print(f"\n[{r['scene']}]")
        for k, v in r.items():
            if k == "scene":
                continue
            print(f"    {k:<28s} {v:.6g}" if isinstance(v, float) else f"    {k:<28s} {v}")

    # PASS/FAIL gates
    s1, s2, s3, s4 = results
    checks = [
        ("sphere gap == SDF (< 1e-12)", s1["max_gap_diff"] < 1e-12),
        ("sphere normal == SDF (1-cos < 1e-12)", s1["max_normal_1mcos"] < 1e-12),
        ("superquadric active-set == true SDF", s2["active_set_vs_true_sdf"] == 1.0),
        ("superquadric |g_rad|>=|g_perp|", s2["conservative_mag_holds"] == 1.0),
        ("normal angle shrinks at contact",
         s2["normal_angle_deg_at_contact"] < s2["normal_angle_deg_at_10pct"]),
        ("two-sphere vA chart==sdf (< 1e-9)", s3["vA_diff"] < 1e-9),
        ("two-sphere impulse chart==sdf (< 1e-6 rel)", s3["impulse_rel_diff"] < 1e-6),
        ("Hertz radius chart==sdf (< 2e-3)", s4["chart_vs_sdf_radius_diff"] < 2e-3),
        ("Hertz detected==geom lens (< 2e-3)", s4["detected_vs_geom_lens_diff"] < 2e-3),
    ]
    print("\n" + "-" * 74)
    all_ok = True
    for name, ok in checks:
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 74)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"results": results,
                   "checks": {n: bool(o) for n, o in checks}},
                  f, indent=2)
    print(f"\nSaved {os.path.join(out_dir, 'summary.json')}")
    return all_ok


if __name__ == "__main__":
    main()
