"""Rigid-body contact: a rotating nonconvex superformula 'cam' drives a free particle.

Two star-shaped Gielis-superformula bodies (solvers/contact/supershape.py). Body A is a
driver: fixed center, prescribed clockwise spin alpha_A(t) = alpha_A0 - omega*t. Body B is
free (mass, inertia) and is pushed/spun by contact with A's lobes.

Contact uses the chart inverse radial gap + matched normal (the transition-map kinematics):
a two-pass node-to-surface penalty + regularized Coulomb friction. Each contact produces a
Newton-3 force pair, so in the --free-A control variant total momentum is conserved; in the
driven cam the axle supplies reaction (momentum NOT conserved by design) and we instead check
work-energy balance on B.

Penalty f = eps_n <-g> n ds and the regularized Coulomb friction mirror
solvers/contact/{penalty,friction}.py (re-expressed in 2D numpy). Output:
runs/supershape_cam_drive[/_free_A]/history.json (mirrors benchmarks/contact/two_sphere_collision_mpm.py).

Run:  python3 benchmarks/contact/supershape_cam_drive.py
      python3 benchmarks/contact/supershape_cam_drive.py --free-A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solvers.contact import supershape as ss   # noqa: E402


@dataclass
class Body:
    params: ss.SuperParams
    c: np.ndarray
    alpha: float = 0.0
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0
    mass: float = 1.0
    inertia: float = 1.0
    chart: object = None          # optional trained NeuralRho2D radial chart (neural detection)

    @classmethod
    def make(cls, params, c, density=1.0, **kw):
        m, I = ss.polar_inertia(params, density=density)
        return cls(params=params, c=np.asarray(c, float), mass=m, inertia=I, **kw)

    def vel_at(self, pts):
        r = pts - self.c
        return self.v + self.omega * np.stack([-r[..., 1], r[..., 0]], axis=-1)

    def max_radius(self):
        th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        return float(ss.radius(th, self.params).max())


def _count_arcs(active):
    if not active.any():
        return 0
    a = active.astype(int)
    rising = int(np.sum((a - np.roll(a, 1)) == 1))
    return rising if rising > 0 else 1


def _contact_pass(intr: Body, obst: Body, theta, dtheta, eps_n, mu, eps_t):
    """Sample intruder boundary, penalize points inside the obstacle.

    Returns dict with forces/torques ON the intruder and ON the obstacle (Newton-3),
    plus n_contacts, max_penetration, friction power (<=0), normal-force sum.
    """
    pts = ss.boundary(theta, intr.c, intr.alpha, intr.params)          # (N,2)
    if obst.chart is not None:
        # NEURAL detection: the obstacle's gap/normal come from a trained radial chart
        # (transition-map detector) instead of the analytical inverse radial gap.  grad is the
        # matched unit normal (re-normalised below) — same (gap, normal) contract.
        import torch
        from solvers.contact.radial_chart_2d import evaluate_radial_gap_2d
        gt, nt = evaluate_radial_gap_2d(torch.as_tensor(pts, dtype=torch.float64),
                                        obst.chart, center=obst.c, alpha=obst.alpha)
        g, grad = gt.numpy(), nt.numpy()
    else:
        g, grad = ss.radial_gap(pts, obst.c, obst.alpha, obst.params)
    active = g < 0.0
    out = dict(F_intr=np.zeros(2), T_intr=0.0, F_obst=np.zeros(2), T_obst=0.0,
               n_contacts=_count_arcs(active), max_pen=0.0, fric_power=0.0, fn_sum=0.0)
    if not active.any():
        return out
    pa = pts[active]
    ga = g[active]
    n_obs = grad[active] / np.clip(np.linalg.norm(grad[active], axis=1, keepdims=True), 1e-12, None)
    tang = ss.tangent(theta, intr.alpha, intr.params)
    ds = np.linalg.norm(tang, axis=1) * dtheta
    fn = eps_n * (-ga) * ds[active]                                    # >=0 normal magnitude
    fN = fn[:, None] * n_obs                                           # normal force on intruder
    # relative tangential velocity (intruder surface rel obstacle surface)
    vrel = intr.vel_at(pa) - obst.vel_at(pa)
    vt = vrel - np.sum(vrel * n_obs, axis=1, keepdims=True) * n_obs
    ft = -mu * fn[:, None] * vt / np.sqrt(np.sum(vt ** 2, axis=1, keepdims=True) + eps_t ** 2)
    f_intr = fN + ft
    f_obst = -f_intr                                                   # Newton's third law
    ri = pa - intr.c
    ro = pa - obst.c
    out["F_intr"] = f_intr.sum(0)
    out["F_obst"] = f_obst.sum(0)
    out["T_intr"] = float(np.sum(ri[:, 0] * f_intr[:, 1] - ri[:, 1] * f_intr[:, 0]))
    out["T_obst"] = float(np.sum(ro[:, 0] * f_obst[:, 1] - ro[:, 1] * f_obst[:, 0]))
    out["max_pen"] = float((-ga).max())
    out["fric_power"] = float(np.sum(ft * vt))                         # <= 0
    out["fn_sum"] = float(fn.sum())
    return out


def step_forces(A, B, theta, dtheta, eps_n, mu, eps_t):
    p1 = _contact_pass(A, B, theta, dtheta, eps_n, mu, eps_t)          # intruder A, obstacle B
    p2 = _contact_pass(B, A, theta, dtheta, eps_n, mu, eps_t)          # intruder B, obstacle A
    F_A = p1["F_intr"] + p2["F_obst"]
    T_A = p1["T_intr"] + p2["T_obst"]
    F_B = p1["F_obst"] + p2["F_intr"]
    T_B = p1["T_obst"] + p2["T_intr"]
    diag = dict(
        # de-duplicated arc count: the same physical contact region is seen by both
        # passes, so take the max (a single-foot CPP would report 1)
        n_contacts=max(p1["n_contacts"], p2["n_contacts"]),
        max_pen=max(p1["max_pen"], p2["max_pen"]),
        fric_power=p1["fric_power"] + p2["fric_power"],
        fn_sum=p1["fn_sum"] + p2["fn_sum"],
    )
    return F_A, T_A, F_B, T_B, diag


def simulate(free_A=False, n_steps=3000, omega_drive=4.0, eps_n=2.0e4, mu=0.3,
             eps_t=1e-3, n_samples=1400, seed_push=2.0, charts=None):
    # --- two nonconvex supershapes (n<1 -> concave lobes) ---
    pA = ss.SuperParams(m=4, n1=0.8, n2=0.8, n3=0.8, scale=1.7)        # 4-lobed cam
    pB = ss.SuperParams(m=7, n1=0.8, n2=0.8, n3=0.8, scale=1.0)        # 7-lobed follower
    A = Body.make(pA, c=[0.0, 0.0], density=1.0)
    B = Body.make(pB, c=[2.45, 0.15], density=1.0)
    if charts is not None:                                            # NEURAL detection (chartA, chartB)
        A.chart, B.chart = charts

    if free_A:
        A.v = np.array([seed_push, 0.0]); A.omega = 0.0; A.alpha = 0.0  # B pushed into A
        B.v = np.array([0.0, 0.0])
        mu = 0.0                                                        # control: frictionless
    else:
        A.alpha = 0.0; A.omega = -omega_drive                          # clockwise driver

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    dtheta = 2.0 * np.pi / n_samples

    # stable dt: penalty + rotation anti-tunneling
    L = B.max_radius() + np.linalg.norm(B.c - A.c)
    m_eff = min(B.mass, B.inertia / L ** 2)
    dt_pen = 0.5 * np.sqrt(m_eff / (2.0 * eps_n))                      # 2 eps_n (two-pass)
    ds_min = (np.linalg.norm(ss.tangent(theta, A.alpha, A.params), axis=1).min()) * dtheta
    dt_rot = 0.4 * ds_min / max(abs(omega_drive) * A.max_radius(), 1e-9) if not free_A else 1e9
    dt = float(min(dt_pen, dt_rot, 5e-4))

    history = []
    impulse = 0.0
    e_inj = 0.0
    e_dis = 0.0
    ke0 = 0.5 * B.mass * B.v @ B.v + 0.5 * B.inertia * B.omega ** 2

    def total_linmom():
        return A.mass * A.v + B.mass * B.v

    def total_angmom():
        la = A.inertia * A.omega + A.mass * (A.c[0] * A.v[1] - A.c[1] * A.v[0])
        lb = B.inertia * B.omega + B.mass * (B.c[0] * B.v[1] - B.c[1] * B.v[0])
        return la + lb

    p0 = total_linmom(); l0 = total_angmom()

    for i in range(n_steps):
        t = (i + 1) * dt
        if not free_A:
            A.alpha = -omega_drive * t
            A.omega = -omega_drive

        F_A, T_A, F_B, T_B, d = step_forces(A, B, theta, dtheta, eps_n, mu, eps_t)

        # integrate B (always) and A (free control only) -- semi-implicit Euler
        B.v = B.v + dt * F_B / B.mass
        B.omega = B.omega + dt * T_B / B.inertia
        B.c = B.c + dt * B.v
        B.alpha = B.alpha + dt * B.omega
        if free_A:
            A.v = A.v + dt * F_A / A.mass
            A.omega = A.omega + dt * T_A / A.inertia
            A.c = A.c + dt * A.v
            A.alpha = A.alpha + dt * A.omega

        ke_B = 0.5 * B.mass * B.v @ B.v + 0.5 * B.inertia * B.omega ** 2
        impulse += np.linalg.norm(F_B) * dt
        e_inj += float(F_B @ B.v + T_B * B.omega) * dt
        e_dis += -d["fric_power"] * dt

        rec = dict(step=i + 1, time=t, alpha_A=float(A.alpha),
                   cB_x=float(B.c[0]), cB_y=float(B.c[1]), alpha_B=float(B.alpha),
                   vB_x=float(B.v[0]), vB_y=float(B.v[1]), omega_B=float(B.omega),
                   ke_B=float(ke_B), n_contacts=int(d["n_contacts"]),
                   max_penetration=float(d["max_pen"]), contact_impulse=float(impulse),
                   energy_injected=float(e_inj), energy_dissipated=float(e_dis))
        if free_A:
            rec["linmom_err"] = float(np.linalg.norm(total_linmom() - p0))
            rec["angmom_err"] = float(abs(total_angmom() - l0))
            rec["ke_total"] = float(ke_B + 0.5 * A.mass * A.v @ A.v + 0.5 * A.inertia * A.omega ** 2)
        history.append(rec)
        if (i + 1) % 200 == 0:
            print("  step %5d | t=%.3e | cB=(%.3f,%.3f) | aB=%.3f | nC=%d | pen=%.2e | KE_B=%.3e"
                  % (i + 1, t, B.c[0], B.c[1], B.alpha, d["n_contacts"], d["max_pen"], ke_B))

    meta = dict(free_A=free_A, n_steps=n_steps, dt=dt, omega_drive=omega_drive,
                eps_n=eps_n, mu=mu, eps_t=eps_t, n_samples=n_samples,
                mass_A=A.mass, inertia_A=A.inertia, mass_B=B.mass, inertia_B=B.inertia,
                params_A=vars(pA), params_B=vars(pB), ke0_B=ke0)
    return history, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-A", action="store_true", help="momentum-conserving control (both free, mu=0)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--omega", type=float, default=4.0)
    ap.add_argument("--eps_n", type=float, default=2.0e4)
    ap.add_argument("--mu", type=float, default=0.3)
    args = ap.parse_args()

    name = "supershape_cam_drive_free_A" if args.free_A else "supershape_cam_drive"
    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {name} ===")
    history, meta = simulate(free_A=args.free_A, n_steps=args.steps,
                             omega_drive=args.omega, eps_n=args.eps_n, mu=args.mu)
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump({"history": history, "meta": meta}, f, indent=2)

    last = history[-1]
    print("\nSummary:")
    print("  B moved:  dcB = (%.3f, %.3f),  d alpha_B = %.3f rad"
          % (last["cB_x"] - history[0]["cB_x"], last["cB_y"] - history[0]["cB_y"], last["alpha_B"]))
    print("  max #contact-arcs over run: %d" % max(h["n_contacts"] for h in history))
    print("  max penetration: %.3e" % max(h["max_penetration"] for h in history))
    if args.free_A:
        print("  momentum error (max): lin=%.2e ang=%.2e"
              % (max(h["linmom_err"] for h in history), max(h["angmom_err"] for h in history)))
    else:
        dKE = last["ke_B"] - meta["ke0_B"]
        print("  work-energy: dKE_B=%.4e  energy_injected=%.4e  dissipated=%.4e  resid=%.2e"
              % (dKE, last["energy_injected"], last["energy_dissipated"],
                 abs(dKE - (last["energy_injected"]))))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
