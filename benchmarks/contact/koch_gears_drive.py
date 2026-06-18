"""Interlocking Koch-snowflake 'gears' (CV-6): a rotating fractal cam drives a free fractal body.

Two Koch snowflakes (level n). Body A is a driver (fixed center, prescribed clockwise spin);
body B is free. Contact uses the recursive chart engine in solvers/contact/koch.py
(pruned O(depth) inside-test + nearest-boundary segment normal) -- NO global SDF grid. The
dynamics loop mirrors benchmarks/contact/supershape_cam_drive.py (two-pass node-to-surface
penalty + regularized Coulomb friction, semi-implicit Euler). --free-A is the
momentum-conserving control (both free, mu=0).

Run:  python3 benchmarks/contact/koch_gears_drive.py [--free-A] [--level N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solvers.contact import koch                                  # noqa: E402


def area_inertia(level, R, density):
    V = koch.snowflake_vertices(level, R=R)[:-1]
    x, y = V[:, 0], V[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    A = 0.5 * np.sum(cross)
    J = (1.0 / 12.0) * np.sum(cross * (x * x + x * xn + xn * xn + y * y + y * yn + yn * yn))
    return density * abs(A), density * abs(J)


@dataclass
class Body:
    level: int
    R: float
    c: np.ndarray
    alpha: float = 0.0
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0
    mass: float = 1.0
    inertia: float = 1.0

    @classmethod
    def make(cls, level, R, c, density=1.0, **kw):
        m, I = area_inertia(level, R, density)
        return cls(level=level, R=R, c=np.asarray(c, float), mass=m, inertia=I, **kw)

    def vel_at(self, pts):
        r = pts - self.c
        return self.v + self.omega * np.stack([-r[..., 1], r[..., 0]], axis=-1)

    def samples(self, density=4):
        """Boundary sample points, densified by inserting (density-1) points per segment
        (thin fractal spikes need enough interior samples for a smooth penalty)."""
        V = koch.snowflake_vertices(self.level, R=self.R, center=self.c, alpha=self.alpha)
        if density <= 1:
            return V[:-1]
        ts = np.linspace(0.0, 1.0, density, endpoint=False)
        segs = V[:-1][:, None, :] + ts[None, :, None] * (V[1:] - V[:-1])[:, None, :]
        return segs.reshape(-1, 2)


def _contact_pass(intr: Body, obst: Body, eps_n, mu, eps_t):
    """Penalize intruder boundary samples that lie inside the obstacle (Newton-3 pair)."""
    out = dict(F_intr=np.zeros(2), T_intr=0.0, F_obst=np.zeros(2), T_obst=0.0,
               n_contacts=0, max_pen=0.0, fric_power=0.0)
    pts = intr.samples()
    ds = koch.perimeter(intr.level, intr.R) / len(pts)            # mean segment length
    active_pts, gs, ns = [], [], []
    inside_flags = []
    for p in pts:
        ins, _ = koch.inside_cost(p, obst.level, R=obst.R, center=obst.c, alpha=obst.alpha)
        inside_flags.append(ins)
        if ins:
            g, _foot, n = koch.nearest_boundary(p, obst.level, R=obst.R, center=obst.c, alpha=obst.alpha)
            active_pts.append(p); gs.append(g); ns.append(n)
    if not active_pts:
        return out
    pa = np.array(active_pts); ga = np.array(gs); na = np.array(ns)
    fn = eps_n * (-ga) * ds                                       # >=0 normal magnitude
    fN = fn[:, None] * na                                         # normal force on intruder (outward of obst)
    vrel = intr.vel_at(pa) - obst.vel_at(pa)
    vt = vrel - np.sum(vrel * na, axis=1, keepdims=True) * na
    ft = -mu * fn[:, None] * vt / np.sqrt(np.sum(vt ** 2, axis=1, keepdims=True) + eps_t ** 2)
    f_intr = fN + ft
    f_obst = -f_intr
    ri = pa - intr.c; ro = pa - obst.c
    out["F_intr"] = f_intr.sum(0); out["F_obst"] = f_obst.sum(0)
    out["T_intr"] = float(np.sum(ri[:, 0] * f_intr[:, 1] - ri[:, 1] * f_intr[:, 0]))
    out["T_obst"] = float(np.sum(ro[:, 0] * f_obst[:, 1] - ro[:, 1] * f_obst[:, 0]))
    out["max_pen"] = float((-ga).max())
    out["fric_power"] = float(np.sum(ft * vt))
    # count contact arcs = runs of consecutive inside samples (circular)
    a = np.array(inside_flags, dtype=int)
    out["n_contacts"] = int(np.sum((a - np.roll(a, 1)) == 1)) or (1 if a.any() else 0)
    return out


def simulate(free_A=False, level=3, n_steps=1600, omega_drive=3.0, eps_n=5.0e4,
             mu=0.2, eps_t=1e-3, R=1.0, damping=0.0):
    # B starts CLEAR of A and clips its spike region (offset impact parameter) so the
    # spinning cam A deflects/spins it. The corrected (repulsive) contact + stiff penalty
    # + densified samples keep penetration shallow; optional light velocity damping bounds
    # the rebound for a clean demo. (Fractal spikes mesh deeply or miss head-on.)
    A = Body.make(level, R, c=[0.0, 0.0], density=1.0)
    B = Body.make(level, R, c=[2.15 * R, 0.25 * R], density=1.0)
    B.v = np.array([-1.2, 0.0])                                    # near head-on (slight offset -> spin)
    if free_A:
        A.omega = 0.0; mu = 0.0                                    # both free, frictionless control
    else:
        A.alpha = 0.0; A.omega = -omega_drive                     # spinning cam driver

    L = R + np.linalg.norm(B.c - A.c)
    m_eff = min(B.mass, B.inertia / L ** 2)
    dt = float(min(0.5 * np.sqrt(m_eff / (2 * eps_n)), 5e-4))

    hist = []; impulse = e_inj = e_dis = 0.0
    ke0 = 0.5 * B.mass * B.v @ B.v + 0.5 * B.inertia * B.omega ** 2
    p0 = A.mass * A.v + B.mass * B.v
    for i in range(n_steps):
        t = (i + 1) * dt
        if not free_A:
            A.alpha = -omega_drive * t; A.omega = -omega_drive
        p1 = _contact_pass(A, B, eps_n, mu, eps_t)
        p2 = _contact_pass(B, A, eps_n, mu, eps_t)
        F_B = p1["F_obst"] + p2["F_intr"]; T_B = p1["T_obst"] + p2["T_intr"]
        F_A = p1["F_intr"] + p2["F_obst"]; T_A = p1["T_intr"] + p2["T_obst"]
        B.v = B.v + dt * F_B / B.mass; B.omega = B.omega + dt * T_B / B.inertia
        B.v *= (1.0 - damping); B.omega *= (1.0 - damping)        # bound rebound (driven demo)
        B.c = B.c + dt * B.v; B.alpha = B.alpha + dt * B.omega
        if free_A:
            A.v = A.v + dt * F_A / A.mass; A.omega = A.omega + dt * T_A / A.inertia
            A.c = A.c + dt * A.v; A.alpha = A.alpha + dt * A.omega
        ke_B = 0.5 * B.mass * B.v @ B.v + 0.5 * B.inertia * B.omega ** 2
        impulse += np.linalg.norm(F_B) * dt
        e_inj += float(F_B @ B.v + T_B * B.omega) * dt
        e_dis += -(p1["fric_power"] + p2["fric_power"]) * dt
        rec = dict(step=i + 1, time=t, alpha_A=float(A.alpha), cB_x=float(B.c[0]), cB_y=float(B.c[1]),
                   alpha_B=float(B.alpha), omega_B=float(B.omega), ke_B=float(ke_B),
                   n_contacts=max(p1["n_contacts"], p2["n_contacts"]),
                   max_penetration=float(max(p1["max_pen"], p2["max_pen"])),
                   contact_impulse=float(impulse), energy_injected=float(e_inj),
                   energy_dissipated=float(e_dis))
        if free_A:
            rec["linmom_err"] = float(np.linalg.norm(A.mass * A.v + B.mass * B.v - p0))
        hist.append(rec)
        if (i + 1) % 150 == 0:
            print("  step %4d t=%.3e cB=(%.3f,%.3f) aB=%.3f nC=%d pen=%.2e"
                  % (i + 1, t, B.c[0], B.c[1], B.alpha, rec["n_contacts"], rec["max_penetration"]))
    meta = dict(free_A=free_A, level=level, n_steps=n_steps, dt=dt, omega_drive=omega_drive,
                eps_n=eps_n, mu=mu, R=R, mass=A.mass, inertia=A.inertia, ke0_B=ke0)
    return hist, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-A", action="store_true")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()
    name = "koch_gears_drive_free_A" if args.free_A else "koch_gears_drive"
    out_dir = os.path.join("runs", name); os.makedirs(out_dir, exist_ok=True)
    print(f"=== {name} (level {args.level}) ===")
    hist, meta = simulate(free_A=args.free_A, level=args.level, n_steps=args.steps)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"history": hist, "meta": meta}, f, indent=2)
    last = hist[-1]
    print("\nB moved: dc=(%.3f,%.3f), dalpha=%.3f rad; max arcs=%d; max pen=%.2e"
          % (last["cB_x"] - hist[0]["cB_x"], last["cB_y"] - hist[0]["cB_y"], last["alpha_B"],
             max(h["n_contacts"] for h in hist), max(h["max_penetration"] for h in hist)))
    if args.free_A:
        print("momentum error (max): %.2e" % max(h["linmom_err"] for h in hist))
    print("saved", os.path.join(out_dir, "history.json"))


if __name__ == "__main__":
    main()
