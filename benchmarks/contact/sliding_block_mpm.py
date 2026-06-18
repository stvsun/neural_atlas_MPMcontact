"""Sliding-block benchmark: regularized Coulomb friction validation.

A thin elastic slab is pinned to a horizontal floor by gravity and
launched with a horizontal velocity ``v0``.  The slab is initialized at
the static equilibrium penetration depth so the contact force balances
gravity from the first step.  Two runs are compared:

    * Frictionless (mu = 0)   — slab keeps moving at v0.
    * Frictioned (mu > 0)     — slab decelerates at a ≈ mu * g.

The expected dynamics from elementary mechanics:

    f_friction = mu * |f_N|       (per-particle, slip regime)
    a          = -mu * g          (uniform horizontal deceleration)
    v(t)       = max(0, v0 - mu*g*t)
    stop time  = v0 / (mu*g)

Run::

    PYTHONPATH=. python benchmarks/contact/sliding_block_mpm.py

Outputs are written to ``runs/sliding_block_mpm/``.
"""

import os
import json

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt
from solvers.contact.friction import compute_friction_force


# ── Analytic floor SDF ───────────────────────────────────────────────


class FloorSDF(torch.nn.Module):
    """phi(x) = x[:,1] - y_floor; outward normal = (0, 1, 0)."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Slab construction ───────────────────────────────────────────────


def make_slab(n_xz: int, density: float, penetration: float):
    """Single-layer slab placed below the floor at uniform penetration."""
    coords = []
    spacing = 0.02
    for ix in range(n_xz):
        for iz in range(n_xz):
            x = (ix - (n_xz - 1) / 2) * spacing
            z = (iz - (n_xz - 1) / 2) * spacing
            coords.append([x, -penetration, z])
    pos = torch.tensor(coords, dtype=torch.float64)
    n = pos.shape[0]
    cell_vol = spacing ** 3
    vols = torch.full((n,), cell_vol, dtype=torch.float64)
    masses = density * vols
    vels = torch.zeros(n, 3, dtype=torch.float64)
    return MaterialPointCloud(pos, vels, vols, masses)


# ── Single run ──────────────────────────────────────────────────────


def run(mu: float, eps_n: float, density: float, g: float, v0: float,
        n_steps: int, label: str):
    pen_eq = density * g / eps_n
    slab = make_slab(n_xz=4, density=density, penetration=pen_eq)
    slab.v[:, 0] = v0

    solver = ChartMPMSolver(
        n_cells=20, extent=0.2,
        gravity=(0.0, -g, 0.0),
        bc_type="free",
    )
    floor = FloorSDF(y_floor=0.0)

    dt = min(1e-5, contact_stable_dt(eps_n, slab.mass.min().item()))
    history = []
    for step_i in range(n_steps):
        gap, normal = evaluate_gap(slab.xi.clone(), floor)
        f_N = compute_contact_force(
            gap, normal, slab.current_volume, eps_n,
        )
        f_T = compute_friction_force(
            slab.v, normal, f_N.norm(dim=1),
            mu=mu, epsilon_t=1e-4,
        )
        diag = solver.step(slab, dt, contact_force=f_N + f_T)

        vcom_x = (
            (slab.mass * slab.v[:, 0]).sum() / slab.mass.sum()
        ).item()
        vcom_y = (
            (slab.mass * slab.v[:, 1]).sum() / slab.mass.sum()
        ).item()
        history.append({
            "step": diag["step"],
            "time": diag["time"],
            "vcom_x": vcom_x,
            "vcom_y": vcom_y,
            "kinetic_energy": diag["kinetic_energy"],
        })

        if (step_i + 1) % 200 == 0:
            print(f"  [{label}] step {diag['step']:5d} | "
                  f"t={diag['time']:.4e} | "
                  f"v_x={vcom_x:.4f} | "
                  f"KE={diag['kinetic_energy']:.4e}")

    return history


# ── Main ────────────────────────────────────────────────────────────


def main():
    out_dir = os.path.join("runs", "sliding_block_mpm")
    os.makedirs(out_dir, exist_ok=True)

    eps_n = 1e6
    density = 1000.0
    g = 9.81
    v0 = 1.0
    mu = 0.3
    n_steps = 4000

    pen_eq = density * g / eps_n
    print(f"epsilon_n           = {eps_n:.2e}")
    print(f"mu                  = {mu}")
    print(f"v0                  = {v0} m/s")
    print(f"gravity             = {g} m/s^2")
    print(f"equilibrium pen.    = {pen_eq * 1000:.3f} mm")
    print(f"expected decel      = {mu * g:.3f} m/s^2")
    print(f"expected stop time  = {v0 / (mu * g):.4f} s")
    print()

    print("Running FRICTIONLESS (mu = 0) ...")
    h_no = run(0.0, eps_n, density, g, v0, n_steps, "no_fric")

    print("\nRunning WITH FRICTION (mu = 0.3) ...")
    h_fr = run(mu, eps_n, density, g, v0, n_steps, "friction")

    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump({
            "frictionless": h_no,
            "friction": h_fr,
            "params": {
                "eps_n": eps_n, "mu": mu, "v0": v0,
                "g": g, "density": density, "n_steps": n_steps,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    no_v_start = h_no[0]["vcom_x"]
    no_v_end = h_no[-1]["vcom_x"]
    fr_v_start = h_fr[0]["vcom_x"]
    fr_v_end = h_fr[-1]["vcom_x"]
    T = h_fr[-1]["time"]
    fr_dv = fr_v_start - fr_v_end
    a_measured = fr_dv / T if T > 0 else 0.0
    a_expected = mu * g

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'metric':<35} {'frictionless':>14} {'friction':>14}")
    print(f"{'-' * 35} {'-' * 14} {'-' * 14}")
    print(f"{'initial v_x (m/s)':<35} {no_v_start:>14.6f} {fr_v_start:>14.6f}")
    print(f"{'final v_x (m/s)':<35} {no_v_end:>14.6f} {fr_v_end:>14.6f}")
    print(f"{'velocity loss (m/s)':<35} "
          f"{no_v_start - no_v_end:>14.6f} {fr_dv:>14.6f}")
    print()
    print(f"Total time            = {T:.4f} s")
    print(f"Measured deceleration = {a_measured:.4f} m/s^2")
    print(f"Analytic mu*g         = {a_expected:.4f} m/s^2")
    if a_expected > 0:
        ratio = a_measured / a_expected
        print(f"Ratio (measured/exp)  = {ratio:.3f}")


if __name__ == "__main__":
    main()
