"""Ball-drop benchmark: elastic ball onto rigid floor via MPM penalty contact.

Demonstrates the contact pipeline:
    1. Analytic floor SDF defines the rigid obstacle.
    2. Ball discretised as MPM particles with Neo-Hookean constitutive model.
    3. Penalty contact force computed each step via evaluate_gap + compute_contact_force.
    4. Forces scattered to grid alongside gravity during P2G.

Run::

    python benchmarks/contact/ball_drop_mpm.py

Outputs are written to ``runs/ball_drop_mpm/``.
"""

import os
import math
import json

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt


# ── Analytic floor SDF ───────────────────────────────────────────────

class FloorSDF(torch.nn.Module):
    """Half-space SDF: phi(x) = x[:,1] - y_floor."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Main ─────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join("runs", "ball_drop_mpm")
    os.makedirs(out_dir, exist_ok=True)

    # Parameters
    density = 1000.0
    E = 1e5
    nu = 0.3
    eps_n = 1e6           # penalty stiffness
    n_per_axis = 4        # particles per axis in ball
    ball_radius = 0.08    # half-extent of particle block
    ball_center_y = 0.02  # start just above floor
    v_initial_y = -0.5    # initial downward velocity
    gravity = (0.0, -9.81, 0.0)
    n_steps = 1000

    floor_sdf = FloorSDF(y_floor=0.0)

    solver = ChartMPMSolver(
        n_cells=16,
        extent=0.5,
        gravity=gravity,
        bc_type="free",
    )

    particles = MaterialPointCloud.create_uniform(
        n_per_axis=n_per_axis,
        extent=ball_radius,
        density=density,
    )
    # Shift ball to initial height and set initial velocity
    particles.xi[:, 1] += ball_center_y
    particles.v[:, 1] = v_initial_y

    # Time step: min of CFL and contact stability
    m_min = particles.mass.min().item()
    dt_contact = contact_stable_dt(eps_n, m_min)
    dt = min(1e-4, dt_contact)
    print(f"dt = {dt:.2e}  (contact dt = {dt_contact:.2e})")
    print(f"Particles: {particles.n_particles}")

    history = []
    for step_i in range(n_steps):
        # Contact detection and force computation
        x_phys = particles.xi.clone()  # identity decoder
        gap, normal = evaluate_gap(x_phys, floor_sdf)
        volume = particles.current_volume
        cf = compute_contact_force(gap, normal, volume, eps_n)

        diag = solver.step(particles, dt, contact_force=cf)

        # Augment diagnostics
        min_gap = gap.min().item()
        max_pen = max(0.0, -min_gap)
        avg_y = particles.xi[:, 1].mean().item()
        avg_vy = particles.v[:, 1].mean().item()

        record = {
            "step": diag["step"],
            "time": diag["time"],
            "kinetic_energy": diag["kinetic_energy"],
            "max_velocity": diag["max_velocity"],
            "min_gap": min_gap,
            "max_penetration": max_pen,
            "avg_y": avg_y,
            "avg_vy": avg_vy,
        }
        history.append(record)

        if (step_i + 1) % 50 == 0:
            print(
                f"  step {diag['step']:5d} | "
                f"t={diag['time']:.4e} | "
                f"KE={diag['kinetic_energy']:.4e} | "
                f"y={avg_y:.4f} | "
                f"vy={avg_vy:.4f} | "
                f"pen={max_pen:.2e}"
            )

    # Save results
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    min_y = min(r["avg_y"] for r in history)
    max_pen = max(r["max_penetration"] for r in history)
    final_vy = history[-1]["avg_vy"]
    print(f"\nSummary:")
    print(f"  Min avg y:         {min_y:.6f}")
    print(f"  Max penetration:   {max_pen:.2e}")
    print(f"  Final avg v_y:     {final_vy:.4f}")

    bounced = any(
        r["avg_vy"] > 0.1 for r in history[len(history) // 2:]
    )
    print(f"  Ball bounced:      {bounced}")


if __name__ == "__main__":
    main()
