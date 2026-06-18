"""Two-sphere collision benchmark: two elastic balls approaching each other.

Demonstrates two-body MPM penalty contact:
    1. Each ball is discretised as an independent MaterialPointCloud on
       its own ChartMPMSolver grid (identity decoder).
    2. A per-ball analytic SDF describes each obstacle.
    3. Each step computes penalty contact forces on ball A against
       ball B's SDF and vice versa.
    4. By Newton's third law and symmetry, the two balls should rebound
       with equal and opposite velocities.

Run::

    python benchmarks/contact/two_sphere_collision_mpm.py

Outputs are written to ``runs/two_sphere_collision_mpm/``.
"""

import os
import json

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt


# ── Analytic ball SDF that tracks a moving centre ────────────────────

class MovingBallSDF(torch.nn.Module):
    """Analytic sphere SDF with a mutable centre tensor.

    phi(x) = |x - c| - r
    """

    def __init__(self, center, radius):
        super().__init__()
        # Registered as a buffer so torch.autograd.grad still works w.r.t. x
        self.register_buffer(
            "center",
            torch.tensor(center, dtype=torch.float64),
        )
        self.radius = radius

    def update_center(self, new_center: torch.Tensor) -> None:
        self.center.copy_(new_center)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x - self.center.unsqueeze(0)
        return d.norm(dim=1) - self.radius


def _centre_of_mass(particles: MaterialPointCloud) -> torch.Tensor:
    m = particles.mass.unsqueeze(1)
    return (m * particles.xi).sum(dim=0) / particles.mass.sum()


def _centre_of_mass_velocity(particles: MaterialPointCloud) -> torch.Tensor:
    m = particles.mass.unsqueeze(1)
    return (m * particles.v).sum(dim=0) / particles.mass.sum()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join("runs", "two_sphere_collision_mpm")
    os.makedirs(out_dir, exist_ok=True)

    # Parameters
    density = 1000.0
    eps_n = 5e7
    n_per_axis = 3
    # Use ball_radius = sdf_radius so particles extend to the SDF surface
    # and contact activates as soon as the two SDF spheres overlap.
    ball_radius = 0.07
    sdf_radius = 0.07
    v_approach = 0.2          # each ball moves toward the other
    n_steps = 4000

    # Ball centres: place them just touching (sep = 2 * sdf_radius).
    half_sep = sdf_radius
    cA0 = torch.tensor([-half_sep, 0.0, 0.0], dtype=torch.float64)
    cB0 = torch.tensor([+half_sep, 0.0, 0.0], dtype=torch.float64)

    # SDFs — one per ball, tracking each ball's centre of mass.
    sdf_A = MovingBallSDF(cA0.tolist(), sdf_radius)
    sdf_B = MovingBallSDF(cB0.tolist(), sdf_radius)

    # Two independent solvers (free BCs, no gravity).
    solver_A = ChartMPMSolver(
        n_cells=16, extent=0.3, gravity=None, bc_type="free",
    )
    solver_B = ChartMPMSolver(
        n_cells=16, extent=0.3, gravity=None, bc_type="free",
    )

    # Particles: create uniform blocks centred at each ball centre, with
    # opposite initial x-velocities.
    particles_A = MaterialPointCloud.create_uniform(
        n_per_axis=n_per_axis, extent=ball_radius, density=density,
    )
    particles_A.xi += cA0.unsqueeze(0)
    particles_A.v[:, 0] = +v_approach  # moving +x

    particles_B = MaterialPointCloud.create_uniform(
        n_per_axis=n_per_axis, extent=ball_radius, density=density,
    )
    particles_B.xi += cB0.unsqueeze(0)
    particles_B.v[:, 0] = -v_approach  # moving -x

    # Time step
    m_min = min(particles_A.mass.min().item(), particles_B.mass.min().item())
    dt_contact = contact_stable_dt(eps_n, m_min)
    dt = min(1e-5, dt_contact)
    print(f"dt = {dt:.2e}  (contact dt = {dt_contact:.2e})")
    print(f"Ball A particles: {particles_A.n_particles}")
    print(f"Ball B particles: {particles_B.n_particles}")

    history = []
    for step_i in range(n_steps):
        # Refresh each SDF's centre to follow the current CoM of that ball
        sdf_A.update_center(_centre_of_mass(particles_A))
        sdf_B.update_center(_centre_of_mass(particles_B))

        # Ball A particles see ball B's SDF, and vice versa
        gap_A, normal_A = evaluate_gap(particles_A.xi.detach(), sdf_B)
        gap_B, normal_B = evaluate_gap(particles_B.xi.detach(), sdf_A)

        cf_A = compute_contact_force(
            gap_A, normal_A, particles_A.current_volume, eps_n,
        )
        cf_B = compute_contact_force(
            gap_B, normal_B, particles_B.current_volume, eps_n,
        )

        diag_A = solver_A.step(particles_A, dt, contact_force=cf_A)
        diag_B = solver_B.step(particles_B, dt, contact_force=cf_B)

        com_A = _centre_of_mass(particles_A)
        com_B = _centre_of_mass(particles_B)
        vcom_A = _centre_of_mass_velocity(particles_A)
        vcom_B = _centre_of_mass_velocity(particles_B)

        # Momentum imbalance (should stay ~0 by Newton 3)
        mom = (
            particles_A.mass.sum() * vcom_A
            + particles_B.mass.sum() * vcom_B
        )

        record = {
            "step": step_i + 1,
            "time": (step_i + 1) * dt,
            "ke_total": diag_A["kinetic_energy"] + diag_B["kinetic_energy"],
            "com_A_x": com_A[0].item(),
            "com_B_x": com_B[0].item(),
            "vcom_A_x": vcom_A[0].item(),
            "vcom_B_x": vcom_B[0].item(),
            "total_momentum_x": mom[0].item(),
            "separation": (com_B[0] - com_A[0]).item(),
        }
        history.append(record)

        if (step_i + 1) % 50 == 0:
            print(
                f"  step {step_i + 1:4d} | "
                f"t={record['time']:.4e} | "
                f"KE={record['ke_total']:.4e} | "
                f"sep={record['separation']:.4f} | "
                f"v_A={record['vcom_A_x']:+.4f}  v_B={record['vcom_B_x']:+.4f} | "
                f"p_x={record['total_momentum_x']:+.2e}"
            )

    # Save results
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary: check deceleration, symmetry, and momentum conservation
    initial = history[0]
    final = history[-1]
    vA0 = initial["vcom_A_x"]
    vB0 = initial["vcom_B_x"]
    vA = final["vcom_A_x"]
    vB = final["vcom_B_x"]
    drift = abs(final["total_momentum_x"])  # should be 0 for symmetric case
    decel_A = (vA0 - vA) / abs(vA0)
    decel_B = (vB - vB0) / abs(vB0)

    print(f"\nSummary:")
    print(f"  Initial v_A_x:      {vA0:+.4f}")
    print(f"  Initial v_B_x:      {vB0:+.4f}")
    print(f"  Final v_A_x:        {vA:+.4f}")
    print(f"  Final v_B_x:        {vB:+.4f}")
    print(f"  Velocity symmetry:  |v_A + v_B| = {abs(vA + vB):.2e}")
    print(f"  Momentum drift:     {drift:.2e}")
    print(f"  Deceleration A:     {100 * decel_A:.1f}%")
    print(f"  Deceleration B:     {100 * decel_B:.1f}%")

    decelerated = decel_A > 0.01 and decel_B > 0.01
    symmetric = abs(vA + vB) < 1e-10
    momentum_conserved = drift < 1e-10
    print(f"  Balls decelerated:      {decelerated}")
    print(f"  Symmetric (v_A+v_B≈0):  {symmetric}")
    print(f"  Momentum conserved:     {momentum_conserved}")


if __name__ == "__main__":
    main()
