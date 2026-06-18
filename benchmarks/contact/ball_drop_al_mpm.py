"""Ball-drop comparison: pure penalty vs augmented Lagrangian (Uzawa).

Drops the same elastic ball under gravity onto a rigid floor twice with
the same penalty stiffness ``epsilon_n``.  One run uses the pure penalty
contact force from ``solvers/contact/penalty.py``; the other uses the
augmented-Lagrangian update from
``solvers/contact/augmented_lagrangian.py``.

The augmented-Lagrangian run should achieve smaller residual penetration
because the persistent multiplier accumulates the contact pressure
needed to enforce non-penetration even at moderate ``epsilon_n``.

Run::

    PYTHONPATH=. python benchmarks/contact/ball_drop_al_mpm.py

Outputs are written to ``runs/ball_drop_al_mpm/``.
"""

import os
import json

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt
from solvers.contact.augmented_lagrangian import AugmentedLagrangianContact


# ── Analytic floor SDF (same as ball_drop_mpm.py) ────────────────────


class FloorSDF(torch.nn.Module):
    """Half-space SDF: phi(x) = x[:,1] - y_floor."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Helpers ──────────────────────────────────────────────────────────


def _make_ball():
    """Identical particle clouds for the two runs."""
    p = MaterialPointCloud.create_uniform(
        n_per_axis=4, extent=0.06, density=1000.0,
    )
    p.xi[:, 1] += 0.05  # start above the floor
    return p


def _run_penalty(eps_n, dt, n_steps, floor):
    solver = ChartMPMSolver(
        n_cells=16, extent=0.5, gravity=(0.0, -9.81, 0.0), bc_type="free",
    )
    particles = _make_ball()

    history = []
    for step_i in range(n_steps):
        x = particles.xi.clone()
        gap, normal = evaluate_gap(x, floor)
        cf = compute_contact_force(
            gap, normal, particles.current_volume, eps_n,
        )
        diag = solver.step(particles, dt, contact_force=cf)
        history.append({
            "step": diag["step"],
            "time": diag["time"],
            "min_y": particles.xi[:, 1].min().item(),
            "avg_y": particles.xi[:, 1].mean().item(),
            "kinetic_energy": diag["kinetic_energy"],
        })
    return particles, history


def _run_al(eps_n, dt, n_steps, floor):
    solver = ChartMPMSolver(
        n_cells=16, extent=0.5, gravity=(0.0, -9.81, 0.0), bc_type="free",
    )
    particles = _make_ball()
    al = AugmentedLagrangianContact(epsilon_n=eps_n)

    history = []
    for step_i in range(n_steps):
        x = particles.xi.clone()
        gap, normal = evaluate_gap(x, floor)
        cf = al.compute_force(gap, normal, particles.current_volume)
        diag = solver.step(particles, dt, contact_force=cf)

        # Update multiplier from gap at the new particle positions
        x_new = particles.xi.clone()
        gap_new, _ = evaluate_gap(x_new, floor)
        al.uzawa_update(gap_new)

        history.append({
            "step": diag["step"],
            "time": diag["time"],
            "min_y": particles.xi[:, 1].min().item(),
            "avg_y": particles.xi[:, 1].mean().item(),
            "kinetic_energy": diag["kinetic_energy"],
            "lambda_max": al.lam.max().item() if al.lam is not None else 0.0,
            "lambda_active_frac": (
                (al.lam > 0).float().mean().item()
                if al.lam is not None else 0.0
            ),
        })
    return particles, history, al


# ── Main ─────────────────────────────────────────────────────────────


def main():
    out_dir = os.path.join("runs", "ball_drop_al_mpm")
    os.makedirs(out_dir, exist_ok=True)

    # Shared parameters
    eps_n = 5e4              # moderate penalty
    n_steps = 2000
    floor = FloorSDF(y_floor=0.0)

    # Time step from contact stability
    dummy = _make_ball()
    m_min = dummy.mass.min().item()
    dt_contact = contact_stable_dt(eps_n, m_min)
    dt = min(5e-5, dt_contact)
    print(f"epsilon_n     = {eps_n:.2e}")
    print(f"dt            = {dt:.2e}  (contact dt = {dt_contact:.2e})")
    print(f"n_steps       = {n_steps}")
    print(f"total time    = {n_steps * dt:.4f} s")
    print(f"particles     = {dummy.n_particles}")
    print()

    # Run penalty version
    print("Running PURE PENALTY ...")
    p_pen, h_pen = _run_penalty(eps_n, dt, n_steps, floor)
    pen_min_y = min(r["min_y"] for r in h_pen)
    pen_final_y = h_pen[-1]["avg_y"]
    pen_final_min_y = h_pen[-1]["min_y"]

    # Run AL version
    print("Running AUGMENTED LAGRANGIAN ...")
    p_al, h_al, al = _run_al(eps_n, dt, n_steps, floor)
    al_min_y = min(r["min_y"] for r in h_al)
    al_final_y = h_al[-1]["avg_y"]
    al_final_min_y = h_al[-1]["min_y"]

    # Save histories
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump({"penalty": h_pen, "augmented_lagrangian": h_al}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    pen_pen_max = max(0.0, -pen_min_y)
    al_pen_max = max(0.0, -al_min_y)
    pen_pen_final = max(0.0, -pen_final_min_y)
    al_pen_final = max(0.0, -al_final_min_y)

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'metric':<35} {'penalty':>12} {'AL':>12}")
    print(f"{'-' * 35} {'-' * 12} {'-' * 12}")
    print(f"{'final avg y':<35} {pen_final_y:>12.6e} {al_final_y:>12.6e}")
    print(f"{'final min y':<35} {pen_final_min_y:>12.6e} {al_final_min_y:>12.6e}")
    print(f"{'max penetration (over time)':<35} {pen_pen_max:>12.6e} {al_pen_max:>12.6e}")
    print(f"{'final penetration':<35} {pen_pen_final:>12.6e} {al_pen_final:>12.6e}")
    if al.lam is not None:
        print(f"{'final λ_max':<35} {'—':>12} {al.lam.max().item():>12.6e}")
        print(f"{'final λ active fraction':<35} {'—':>12} {(al.lam > 0).float().mean().item():>12.4f}")
    print()

    if al_pen_final < pen_pen_final:
        ratio = al_pen_final / max(pen_pen_final, 1e-15)
        print(f"AL reduces final penetration by {(1 - ratio) * 100:.1f}% "
              f"({pen_pen_final:.3e} → {al_pen_final:.3e}).")
    else:
        print("AL did not reduce final penetration "
              "(check parameters / step count).")


if __name__ == "__main__":
    main()
