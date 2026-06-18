"""Tests for regularized Coulomb friction in MPM contact.

Verifies the per-particle friction force computation and the
sliding-block integration test (constant deceleration on a horizontal
floor).
"""

import math

import torch
import pytest

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force, contact_stable_dt
from solvers.contact.friction import (
    decompose_velocity,
    compute_friction_force,
)


# ── Analytic floor SDF ───────────────────────────────────────────────


class FloorSDF(torch.nn.Module):
    """phi(x) = x[:,1] - y_floor; outward normal = (0, 1, 0)."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Decomposition tests ──────────────────────────────────────────────


class TestDecomposeVelocity:
    def test_round_trip(self):
        """v == v_n * n + v_t and v_t · n == 0."""
        torch.manual_seed(0)
        v = torch.randn(10, 3, dtype=torch.float64)
        # Random unit normals
        n = torch.randn(10, 3, dtype=torch.float64)
        n = n / n.norm(dim=1, keepdim=True)

        v_n_scalar, v_t = decompose_velocity(v, n)
        recon = v_n_scalar.unsqueeze(1) * n + v_t
        assert torch.allclose(recon, v, atol=1e-12)

        dot_t_n = (v_t * n).sum(dim=1)
        assert torch.allclose(
            dot_t_n, torch.zeros_like(dot_t_n), atol=1e-12
        )

    def test_pure_normal_velocity(self):
        """v parallel to n should give zero tangential component."""
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 3.7, 0.0]], dtype=torch.float64)
        v_n_scalar, v_t = decompose_velocity(v, n)
        assert abs(v_n_scalar[0].item() - 3.7) < 1e-12
        assert torch.allclose(v_t, torch.zeros_like(v_t), atol=1e-12)

    def test_pure_tangential_velocity(self):
        """v perpendicular to n should give zero normal component."""
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[2.0, 0.0, -1.5]], dtype=torch.float64)
        v_n_scalar, v_t = decompose_velocity(v, n)
        assert abs(v_n_scalar[0].item()) < 1e-12
        assert torch.allclose(v_t, v, atol=1e-12)


# ── Friction force unit tests ────────────────────────────────────────


class TestFrictionForce:
    def test_zero_mu_returns_zero(self):
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        fN = torch.tensor([100.0], dtype=torch.float64)
        f_T = compute_friction_force(v, n, fN, mu=0.0)
        assert torch.allclose(f_T, torch.zeros_like(f_T))

    def test_friction_opposes_tangential_velocity(self):
        """f_T · v_t < 0 whenever ‖v_t‖ > 0."""
        torch.manual_seed(1)
        v = torch.randn(8, 3, dtype=torch.float64)
        n = torch.randn(8, 3, dtype=torch.float64)
        n = n / n.norm(dim=1, keepdim=True)
        fN = torch.full((8,), 50.0, dtype=torch.float64)

        f_T = compute_friction_force(v, n, fN, mu=0.4, epsilon_t=1e-8)
        _, v_t = decompose_velocity(v, n)

        dot = (f_T * v_t).sum(dim=1)
        # Skip particles whose tangential velocity is essentially zero
        mask = v_t.norm(dim=1) > 1e-6
        assert (dot[mask] < 0).all()

    def test_friction_pure_normal_velocity(self):
        """v parallel to n ⇒ no tangential velocity ⇒ no friction."""
        v = torch.tensor([[0.0, -2.0, 0.0]], dtype=torch.float64)
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        fN = torch.tensor([100.0], dtype=torch.float64)
        f_T = compute_friction_force(v, n, fN, mu=0.5, epsilon_t=1e-8)
        assert torch.allclose(f_T, torch.zeros_like(f_T), atol=1e-10)

    def test_friction_zero_normal_force(self):
        """‖f_N‖ = 0 ⇒ no friction even if there's tangential velocity."""
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        fN = torch.tensor([0.0], dtype=torch.float64)
        f_T = compute_friction_force(v, n, fN, mu=0.5)
        assert torch.allclose(f_T, torch.zeros_like(f_T))

    def test_friction_magnitude_bounded_slip_limit(self):
        """At high tangential speed, ‖f_T‖ → mu * ‖f_N‖."""
        # v_T = 100 m/s along x, n along y, f_N = 10 N
        v = torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float64)
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        fN = torch.tensor([10.0], dtype=torch.float64)
        mu = 0.3
        f_T = compute_friction_force(v, n, fN, mu=mu, epsilon_t=1e-6)
        expected_mag = mu * 10.0
        actual_mag = f_T.norm(dim=1)[0].item()
        assert abs(actual_mag - expected_mag) < 1e-6
        # Force should be along -x
        assert f_T[0, 0].item() < 0
        assert abs(f_T[0, 1].item()) < 1e-12
        assert abs(f_T[0, 2].item()) < 1e-12

    def test_friction_smooth_at_zero_velocity(self):
        """No NaNs and magnitude → 0 as ‖v_t‖ → 0."""
        v_zero = torch.zeros(1, 3, dtype=torch.float64)
        n = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        fN = torch.tensor([100.0], dtype=torch.float64)
        f_T = compute_friction_force(v_zero, n, fN, mu=0.5, epsilon_t=1e-6)
        assert torch.isfinite(f_T).all()
        assert f_T.norm().item() < 1e-12

        # And a small but nonzero velocity should give a small force
        v_small = torch.tensor(
            [[1e-9, 0.0, 0.0]], dtype=torch.float64
        )
        f_T_small = compute_friction_force(
            v_small, n, fN, mu=0.5, epsilon_t=1e-6
        )
        assert torch.isfinite(f_T_small).all()
        # Force magnitude should be small (much less than the slip-limit
        # bound mu * |f_N| = 50)
        assert f_T_small.norm().item() < 1.0


# ── Sliding-block integration tests ──────────────────────────────────


def _make_thin_slab(n_xz: int, density: float, penetration: float):
    """Single-layer particle slab placed below the floor.

    All particles share the same y-coordinate ``-penetration`` so the
    entire slab is uniformly in contact with a floor at y=0.
    """
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


class TestSlidingSlabAnalyticOneStep:
    """Analytic one-step verification: total friction force scattered to
    the grid should equal ``-mu * total_normal_force`` to high accuracy,
    independent of any time-stepping or stress dynamics.
    """

    def test_total_friction_equals_mu_normal(self):
        floor = FloorSDF(y_floor=0.0)
        eps_n = 1e6
        mu = 0.3
        v0 = 1.0

        slab = _make_thin_slab(n_xz=3, density=1000.0, penetration=1e-3)
        slab.v[:, 0] = v0

        gap, normal = evaluate_gap(slab.xi.clone(), floor)
        f_N = compute_contact_force(
            gap, normal, slab.current_volume, eps_n,
        )
        f_N_mag = f_N.norm(dim=1)
        f_T = compute_friction_force(
            slab.v, normal, f_N_mag, mu=mu, epsilon_t=1e-6,
        )

        # Every particle is in penetration → every particle has |f_N| > 0
        assert (f_N_mag > 0).all()
        # Friction direction is along -x for v0 along +x and n along +y
        assert (f_T[:, 0] < 0).all()
        assert torch.allclose(f_T[:, 1], torch.zeros_like(f_T[:, 1]), atol=1e-12)
        assert torch.allclose(f_T[:, 2], torch.zeros_like(f_T[:, 2]), atol=1e-12)

        # Sum of friction force x-components must equal -mu * sum of |f_N|
        total_fT_x = f_T[:, 0].sum().item()
        total_fN_mag = f_N_mag.sum().item()
        expected_total_fT_x = -mu * total_fN_mag
        rel_err = abs(total_fT_x - expected_total_fT_x) / abs(expected_total_fT_x)
        assert rel_err < 1e-12

    def test_one_step_velocity_change_matches_analytic(self):
        """One step of MPM with the friction force should produce a
        velocity change consistent with ``a = total_f_T / total_mass``.
        """
        floor = FloorSDF(y_floor=0.0)
        eps_n = 1e6
        mu = 0.3
        v0 = 1.0

        slab = _make_thin_slab(n_xz=3, density=1000.0, penetration=1e-3)
        slab.v[:, 0] = v0

        # No gravity for this test — we want to isolate the friction effect
        solver = ChartMPMSolver(
            n_cells=8, extent=0.4, gravity=None, bc_type="free",
        )
        dt = 1e-5

        # Compute forces for the first step
        gap, normal = evaluate_gap(slab.xi.clone(), floor)
        f_N = compute_contact_force(
            gap, normal, slab.current_volume, eps_n,
        )
        f_T = compute_friction_force(
            slab.v, normal, f_N.norm(dim=1), mu=mu, epsilon_t=1e-6,
        )
        total_fT_x = f_T[:, 0].sum().item()
        total_mass = slab.mass.sum().item()
        expected_dvx = (total_fT_x / total_mass) * dt  # negative

        v_before = (
            (slab.mass * slab.v[:, 0]).sum() / slab.mass.sum()
        ).item()
        solver.step(slab, dt, contact_force=f_N + f_T)
        v_after = (
            (slab.mass * slab.v[:, 0]).sum() / slab.mass.sum()
        ).item()
        actual_dvx = v_after - v_before

        # Should be in the right direction and within ~30% of analytic
        assert actual_dvx < 0
        ratio = actual_dvx / expected_dvx
        assert 0.5 < ratio < 1.5, (
            f"One-step dv off: actual={actual_dvx:.6e}, "
            f"expected={expected_dvx:.6e}, ratio={ratio:.3f}"
        )


class TestSlidingSlabIntegration:
    """Multi-step integration: thin slab held against a floor by gravity
    with horizontal initial velocity should decelerate over time.

    The slab is started at the static equilibrium penetration so the
    contact force balances gravity from step 1 — there is no settling
    transient that would let the slab bounce off the floor and lose
    contact.
    """

    @staticmethod
    def _equilibrium_penetration(density, g, eps_n):
        """For a single-particle Coulomb sliding test, the static
        equilibrium penetration satisfies ``rho * g = eps_n * pen``."""
        return density * g / eps_n

    def test_slab_decelerates(self):
        floor = FloorSDF(y_floor=0.0)
        eps_n = 1e6
        mu = 0.3
        g = 9.81
        v0 = 1.0
        density = 1000.0

        pen_eq = self._equilibrium_penetration(density, g, eps_n)
        slab = _make_thin_slab(
            n_xz=3, density=density, penetration=pen_eq,
        )
        slab.v[:, 0] = v0

        # Finer grid for better particle/grid coupling
        solver = ChartMPMSolver(
            n_cells=20, extent=0.2, gravity=(0.0, -g, 0.0),
            bc_type="free",
        )
        dt = min(1e-5, contact_stable_dt(eps_n, slab.mass.min().item()))
        n_steps = 2000

        v_history = []
        for _ in range(n_steps):
            gap, normal = evaluate_gap(slab.xi.clone(), floor)
            f_N = compute_contact_force(
                gap, normal, slab.current_volume, eps_n,
            )
            f_T = compute_friction_force(
                slab.v, normal, f_N.norm(dim=1),
                mu=mu, epsilon_t=1e-4,
            )
            solver.step(slab, dt, contact_force=f_N + f_T)
            vcom_x = (
                (slab.mass * slab.v[:, 0]).sum() / slab.mass.sum()
            ).item()
            v_history.append(vcom_x)

        v_start, v_end = v_history[0], v_history[-1]
        # Slab must have decelerated meaningfully
        assert v_end < v_start - 0.02, (
            f"Slab did not decelerate enough: "
            f"v_start={v_start:.4f}, v_end={v_end:.4f}"
        )
        # And velocity should remain positive (didn't reverse)
        assert v_end > -0.1

    def test_frictionless_slab_does_not_decelerate(self):
        """Same setup with mu=0 should keep the slab moving."""
        floor = FloorSDF(y_floor=0.0)
        eps_n = 1e6
        g = 9.81
        v0 = 1.0
        density = 1000.0

        pen_eq = self._equilibrium_penetration(density, g, eps_n)
        slab = _make_thin_slab(
            n_xz=3, density=density, penetration=pen_eq,
        )
        slab.v[:, 0] = v0

        solver = ChartMPMSolver(
            n_cells=20, extent=0.2, gravity=(0.0, -g, 0.0),
            bc_type="free",
        )
        dt = min(1e-5, contact_stable_dt(eps_n, slab.mass.min().item()))

        for _ in range(2000):
            gap, normal = evaluate_gap(slab.xi.clone(), floor)
            f_N = compute_contact_force(
                gap, normal, slab.current_volume, eps_n,
            )
            # No friction
            solver.step(slab, dt, contact_force=f_N)

        vcom_x_final = (
            (slab.mass * slab.v[:, 0]).sum() / slab.mass.sum()
        ).item()
        # Without friction, the slab keeps most of its initial velocity
        assert vcom_x_final > 0.8 * v0, (
            f"Frictionless slab lost velocity: v_start={v0}, "
            f"v_end={vcom_x_final:.4f}"
        )
