"""Tests for the curved-chart MPM velocity-gradient fix (audit Option 3).

These tests verify that ``ChartMPMSolver`` handles non-identity chart
decoders correctly by computing the chart Jacobian per step and
chain-ruling the ξ-space shape-function gradients into physical space
before the P2G internal-force scatter and G2P velocity-gradient gather.

See ``docs/mpm_velocity_gradient_audit.md`` for the underlying math.
"""

import copy

import numpy as np
import torch
import pytest

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.grid import ChartGrid
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.mpm.transfers import particle_to_grid, grid_to_particle
from solvers.contact.gap import evaluate_gap
from solvers.contact.penalty import compute_contact_force


# ── Analytic affine decoder ──────────────────────────────────────────


class AffineDecoder(torch.nn.Module):
    """``phi(xi) = alpha * xi`` — constant Jacobian ``J = alpha * I``.

    Ignores all five frame parameters (seed, t1, t2, n, chart_scale)
    so the analytic ``J`` is exactly ``alpha * I`` everywhere.  This is
    the clean test fixture the audit calls for.
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        xi: torch.Tensor,
        seed: torch.Tensor = None,
        t1: torch.Tensor = None,
        t2: torch.Tensor = None,
        n: torch.Tensor = None,
        chart_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.alpha * xi


def _dummy_frame(device=None, dtype=torch.float64):
    """Frame inputs the AffineDecoder ignores, but which must still be
    passed because ``chart_map_and_jacobian`` forwards them.
    """
    return {
        "seed": torch.zeros(3, device=device, dtype=dtype),
        "t1": torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        "t2": torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        "n_vec": torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        "chart_scale": torch.tensor(1.0, device=device, dtype=dtype),
    }


# ── Analytic floor SDF (reused from other contact tests) ─────────────


class FloorSDF(torch.nn.Module):
    """phi(x) = x[:,1] - y_floor; outward normal = (0, 1, 0)."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Test 1: affine decoder with alpha = 2 ────────────────────────────


class TestAffineDecoderVelocityGradient:
    @staticmethod
    def _build_grid_with_uniform_velocity_field(
        alpha: float, gamma: float,
    ):
        """Create a grid where the physical-space velocity field is

            v_phys(x) = (gamma * x[1], 0, 0)

        i.e. a simple shear with spatial gradient L[0,1] = gamma
        (all other entries zero).  Under phi(xi) = alpha * xi, a grid
        node at chart-local position xi has physical position
        x = alpha * xi, so its physical-space velocity there is
        (gamma * alpha * xi[1], 0, 0).
        """
        grid = ChartGrid(n_cells=8, extent=0.5)
        # Fill grid.velocity with the prescribed field
        grid.velocity[:, 0] = gamma * alpha * grid.positions[:, 1]
        grid.velocity[:, 1] = 0.0
        grid.velocity[:, 2] = 0.0
        # Non-zero mass so compute_velocity-style pipelines don't
        # zero it out (even though we're not actually calling
        # compute_velocity here)
        grid.mass[:] = 1.0
        return grid

    def _run_g2p_for_alpha(self, alpha: float, gamma: float = 0.1):
        """Run a single G2P sweep on an affine-decoder test setup
        and return the resulting per-particle velocity gradient.
        """
        # Build a curved-chart MPM solver
        decoder = AffineDecoder(alpha=alpha).double()
        frame = _dummy_frame()
        solver = ChartMPMSolver(
            n_cells=8, extent=0.5,
            bc_type="free",
            decoder=decoder,
            **frame,
        )

        # Replace the freshly-made grid with one carrying the
        # prescribed velocity field
        solver.grid = self._build_grid_with_uniform_velocity_field(
            alpha=alpha, gamma=gamma,
        )

        # Particles at known chart-local positions — use n_per_axis=4
        # so none of them lands exactly on a grid node (positions are
        # {-0.3, -0.1, +0.1, +0.3} with an 8-cell grid of step 0.125).
        # Linear B-spline gradients are degenerate on nodes, so any
        # test that hits a node sees spurious zeros.
        particles = MaterialPointCloud.create_uniform(
            n_per_axis=4, extent=0.3, density=1000.0,
        )

        # Compute J_inv_T the same way ChartMPMSolver.step would
        J_inv_T = solver._compute_J_inv_T(particles)

        # Run G2P with the transform
        grid_to_particle(
            particles, solver.grid, dt=1e-5, J_inv_T=J_inv_T,
        )

        # Now particles.F has been updated.  Re-run just the gradient
        # computation so we can inspect vel_grad directly.
        # (grid_to_particle already updated F; we derive vel_grad back
        # from F via F_new = (I + dt L) F_old with F_old = I:
        #     vel_grad ≈ (F_new - I) / dt
        dt = 1e-5
        L_measured = (particles.F - torch.eye(3, dtype=torch.float64)) / dt
        return L_measured

    def test_affine_decoder_j_eq_2I_velocity_gradient(self):
        """With alpha = 2 and v_phys = (gamma x[1], 0, 0), the recovered
        spatial velocity gradient must be the analytic L[0,1] = gamma
        (not the buggy 2 * gamma).
        """
        alpha = 2.0
        gamma = 0.1
        L = self._run_g2p_for_alpha(alpha=alpha, gamma=gamma)
        # L[:, 0, 1] should be gamma for every particle
        L01 = L[:, 0, 1]
        assert torch.allclose(
            L01, torch.full_like(L01, gamma), atol=1e-7,
        ), (
            f"Expected L[0,1] == {gamma}, got min={L01.min():.6e}, "
            f"max={L01.max():.6e}"
        )
        # All other entries should be ~0
        L00 = L[:, 0, 0]
        L11 = L[:, 1, 1]
        assert torch.allclose(L00, torch.zeros_like(L00), atol=1e-7)
        assert torch.allclose(L11, torch.zeros_like(L11), atol=1e-7)

    def test_affine_decoder_j_eq_half_I_velocity_gradient(self):
        """Mirror test with alpha = 0.5.  Without the fix vel_grad[0,1]
        would be 0.5 * gamma instead of gamma.
        """
        alpha = 0.5
        gamma = 0.1
        L = self._run_g2p_for_alpha(alpha=alpha, gamma=gamma)
        L01 = L[:, 0, 1]
        assert torch.allclose(
            L01, torch.full_like(L01, gamma), atol=1e-7,
        ), (
            f"Expected L[0,1] == {gamma}, got min={L01.min():.6e}, "
            f"max={L01.max():.6e}"
        )


# ── Test 2: static equilibrium on a curved chart ─────────────────────


class TestStaticEquilibriumCurvedChart:
    def test_curved_chart_particles_at_rest_stay_at_rest(self):
        """Particles at rest with F = I should remain at rest under
        an arbitrary non-identity decoder (alpha = 2), with no gravity
        and fixed BCs.  Checks that the P2G internal force and the
        G2P update are internally consistent on the curved chart.
        """
        decoder = AffineDecoder(alpha=2.0).double()
        frame = _dummy_frame()
        solver = ChartMPMSolver(
            n_cells=8, extent=0.5,
            gravity=None,
            bc_type="fixed",
            decoder=decoder,
            **frame,
        )
        particles = MaterialPointCloud.create_uniform(
            n_per_axis=3, extent=0.3, density=1000.0,
        )
        # Run 10 steps
        for _ in range(10):
            diag = solver.step(particles, dt=1e-4)
            assert diag["kinetic_energy"] < 1e-12, (
                f"KE leaked: {diag['kinetic_energy']:.3e}"
            )
        # Particle positions must be within epsilon of their initial
        # values (rest remains rest)
        assert particles.v.abs().max().item() < 1e-12


# ── Test 3: identity decoder matches no-decoder path ─────────────────


class TestIdentityDecoderMatchesNoDecoderPath:
    def test_alpha_1_affine_matches_no_decoder(self):
        """ChartMPMSolver(decoder=AffineDecoder(alpha=1), ...) must
        produce byte-identical particles.F and particles.v after 5
        steps as ChartMPMSolver() with no decoder.  The whole point
        of the fix is that it should NOT change the identity path.
        """
        torch.manual_seed(42)
        # Shared initial conditions
        p_a = MaterialPointCloud.create_uniform(
            n_per_axis=3, extent=0.3, density=1000.0,
        )
        p_a.v[:, 0] = 0.1  # give them some velocity so there's action
        p_b = copy.deepcopy(p_a)

        # Solver A: no decoder (identity path)
        solver_a = ChartMPMSolver(
            n_cells=8, extent=0.5,
            gravity=(0.0, 0.0, 0.0),
            bc_type="free",
        )

        # Solver B: explicit alpha=1 decoder (curved path that
        # happens to be identity)
        decoder = AffineDecoder(alpha=1.0).double()
        frame = _dummy_frame()
        solver_b = ChartMPMSolver(
            n_cells=8, extent=0.5,
            gravity=(0.0, 0.0, 0.0),
            bc_type="free",
            decoder=decoder,
            **frame,
        )

        for _ in range(5):
            solver_a.step(p_a, dt=1e-4)
            solver_b.step(p_b, dt=1e-4)

        # Particles.F and particles.v must match to machine precision
        assert torch.allclose(p_a.F, p_b.F, atol=1e-12), (
            "F diverged between identity and alpha=1 paths"
        )
        assert torch.allclose(p_a.v, p_b.v, atol=1e-12), (
            "v diverged between identity and alpha=1 paths"
        )
        assert torch.allclose(p_a.xi, p_b.xi, atol=1e-12), (
            "xi diverged between identity and alpha=1 paths"
        )


# ── Test 4: contact invariance under decoder ─────────────────────────


class TestContactInvariantUnderDecoder:
    def test_ball_contact_force_unchanged_under_alpha_1_decoder(self):
        """Contact forces are computed in physical space via evaluate_gap
        on x_phys.  Under alpha = 1, x_phys == xi, so the contact
        response should be identical whether or not a decoder is
        supplied.  This is a regression guarantee that the fix does
        not inadvertently change the contact path.
        """
        torch.manual_seed(0)
        # Two particle clouds, same initial state
        p_a = MaterialPointCloud.create_uniform(
            n_per_axis=3, extent=0.04, density=1000.0,
        )
        p_a.xi[:, 1] -= 0.005  # pull into the floor a bit
        p_b = copy.deepcopy(p_a)

        # Two solvers
        solver_a = ChartMPMSolver(
            n_cells=8, extent=0.3,
            gravity=(0.0, -9.81, 0.0),
            bc_type="free",
        )
        decoder = AffineDecoder(alpha=1.0).double()
        frame = _dummy_frame()
        solver_b = ChartMPMSolver(
            n_cells=8, extent=0.3,
            gravity=(0.0, -9.81, 0.0),
            bc_type="free",
            decoder=decoder,
            **frame,
        )

        floor = FloorSDF(y_floor=0.0)
        eps_n = 1e5

        for _ in range(20):
            # Compute and step solver A
            x_a = p_a.xi.clone()
            gap_a, normal_a = evaluate_gap(x_a, floor)
            cf_a = compute_contact_force(
                gap_a, normal_a, p_a.current_volume, eps_n,
            )
            solver_a.step(p_a, 1e-5, contact_force=cf_a)

            # Compute and step solver B (must use same contact force)
            x_b = p_b.xi.clone()
            gap_b, normal_b = evaluate_gap(x_b, floor)
            cf_b = compute_contact_force(
                gap_b, normal_b, p_b.current_volume, eps_n,
            )
            solver_b.step(p_b, 1e-5, contact_force=cf_b)

        # Trajectories must match to high precision
        assert torch.allclose(p_a.xi, p_b.xi, atol=1e-10), (
            "Contact trajectories diverged under alpha=1 decoder"
        )
        assert torch.allclose(p_a.v, p_b.v, atol=1e-10)
        assert torch.allclose(p_a.F, p_b.F, atol=1e-10)
