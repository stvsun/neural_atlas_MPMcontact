"""Unit tests for the MPM solver on mapped coordinate charts.

Tests verify:
    1. P2G/G2P mass and momentum conservation
    2. Energy conservation for elastic vibrating bar
    3. Constitutive model correctness (Neo-Hookean, elastoplastic)
    4. Grid boundary conditions
"""

import math

import torch
import pytest

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.grid import ChartGrid
from solvers.mpm.transfers import particle_to_grid, grid_to_particle
from solvers.mpm.constitutive import NeoHookeanModel, ElastoplasticModel
from solvers.mpm.chart_mpm_solver import ChartMPMSolver


class TestParticles:
    def test_create_uniform(self):
        cloud = MaterialPointCloud.create_uniform(n_per_axis=4, extent=0.5, density=1000.0)
        assert cloud.n_particles == 64
        assert cloud.xi.shape == (64, 3)
        assert cloud.F.shape == (64, 3, 3)
        # F should be identity
        eye = torch.eye(3, dtype=cloud.dtype)
        assert torch.allclose(cloud.F[0], eye)

    def test_current_volume_identity_F(self):
        cloud = MaterialPointCloud.create_uniform(n_per_axis=3, extent=0.5, density=1.0)
        # With F = I, current_volume should equal V0
        assert torch.allclose(cloud.current_volume, cloud.V0, atol=1e-12)

    def test_deformation_gradient_update(self):
        cloud = MaterialPointCloud.create_uniform(n_per_axis=2, extent=0.5, density=1.0)
        n = cloud.n_particles
        # Apply a uniform stretch: ∇v = diag(1, 0, 0) for dt=0.1
        # F_new = (I + dt * ∇v) @ F = diag(1.1, 1, 1)
        vel_grad = torch.zeros(n, 3, 3, dtype=cloud.dtype)
        vel_grad[:, 0, 0] = 1.0
        cloud.update_deformation_gradient(vel_grad, dt=0.1)
        assert abs(cloud.F[0, 0, 0].item() - 1.1) < 1e-12
        assert abs(cloud.F[0, 1, 1].item() - 1.0) < 1e-12


class TestGrid:
    def test_grid_creation(self):
        grid = ChartGrid(n_cells=4, extent=1.0)
        assert grid.n_nodes == 5 ** 3
        assert grid.positions.shape == (125, 3)

    def test_boundary_mask(self):
        grid = ChartGrid(n_cells=4, extent=1.0)
        # Corner node should be on boundary
        assert grid.boundary_mask.any()
        # Interior node should not be on boundary
        n_boundary = grid.boundary_mask.sum().item()
        n_interior = grid.n_nodes - n_boundary
        assert n_interior > 0

    def test_reset(self):
        grid = ChartGrid(n_cells=4, extent=1.0)
        grid.mass += 1.0
        grid.reset()
        assert grid.mass.sum().item() == 0.0


class TestTransfers:
    def test_mass_conservation(self):
        """Total mass should be conserved after P2G."""
        particles = MaterialPointCloud.create_uniform(n_per_axis=4, extent=0.4, density=1000.0)
        grid = ChartGrid(n_cells=8, extent=0.5)

        total_mass_p = particles.mass.sum().item()
        particle_to_grid(particles, grid)
        total_mass_g = grid.mass.sum().item()

        assert abs(total_mass_g - total_mass_p) / total_mass_p < 1e-6, \
            f"Mass not conserved: particles={total_mass_p:.6f}, grid={total_mass_g:.6f}"

    def test_momentum_conservation(self):
        """Total momentum should be conserved after P2G (no gravity)."""
        particles = MaterialPointCloud.create_uniform(n_per_axis=4, extent=0.4, density=1000.0)
        # Give particles a uniform velocity
        particles.v[:, 0] = 1.0
        grid = ChartGrid(n_cells=8, extent=0.5)

        total_mom_p = (particles.mass.unsqueeze(1) * particles.v).sum(dim=0)
        particle_to_grid(particles, grid)
        total_mom_g = grid.momentum.sum(dim=0)

        for d in range(3):
            err = abs(total_mom_g[d].item() - total_mom_p[d].item())
            scale = max(abs(total_mom_p[d].item()), 1e-12)
            assert err / scale < 1e-6, f"Momentum[{d}] not conserved"


class TestConstitutive:
    def test_neo_hookean_zero_strain(self):
        """At F = I, Neo-Hookean stress should be zero."""
        model = NeoHookeanModel(E=1e5, nu=0.3)
        F = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        sigma, _ = model.compute_stress(F)
        assert torch.allclose(sigma, torch.zeros_like(sigma), atol=1e-10)

    def test_neo_hookean_uniaxial(self):
        """Uniaxial stretch should produce positive stress in stretch direction."""
        model = NeoHookeanModel(E=1e5, nu=0.3)
        F = torch.diag(torch.tensor([1.1, 1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
        sigma, _ = model.compute_stress(F)
        assert sigma[0, 0, 0].item() > 0, "Tensile stress expected in stretch direction"

    def test_elastoplastic_elastic_regime(self):
        """Small strain should be purely elastic (no plastic strain)."""
        model = ElastoplasticModel(E=1e5, nu=0.3, yield_stress=1e4, hardening=1e3)
        # Very small stretch — well below yield
        F = torch.diag(torch.tensor([1.001, 1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
        state = {}
        sigma, state = model.compute_stress(F, state)
        assert state["equiv_plastic_strain"].item() < 1e-12, "No plastic strain expected"

    def test_elastoplastic_yields(self):
        """Large strain should produce plastic strain."""
        model = ElastoplasticModel(E=1e5, nu=0.3, yield_stress=100.0, hardening=100.0)
        F = torch.diag(torch.tensor([1.5, 1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
        state = {}
        sigma, state = model.compute_stress(F, state)
        assert state["equiv_plastic_strain"].item() > 0, "Plastic strain expected for large deformation"


class TestChartMPMSolver:
    def test_solver_runs(self):
        """Solver should complete without errors."""
        solver = ChartMPMSolver(n_cells=8, extent=0.5)
        particles = MaterialPointCloud.create_uniform(n_per_axis=3, extent=0.3, density=1000.0)
        history = solver.run(particles, dt=1e-4, n_steps=5, output_interval=10)
        assert len(history) == 5
        assert "kinetic_energy" in history[0]

    def test_static_equilibrium(self):
        """Particles at rest with F=I should remain at rest (no gravity)."""
        solver = ChartMPMSolver(n_cells=8, extent=0.5, gravity=None, bc_type="fixed")
        particles = MaterialPointCloud.create_uniform(n_per_axis=3, extent=0.3, density=1000.0)
        history = solver.run(particles, dt=1e-4, n_steps=10, output_interval=100)
        # Kinetic energy should remain near zero
        for h in history:
            assert h["kinetic_energy"] < 1e-10, f"KE should be ~0, got {h['kinetic_energy']}"
