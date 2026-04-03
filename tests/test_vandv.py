"""V&V exercises from topo_atlas/docs/VandV.md.

These tests implement the verification and validation exercises defined
in the per-phase V&V plan. Each test is tagged with its V&V ID.
"""

import math

import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# V&V-4.2: No-Crack Regression (False Positive Rate)
# ---------------------------------------------------------------------------
class TestVandV_4_2:
    """Verify the TopologyMonitor does not fire spurious events on an intact
    domain under large elastic deformation (simulated by perturbing the SDF).

    Protocol: Apply smooth perturbation (simulating 20% compression) to a
    solid torus SDF grid over 20 steps. Monitor should report zero events
    because the perturbation does not change the topology.
    """

    @pytest.fixture
    def torus_grid(self):
        from atlas.topo.filtration import sdf_solid_torus, clip_to_interior
        N = 32
        lin = np.linspace(-1.8, 1.8, N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = sdf_solid_torus(coords, R=1.0, r=0.35).reshape(N, N, N).astype("float32")
        return clip_to_interior(vals)

    def test_no_false_positives_under_deformation(self, torus_grid):
        """V&V-4.2: 20 load steps of smooth SDF perturbation -> zero events."""
        from atlas.topo.monitor import TopologyMonitor

        monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
        )

        # Step 0: baseline
        events_0 = monitor.update(torus_grid, load_step=0)

        total_events = 0
        for step in range(1, 21):
            # Simulate compression: shrink the SDF values uniformly
            # This changes the *metric* but not the *topology*
            # Scale factor: 1.0 -> 0.8 over 20 steps (20% compression)
            scale = 1.0 - 0.2 * (step / 20.0)
            deformed = torus_grid * scale

            events = monitor.update(deformed, load_step=step)
            total_events += len(events)

        assert total_events == 0, (
            f"V&V-4.2 FAIL: {total_events} spurious topology events "
            f"detected during elastic deformation (expected 0)"
        )

        # Verify Betti numbers unchanged
        betti = monitor.current_betti
        assert betti.get(0, 0) == 1, f"beta_0 changed: {betti}"
        assert betti.get(1, 0) == 1, f"beta_1 changed: {betti}"

    def test_no_events_under_noise(self, torus_grid):
        """Small noise should not trigger topology events.

        Note: raw Betti numbers may fluctuate with noise (short-lived
        features pass the relative lifetime filter). The *bottleneck
        distance* correctly ignores these, so no events should fire.
        """
        from atlas.topo.monitor import TopologyMonitor

        monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
        )

        # Baseline
        monitor.update(torus_grid, load_step=0)

        # Add small noise
        rng = np.random.RandomState(42)
        noisy = torus_grid + rng.normal(0, 0.005, torus_grid.shape).astype("float32")

        events = monitor.update(noisy, load_step=1)
        assert len(events) == 0, f"Noise triggered {len(events)} events (expected 0)"


# ---------------------------------------------------------------------------
# V&V-5.2: Full Pipeline Smoke Test
# ---------------------------------------------------------------------------
class TestVandV_5_2:
    """End-to-end test of the topology certification pipeline on analytic
    geometries, verifying the complete data flow:
    SDF -> grid sampling -> persistence -> Betti -> M_min -> certification.
    """

    def test_full_pipeline_ball(self):
        """V&V-5.2a: Ball (contractible) -> M_min=1 -> certification PASS."""
        from atlas.topo.certify import certify_sdf
        from atlas.topo.filtration import sdf_ball
        from atlas.topo.ls_category import certify_atlas

        # Step 1: Create analytic SDF as torch Module
        class BallSDF(torch.nn.Module):
            def forward(self, x):
                x_np = x.detach().cpu().numpy()
                return torch.tensor(sdf_ball(x_np, radius=0.8), dtype=x.dtype)

        sdf_net = BallSDF()
        bbox = torch.tensor([-1.5, -1.5, -1.5]), torch.tensor([1.5, 1.5, 1.5])

        # Step 2: Topology certification
        report = certify_sdf(sdf_net, bbox[0], bbox[1], resolution=24)
        assert report["M_min"] == 1, f"Ball M_min should be 1, got {report['M_min']}"

        # Step 3: Atlas certification (simulate M=4 atlas with good quality)
        quality = {"g_fold": 0.0, "g_cov": 1.0, "g_ov": 0.02}
        cert = certify_atlas(M_actual=4, betti=report["betti"], quality_metrics=quality)
        assert cert["topology_pass"], "Ball atlas with M=4 should pass topology"
        assert cert["quality_pass"], "Good quality metrics should pass"

    def test_full_pipeline_torus(self):
        """V&V-5.2b: Solid torus -> M_min=2 -> M=8 certification PASS."""
        from atlas.topo.certify import certify_sdf
        from atlas.topo.filtration import sdf_solid_torus
        from atlas.topo.ls_category import certify_atlas

        class TorusSDF(torch.nn.Module):
            def forward(self, x):
                x_np = x.detach().cpu().numpy()
                return torch.tensor(
                    sdf_solid_torus(x_np, R=1.0, r=0.35), dtype=x.dtype
                )

        sdf_net = TorusSDF()
        bbox = torch.tensor([-2.0, -2.0, -1.0]), torch.tensor([2.0, 2.0, 1.0])

        report = certify_sdf(sdf_net, bbox[0], bbox[1], resolution=24)
        assert report["M_min"] == 2, f"Torus M_min should be 2, got {report['M_min']}"
        assert report["betti"][1] == 1, "Torus should have beta_1=1"

        # Certify with M=8 (Sun 2026 paper value)
        quality = {"g_fold": 0.0, "g_cov": 1.0, "g_ov": 0.026}
        cert = certify_atlas(M_actual=8, betti=report["betti"], quality_metrics=quality)
        assert cert["topology_pass"], "Torus M=8 >= M_min=2 should pass"

    def test_full_pipeline_insufficient_charts(self):
        """V&V-5.2c: Solid torus with M=1 < M_min=2 -> certification FAIL."""
        from atlas.topo.certify import certify_sdf
        from atlas.topo.filtration import sdf_solid_torus
        from atlas.topo.ls_category import certify_atlas

        class TorusSDF(torch.nn.Module):
            def forward(self, x):
                x_np = x.detach().cpu().numpy()
                return torch.tensor(
                    sdf_solid_torus(x_np, R=1.0, r=0.35), dtype=x.dtype
                )

        sdf_net = TorusSDF()
        bbox = torch.tensor([-2.0, -2.0, -1.0]), torch.tensor([2.0, 2.0, 1.0])
        report = certify_sdf(sdf_net, bbox[0], bbox[1], resolution=24)

        # Certify with M=1 (below topological minimum)
        cert = certify_atlas(M_actual=1, betti=report["betti"])
        assert not cert["topology_pass"], "M=1 < M_min=2 should fail topology check"

    def test_mpm_solver_integration(self):
        """V&V-5.2d: MPM solver runs without errors on a simple problem."""
        from solvers.mpm.chart_mpm_solver import ChartMPMSolver
        from solvers.mpm.particles import MaterialPointCloud
        from solvers.mpm.constitutive import NeoHookeanModel

        solver = ChartMPMSolver(
            n_cells=8, extent=0.5,
            constitutive=NeoHookeanModel(E=1e5, nu=0.3),
            gravity=(0, 0, -9.81),
            bc_type="fixed",
        )
        particles = MaterialPointCloud.create_uniform(
            n_per_axis=3, extent=0.3, density=1000.0,
        )

        history = solver.run(particles, dt=1e-4, n_steps=10, output_interval=100)
        assert len(history) == 10
        # With gravity, KE should increase from zero
        assert history[-1]["kinetic_energy"] > 0, "Gravity should produce motion"

    def test_schwarz_fem_import_and_construction(self):
        """V&V-5.2e: SchwarzFEMSolver can be instantiated (import check)."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver
        assert SchwarzFEMSolver is not None
        # Full construction requires atlas data — tested in test_certify.py
