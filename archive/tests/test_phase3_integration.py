"""Phase 3 integration tests: live chart spawning in Schwarz solvers.

Tests verify the complete pipeline:
    TopologyMonitor -> TopologyEvent -> ChartSpawner -> SpawnedChartPair
    -> SchwarzSolver.add_charts() -> solver continues with new charts

Also tests the biaxial tension scenario end-to-end.
"""

import math

import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# Warm-start decoder
# ---------------------------------------------------------------------------
class TestDecoderWarmStart:
    def test_warm_start_copies_weights(self):
        """warm_start_from() should copy all weights from parent."""
        from common.models import ChartDecoder

        parent = ChartDecoder(width=32, depth=3).float()
        # Modify parent weights so they're non-default
        with torch.no_grad():
            for p in parent.parameters():
                p.fill_(0.42)

        child = ChartDecoder(width=32, depth=3).float()
        child.warm_start_from(parent)

        # All parameters should match
        for p_par, p_child in zip(parent.parameters(), child.parameters()):
            assert torch.allclose(p_par, p_child), "Warm-start did not copy weights"

    def test_warm_start_is_deep_copy(self):
        """Modifying child should not affect parent after warm-start."""
        from common.models import ChartDecoder

        parent = ChartDecoder(width=16, depth=2).float()
        child = ChartDecoder(width=16, depth=2).float()
        child.warm_start_from(parent)

        # Modify child
        with torch.no_grad():
            child.raw_scale.fill_(99.0)

        assert parent.raw_scale.item() != 99.0, "Warm-start should be a deep copy"

    def test_warm_started_decoder_produces_same_output(self):
        """Parent and warm-started child should produce identical output."""
        from common.models import ChartDecoder

        parent = ChartDecoder(width=16, depth=2).float()
        child = ChartDecoder(width=16, depth=2).float()
        child.warm_start_from(parent)

        seed = torch.zeros(3, dtype=torch.float32)
        t1 = torch.tensor([1, 0, 0], dtype=torch.float32)
        t2 = torch.tensor([0, 1, 0], dtype=torch.float32)
        n = torch.tensor([0, 0, 1], dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32)
        xi = torch.randn(5, 3, dtype=torch.float32)

        with torch.no_grad():
            out_parent = parent(xi, seed=seed, t1=t1, t2=t2, n=n, chart_scale=scale)
            out_child = child(xi, seed=seed, t1=t1, t2=t2, n=n, chart_scale=scale)

        assert torch.allclose(out_parent, out_child, atol=1e-6)


# ---------------------------------------------------------------------------
# SchwarzFEMSolver.add_charts()
# ---------------------------------------------------------------------------
class TestSchwarzFEMAddCharts:
    def _make_mock_atlas(self, n_charts=2):
        """Create minimal mock atlas data for testing."""
        seeds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])[:n_charts]
        frames = np.array([[1, 0, 0], [0, 1, 0]])[:n_charts]
        normals = np.array([[0, 0, 1], [0, 0, 1]])[:n_charts]
        radii = np.array([0.5, 0.5])[:n_charts]
        membership = np.ones((10, n_charts), dtype=np.uint8)

        return {
            "seed_points": seeds,
            "frame_t1": frames,
            "frame_t2": np.array([[0, 1, 0], [1, 0, 0]])[:n_charts],
            "frame_n": normals,
            "support_radii": radii,
            "membership": membership,
        }

    def test_add_charts_increases_count(self):
        """add_charts() should increase n_charts by 2 per pair."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver
        from common.models import ChartDecoder, MaskNet
        from atlas.topo.chart_spawn import SpawnedChartPair

        atlas_data = self._make_mock_atlas(n_charts=2)
        decoders = [ChartDecoder(width=16, depth=2).double() for _ in range(2)]
        masks = [MaskNet(width=16, depth=2).double() for _ in range(2)]

        solver = SchwarzFEMSolver(
            atlas_data=atlas_data,
            decoders=decoders,
            masks=masks,
            n_cells=4,
        )
        assert solver.n_charts == 2

        pair = SpawnedChartPair(
            seed_plus=np.array([0.5, 0.5, 0.0]),
            seed_minus=np.array([0.5, -0.5, 0.0]),
            frame_plus=np.eye(3),
            frame_minus=np.eye(3),
            radius=0.3,
            parent_chart=0,
            edge_type="crack",
            activation_step=1,
        )

        n_added = solver.add_charts([pair])
        assert n_added == 2
        assert solver.n_charts == 4
        assert len(solver.decoders) == 4
        assert len(solver.fem_solvers) == 4

    def test_spawned_decoder_is_warm_started(self):
        """New decoders should have same weights as parent."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver
        from common.models import ChartDecoder, MaskNet
        from atlas.topo.chart_spawn import SpawnedChartPair

        atlas_data = self._make_mock_atlas(n_charts=1)
        parent = ChartDecoder(width=16, depth=2).double()
        decoders = [parent]
        masks = [MaskNet(width=16, depth=2).double()]

        solver = SchwarzFEMSolver(
            atlas_data=atlas_data, decoders=decoders, masks=masks, n_cells=4,
        )

        pair = SpawnedChartPair(
            seed_plus=np.array([0.3, 0.0, 0.0]),
            seed_minus=np.array([-0.3, 0.0, 0.0]),
            frame_plus=np.eye(3), frame_minus=np.eye(3),
            radius=0.2, parent_chart=0,
            edge_type="crack", activation_step=0,
        )
        solver.add_charts([pair])

        # The two new decoders (indices 1, 2) should have parent's weights
        for new_dec in solver.decoders[1:]:
            for p_par, p_new in zip(parent.parameters(), new_dec.parameters()):
                assert torch.allclose(p_par, p_new), "Spawned decoder not warm-started"


# ---------------------------------------------------------------------------
# Full biaxial tension scenario
# ---------------------------------------------------------------------------
class TestBiaxialTensionPhase3:
    """End-to-end: biaxial tension with topology monitoring and chart spawning."""

    def test_monitor_spawns_charts_on_crack_nucleation(self):
        """When a crack nucleates in the biaxial test, the monitor should
        fire an event and the spawner should produce a chart pair.
        """
        from atlas.topo.monitor import TopologyMonitor
        from atlas.topo.chart_spawn import ChartSpawner
        from atlas.topo.filtration import clip_to_interior
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark

        bench = BiaxialTensionBenchmark(R=3.0, L=2.0)
        monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1),
        )
        spawner = ChartSpawner()

        # Step 0: intact
        grid_intact = clip_to_interior(bench.sdf_intact())
        monitor.update(grid_intact, load_step=0)

        # Step 1: crack nucleates
        grid_cracked = clip_to_interior(bench.sdf_cracked(delta=0.1))
        events = monitor.update(grid_cracked, load_step=1)

        assert len(events) >= 1, "Monitor should detect crack"

        # Spawn charts for the event
        existing_seeds = np.array([[0.0, 0.0, 0.0]])
        existing_frames = np.array([np.eye(3)])
        extent = max(bench.R, bench.L) * 1.3
        bbox_min = np.array([-extent, -extent, -bench.L])
        bbox_max = np.array([extent, extent, bench.L])

        pair = spawner.spawn_from_event(
            events[0], existing_seeds, existing_frames,
            grid_cracked, bbox_min, bbox_max,
        )

        assert pair is not None
        assert pair.edge_type == "crack"
        assert pair.radius > 0
        # Seeds should be on opposite sides of the feature
        dist = np.linalg.norm(pair.seed_plus - pair.seed_minus)
        assert dist > 0, "Chart seeds should be separated"

    def test_fem_solver_survives_add_charts(self):
        """SchwarzFEMSolver should remain functional after adding charts."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver
        from common.models import ChartDecoder, MaskNet
        from atlas.topo.chart_spawn import SpawnedChartPair

        # Build a minimal 1-chart solver
        atlas_data = {
            "seed_points": np.array([[0.0, 0.0, 0.0]]),
            "frame_t1": np.array([[1.0, 0.0, 0.0]]),
            "frame_t2": np.array([[0.0, 1.0, 0.0]]),
            "frame_n": np.array([[0.0, 0.0, 1.0]]),
            "support_radii": np.array([1.0]),
            "membership": np.ones((5, 1), dtype=np.uint8),
        }
        decoders = [ChartDecoder(width=16, depth=2).double()]
        masks = [MaskNet(width=16, depth=2).double()]

        solver = SchwarzFEMSolver(
            atlas_data=atlas_data, decoders=decoders, masks=masks, n_cells=4,
        )

        # Add a chart pair
        pair = SpawnedChartPair(
            seed_plus=np.array([0.5, 0.0, 0.0]),
            seed_minus=np.array([-0.5, 0.0, 0.0]),
            frame_plus=np.eye(3), frame_minus=np.eye(3),
            radius=0.4, parent_chart=0,
            edge_type="crack", activation_step=0,
        )
        solver.add_charts([pair])

        # Solver should still have valid state
        assert solver.n_charts == 3
        assert len(solver.fem_solvers) == 3
        assert len(solver.neighbors) == 3
        assert len(solver.color_groups) >= 1

        # Each FEM solver should have nodes
        for i, fs in enumerate(solver.fem_solvers):
            assert fs.n_nodes > 0, f"Chart {i} has no nodes"

    def test_mpm_solver_add_charts(self):
        """SchwarzMPMSolver.add_charts() should create new solvers."""
        from solvers.mpm.schwarz_mpm import SchwarzMPMSolver
        from common.models import ChartDecoder, MaskNet
        from atlas.topo.chart_spawn import SpawnedChartPair

        atlas_data = {
            "seed_points": np.array([[0.0, 0.0, 0.0]]),
            "frame_t1": np.array([[1.0, 0.0, 0.0]]),
            "frame_t2": np.array([[0.0, 1.0, 0.0]]),
            "frame_n": np.array([[0.0, 0.0, 1.0]]),
            "support_radii": np.array([1.0]),
            "membership": np.ones((5, 1), dtype=np.uint8),
        }
        decoders = [ChartDecoder(width=16, depth=2).double()]
        masks = [MaskNet(width=16, depth=2).double()]

        solver = SchwarzMPMSolver(
            atlas_data=atlas_data, decoders=decoders, masks=masks,
            n_cells=4,
        )
        assert solver.n_charts == 1

        pair = SpawnedChartPair(
            seed_plus=np.array([0.3, 0.0, 0.0]),
            seed_minus=np.array([-0.3, 0.0, 0.0]),
            frame_plus=np.eye(3), frame_minus=np.eye(3),
            radius=0.3, parent_chart=0,
            edge_type="crack", activation_step=0,
        )
        n = solver.add_charts([pair])
        assert n == 2
        assert solver.n_charts == 3
        assert len(solver.solvers) == 3
        assert len(solver.particles) == 3
