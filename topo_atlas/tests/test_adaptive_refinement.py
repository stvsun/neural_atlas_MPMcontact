"""Tests for persistence-driven adaptive mesh refinement.

Validates that TopologyEvent.lifetime drives SpawnedChartPair.recommended_n_cells
and recommended_element_order, enabling topology-aware hp-adaptivity.
"""
import numpy as np
import pytest

from atlas.topo.chart_spawn import ChartSpawner, SpawnedChartPair
from atlas.topo.monitor import TopologyEvent


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

def _make_event(lifetime: float, dimension: int = 1, load_step: int = 5):
    """Create a TopologyEvent with specified lifetime and dimension."""
    return TopologyEvent(
        load_step=load_step,
        event_type=f"new_H{dimension}",
        dimension=dimension,
        birth_value=-0.3,
        lifetime=lifetime,
        localization=np.array([1.0, 0.0, 0.0]),
        bottleneck_change=lifetime * 0.5,
    )


def _make_grid(sdf_range: float = 1.0, resolution: int = 16):
    """Create a dummy SDF grid with known range."""
    grid = np.linspace(-sdf_range / 2, sdf_range / 2, resolution)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    return gx  # values range from -sdf_range/2 to +sdf_range/2


def _spawn_pair(lifetime: float, dimension: int = 1, base_n_cells: int = 8):
    """Run the full spawn pipeline and return the pair."""
    event = _make_event(lifetime, dimension)
    spawner = ChartSpawner(
        default_radius=0.3, verbose=False, base_n_cells=base_n_cells
    )
    grid = _make_grid(sdf_range=1.0)
    bbox_min = np.array([-0.5, -0.5, -0.5])
    bbox_max = np.array([0.5, 0.5, 0.5])
    seeds = np.array([[0.0, 0.0, 0.0]])
    frames = np.eye(3)[None, :, :]  # (1, 3, 3)
    return spawner.spawn_from_event(
        event, seeds, frames, grid, bbox_min, bbox_max
    )


# ────────────────────────────────────────────────────────────────────
# Test A4.1: SpawnedChartPair has the new fields
# ────────────────────────────────────────────────────────────────────

class TestAdaptiveFields:
    def test_fields_exist(self):
        pair = SpawnedChartPair(
            seed_plus=np.zeros(3),
            seed_minus=np.zeros(3),
            frame_plus=np.eye(3),
            frame_minus=np.eye(3),
            radius=0.3,
            parent_chart=0,
        )
        assert hasattr(pair, "recommended_n_cells")
        assert hasattr(pair, "recommended_element_order")

    def test_default_values(self):
        pair = SpawnedChartPair(
            seed_plus=np.zeros(3),
            seed_minus=np.zeros(3),
            frame_plus=np.eye(3),
            frame_minus=np.eye(3),
            radius=0.3,
            parent_chart=0,
        )
        assert pair.recommended_n_cells == 8
        assert pair.recommended_element_order == 1


# ────────────────────────────────────────────────────────────────────
# Test A4.2: Short-lived feature → base n_cells
# ────────────────────────────────────────────────────────────────────

class TestShortLifetime:
    def test_short_lifetime_h1_gets_minimum_12(self):
        """Even short-lived H1 features get at least 12 cells."""
        pair = _spawn_pair(lifetime=0.01, dimension=1, base_n_cells=8)
        assert pair.recommended_n_cells >= 12

    def test_short_lifetime_h0_gets_base(self):
        """Short-lived H0 features get base n_cells."""
        pair = _spawn_pair(lifetime=0.01, dimension=0, base_n_cells=8)
        assert pair.recommended_n_cells == 8


# ────────────────────────────────────────────────────────────────────
# Test A4.3: Long-lived H1 feature → n_cells >= 12
# ────────────────────────────────────────────────────────────────────

class TestLongLifetime:
    def test_long_lived_h1(self):
        pair = _spawn_pair(lifetime=0.5, dimension=1, base_n_cells=8)
        assert pair.recommended_n_cells >= 12

    def test_longer_lifetime_increases_n_cells(self):
        pair_short = _spawn_pair(lifetime=0.1, dimension=1)
        pair_long = _spawn_pair(lifetime=0.8, dimension=1)
        assert pair_long.recommended_n_cells >= pair_short.recommended_n_cells


# ────────────────────────────────────────────────────────────────────
# Test A4.4: Very long-lived feature → capped at 3× base
# ────────────────────────────────────────────────────────────────────

class TestCapped:
    def test_capped_at_3x_base(self):
        pair = _spawn_pair(lifetime=10.0, dimension=1, base_n_cells=8)
        assert pair.recommended_n_cells <= 3 * 8  # = 24

    def test_capped_at_3x_base_custom(self):
        pair = _spawn_pair(lifetime=10.0, dimension=1, base_n_cells=10)
        assert pair.recommended_n_cells <= 3 * 10  # = 30


# ────────────────────────────────────────────────────────────────────
# Test A4.5: H1 event → recommended_element_order = 2
# ────────────────────────────────────────────────────────────────────

class TestElementOrder:
    def test_h1_gets_p2(self):
        pair = _spawn_pair(lifetime=0.3, dimension=1)
        assert pair.recommended_element_order == 2

    def test_h2_gets_p2(self):
        pair = _spawn_pair(lifetime=0.3, dimension=2)
        assert pair.recommended_element_order == 2

    def test_h0_gets_p1(self):
        pair = _spawn_pair(lifetime=0.3, dimension=0)
        assert pair.recommended_element_order == 1


# ────────────────────────────────────────────────────────────────────
# Test A4.6: Full pipeline integration
# ────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_spawner_produces_adaptive_pair(self):
        """ChartSpawner.spawn_from_event produces a pair with adaptive fields."""
        event = _make_event(lifetime=0.4, dimension=1)
        spawner = ChartSpawner(
            default_radius=0.5, verbose=False, base_n_cells=8
        )
        grid = _make_grid(sdf_range=1.0)
        bbox_min = np.array([-0.5, -0.5, -0.5])
        bbox_max = np.array([0.5, 0.5, 0.5])
        seeds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        frames = np.stack([np.eye(3), np.eye(3)])

        pair = spawner.spawn_from_event(
            event, seeds, frames, grid, bbox_min, bbox_max
        )

        assert isinstance(pair, SpawnedChartPair)
        assert pair.recommended_n_cells >= 12  # H1 minimum
        assert pair.recommended_element_order == 2  # H1 → P2
        assert pair.edge_type == "crack"
        assert pair.activation_step == 5

    def test_h2_event_produces_h2_refinement(self):
        """H2 (void/cavitation) events produce at least 10 cells."""
        event = _make_event(lifetime=0.2, dimension=2)
        spawner = ChartSpawner(verbose=False, base_n_cells=6)
        grid = _make_grid()
        pair = spawner.spawn_from_event(
            event,
            np.zeros((1, 3)),
            np.eye(3)[None],
            grid,
            np.full(3, -0.5),
            np.full(3, 0.5),
        )
        assert pair.recommended_n_cells >= 10

    def test_spawn_log_records_adaptive_fields(self):
        """Spawn log retains recommended_n_cells and element_order."""
        spawner = ChartSpawner(verbose=False, base_n_cells=8)
        grid = _make_grid()
        pair = spawner.spawn_from_event(
            _make_event(0.6, 1),
            np.zeros((1, 3)),
            np.eye(3)[None],
            grid,
            np.full(3, -0.5),
            np.full(3, 0.5),
        )
        log = spawner.spawn_log
        assert len(log) == 1
        assert log[0].recommended_n_cells == pair.recommended_n_cells
        assert log[0].recommended_element_order == pair.recommended_element_order
