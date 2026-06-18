"""Tests for topology-aware contact detection via persistent homology.

Verifies the ``combined_sdf_grid`` helper and the
``ContactTopologyMonitor`` class: Betti-number computation, event
emission on β₀ transitions, and contact localization.
"""

import numpy as np
import torch
import pytest

from solvers.contact.contact_pair import ContactBody
from solvers.contact.contact_topology import (
    ContactTopologyEvent,
    ContactTopologyMonitor,
    combined_sdf_grid,
)
from solvers.contact.contact_chart_spawn import (
    compute_contact_normal,
    spawn_contact_chart_pair,
)


# ── Analytic SDFs (mutable centres so we can move bodies) ────────────


class MovableSphereSDF(torch.nn.Module):
    """phi(x) = |x - c| - r   with a mutable centre tensor."""

    def __init__(self, center, radius):
        super().__init__()
        self.register_buffer(
            "center",
            torch.tensor(center, dtype=torch.float64),
        )
        self.radius = float(radius)

    def set_center(self, new_center):
        self.center.copy_(torch.as_tensor(new_center, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center.unsqueeze(0)).norm(dim=1) - self.radius


def _make_body(body_id: int, center, radius: float) -> ContactBody:
    """Wrap a movable sphere as a ContactBody."""
    sdf = MovableSphereSDF(center=center, radius=radius).double()
    return ContactBody(
        body_id=body_id,
        sdf_net=sdf,
        seeds=torch.tensor([center], dtype=torch.float64),
        support_radii=torch.tensor([radius], dtype=torch.float64),
    )


# ── Combined SDF grid tests ──────────────────────────────────────────


class TestCombinedSDFGrid:
    def test_single_body_grid_contains_negative_values(self):
        """With one body the combined SDF is the body SDF itself, and
        the interior of the sphere shows up as negative values.
        """
        body = _make_body(0, center=(0.0, 0.0, 0.0), radius=0.5)
        bbox_min = np.array([-1.0, -1.0, -1.0])
        bbox_max = np.array([1.0, 1.0, 1.0])
        grid = combined_sdf_grid([body], bbox_min, bbox_max, resolution=16)
        assert grid.shape == (16, 16, 16)
        assert grid.min() < -0.3   # deep interior on a 16³ grid
        assert grid.max() > 0.3    # far corner outside

    def test_two_separated_bodies_min_is_negative(self):
        """Combined grid over two interior regions has negative min."""
        body_a = _make_body(0, center=(-0.5, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(0.5, 0.0, 0.0), radius=0.25)
        bbox_min = np.array([-1.0, -0.5, -0.5])
        bbox_max = np.array([1.0, 0.5, 0.5])

        grid = combined_sdf_grid(
            [body_a, body_b], bbox_min, bbox_max, resolution=16,
        )
        # Both sphere interiors produce negative values on the grid
        assert grid.min() < -0.1
        # Far corner of the bbox is outside both spheres
        assert grid.max() > 0.3

    def test_empty_body_list_raises(self):
        with pytest.raises(ValueError):
            combined_sdf_grid([], np.zeros(3), np.ones(3))


# ── Betti-number tests ──────────────────────────────────────────────


class TestCurrentBetti:
    def test_two_separated_spheres_beta0_is_2(self):
        """Well-separated bodies should give β₀ = 2."""
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        betti = monitor.current_betti()
        assert betti[0] == 2

    def test_two_overlapping_spheres_beta0_is_1(self):
        """Overlapping bodies should give β₀ = 1."""
        body_a = _make_body(0, center=(-0.15, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.15, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.0, -0.5, -0.5]),
            bbox_max=np.array([1.0, 0.5, 0.5]),
            resolution=16,
        )
        betti = monitor.current_betti()
        assert betti[0] == 1

    def test_single_body_beta0_is_1(self):
        body = _make_body(0, center=(0.0, 0.0, 0.0), radius=0.4)
        monitor = ContactTopologyMonitor(
            bodies=[body],
            bbox_min=np.array([-1.0, -1.0, -1.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
            resolution=16,
        )
        assert monitor.current_betti()[0] == 1


# ── Event emission tests ─────────────────────────────────────────────


class TestEventEmission:
    def test_first_update_returns_no_events(self):
        """The baseline update always returns []."""
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        events = monitor.update(load_step=0)
        assert events == []
        assert monitor.prev_beta0 == 2

    def test_stationary_no_events_after_baseline(self):
        """Consecutive updates without moving anything → no events."""
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        monitor.update(load_step=0)
        events = monitor.update(load_step=1)
        assert events == []

    def test_first_contact_event_emitted(self):
        """Moving two separated bodies into overlap emits
        ``first_contact`` with the right Betti-number transition.
        """
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        # Baseline with the two bodies separated
        monitor.update(load_step=0)
        assert monitor.prev_beta0 == 2

        # Move them into overlap
        body_a.sdf_net.set_center((-0.1, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.1, 0.0, 0.0))
        events = monitor.update(load_step=1)

        # Exactly one event, first_contact, β₀ 2 → 1
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "first_contact"
        assert ev.load_step == 1
        assert ev.beta0_before == 2
        assert ev.beta0_after == 1
        assert ev.location is not None
        assert ev.location.shape == (3,)

    def test_separation_event_emitted(self):
        """Reverse the motion: overlapping → separated emits
        ``separation``.
        """
        body_a = _make_body(0, center=(-0.1, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.1, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        monitor.update(load_step=0)
        assert monitor.prev_beta0 == 1

        body_a.sdf_net.set_center((-0.6, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.6, 0.0, 0.0))
        events = monitor.update(load_step=1)

        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "separation"
        assert ev.beta0_before == 1
        assert ev.beta0_after == 2
        # Separation events carry no physical location
        assert ev.location is None

    def test_event_history_accumulates(self):
        """``event_history`` accumulates emitted events across updates."""
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )
        monitor.update(load_step=0)

        # Close them
        body_a.sdf_net.set_center((-0.1, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.1, 0.0, 0.0))
        monitor.update(load_step=1)

        # Separate them
        body_a.sdf_net.set_center((-0.6, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.6, 0.0, 0.0))
        monitor.update(load_step=2)

        assert len(monitor.event_history) == 2
        assert [e.event_type for e in monitor.event_history] == [
            "first_contact", "separation",
        ]


# ── Localization test ───────────────────────────────────────────────


class TestContactLocalization:
    def test_contact_location_between_centres(self):
        """First-contact location lies inside the bounding box of the
        two body centres — i.e., between them, in the overlap region.
        """
        c_a = np.array([-0.1, 0.0, 0.0])
        c_b = np.array([+0.1, 0.0, 0.0])
        body_a = _make_body(0, center=c_a, radius=0.25)
        body_b = _make_body(1, center=c_b, radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.0, -0.5, -0.5]),
            bbox_max=np.array([1.0, 0.5, 0.5]),
            resolution=20,
        )
        # Force a first_contact event: baseline separated, then bring
        # together.
        body_a.sdf_net.set_center((-0.6, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.6, 0.0, 0.0))
        monitor.update(load_step=0)  # baseline

        body_a.sdf_net.set_center(c_a)
        body_b.sdf_net.set_center(c_b)
        events = monitor.update(load_step=1)
        assert len(events) == 1

        loc = events[0].location
        # Location should be inside the convex hull spanned by the two
        # centres with a generous tolerance for grid resolution.
        lo = np.minimum(c_a, c_b) - 0.2
        hi = np.maximum(c_a, c_b) + 0.2
        assert ((loc >= lo).all() and (loc <= hi).all()), (
            f"Contact location {loc} outside expected box [{lo}, {hi}]"
        )


# ── Integration test: multi-step approach ──────────────────────────


class TestMonitorIntegration:
    def test_monitor_across_approach_sweep(self):
        """Sweep two body centres from far apart to overlapping.

        Over many update calls, exactly one ``first_contact`` event
        should be emitted — at the moment β₀ transitions from 2 to 1.
        """
        body_a = _make_body(0, center=(-0.6, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.6, 0.0, 0.0), radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([1.2, 0.6, 0.6]),
            resolution=16,
        )

        # Sweep the centres from ±0.6 to ±0.1 in 6 steps
        distances = np.linspace(0.6, 0.1, 6)
        first_contact_step = None
        for step, d in enumerate(distances):
            body_a.sdf_net.set_center((-d, 0.0, 0.0))
            body_b.sdf_net.set_center((+d, 0.0, 0.0))
            events = monitor.update(load_step=step)
            for e in events:
                if e.event_type == "first_contact":
                    first_contact_step = step
                    break

        # Exactly one first_contact recorded, somewhere in the sweep
        first_contact_events = [
            e for e in monitor.event_history
            if e.event_type == "first_contact"
        ]
        assert len(first_contact_events) == 1
        assert first_contact_step is not None
        # Sanity: the transition happened after the baseline step 0
        assert first_contact_step > 0


# ── Contact chart spawning tests ─────────────────────────────────────


class TestComputeContactNormal:
    def test_sphere_normal_is_radial(self):
        """For a sphere SDF, the normal at any exterior point is radial."""
        body = _make_body(0, center=(0.0, 0.0, 0.0), radius=0.25)
        # Point outside along +x
        normal = compute_contact_normal(body, np.array([0.5, 0.0, 0.0]))
        assert np.allclose(normal, [1.0, 0.0, 0.0], atol=1e-10)

        # Point outside along +y
        normal = compute_contact_normal(body, np.array([0.0, 0.7, 0.0]))
        assert np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-10)

        # Unit magnitude
        n = compute_contact_normal(
            body, np.array([0.3, 0.4, 0.12])
        )
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_offset_sphere_normal_points_outward(self):
        """Normal of an off-centre sphere points away from its centre."""
        center = np.array([0.5, 0.0, 0.0])
        body = _make_body(0, center=tuple(center), radius=0.25)
        probe = np.array([1.0, 0.0, 0.0])
        normal = compute_contact_normal(body, probe)
        expected = (probe - center) / np.linalg.norm(probe - center)
        assert np.allclose(normal, expected, atol=1e-10)


class TestSpawnContactChartPair:
    @staticmethod
    def _two_spheres_overlap():
        body_a = _make_body(0, center=(-0.1, 0.0, 0.0), radius=0.25)
        body_b = _make_body(1, center=(+0.1, 0.0, 0.0), radius=0.25)
        return body_a, body_b

    @staticmethod
    def _overlapping_event():
        """Construct a dummy first_contact event at the origin."""
        return ContactTopologyEvent(
            event_type="first_contact",
            load_step=5,
            beta0_before=2,
            beta0_after=1,
            location=np.array([0.0, 0.0, 0.0]),
        )

    def test_seeds_straddle_contact_location(self):
        body_a, body_b = self._two_spheres_overlap()
        event = self._overlapping_event()
        pair = spawn_contact_chart_pair(
            event, [body_a, body_b], radius=0.2,
        )
        center = np.asarray(event.location)
        d_plus = pair.seed_plus - center
        d_minus = pair.seed_minus - center
        # Symmetric about the contact location
        assert np.allclose(d_plus, -d_minus, atol=1e-12)
        # Correct magnitude (0.5 * radius)
        assert abs(np.linalg.norm(d_plus) - 0.1) < 1e-12

    def test_frames_orthonormal_right_handed(self):
        body_a, body_b = self._two_spheres_overlap()
        pair = spawn_contact_chart_pair(
            self._overlapping_event(), [body_a, body_b], radius=0.2,
        )
        for frame in (pair.frame_plus, pair.frame_minus):
            t1, t2, n = frame[0], frame[1], frame[2]
            # Each vector is unit length
            assert abs(np.linalg.norm(t1) - 1.0) < 1e-12
            assert abs(np.linalg.norm(t2) - 1.0) < 1e-12
            assert abs(np.linalg.norm(n) - 1.0) < 1e-12
            # Pairwise orthogonal
            assert abs(np.dot(t1, t2)) < 1e-12
            assert abs(np.dot(t1, n)) < 1e-12
            assert abs(np.dot(t2, n)) < 1e-12
            # Right-handed: t1 × t2 = n
            cross = np.cross(t1, t2)
            assert np.allclose(cross, n, atol=1e-12)

    def test_edge_type_and_activation_step(self):
        event = self._overlapping_event()
        body_a, body_b = self._two_spheres_overlap()
        pair = spawn_contact_chart_pair(
            event, [body_a, body_b], radius=0.2,
        )
        assert pair.edge_type == "contact"
        assert pair.activation_step == 5
        assert abs(pair.radius - 0.2) < 1e-12

    def test_parent_chart_picks_nearest_seed(self):
        body_a, body_b = self._two_spheres_overlap()
        event = ContactTopologyEvent(
            event_type="first_contact",
            load_step=0,
            beta0_before=2,
            beta0_after=1,
            location=np.array([0.05, 0.0, 0.0]),
        )
        # Four existing seeds.  Seed 2 is closest to (0.05, 0, 0).
        existing = np.array([
            [-1.0, 0.0, 0.0],
            [+1.0, 0.0, 0.0],
            [+0.1, 0.0, 0.0],   # nearest
            [+0.5, 0.5, 0.0],
        ])
        pair = spawn_contact_chart_pair(
            event, [body_a, body_b],
            existing_seeds=existing, radius=0.2,
        )
        assert pair.parent_chart == 2

    def test_separation_event_rejected(self):
        body_a, body_b = self._two_spheres_overlap()
        ev = ContactTopologyEvent(
            event_type="separation",
            load_step=3,
            beta0_before=1,
            beta0_after=2,
            location=None,
        )
        with pytest.raises(ValueError):
            spawn_contact_chart_pair(ev, [body_a, body_b])

    def test_invalid_body_index_rejected(self):
        event = self._overlapping_event()
        body_a, body_b = self._two_spheres_overlap()
        with pytest.raises(ValueError):
            spawn_contact_chart_pair(
                event, [body_a, body_b], normal_from_body=2,
            )

    def test_schwarz_mpm_integration(self):
        """Feed a spawned contact pair into SchwarzMPMSolver.add_charts
        and verify the solver grows cleanly."""
        from solvers.mpm.schwarz_mpm import SchwarzMPMSolver
        from common.models import ChartDecoder, MaskNet

        # Build a minimal single-chart solver
        atlas_data = {
            "seed_points": np.array([[0.0, 0.0, 0.0]]),
            "frame_t1": np.array([[1.0, 0.0, 0.0]]),
            "frame_t2": np.array([[0.0, 1.0, 0.0]]),
            "frame_n": np.array([[0.0, 0.0, 1.0]]),
            "support_radii": np.array([0.5]),
            "membership": np.ones((5, 1), dtype=np.uint8),
        }
        decoders = [ChartDecoder(width=16, depth=2).double()]
        masks = [MaskNet(width=16, depth=2).double()]
        solver = SchwarzMPMSolver(
            atlas_data=atlas_data, decoders=decoders, masks=masks,
            n_cells=4,
        )
        assert solver.n_charts == 1

        # Build bodies and spawn a contact chart pair
        body_a, body_b = self._two_spheres_overlap()
        pair = spawn_contact_chart_pair(
            self._overlapping_event(), [body_a, body_b],
            existing_seeds=np.array([[0.0, 0.0, 0.0]]),
            radius=0.2,
        )
        added = solver.add_charts([pair])
        assert added == 2
        assert solver.n_charts == 3
        assert len(solver.solvers) == 3


# ── Dtype/device regression tests for the audit fixes ────────────────


class _Float32SphereSDF(torch.nn.Module):
    """Sphere SDF whose internal state is float32 (simulates an SDF
    trained on GPU/MPS).  Forward pass casts input to match.
    """

    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float32),
        )
        self.radius = float(radius)
        # Add a trainable parameter so _infer_sdf_device_dtype picks
        # up float32.
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(dtype=torch.float32)
        return self.scale * (
            (x32 - self.center.unsqueeze(0)).norm(dim=1) - self.radius
        )


class TestDtypeDevice:
    def test_combined_sdf_grid_float32_body(self):
        """A float32 SDF must not crash combined_sdf_grid."""
        body = ContactBody(
            body_id=0,
            sdf_net=_Float32SphereSDF(
                center=(0.0, 0.0, 0.0), radius=0.3
            ),
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.3], dtype=torch.float64),
        )
        grid = combined_sdf_grid(
            [body],
            bbox_min=np.array([-0.6, -0.6, -0.6]),
            bbox_max=np.array([+0.6, +0.6, +0.6]),
            resolution=8,
        )
        assert grid.shape == (8, 8, 8)
        assert grid.min() < -0.1
        assert grid.max() > 0.3

    def test_combined_sdf_grid_mixed_dtype_bodies(self):
        """Mixing float32 and float64 SDFs should still work."""
        body_a = ContactBody(
            body_id=0,
            sdf_net=_Float32SphereSDF(
                center=(-0.3, 0.0, 0.0), radius=0.2,
            ),
            seeds=torch.tensor([[-0.3, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.2], dtype=torch.float64),
        )
        body_b = _make_body(
            1, center=(+0.3, 0.0, 0.0), radius=0.2,
        )
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-0.8, -0.4, -0.4]),
            bbox_max=np.array([+0.8, +0.4, +0.4]),
            resolution=12,
        )
        betti = monitor.current_betti()
        assert betti[0] == 2

    def test_compute_contact_normal_float32_body(self):
        """compute_contact_normal must not crash on a float32 SDF."""
        body = ContactBody(
            body_id=0,
            sdf_net=_Float32SphereSDF(
                center=(0.0, 0.0, 0.0), radius=0.25,
            ),
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.25], dtype=torch.float64),
        )
        # Point outside along +x — normal should be radial
        normal = compute_contact_normal(body, np.array([0.5, 0.0, 0.0]))
        assert normal.dtype == np.float64
        assert normal.shape == (3,)
        assert np.allclose(normal, [1.0, 0.0, 0.0], atol=1e-5)


# ── SchwarzMPMSolver.observe_contact_topology integration tests ───────


class TestObserveContactTopology:
    @staticmethod
    def _make_solver_and_monitor(body_a_center, body_b_center):
        """Build a minimal single-chart SchwarzMPMSolver plus a
        ``ContactTopologyMonitor`` watching two movable ball SDFs.
        """
        from solvers.mpm.schwarz_mpm import SchwarzMPMSolver
        from common.models import ChartDecoder, MaskNet

        atlas_data = {
            "seed_points": np.array([[0.0, 0.0, 0.0]]),
            "frame_t1": np.array([[1.0, 0.0, 0.0]]),
            "frame_t2": np.array([[0.0, 1.0, 0.0]]),
            "frame_n": np.array([[0.0, 0.0, 1.0]]),
            "support_radii": np.array([0.5]),
            "membership": np.ones((5, 1), dtype=np.uint8),
        }
        decoders = [ChartDecoder(width=16, depth=2).double()]
        masks = [MaskNet(width=16, depth=2).double()]
        solver = SchwarzMPMSolver(
            atlas_data=atlas_data, decoders=decoders, masks=masks,
            n_cells=4, bc_type="free",
        )

        body_a = _make_body(0, center=body_a_center, radius=0.25)
        body_b = _make_body(1, center=body_b_center, radius=0.25)
        monitor = ContactTopologyMonitor(
            bodies=[body_a, body_b],
            bbox_min=np.array([-1.2, -0.6, -0.6]),
            bbox_max=np.array([+1.2, +0.6, +0.6]),
            resolution=16,
        )
        return solver, monitor, body_a, body_b

    def test_interval_validation(self):
        solver, monitor, _, _ = self._make_solver_and_monitor(
            (-0.6, 0.0, 0.0), (+0.6, 0.0, 0.0),
        )
        with pytest.raises(ValueError):
            solver.observe_contact_topology(monitor, check_interval=0)

    def test_stepping_without_observer_still_works(self):
        """Nothing should break when no monitor is registered."""
        solver, _, _, _ = self._make_solver_and_monitor(
            (-0.6, 0.0, 0.0), (+0.6, 0.0, 0.0),
        )
        # Initialize minimal particles so step() is non-trivial
        solver.initialize_particles(n_per_axis=2, density=1000.0)
        diag = solver.step(1e-4)
        # The diagnostics dict should have the new key but with an
        # empty event list (because no monitor is registered).
        assert "contact_topology_events" in diag
        assert diag["contact_topology_events"] == []

    def test_stationary_bodies_emit_baseline_only(self):
        """First tick establishes the baseline and returns no events;
        subsequent ticks on a stationary configuration also return no
        events.
        """
        solver, monitor, _, _ = self._make_solver_and_monitor(
            (-0.6, 0.0, 0.0), (+0.6, 0.0, 0.0),
        )
        solver.initialize_particles(n_per_axis=2, density=1000.0)
        solver.observe_contact_topology(monitor, check_interval=1)

        diag1 = solver.step(1e-4)
        diag2 = solver.step(1e-4)
        # First step: baseline only → empty event list
        assert diag1["contact_topology_events"] == []
        assert diag2["contact_topology_events"] == []
        # Monitor's cached prev_beta0 should be 2 after the baseline
        assert monitor.prev_beta0 == 2

    def test_first_contact_event_propagated_through_step(self):
        """Move the two bodies into overlap between successive ``step``
        calls.  The second step should propagate a ``first_contact``
        event through ``diag['contact_topology_events']`` and also
        append it to ``solver._contact_topology_events``.
        """
        solver, monitor, body_a, body_b = self._make_solver_and_monitor(
            (-0.6, 0.0, 0.0), (+0.6, 0.0, 0.0),
        )
        solver.initialize_particles(n_per_axis=2, density=1000.0)
        solver.observe_contact_topology(monitor, check_interval=1)

        # Step 1: baseline with bodies separated
        diag1 = solver.step(1e-4)
        assert diag1["contact_topology_events"] == []

        # Move the bodies into overlap (by mutating the SDF centres)
        body_a.sdf_net.set_center((-0.1, 0.0, 0.0))
        body_b.sdf_net.set_center((+0.1, 0.0, 0.0))

        # Step 2: should report the first_contact event
        diag2 = solver.step(1e-4)
        events = diag2["contact_topology_events"]
        assert len(events) == 1
        assert events[0].event_type == "first_contact"
        assert events[0].beta0_before == 2
        assert events[0].beta0_after == 1

        # And the solver has accumulated the event into its own list
        assert len(solver._contact_topology_events) == 1
        assert (
            solver._contact_topology_events[0].event_type
            == "first_contact"
        )

    def test_check_interval_amortises_monitor_calls(self):
        """With ``check_interval=3``, the monitor should only tick on
        steps where ``step_count % 3 == 0``.  Over 5 steps, that's
        steps 3 and 0... actually, we tick post-step so 3.
        """
        solver, monitor, _, _ = self._make_solver_and_monitor(
            (-0.6, 0.0, 0.0), (+0.6, 0.0, 0.0),
        )
        solver.initialize_particles(n_per_axis=2, density=1000.0)
        solver.observe_contact_topology(monitor, check_interval=3)

        # Count how many times the monitor had a baseline established
        # (equivalently, non-None prev_beta0 indicates at least one tick)
        assert monitor.prev_beta0 is None
        solver.step(1e-4)   # step_count → 1 → 1 % 3 != 0 → no tick
        assert monitor.prev_beta0 is None
        solver.step(1e-4)   # step_count → 2 → 2 % 3 != 0 → no tick
        assert monitor.prev_beta0 is None
        solver.step(1e-4)   # step_count → 3 → 3 % 3 == 0 → tick
        assert monitor.prev_beta0 == 2
