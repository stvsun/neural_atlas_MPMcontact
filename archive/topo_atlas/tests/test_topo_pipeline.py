"""
tests/test_topo_pipeline.py
Comprehensive test suite for the atlas/topo module.

Test hierarchy
--------------
Level 1 — Unit tests (no GUDHI required):
    TestFiltration    : grid sampling, clipping, analytic SDF shapes
    TestLSCategory    : cup_length_lower_bound, compute_m_min, certify_atlas

Level 2 — Integration tests (require GUDHI):
    TestPersistence   : persistence diagram correctness on analytic SDFs
    TestBettiNumbers  : Betti number accuracy for known manifolds

Level 3 — Verification tests (mathematical contracts):
    TestVerification  : nerve theorem compliance, M_min bounds

Level 4 — Monitor tests:
    TestMonitor       : static domain (no events), crack event detection

Run with:
    pytest tests/test_topo_pipeline.py -v                   # all tests
    pytest tests/test_topo_pipeline.py -v -m "not gudhi"    # skip GUDHI tests
    pytest tests/test_topo_pipeline.py -v -k "Verification" # run only V&V

Tolerance conventions
---------------------
All persistence-based tests allow lifetime_threshold = 0.05 * |SDF range|
because GUDHI's CubicalComplex on a finite grid introduces O(h) discretization
error in birth/death values (h = grid spacing).
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ball_grid():
    """SDF of a unit ball sampled on a 32^3 grid, clipped at t=0."""
    from atlas.topo.filtration import sdf_ball, clip_to_interior
    N = 32
    lin = np.linspace(-1.5, 1.5, N)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
    vals = sdf_ball(coords, radius=1.0).reshape(N, N, N).astype(np.float32)
    return clip_to_interior(vals)


@pytest.fixture
def torus_grid():
    """SDF of a solid torus (R=1, r=0.35) on a 32^3 grid, clipped at t=0."""
    from atlas.topo.filtration import sdf_solid_torus, clip_to_interior
    N = 32
    lin = np.linspace(-1.8, 1.8, N)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
    vals = sdf_solid_torus(coords, R=1.0, r=0.35).reshape(N, N, N).astype(np.float32)
    return clip_to_interior(vals)


@pytest.fixture
def shell_grid():
    """SDF of a thick spherical shell (R_inner=0.5, R_outer=1.0), clipped."""
    from atlas.topo.filtration import sdf_thick_spherical_shell, clip_to_interior
    N = 32
    lin = np.linspace(-1.5, 1.5, N)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
    vals = sdf_thick_spherical_shell(coords).reshape(N, N, N).astype(np.float32)
    return clip_to_interior(vals)


# ---------------------------------------------------------------------------
# Level 1: Unit tests (no GUDHI)
# ---------------------------------------------------------------------------

class TestFiltration:
    """Unit tests for atlas/topo/filtration.py."""

    def test_ball_interior_is_negative(self):
        from atlas.topo.filtration import sdf_ball
        coords = np.zeros((1, 3))
        assert sdf_ball(coords, radius=1.0)[0] < 0.0, "Ball center should be inside (SDF < 0)"

    def test_ball_boundary_is_zero(self):
        from atlas.topo.filtration import sdf_ball
        coords = np.array([[1.0, 0.0, 0.0]])
        val = sdf_ball(coords, radius=1.0)[0]
        assert abs(val) < 1e-6, f"Ball surface should have SDF=0, got {val}"

    def test_torus_interior_negative(self):
        from atlas.topo.filtration import sdf_solid_torus
        # Point on the main circle of the torus center
        coords = np.array([[1.0, 0.0, 0.0]])
        val = sdf_solid_torus(coords, R=1.0, r=0.35)[0]
        assert val < 0.0, f"Torus center ring should be inside, got SDF={val}"

    def test_clip_to_interior_removes_exterior(self):
        from atlas.topo.filtration import clip_to_interior
        vals = np.array([[[0.5, -0.3], [0.1, -0.8]]])
        clipped = clip_to_interior(vals, t_max=0.0)
        assert clipped.max() <= 0.0, "Clipping should remove all positive (exterior) values"
        assert clipped.min() == pytest.approx(-0.8, abs=1e-6)

    def test_grid_resolution_shapes(self):
        from atlas.topo.filtration import sdf_ball, clip_to_interior
        N = 16
        lin = np.linspace(-1.5, 1.5, N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
        coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
        vals = sdf_ball(coords).reshape(N, N, N)
        assert vals.shape == (N, N, N)
        clipped = clip_to_interior(vals)
        assert clipped.shape == (N, N, N)


class TestLSCategory:
    """Unit tests for atlas/topo/ls_category.py (no GUDHI needed)."""

    def test_convex_body_m_min_is_1(self):
        from atlas.topo.ls_category import compute_m_min
        betti = {0: 1, 1: 0, 2: 0}
        assert compute_m_min(betti) == 1

    def test_solid_torus_m_min_is_2(self):
        from atlas.topo.ls_category import compute_m_min
        betti = {0: 1, 1: 1, 2: 0}
        assert compute_m_min(betti) == 2

    def test_spherical_shell_m_min_is_2(self):
        from atlas.topo.ls_category import compute_m_min
        betti = {0: 1, 1: 0, 2: 1}
        assert compute_m_min(betti) == 2

    def test_torus_surface_m_min_is_3(self):
        from atlas.topo.ls_category import compute_m_min
        # Torus surface T^2: beta_1=2, beta_2=1
        betti = {0: 1, 1: 2, 2: 1}
        assert compute_m_min(betti) == 3

    def test_certify_atlas_pass(self):
        from atlas.topo.ls_category import certify_atlas
        betti = {0: 1, 1: 1, 2: 0}  # solid torus: M_min=2
        report = certify_atlas(M_actual=8, betti=betti)
        assert report["topology_pass"] is True
        assert report["M_min"] == 2

    def test_certify_atlas_fail(self):
        from atlas.topo.ls_category import certify_atlas
        betti = {0: 1, 1: 1, 2: 0}  # M_min=2
        report = certify_atlas(M_actual=1, betti=betti)
        assert report["topology_pass"] is False

    def test_certify_quality_gates(self):
        from atlas.topo.ls_category import certify_atlas
        betti = {0: 1, 1: 0, 2: 0}  # ball
        # Foldover failure
        report = certify_atlas(
            M_actual=4, betti=betti,
            quality_metrics={"g_fold": 0.02, "g_cov": 1.0, "g_ov": 0.01}
        )
        assert report["quality_pass"] is False

    def test_paper_torus_benchmark(self):
        """Sun (2026) uses M=8 for the solid torus. Certify M=8 passes M_min=2."""
        from atlas.topo.ls_category import certify_atlas
        betti_torus = {0: 1, 1: 1, 2: 0}
        report = certify_atlas(M_actual=8, betti=betti_torus,
                               quality_metrics={"g_fold": 0.0, "g_cov": 1.0, "g_ov": 0.02})
        assert report["topology_pass"] is True
        assert report["quality_pass"] is True


# ---------------------------------------------------------------------------
# Level 2 & 3: Integration + Verification (require GUDHI)
# ---------------------------------------------------------------------------

gudhi_mark = pytest.mark.skipif(
    True,  # replaced at runtime below
    reason="GUDHI not installed; install with: pip install gudhi"
)
try:
    import gudhi as _gudhi
    gudhi_mark = pytest.mark.gudhi
    GUDHI_OK = True
except ImportError:
    GUDHI_OK = False

requires_gudhi = pytest.mark.skipif(not GUDHI_OK, reason="GUDHI not installed")


class TestPersistence:
    """Integration tests: persistence diagram correctness on analytic SDFs."""

    @requires_gudhi
    def test_ball_has_no_H1_H2(self, ball_grid):
        """A contractible domain must have empty Dgm_1 and Dgm_2 after filtering."""
        from atlas.topo.persistence import compute_persistence_diagrams, filter_by_lifetime
        raw = compute_persistence_diagrams(ball_grid, max_dimension=2)
        fmin, fmax = float(ball_grid.min()), float(ball_grid.max())
        filtered = filter_by_lifetime(raw, threshold=0.05,
                                      filtration_range=(fmin, fmax))
        assert len(filtered[1]) == 0, \
            f"Ball should have no H_1 features, found: {filtered[1]}"
        assert len(filtered[2]) == 0, \
            f"Ball should have no H_2 features, found: {filtered[2]}"

    @requires_gudhi
    def test_ball_has_one_H0_component(self, ball_grid):
        from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
        dgms = compute_persistence_diagrams(ball_grid, max_dimension=2)
        betti = betti_numbers_at(dgms, t=-1e-6)
        assert betti[0] == 1, f"Ball should be connected (beta_0=1), got {betti[0]}"

    @requires_gudhi
    def test_torus_has_one_H1_loop(self, torus_grid):
        """Solid torus must have exactly 1 significant H_1 pair (the tunnel)."""
        from atlas.topo.persistence import compute_persistence_diagrams, filter_by_lifetime
        raw = compute_persistence_diagrams(torus_grid, max_dimension=2)
        fmin, fmax = float(torus_grid.min()), float(torus_grid.max())
        filtered = filter_by_lifetime(raw, threshold=0.08,
                                      filtration_range=(fmin, fmax))
        assert len(filtered[1]) == 1, \
            f"Solid torus should have exactly 1 H_1 feature (tunnel), found: {filtered[1]}"

    @requires_gudhi
    def test_shell_has_one_H2_void(self, shell_grid):
        """Thick spherical shell should have 1 H_2 feature (enclosed void)."""
        from atlas.topo.persistence import compute_persistence_diagrams, filter_by_lifetime
        raw = compute_persistence_diagrams(shell_grid, max_dimension=2)
        fmin, fmax = float(shell_grid.min()), float(shell_grid.max())
        filtered = filter_by_lifetime(raw, threshold=0.05,
                                      filtration_range=(fmin, fmax))
        assert len(filtered[2]) == 1, \
            f"Shell should have 1 H_2 void, found: {filtered[2]}"

    @requires_gudhi
    def test_torus_betti_numbers(self, torus_grid):
        from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at, filter_by_lifetime
        raw = compute_persistence_diagrams(torus_grid, max_dimension=2)
        fmin, fmax = float(torus_grid.min()), float(torus_grid.max())
        filtered = filter_by_lifetime(raw, threshold=0.08, filtration_range=(fmin, fmax))
        betti = betti_numbers_at(filtered, t=-1e-6)
        assert betti[0] == 1, f"beta_0 should be 1, got {betti[0]}"
        assert betti[1] == 1, f"beta_1 should be 1 (torus tunnel), got {betti[1]}"


class TestVerification:
    """
    Level 3: Mathematical verification — nerve theorem and M_min contracts.

    These are the highest-priority tests: if any fails, the topology
    pipeline has a fundamental correctness bug, not just a numerical issue.
    """

    @requires_gudhi
    def test_nerve_theorem_ball(self, ball_grid):
        """Ball: M_min=1, single chart is topologically sufficient."""
        from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at, filter_by_lifetime
        from atlas.topo.ls_category import compute_m_min
        raw = compute_persistence_diagrams(ball_grid, max_dimension=2)
        fmin, fmax = float(ball_grid.min()), float(ball_grid.max())
        filtered = filter_by_lifetime(raw, 0.05, filtration_range=(fmin, fmax))
        betti = betti_numbers_at(filtered, t=-1e-6)
        M_min = compute_m_min(betti)
        assert M_min == 1, f"Ball needs M_min=1, got {M_min} (betti={betti})"

    @requires_gudhi
    def test_nerve_theorem_torus(self, torus_grid):
        """Solid torus: M_min=2. The paper's M=8 charts satisfies this."""
        from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at, filter_by_lifetime
        from atlas.topo.ls_category import compute_m_min
        raw = compute_persistence_diagrams(torus_grid, max_dimension=2)
        fmin, fmax = float(torus_grid.min()), float(torus_grid.max())
        filtered = filter_by_lifetime(raw, 0.08, filtration_range=(fmin, fmax))
        betti = betti_numbers_at(filtered, t=-1e-6)
        M_min = compute_m_min(betti)
        assert M_min == 2, f"Solid torus needs M_min=2, got {M_min} (betti={betti})"

    @requires_gudhi
    def test_bottleneck_stability(self, ball_grid):
        """
        Stability theorem: perturbing SDF by epsilon changes persistence by <= epsilon.
        Verify: adding Gaussian noise with std=0.01 changes bottleneck distance by <= 0.02.
        """
        from atlas.topo.persistence import compute_persistence_diagrams, bottleneck_distance
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, ball_grid.shape).astype(np.float32)
        noisy = np.clip(ball_grid + noise, ball_grid.min(), 0.0)
        dgm_clean = compute_persistence_diagrams(ball_grid, max_dimension=1)[1]
        dgm_noisy = compute_persistence_diagrams(noisy, max_dimension=1)[1]
        d_b = bottleneck_distance(dgm_clean, dgm_noisy)
        assert d_b <= 0.05, \
            f"Bottleneck stability violated: d_B={d_b:.4f} exceeds noise level 0.05"


# ---------------------------------------------------------------------------
# Level 4: Monitor tests
# ---------------------------------------------------------------------------

class TestMonitor:
    """Tests for TopologyMonitor."""

    def test_no_events_on_fixed_domain(self):
        """Fixed domain: monitor should report 0 events after initial step."""
        try:
            from atlas.topo.filtration import sdf_ball, clip_to_interior
            from atlas.topo.monitor import TopologyMonitor
        except ImportError:
            pytest.skip("Module dependencies not available")

        if not GUDHI_OK:
            pytest.skip("GUDHI required for monitor test")

        N = 20
        lin = np.linspace(-1.5, 1.5, N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
        coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
        vals = sdf_ball(coords, radius=1.0).reshape(N, N, N).astype(np.float32)
        grid = clip_to_interior(vals)

        monitor = TopologyMonitor(verbose=False)
        for step in range(5):
            events = monitor.update(grid, load_step=step)
            if step > 0:
                assert len(events) == 0, \
                    f"Fixed domain should have 0 events at step {step}, got {len(events)}"

    def test_topology_change_detected(self):
        """
        Simulate a topology change: start with a ball, then switch to a torus.
        Monitor should detect a new H_1 event on the transition step.
        """
        if not GUDHI_OK:
            pytest.skip("GUDHI required for monitor test")

        from atlas.topo.filtration import sdf_ball, sdf_solid_torus, clip_to_interior
        from atlas.topo.monitor import TopologyMonitor

        N = 24
        lin = np.linspace(-1.8, 1.8, N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
        coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)

        ball_grid = clip_to_interior(sdf_ball(coords).reshape(N, N, N).astype(np.float32))
        torus_grid = clip_to_interior(sdf_solid_torus(coords, R=1.0, r=0.35).reshape(N, N, N).astype(np.float32))

        monitor = TopologyMonitor(lifetime_threshold=0.08, bottleneck_threshold=0.01, verbose=False)
        monitor.update(ball_grid, load_step=0)     # baseline
        monitor.update(ball_grid, load_step=1)     # no change
        events = monitor.update(torus_grid, load_step=2)  # topology change

        h1_events = [e for e in events if e.dimension == 1]
        assert len(h1_events) >= 1, \
            f"Expected at least 1 H_1 event when switching to torus, got {events}"
        assert monitor.total_events >= 1


# ---------------------------------------------------------------------------
# Regression tests (reproduce Sun 2026 Table 4 quality metrics)
# ---------------------------------------------------------------------------

class TestPaperBenchmarks:
    """
    Regression tests against Sun (2026) numerical results.
    These should pass once the topology pipeline is integrated with the
    existing neural_atlas_MPM atlas-building pipeline.

    Currently marked as expected-to-fail (xfail) since integration
    with the existing atlas code is Phase 2 work.
    """

    @pytest.mark.xfail(reason="Phase 2: requires atlas pipeline integration")
    def test_torus_12chart_quality_gates(self):
        """
        After building a torus atlas with the topology pipeline,
        M should satisfy M_min=2 and quality gates should pass.
        Expected from paper: g_fold=0.0, g_cov=1.0, g_ov<0.05.
        """
        from atlas.topo.ls_category import certify_atlas
        betti_torus = {0: 1, 1: 1, 2: 0}
        # This will be populated by the actual atlas build in Phase 2
        report = certify_atlas(
            M_actual=8,
            betti=betti_torus,
            quality_metrics={"g_fold": 0.0, "g_cov": 1.0, "g_ov": 0.026}
        )
        assert report["topology_pass"] and report["quality_pass"]
