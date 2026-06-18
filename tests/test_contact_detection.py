"""Tests for SDF gap evaluation and contact detection infrastructure.

Verifies:
    1. Gap accuracy against analytic sphere SDF
    2. Contact normal accuracy (radial direction for sphere)
    3. Broad-phase culling (well-separated vs overlapping bodies)
    4. Narrow-phase detection returns correct active set
"""

import math

import torch
import pytest

from solvers.contact.gap import evaluate_gap
from solvers.contact.contact_pair import ContactBody, ContactPair
from solvers.contact.contact_manager import ContactManager


# ── Analytic SDF helpers ─────────────────────────────────────────────

class SphereSDF(torch.nn.Module):
    """Analytic sphere SDF: phi(x) = |x - c| - r."""

    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float64)
        )
        self.radius = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x - self.center.unsqueeze(0)
        return d.norm(dim=1) - self.radius


class FloorSDF(torch.nn.Module):
    """Analytic half-space SDF: phi(x) = x[:,1] - y_floor."""

    def __init__(self, y_floor=0.0):
        super().__init__()
        self.y_floor = y_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1] - self.y_floor


# ── Gap evaluation tests ─────────────────────────────────────────────

class TestEvaluateGap:
    def test_gap_analytic_sphere(self):
        """Gap values should match closed-form sphere SDF."""
        sdf = SphereSDF(center=(0, 0, 0), radius=1.0)
        # Points at known distances
        pts = torch.tensor(
            [
                [2.0, 0.0, 0.0],   # outside: gap = 1.0
                [1.0, 0.0, 0.0],   # on surface: gap = 0.0
                [0.5, 0.0, 0.0],   # inside: gap = -0.5
            ],
            dtype=torch.float64,
        )
        gap, normal = evaluate_gap(pts, sdf)

        assert abs(gap[0].item() - 1.0) < 1e-10
        assert abs(gap[1].item() - 0.0) < 1e-10
        assert abs(gap[2].item() + 0.5) < 1e-10

    def test_gap_normal_accuracy_sphere(self):
        """Normals should point radially outward for a sphere SDF."""
        sdf = SphereSDF(center=(1.0, 2.0, 3.0), radius=0.5)
        center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Random points around the sphere
        torch.manual_seed(42)
        dirs = torch.randn(20, 3, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        pts = center.unsqueeze(0) + 0.8 * dirs  # outside sphere

        gap, normal = evaluate_gap(pts, sdf)

        # Expected normal = (x - center) / |x - center|
        expected = (pts - center.unsqueeze(0))
        expected = expected / expected.norm(dim=1, keepdim=True)

        cos_angle = (normal * expected).sum(dim=1)
        # Angle error < 1 degree
        assert cos_angle.min().item() > math.cos(math.radians(1.0))

    def test_gap_floor_sdf(self):
        """Half-space SDF should give constant normal (0, 1, 0)."""
        sdf = FloorSDF(y_floor=0.0)
        pts = torch.tensor(
            [
                [0.0, 0.5, 0.0],   # above: gap = 0.5
                [0.0, -0.1, 0.0],  # below: gap = -0.1
                [3.0, 0.0, -2.0],  # on surface: gap = 0.0
            ],
            dtype=torch.float64,
        )
        gap, normal = evaluate_gap(pts, sdf)

        assert abs(gap[0].item() - 0.5) < 1e-10
        assert abs(gap[1].item() + 0.1) < 1e-10
        assert abs(gap[2].item()) < 1e-10

        # Normal should be (0, 1, 0) everywhere
        expected_n = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        for i in range(3):
            assert torch.allclose(normal[i], expected_n, atol=1e-10)

    def test_gap_empty_input(self):
        """Empty candidate tensor should return empty results."""
        sdf = SphereSDF()
        pts = torch.zeros(0, 3, dtype=torch.float64)
        gap, normal = evaluate_gap(pts, sdf)
        assert gap.shape == (0,)
        assert normal.shape == (0, 3)

    def test_gap_handles_2d_sdf_output(self):
        """SDFs that return (N, 1) must be flattened to (N,) — this
        guards against a silent shape-propagation bug where downstream
        broadcasting turns the gap into a 2D matrix.
        """
        class TwoDSphereSDF(torch.nn.Module):
            """Like SphereSDF but returns shape (N, 1) instead of (N,)."""
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d = x.norm(dim=1, keepdim=True) - 1.0
                assert d.shape == (x.shape[0], 1)
                return d  # (N, 1)

        sdf = TwoDSphereSDF()
        pts = torch.tensor(
            [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64,
        )
        gap, normal = evaluate_gap(pts, sdf)
        # Output must be exactly (N,) and (N, 3)
        assert gap.shape == (2,)
        assert normal.shape == (2, 3)
        assert abs(gap[0].item() - 1.0) < 1e-10
        assert abs(gap[1].item() + 0.5) < 1e-10

    def test_gap_inside_no_grad_block(self):
        """``evaluate_gap`` must work even when the caller is inside a
        ``torch.no_grad()`` block — the function forces grad-recording
        internally.
        """
        sdf = SphereSDF(center=(0, 0, 0), radius=1.0)
        pts = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
        with torch.no_grad():
            gap, normal = evaluate_gap(pts, sdf)
        assert abs(gap[0].item() + 0.5) < 1e-10
        # Normal should be radial (+x for a point at (0.5, 0, 0))
        assert abs(normal[0, 0].item() - 1.0) < 1e-10

    def test_gap_float32_input(self):
        """float32 inputs should work without dtype errors."""
        sdf = SphereSDF(center=(0, 0, 0), radius=1.0)
        # The SDF's ``center`` buffer is float64, but x_candidates is
        # float32 — evaluate_gap has to tolerate the mismatch or the
        # SDF forward pass will crash.  For this analytic SDF, torch
        # will upcast automatically.
        pts = torch.tensor(
            [[2.0, 0.0, 0.0]], dtype=torch.float32,
        )
        gap, normal = evaluate_gap(pts, sdf)
        assert gap.shape == (1,)
        assert normal.shape == (1, 3)


# ── Broad-phase tests ────────────────────────────────────────────────

class TestBroadPhase:
    def test_culling_well_separated(self):
        """Well-separated bodies should produce no broad-phase pairs."""
        body_a = ContactBody(
            body_id=0,
            sdf_net=SphereSDF(center=(0, 0, 0)),
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.5], dtype=torch.float64),
        )
        body_b = ContactBody(
            body_id=1,
            sdf_net=SphereSDF(center=(10, 0, 0)),
            seeds=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.5], dtype=torch.float64),
        )
        mgr = ContactManager([body_a, body_b], margin=0.1)
        pairs = mgr.broad_phase(body_a, body_b)
        assert len(pairs) == 0

    def test_contact_overlapping(self):
        """Overlapping bodies should return chart pairs."""
        body_a = ContactBody(
            body_id=0,
            sdf_net=SphereSDF(center=(0, 0, 0)),
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([1.0], dtype=torch.float64),
        )
        body_b = ContactBody(
            body_id=1,
            sdf_net=SphereSDF(center=(1.5, 0, 0)),
            seeds=torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([1.0], dtype=torch.float64),
        )
        mgr = ContactManager([body_a, body_b], margin=0.1)
        pairs = mgr.broad_phase(body_a, body_b)
        assert len(pairs) == 1
        assert pairs[0] == (0, 0)

    def test_multi_chart_broad_phase(self):
        """Multi-chart body should produce multiple pairs when close."""
        body_a = ContactBody(
            body_id=0,
            sdf_net=SphereSDF(),
            seeds=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64
            ),
            support_radii=torch.tensor([0.5, 0.5], dtype=torch.float64),
        )
        body_b = ContactBody(
            body_id=1,
            sdf_net=SphereSDF(),
            seeds=torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([0.5], dtype=torch.float64),
        )
        mgr = ContactManager([body_a, body_b], margin=0.1)
        pairs = mgr.broad_phase(body_a, body_b)
        # Both charts of A should pair with B's single chart
        assert len(pairs) == 2

    def test_broad_phase_empty_bodies(self):
        """Empty seeds should return an empty list (no crash)."""
        body_a = ContactBody(
            body_id=0,
            sdf_net=SphereSDF(),
            seeds=torch.zeros(0, 3, dtype=torch.float64),
            support_radii=torch.zeros(0, dtype=torch.float64),
        )
        body_b = ContactBody(
            body_id=1,
            sdf_net=SphereSDF(),
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([1.0], dtype=torch.float64),
        )
        mgr = ContactManager([body_a, body_b], margin=0.1)
        assert mgr.broad_phase(body_a, body_b) == []
        assert mgr.broad_phase(body_b, body_a) == []

    def test_vectorized_broad_phase_matches_reference(self):
        """Parity check: the vectorized implementation must give the
        same pairs as a naive Python-loop reference, for a randomized
        multi-chart configuration.
        """
        torch.manual_seed(42)
        margin = 0.05

        def reference_broad_phase(body_A, body_B, margin):
            pairs = []
            for i in range(body_A.seeds.shape[0]):
                for j in range(body_B.seeds.shape[0]):
                    d = (body_A.seeds[i] - body_B.seeds[j]).norm().item()
                    r = (
                        body_A.support_radii[i]
                        + body_B.support_radii[j]
                    ).item()
                    if d < r + margin:
                        pairs.append((i, j))
            return pairs

        for trial in range(5):
            M_A = int(torch.randint(1, 15, (1,)).item())
            M_B = int(torch.randint(1, 15, (1,)).item())
            seeds_A = torch.randn(M_A, 3, dtype=torch.float64)
            seeds_B = torch.randn(M_B, 3, dtype=torch.float64)
            radii_A = 0.2 + 0.5 * torch.rand(M_A, dtype=torch.float64)
            radii_B = 0.2 + 0.5 * torch.rand(M_B, dtype=torch.float64)

            body_A = ContactBody(
                body_id=0,
                sdf_net=SphereSDF(),
                seeds=seeds_A,
                support_radii=radii_A,
            )
            body_B = ContactBody(
                body_id=1,
                sdf_net=SphereSDF(),
                seeds=seeds_B,
                support_radii=radii_B,
            )

            mgr = ContactManager([body_A, body_B], margin=margin)
            vec_pairs = sorted(mgr.broad_phase(body_A, body_B))
            ref_pairs = sorted(
                reference_broad_phase(body_A, body_B, margin),
            )
            assert vec_pairs == ref_pairs, (
                f"Trial {trial}: vectorized={vec_pairs} "
                f"vs reference={ref_pairs}"
            )


# ── Narrow-phase tests ───────────────────────────────────────────────

class TestNarrowPhase:
    def test_detect_penetration(self):
        """Penetrating particles should be detected."""
        sdf_a = SphereSDF(center=(0, 0, 0), radius=1.0)
        body_a = ContactBody(
            body_id=0, sdf_net=sdf_a,
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([1.5], dtype=torch.float64),
        )
        mgr = ContactManager([body_a])

        # Particles: one inside, one outside
        x_phys = torch.tensor(
            [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
        )
        pair = mgr.detect_mpm(body_a, body_B_id=1, chart_id_B=0, x_phys=x_phys)

        assert pair is not None
        assert pair.particle_indices.shape[0] == 1
        assert pair.particle_indices[0].item() == 0
        assert pair.gap[0].item() < 0

    def test_no_penetration_returns_none(self):
        """All-exterior particles should yield None."""
        sdf_a = SphereSDF(center=(0, 0, 0), radius=1.0)
        body_a = ContactBody(
            body_id=0, sdf_net=sdf_a,
            seeds=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            support_radii=torch.tensor([1.5], dtype=torch.float64),
        )
        mgr = ContactManager([body_a])

        x_phys = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64
        )
        pair = mgr.detect_mpm(body_a, body_B_id=1, chart_id_B=0, x_phys=x_phys)
        assert pair is None
