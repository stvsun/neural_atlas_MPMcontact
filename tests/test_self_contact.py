"""Tests for the self-contact manager.

Covers construction (surface vs bulk classification), detection
(home-position vs folded), force computation (zeroed out for inactive
particles), input validation, and a folding-slab integration test.
"""

import math

import numpy as np
import torch
import pytest

from solvers.contact.contact_pair import ContactBody
from solvers.contact.self_contact import SelfContactManager


# ── Analytic SDFs ────────────────────────────────────────────────────


class SphereSDF(torch.nn.Module):
    """phi(x) = |x - c| - r."""

    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float64),
        )
        self.radius = float(radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center.unsqueeze(0)).norm(dim=1) - self.radius


class SlabSDF(torch.nn.Module):
    """Axis-aligned slab: phi(x) = max(|x - c| - half_extent).

    For simplicity here we use the L_inf variant, which gives a cubical
    slab.  ``half_extent`` is a (3,) tensor of half-lengths.
    """

    def __init__(self, center, half_extent):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float64),
        )
        self.register_buffer(
            "half_extent",
            torch.tensor(half_extent, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = (x - self.center.unsqueeze(0)).abs() - self.half_extent.unsqueeze(0)
        outside = torch.clamp(d, min=0.0).norm(dim=1)
        inside = torch.clamp(d.max(dim=1).values, max=0.0)
        return outside + inside


def _make_body(body_id, sdf):
    return ContactBody(
        body_id=body_id,
        sdf_net=sdf,
        seeds=torch.zeros(1, 3, dtype=torch.float64),
        support_radii=torch.tensor([1.0], dtype=torch.float64),
    )


# ── Construction tests ──────────────────────────────────────────────


class TestConstruction:
    def test_surface_vs_bulk_classification(self):
        """Particles near the surface are flagged; deep interior not."""
        body = _make_body(0, SphereSDF(radius=1.0))
        # Two particles on the surface, two deep in the interior
        pts = torch.tensor(
            [
                [1.0, 0.0, 0.0],    # gap = 0  → surface
                [0.0, 1.0, 0.0],    # gap = 0  → surface
                [0.0, 0.0, 0.0],    # gap = -1 → bulk
                [0.2, 0.1, 0.0],    # gap = -0.78 → bulk
            ],
            dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts, surface_tol=0.05)
        assert mgr.state.is_surface.tolist() == [True, True, False, False]
        assert abs(mgr.state.initial_gap[0].item()) < 1e-10
        assert mgr.state.initial_gap[2].item() < -0.5

    def test_bad_shape_raises(self):
        body = _make_body(0, SphereSDF(radius=1.0))
        with pytest.raises(ValueError):
            SelfContactManager(body, torch.zeros(5))   # 1-D

    def test_n_surface_particles(self):
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],   # bulk
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts, surface_tol=0.05)
        assert mgr.n_surface_particles() == 2


# ── Detection tests ─────────────────────────────────────────────────


class TestDetect:
    def test_stationary_particles_inactive(self):
        """Particles that haven't moved are NOT flagged."""
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],     # bulk, gap = -0.5
            ],
            dtype=torch.float64,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=1e-2,
        )
        gap, normal, active = mgr.detect(pts)
        # No particle has moved → no active self-contact
        assert not active.any()

    def test_bulk_particle_moved_deeper_still_inactive(self):
        """Deep interior particles must NEVER be flagged, even if they
        move deeper than where they started.
        """
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[0.5, 0.0, 0.0]],       # bulk, gap = -0.5
            dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts, surface_tol=0.05)
        # Move it deeper into the body
        moved = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        _, _, active = mgr.detect(moved)
        assert not active.any()

    def test_surface_particle_moved_deeper_flagged(self):
        """Surface particle that moves notably deeper → active."""
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[1.0, 0.0, 0.0]],       # on surface, gap = 0
            dtype=torch.float64,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.02,
        )
        # Move radially inward by 0.05 → current gap = -0.05, delta = -0.05
        moved = torch.tensor([[0.95, 0.0, 0.0]], dtype=torch.float64)
        gap, normal, active = mgr.detect(moved)
        assert active.any()
        assert gap[0].item() < -0.02

    def test_surface_particle_small_wobble_not_flagged(self):
        """A surface particle that wobbles within ``penetration_delta``
        must NOT be flagged (no false positives from numerical noise).
        """
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.02,
        )
        # Move inward by 1e-3 (much less than threshold 2e-2)
        wobble = torch.tensor(
            [[0.999, 0.0, 0.0]], dtype=torch.float64,
        )
        _, _, active = mgr.detect(wobble)
        assert not active.any()

    def test_shape_mismatch_raises(self):
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts)
        with pytest.raises(ValueError):
            mgr.detect(torch.zeros(2, 3, dtype=torch.float64))


# ── Force tests ─────────────────────────────────────────────────────


class TestComputeForce:
    def test_force_zero_for_stationary_particles(self):
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts)
        vols = torch.ones(2, dtype=torch.float64)
        f = mgr.compute_force(pts, vols, epsilon_n=1e4)
        assert torch.allclose(f, torch.zeros_like(f))

    def test_force_zero_for_bulk_particles(self):
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[0.5, 0.0, 0.0]],    # bulk
            dtype=torch.float64,
        )
        mgr = SelfContactManager(body, pts, surface_tol=0.05)
        vols = torch.ones(1, dtype=torch.float64)
        # Move bulk particle arbitrarily — still no force
        moved = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)
        f = mgr.compute_force(moved, vols, epsilon_n=1e4)
        assert torch.allclose(f, torch.zeros_like(f))

    def test_force_on_folded_surface_particle(self):
        """Surface particle that folded inward by 0.1 → nonzero force
        along the outward normal (radial for a sphere).
        """
        body = _make_body(0, SphereSDF(radius=1.0))
        pts = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.01,
        )
        vols = torch.tensor([0.001], dtype=torch.float64)
        # Move inward by 0.1
        moved = torch.tensor([[0.9, 0.0, 0.0]], dtype=torch.float64)
        f = mgr.compute_force(moved, vols, epsilon_n=1e4)
        # Force should be along +x (outward normal at that point)
        assert f[0, 0].item() > 0
        assert abs(f[0, 1].item()) < 1e-10
        assert abs(f[0, 2].item()) < 1e-10
        # Magnitude: eps_n * (0.1 - 0) * V_p  =  1e4 * 0.1 * 1e-3 = 1.0
        expected = 1e4 * 0.1 * 1e-3
        assert abs(f.norm(dim=1)[0].item() - expected) < 1e-6

    def test_force_independent_of_absolute_gap(self):
        """Force scales with the DELTA from initial, not the absolute
        current gap — otherwise bulk surface-adjacent particles with
        large absolute gap but no motion would spuriously contribute.
        """
        body = _make_body(0, SphereSDF(radius=1.0))
        # Both particles start near the surface
        pts = torch.tensor(
            [[0.99, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.01,
        )
        vols = torch.ones(2, dtype=torch.float64)
        # Both particles move inward to gap = -0.05 each (absolute)
        moved = torch.tensor(
            [[0.95, 0.0, 0.0], [0.95, 0.0, 0.0]],
            dtype=torch.float64,
        )
        f = mgr.compute_force(moved, vols, epsilon_n=1e4)
        # Delta for particle 0: (-0.05) - (-0.01) = -0.04  → pen = 0.04
        # Delta for particle 1: (-0.05) -  0     = -0.05  → pen = 0.05
        # Forces should differ by the delta ratio (0.04 vs 0.05)
        m0 = f[0].norm().item()
        m1 = f[1].norm().item()
        assert abs(m0 / 0.04 - m1 / 0.05) < 1e-8


# ── Integration test: folding slab ───────────────────────────────────


class TestFoldingSlab:
    def test_folding_detected(self):
        """Build a 1D strip of surface particles along a slab, then
        fold half of them onto the other half. The manager should
        flag the folded half as self-contacting.
        """
        # Slab: cube of half-extent 0.5
        body = _make_body(0, SlabSDF(center=(0, 0, 0), half_extent=(0.5, 0.5, 0.5)))

        # 10 particles on the top face of the slab (y = 0.5)
        n = 10
        xs = torch.linspace(-0.4, 0.4, n, dtype=torch.float64)
        pts = torch.stack(
            [xs, torch.full_like(xs, 0.5), torch.zeros_like(xs)],
            dim=1,
        )
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.02,
        )
        # All 10 should be surface particles
        assert mgr.n_surface_particles() == n

        # Fold the right half of the particles onto the left:
        # move particles at positive x to the opposite side of the slab.
        folded = pts.clone()
        folded[n // 2:, 1] = -0.4   # move to near the bottom face
        _, _, active = mgr.detect(folded)
        # The folded particles are well inside the slab (gap ≈ -0.1)
        # vs. their initial gap of 0.0, so they should be flagged.
        assert active[n // 2:].all()
        # The unfolded particles should NOT be flagged.
        assert not active[: n // 2].any()

    def test_folding_force_pushes_outward(self):
        """The self-contact force on folded particles should push them
        back toward the slab boundary they crossed.
        """
        body = _make_body(0, SlabSDF(center=(0, 0, 0), half_extent=(0.5, 0.5, 0.5)))
        pts = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64)
        mgr = SelfContactManager(
            body, pts, surface_tol=0.05, penetration_delta=0.02,
        )
        # Fold: move from top face (y=0.5) to near the bottom face
        folded = torch.tensor([[0.0, -0.45, 0.0]], dtype=torch.float64)
        vols = torch.tensor([1e-3], dtype=torch.float64)
        f = mgr.compute_force(folded, vols, epsilon_n=1e4)
        # Force on the folded particle should point toward -y (the
        # nearest slab face) since that's the local outward normal at
        # (0, -0.45, 0).
        assert f[0, 1].item() < 0, f"Expected negative y-force, got {f[0, 1]}"
        # Magnitude should be nonzero
        assert f.norm().item() > 0
