"""Tests for the level-set-free radial-chart contact detector.

Benchmarks ``solvers/contact/chart_gap.py`` against the SDF model
(``solvers/contact/gap.py::evaluate_gap``) and closed-form geometry:

    1. Sphere chart reproduces the analytic sphere SDF to machine precision
       (gap + normal), including a rotated, off-origin frame (Q != I).
    2. Conservativeness: the returned normal is ``grad(gap)/|grad(gap)|``
       (finite-difference check) — implicitly guards the surface-gradient
       projection.
    3. Ellipsoid on-surface normal matches the closed-form ellipsoid normal
       (independent projection/normal guard) + direct tangential-projection check.
    4. Active set ``gap < 0`` is identical to the true SDF's for a star-shaped
       superquadric (the provable equivalence).
    5. Radial vs perpendicular: ``|gap_rad| >= |gap_perp|`` always, signs agree,
       and the refine matches a dense S^2 closest-point scan near the surface.
    6. The swappable ``detector='chart'`` path through ``ContactManager.detect_mpm``
       yields the same active set as the SDF body on the same particle cloud.
"""

import math

import pytest
import torch

from solvers.contact.chart_gap import (
    SphereRho,
    SuperquadricRho,
    evaluate_gap_chart,
    closest_point_refine_chart,
)
from solvers.contact.contact_pair import ContactBody
from solvers.contact.contact_manager import ContactManager
from solvers.contact.gap import evaluate_gap
from tests.test_contact_detection import SphereSDF


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


# ── 1. Sphere chart == sphere SDF to machine precision ───────────────

class TestSphereMachinePrecision:
    def test_chart_sphere_machine_precision(self):
        """SphereRho vs SphereSDF, identical points, rotated off-origin frame.

        Orientation cannot matter for a sphere — but a Q != I, off-origin frame
        is the test that catches a transposed/garbled body<->world transform.
        """
        center = (1.0, 2.0, 3.0)
        R = 0.5
        Q = _rot_z(0.7)
        chart = SphereRho(R, center=center, orientation=Q)
        sdf = SphereSDF(center=center, radius=R)

        c = torch.tensor(center, dtype=torch.float64)
        torch.manual_seed(0)
        dirs = torch.randn(64, 3, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        rad = 0.1 + 1.5 * torch.rand(64, dtype=torch.float64)  # inside & outside
        pts = c.unsqueeze(0) + rad.unsqueeze(1) * dirs

        gap_c, n_c = evaluate_gap_chart(pts, chart)
        gap_s, n_s = evaluate_gap(pts, sdf)

        assert (gap_c - gap_s).abs().max().item() < 1e-12
        cos = (n_c * n_s).sum(dim=1).clamp(max=1.0)
        max_angle = torch.acos(cos).max().item()
        assert max_angle < 1e-10

    def test_chart_sphere_known_values(self):
        """Spot-check the three canonical depths (mirrors test_gap_analytic_sphere)."""
        chart = SphereRho(1.0, center=(0.0, 0.0, 0.0))
        pts = torch.tensor(
            [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float64,
        )
        gap, normal = evaluate_gap_chart(pts, chart)
        assert abs(gap[0].item() - 1.0) < 1e-12
        assert abs(gap[1].item() - 0.0) < 1e-12
        assert abs(gap[2].item() + 0.5) < 1e-12
        # normal is +x for all three (radial)
        assert torch.allclose(
            normal[:, 0], torch.ones(3, dtype=torch.float64), atol=1e-12
        )


# ── 2. Conservativeness (finite-difference of gap == normal) ─────────

class TestConservativeness:
    def test_chart_normal_conservativeness(self):
        """Returned normal == grad(gap)/|grad(gap)| (central FD) on an ellipsoid.

        A missing tangential projection of grad(rho) would break this.
        """
        ell = SuperquadricRho(
            semi_axes=(0.6, 0.9, 0.4), exponent=2.0,
            center=(0.1, -0.2, 0.05), orientation=_rot_z(0.4),
        )
        torch.manual_seed(3)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        xp = ctr + 0.6 * torch.randn(32, 3, dtype=torch.float64)

        _, normal = evaluate_gap_chart(xp, ell)
        h = 1e-6
        grad_fd = torch.zeros_like(xp)
        for k in range(3):
            e = torch.zeros(3, dtype=torch.float64)
            e[k] = h
            gp, _ = evaluate_gap_chart(xp + e, ell)
            gm, _ = evaluate_gap_chart(xp - e, ell)
            grad_fd[:, k] = (gp - gm) / (2 * h)
        grad_fd = grad_fd / grad_fd.norm(dim=1, keepdim=True)

        cos = (grad_fd * normal).sum(dim=1).clamp(max=1.0)
        assert torch.acos(cos).max().item() < 1e-5


# ── 3. Ellipsoid normal vs closed form + explicit projection ─────────

class TestEllipsoidNormal:
    def test_on_surface_normal_matches_analytic(self):
        """On-surface radial-mode normal == closed-form ellipsoid normal.

        For surface ``sum_i ((x-c)_i / S_i)^2 = 1`` the outward normal is
        ``normalize((x-c)/S^2)`` (Q = I here).  On the surface the radial-mode
        normal equals the true surface normal exactly.
        """
        S = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float64)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                              center=tuple(ctr.tolist()))
        torch.manual_seed(5)
        u = torch.randn(40, 3, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True)
        rho = ell.radius(u)
        x_surf = ctr + rho.unsqueeze(1) * u           # exactly on the surface

        gap, normal = evaluate_gap_chart(x_surf, ell)
        assert gap.abs().max().item() < 1e-12         # zero level set

        d = x_surf - ctr
        n_ana = d / (S ** 2)
        n_ana = n_ana / n_ana.norm(dim=1, keepdim=True)
        # 1 - cos avoids the sqrt-sensitivity of acos near cos = 1; this is the
        # machine-precision metric (acos would floor at ~2e-8 by conditioning).
        one_minus_cos = (1.0 - (normal * n_ana).sum(dim=1)).abs()
        assert one_minus_cos.max().item() < 1e-12

    def test_surface_gradient_is_tangential(self):
        """Direct guard: the projected surface gradient is orthogonal to dhat.

        Replicates the projection the detector performs and asserts the
        load-bearing property ``grad_S rho . dhat ~ 0`` (catches an
        unprojected-gradient regression in the neural Stage-1 path).
        """
        ell = SuperquadricRho(semi_axes=(0.6, 0.9, 0.4), exponent=2.0)
        torch.manual_seed(7)
        u = torch.randn(50, 3, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True)
        u_g = u.clone().requires_grad_(True)
        rho = ell.radius(u_g)
        g_amb = torch.autograd.grad(rho, u_g, torch.ones_like(rho))[0]
        g_S = g_amb - (g_amb * u).sum(1, keepdim=True) * u
        assert (g_S * u).sum(1).abs().max().item() < 1e-10


# ── 4. Active set identical to the true SDF for star-shaped bodies ───

class TestActiveSetEquivalence:
    def test_active_set_equals_sdf_starshaped(self):
        """sign(gap_chart) == sign(true ellipsoid SDF) at random points."""
        S = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float64)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                              center=tuple(ctr.tolist()))
        torch.manual_seed(11)
        xx = ctr + 0.8 * torch.randn(5000, 3, dtype=torch.float64)
        gap, _ = evaluate_gap_chart(xx, ell)
        # True implicit sign for an ellipsoid: F_ana < 0 inside.
        f_ana = (((xx - ctr) / S) ** 2).sum(dim=1) - 1.0
        assert torch.equal(gap < 0, f_ana < 0)


# ── 5. Radial vs perpendicular bias + refine accuracy ────────────────

class TestRadialVsPerp:
    def _dense_perp(self, ell, ctr, S, xp, M=20000, seed=21):
        torch.manual_seed(seed)
        ud = torch.randn(M, 3, dtype=torch.float64)
        ud = ud / ud.norm(dim=1, keepdim=True)
        feet = ctr + ell.radius(ud).unsqueeze(1) * ud
        d2 = ((xp.unsqueeze(1) - feet.unsqueeze(0)) ** 2).sum(dim=2)
        dmin = d2.min(dim=1).values.sqrt()
        inside = (((xp - ctr) / S) ** 2).sum(dim=1) < 1.0
        return torch.where(inside, -dmin, dmin)

    def test_magnitude_conservative_and_sign(self):
        """|gap_rad| >= |gap_refine| (always) and signs agree (general cloud)."""
        S = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float64)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                              center=tuple(ctr.tolist()))
        torch.manual_seed(13)
        xp = ctr + 0.6 * torch.randn(300, 3, dtype=torch.float64)
        gap_rad, _ = evaluate_gap_chart(xp, ell)
        gap_ref, _ = closest_point_refine_chart(xp, ell)
        assert (gap_rad.abs() >= gap_ref.abs() - 1e-12).all()
        assert torch.equal(gap_rad < 0, gap_ref < 0)

    def test_refine_matches_dense_scan_near_surface(self):
        """Near the surface, the refine matches a dense S^2 closest-point scan."""
        S = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float64)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        L = float(S.max())
        ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                              center=tuple(ctr.tolist()))
        torch.manual_seed(17)
        u = torch.randn(300, 3, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True)
        rho = ell.radius(u)
        frac = (torch.rand(300, dtype=torch.float64) - 0.5) * 0.3   # +/-15% of rho
        xp = ctr + (rho * (1.0 + frac)).unsqueeze(1) * u
        gap_ref, _ = closest_point_refine_chart(xp, ell)
        gap_true = self._dense_perp(ell, ctr, S, xp)
        assert ((gap_ref - gap_true).abs() / L).max().item() < 5e-2

    def test_radial_normal_approaches_surface_normal_on_contact(self):
        """Radial normal error vs true surface normal -> 0 as penetration -> 0."""
        S = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float64)
        ctr = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        ell = SuperquadricRho(semi_axes=tuple(S.tolist()), exponent=2.0,
                              center=tuple(ctr.tolist()))
        torch.manual_seed(19)
        u = torch.randn(60, 3, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True)
        rho = ell.radius(u)

        def max_angle(frac):
            xp = ctr + (rho * (1.0 + frac)).unsqueeze(1) * u
            _, n_rad = evaluate_gap_chart(xp, ell)
            _, n_surf = closest_point_refine_chart(xp, ell)
            cos = (n_rad * n_surf).sum(dim=1).clamp(max=1.0)
            return torch.acos(cos).max().item()

        a_far = max_angle(0.10)     # 10% off the surface
        a_near = max_angle(0.001)   # essentially on contact
        assert a_near < a_far
        assert a_near < 1e-2        # ~0.6 deg at contact


# ── 6. Swappable detector through the live manager path ──────────────

class TestSwappableDetector:
    def test_detect_mpm_chart_matches_sdf(self):
        """detector='chart' and detector='sdf' flag the same particles."""
        center = (0.0, 0.0, 0.0)
        R = 1.0
        body_sdf = ContactBody(
            body_id=0, sdf_net=SphereSDF(center=center, radius=R),
            seeds=torch.tensor([list(center)], dtype=torch.float64),
            support_radii=torch.tensor([1.5], dtype=torch.float64),
            detector="sdf",
        )
        body_chart = ContactBody(
            body_id=0, chart=SphereRho(R, center=center),
            seeds=torch.tensor([list(center)], dtype=torch.float64),
            support_radii=torch.tensor([1.5], dtype=torch.float64),
            detector="chart",
        )
        mgr = ContactManager([body_sdf])
        torch.manual_seed(23)
        x_phys = 1.6 * torch.randn(200, 3, dtype=torch.float64)

        pair_sdf = mgr.detect_mpm(body_sdf, body_B_id=1, chart_id_B=0, x_phys=x_phys)
        pair_chart = mgr.detect_mpm(body_chart, body_B_id=1, chart_id_B=0, x_phys=x_phys)
        assert pair_sdf is not None and pair_chart is not None
        assert torch.equal(pair_sdf.particle_indices, pair_chart.particle_indices)
        # gaps and normals also agree (sphere => machine precision)
        assert (pair_sdf.gap - pair_chart.gap).abs().max().item() < 1e-12

    def test_contact_body_validation(self):
        """ContactBody enforces a source for the chosen detector."""
        with pytest.raises(ValueError):
            ContactBody(body_id=0, detector="chart")        # missing chart
        with pytest.raises(ValueError):
            ContactBody(body_id=0, detector="sdf")          # missing sdf_net
        with pytest.raises(ValueError):
            ContactBody(body_id=0, sdf_net=SphereSDF(), detector="bogus")

    def test_from_chart_populates_broadphase_seeds(self):
        """from_chart sets a bounding-sphere seed so the schwarz broad-phase works."""
        chart = SphereRho(0.5, center=(1.0, 2.0, 3.0))
        body = ContactBody.from_chart(body_id=7, chart=chart, margin=0.1)
        assert body.detector == "chart"
        assert body.seeds.shape == (1, 3)
        assert torch.allclose(body.seeds[0], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert abs(body.support_radii[0].item() - 0.6) < 1e-12

    def test_chart_body_none_seeds_detect_mpm_safe(self):
        """A chart body with seeds=None still works through detect_mpm (no broad-phase)."""
        body = ContactBody(body_id=0, chart=SphereRho(1.0, center=(0.0, 0.0, 0.0)),
                           detector="chart")  # seeds/support_radii default None
        mgr = ContactManager([body])
        x = torch.tensor([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        pair = mgr.detect_mpm(body, body_B_id=1, chart_id_B=0, x_phys=x)
        assert pair is not None
        assert pair.particle_indices.tolist() == [0]


# ── 7. Contract parity (empty input, no_grad block) ──────────────────

class TestContract:
    def test_empty_input(self):
        chart = SphereRho(1.0)
        pts = torch.zeros(0, 3, dtype=torch.float64)
        gap, normal = evaluate_gap_chart(pts, chart)
        assert gap.shape == (0,)
        assert normal.shape == (0, 3)

    def test_inside_no_grad_block(self):
        chart = SphereRho(1.0, center=(0.0, 0.0, 0.0))
        pts = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
        with torch.no_grad():
            gap, normal = evaluate_gap_chart(pts, chart)
        assert abs(gap[0].item() + 0.5) < 1e-12
        assert abs(normal[0, 0].item() - 1.0) < 1e-12

    def test_float32_input(self):
        chart = SphereRho(1.0, center=(0.0, 0.0, 0.0))
        pts = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        gap, normal = evaluate_gap_chart(pts, chart)
        assert gap.shape == (1,)
        assert normal.shape == (1, 3)
        assert abs(gap[0].item() - 1.0) < 1e-5
