"""Tests for topology certification pipeline.

Tests the glue code in atlas/topo/certify.py that connects
SDF networks to persistent homology.
"""

import numpy as np
import torch
import pytest

from atlas.topo.filtration import sdf_ball, sdf_solid_torus, clip_to_interior
from atlas.topo.certify import certify_sdf


class AnalyticSDF(torch.nn.Module):
    """Wraps an analytic SDF function as a torch Module for testing."""
    def __init__(self, sdf_fn):
        super().__init__()
        self.sdf_fn = sdf_fn

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        vals = self.sdf_fn(x_np)
        return torch.tensor(vals, dtype=x.dtype, device=x.device)


class TestCertifySDF:
    def test_ball_m_min_is_1(self):
        """A ball is contractible: M_min = 1."""
        sdf_net = AnalyticSDF(lambda x: sdf_ball(x, radius=0.8))
        bbox_min = torch.tensor([-1.5, -1.5, -1.5])
        bbox_max = torch.tensor([1.5, 1.5, 1.5])

        report = certify_sdf(sdf_net, bbox_min, bbox_max, resolution=24)

        if report["has_gudhi"]:
            assert report["M_min"] == 1, f"Ball should have M_min=1, got {report['M_min']}"
            assert report["betti"][0] == 1, "Ball should have beta_0=1"
            assert report["betti"].get(1, 0) == 0, "Ball should have beta_1=0"
        else:
            # Without GUDHI, defaults to M_min=1 (contractible assumption)
            assert report["M_min"] == 1

    def test_solid_torus_m_min_is_2(self):
        """A solid torus has a 1-loop: M_min = 2."""
        sdf_net = AnalyticSDF(lambda x: sdf_solid_torus(x, R=1.0, r=0.35))
        bbox_min = torch.tensor([-2.0, -2.0, -1.0])
        bbox_max = torch.tensor([2.0, 2.0, 1.0])

        report = certify_sdf(sdf_net, bbox_min, bbox_max, resolution=24)

        if report["has_gudhi"]:
            assert report["M_min"] == 2, f"Solid torus should have M_min=2, got {report['M_min']}"
            assert report["betti"][0] == 1, "Solid torus should have beta_0=1"
            assert report["betti"].get(1, 0) == 1, "Solid torus should have beta_1=1"
        else:
            pytest.skip("GUDHI not available")

    def test_report_has_expected_keys(self):
        """Certification report should have standard keys."""
        sdf_net = AnalyticSDF(lambda x: sdf_ball(x, radius=0.5))
        bbox_min = torch.tensor([-1.0, -1.0, -1.0])
        bbox_max = torch.tensor([1.0, 1.0, 1.0])

        report = certify_sdf(sdf_net, bbox_min, bbox_max, resolution=16)

        assert "M_min" in report
        assert "betti" in report
        assert "explanation" in report
        assert "has_gudhi" in report
        assert "grid_vals" in report
        assert isinstance(report["M_min"], int)
        assert isinstance(report["betti"], dict)

    def test_grid_vals_shape(self):
        """Grid values should match resolution^3."""
        resolution = 16
        sdf_net = AnalyticSDF(lambda x: sdf_ball(x, radius=0.5))
        bbox_min = torch.tensor([-1.0, -1.0, -1.0])
        bbox_max = torch.tensor([1.0, 1.0, 1.0])

        report = certify_sdf(sdf_net, bbox_min, bbox_max, resolution=resolution)
        assert report["grid_vals"].shape == (resolution, resolution, resolution)


class TestInvertDecoder:
    """Test the Newton decoder inversion in common/geometry.py."""

    def test_identity_decoder_inversion(self):
        """For an identity-like decoder, inversion should recover input."""
        from common.models import ChartDecoder
        from common.geometry import invert_decoder

        # Use float32 to match default Linear layer weights
        decoder = ChartDecoder(width=16, depth=2).float()
        # Override raw_scale to float32
        decoder.raw_scale = torch.nn.Parameter(torch.tensor(-1.8, dtype=torch.float32))

        seed = torch.zeros(3, dtype=torch.float32)
        t1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        t2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        n_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32)

        x_target = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1]], dtype=torch.float32)

        with torch.enable_grad():
            xi_star = invert_decoder(
                decoder, x_target, seed, t1, t2, n_vec, scale,
                max_iter=30, tol=1e-6,
            )

        with torch.no_grad():
            x_recovered = decoder(xi_star, seed=seed, t1=t1, t2=t2, n=n_vec, chart_scale=scale)

        err = torch.linalg.norm(x_recovered - x_target, dim=1)
        assert err.max().item() < 1e-4, f"Inversion error too large: {err.max().item()}"


class TestSchwarzFEMSolver:
    """Basic tests for the SchwarzFEMSolver class."""

    def test_import(self):
        """SchwarzFEMSolver should be importable."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver
        assert SchwarzFEMSolver is not None

    def test_build_neighbors(self):
        """Neighbor graph should be built correctly from membership."""
        from solvers.fem.schwarz_fem import SchwarzFEMSolver

        # 2 charts with some overlap
        membership = np.array([
            [True, False],
            [True, True],   # overlap
            [True, True],   # overlap
            [False, True],
        ])

        solver = SchwarzFEMSolver.__new__(SchwarzFEMSolver)
        solver.n_charts = 2
        neighbors = solver._build_neighbors(membership)

        assert 1 in neighbors[0], "Chart 0 should neighbor chart 1"
        assert 0 in neighbors[1], "Chart 1 should neighbor chart 0"


class TestSchwarzMPMSolver:
    """Basic tests for the SchwarzMPMSolver class."""

    def test_import(self):
        """SchwarzMPMSolver should be importable."""
        from solvers.mpm.schwarz_mpm import SchwarzMPMSolver
        assert SchwarzMPMSolver is not None
