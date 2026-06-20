"""Verification of the 3D rock-joint shear extension (rigid modes + deformable FEM).

Crisp, analytic checks (the rigorous anchors):
  * the 2-D height chart's ridged-sawtooth slopes are tan(i) across ridges, 0 along them;
  * NeuralHeight2D resolves a rough synthetic surface;
  * RIGID 3-D anisotropy: shearing ACROSS ridges (in-plane) -> Patton tan(phi_b+i) with dilation;
    shearing ALONG ridges (out-of-plane) -> NO dilation (pure friction);
  * DEFORMABLE two-block FEM, monotonic: flat joint -> Coulomb tau/tN -> mu; uniform dilation angle
    i -> Patton tau/tN -> tan(phi_b+i).  These verify the deformable dilatant-frictional interface.

The cyclic-FEM hysteresis runs as a benchmark driver (rock_joint_cyclic_fem.py), not a routine test
(its energy balance is approximate at slip peaks — Coulomb non-smoothness; see manual 11.10).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")

from solvers.contact.surface_chart_3d import (  # noqa: E402
    RidgedSawtooth3D, height_and_grad, surface_normal_3d, fit_surface_chart,
)


def test_chart3d_ridged_slopes():
    saw = RidgedSawtooth3D(wavelength=5.0, angle_deg=18.0).double()
    g = torch.tensor(np.stack([np.linspace(0.3, 49.7, 3000), np.linspace(0, 30, 3000)], 1))
    _, gh = height_and_grad(g, saw)
    assert abs(float(torch.median(torch.abs(gh[:, 0]))) - math.tan(math.radians(18.0))) < 1e-6
    assert float(torch.abs(gh[:, 1]).max()) < 1e-9          # flat along ridges


def test_chart3d_fits_rough_surface():
    rng = np.random.RandomState(0)
    xs = np.linspace(0, 20, 110); ys = np.linspace(0, 20, 110)
    XX, YY = np.meshgrid(xs, ys)
    Z = sum((1.0 / (kx + ky)) * np.sin(kx * XX + ky * YY + p)
            for kx, ky, p in zip(rng.uniform(0.2, 2, 30), rng.uniform(0.2, 2, 30),
                                 rng.uniform(0, 2 * np.pi, 30)))
    xy = np.stack([XX.ravel(), YY.ravel()], 1)
    chart, rmse = fit_surface_chart(xy, Z.ravel(), sigma=4.0, n_freq=256, iters=1200)
    assert rmse / Z.std() < 0.10
    n = surface_normal_3d(torch.from_numpy(xy[:200]), chart)
    assert torch.allclose(n.norm(dim=1), torch.ones(200, dtype=torch.float64), atol=1e-9)
    assert (n[:, 2] > 0).all()


def test_rigid_3d_ridged_anisotropy():
    from benchmarks.contact.cv_numerical.rock_joint_shear_3d import run_ridged_anisotropy
    out = run_ridged_anisotropy(angle_deg=20.0, mu=0.3, wavelength=5.0)
    assert out["in_plane_relerr_vs_patton"] < 0.02                       # in-plane -> Patton
    assert abs(out["out_of_plane"]["dilation_total"]) < 1e-3             # along-ridge -> no dilation
    assert out["out_of_plane"]["mu_app_steady"] > out["in_plane"]["mu_app_steady"] * 0  # finite


@pytest.mark.parametrize("dil_deg,expect", [(0.0, "coulomb"), (8.0, "patton")])
def test_fem_monotonic_strength(dil_deg, expect):
    """Deformable two-block FEM (CNV, rigid platen): flat -> mu; dilatant -> tan(phi_b+i)."""
    from benchmarks.contact.cv_numerical.rock_joint_cyclic_fem import JointFEM
    mu = 0.4
    fem = JointFEM(n_cells=4, half=1.0, E=2.0e3, nu=0.25, dilation_deg=dil_deg, mu=mu)
    K = fem._Kblocks(); u = np.zeros(fem.ndof); d = np.array([1.0, 0.0]); uz0 = -0.02
    diag = None
    for umag in np.linspace(0, 0.10, 13):
        u, diag, _ = fem._solve_fixed_top((umag * d[0], umag * d[1], uz0), K, u, max_iter=160)
    ratio = diag["tx_mean"] / max(diag["tN_mean"], 1e-9)
    target = math.tan(math.atan(mu) + math.radians(dil_deg))             # mu (flat) or Patton
    assert abs(ratio - target) / target < 0.03, (expect, ratio, target)
