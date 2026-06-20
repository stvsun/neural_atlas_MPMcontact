"""Verification of the rock-joint direct-shear capstone.

Fast, self-contained checks (no trained .pt, no downloads required):
  * the analytic sawtooth flank slope equals tan(i);
  * the L1 ANCHOR — mating-sawtooth direct shear under plain Coulomb emergently reproduces the
    closed-form Patton law mu_app = tan(phi_b + i);
  * the Fourier-feature height chart resolves a rough synthetic profile (small RMSE) while a
    capacity-matched plain coordinate MLP (the ambient-SDF class) does not — and crucially SMOOTHS
    the asperity slopes (the directional, contact-relevant claim).

The real-Inada-surface checks skip cleanly if data/inada_joint/*.npz is absent (gitignored raw data).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from solvers.contact.profile_chart_2d import (  # noqa: E402
    AnalyticSawtooth1D, NeuralHeight1D, fit_height_chart, height_and_grad, surface_normal,
)
from benchmarks.contact.cv_numerical import rock_joint_shear as rjs  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "inada_joint")


def test_sawtooth_flank_slope_is_tan_i():
    saw = AnalyticSawtooth1D(wavelength=5.0, angle_deg=18.0).double()
    x = torch.linspace(0.3, 49.7, 5000, dtype=torch.float64)
    _, dh = height_and_grad(x, saw)
    med = float(torch.median(torch.abs(dh)))
    assert abs(med - math.tan(math.radians(18.0))) < 1e-6


@pytest.mark.parametrize("angle,mu", [(15.0, 0.3), (25.0, 0.2)])
def test_patton_emergent_law(angle, mu):
    """L1 anchor: plain Coulomb on the resolved tilted sawtooth contact -> tan(phi_b + i)."""
    res = rjs.run_sawtooth_patton(angle_deg=angle, mu=mu)
    s = res["summary"]
    assert s["mu_app_relerr_vs_patton"] < 0.02, s            # within 2% of the closed form
    # dilation rate should be ~tan(i) (the block rides up the flank)
    h = res["history"]
    rate = (h["dilation"][-1]) / max(h["ux"][-1], 1e-9)
    assert abs(rate - math.tan(math.radians(angle))) < 0.08, rate


def test_height_chart_resolves_rough_profile():
    rng = np.random.RandomState(1)
    xs = np.linspace(0.0, 50.0, 4000)
    ks = np.arange(1, 80)
    zs = sum((k ** -1.0) * np.sin(2 * np.pi * k * xs / 50.0 + p)
             for k, p in zip(ks, rng.uniform(0, 2 * np.pi, ks.size)))
    chart, rmse = fit_height_chart(xs, zs, f_max=150.0, n_freq=128, iters=2000)
    assert rmse / zs.std() < 0.05, rmse / zs.std()
    n = surface_normal(torch.from_numpy(xs[:100]), chart)
    assert torch.allclose(n.norm(dim=1), torch.ones(100, dtype=torch.float64), atol=1e-9)


def test_plain_mlp_smooths_slopes_vs_chart():
    """Directional thesis check: at matched width/depth the plain coordinate MLP (ambient-SDF class)
    UNDER-resolves the asperity slopes relative to the Fourier chart and the reference."""
    rng = np.random.RandomState(2)
    xs = np.linspace(0.0, 50.0, 4000)
    ks = np.arange(1, 120)
    zs = sum((k ** -0.9) * np.sin(2 * np.pi * k * xs / 50.0 + p)
             for k, p in zip(ks, rng.uniform(0, 2 * np.pi, ks.size)))
    ref_rms_slope = np.sqrt(np.mean(np.gradient(zs, xs) ** 2))
    cF, _ = fit_height_chart(xs, zs, f_max=200.0, n_freq=128, iters=2000, plain=False)
    cP, _ = fit_height_chart(xs, zs, f_max=200.0, n_freq=128, iters=2000, plain=True)
    _, sF = height_and_grad(torch.from_numpy(xs), cF)
    _, sP = height_and_grad(torch.from_numpy(xs), cP)
    rms_F = float(torch.sqrt(torch.mean(sF ** 2)))
    rms_P = float(torch.sqrt(torch.mean(sP ** 2)))
    # plain MLP loses slope energy (smooths); chart retains far more of the reference
    assert rms_P < 0.7 * ref_rms_slope, (rms_P, ref_rms_slope)
    assert rms_F > rms_P, (rms_F, rms_P)


@pytest.mark.skipif(not os.path.exists(os.path.join(_DATA, "inada_rough_profile.npz")),
                    reason="Inada profile not present (run characterize_inada_joint.py)")
def test_real_inada_chart_reproduces_surface():
    d = np.load(os.path.join(_DATA, "inada_rough_profile.npz"))
    x, z = d["x_mm"].astype(float), d["footwall_mm"].astype(float)
    chart, rmse = fit_height_chart(x, z, f_max=200.0, n_freq=64, iters=1500)
    # the chart represents the measured surface to a small fraction of its RMS roughness
    assert rmse / z.std() < 0.05, rmse / z.std()
