#!/usr/bin/env python3
"""Verify the CV-7 contact mechanics via the TRANSITION MAP (chart-based, level-set-free) detector.

The contact-detection contract is (gap, normal).  For the single-valued rough rock joint the transition
map is the HEIGHT-CHART projection: a query point p=(x,y,z) maps to the surface point (x,y,h_theta(x,y)),
giving the normal gap and the chart normal WITHOUT an ambient level set.  (The radial transition-map
chart `radial_chart_2d.NeuralRho2D` assumes a STAR-SHAPED body; a height field is not star-shaped, so
the height-chart projection is the correct transition map here.)

We verify the transition-map detector against the analytic gap/normal, and contrast it with the ambient
neural SDF (the level set) — the detector that drives the contact forces must be accurate, and the
transition map (boundary-fitted chart) beats the smoothing level set.

Run:  python3 benchmarks/contact/cv_numerical/cv7_transition_map_verify.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.contact.surface_chart_3d import fit_surface_chart, height_and_grad  # noqa: E402
from solvers.fem.rough_block_decoder import band_limited_rough_surface           # noqa: E402
from benchmarks.contact.cv_numerical.cv7_decoder_verify import train_ambient_sdf3d  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(_ROOT, "runs", "cv7_transition_map")
AMP = 0.10


def analytic_gap_normal(xy, z):
    """Analytic normal gap + unit normal of the surface z=h_true(x,y) (finite-difference gradient)."""
    eps = 1e-4
    h = band_limited_rough_surface(xy[:, 0], xy[:, 1], amp=AMP)
    hx = (band_limited_rough_surface(xy[:, 0] + eps, xy[:, 1], amp=AMP) - h) / eps
    hy = (band_limited_rough_surface(xy[:, 0], xy[:, 1] + eps, amp=AMP) - h) / eps
    sec = np.sqrt(1 + hx ** 2 + hy ** 2)
    n = np.stack([-hx / sec, -hy / sec, 1.0 / sec], 1)
    gap = (z - h) / sec                                            # normal-projected gap
    return gap, n


def transition_map_gap_normal(xy, z, chart):
    """Chart-based (transition-map) gap + normal: project onto z=h_theta(x,y) via the height chart."""
    h, gh = height_and_grad(torch.from_numpy(xy), chart)
    h = h.numpy(); hx = gh[:, 0].numpy(); hy = gh[:, 1].numpy()
    sec = np.sqrt(1 + hx ** 2 + hy ** 2)
    n = np.stack([-hx / sec, -hy / sec, 1.0 / sec], 1)
    gap = (z - h) / sec
    return gap, n


def sdf_gap_normal(P, sdf):
    """Ambient-SDF (level-set) gap + normal: phi(p) and grad phi/|grad phi| (autograd)."""
    Pt = torch.tensor(P, requires_grad=True)
    phi = sdf(Pt)
    g = torch.autograd.grad(phi, Pt, torch.ones_like(phi))[0]
    n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach().numpy()
    return phi.detach().numpy(), n


def main():
    os.makedirs(RUN, exist_ok=True)
    rng = np.random.RandomState(0)
    # training data for the height chart (dense surface samples)
    gx = np.linspace(-1, 1, 130); GX, GY = np.meshgrid(gx, gx)
    xy_tr = np.stack([GX.ravel(), GY.ravel()], 1)
    z_tr = band_limited_rough_surface(xy_tr[:, 0], xy_tr[:, 1], amp=AMP)
    print("=== train the height chart (transition-map detector) ===")
    chart, rmse = fit_surface_chart(xy_tr, z_tr, sigma=4.0, n_freq=256, iters=3000)
    surf_rms = float(z_tr.std())
    print(f"  chart reconstruction RMSE {rmse:.3e} ({rmse/surf_rms*100:.2f}% of RMS)")
    print("=== train the ambient SDF (level set) ===")
    sdf, sdf_rmse = train_ambient_sdf3d(lambda x, y: band_limited_rough_surface(x, y, amp=AMP),
                                        band=2.5 * AMP)

    # query points in a near-contact band around the surface
    nq = 4000
    xyq = rng.uniform(-0.9, 0.9, (nq, 2))
    hq = band_limited_rough_surface(xyq[:, 0], xyq[:, 1], amp=AMP)
    zq = hq + rng.uniform(-0.04, 0.04, nq)                          # near-surface band
    P = np.stack([xyq[:, 0], xyq[:, 1], zq], 1)

    g_ana, n_ana = analytic_gap_normal(xyq, zq)
    g_tm, n_tm = transition_map_gap_normal(xyq, zq, chart)
    g_sdf, n_sdf = sdf_gap_normal(P, sdf)

    def ang(na, nb):
        d = np.clip((na * nb).sum(1), -1, 1)
        return np.degrees(np.arccos(np.abs(d)))                    # unsigned normal angle

    res = dict(
        chart_recon_pct=rmse / surf_rms * 100,
        transition_map=dict(
            gap_rmse=float(np.sqrt(np.mean((g_tm - g_ana) ** 2))),
            gap_rmse_rel=float(np.sqrt(np.mean((g_tm - g_ana) ** 2)) / surf_rms),
            normal_median_deg=float(np.median(ang(n_tm, n_ana))),
            normal_p90_deg=float(np.percentile(ang(n_tm, n_ana), 90))),
        ambient_sdf=dict(
            gap_rmse=float(np.sqrt(np.mean((g_sdf - g_ana) ** 2))),
            gap_rmse_rel=float(np.sqrt(np.mean((g_sdf - g_ana) ** 2)) / surf_rms),
            normal_median_deg=float(np.median(ang(n_sdf, n_ana))),
            normal_p90_deg=float(np.percentile(ang(n_sdf, n_ana), 90))),
    )
    json.dump(res, open(os.path.join(RUN, "verify.json"), "w"), indent=2)
    tm, sd = res["transition_map"], res["ambient_sdf"]
    print("\n=== transition-map vs level-set contact DETECTION (vs analytic) ===")
    print(f"  transition map (height chart): gap RMSE {tm['gap_rmse']:.3e} ({tm['gap_rmse_rel']*100:.2f}% of RMS), "
          f"normal median {tm['normal_median_deg']:.2f} deg (p90 {tm['normal_p90_deg']:.2f})")
    print(f"  ambient SDF (level set):       gap RMSE {sd['gap_rmse']:.3e} ({sd['gap_rmse_rel']*100:.2f}% of RMS), "
          f"normal median {sd['normal_median_deg']:.2f} deg (p90 {sd['normal_p90_deg']:.2f})")
    print(f"  -> transition-map gap {sd['gap_rmse']/max(tm['gap_rmse'],1e-12):.1f}x more accurate; "
          f"normal {sd['normal_median_deg']/max(tm['normal_median_deg'],1e-9):.1f}x sharper")
    print(f"  saved -> {os.path.join(RUN, 'verify.json')}")


if __name__ == "__main__":
    main()
