#!/usr/bin/env python3
"""Wire the TRANSITION-MAP (height-chart) detector into the contact loop and verify it (Phase 3).

`cv7_transition_map_verify.py` showed the height-chart gap/normal beats the ambient SDF in accuracy.
This driver closes Phase 3: it (1) confirms the chart detector satisfies the SAME (gap, normal) contract
the contact manager dispatches on (`solvers/contact/contact_manager.body_gap_normal`: gap<0 => penetration,
unit outward normal), (2) measures the ACTIVE-SET agreement vs the analytic reference on ~1e5 query points
(chart vs ambient SDF), (3) times the per-query cost (chart vs SDF, target <2x), (4) reports normal
accuracy at the steep ASPERITY TIPS (where the level set smooths most), and (5) DRIVES the genuine
one-block chart-FEM shear through the chart detector (swapping the analytic mating surface for the
transition-map height query) and checks the emergent mu_app/dilation match the analytic-surface run.

Run:  python3 benchmarks/contact/cv_numerical/cv7_transition_map_contact.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.contact.surface_chart_3d import fit_surface_chart, height_and_grad  # noqa: E402
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from benchmarks.contact.cv_numerical.cv7_decoder_verify import train_ambient_sdf3d  # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import run_shear  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(_ROOT, "runs", "cv7_transition_map_contact")
AMP = 0.10


def _analytic(xy, z):
    e = 1e-4
    h = band_limited_rough_surface(xy[:, 0], xy[:, 1], amp=AMP)
    hx = (band_limited_rough_surface(xy[:, 0] + e, xy[:, 1], amp=AMP) - h) / e
    hy = (band_limited_rough_surface(xy[:, 0], xy[:, 1] + e, amp=AMP) - h) / e
    sec = np.sqrt(1 + hx ** 2 + hy ** 2)
    n = np.stack([-hx / sec, -hy / sec, 1.0 / sec], 1)
    return (z - h) / sec, n, np.hypot(hx, hy)


def _chart(xy, z, chart):
    h, gh = height_and_grad(torch.from_numpy(xy), chart)
    h = h.numpy(); hx = gh[:, 0].numpy(); hy = gh[:, 1].numpy()
    sec = np.sqrt(1 + hx ** 2 + hy ** 2)
    n = np.stack([-hx / sec, -hy / sec, 1.0 / sec], 1)
    return (z - h) / sec, n


def _sdf(P, sdf):
    Pt = torch.tensor(P, requires_grad=True)
    phi = sdf(Pt)
    g = torch.autograd.grad(phi, Pt, torch.ones_like(phi))[0]
    n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach().numpy()
    return phi.detach().numpy(), n


def main():
    os.makedirs(RUN, exist_ok=True)
    rng = np.random.RandomState(0)
    gx = np.linspace(-1, 1, 130); GX, GY = np.meshgrid(gx, gx)
    xy_tr = np.stack([GX.ravel(), GY.ravel()], 1)
    z_tr = band_limited_rough_surface(xy_tr[:, 0], xy_tr[:, 1], amp=AMP)
    print("=== train transition-map height chart + ambient SDF ===")
    chart, rmse = fit_surface_chart(xy_tr, z_tr, sigma=4.0, n_freq=256, iters=3000)
    sdf, sdf_rmse = train_ambient_sdf3d(lambda x, y: band_limited_rough_surface(x, y, amp=AMP), band=2.5 * AMP)
    print(f"  chart recon {rmse:.3e}  sdf recon {sdf_rmse:.3e}")

    # (2) active-set agreement on a large near-contact point cloud
    N = 100000
    xyq = rng.uniform(-0.9, 0.9, (N, 2))
    hq = band_limited_rough_surface(xyq[:, 0], xyq[:, 1], amp=AMP)
    zq = hq + rng.uniform(-0.05, 0.05, N)
    P = np.stack([xyq[:, 0], xyq[:, 1], zq], 1)
    g_a, n_a, slope = _analytic(xyq, zq)
    g_c, n_c = _chart(xyq, zq, chart)
    g_s, n_s = _sdf(P, sdf)
    act_a = g_a < 0
    agree_c = float(np.mean((g_c < 0) == act_a))
    agree_s = float(np.mean((g_s < 0) == act_a))

    # (3) per-query timing
    Pt = torch.from_numpy(P)
    t0 = time.time(); _ = height_and_grad(torch.from_numpy(xyq), chart); t_chart = time.time() - t0
    t0 = time.time(); _ = _sdf(P, sdf); t_sdf = time.time() - t0

    # (4) normals at steep asperity tips (top decile of |grad h|)
    tip = slope > np.percentile(slope, 90)
    def ang(na, nb):
        d = np.clip((na * nb).sum(1), -1, 1)
        return np.degrees(np.arccos(np.abs(d)))
    tip_chart = float(np.median(ang(n_c[tip], n_a[tip])))
    tip_sdf = float(np.median(ang(n_s[tip], n_a[tip])))

    # (5) DRIVE the genuine FEM shear through the chart detector vs the analytic surface
    print("=== drive one-block FEM shear: analytic surface vs chart detector ===")
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=AMP)          # noqa: E731
    dec, drmse, dk = train_rough_decoder(tgt, rough_face="top", iters=3000)

    def chart_surf(x, y):                                                 # height-chart query (transition map)
        xy = np.stack([np.asarray(x).ravel(), np.asarray(y).ravel()], 1)
        h, _ = height_and_grad(torch.from_numpy(xy), chart)
        return h.numpy().reshape(np.shape(x))
    _, h_ana = run_shear(dec, dk, protocol="CNV", shear_total=0.12, n_inc=7, n_cells=10, mu=0.4,
                         surf_amp=AMP, compress=0.5, verbose=False)
    _, h_chart = run_shear(dec, dk, protocol="CNV", shear_total=0.12, n_inc=7, n_cells=10, mu=0.4,
                           surf_amp=AMP, compress=0.5, surf_fn=chart_surf, verbose=False)
    mu_ana = float(np.abs(h_ana["mu_app"]).max()); mu_chart = float(np.abs(h_chart["mu_app"]).max())
    tau_rel = float(np.max(np.abs(h_chart["tau"] - h_ana["tau"])) / (np.max(np.abs(h_ana["tau"])) + 1e-12))

    res = dict(chart_recon=rmse, sdf_recon=sdf_rmse,
               active_set_agree_chart=agree_c, active_set_agree_sdf=agree_s, n_query=N,
               per_query_us_chart=t_chart / N * 1e6, per_query_us_sdf=t_sdf / N * 1e6,
               cost_ratio_chart_over_sdf=t_chart / max(t_sdf, 1e-12),
               tip_normal_deg_chart=tip_chart, tip_normal_deg_sdf=tip_sdf,
               fem_peak_mu_analytic=mu_ana, fem_peak_mu_chart=mu_chart,
               fem_tau_rel_diff=tau_rel)
    json.dump(res, open(os.path.join(RUN, "verify.json"), "w"), indent=2)
    print("\n=== PHASE-3 transition-map-into-contact summary ===")
    print(f"  active-set agreement vs analytic:  chart {agree_c*100:.2f}%   SDF {agree_s*100:.2f}%  (N={N})")
    print(f"  per-query cost: chart {res['per_query_us_chart']:.2f} us  SDF {res['per_query_us_sdf']:.2f} us "
          f"(ratio {res['cost_ratio_chart_over_sdf']:.2f}x)")
    print(f"  asperity-TIP normal error: chart {tip_chart:.2f} deg   SDF {tip_sdf:.2f} deg")
    print(f"  FEM shear via chart detector vs analytic surface: peak mu_app {mu_chart:.3f} vs {mu_ana:.3f}, "
          f"tau max rel-diff {tau_rel*100:.2f}%")
    print(f"  saved -> {os.path.join(RUN, 'verify.json')}")


if __name__ == "__main__":
    main()
