#!/usr/bin/env python3
"""Multi-scale roughness / spectral-cutoff sweep -> the dilatancy-vs-roughness LAW (Phase 2).

For a grid of band-limited rough surfaces (varying amplitude and spectral cutoff k_max), this:
  1. trains + VERIFIES a Fourier rough-block decoder (reconstruction RMSE, det J > 0), then
  2. runs the GENUINE monotonic shear (one deformable decoder block vs the rigid mating rough face,
     `rock_joint_decoder_shear.run_shear`, CNL), and records peak mu_app and total emergent dilation.

Output: the emergent dilatancy law  peak mu_app(RMS, k_max)  and  dilation(RMS, k_max) — measured, not
imposed.  Each config is independent -> run in parallel across the cluster cores.

Run:  python3 benchmarks/contact/cv_numerical/cv7_roughness_sweep.py            (parallel sweep)
      python3 benchmarks/contact/cv_numerical/cv7_roughness_sweep.py --serial   (debug, sequential)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)
RUN = os.path.join(_ROOT, "runs", "cv7_roughness_sweep")


def _one_config(cfg):
    """Train+verify a decoder for (amp,k_max) and run the genuine monotonic shear.  Self-contained
    (importable in a worker process)."""
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    from solvers.fem.rough_block_decoder import (band_limited_rough_surface, train_rough_decoder,
                                                 verify_decoder)
    from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import run_shear
    amp, kmax, n_cells, mu, iters = cfg["amp"], cfg["kmax"], cfg["n_cells"], cfg["mu"], cfg["iters"]
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=amp, k_max=kmax)      # noqa: E731
    rng = np.random.RandomState(7)
    rms = float(np.std(tgt(rng.uniform(-1, 1, 8000), rng.uniform(-1, 1, 8000))))
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", k_max=max(6.0, 2.0 * kmax + 2.0),
                                        iters=iters)
    v = verify_decoder(dec, dk, n_cells=n_cells)
    js, hist = run_shear(dec, dk, protocol="CNL", sigma_n=2.0, shear_total=0.16, n_inc=11,
                         n_cells=n_cells, mu=mu, surf_amp=amp, eps_n=1.2e4, W=2.0 * (2.0) ** 2,
                         surf_fn=lambda x, y: band_limited_rough_surface(x, y, amp=amp, k_max=kmax),
                         verbose=False)
    return dict(amp=amp, kmax=kmax, surf_rms=rms, recon_rmse=rmse, recon_pct=100 * rmse / max(rms, 1e-12),
                detJ_min=v["detJ_min"], all_valid=bool(v["all_valid"]),
                peak_mu_app=float(np.abs(hist["mu_app"]).max()),
                total_dilation=float(hist["dilation"][-1]),
                max_resid=float(hist["resid"].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--n_cells", type=int, default=10)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    # amplitude sweep at fixed cutoff + spectral-cutoff sweep at fixed amplitude
    cfgs = []
    for amp in (0.04, 0.06, 0.08, 0.10, 0.12):
        cfgs.append(dict(amp=amp, kmax=2.2, family="amp"))
    for kmax in (1.2, 1.8, 2.6, 3.4, 4.2):
        cfgs.append(dict(amp=0.08, kmax=kmax, family="kmax"))
    for c in cfgs:
        c.update(n_cells=args.n_cells, mu=args.mu, iters=args.iters)
    mode_str = "serial" if args.serial else f"{args.workers} workers"
    print(f"=== roughness sweep: {len(cfgs)} configs ({mode_str}) ===")
    if args.serial:
        results = [_one_config(c) for c in cfgs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_one_config, cfgs))
    for c, r in zip(cfgs, results):
        r["family"] = c["family"]
        print(f"  amp={r['amp']:.3f} kmax={r['kmax']:.1f} RMS={r['surf_rms']:.4f} "
              f"recon={r['recon_pct']:.1f}% detJmin={r['detJ_min']:.3f} "
              f"peak_mu={r['peak_mu_app']:.3f} dil={r['total_dilation']:+.4f} resid={r['max_resid']:.1e}")
    json.dump(results, open(os.path.join(RUN, "results.json"), "w"), indent=2)
    print(f"  saved -> {os.path.join(RUN, 'results.json')}")


if __name__ == "__main__":
    main()
