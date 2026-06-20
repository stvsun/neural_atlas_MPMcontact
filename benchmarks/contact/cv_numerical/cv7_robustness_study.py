#!/usr/bin/env python3
"""CV-7 V&V: robustness of the atlas-beats-level-set result across roughness REALIZATIONS.

Manuscript credibility: the SDF's under-prediction of the emergent (frictionless) geometric dilatancy
must be a statistical fact, not a single cherry-picked surface.  For N independent band-limited rough
surfaces (different random seeds) this trains the atlas decoder and an ambient SDF, runs the genuine
frictionless shear on each, and reports the atlas vs SDF peak dilatancy + the under-prediction, with
mean ± std.  Self-contained (new runs/cv7_robustness/); does not touch other CV-7 files.

Run:  python3 benchmarks/contact/cv_numerical/cv7_robustness_study.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import DecoderJointShear         # noqa: E402
from benchmarks.contact.cv_numerical.cv7_decoder_verify import train_ambient_sdf3d             # noqa: E402
from benchmarks.contact.cv_numerical.cv7_atlas_vs_sdf_shear import sdf_levelset_fn             # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(_ROOT, "runs", "cv7_robustness")
AMP = 0.06


def frictionless_peak(dec, dk, surf_fn, n_cells=12, eps_n=2.0e4):
    js = DecoderJointShear(dec, dk, n_cells=n_cells, mu=0.0, surf_amp=AMP, eps_n=eps_n)
    js._surf_fn = surf_fn
    u = np.zeros(3 * js.N); z_p = 1.0 - 0.5 * AMP; peak = 0.0
    for ux in np.linspace(0, 0.15, 9):
        u, diag = js.solve_fixed(ux, z_p, u, max_iter=140)
        Fn = js.normal_force(u, ux, z_p); U = u.reshape(js.N, 3)
        fc, _ = js.contact(U, ux, z_p); Fx = float(fc[js.top, 0].sum())
        peak = max(peak, Fx / max(Fn, 1e-9))
    return peak


def main():
    os.makedirs(RUN, exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for sd in seeds:
        tgt = lambda x, y, s=sd: band_limited_rough_surface(x, y, amp=AMP, seed=s)
        dec, rmse, dk = train_rough_decoder(lambda x, y: tgt(x, y), iters=3500)
        atlas = frictionless_peak(dec, dk, tgt)
        sdf, sdf_rmse = train_ambient_sdf3d(lambda x, y: tgt(x, y), band=2.5 * AMP, iters=3500)
        h_sdf = sdf_levelset_fn(sdf, band=2.5 * AMP)
        dec_s, _, dk_s = train_rough_decoder(lambda x, y: h_sdf(x, y), iters=3500)
        sdf_peak = frictionless_peak(dec_s, dk_s, lambda x, y: h_sdf(x, y))
        under = 100 * (atlas - sdf_peak) / max(atlas, 1e-9)
        rows.append(dict(seed=sd, atlas_recon_pct=rmse / band_limited_rough_surface(
            np.random.RandomState(9).uniform(-1, 1, 3000), np.random.RandomState(8).uniform(-1, 1, 3000),
            amp=AMP, seed=sd).std() * 100,
            atlas_peak=atlas, sdf_peak=sdf_peak, sdf_underprediction_pct=under))
        print(f"  seed {sd}: atlas mu_app={atlas:.3f}  SDF={sdf_peak:.3f}  -> SDF under-predicts {under:.0f}%")
    under = np.array([r["sdf_underprediction_pct"] for r in rows])
    res = dict(rows=rows, under_mean=float(under.mean()), under_std=float(under.std()),
               atlas_mean=float(np.mean([r["atlas_peak"] for r in rows])),
               sdf_mean=float(np.mean([r["sdf_peak"] for r in rows])))
    json.dump(res, open(os.path.join(RUN, "results.json"), "w"), indent=2)
    print(f"\n  ROBUSTNESS ({len(seeds)} realizations): SDF under-predicts the emergent dilatancy by "
          f"{res['under_mean']:.0f}% ± {res['under_std']:.0f}%  (atlas {res['atlas_mean']:.3f} vs SDF {res['sdf_mean']:.3f})")
    print(f"  saved {os.path.join(RUN, 'results.json')}")


if __name__ == "__main__":
    main()
