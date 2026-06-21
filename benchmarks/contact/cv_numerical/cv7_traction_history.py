#!/usr/bin/env python3
"""Global shear-vs-normal traction history of the genuine rough-joint shear, for several meshes.

Runs the genuine one-block CNL shear (`rock_joint_decoder_shear.run_shear`) at each n_cells and saves
the GLOBAL traction history — shear traction tau(u), normal traction sigma_n(u), mobilized friction
mu_app(u), dilation(u) — so the macroscopic shear-normal response and its MESH CONVERGENCE can be
plotted (all meshes should overlay).

Run:  python3 benchmarks/contact/cv_numerical/cv7_traction_history.py --cells 6,8,10,12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import run_shear              # noqa: E402

RUN = os.path.join(_ROOT, "runs", "cv7_refinement")
torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="6,8,10,12")
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--n_inc", type=int, default=21)            # finer load steps for smooth curves
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    out = os.path.join(RUN, "traction_history.json")
    cells = [int(c) for c in args.cells.split(",")]

    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=args.amp)     # noqa: E731
    print(f"=== train decoder (amp={args.amp}) ===", flush=True)
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=args.iters)
    print(f"  recon RMSE {rmse:.3e}", flush=True)

    res = json.load(open(out)) if os.path.exists(out) else {}
    for nc in cells:
        if str(nc) in res:
            print(f"  (n_cells={nc} done, skip)", flush=True); continue
        js, hist = run_shear(dec, dk, protocol="CNL", sigma_n=2.0, shear_total=0.16, n_inc=args.n_inc,
                             n_cells=nc, mu=args.mu, surf_amp=args.amp, eps_n=None, W=2.0 * (2.0) ** 2,
                             verbose=False)
        res[str(nc)] = dict(n_cells=nc, n_elem=int(js.solver.n_elements),
                            u=np.asarray(hist["u_x"]).tolist(), tau=np.asarray(hist["tau"]).tolist(),
                            sigma_n=np.asarray(hist["sigma_n"]).tolist(),
                            mu_app=np.asarray(hist["mu_app"]).tolist(),
                            dilation=np.asarray(hist["dilation"]).tolist())
        json.dump(res, open(out, "w"), indent=2)                # progressive
        print(f"  n_cells={nc:2d} ({js.solver.n_elem if hasattr(js.solver,'n_elem') else js.solver.n_elements} tets): "
              f"peak tau={max(np.abs(hist['tau'])):.3f}, sigma_n~{np.mean(hist['sigma_n']):.3f} -> {out}", flush=True)
    print("=== traction-history capture complete ===", flush=True)


if __name__ == "__main__":
    main()
