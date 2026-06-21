#!/usr/bin/env python3
"""Mesh-refinement (convergence) study of the genuine rough-joint shear (P2 problem).

A single trained rough-block decoder (the geometry is mesh-independent) is solved at increasing chart-FEM
resolutions n_cells; for each mesh we record the emergent peak apparent friction, total dilation, the
Newton residual, and the worst chart-Jacobian det — so the convergence of the EMERGENT quantities (and
the well-posedness of the chart-FEM) can be tracked as h -> 0.  Results are appended to a JSON after EACH
level so a progressive convergence plot can be made while the finer (slower) meshes are still running.

Run:  python3 benchmarks/contact/cv_numerical/cv7_refinement_study.py --cells 6,8,10,12,14,16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
    ap.add_argument("--cells", default="6,8,10,12,14,16")
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--n_inc", type=int, default=13)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    out = os.path.join(RUN, "results.json")
    cells = [int(c) for c in args.cells.split(",")]

    # train the decoder ONCE — the rough geometry is the same for every mesh (mesh-independent chart)
    print(f"=== training the rough-block decoder (amp={args.amp}) ===", flush=True)
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=args.amp)     # noqa: E731
    rng = np.random.RandomState(1)
    surf_rms = float(np.std(tgt(rng.uniform(-1, 1, 8000), rng.uniform(-1, 1, 8000))))
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=args.iters)
    print(f"  decoder recon RMSE {rmse:.3e} ({100*rmse/surf_rms:.2f}% of RMS {surf_rms:.4f})", flush=True)

    results = json.load(open(out)) if os.path.exists(out) else []
    # a level is "done" only if BOTH its scalar record AND its 3-D field dump exist
    done = {r["n_cells"] for r in results
            if os.path.exists(os.path.join(RUN, f"field_n{r['n_cells']}.npz"))}
    results = [r for r in results if r["n_cells"] in done]
    for nc in cells:
        if nc in done:
            print(f"  (n_cells={nc} already done, skip)", flush=True); continue
        t0 = time.time()
        js, hist = run_shear(dec, dk, protocol="CNL", sigma_n=2.0, shear_total=0.16, n_inc=args.n_inc,
                             n_cells=nc, mu=args.mu, surf_amp=args.amp, eps_n=None, W=2.0 * (2.0) ** 2,
                             verbose=False)
        # --- dump the 3-D von Mises stress FIELD at the final shear state (for the 3-D contour) ---
        U = js.u_final.reshape(js.N, 3)
        F = js.solver.compute_F(torch.from_numpy(U))
        sig = js.stress_fn(F).detach().numpy()                       # (M,3,3) small-strain Cauchy
        s = 0.5 * (sig + np.transpose(sig, (0, 2, 1)))
        trc = (s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]) / 3.0
        dev = s - trc[:, None, None] * np.eye(3)[None]
        vm = np.sqrt(1.5 * (dev ** 2).sum((1, 2)))                   # per-element von Mises
        np.savez(os.path.join(RUN, f"field_n{nc}.npz"),
                 nodes_ref=js.xyz0, u=U, elements=js.solver.elements.numpy(), vm=vm,
                 n_cells=nc, n_elem=js.solver.n_elements, surf_amp=args.amp, z_p=js.z_p_final)
        resid_rel = (np.asarray(hist["resid"]) / max(js.eps_n * js.A_top, 1e-30))
        rec = dict(n_cells=nc, h=2.0 / nc, n_elem=int(js.solver.n_elements), n_nodes=int(js.N),
                   n_dof=int(3 * js.N), peak_mu_app=float(np.abs(hist["mu_app"]).max()),
                   total_dilation=float(hist["dilation"][-1]),
                   final_mu_app=float(hist["mu_app"][-1]),
                   max_resid_rel=float(resid_rel.max()), mean_resid_rel=float(resid_rel.mean()),
                   detJ_min=float(js.solver.geom_detJ.min()), detJ_max=float(js.solver.geom_detJ.max()),
                   recon_rmse=rmse, surf_rms=surf_rms, runtime_s=time.time() - t0)
        results.append(rec)
        results = sorted(results, key=lambda r: r["n_cells"])
        json.dump(results, open(out, "w"), indent=2)                      # append-after-each (progressive)
        print(f"  n_cells={nc:2d} ({rec['n_elem']:6d} tets, {rec['n_dof']:6d} DOF): "
              f"peak_mu={rec['peak_mu_app']:.4f} dil={rec['total_dilation']:+.4f} "
              f"detJmin={rec['detJ_min']:.3f} resid={rec['max_resid_rel']:.1e} "
              f"t={rec['runtime_s']:.0f}s  -> {out}", flush=True)
    print(f"=== refinement study complete ({len(results)} levels) -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
