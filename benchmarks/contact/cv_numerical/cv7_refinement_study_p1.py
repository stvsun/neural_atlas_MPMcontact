#!/usr/bin/env python3
"""Mesh-refinement (convergence) study of the GENUINE TWO-BLOCK rough-joint shear (P1 problem).

The two-deformable-block analogue of `cv7_refinement_study.py`.  Both rough decoders are trained ONCE
(geometry is mesh-independent); the two-block chart-FEM is then solved at increasing n_cells, recording
the emergent peak mu_app / dilation, the (harder) two-block FRICTION residual, and the chart det J on
BOTH blocks — and dumping the per-element von Mises field of BOTH deformable blocks for the 3-D contour.
Results + field dumps are written after EACH level (progressive plotting).

HONEST: the two-deformable moving-master node-to-surface friction converges far worse than the one-block
P2 case (resid ~1-6%, not 1e-9) — the refinement shows whether the emergent quantities still converge.

Run:  python3 benchmarks/contact/cv_numerical/cv7_refinement_study_p1.py --cells 6,8,10,12
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
from benchmarks.contact.cv_numerical.rock_joint_two_block import (   # noqa: E402
    TwoBlockJointShear, _make_blocks, run_shear)

RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1")
torch.set_default_dtype(torch.float64)


def _von_mises(solver, u_block, stress_fn):
    F = solver.compute_F(torch.from_numpy(u_block.reshape(solver.n_nodes, 3)))
    sig = stress_fn(F).detach().numpy()
    s = 0.5 * (sig + np.transpose(sig, (0, 2, 1)))
    tr = (s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]) / 3.0
    dev = s - tr[:, None, None] * np.eye(3)[None]
    return np.sqrt(1.5 * (dev ** 2).sum((1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="6,8,10,12")
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--mode", default="in_plane")
    ap.add_argument("--n_inc", type=int, default=13)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    out = os.path.join(RUN, "results.json")
    cells = [int(c) for c in args.cells.split(",")]

    # train BOTH decoders once (mesh-independent) by building the coarsest block set
    print(f"=== training the two rough-block decoders (amp={args.amp}) ===", flush=True)
    _, info = _make_blocks(amp=args.amp, n_cells=cells[0], mu=args.mu, iters=args.iters, verbose=True)
    decL, dkL, decU, dkU = info["decL"], info["dkL"], info["decU"], info["dkU"]
    E = 2.0e3

    results = json.load(open(out)) if os.path.exists(out) else []
    done = {r["n_cells"] for r in results
            if os.path.exists(os.path.join(RUN, f"field_n{r['n_cells']}.npz"))}
    results = [r for r in results if r["n_cells"] in done]
    for nc in cells:
        if nc in done:
            print(f"  (n_cells={nc} done, skip)", flush=True); continue
        t0 = time.time()
        js = TwoBlockJointShear(decL, dkL, decU, dkU, E=E, mu=args.mu, n_cells=nc, surf_amp=args.amp)
        _, _, pay = run_shear(mode=args.mode, protocol="CNL", amp=args.amp, n_cells=nc, mu=args.mu, E=E,
                              shear_total=0.16, n_inc=args.n_inc, js=js, info=info, verbose=False)
        hist = pay["history"]
        U = js.u_final.reshape(-1, 3); uL = U[:js.NL]; uU = U[js.NL:]
        vmL = _von_mises(js.solL, uL.reshape(-1), js.stress_fn)
        vmU = _von_mises(js.solU, uU.reshape(-1), js.stress_fn)
        np.savez(os.path.join(RUN, f"field_n{nc}.npz"),
                 nodes_L=js.pL, u_L=uL, elem_L=js.solL.elements.numpy(), vm_L=vmL,
                 nodes_U=js.pU, u_U=uU, elem_U=js.solU.elements.numpy(), vm_U=vmU,
                 n_cells=nc, n_elem=int(js.solL.n_elements + js.solU.n_elements))
        resid_rel = np.asarray(hist["resid_rel"]) if "resid_rel" in hist else \
            (np.asarray(hist["resid"]) / max(js.eps_n * js.A_slave, 1e-30))
        rec = dict(n_cells=nc, h=2.0 / nc, n_elem=int(js.solL.n_elements + js.solU.n_elements),
                   n_dof=int(js.ndof), peak_mu_app=float(np.abs(hist["mu_app"]).max()),
                   total_dilation=float(hist["dilation"][-1]),
                   max_resid_rel=float(resid_rel.max()), mean_resid_rel=float(resid_rel.mean()),
                   detJ_min=float(min(js.solL.geom_detJ.min(), js.solU.geom_detJ.min())),
                   recon_rmse=info["rmse_L"], runtime_s=time.time() - t0)
        results.append(rec); results = sorted(results, key=lambda r: r["n_cells"])
        json.dump(results, open(out, "w"), indent=2)
        print(f"  n_cells={nc:2d} ({rec['n_elem']:6d} tets, {rec['n_dof']:6d} DOF): "
              f"peak_mu={rec['peak_mu_app']:.4f} dil={rec['total_dilation']:+.4f} "
              f"detJmin={rec['detJ_min']:.3f} resid={rec['max_resid_rel']:.1e} "
              f"t={rec['runtime_s']:.0f}s  -> {out}", flush=True)
    print(f"=== P1 refinement study complete ({len(results)} levels) ===", flush=True)


if __name__ == "__main__":
    main()
