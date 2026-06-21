#!/usr/bin/env python3
"""Capture the von Mises stress FIELD at 4 loading times, for 4 mesh resolutions — feeds the 4x4
(mesh x time) 3-D contour figure of the genuine rough-joint shear.

For each n_cells the genuine one-block CNL shear is run once; 4 snapshots along the trajectory
(u_x ~ 0.04, 0.08, 0.12, 0.16) are saved (deformed block + per-element von Mises + the RIGID mating
surface state z_p, u_x), so the renderer can show the rigid top surface and the stress evolution.

Run:  python3 benchmarks/contact/cv_numerical/cv7_time_evolution_fields.py --cells 6,8,10,12
"""
from __future__ import annotations

import argparse
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


def _vm(solver, u, stress_fn):
    F = solver.compute_F(torch.from_numpy(u.reshape(solver.n_nodes, 3)))
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
    ap.add_argument("--n_inc", type=int, default=13)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    cells = [int(c) for c in args.cells.split(",")]
    picks = [3, 6, 9, args.n_inc - 1]                                # 4 loading times along the trajectory

    print(f"=== train decoder (amp={args.amp}) ===", flush=True)
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=args.amp)     # noqa: E731
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=args.iters)
    print(f"  recon RMSE {rmse:.3e}", flush=True)

    for nc in cells:
        outf = os.path.join(RUN, f"field_te_n{nc}.npz")
        if os.path.exists(outf):
            print(f"  (n_cells={nc} already captured, skip)", flush=True); continue
        js, hist = run_shear(dec, dk, protocol="CNL", sigma_n=2.0, shear_total=0.16, n_inc=args.n_inc,
                             n_cells=nc, mu=args.mu, surf_amp=args.amp, eps_n=None, W=2.0 * (2.0) ** 2,
                             verbose=False)
        traj = js.traj
        u_t = np.stack([traj[j][2].reshape(js.N, 3) for j in picks])     # (4,N,3)
        vm_t = np.stack([_vm(js.solver, traj[j][2], js.stress_fn) for j in picks])  # (4,M)
        ux_t = np.array([traj[j][0] for j in picks]); zp_t = np.array([traj[j][1] for j in picks])
        np.savez(outf, nodes_ref=js.xyz0, elements=js.solver.elements.numpy(), u_t=u_t, vm_t=vm_t,
                 ux_t=ux_t, zp_t=zp_t, surf_amp=args.amp, n_cells=nc, n_elem=js.solver.n_elements)
        print(f"  n_cells={nc:2d} ({js.solver.n_elements} tets): times u_x={np.round(ux_t,3).tolist()} "
              f"-> {outf}", flush=True)
    print("=== time-evolution capture complete ===", flush=True)


if __name__ == "__main__":
    main()
