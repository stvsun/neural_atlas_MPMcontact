#!/usr/bin/env python3
"""TWO-BLOCK rough-joint shear: von Mises stress FIELD at 4 load increments x several meshes (P1).

For the mesh x time evolution figure.  Both decoders trained once; the two-block chart-FEM is solved at
each n_cells, and the per-element von Mises field of BOTH blocks is dumped at 4 shear increments
(u_x = 25,50,75,100% of the total), so a 4x4 (mesh x loading-time) contour matrix can be made.

Run:  python3 benchmarks/contact/cv_numerical/cv7_refinement_time_p1.py --cells 6,8,10,12
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

RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1_time")
torch.set_default_dtype(torch.float64)


def _vm(solver, u_block, stress_fn):
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
    ap.add_argument("--n_inc", type=int, default=13)            # 4 snapshots at j = n_inc//4*(1..4)-1
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    cells = [int(c) for c in args.cells.split(",")]
    print(f"=== training the two rough-block decoders (amp={args.amp}) ===", flush=True)
    _, info = _make_blocks(amp=args.amp, n_cells=cells[0], mu=args.mu, iters=args.iters, verbose=True)
    decL, dkL, decU, dkU = info["decL"], info["dkL"], info["decU"], info["dkU"]
    E = 2.0e3
    # the 4 increment indices (25/50/75/100% of the shear) for n_inc points spanning [0, shear_total]
    q = args.n_inc - 1
    idx4 = [round(q * f) for f in (0.25, 0.5, 0.75, 1.0)]
    for nc in cells:
        if os.path.exists(os.path.join(RUN, f"field_n{nc}.npz")):
            print(f"  (n_cells={nc} done, skip)", flush=True); continue
        t0 = time.time()
        js = TwoBlockJointShear(decL, dkL, decU, dkU, E=E, mu=args.mu, n_cells=nc, surf_amp=args.amp)
        run_shear(mode=args.mode, protocol="CNL", amp=args.amp, n_cells=nc, mu=args.mu, E=E,
                  shear_total=0.16, n_inc=args.n_inc, js=js, info=info, verbose=False)
        UL, VML, UU, VMU, times = [], [], [], [], []
        for j in idx4:
            ux, u = js.u_snaps[j]
            U = u.reshape(-1, 3); uL = U[:js.NL]; uU = U[js.NL:]
            UL.append(uL); VML.append(_vm(js.solL, uL.reshape(-1), js.stress_fn))
            UU.append(uU); VMU.append(_vm(js.solU, uU.reshape(-1), js.stress_fn))
            times.append(ux)
        np.savez(os.path.join(RUN, f"field_n{nc}.npz"),
                 nodes_L=js.pL, elem_L=js.solL.elements.numpy(), nodes_U=js.pU,
                 elem_U=js.solU.elements.numpy(),
                 U_L=np.array(UL), VM_L=np.array(VML), U_U=np.array(UU), VM_U=np.array(VMU),
                 times=np.array(times), n_cells=nc, n_elem=int(js.solL.n_elements + js.solU.n_elements))
        print(f"  n_cells={nc:2d} ({int(js.solL.n_elements + js.solU.n_elements):6d} tets): "
              f"4 snapshots at u_x={[round(t, 3) for t in times]}  t={time.time() - t0:.0f}s", flush=True)
    print(f"=== P1 time study complete ===", flush=True)


if __name__ == "__main__":
    main()
