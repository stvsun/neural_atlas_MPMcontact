#!/usr/bin/env python3
"""Compare the two surface-to-surface couplings: exact monotone (1-D OT) vs entropic Sinkhorn.

The user's "perform both and compare".  The exact :class:`MonotoneCoupling1D` (closed-form 1-D
optimal transport, bias-free) is the anchor; :class:`SinkhornCoupling1D` (entropic OT) must converge
to it as the regularization ``eps -> 0``.  This driver reports:

  1. CONSISTENCY: relative L2 error of the Sinkhorn barycentric map vs the monotone map at several
     ``eps`` (-> 0), with iteration counts (the cost/accuracy trade-off).
  2. CAPABILITY: the monotone map is 1-D-only (single-valued profiles); the Sinkhorn coupling
     generalizes to genuine 2-D surfaces (3-D contact, exercised by the rock-joint M6).

Run:  python3 benchmarks/contact/cv_numerical/measure_coupling_compare.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.contact.measure_coupling.coupling import (                  # noqa: E402
    MonotoneCoupling1D, SinkhornCoupling1D, measure_coupling_compare)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "measure_coupling_compare")


def _profiles():
    """Two mismatched wavy profiles (slave/master faces of a 2-D joint)."""
    x = np.linspace(0.0, 10.0, 201)
    slave = dict(x=x, h=0.30 * np.sin(0.70 * x), hp=0.30 * 0.70 * np.cos(0.70 * x))
    master = dict(x=x, h=0.20 * np.sin(0.70 * x + 0.4) - 0.10,
                  hp=0.20 * 0.70 * np.cos(0.70 * x + 0.4))
    return slave, master


def run(verbose=True):
    slave, master = _profiles()
    xi = np.linspace(1.0, 9.0, 81)
    rows = measure_coupling_compare(slave, master, xi,
                                    eps_list=(0.3, 0.1, 0.03, 0.01, 0.003))
    best = min(r[2] for r in rows)
    out = {"eps": [r[0] for r in rows], "iters": [r[1] for r in rows],
           "relerr_vs_monotone": [r[2] for r in rows], "best_relerr": float(best),
           "consistency_PASS": bool(best < 0.01)}
    if verbose:
        print("  Monotone (exact 1-D OT) vs Sinkhorn (entropic) — barycentric map consistency:")
        print("    eps        iters   relerr vs monotone")
        for e, n, err in rows:
            print(f"    {e:<8.3f}  {n:5d}   {err*100:7.3f}%")
        print(f"    best = {best*100:.3f}%  ->  {'PASS' if out['consistency_PASS'] else 'CHECK'} "
              f"(Sinkhorn -> monotone, <1%)")
        print("  capability: monotone is single-valued-1-D only; Sinkhorn generalizes to 2-D "
              "surfaces (3-D contact, see rock-joint M6).")
    return out


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    out = run()
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
