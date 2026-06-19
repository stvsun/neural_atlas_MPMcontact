#!/usr/bin/env python3
"""CV-5 dynamics — NEURAL transition-map detection drives the same rigid-body contact dynamics
as the analytical chart (plan M6).

The superformula cam-drive (`benchmarks/contact/supershape_cam_drive.py`) detects contact via the
analytical inverse radial gap.  Here we swap in TRAINED neural radial charts (NeuralRho2D, from
`atlas/charts/train_radial_chart.py`) for both bodies and run the same integrator, then check that
the neural-detection trajectory matches the analytical-detection trajectory and that linear/angular
momentum are conserved in the free-free control.  This is CV-5's L1: the neural chart reproduces the
contact MECHANICS, not just the static gap/normal (L0, test_cv5_supershape_neural_radial_chart_L0).

Run:  python3 benchmarks/contact/cv_numerical/cv5_cam_neural.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from benchmarks.contact import supershape_cam_drive as cam           # noqa: E402
from atlas.charts.train_radial_chart import load_trained_radial_chart  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR = os.path.join(_ROOT, "runs", "cv5_cam_neural")


def run(n_steps=500, free_A=True, verbose=True):
    chA = load_trained_radial_chart("cam_m4")      # m=4 cam radial chart
    chB = load_trained_radial_chart("cam_m7")      # m=7 follower radial chart
    if chA is None or chB is None:
        raise RuntimeError("train cam charts first: atlas/charts/train_radial_chart.py (cam_m4, cam_m7)")

    # free-free control (frictionless): the clean momentum-conservation regime
    h_ana, _ = cam.simulate(free_A=free_A, n_steps=n_steps)
    h_neu, _ = cam.simulate(free_A=free_A, n_steps=n_steps, charts=(chA, chB))

    fa, fn = h_ana[-1], h_neu[-1]
    L = 2.45                                        # initial separation scale
    traj_pos_err = float(np.hypot(fa["cB_x"] - fn["cB_x"], fa["cB_y"] - fn["cB_y"]) / L)
    traj_ang_err = float(abs(fa["alpha_B"] - fn["alpha_B"]))
    # contact-set agreement over the run
    nc_match = float(np.mean([a["n_contacts"] == n["n_contacts"] for a, n in zip(h_ana, h_neu)]))
    m = {
        "n_steps": n_steps, "free_A": free_A,
        "final_cB_ana": [fa["cB_x"], fa["cB_y"]], "final_cB_neu": [fn["cB_x"], fn["cB_y"]],
        "traj_pos_relerr": traj_pos_err, "traj_angle_err_rad": traj_ang_err,
        "n_contacts_match_frac": nc_match,
        "linmom_err_neu": float(h_neu[-1].get("linmom_err", float("nan"))),
        "angmom_err_neu": float(h_neu[-1].get("angmom_err", float("nan"))),
    }
    if verbose:
        print(f"  CV-5 cam dynamics: neural vs analytical detection ({n_steps} steps, free_A={free_A})")
        print(f"    final B position: ana=({fa['cB_x']:.4f},{fa['cB_y']:.4f})  "
              f"neu=({fn['cB_x']:.4f},{fn['cB_y']:.4f})  rel-err={traj_pos_err*100:.3f}%")
        print(f"    final B angle err: {traj_ang_err:.2e} rad   contact-count match: {nc_match*100:.1f}%")
        print(f"    neural-run momentum: lin {m['linmom_err_neu']:.2e}, ang {m['angmom_err_neu']:.2e}")
    return m


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    m = run()
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    ok = (m["traj_pos_relerr"] < 0.02 and m["n_contacts_match_frac"] > 0.9
          and m["linmom_err_neu"] < 1e-8)
    print(f"\n  CV-5 neural-dynamics vs analytical: {'PASS' if ok else 'CHECK'} "
          f"(trajectory match < 2%, momentum conserved)")


if __name__ == "__main__":
    main()
