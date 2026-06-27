"""Focused checks for the CV-5 OT-gap driver (benchmarks/contact/cv_numerical/cv5_ot_gap.py).

The transition-map chart resolves the superformula cusps that an ambient grid SDF smooths, and the
consistent (mortar-structured) OT assembly passes the contact patch test on the charted surface.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact.cv_numerical import cv5_ot_gap as cv5  # noqa: E402


def test_cv5_ot_gap_runs_and_passes():
    metrics, _ = cv5.run(verbose=False)
    A = metrics["part_A_chart_vs_sdf"]
    # NEW: refined chart map is ~exact and grid-independent
    assert A["chart_refine"]["rms"] < 1e-3
    assert A["chart_normal_cusp_deg"] < 0.1            # chart cusp normal essentially exact
    # OLD vs NEW: the ambient SDF degrades materially at the cusps (the CV-5 advantage)
    assert A["sdf_over_chart_max_coarse"] > 5.0        # coarse-SDF max gap err >> chart
    assert A["sdf_normal_cusp_over_chart_coarse"] > 3.0
    # PART B: the consistent OT assembly passes the contact patch test on the charted surface
    assert metrics["part_B_patch_test"]["PASS"]
    assert metrics["status"] == "pass"


def test_cv5_patch_test_uniform_pressure_exact_load():
    out = cv5._patch_clean(eps_n=1.0e4, delta=0.01)
    assert out["uniformity"] < 1e-3                    # uniform pressure across non-uniform nodes
    assert out["F_relerr"] < 1e-10                     # exact line load (no tributary over-count)
    assert out["PASS"]


def test_cv5_round1_polish_beats_round0():
    """ROUND-1: the Brent closest-point polish drives the chart gap RMS from the round-0 1.18e-4
    (coarse equispaced scan + single parabolic step) to the float64 floor -- the chart is analytically
    exact; the round-0 error was purely the 1-D scan discretization."""
    metrics, _ = cv5.run(verbose=False, polish=True)
    A = metrics["part_A_chart_vs_sdf"]
    assert A["improved"]                                  # measured lower than round 0
    assert A["improved_rms"] < A["round0_rms"]
    assert A["improved_rms"] < 1.18e-4                    # below the round-0 number to beat
    assert A["improved_rms"] < 1e-8                       # at the analytic/float64 floor
    assert A["improvement_factor"] > 1.0e4               # orders of magnitude, not noise
    # A/B: with polish disabled the driver reproduces the round-0 RMS
    m0, _ = cv5.run(verbose=False, polish=False)
    assert abs(m0["part_A_chart_vs_sdf"]["chart_refine"]["rms"] - A["round0_rms"]) < 1e-12


def test_cv5_chart_gap_independent_of_grid():
    """The refined chart gap does not change with the (absent) grid; the SDF does -- show the SDF
    monotone-ish improvement vs the flat chart line."""
    metrics, _ = cv5.run(verbose=False)
    sdf = metrics["part_A_chart_vs_sdf"]["ambient_sdf"]
    grids = sorted(sdf.keys())
    coarse = sdf[grids[0]]["rms"]
    fine = sdf[grids[-1]]["rms"]
    assert fine < coarse                               # SDF needs a finer grid to approach the chart
