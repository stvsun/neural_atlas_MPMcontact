"""Focused regression test for the CV-2 OT gap-field driver (Cattaneo-Mindlin + large-def).

Part A: the OT tangential traction FIELD on the exact half-space must recover the closed-form
Cattaneo stick radius and traction field (sub-percent), matching cv2b_cattaneo_field.
Part B: the large-deformation gap-field update must produce a MONOTONE outward shift of the
deformed contact radius with the approach delta/R (the frozen small-strain Cattaneo cannot).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.contact.cv_numerical import cv2_ot_gap as drv  # noqa: E402


def test_partA_ot_tangential_field_matches_cattaneo():
    A = drv.run_partA(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400)
    # stick radius and tangential traction field vs the closed form
    assert A["c_relerr"] < 0.02, f"stick radius err {A['c_relerr']}"
    assert A["field_L2_interior"] < 0.05, f"tangential field L2 {A['field_L2_interior']}"
    assert A["mean_c_relerr"] < 0.02, f"c/a law sweep mean err {A['mean_c_relerr']}"


def test_partA_subgrid_debias_beats_round0():
    # the sub-grid (midpoint-bracket) stick radius + finer mesh must beat the round-0 0.41% floor.
    A400 = drv.run_partA(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400)
    A800 = drv.run_partA(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=800)
    assert A400["mean_c_relerr"] < 0.0041, (
        f"sub-grid debias at n=400 should beat round-0 0.41%: {A400['mean_c_relerr']}")
    assert A800["mean_c_relerr"] < A400["mean_c_relerr"], "finer mesh should reduce the error"
    assert A800["mean_c_relerr"] < 0.0020, f"n=800 mean c/a err {A800['mean_c_relerr']}"


def test_partB_frozen_estimate_monotone_outward():
    # the round-0 FROZEN single-pass lateral-swell estimate stretches the contact radius OUTWARD,
    # growing with delta/R (kept as a labeled first-order estimate).
    B = drv.run_partB(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400)
    shifts = [row["a_shift_frozen_pct"] for row in B]
    assert all(s > 0.0 for s in shifts), f"frozen a-shift must be positive (outward): {shifts}"
    assert shifts == sorted(shifts), f"frozen a-shift must grow with delta/R: {shifts}"
    assert shifts[-1] > 10.0, f"finite delta/R must give a sizeable frozen shift: {shifts}"


def test_partB_converged_fixedpoint_self_consistent():
    # the UPGRADE: the deformed-config fixed point converges, and the self-consistent contact radius
    # collapses the spurious frozen swell back toward the indenter-set radius (|shift| << frozen).
    B = drv.run_partB(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400)
    for row in B:
        assert row["ld_fixedpoint_converged"], f"fixed point must converge: {row}"
        assert row["ld_fixedpoint_res"] < 1e-6, f"fixed-point residual {row['ld_fixedpoint_res']}"
        assert abs(row["a_shift_pct"]) < abs(row["a_shift_frozen_pct"]), (
            "converged shift must be smaller than the frozen single-pass estimate")
        # the OT normal gap field deepens (more penetration) with delta/R
    gN = [row["gN_min_largedef"] for row in B]
    assert gN == sorted(gN, reverse=True), f"gap field must deepen with delta/R: {gN}"


def test_partB_largedef_vs_smallstrain_ca_drops():
    B = drv.run_partB(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400)
    # the small-strain (frozen-geometry) c/a is constant; the genuine large-def c/a drops with
    # delta/R from the deformed pressure redistribution (the emergent Cattaneo signature).
    ss = [row["c_over_a_smallstrain"] for row in B]
    ld = [row["c_over_a_largedef"] for row in B]
    assert max(ss) - min(ss) < 0.02, f"small-strain c/a should be ~constant: {ss}"
    assert ld[-1] < ld[0], f"large-def c/a should drop with delta/R: {ld}"


def test_old_vs_new_ot_beats_lumped():
    m, _ = drv.run(R=1.0, Estar=1.0, F=0.02, mu=0.5, n=400, verbose=False)
    new_err = m["old_vs_new"]["new_OT_mean_c_relerr"]
    old = m["old_vs_new"]["old_lumped_fem"]
    assert new_err < 0.02, f"OT field mean c/a err {new_err}"
    if old:  # the lumped baseline metrics.json is present
        assert new_err < old["mean_c_relerr"], (
            f"OT ({new_err}) should beat lumped FEM ({old['mean_c_relerr']})")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
