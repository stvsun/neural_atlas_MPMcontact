"""T1 — tests for the neural-chart OT verification driver (cv_neural_chart_verify).

Two layers, mirroring the driver:
  * L1 OT-path patch test (training-independent, MUST pass exactly): the consistent mortar
    assembly transmits a uniform pressure / exact line load across non-uniform nodes, and is
    well-formed (symmetric Kc) on a NEURAL-radial-sampled superformula bottom arc.
  * L0 neural-vs-analytical accuracy (fast quick-mode training): the trained NeuralRho2D radial
    chart reaches the documented CV-5 L0 floor (gap RMSE/L < 6e-3, normal median < 2 deg).  The
    disc SDF L0 floor needs the heavy Euler run, so quick-mode only asserts the probe-ring
    pipeline runs and the field is sane (not the publication 7e-3 floor).

These are fast (radial chart trains in ~2 s quick); the disc SDF (~12 s quick) is exercised but its
publication L0 is gated on the full/Euler run emitted by the driver.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.contact.cv_numerical import cv_neural_chart_verify as cv  # noqa: E402


def test_ot_flat_facet_patch_exact():
    """The OT measure-coupling patch test is machine-exact, independent of node spacing."""
    out = cv._flat_facet_patch_test()
    assert out["PASS"], out
    assert out["uniformity"] < 1e-10
    assert out["F_relerr"] < 1e-10
    assert out["total_force_abserr"] < 1e-9


def test_neural_radial_L0_and_ot_patch():
    """Train the neural radial chart (quick) and verify L0 floor + OT-path consistency."""
    from atlas.charts.train_radial_chart import train_superformula_radial
    from solvers.contact import supershape as ss

    model, m = train_superformula_radial(width=48, depth=3, epochs=1200, n_train=2500,
                                          n_eval=2000, save=False, verbose=False)
    # L0: documented CV-5 radial-chart floor (passes even at quick training budget)
    assert m["gap_rmse_rel"] < 6e-3, m
    assert m["normal_angle_median_deg"] < 2.0, m

    # L1: OT patch test on the NEURAL-sampled bottom arc -> consistent mortar assembly
    params = ss.SuperParams(**m["params"])
    patch = cv._neural_radial_ot_patch(model, params)
    assert patch["n_active"] >= 8, patch                  # a meaningful contact band, not a point
    assert patch["Kc_symmetry"] < 1e-9, patch             # OT tangent symmetric (SPSD structure)
    assert patch["F_consistent"] > 0.0
    # mortar-consistent and node-lumped line loads DIFFER on a curved arc with varying penetration
    # (this is the genuine mortar-vs-lumping effect, not a target); just sanity-bound it finite.
    assert 0.0 <= patch["lumped_vs_consistent_relerr"] < 1.0, patch


@pytest.mark.slow
def test_disc_sdf_L0_pipeline():
    """Disc SDF probe-ring L0 runs and is sane (publication 7e-3 floor needs the Euler run)."""
    from atlas.sdf.train_analytical_sdf import train_shape_sdf

    model, _ = train_shape_sdf("disc", width=64, depth=4, epochs=900, n_near=2500,
                               n_bulk=800, n_eval=1200, save=False, verbose=False)
    L0 = cv._disc_sdf_L0(model, R=1.0)
    # in-plane field is sane: planar (small |n_z|), normals roughly radial, finite gap error
    assert L0["max_abs_nz"] < 0.25, L0
    assert L0["normal_angle_median_deg"] < 5.0, L0
    assert np.isfinite(L0["gap_rmse_rel"]) and L0["gap_rmse_rel"] < 0.1, L0
