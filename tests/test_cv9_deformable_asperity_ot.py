"""CV-9b verification: deformable elastic asperity in OT measure-coupling contact (Track T4).

Fast, self-contained (numpy + scipy FEM; no torch / trained .pt / downloads).  The asperity body is
ELASTIC (Tri2D block with a cosine bump on its contact face) pressed by a rigid flat platen; contact
is the OT measure coupling, recomputed on the DEFORMED surface every Newton iteration.  Asserts:

  * Newton CONVERGES (relative residual -> small) each load step;
  * FORCE BALANCE: the integrated contact load equals the clamped-boundary reaction (machine-zero);
  * the OT gap field UPDATES with deformation -- the contact half-width a(delta) grows MONOTONICALLY
    and the surface genuinely displaces (not the rigid baseline);
  * elastic resistance: the real contact a is SMALLER than the naive rigid (undeformed) geometric
    overlap a_rigid at every step (a deformable asperity flattens and resists, redistributing load);
  * SMALL-LOAD Hertz: the first increment follows the 2-D line-contact law a = 2 sqrt(P' R_tip/(pi E*))
    for the parabolic tip (within coarse-mesh tolerance).

Coarse mesh keeps the run < ~30 s; the publication mesh runs on Euler (see the driver docstring).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("scipy")

from benchmarks.contact.cv_numerical import cv9_deformable_asperity_ot as cv9b  # noqa: E402


@pytest.fixture(scope="module")
def asperity():
    return cv9b.run(W=1.5, D=1.0, A_a=0.10, lam=1.6, n_x=50, n_y=26, E=1000.0, nu=0.3,
                    delta=0.04, n_steps=5, max_iter=40, verbose=False)


def test_newton_converges(asperity):
    assert asperity["res_rel_max"] < 1e-4, asperity["res_rel_max"]


def test_force_balance_machine_zero(asperity):
    # integrated contact load == clamped-top reaction (Newton's third law / global equilibrium).
    assert asperity["balance_max"] < 1e-6, asperity["balance_max"]


def test_ot_gap_updates_with_deformation(asperity):
    # the contact half-width grows monotonically AND the surface actually displaced -> the OT gap
    # field is being recomputed on the deformed config (the deformable signature).
    assert asperity["monotone_growth"] is True
    assert asperity["surf_disp_max"] > 1e-4, asperity["surf_disp_max"]
    a = np.asarray(asperity["a_seq"])
    assert np.all(np.diff(a) > -1e-9), a


def test_elastic_resistance_below_rigid_overlap(asperity):
    # an elastic asperity flattens and resists -> real contact a < naive rigid geometric overlap.
    assert asperity["deformable_below_rigid"] is True
    a = np.asarray(asperity["a_seq"])
    a_rigid = np.asarray(asperity["a_rigid_seq"])
    assert np.all(a <= a_rigid + 1e-9)


def test_small_load_hertz(asperity):
    # first increment follows the 2-D Hertz line-contact law for the parabolic tip (coarse tol).
    assert asperity["hertz_relerr_first"] < 0.30, asperity["hertz_relerr_first"]
