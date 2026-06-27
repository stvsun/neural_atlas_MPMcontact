"""Focused checks for the CV-3 Brazilian-disc OPTIMAL-TRANSPORT gap-field driver.

The disc is pressed between two rigid flat platens; the diametral load P EMERGES from the
measure-coupling (OT) gap field + consistent ``assemble_contact`` Galerkin contact, and the centre
stress is validated against the closed form ``+2P/(pi D t)`` / ``-6P/(pi D t)``.

A modest mesh (``n_rings=40``) is used so the test runs in ~1-2 s.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact.cv_numerical import cv3_ot_gap as cv3


@pytest.fixture(scope="module")
def result():
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return cv3.run(n_rings=40, verbose=False)


def test_runs_and_converges(result):
    assert result["converged"], "load-stepped Newton did not converge"
    assert result["P_emergent"] > 0.0, "no emergent diametral load"


def test_symmetric_load(result):
    # D4 symmetry: the two platen reactions must be equal
    assert result["load_imbalance"] < 0.02, result["load_imbalance"]


def test_centre_tension_compression(result):
    # the Brazilian signature: sigma_xx TENSILE, sigma_yy COMPRESSIVE at the centre
    assert result["center_sxx_fem"] > 0.0, "centre sigma_xx must be tensile"
    assert result["center_syy_fem"] < 0.0, "centre sigma_yy must be compressive"


def test_centre_stress_vs_closed_form(result):
    assert result["center_sxx_relerr"] < 0.05, result["center_sxx_relerr"]
    assert result["center_syy_relerr"] < 0.05, result["center_syy_relerr"]


def test_centre_ratio(result):
    # sigma_yy / sigma_xx = -3 at the centre (closed form)
    assert abs(result["center_ratio_fem"] - (-3.0)) < 0.1, result["center_ratio_fem"]


def test_interior_field(result):
    assert result["field_rms_rel"] < 0.03, result["field_rms_rel"]


def test_contact_halfwidth_order(result):
    # the emergent contact half-width should be within a factor ~1.3 of the 2-D Hertz prediction
    # (the patch is only a few elements wide at this resolution, so this is a coarse bound)
    assert result["a_relerr"] < 0.20, result["a_relerr"]


def test_old_vs_new_centre(result):
    # the OT field driver should be at least as accurate as the old Neumann driver at the same P
    cmp = cv3.compare_old(result, verbose=False)
    assert cmp["new_center_sxx_relerr"] <= cmp["old_center_sxx_relerr"] + 0.01
    assert cmp["new_center_syy_relerr"] <= cmp["old_center_syy_relerr"] + 0.01
    assert not cmp["old_has_contact_halfwidth"]   # old driver has no contact -> no a(P)
