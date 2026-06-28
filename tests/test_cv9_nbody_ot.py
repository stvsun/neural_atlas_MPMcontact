"""CV-9a verification: N-body elastic disc array with mutual OT measure-coupling contact (Track T4).

Fast, self-contained (numpy + scipy FEM; no torch / trained .pt / downloads).  Asserts the genuine
multi-body OT contact recovers the equibiaxial closed form and the exact conservation laws:

  * the centre disc of a 3x3 array, equibiaxially confined by four overlapped neighbours, develops the
    closed-form MEAN compression  (sigma_xx + sigma_yy)/2 = -2 N / (pi R t)
    (``postprocessing.contact_fields.nine_disc_unit_cell_field``) to < 3 %, where N is the per-face
    OT contact force (an OUTPUT of the solve, never prescribed);
  * GLOBAL force balance: the sum of every contact nodal force is machine-zero (Newton's third law is
    enforced exactly by the OT-mortar pushforward of the slave reaction onto the master rim);
  * a single isolated disc (no pairs) carries NO contact force and zero stress (sanity baseline).

The per-component D4 anisotropy (~10 %) is a documented concentric-ring-mesh artifact, NOT checked
here; the physically meaningful equibiaxial MEAN is the closed-form target.  Coarse meshes keep the
run < ~30 s; the publication mesh runs on Euler (see the driver docstring).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("scipy")

from benchmarks.contact.cv_numerical import cv9_nbody_array_ot as cv9a  # noqa: E402


@pytest.fixture(scope="module")
def array_3x3():
    # coarse but converged: 3x3, n_rings=8, equibiaxial overlap load
    return cv9a.run(n_discs=3, n_rings=8, overlap=0.025, eps_n=2500.0, max_iter=80,
                    relax=0.7, n_steps=5, arc_half=0.30, verbose=False)


def test_global_force_balance_machine_zero(array_3x3):
    # Newton's third law: every disc-disc reaction is equal-and-opposite by the OT-mortar pushforward.
    assert array_3x3["global_balance"] < 1e-9, array_3x3["global_balance"]


def test_centre_disc_equibiaxial_mean_matches_closed_form(array_3x3):
    m = array_3x3
    # mean (sxx+syy)/2 vs -2N/(pi R t); N is the measured per-face OT contact force.
    assert m["center_mean_relerr"] < 0.03, (m["center_mean_fem"], m["center_exact"],
                                            m["center_mean_relerr"])
    # the stress is genuinely compressive (equibiaxial confinement, not tension)
    assert m["center_sxx_fem"] < 0.0 and m["center_syy_fem"] < 0.0
    # N emerged positive (confining) from the OT field
    assert m["N_per_face"] > 0.0


def test_centre_disc_has_four_loaded_faces(array_3x3):
    # the interior disc of a 3x3 is confined on all four sides -> four OT contact faces.
    assert len(array_3x3["centre_face_N"]) == 4, array_3x3["centre_face_N"]


def test_single_disc_no_contact_no_stress():
    # baseline: one disc has no neighbour pairs -> zero contact force, zero stress, fast convergence.
    m = cv9a.run(n_discs=1, n_rings=8, overlap=0.04, eps_n=2000.0, max_iter=40,
                 relax=0.7, n_steps=2, verbose=False)
    assert m["n_pairs"] == 0
    assert abs(m["center_sxx_fem"]) < 1e-6 and abs(m["center_syy_fem"]) < 1e-6
    assert m["global_balance"] < 1e-12


def test_pair_third_law_exact():
    # a single disc-disc OT pair: the integrated slave and master contact forces are equal-and-opposite
    # to machine precision (the OT-mortar pushforward of the reaction).
    from solvers.contact.measure_coupling import TractionField
    A = cv9a.Disc(np.array([0.0, 0.0]), 1.0, 8, 1000.0, 0.25, 1.0)
    B = cv9a.Disc(np.array([2.0, 0.0]), 1.0, 8, 1000.0, 0.25, 1.0)
    tr = TractionField(2000.0)
    uA = np.zeros((A.sol.n_nodes, 2))
    uB = np.zeros((B.sol.n_nodes, 2))
    res = cv9a._pair_forces(A, B, np.array([1.0, 0.0]), 0.30, uA, uB, tr, 3, pen_offset=0.04)
    assert res is not None
    fA2, fB2, slaveA_ids, masterB_ids, Kss, Ksm, Kmm = res
    total = fA2.sum(axis=0) + fB2.sum(axis=0)
    assert np.linalg.norm(total) < 1e-10, total
    # the contact is repulsive: A (left) pushed in -x, B (right) pushed in +x
    assert fA2.sum(axis=0)[0] < 0.0 and fB2.sum(axis=0)[0] > 0.0
    # the FULL 4-block tangent assembled on the shared (A,B) dof system is symmetric and SPSD:
    # [[K_ss, K_sm],[K_sm^T, K_mm]] (frictionless mortar -> consistent, symmetric, SPSD).
    from scipy.sparse import bmat
    Kpair = bmat([[Kss, Ksm], [Ksm.T, Kmm]]).toarray()
    assert np.allclose(Kpair, Kpair.T, atol=1e-10), "pair tangent must be symmetric"
    assert np.linalg.eigvalsh(Kpair).min() > -1e-8, "pair tangent must be SPSD"
    # the cross block is non-trivial: the master-coupling D_IK block is actually present.
    assert abs(Ksm).sum() > 0.0, "K_sm (slave-master coupling) must be non-zero"
