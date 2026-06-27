"""Focused test for the CV-1 OT gap-field large-deformation driver.

Asserts, on a coarse fast mesh:
  (a) CONSISTENCY: at small load the OT closest-point gap field on a soft Neo-Hookean body recovers
      the Hertz half-width a(F) (a few percent) and the half-ellipse pressure field (interior L2);
  (b) GAP-FIELD UPDATE: at large prescribed displacement the updated-gap large-deformation NH solve
      diverges measurably from the small-strain linear-elastic solve (geometric stiffening).

Reuses the measure-coupling assembly + the contact_fields closed forms; trains nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "contact", "cv_numerical"))
from cv1_ot_gap import solve_contact, NeoHookeanCST2D                    # noqa: E402
from solvers.fem.tri2d import graded_box_mesh, Tri2DFEMSolver            # noqa: E402


def test_neo_hookean_element_consistency():
    """NH tangent at u=0 == linear elasticity; zero residual at u=0; FD-consistent tangent."""
    nodes, tris, top, bot = graded_box_mesh(0.3, 0.3, 6, 6, 1.0)
    E, nu = 1.0, 0.3
    nh = NeoHookeanCST2D(nodes, tris, E, nu)
    lin = Tri2DFEMSolver(nodes, tris, E, nu, thickness=1.0, mode="plane_strain")
    K0 = nh.tangent(np.zeros((nh.N, 2))).toarray()
    Kl = lin.assemble().toarray()
    assert np.abs(K0 - Kl).max() < 1e-9 * np.abs(Kl).max(), "NH tangent(u=0) must match linear elasticity"
    assert np.abs(nh.internal_force(np.zeros((nh.N, 2)))).max() < 1e-12, "zero residual at u=0"
    # finite-difference tangent column check
    rng = np.random.default_rng(0)
    u = 0.002 * rng.normal(size=(nh.N, 2))
    f = nh.internal_force(u).reshape(-1)
    K = nh.tangent(u).toarray()
    eps, i = 1e-6, 2 * 4 + 1
    up = u.copy(); up.reshape(-1)[i] += eps
    fd = (nh.internal_force(up).reshape(-1) - f) / eps
    assert np.abs(fd - K[:, i]).max() < 1e-3, "tangent must be FD-consistent"


def test_cv1_ot_gap_consistency_small_load():
    """Small-load OT updated-gap large-def NH recovers Hertz a(F) and the pressure field."""
    E, nu, Rc, W, D = 1.0, 0.3, 1.0, 0.6, 0.6
    n_x, n_y, grade = 80, 40, 3.0
    eps_n = 60.0 * E / (W / n_x)
    m = solve_contact(E, nu, Rc, delta=0.012, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                      eps_n=eps_n, frozen_gap=False, max_iter=40, relax=1.0, n_load=2,
                      quad_order=3, large_def=True)
    assert m["converged"], "small-load NH solve must converge"
    assert m["F_line"] > 0, "must establish contact"
    assert m["a_relerr"] < 0.08, f"a(F) within 8% of Hertz, got {m['a_relerr']*100:.2f}%"
    assert m["field_L2_interior"] < 0.12, \
        f"interior pressure field L2 < 12%, got {m['field_L2_interior']*100:.2f}%"


def test_cv1_ot_gap_large_deformation_update():
    """Large-load updated-gap NH diverges measurably from the small-strain linear solve."""
    E, nu, Rc, W, D = 1.0, 0.3, 1.0, 0.6, 0.6
    n_x, n_y, grade = 80, 40, 3.0
    eps_n = 60.0 * E / (W / n_x)
    delta = 0.10
    nh = solve_contact(E, nu, Rc, delta=delta, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                       eps_n=eps_n, frozen_gap=False, max_iter=60, relax=1.0, n_load=6,
                       quad_order=3, large_def=True)
    lin = solve_contact(E, nu, Rc, delta=delta, W=W, D=D, n_x=n_x, n_y=n_y, grade=grade,
                        eps_n=eps_n, frozen_gap=False, max_iter=60, relax=1.0, n_load=1,
                        quad_order=3, large_def=False)
    assert nh["converged"] and lin["converged"], "both large-load solves must converge"
    dF = abs(nh["F_line"] - lin["F_line"]) / lin["F_line"]
    dp0 = abs(nh["p0_fem"] - lin["p0_fem"]) / lin["p0_fem"]
    assert dF > 0.05, f"large-def gap-field update must shift load > 5%, got {dF*100:.2f}%"
    # NH geometric stiffening -> larger contact load/pressure than the small-strain solve
    assert nh["F_line"] > lin["F_line"], "Neo-Hookean stiffening should raise the contact load"
    assert dp0 > 0.03, f"peak pressure must shift > 3%, got {dp0*100:.2f}%"
