"""V&V for the measure-coupling field-traction contact (solvers/contact/measure_coupling/).

Fast suite (pure numpy + scipy; no torch / no .pt checkpoints) covering the module gates:
  * quadrature partition-of-unity + exact integration
  * consistent assembly self-test (force split, mass-matrix coupling, SPSD tangent)
  * CONTACT PATCH TEST: uniform pressure transmitted exactly across non-uniform nodes (~1e-12)
  * Sinkhorn -> monotone coupling consistency (the two couplings agree as eps -> 0)
  * half-space BEM cross-check: TractionField penalty == closed-form Hertz (isolates coupling math)
  * 2-D line Hertz FEM: pressure FIELD matches the half-ellipse; a(F)-E* law
  * 3-D axisymmetric Hertz field; Cattaneo-Mindlin tangential field + stick-radius law
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.contact.measure_coupling import quadrature, assemble_contact, TractionField  # noqa: E402
from solvers.contact.measure_coupling.coupling import measure_coupling_compare              # noqa: E402
from solvers.contact.measure_coupling import halfspace_bem                                  # noqa: E402


def test_quadrature_partition_of_unity_and_exact_integral():
    N, _ = quadrature.lagrange_p1(np.array([-1.0, -0.3, 0.5, 1.0]))
    assert np.allclose(N.sum(axis=1), 1.0)
    xs = np.array([0.0, 0.5, 2.0, 3.0])               # non-uniform segments, total length 3
    q = quadrature.segment_quadrature(np.column_stack([xs, np.zeros_like(xs)]), order=3)
    assert abs(q["wds"].sum() - 3.0) < 1e-12
    # integrate f(x)=2x+1 over [0,3] = 12
    assert abs((q["wds"] * (2.0 * q["Xq"][:, 0] + 1.0)).sum() - 12.0) < 1e-10


def test_assembly_consistent_force_and_tangent():
    eps_n, pen0 = 500.0, 0.02
    surf = np.array([[0.0, 0.0], [1.0, 0.0]])

    def eval_gap(X):
        return np.full(len(X), -pen0), np.tile([0.0, 1.0], (len(X), 1))

    f, Kc, diag = assemble_contact(surf, np.array([0, 1]), 4, eval_gap, TractionField(eps_n))
    pN = eps_n * pen0
    assert np.allclose(f.sum(axis=0), [0.0, pN * 1.0])            # total = p*L*n
    assert np.allclose(f[0], f[1])                               # 50/50 split
    Kd = Kc.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)                     # symmetric
    yy = Kd[np.ix_([1, 3], [1, 3])]
    assert np.allclose(yy, eps_n * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]]), atol=1e-10)
    assert np.linalg.eigvalsh(Kd).min() > -1e-12                 # SPSD


def test_contact_patch_test_uniform_pressure_exact():
    from benchmarks.contact.cv_numerical.cv1b_hertz_field import patch_test
    res = patch_test(verbose=False)
    assert res["uniformity"] < 1e-9, res                         # pressure uniform
    assert res["force_relerr"] < 1e-10, res                      # line load exact
    assert res["total_force_abserr"] < 1e-9, res                 # nodal force sum exact
    assert res["PASS"]


def test_sinkhorn_converges_to_monotone():
    x = np.linspace(0.0, 10.0, 161)
    slave = dict(x=x, h=0.3 * np.sin(0.7 * x), hp=0.21 * np.cos(0.7 * x))
    master = dict(x=x, h=0.2 * np.sin(0.7 * x + 0.4) - 0.1, hp=0.14 * np.cos(0.7 * x + 0.4))
    rows = measure_coupling_compare(slave, master, np.linspace(1, 9, 41),
                                    eps_list=(0.1, 0.03, 0.01))
    assert min(r[2] for r in rows) < 0.01                       # best relerr < 1%


def test_halfspace_bem_coupling_vs_closed_form():
    res = halfspace_bem.cross_check(R=1.0, Estar=1.0, P=0.01, n=601, verbose=False)
    assert res["L2_lcp_vs_hertz"] < 0.01                        # BEM reproduces Hertz
    assert res["L2_penalty_vs_hertz"] < 0.01                    # TractionField penalty too
    assert res["L2_coupling_vs_bem"] < 0.01                     # coupling == exact LCP (<1%)


def test_hertz_field_2d_fem():
    from benchmarks.contact.cv_numerical.cv1b_hertz_field import run
    m, _, _ = run(n_x=90, n_y=45, verbose=False)
    assert m["converged"]
    assert m["field_L2_interior"] < 0.06                        # pressure FIELD matches half-ellipse
    assert m["a_relerr"] < 0.06                                 # a(F)-E* law


def test_hertz_field_3d_axisymmetric():
    from benchmarks.contact.cv_numerical.cv1c_hertz3d_field import run
    m, _ = run(n=300, verbose=False)
    assert m["delta_relerr"] < 0.02                            # approach delta = a^2/R
    assert m["field_L2_interior_penalty"] < 0.05               # 3-D pressure field
    assert abs(m["a_pen"] - m["a_lcp"]) < 1e-6                 # penalty == LCP


def test_cattaneo_tangential_field():
    from benchmarks.contact.cv_numerical.cv2b_cattaneo_field import run
    m, _ = run(n=300, verbose=False)
    assert m["field_L2_interior"] < 0.10                       # tangential traction field
    assert m["mean_c_relerr"] < 0.05                           # stick-radius law c/a=(1-Q/muF)^(1/3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
