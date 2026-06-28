"""V&V for CV-8: two-body deformable OT contact via the CLOSEST-POINT transition map.

Fast suite (pure numpy + scipy; no torch / no .pt) covering:
  * ClosestPointCoupling1D: orthogonal foot, partition-of-unity host weights, band mass-screen.
  * assemble_two_body_contact(correspondence="closest_point"): exact force balance (Newton's 3rd
    law), symmetric SPSD 4-block tangent, tangent == FD df/du under frozen geometry.
  * the CV-8 patch test still passes (monotone full-contact path NOT regressed).
  * the CV-8 deformable Hertz localizes to a CENTRAL patch (a/W ~ 0.23) and matches the analytical
    line-contact half-width a and peak pressure p0 to the acceptance gates (<10% / <12%).
  * the large-sliding patch centroid tracks the platen MONOTONICALLY (no saltation).

The Hertz / sliding solves use a coarse-but-decisive mesh so the test stays fast (~10-20 s); the full
mesh-convergence sweep is the benchmark driver (``cv8_deformable_ot.py --mode cylinder``).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.contact.measure_coupling.coupling import ClosestPointCoupling1D  # noqa: E402
from solvers.contact.measure_coupling.two_body import assemble_two_body_contact  # noqa: E402


# ----------------------------------------------------------------------------------------------
#  ClosestPointCoupling1D
# ----------------------------------------------------------------------------------------------
def test_closest_point_orthogonal_foot_and_partition_of_unity():
    mx = np.linspace(0.0, 1.0, 11)
    master = dict(x=mx, h=np.zeros_like(mx), hp=np.zeros_like(mx))
    sx = np.linspace(0.0, 1.0, 7)
    slave = dict(x=sx, h=np.full_like(sx, -0.01), hp=np.zeros_like(sx))
    cp = ClosestPointCoupling1D(slave, master, contact_band=0.05)
    Xm, seg, N0, N1, mass = cp.map_full(np.array([0.23, 0.5, 0.77]))
    assert np.allclose(Xm[:, 0], [0.23, 0.5, 0.77], atol=1e-9)     # vertical foot on a flat master
    assert np.allclose(N0 + N1, 1.0, atol=1e-12)                   # host weights are a PoU
    assert np.all(mass == 1.0)                                     # 0.01 within band 0.05


def test_closest_point_band_mass_screen():
    mx = np.linspace(0.0, 1.0, 11)
    master = dict(x=mx, h=np.zeros_like(mx), hp=np.zeros_like(mx))
    sx = np.linspace(0.0, 1.0, 7)
    far = dict(x=sx, h=np.full_like(sx, -0.2), hp=np.zeros_like(sx))  # 0.2 below master
    cp = ClosestPointCoupling1D(far, master, contact_band=0.05)
    _, _, _, _, mass = cp.map_full(np.array([0.5]))
    assert mass[0] == 0.0                                          # out of band -> no mass


# ----------------------------------------------------------------------------------------------
#  assemble_two_body_contact(correspondence="closest_point")
# ----------------------------------------------------------------------------------------------
def test_two_body_closest_point_force_balance_and_spsd():
    pen, eps_n = 0.02, 500.0
    sxy = np.array([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0]])
    sid = np.array([0, 1, 2])
    mxy = np.array([[0.0, pen], [0.55, pen], [1.0, pen]])         # non-matching node spacing
    mid = np.array([3, 4, 5])
    f, Kc, diag = assemble_two_body_contact(sxy, sid, mxy, mid, 12, eps_n, order=3,
                                            correspondence="closest_point")
    pN = eps_n * pen
    assert abs(diag["force_balance"]) < 1e-10                     # Newton's 3rd law (net = 0)
    assert abs(diag["F_line"] - pN * 1.0) < 1e-10
    assert abs(f[[0, 1, 2], 1].sum() - pN) < 1e-10               # slave carries +F
    assert abs(f[[3, 4, 5], 1].sum() + pN) < 1e-10              # master carries -F
    Kd = Kc.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)                      # symmetric
    assert np.linalg.eigvalsh(Kd).min() > -1e-9                   # SPSD


def test_two_body_closest_point_tangent_equals_fd():
    eps_n = 500.0
    sx = np.array([[0.0, 0.0], [0.37, 0.0], [0.71, 0.0], [1.0, 0.0]])
    sid = np.array([0, 1, 2, 3])
    mx = np.column_stack([np.array([0.0, 0.42, 0.83, 1.0]),
                          0.015 + 0.01 * np.array([0.0, 0.42, 0.83, 1.0]) ** 2])  # curved master
    mid = np.array([4, 5, 6, 7])
    N = 8
    ndof = 2 * N
    base = np.vstack([sx, mx])

    from solvers.contact.measure_coupling.quadrature import gauss_legendre_1d
    from solvers.contact.measure_coupling.traction import TractionField
    from solvers.contact.measure_coupling.two_body import _profile

    order = 3
    xi_g, w_g = gauss_legendre_1d(order); ss = 0.5 * (1 + xi_g); Nref = np.stack([1 - ss, ss], 1)
    sprof = _profile(sx); sec = np.sqrt(1 + sprof["hp"] ** 2)
    ns = np.column_stack([-sprof["hp"] / sec, 1 / sec])
    cpc = ClosestPointCoupling1D(sprof, _profile(mx))
    frozen = []
    for k in range(3):
        for q in range(order):
            Nq = Nref[q]; P0, P1 = sx[k], sx[k + 1]; L = np.linalg.norm(P1 - P0)
            wq = w_g[q] * 0.5 * L
            nq = Nq[0] * ns[k] + Nq[1] * ns[k + 1]; nq = nq / np.linalg.norm(nq)
            xs0 = Nq[0] * P0 + Nq[1] * P1
            _Xm, seg, Nm0, Nm1, _m = cpc.map_full(np.array([xs0[0]]))
            frozen.append((k, q, wq, nq, Nq, int(seg[0]), float(Nm0[0]), float(Nm1[0])))

    def resid_frozen(u):
        cur = base + u.reshape(N, 2); f = np.zeros((N, 2)); tr = TractionField(eps_n)
        for (k, q, wq, nhat, Nq, jm, Nm0, Nm1) in frozen:
            xs = Nq[0] * cur[k] + Nq[1] * cur[k + 1]
            xm = Nm0 * cur[4 + jm] + Nm1 * cur[4 + jm + 1]
            gN = (xs - xm) @ nhat
            t = tr.evaluate(np.array([gN]), nhat[None, :])["t"][0]
            f[k] += wq * Nq[0] * t; f[k + 1] += wq * Nq[1] * t
            f[4 + jm] += -wq * Nm0 * t; f[4 + jm + 1] += -wq * Nm1 * t
        return -f.reshape(-1)

    _, Kc0, _ = assemble_two_body_contact(base[:4], sid, base[4:], mid, ndof, eps_n, order=3,
                                          correspondence="closest_point")
    K_ana = Kc0.toarray()
    h = 1e-7
    K_fd = np.zeros((ndof, ndof))
    u0 = np.zeros(ndof)
    for j in range(ndof):
        up = u0.copy(); up[j] += h
        um = u0.copy(); um[j] -= h
        K_fd[:, j] = (resid_frozen(up) - resid_frozen(um)) / (2 * h)
    scale = max(np.abs(K_ana).max(), np.abs(K_fd).max(), 1e-30)
    rel = np.abs(K_ana - K_fd).max() / scale
    assert rel < 1e-6, rel


# ----------------------------------------------------------------------------------------------
#  CV-8 driver-level gates (patch NOT regressed; Hertz localizes + matches; sliding tracks)
# ----------------------------------------------------------------------------------------------
def test_cv8_patch_test_not_regressed():
    from benchmarks.contact.cv_numerical.cv8_deformable_ot import patch_test
    pt = patch_test(verbose=False)
    assert pt["coupling_pou_err"] < 1e-10, pt["coupling_pou_err"]
    assert pt["coupling_net_resultant"] < 1e-8, pt["coupling_net_resultant"]
    assert pt["coupling_transmit_err"] < 1e-8, pt["coupling_transmit_err"]


def test_cv8_deformable_hertz_localizes_and_matches():
    from benchmarks.contact.cv_numerical.cv8_deformable_ot import hertz_test
    r = hertz_test(n_lower=(128, 16), n_upper=(128, 16), R=2.0, delta=0.02, n_load=6,
                   slide=0.0, n_slide=0, verbose=False)
    # localizes to a CENTRAL Hertz patch (NOT smeared across the whole interface)
    assert 0.15 < r["aW"] < 0.35, r["aW"]
    # half-width and peak pressure match the analytical line contact
    assert r["a_relerr"] < 0.10, r["a_relerr"]
    assert r["p0_relerr"] < 0.12, r["p0_relerr"]
    # exact two-body force balance (Newton's 3rd law)
    assert r["force_balance"] < 1e-10, r["force_balance"]


def test_cv8_large_sliding_centroid_monotone():
    from benchmarks.contact.cv_numerical.cv8_deformable_ot import hertz_test
    r = hertz_test(n_lower=(128, 16), n_upper=(128, 16), R=2.0, delta=0.02, n_load=6,
                   slide=0.06, n_slide=6, verbose=False)
    assert r["slide_centroid_monotone"], r["slide_hist"]
    assert r["slide_F_cov"] < 0.05, r["slide_F_cov"]             # normal load stays ~constant


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
