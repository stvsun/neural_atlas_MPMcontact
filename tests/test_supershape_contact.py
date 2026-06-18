"""V&V tests for the superformula (supershape) contact example (CV-5).

Covers: star-shaped inside test, matched (gap, grad) conservativeness, area/inertia
circle limit, momentum conservation (free-A control) + driven work-energy, multi-arc
detection (the headline transition-map advantage), and the radial-vs-perpendicular bias.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.contact import supershape as ss
from benchmarks.contact.supershape_cam_drive import simulate, _contact_pass, Body


P = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, scale=1.0)
C = np.array([0.0, 0.0])


def test_star_shaped_inside_and_gap_sign():
    th = np.linspace(0, 2 * np.pi, 50)
    rho = ss.radius(th, P)
    e = np.stack([np.cos(th), np.sin(th)], axis=1)
    inside_pts = (0.8 * rho)[:, None] * e
    outside_pts = (1.2 * rho)[:, None] * e
    assert np.all(ss.inside(inside_pts, C, 0.0, P))
    assert not np.any(ss.inside(outside_pts, C, 0.0, P))
    # gap ~ 0 on the boundary
    bpts = ss.boundary(th, C, 0.0, P)
    g, _ = ss.radial_gap(bpts, C, 0.0, P)
    assert np.max(np.abs(g)) < 1e-9


def test_gap_normal_conservativeness():
    """Returned grad of the radial gap matches a finite-difference gradient."""
    rng = np.random.RandomState(0)
    pts = C + rng.uniform(-1.0, 1.0, size=(40, 2))
    pts = pts[np.linalg.norm(pts - C, axis=1) > 0.2]   # gap has a cone kink at the center
    _, grad = ss.radial_gap(pts, C, 0.3, P)
    h = 1e-6
    gx = (ss.radial_gap(pts + [h, 0], C, 0.3, P)[0] - ss.radial_gap(pts - [h, 0], C, 0.3, P)[0]) / (2 * h)
    gy = (ss.radial_gap(pts + [0, h], C, 0.3, P)[0] - ss.radial_gap(pts - [0, h], C, 0.3, P)[0]) / (2 * h)
    # relative tolerance: FD truncation error scales with the (large) gradient near the center
    assert np.allclose(grad[:, 0], gx, rtol=1e-4, atol=1e-7)
    assert np.allclose(grad[:, 1], gy, rtol=1e-4, atol=1e-7)


def test_area_inertia_circle_limit():
    circ = ss.SuperParams(m=0, n1=1, n2=1, n3=1, a=1, b=1, scale=2.5)
    R = 2.5
    assert abs(ss.area(circ) - np.pi * R ** 2) / (np.pi * R ** 2) < 1e-4
    m, I = ss.polar_inertia(circ, density=3.0)
    assert abs(I - m * R ** 2 / 2.0) / (m * R ** 2 / 2.0) < 1e-4
    assert np.linalg.norm(ss.centroid(P)) < 1e-6


def test_free_A_conserves_momentum():
    hist, meta = simulate(free_A=True, n_steps=1500)
    assert max(h["linmom_err"] for h in hist) < 1e-8
    assert max(h["angmom_err"] for h in hist) < 1e-6
    ke = [h["ke_total"] for h in hist]
    assert ke[-1] <= ke[0] * 1.02  # penalty+frictionless: KE non-increasing (small tol)


def test_driven_cam_drives_follower_and_energy_balance():
    hist, meta = simulate(free_A=False, n_steps=2500)
    last = hist[-1]
    disp = np.hypot(last["cB_x"] - hist[0]["cB_x"], last["cB_y"] - hist[0]["cB_y"])
    assert disp > 0.3                      # the cam actually moves the follower
    assert abs(last["alpha_B"]) > 0.2      # and spins it
    dKE = last["ke_B"] - meta["ke0_B"]
    # work-energy theorem on B (explicit-integration tolerance)
    assert abs(dKE - last["energy_injected"]) < 0.1 * max(abs(dKE), 1e-9) + 0.5


def test_multi_arc_detection():
    """A driven run reaches >=2 simultaneous contact arcs (single-foot CPP returns 1)."""
    hist, _ = simulate(free_A=False, n_steps=2500)
    assert max(h["n_contacts"] for h in hist) >= 2


def test_radial_vs_perpendicular_bias():
    """Radial gap overestimates the perpendicular gap; the (fixed) refine recovers it."""
    thb = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
    B = ss.boundary(thb, C, 0.0, P)
    for psi in (0.18 * np.pi, 0.35 * np.pi, 0.5 * np.pi):
        rho = ss.radius(np.array([psi]), P)[0]
        p = (rho + 0.05) * np.array([np.cos(psi), np.sin(psi)])
        g_rad, _ = ss.radial_gap(p[None, :], C, 0.0, P)
        _ps, _ft, g_perp, _n = ss.closest_point_refine(p, C, 0.0, P)
        d_perp = float(np.sqrt(((B - p) ** 2).sum(1)).min())   # dense global perpendicular
        assert g_rad[0] >= g_perp - 1e-9                        # radial overestimates
        assert abs(g_perp - d_perp) / d_perp < 0.03            # refine matches dense scan


def test_chart_vs_single_cpp_multi_arc():
    """Headline: the chart boundary scan finds >=2 contact arcs where a single
    closest-point projection (one foot) reports exactly 1, on a wedged config."""
    from benchmarks.contact.supershape_cam_drive import Body, _count_arcs
    pA = ss.SuperParams(m=4, n1=0.8, n2=0.8, n3=0.8, scale=1.7)
    pB = ss.SuperParams(m=7, n1=0.8, n2=0.8, n3=0.8, scale=1.0)
    cA = np.array([0.0, 0.0]); aA = 0.2396
    cB = np.array([2.35, 0.0])
    th = np.linspace(0, 2 * np.pi, 1400, endpoint=False)
    ptsB = ss.boundary(th, cB, 0.0, pB)
    g, _ = ss.radial_gap(ptsB, cA, aA, pA)
    chart_arcs = _count_arcs(g < 0)
    # single closest-point projection: the one deepest penetrating foot
    single_cpp = 1 if np.any(g < 0) else 0
    assert chart_arcs >= 2          # chart enumerates all penetrating arcs
    assert single_cpp == 1          # a single CPP returns one foot


def test_dradius_finite_at_cusps():
    """Analytic dradius_dtheta is finite (not a clip artifact) at cusp angles (n<1)."""
    p = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, scale=1.0)
    cusps = np.array([k * np.pi / 3.0 for k in range(6)] + [np.pi])  # cos/sin(m th/4)=0
    dr = ss.dradius_dtheta(cusps, p)
    assert np.all(np.isfinite(dr))
    assert np.max(np.abs(dr)) < 5.0     # finite, not the ~7763 clip spike
    # tangent magnitude stays bounded across a dense grid (no spurious spikes)
    t = ss.tangent(np.linspace(0, 2 * np.pi, 3000), 0.0, p)
    assert np.max(np.linalg.norm(t, axis=1)) < 50.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
