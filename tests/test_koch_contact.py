"""V&V tests for the Koch-snowflake fractal contact chart (CV-6).

Covers: exact IFS geometry, inside-test vs brute-force, the O(depth) resolution-independent
cost (the headline advantage vs a 9^n SDF grid), the (verified) loss of star-shapedness at
level>=2, repulsive contact-normal sign, contact-arc growth with depth, and momentum
conservation of the rigid-body contact engine.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.contact import koch


def test_ifs_geometry_exact():
    for n in range(5):
        V = koch.snowflake_vertices(n)
        assert len(V) == koch.n_segments(n) + 1          # 3*4^n + 1 (closed)
        assert np.allclose(V[0], V[-1])                  # closed loop
    assert abs(koch.perimeter(3) / koch.perimeter(0) - (4 / 3) ** 3) < 1e-9


def test_inside_matches_brute_force():
    Vb = koch.snowflake_vertices(4)

    def brute(x):
        c = False; j = len(Vb) - 1
        for i in range(len(Vb)):
            xi, yi = Vb[i]; xj, yj = Vb[j]
            if ((yi > x[1]) != (yj > x[1])) and \
               (x[0] < (xj - xi) * (x[1] - yi) / (yj - yi) + xi):
                c = not c
            j = i
        return c

    rng = np.random.RandomState(0)
    X = rng.uniform(-1.2, 1.2, size=(600, 2))
    assert sum(koch.inside_cost(x, 4)[0] != brute(x) for x in X) == 0


def test_resolution_independent_cost():
    """The pruned inside-query cost is ~constant in depth (<< 4^n, vs SDF 9^n)."""
    near = np.array([1.03 * v for v in koch.snowflake_vertices(2)[:-1]])
    cost = {n: np.mean([koch.inside_cost(x, n)[1] for x in near]) for n in (2, 6, 10)}
    assert cost[10] < 80                                  # bounded, not exponential
    assert cost[10] < 0.01 * 4 ** 10                      # vastly below the segment count
    assert cost[10] < koch.sdf_grid_cells(10)             # and below the SDF grid


def test_not_star_shaped_from_level_2():
    """Documented fact: star-shaped only at levels 0-1 (so no radial inverse-chart gap)."""
    def star(level):
        V = koch.snowflake_vertices(level); c = V[:-1].mean(0)
        th = np.unwrap(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))
        d = np.diff(th)
        return np.all(d > 0) or np.all(d < 0)
    assert star(1)
    assert not star(2)
    assert not star(4)


def test_contact_normal_repulsive():
    """The returned normal must be the gap-ascent (outward) direction for EVERY penetrating
    point -- including the ~12% whose nearest foot is a polyline VERTEX (spike/corner), where
    a bare segment normal is wrong. Sample the whole interior on a grid so vertex-foot cases
    are exercised (the previous single-sample test gave false confidence)."""
    level = 4
    gx = np.linspace(-0.9, 0.9, 41)
    interior = [np.array([a, b]) for a in gx for b in gx if koch.inside(np.array([a, b]), level)]
    assert len(interior) > 200                            # the grid really lands inside
    h = 1e-4
    n_vertex_foot = 0
    for p in interior:
        g0, foot, n = koch.nearest_boundary(p, level)
        assert g0 < 0.0                                   # interior => negative gap
        assert abs(np.linalg.norm(n) - 1.0) < 1e-9        # unit normal
        g1, _f, _n = koch.nearest_boundary(p + h * n, level)
        assert g1 > g0 - 1e-12                            # +n is gap-ascent (repulsive) everywhere
        # count vertex-foot cases: foot near a polyline vertex (the bug's failure region)
        if np.min(np.linalg.norm(koch.snowflake_vertices(level) - foot, axis=1)) < 1e-6:
            n_vertex_foot += 1
    assert n_vertex_foot > 0                              # the failure region is actually covered
    # an exterior point has positive gap
    assert koch.nearest_boundary(np.array([2.0, 0.0]), level)[0] > 0.0


def test_nearest_boundary_distance_exact():
    """The signed-gap MAGNITUDE must equal the true polyline distance (regression: a pruning
    bug once squared a negative |x-mid|-rbound and skipped the nearest segment, overestimating
    interior distances ~20x while keeping the SIGN correct -- which the sign/repulsion tests
    could not catch)."""
    level = 4
    V = koch.snowflake_vertices(level)
    segP, segQ = V[:-1], V[1:]

    def brute_dist(x):
        d = segQ - segP
        L2 = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", x - segP, d) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        foot = segP + t[:, None] * d
        return float(np.min(np.linalg.norm(x - foot, axis=1)))

    rng = np.random.RandomState(3)
    X = rng.uniform(-1.05, 1.05, size=(800, 2))
    err = np.array([abs(abs(koch.nearest_boundary(x, level)[0]) - brute_dist(x)) for x in X])
    assert err.max() < 1e-12                              # exact nearest-point on the polyline


def test_contact_arcs_grow_with_depth():
    def arcs(level, dB=1.7):
        B = koch.snowflake_vertices(level, center=[dB, 0.0])[:-1]
        ins = np.array([koch.inside_cost(p, level, center=[0, 0])[0] for p in B], dtype=int)
        return int(np.sum((ins - np.roll(ins, 1)) == 1)) or (1 if ins.any() else 0)
    assert arcs(4) >= arcs(2) >= 1                        # more fractal detail -> more contacts


def test_engine_conserves_momentum():
    """Free-free control: the rigid-body contact engine conserves linear momentum."""
    from benchmarks.contact.koch_gears_drive import simulate
    hist, _ = simulate(free_A=True, level=2, n_steps=400)
    assert max(h["linmom_err"] for h in hist) < 1e-8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
