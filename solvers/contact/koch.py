"""Koch-snowflake fractal contact chart (CV-6): exact recursive boundary + O(depth) detection.

The Koch snowflake is defined by an iterated function system (IFS): each edge is replaced
by 4 sub-edges (scale 1/3) with an outward bump, recursively. This is an EXACT,
resolution-independent description (the IFS is O(1) storage) that can be evaluated to any
depth on demand. We use it as the analytical coordinate chart for a fractal contact body.

The CV-6 point, on the honest axes: a uniform SDF grid must resolve the smallest feature
L*3^-n EVERYWHERE -> (3^n)^2 = 9^n cells (an adaptive/narrow-band SDF needs ~O(4^n) boundary
cells -- smaller, still exponential), and both bake in a maximum resolution. The recursive
chart stores the IFS (O(1) storage) and the inside/outside DETECTION test (`inside_cost`)
prunes to O(depth) per query (measured ~21 nodes, bounded in n), refining to ANY depth on
demand. A precomputed SDF is actually CHEAPER per query (one lookup); the chart's wins are
STORAGE and RESOLUTION-INDEPENDENCE, not per-query speed. (A fixed-capacity neural SDF is
likewise resolution-capped -- it cannot keep refining self-similar detail without growing its
parameters. That ceiling is now MEASURED, not just argued: a single fixed-width SDFNet trained
on this exact signed distance shows its gap RMSE and Eikonal residual rise with fractal level n
-- see benchmarks/contact/koch_neural_ceiling.py, figures/koch_neural_ceiling_pub.png, and the
manual's CV-6 section / §11.6.)

Cost caveat: the EXACT signed gap + normal (`nearest_boundary`) is more expensive than
detection -- the descent visits the boundary detail clustered near the query, which on a fractal
grows ~O(3^n) for a fixed-distance point (intrinsic: ~3^n tiny segments lie near any boundary
point). It is exact (verified == brute-force polyline distance) and is evaluated only at the
penetrating points an inside-test already flagged, at a finite practical level (n=3: ~110 nodes).

Honest scope (verified):
- The Koch snowflake is star-shaped only at levels 0-1; from level 2 it is NOT (outward
  bumps on near-radial spike flanks make rho(theta) multivalued). So there is NO radial
  inverse-chart gap here (unlike the supershape, CV-5) -- contact uses the PARAMETRIC chart
  with a pruned recursive inside-test / nearest-boundary descent.
- At the true fractal limit there is no tangent/normal; contact mechanics is well-posed only
  at a finite pre-fractal level n (n may be large for the chart, vs grid-memory-bounded SDF).
- "Analytical chart" here = exact recursive (piecewise-linear at level n), not smooth C^1.

Pure numpy (matches solvers/contact/supershape.py).
"""
from __future__ import annotations

import numpy as np

_R60 = np.array([[np.cos(-np.pi / 3), -np.sin(-np.pi / 3)],
                 [np.sin(-np.pi / 3), np.cos(-np.pi / 3)]])   # -60 deg => outward bump (CCW)
_INV_SQRT3 = 1.0 / np.sqrt(3.0)


def n_segments(level):
    """Boundary segments at level n: 3 * 4^n."""
    return 3 * 4 ** level


def perimeter(level, R=1.0):
    """Snowflake perimeter: base 3*sqrt(3)*R times (4/3)^n  (-> infinity)."""
    base = 3.0 * np.sqrt(3.0) * R           # side of equilateral triangle, circumradius R
    return base * (4.0 / 3.0) ** level


def sdf_grid_cells(level):
    """Uniform SDF grid cells to resolve level n (spacing < L*3^-n): (3^n)^2 = 9^n."""
    return 9.0 ** level


# --- canonical snowflake (centroid at origin, circumradius R, orientation 0) ---

def _base_vertices(R=1.0):
    angs = np.deg2rad([90.0, 210.0, 330.0])                 # CCW equilateral triangle
    return [R * np.array([np.cos(a), np.sin(a)]) for a in angs]


def _koch_curve(p, q, level):
    """Vertices of the Koch curve on segment p->q at *level* (excludes q)."""
    if level == 0:
        return [np.asarray(p, float)]
    p = np.asarray(p, float); q = np.asarray(q, float)
    d = q - p
    a = p + d / 3.0
    b = p + 2.0 * d / 3.0
    apex = a + _R60 @ (b - a)
    return (_koch_curve(p, a, level - 1) + _koch_curve(a, apex, level - 1)
            + _koch_curve(apex, b, level - 1) + _koch_curve(b, q, level - 1))


def snowflake_vertices(level, R=1.0, center=(0.0, 0.0), alpha=0.0):
    """Closed boundary polyline (M=3*4^level + 1, last==first), in world frame."""
    V = _base_vertices(R)
    pts = []
    for i in range(3):
        pts += _koch_curve(V[i], V[(i + 1) % 3], level)
    pts.append(V[0])
    P = np.array(pts)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rm = np.array([[ca, -sa], [sa, ca]])
    return (P @ Rm.T) + np.asarray(center)


# --- geometric primitives ---

def _in_triangle(x, A, B, C):
    d1 = (x[0] - B[0]) * (A[1] - B[1]) - (A[0] - B[0]) * (x[1] - B[1])
    d2 = (x[0] - C[0]) * (B[1] - C[1]) - (B[0] - C[0]) * (x[1] - C[1])
    d3 = (x[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (x[1] - A[1])
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def _to_body(x, center, alpha):
    d = np.asarray(x, float) - np.asarray(center)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([d[0] * ca + d[1] * sa, -d[0] * sa + d[1] * ca])   # R(-alpha)(x-c)


# --- O(depth) recursive inside-test (pruned by per-node bounding circle) ---

def _edge_bump_inside(x, p, q, level, cost):
    """True if x is inside the outward Koch bump region on edge p->q (body frame)."""
    cost[0] += 1
    if level == 0:
        return False
    mid = 0.5 * (p + q)
    r = np.linalg.norm(q - p) * _INV_SQRT3            # Koch curve fits in this circle
    if np.linalg.norm(x - mid) > r:                   # PRUNE: x not near this edge's bumps
        return False
    d = q - p
    a = p + d / 3.0
    b = p + 2.0 * d / 3.0
    apex = a + _R60 @ (b - a)
    if _in_triangle(x, a, apex, b):
        return True
    return (_edge_bump_inside(x, p, a, level - 1, cost)
            or _edge_bump_inside(x, a, apex, level - 1, cost)
            or _edge_bump_inside(x, apex, b, level - 1, cost)
            or _edge_bump_inside(x, b, q, level - 1, cost))


def inside_cost(x, level, R=1.0, center=(0.0, 0.0), alpha=0.0):
    """(inside?, nodes_visited) for a single point. nodes_visited ~ O(level) by pruning."""
    xb = _to_body(x, center, alpha)
    cost = [0]
    V = _base_vertices(R)
    if _in_triangle(xb, V[0], V[1], V[2]):
        return True, cost[0]
    for i in range(3):
        if _edge_bump_inside(xb, V[i], V[(i + 1) % 3], level, cost):
            return True, cost[0]
    return False, cost[0]


def inside(x, level, R=1.0, center=(0.0, 0.0), alpha=0.0):
    """Boolean inside-test (single point or (N,2) array)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return inside_cost(x, level, R, center, alpha)[0]
    return np.array([inside_cost(xi, level, R, center, alpha)[0] for xi in x])


# --- nearest boundary segment (pruned recursive descent) -> penetration + normal ---

def _nearest_edge(x, p, q, level, best):
    """Descend edge p->q, updating best=[dist2, foot, seg_dir] for the nearest leaf segment."""
    mid = 0.5 * (p + q)
    half = 0.5 * np.linalg.norm(q - p)
    rbound = half * (1.0 + 2.0 * _INV_SQRT3)          # circle containing this Koch node
    # prune: skip iff the node's bounding circle lies entirely farther than the current best.
    # The min distance from x to the circle is max(0, |x-mid| - rbound); only prune when x is
    # OUTSIDE the circle (|x-mid| > rbound) -- else squaring a negative would falsely prune a
    # node that may CONTAIN the nearest segment (the bug that overestimated interior distances).
    dmid = np.linalg.norm(x - mid)
    if best[0] < np.inf and dmid > rbound and (dmid - rbound) ** 2 > best[0]:
        return
    if level == 0:
        # leaf segment p->q: closest point
        d = q - p
        L2 = d @ d
        t = 0.0 if L2 == 0 else np.clip((x - p) @ d / L2, 0.0, 1.0)
        foot = p + t * d
        dist2 = (x - foot) @ (x - foot)
        if dist2 < best[0]:
            best[0] = dist2; best[1] = foot; best[2] = d
        return
    dd = q - p
    a = p + dd / 3.0
    b = p + 2.0 * dd / 3.0
    apex = a + _R60 @ (b - a)
    for (pp, qq) in ((p, a), (a, apex), (apex, b), (b, q)):
        _nearest_edge(x, pp, qq, level - 1, best)


def nearest_boundary(x, level, R=1.0, center=(0.0, 0.0), alpha=0.0):
    """Signed gap (negative inside), foot point, and outward unit normal at the nearest
    boundary segment. Uses the parametric chart (no radial assumption)."""
    xb = _to_body(x, center, alpha)
    best = [np.inf, None, None]
    V = _base_vertices(R)
    for i in range(3):
        _nearest_edge(xb, V[i], V[(i + 1) % 3], level, best)
    dist = np.sqrt(best[0])
    foot_b = best[1]
    seg = best[2]
    ins, _ = inside_cost(x, level, R, center, alpha)
    g = -dist if ins else dist                        # signed gap (negative inside)
    # Outward unit normal = gradient of the signed distance: sign-corrected (x - foot)/|x - foot|.
    # On a segment INTERIOR this equals the perpendicular segment normal; at a VERTEX foot
    # (spike tip / corner) it is the TRUE gap-ascent direction, where the bare segment normal
    # is wrong. Sign: outward for exterior (+(x-foot)), outward for interior (foot-x), so the
    # penalty force eps_n<-g> n always pushes a penetrating point OUT (verified by test).
    diff = (foot_b - xb) if ins else (xb - foot_b)    # outward (gap-ascent) direction, body frame
    nrm = np.linalg.norm(diff)
    if nrm < 1e-12:                                   # x exactly on the boundary (g~0): segment normal
        n_b = np.array([seg[1], -seg[0]])
        n_b = n_b / np.clip(np.linalg.norm(n_b), 1e-12, None)
    else:
        n_b = diff / nrm
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rm = np.array([[ca, -sa], [sa, ca]])
    foot = Rm @ foot_b + np.asarray(center)
    n = Rm @ n_b
    return g, foot, n


def _self_test():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("Koch snowflake chart self-test:")
    for n in range(5):
        V = snowflake_vertices(n)
        chk(f"level {n}: vertex count = 3*4^n+1 = {n_segments(n)+1}", len(V) == n_segments(n) + 1)
    # perimeter grows as (4/3)^n
    p2 = perimeter(2) / perimeter(0)
    chk("perimeter ratio P2/P0 = (4/3)^2", abs(p2 - (4 / 3) ** 2) < 1e-9)
    # inside-test vs brute-force even-odd at level 4
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.2, 1.2, size=(400, 2))
    Vb = snowflake_vertices(4)
    def brute(x):
        c = False; j = len(Vb) - 1
        for i in range(len(Vb)):
            xi, yi = Vb[i]; xj, yj = Vb[j]
            if ((yi > x[1]) != (yj > x[1])) and (x[0] < (xj - xi) * (x[1] - yi) / (yj - yi) + xi):
                c = not c
            j = i
        return c
    mism = sum(inside_cost(x, 4)[0] != brute(x) for x in X)
    chk(f"inside vs brute-force (level 4): {mism}/400 mismatches", mism == 0)
    # O(depth) cost: mean nodes visited grows ~linearly, NOT 4^n
    print("  cost (mean nodes visited per inside query, near boundary):")
    near = np.array([1.02 * v for v in snowflake_vertices(2)[:-1:7]])
    for n in (2, 4, 6, 8, 10):
        costs = [inside_cost(x, n)[1] for x in near]
        print(f"    level {n:2d}: mean nodes={np.mean(costs):6.1f}   (4^n={4**n:>8d}, 9^n={9**n:.1e})")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    _self_test()
