"""Analytical CV shapes — the single source of geometry for the NUMERICAL CV suite.

Each shape exposes a uniform interface so the three neural objects (neural SDF, neural
radial chart, ChartDecoder) and the analytical L0 references can all be driven from one place:

    shape.dim                      -> 2 or 3 (native dimension)
    shape.sdf(pts)                 -> signed distance (neg inside), native dim, numpy
    shape.sample_surface(n, seed)  -> (pts, normals) on the boundary, native dim
    shape.sample_near(n, band, s)  -> points in a |gap|~band shell around the boundary
    shape.inside(pts)              -> bool mask
    shape.body_size()              -> characteristic L (max boundary radius from centroid)

2-D shapes (Disc, Superformula, Koch) carry the IN-PLANE signed distance; the SDF trainer
lifts them to a 3-D prism at z=0 (atlas/sdf/train_analytical_sdf.py) so a 3-D SDFNet learns a
true prism field (|grad phi|=1, n_z~0). Sphere is natively 3-D. Closed forms are used where
they exist (sphere, circle); Superformula/Koch use the verified Euclidean references
(dense-boundary nearest / koch.nearest_boundary).

Pure numpy (matches solvers/contact/{supershape,koch}.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solvers.contact import supershape as ss          # noqa: E402
from solvers.contact import koch as koch_mod           # noqa: E402


class AnalyticalShape:
    """Base interface. Subclasses set `dim` and implement `sdf`, `sample_surface`, `inside`."""

    dim = 2
    name = "shape"

    def sdf(self, pts):                       # noqa: D401
        raise NotImplementedError

    def inside(self, pts):
        return self.sdf(pts) < 0.0

    def sample_surface(self, n, seed=0):
        raise NotImplementedError

    def body_size(self):
        raise NotImplementedError

    def sample_near(self, n, band, seed=0):
        """Points in a Gaussian shell of width `band` around the boundary."""
        pts, _ = self.sample_surface(n, seed)
        rng = np.random.RandomState(seed + 7)
        return pts + rng.randn(n, self.dim) * band

    def sample_bulk(self, n, seed=0, pad=1.4):
        """Uniform points in the padded bounding box."""
        L = self.body_size() * pad
        rng = np.random.RandomState(seed + 3)
        return rng.uniform(-L, L, size=(n, self.dim))


class Sphere(AnalyticalShape):
    """3-D sphere of radius R centred at the origin (CV-1 Hertz)."""

    dim = 3
    name = "sphere"

    def __init__(self, R=1.0):
        self.R = float(R)

    def sdf(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return np.linalg.norm(p, axis=1) - self.R

    def normal(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)

    def sample_surface(self, n, seed=0):
        rng = np.random.RandomState(seed)
        v = rng.randn(n, 3)
        v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        return self.R * v, v

    def body_size(self):
        return self.R


class Disc(AnalyticalShape):
    """2-D disc (circle) of radius R (CV-3 Brazilian, CV-4 nine-disc, 2-D line Hertz).

    Closed-form Euclidean SDF in the plane: g = |xy| - R. Alias `Cylinder` (its 3-D
    extrusion is an infinite cylinder, the 2-D line-contact body)."""

    dim = 2
    name = "disc"

    def __init__(self, R=1.0):
        self.R = float(R)

    def sdf(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return np.linalg.norm(p, axis=1) - self.R

    def normal(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)

    def sample_surface(self, n, seed=0):
        rng = np.random.RandomState(seed)
        th = rng.uniform(0, 2 * np.pi, n)
        v = np.column_stack([np.cos(th), np.sin(th)])
        return self.R * v, v

    def body_size(self):
        return self.R


Cylinder = Disc          # a 2-D circle is the cross-section of the 3-D line-contact cylinder


class Superformula(AnalyticalShape):
    """2-D Gielis superformula boundary (CV-5). EUCLIDEAN signed distance via a dense-boundary
    nearest-foot search (matches the harness reference and `closest_point_refine`); NOT the
    radial gap (which is the biased single-body chart, see verification manual §7/§11.2)."""

    dim = 2
    name = "supershape"

    def __init__(self, params=None, n_boundary=24000):
        self.p = params or ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, scale=1.0)
        self.c = np.zeros(2)
        # Dense boundary resampled UNIFORMLY IN ARC LENGTH (not theta): uniform-theta sampling
        # leaves gaps near the cusps (n1<1), which would inflate the nearest-foot SDF error
        # there.  Fine theta sweep -> cumulative arc length -> uniform-arc resample.
        thf = np.linspace(0, 2 * np.pi, 400000, endpoint=False)
        Bf = ss.boundary(thf, self.c, 0.0, self.p)
        seg = np.linalg.norm(np.diff(Bf, axis=0, append=Bf[:1]), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])            # (len+1,)
        s_unif = np.linspace(0.0, s[-1], n_boundary, endpoint=False)
        idx = np.searchsorted(s, s_unif, side="right") - 1
        self._B = Bf[np.clip(idx, 0, len(Bf) - 1)]            # (M, 2) arc-uniform boundary
        self._L = float(ss.radius(np.linspace(0, 2 * np.pi, 1024), self.p).max())

    def _nearest(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        out_d = np.empty(len(p))
        out_foot = np.empty((len(p), 2))
        for i in range(0, len(p), 256):                        # chunk to bound memory
            chunk = p[i:i + 256]
            d2 = np.sum((chunk[:, None, :] - self._B[None, :, :]) ** 2, axis=2)
            k = np.argmin(d2, axis=1)
            out_d[i:i + 256] = np.sqrt(d2[np.arange(len(chunk)), k])
            out_foot[i:i + 256] = self._B[k]
        return out_d, out_foot

    def inside(self, pts):
        return ss.inside(np.atleast_2d(np.asarray(pts, float)), self.c, 0.0, self.p)

    def sdf(self, pts):
        d, _ = self._nearest(pts)
        ins = self.inside(pts)
        return np.where(ins, -d, d)

    def sample_surface(self, n, seed=0):
        rng = np.random.RandomState(seed)
        th = rng.uniform(0, 2 * np.pi, n)
        pts = ss.boundary(th, self.c, 0.0, self.p)
        nrm = ss.outward_normal(th, self.c, 0.0, self.p)
        return pts, nrm

    def body_size(self):
        return self._L


class Koch(AnalyticalShape):
    """2-D Koch snowflake at fractal level (CV-6). EXACT signed distance via the verified
    `koch.nearest_boundary` (== brute-force polyline distance)."""

    dim = 2
    name = "koch"

    def __init__(self, level=1, R=1.0):
        self.level = int(level)
        self.R = float(R)

    def sdf(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return np.array([koch_mod.nearest_boundary(q, self.level, R=self.R)[0] for q in p])

    def inside(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return koch_mod.inside(p, self.level, R=self.R)

    def sample_surface(self, n, seed=0):
        rng = np.random.RandomState(seed)
        V = koch_mod.snowflake_vertices(self.level, R=self.R)
        P, Q = V[:-1], V[1:]
        idx = rng.randint(0, len(P), size=n)
        t = rng.rand(n)
        on = P[idx] + t[:, None] * (Q[idx] - P[idx])
        nrm = np.array([koch_mod.nearest_boundary(p + 1e-6 * np.array([1, 0]), self.level, R=self.R)[2]
                        for p in on])
        return on, nrm

    def body_size(self):
        V = koch_mod.snowflake_vertices(self.level, R=self.R)[:-1]
        c = V.mean(axis=0)
        return float(np.linalg.norm(V - c, axis=1).max())


_REGISTRY = {
    "sphere": lambda: Sphere(1.0),
    "disc": lambda: Disc(1.0),
    "cylinder": lambda: Cylinder(1.0),
    "supershape": lambda: Superformula(),
    "koch": lambda: Koch(1),
}


def get_shape(name, **kwargs):
    """Factory: `get_shape('sphere')`, `get_shape('koch', level=3)`, etc."""
    name = name.lower()
    if name == "koch":
        return Koch(kwargs.get("level", 1), kwargs.get("R", 1.0))
    if name in ("sphere", "disc", "cylinder"):
        cls = {"sphere": Sphere, "disc": Disc, "cylinder": Cylinder}[name]
        return cls(kwargs.get("R", 1.0))
    if name == "supershape":
        return Superformula(kwargs.get("params"))
    raise ValueError(f"unknown shape '{name}'")


if __name__ == "__main__":
    # self-test: sign correctness + zero-level-set + closed-form agreement
    for nm in ("sphere", "disc", "supershape", "koch"):
        s = get_shape(nm, level=2) if nm == "koch" else get_shape(nm)
        pts, nrm = s.sample_surface(200, 0)
        g_on = s.sdf(pts)
        # sphere/disc/koch have an EXACT signed distance; supershape uses a dense-boundary
        # nearest-foot approximation, so on-boundary points read ~half the sample spacing.
        tol_on = 1e-3 if nm == "supershape" else 1e-6
        assert np.max(np.abs(g_on)) < tol_on, (nm, np.max(np.abs(g_on)))
        inside_pt = np.zeros((1, s.dim))
        assert s.sdf(inside_pt)[0] < 0, nm                     # centre is inside
        print(f"  [{nm:10s}] dim={s.dim} L={s.body_size():.4f} "
              f"max|sdf on boundary|={np.max(np.abs(g_on)):.2e}  OK")
    print("analytical shape library self-test PASSED")
