"""2-D linear-elastic FEM on constant-strain triangles (plane stress / plane strain).

The "true 2-D element path" for the numerical CV suite (plan B2): CV-3 Brazilian and CV-4
nine-disc are 2-D plane-stress BVPs whose sharp/structured stress fields are far cleaner on a
real 2-D mesh than on a thin 3-D slab.

The solver assembles on PHYSICAL node positions, so the coordinate chart enters simply by
supplying those positions: a direct analytical mesh (identity chart) for the baseline, or
``decoder(reference_mesh)`` for a NEURAL ChartDecoder chart (the geometry distortion of the
trained chart then shows up directly in the recovered field). Standard CST elements; dense
solve (the CV meshes are a few hundred–thousand nodes). Pure numpy + scipy.

Reference: Zienkiewicz & Taylor, *The Finite Element Method*; CST = Turner triangle.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def plane_C(E: float, nu: float, mode: str = "plane_stress") -> np.ndarray:
    """3x3 Voigt elasticity matrix C for [sxx, syy, sxy] = C [exx, eyy, gxy]."""
    if mode == "plane_stress":
        f = E / (1.0 - nu ** 2)
        return f * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    if mode == "plane_strain":
        f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return f * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0],
                             [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
    raise ValueError(mode)


def _tri_B_area(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """CST strain-displacement matrix B (3x6) and signed area for one triangle (3x2 coords)."""
    (x1, y1), (x2, y2), (x3, y3) = xy
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * detJ
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ        # dN_i/dx
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ        # dN_i/dy
    B = np.zeros((3, 6))
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    return B, area


class Tri2DFEMSolver:
    """Plane CST FEM. Assembles on physical nodes; chart-agnostic."""

    def __init__(self, nodes: np.ndarray, elements: np.ndarray, E: float, nu: float,
                 thickness: float = 1.0, mode: str = "plane_stress"):
        self.nodes = np.asarray(nodes, float)              # (N, 2) physical
        self.elements = np.asarray(elements, int)          # (M, 3)
        self.E, self.nu, self.t, self.mode = E, nu, thickness, mode
        self.C = plane_C(E, nu, mode)
        self.n_nodes = len(self.nodes)
        self.n_dof = 2 * self.n_nodes
        self._B = np.empty((len(self.elements), 3, 6))
        self._area = np.empty(len(self.elements))
        for e, tri in enumerate(self.elements):
            self._B[e], self._area[e] = _tri_B_area(self.nodes[tri])
        assert (self._area > 0).all(), "non-positive triangle area (check winding)"

    def _elem_dofs(self) -> np.ndarray:
        """(M, 6) global dof indices per element [ux0,uy0,ux1,uy1,ux2,uy2]."""
        e = self.elements
        return np.stack([2 * e[:, 0], 2 * e[:, 0] + 1, 2 * e[:, 1], 2 * e[:, 1] + 1,
                         2 * e[:, 2], 2 * e[:, 2] + 1], axis=1)

    def assemble(self):
        """Global stiffness K as a scipy CSR sparse matrix (vectorized COO triplets)."""
        from scipy.sparse import coo_matrix
        Ke = self.t * self._area[:, None, None] * np.einsum(
            "eki,kl,elj->eij", self._B, self.C, self._B)     # (M, 6, 6)
        dofs = self._elem_dofs()                             # (M, 6)
        rows = np.broadcast_to(dofs[:, :, None], Ke.shape).reshape(-1)
        cols = np.broadcast_to(dofs[:, None, :], Ke.shape).reshape(-1)
        return coo_matrix((Ke.reshape(-1), (rows, cols)),
                          shape=(self.n_dof, self.n_dof)).tocsr()

    def solve(self, fixed_dofs: np.ndarray, f_ext: np.ndarray,
              prescribed: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve K u = f with Dirichlet `fixed_dofs` (indices into the 2N dof vector), sparse.

        f_ext : (N, 2) nodal forces. prescribed : (n_fixed,) values for fixed_dofs (default 0).
        Returns u (N, 2)."""
        from scipy.sparse.linalg import spsolve
        K = self.assemble()
        f = np.asarray(f_ext, float).reshape(-1).copy()
        u = np.zeros(self.n_dof)
        fixed = np.asarray(fixed_dofs, int)
        if prescribed is not None:
            u[fixed] = prescribed
            f = f - K @ u                                   # move prescribed to RHS
        free = np.setdiff1d(np.arange(self.n_dof), fixed)
        # two successive slices avoid scipy's np.ix_ int32 overflow on large sparse matrices
        Kff = K.tocsr()[free][:, free].tocsc()
        u[free] = spsolve(Kff, f[free])
        return u.reshape(self.n_nodes, 2)

    def element_stress(self, u: np.ndarray) -> np.ndarray:
        """Per-element Voigt stress (M, 3) = [sxx, syy, sxy] (vectorized)."""
        u = np.asarray(u, float).reshape(self.n_nodes, 2)
        ue = u[self.elements].reshape(len(self.elements), 6)     # (M, 6)
        strain = np.einsum("eij,ej->ei", self._B, ue)            # (M, 3)
        return strain @ self.C.T

    def element_centroids(self) -> np.ndarray:
        return self.nodes[self.elements].mean(axis=1)

    def node_stress(self, u: np.ndarray) -> np.ndarray:
        """Area-weighted nodal stress (N, 3) by averaging incident element stresses."""
        es = self.element_stress(u)
        acc = np.zeros((self.n_nodes, 3))
        wsum = np.zeros(self.n_nodes)
        for e, tri in enumerate(self.elements):
            for nd in tri:
                acc[nd] += self._area[e] * es[e]
                wsum[nd] += self._area[e]
        return acc / np.clip(wsum[:, None], 1e-30, None)

    def stress_at(self, u: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Voigt stress sampled at arbitrary points (nearest element centroid). (K, 3)."""
        cen = self.element_centroids()
        es = self.element_stress(u)
        pts = np.atleast_2d(np.asarray(pts, float))
        idx = np.array([np.argmin(np.sum((cen - p) ** 2, axis=1)) for p in pts])
        return es[idx]


def disc_mesh(R: float = 1.0, n_rings: int = 16, center: Tuple[float, float] = (0.0, 0.0)
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentric-ring disc triangulation (Delaunay of the ring points).

    Returns (nodes (N,2), tris (M,3 CCW), boundary_node_idx). Ring density grows with radius so
    triangles stay near-equilateral. The boundary ring (i=n_rings) gives the rim nodes."""
    from scipy.spatial import Delaunay
    pts = [np.array([0.0, 0.0])]
    bnd = []
    for i in range(1, n_rings + 1):
        r = R * i / n_rings
        n_i = max(6, int(round(2 * np.pi * i)))            # ~uniform arc spacing
        ang = np.linspace(0, 2 * np.pi, n_i, endpoint=False) + (0.5 * np.pi / n_i) * (i % 2)
        ring = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        if i == n_rings:
            bnd = list(range(len(pts), len(pts) + n_i))
        pts.extend(ring)
    nodes = np.array(pts)
    tri = Delaunay(nodes).simplices
    # enforce CCW winding (positive area)
    v = nodes[tri]
    area2 = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) -
             (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    flip = area2 < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    nodes = nodes + np.asarray(center)
    return nodes, tri, np.array(bnd, int)


def graded_box_mesh(W: float, D: float, n_x: int = 80, n_y: int = 48, grade: float = 3.0):
    """Half-plane box [-W, W] x [-D, 0], structured grid GRADED toward the top-centre contact
    zone (x->0, y->0), each cell split into two CCW triangles.

    The Hertz contact patch is tiny, so the mesh must refine where contact happens.  `grade`>1
    clusters nodes near x=0 and y=0 (power-law).  Returns (nodes, tris, top_idx, bottom_idx)
    where top_idx are the y=0 surface nodes (left->right) and bottom_idx the y=-D nodes."""
    s = np.linspace(-1.0, 1.0, n_x + 1)
    xs = W * np.sign(s) * np.abs(s) ** grade                  # cluster near x=0
    v = np.linspace(0.0, 1.0, n_y + 1)
    ys = -D * (1.0 - v) ** grade                              # cluster near y=0 (top)
    XX, YY = np.meshgrid(xs, ys)                              # (n_y+1, n_x+1)
    nodes = np.column_stack([XX.ravel(), YY.ravel()])
    nx1 = n_x + 1

    def nid(iy, ix):
        return iy * nx1 + ix

    tris = []
    for iy in range(n_y):
        for ix in range(n_x):
            a, b, c, d = nid(iy, ix), nid(iy, ix + 1), nid(iy + 1, ix + 1), nid(iy + 1, ix)
            tris.append([a, b, d])                            # CCW (y increases with iy)
            tris.append([b, c, d])
    tris = np.array(tris, int)
    top_idx = np.array([nid(n_y, ix) for ix in range(nx1)])   # y=0 row
    bottom_idx = np.array([nid(0, ix) for ix in range(nx1)])  # y=-D row
    # enforce CCW (the grid winding above is CCW already, but guard against grading flips)
    vtx = nodes[tris]
    area2 = ((vtx[:, 1, 0] - vtx[:, 0, 0]) * (vtx[:, 2, 1] - vtx[:, 0, 1]) -
             (vtx[:, 2, 0] - vtx[:, 0, 0]) * (vtx[:, 1, 1] - vtx[:, 0, 1]))
    tris[area2 < 0] = tris[area2 < 0][:, [0, 2, 1]]
    return nodes, tris, top_idx, bottom_idx


if __name__ == "__main__":
    # self-test: uniaxial tension patch -> uniform sigma_xx, zero sigma_yy
    nodes, tris, bnd = disc_mesh(1.0, 10)
    print(f"  disc mesh: {len(nodes)} nodes, {len(tris)} triangles, {len(bnd)} rim nodes")
    # square patch test instead (exact): unit square, prescribe u = eps*x
    gx = np.linspace(0, 1, 6)
    XX, YY = np.meshgrid(gx, gx)
    sq_nodes = np.column_stack([XX.ravel(), YY.ravel()])
    from scipy.spatial import Delaunay
    sq_tris = Delaunay(sq_nodes).simplices
    E, nu, eps = 200.0, 0.3, 1e-3
    sol = Tri2DFEMSolver(sq_nodes, sq_tris, E, nu, thickness=1.0, mode="plane_stress")
    # prescribe exact linear field u=(eps*x, -nu*eps*y) (uniaxial stress) on the boundary
    on_b = (np.abs(sq_nodes[:, 0]) < 1e-9) | (np.abs(sq_nodes[:, 0] - 1) < 1e-9) | \
           (np.abs(sq_nodes[:, 1]) < 1e-9) | (np.abs(sq_nodes[:, 1] - 1) < 1e-9)
    fixed, presc = [], []
    for i in np.where(on_b)[0]:
        fixed += [2 * i, 2 * i + 1]
        presc += [eps * sq_nodes[i, 0], -nu * eps * sq_nodes[i, 1]]
    u = sol.solve(np.array(fixed), np.zeros((sol.n_nodes, 2)), np.array(presc))
    s = sol.element_stress(u)
    sxx_exp = E * eps
    print(f"  uniaxial patch: sigma_xx mean={s[:,0].mean():.4f} exp={sxx_exp:.4f} "
          f"(err {abs(s[:,0].mean()-sxx_exp)/sxx_exp*100:.2e}%), "
          f"|sigma_yy|max={np.abs(s[:,1]).max():.2e}")
    assert abs(s[:, 0].mean() - sxx_exp) / sxx_exp < 1e-9
    assert np.abs(s[:, 1]).max() < 1e-9
    print("  tri2d patch test PASSED")
