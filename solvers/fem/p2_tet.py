"""P2 (quadratic) tetrahedral reference element.

Provides shape functions, gradients, and quadrature rules for 10-node
quadratic tetrahedra used in p-refined crack-tip charts.

Node layout:
  Vertices:  0, 1, 2, 3
  Edge midpoints:  4=(0,1), 5=(0,2), 6=(0,3), 7=(1,2), 8=(1,3), 9=(2,3)

Barycentric coordinates: L = (L1, L2, L3), L4 = 1 - L1 - L2 - L3
where L_i is the barycentric coordinate for vertex i.

Shape functions (quadratic in L):
  Vertex nodes:     N_i = L_i * (2*L_i - 1)           for i = 1..4
  Edge midpoints:   N_{ij} = 4 * L_i * L_j             for edges (i,j)

Reference:
  Zienkiewicz, Taylor & Zhu (2013), "The Finite Element Method",
  Chapter 9: Higher-order elements.
"""

import numpy as np
import torch
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────
# Edge connectivity: 6 edges of a tet, each defined by 2 vertex indices
# ─────────────────────────────────────────────────────────────────────

EDGE_PAIRS = [
    (0, 1),  # edge 0 → midpoint node 4
    (0, 2),  # edge 1 → midpoint node 5
    (0, 3),  # edge 2 → midpoint node 6
    (1, 2),  # edge 3 → midpoint node 7
    (1, 3),  # edge 4 → midpoint node 8
    (2, 3),  # edge 5 → midpoint node 9
]


# ─────────────────────────────────────────────────────────────────────
# 4-point Gauss quadrature for tetrahedra (degree 2, exact for cubics)
# Points in barycentric coordinates (L1, L2, L3); L4 = 1 - L1 - L2 - L3
# ─────────────────────────────────────────────────────────────────────

_a = 0.1381966011250105  # (5 - sqrt(5)) / 20
_b = 0.5854101966249685  # (5 + 3*sqrt(5)) / 20

QUAD_POINTS_BARY = np.array([
    [_a, _a, _a],  # L4 = _b
    [_b, _a, _a],  # L4 = _a
    [_a, _b, _a],  # L4 = _a
    [_a, _a, _b],  # L4 = _a
], dtype=np.float64)

# Weights: each = 1/4 of reference tet volume (1/6)
QUAD_WEIGHTS = np.full(4, 1.0 / 24.0, dtype=np.float64)  # sum = 1/6


# ─────────────────────────────────────────────────────────────────────
# Shape functions
# ─────────────────────────────────────────────────────────────────────

def p2_shape_functions(L: np.ndarray) -> np.ndarray:
    """Evaluate 10 P2 shape functions at barycentric coordinates.

    Parameters
    ----------
    L : ndarray (..., 3)
        Barycentric coordinates (L1, L2, L3). L4 = 1 - L1 - L2 - L3.

    Returns
    -------
    N : ndarray (..., 10)
        Shape function values.
    """
    L1, L2, L3 = L[..., 0], L[..., 1], L[..., 2]
    L4 = 1.0 - L1 - L2 - L3

    N = np.empty(L.shape[:-1] + (10,), dtype=L.dtype)

    # Vertex nodes: N_i = L_i * (2*L_i - 1)
    N[..., 0] = L1 * (2 * L1 - 1)
    N[..., 1] = L2 * (2 * L2 - 1)
    N[..., 2] = L3 * (2 * L3 - 1)
    N[..., 3] = L4 * (2 * L4 - 1)

    # Edge midpoints: N_{ij} = 4 * L_i * L_j
    N[..., 4] = 4 * L1 * L2   # edge (0,1)
    N[..., 5] = 4 * L1 * L3   # edge (0,2)
    N[..., 6] = 4 * L1 * L4   # edge (0,3)
    N[..., 7] = 4 * L2 * L3   # edge (1,2)
    N[..., 8] = 4 * L2 * L4   # edge (1,3)
    N[..., 9] = 4 * L3 * L4   # edge (2,3)

    return N


def p2_shape_gradients(L: np.ndarray) -> np.ndarray:
    """Evaluate gradients of 10 P2 shape functions w.r.t. (L1, L2, L3).

    Parameters
    ----------
    L : ndarray (..., 3)
        Barycentric coordinates.

    Returns
    -------
    dN_dL : ndarray (..., 10, 3)
        Gradient of each shape function w.r.t. (L1, L2, L3).
        Note: dN/dL4 = -dN/dL1 - dN/dL2 - dN/dL3 is implicit via chain rule.
    """
    L1, L2, L3 = L[..., 0], L[..., 1], L[..., 2]
    L4 = 1.0 - L1 - L2 - L3

    shape = L.shape[:-1] + (10, 3)
    dN = np.zeros(shape, dtype=L.dtype)

    # dN0/dL = d[L1*(2L1-1)]/dL = (4L1 - 1) * dL1/dL
    # dL1/dL1 = 1, dL1/dL2 = 0, dL1/dL3 = 0
    dN[..., 0, 0] = 4 * L1 - 1  # dN0/dL1
    # dN0/dL2 = 0, dN0/dL3 = 0

    dN[..., 1, 1] = 4 * L2 - 1  # dN1/dL2

    dN[..., 2, 2] = 4 * L3 - 1  # dN2/dL3

    # N3 = L4*(2L4-1), L4 = 1-L1-L2-L3
    # dN3/dLk = -(4L4-1) for k=1,2,3
    dN3_common = -(4 * L4 - 1)
    dN[..., 3, 0] = dN3_common
    dN[..., 3, 1] = dN3_common
    dN[..., 3, 2] = dN3_common

    # N4 = 4*L1*L2
    dN[..., 4, 0] = 4 * L2
    dN[..., 4, 1] = 4 * L1

    # N5 = 4*L1*L3
    dN[..., 5, 0] = 4 * L3
    dN[..., 5, 2] = 4 * L1

    # N6 = 4*L1*L4 = 4*L1*(1-L1-L2-L3)
    dN[..., 6, 0] = 4 * (L4 - L1)      # = 4*(1-2L1-L2-L3)
    dN[..., 6, 1] = -4 * L1
    dN[..., 6, 2] = -4 * L1

    # N7 = 4*L2*L3
    dN[..., 7, 1] = 4 * L3
    dN[..., 7, 2] = 4 * L2

    # N8 = 4*L2*L4 = 4*L2*(1-L1-L2-L3)
    dN[..., 8, 0] = -4 * L2
    dN[..., 8, 1] = 4 * (L4 - L2)      # = 4*(1-L1-2L2-L3)
    dN[..., 8, 2] = -4 * L2

    # N9 = 4*L3*L4 = 4*L3*(1-L1-L2-L3)
    dN[..., 9, 0] = -4 * L3
    dN[..., 9, 1] = -4 * L3
    dN[..., 9, 2] = 4 * (L4 - L3)      # = 4*(1-L1-L2-2L3)

    return dN


# ─────────────────────────────────────────────────────────────────────
# Coordinate conversions
# ─────────────────────────────────────────────────────────────────────

def barycentric_to_cartesian(
    L: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Convert barycentric coordinates to Cartesian.

    Parameters
    ----------
    L : ndarray (..., 3)
        Barycentric (L1, L2, L3). L4 = 1 - sum.
    vertices : ndarray (4, 3)
        Tet vertex positions.

    Returns
    -------
    x : ndarray (..., 3)
    """
    L1, L2, L3 = L[..., 0:1], L[..., 1:2], L[..., 2:3]
    L4 = 1.0 - L1 - L2 - L3
    return L1 * vertices[0] + L2 * vertices[1] + L3 * vertices[2] + L4 * vertices[3]


def cartesian_to_barycentric(
    x: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Convert Cartesian to barycentric coordinates.

    Parameters
    ----------
    x : ndarray (..., 3)
    vertices : ndarray (4, 3)

    Returns
    -------
    L : ndarray (..., 3)  — (L1, L2, L3), with L4 = 1 - sum
    """
    # Transform: x = v0*L1 + v1*L2 + v2*L3 + v3*L4
    # = v3 + (v0-v3)*L1 + (v1-v3)*L2 + (v2-v3)*L3
    T = np.column_stack([
        vertices[0] - vertices[3],
        vertices[1] - vertices[3],
        vertices[2] - vertices[3],
    ])  # (3, 3)
    dx = x - vertices[3]  # (..., 3)
    L = dx @ np.linalg.inv(T).T  # (..., 3)
    return L


# ─────────────────────────────────────────────────────────────────────
# P2 edge midpoint generation for mesh
# ─────────────────────────────────────────────────────────────────────

def build_p2_connectivity(
    elements_p1: np.ndarray, n_nodes_p1: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand P1 tet connectivity to P2 by adding shared edge midpoints.

    Parameters
    ----------
    elements_p1 : ndarray (M, 4)
        P1 tet connectivity (vertex indices).
    n_nodes_p1 : int
        Number of P1 (vertex) nodes.

    Returns
    -------
    elements_p2 : ndarray (M, 10)
        P2 tet connectivity [v0, v1, v2, v3, m01, m02, m03, m12, m13, m23].
    n_nodes_p2 : int
        Total number of nodes (vertices + edge midpoints).
    """
    edge_to_mid = {}  # (min_idx, max_idx) → midpoint global index
    next_mid_idx = n_nodes_p1

    M = len(elements_p1)
    elements_p2 = np.zeros((M, 10), dtype=elements_p1.dtype)
    elements_p2[:, :4] = elements_p1

    for e in range(M):
        verts = elements_p1[e]  # (4,)
        for k, (a, b) in enumerate(EDGE_PAIRS):
            edge_key = (min(verts[a], verts[b]), max(verts[a], verts[b]))
            if edge_key not in edge_to_mid:
                edge_to_mid[edge_key] = next_mid_idx
                next_mid_idx += 1
            elements_p2[e, 4 + k] = edge_to_mid[edge_key]

    n_nodes_p2 = next_mid_idx
    return elements_p2, n_nodes_p2


def compute_midpoint_positions(
    nodes_p1: np.ndarray,
    elements_p2: np.ndarray,
    n_nodes_p1: int,
    n_nodes_p2: int,
) -> np.ndarray:
    """Compute positions of edge midpoint nodes by averaging vertices.

    Parameters
    ----------
    nodes_p1 : ndarray (N_p1, 3)
        P1 vertex node positions.
    elements_p2 : ndarray (M, 10)
        P2 element connectivity.
    n_nodes_p1, n_nodes_p2 : int

    Returns
    -------
    nodes_p2 : ndarray (N_p2, 3)
        All node positions (vertices + midpoints).
    """
    nodes_p2 = np.zeros((n_nodes_p2, 3), dtype=nodes_p1.dtype)
    nodes_p2[:n_nodes_p1] = nodes_p1

    # Mark which midpoints have been computed
    computed = np.zeros(n_nodes_p2, dtype=bool)
    computed[:n_nodes_p1] = True

    for e in range(len(elements_p2)):
        verts = elements_p2[e, :4]
        for k, (a, b) in enumerate(EDGE_PAIRS):
            mid_idx = elements_p2[e, 4 + k]
            if not computed[mid_idx]:
                nodes_p2[mid_idx] = 0.5 * (nodes_p1[verts[a]] + nodes_p1[verts[b]])
                computed[mid_idx] = True

    return nodes_p2


# ─────────────────────────────────────────────────────────────────────
# P2 shape function gradients in physical space (at quadrature points)
# ─────────────────────────────────────────────────────────────────────

def p2_dNdx_at_quadpoints(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute P2 shape function gradients in Cartesian space at quad points.

    For a single element defined by 4 vertex positions.

    Parameters
    ----------
    vertices : ndarray (4, 3)
        Vertex positions of the tet.

    Returns
    -------
    dNdx : ndarray (4, 10, 3)
        Shape function gradients at each of 4 quad points.
    detJ : ndarray (4,)
        Jacobian determinant at each quad point (constant for linear geom).
    """
    # The Jacobian dx/dL is constant for linear (P1) geometry:
    # dx/dL = [v0-v3 | v1-v3 | v2-v3]^T  (3×3)
    J_geom = np.column_stack([
        vertices[0] - vertices[3],
        vertices[1] - vertices[3],
        vertices[2] - vertices[3],
    ])  # (3, 3)

    detJ_val = abs(np.linalg.det(J_geom))
    J_inv = np.linalg.inv(J_geom)  # (3, 3)

    # dN/dx = dN/dL @ J^{-1}
    n_qp = len(QUAD_POINTS_BARY)
    dNdx = np.zeros((n_qp, 10, 3), dtype=np.float64)

    for q in range(n_qp):
        L = QUAD_POINTS_BARY[q]
        dN_dL = p2_shape_gradients(L)  # (10, 3)
        dNdx[q] = dN_dL @ J_inv       # (10, 3)

    detJ = np.full(n_qp, detJ_val)
    return dNdx, detJ
