"""1-D Gauss-Legendre quadrature and boundary shape functions for surface-integrated contact.

The repo has no 1-D Gauss helper (only the 4-point tet rule in ``solvers/fem/p2_tet.py`` and
``np.trapz`` in postprocessing).  This module supplies the surface-quadrature primitives the
measure-coupling contact assembly integrates the traction field against.

A contact boundary is an ordered polyline of surface nodes ``X (n_s, 2)`` (the deformed FEM
boundary, or a chart sampled at its nodes).  :func:`segment_quadrature` lays a Gauss-Legendre rule
on every segment and returns, per quadrature point, the physical position, the arclength weight
``w*ds``, the P1 boundary shape-function values at the two host nodes, and the host-node local
indices — exactly what :func:`solvers.contact.measure_coupling.assembly.assemble_contact` needs to
integrate ``f_I = sum_q (w ds) N_I traction_q``.
"""
from __future__ import annotations

import numpy as np


def gauss_legendre_1d(order: int = 3):
    """Reference Gauss-Legendre nodes/weights on ``[-1, 1]`` (``numpy.polynomial.legendre.leggauss``).

    ``order`` points integrate a polynomial of degree ``2*order-1`` exactly.  The default
    ``order=3`` is exact for the P1*P1*(linear-Jacobian) products that arise in the consistent
    contact mass/tangent and for a quadratic traction profile within a segment.
    """
    xi, w = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(xi, float), np.asarray(w, float)


def lagrange_p1(loc: np.ndarray):
    """Linear boundary shape functions on the reference segment ``loc`` in ``[-1, 1]``.

    Returns ``N (Q, 2) = [(1-loc)/2, (1+loc)/2]`` (partition of unity: rows sum to 1) and
    ``dN/dloc (Q, 2)``.
    """
    loc = np.asarray(loc, float)
    N = np.stack([0.5 * (1.0 - loc), 0.5 * (1.0 + loc)], axis=1)
    dN = np.stack([-0.5 * np.ones_like(loc), 0.5 * np.ones_like(loc)], axis=1)
    return N, dN


def segment_quadrature(surf_xy: np.ndarray, order: int = 3) -> dict:
    """Lay a Gauss-Legendre rule on every segment of an ordered surface polyline.

    Parameters
    ----------
    surf_xy : (n_s, 2) ordered surface node positions (deformed config).  Segments are the
        consecutive pairs ``(k, k+1)``.
    order : Gauss points per segment.

    Returns
    -------
    dict with arrays over all ``Q = (n_s-1)*order`` quadrature points:
        ``Xq`` (Q, 2) physical position of each quadrature point;
        ``wds`` (Q,) arclength quadrature weight ``w_ref * |P1-P0|/2``;
        ``N`` (Q, 2) P1 shape-function values ``[N0, N1]`` at the quadrature point;
        ``i0`` (Q,) local index of the segment's first node (into ``surf_xy``);
        ``i1`` (Q,) local index of the segment's second node;
        ``seg`` (Q,) owning segment index.

    The segment length carries the arclength Jacobian, so no separate ``sqrt(1+h'^2)`` factor is
    needed for a polyline surface; for a chart sampled at its nodes the chord length is a
    2nd-order-accurate approximation to the arclength, refined with node density.
    """
    surf_xy = np.asarray(surf_xy, float)
    n_s = len(surf_xy)
    if n_s < 2:
        raise ValueError("segment_quadrature needs at least 2 surface nodes")
    xi, w = gauss_legendre_1d(order)
    s = 0.5 * (1.0 + xi)                       # map [-1, 1] -> [0, 1]
    N0 = 1.0 - s
    N1 = s
    Xq, wds, Nq, I0, I1, SEG = [], [], [], [], [], []
    for k in range(n_s - 1):
        P0, P1 = surf_xy[k], surf_xy[k + 1]
        L = float(np.linalg.norm(P1 - P0))
        Xq.append(np.outer(N0, P0) + np.outer(N1, P1))         # (order, 2)
        wds.append(w * 0.5 * L)
        Nq.append(np.stack([N0, N1], axis=1))
        I0.append(np.full(order, k, dtype=int))
        I1.append(np.full(order, k + 1, dtype=int))
        SEG.append(np.full(order, k, dtype=int))
    return dict(
        Xq=np.concatenate(Xq, axis=0),
        wds=np.concatenate(wds, axis=0),
        N=np.concatenate(Nq, axis=0),
        i0=np.concatenate(I0, axis=0),
        i1=np.concatenate(I1, axis=0),
        seg=np.concatenate(SEG, axis=0),
    )


if __name__ == "__main__":
    # self-test: partition of unity + exact integration of constants/linears on a polyline.
    N, dN = lagrange_p1(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(N.sum(axis=1), 1.0), "P1 shape functions must be a partition of unity"

    # straight surface from (0,0) to (3,0) with 3 unequal segments: total length 3
    xs = np.array([0.0, 0.5, 2.0, 3.0])
    surf = np.column_stack([xs, np.zeros_like(xs)])
    q = segment_quadrature(surf, order=3)
    L_quad = q["wds"].sum()
    assert abs(L_quad - 3.0) < 1e-12, f"sum of arclength weights {L_quad} != length 3"

    # integral of a linear field f(x)=2x+1 over [0,3] = [x^2+x]_0^3 = 12
    fval = 2.0 * q["Xq"][:, 0] + 1.0
    I = float((q["wds"] * fval).sum())
    assert abs(I - 12.0) < 1e-10, f"quadrature integral {I} != 12"
    print(f"  quadrature self-test OK: length={L_quad:.12f}, int(2x+1, 0..3)={I:.12f}")
