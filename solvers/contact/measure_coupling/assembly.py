"""Consistent Galerkin (segment-to-segment / mortar-structured) contact assembly.

The contact traction is a finite-element FIELD: nodal tractions ``t_J`` (from the per-node
measure-coupling gap, evaluated by the exact closest-point/correspondence to the master) are
interpolated by the boundary shape functions and integrated by Gauss-Legendre quadrature::

    p(x) = sum_J N_J(x) p_J,   t(x) = sum_J N_J(x) t_J
    f_I  = int N_I t(x) ds = sum_J M_IJ t_J,         M_IJ = int N_I N_J ds   (consistent mass)
    K_c  = sum_J M_IJ eps_n (n_J (x) n_J)   on the active set                (consistent tangent)

The ``M_IJ`` mass coupling ties ADJACENT slave nodes (the off-diagonal mortar structure).  The
existing node-collocated penalty (``cv1_hertz_fem.py:138-139``, ``rock_joint_*``) instead LUMPS the
mass (diagonal tributary weights) and drops the coupling — a node-spacing sawtooth across
non-matching meshes.

Why evaluate the gap at NODES (not at the Gauss points): for a curved rigid master the penalty
penetration is sub-element-tiny and a straight slave facet penetrates the curved master
non-uniformly within an element (chord-vs-arc), so a raw Gauss-point gap would sample that faceting
noise.  Evaluating the closest-point gap at the nodes (exact there) and interpolating gives the
conforming pressure field — the same reason a consistent penalty uses nodal pressures.  For two
flat/charted surfaces (patch test, rock joint) node- and Gauss-evaluation coincide.

``K_c`` omits the geometric term ``p_N d n/d u`` (small-rotation / rigid-counterface regime), the
same documented approximation as the existing penalty drivers.
"""
from __future__ import annotations

import numpy as np

from .quadrature import gauss_legendre_1d


def assemble_contact(surf_xy, node_ids, n_dof, eval_gap, traction,
                     order: int = 3, eval_slip=None):
    """Assemble consistent contact force and tangent over an ordered slave surface polyline.

    Parameters
    ----------
    surf_xy : (n_s, 2) ordered deformed slave surface node positions.
    node_ids : (n_s,) global node indices of those surface nodes.
    n_dof : int, total dofs (2 * n_nodes_total).
    eval_gap : callable ``X (n,2) -> (gN (n,), n (n,2))`` (rigid ``gap_normal`` or
        :meth:`GapField.eval_gap`); evaluated at the surface NODES.
    traction : :class:`TractionField`.
    order : Gauss points per segment (mass-matrix exactness).
    eval_slip : callable ``X (n,2) -> slip (n,2)``, optional (friction).

    Returns
    -------
    f : (n_nodes, 2) consistent contact force per node.
    Kc : scipy.sparse CSR (n_dof, n_dof) consistent contact tangent.
    diag : dict with nodal field (x, pN, n, gN), the line load F, and the interpolated
        Gauss-point field (Xq, pN_q) for plotting.
    """
    from scipy.sparse import coo_matrix

    surf_xy = np.asarray(surf_xy, float)
    node_ids = np.asarray(node_ids, int)
    n_s = len(surf_xy)
    n_nodes = n_dof // 2

    # --- nodal traction from the closest-point/correspondence gap at the nodes ---
    gN_n, n_n = eval_gap(surf_xy)
    slip_n = eval_slip(surf_xy) if eval_slip is not None else None
    tr = traction.evaluate(gN_n, n_n, slip=slip_n)
    t_node, deps_node, pN_node = tr["t"], tr["deps"], tr["pN"]

    xi, w = gauss_legendre_1d(order)
    s = 0.5 * (1.0 + xi)
    Nref = np.stack([1.0 - s, s], axis=1)                  # (order, 2)

    f = np.zeros((n_nodes, 2))
    rows, cols, vals = [], [], []
    Xq_all, pNq_all, wds_all = [], [], []
    for k in range(n_s - 1):
        P0, P1 = surf_xy[k], surf_xy[k + 1]
        L = float(np.linalg.norm(P1 - P0))
        wds = w * 0.5 * L                                  # (order,)
        m = np.einsum("q,qa,qb->ab", wds, Nref, Nref)      # local consistent mass (2,2)
        loc = (k, k + 1)
        gid = (node_ids[k], node_ids[k + 1])
        tt = (t_node[loc[0]], t_node[loc[1]])
        # consistent force  f_I += sum_J M_IJ t_J
        for a in range(2):
            f[gid[a]] += m[a, 0] * tt[0] + m[a, 1] * tt[1]
        # consistent tangent  K_IJ += M_IJ eps_n (n_J (x) n_J)  (active J)
        for a in range(2):
            for b in range(2):
                dj = deps_node[loc[b]]
                if dj != 0.0:
                    nb = n_n[loc[b]]
                    blk = m[a, b] * dj * np.outer(nb, nb)
                    Ia, Ib = gid[a], gid[b]
                    for di in range(2):
                        for dk in range(2):
                            rows.append(2 * Ia + di)
                            cols.append(2 * Ib + dk)
                            vals.append(blk[di, dk])
        # interpolated Gauss-point pressure field (plotting / line-load integral)
        Xq_all.append(np.outer(Nref[:, 0], P0) + np.outer(Nref[:, 1], P1))
        pNq_all.append(Nref[:, 0] * pN_node[loc[0]] + Nref[:, 1] * pN_node[loc[1]])
        wds_all.append(wds)

    if rows:
        Kc = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    else:
        Kc = coo_matrix((n_dof, n_dof)).tocsr()

    Xq = np.concatenate(Xq_all, axis=0) if Xq_all else np.zeros((0, 2))
    pNq = np.concatenate(pNq_all, axis=0) if pNq_all else np.zeros(0)
    wds_cat = np.concatenate(wds_all, axis=0) if wds_all else np.zeros(0)
    F_line = float((wds_cat * pNq).sum())                  # int p ds = trapezoid of nodal pressures
    diag = dict(x=surf_xy[:, 0], pN=pN_node, n=n_n, gN=gN_n,
                Xq=Xq, pN_q=pNq, wds=wds_cat, F_line=F_line, tT=tr["tT"])
    return f, Kc, diag


if __name__ == "__main__":
    from .traction import TractionField

    # --- self-test: a single flat segment under uniform penetration ---
    pen0, eps_n = 0.02, 500.0
    surf = np.array([[0.0, 0.0], [1.0, 0.0]])
    node_ids = np.array([0, 1])

    def eval_gap(X):
        Q = len(X)
        return np.full(Q, -pen0), np.tile([0.0, 1.0], (Q, 1))

    f, Kc, diag = assemble_contact(surf, node_ids, 4, eval_gap, TractionField(eps_n), order=3)
    pN = eps_n * pen0
    assert np.allclose(f[0], [0.0, 0.5 * pN]), f[0]
    assert np.allclose(f[1], [0.0, 0.5 * pN]), f[1]
    assert np.allclose(f.sum(axis=0), [0.0, pN * 1.0]), f.sum(axis=0)
    assert abs(diag["F_line"] - pN * 1.0) < 1e-12, diag["F_line"]
    Kd = Kc.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12), "K_c must be symmetric"
    yy = Kd[np.ix_([1, 3], [1, 3])]
    expect = eps_n * 1.0 * np.array([[1.0 / 3, 1.0 / 6], [1.0 / 6, 1.0 / 3]])
    assert np.allclose(yy, expect, atol=1e-10), (yy, expect)
    assert np.linalg.eigvalsh(Kd).min() > -1e-12, "K_c must be SPSD"
    print("  assemble_contact self-test OK: 50/50 force split, F_line exact,")
    print("    consistent mass-matrix coupling, symmetric SPSD tangent")
