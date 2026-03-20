#!/usr/bin/env python3
"""Reconstruct physical torus elastoplastic fields from saved atlas checkpoints.

This module is the source of truth for torus forward-BVP diagnostics and
postprocessing.  It rebuilds the chart solvers for a saved checkpoint,
reconstructs both chart-space and physical-space kinematics, clusters
coincident physical nodes across overlapping charts, and computes a compact
diagnostic report that helps distinguish solver issues from export artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.spatial import cKDTree

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.run_forward_bvp_schwarz import (
    N_CHARTS,
    PHI_HALFWIDTH,
    R_MAJOR,
    R_MINOR,
    TorusChartDecoder,
    TorusSDF,
)


@dataclass
class ReconstructionResult:
    checkpoint_path: Path
    config: Dict[str, Any]
    charts: List[Dict[str, Any]]
    unique_points: Dict[str, np.ndarray]
    unique_surface_points: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    classification: str


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _infer_n_cells_from_checkpoint(ckpt: Dict[str, Any]) -> int:
    n_nodes = int(ckpt["u_charts"][0].shape[0])
    npa = round(n_nodes ** (1.0 / 3.0))
    if npa ** 3 != n_nodes:
        raise ValueError(
            f"Could not infer n_cells from {n_nodes} nodes; expected a cubic node count."
        )
    return npa - 1


def _load_checkpoint(checkpoint_path: str | Path) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def build_torus_context_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Rebuild the torus chart context needed to interpret a saved checkpoint."""
    ckpt = _load_checkpoint(checkpoint_path)
    n_charts = int(ckpt.get("n_charts", len(ckpt["u_charts"])))
    n_cells = int(ckpt.get("n_cells", _infer_n_cells_from_checkpoint(ckpt)))
    support_r = float(ckpt.get("support_r", 1.0))
    phi_halfwidth = float(ckpt.get("phi_halfwidth", PHI_HALFWIDTH))
    phi_centers = ckpt.get(
        "chart_phi_centers",
        [i * 2.0 * math.pi / n_charts for i in range(n_charts)],
    )

    sdf = TorusSDF()
    decoders = [
        TorusChartDecoder(phi_center=float(phi_c), phi_halfwidth=phi_halfwidth)
        for phi_c in phi_centers
    ]
    solvers = [
        ChartVectorFEMSolver(
            n_cells=n_cells,
            support_r=support_r,
            chart_decoder=dec,
            sdf_oracle=sdf,
            sdf_threshold=-0.005,
            device=device,
            dtype=dtype,
        )
        for dec in decoders
    ]

    return {
        "checkpoint": ckpt,
        "n_charts": n_charts,
        "n_cells": n_cells,
        "support_r": support_r,
        "phi_halfwidth": phi_halfwidth,
        "phi_centers": [float(v) for v in phi_centers],
        "solvers": solvers,
        "decoders": decoders,
    }


def _cluster_points(points: np.ndarray, decimals: int = 10) -> Dict[str, Any]:
    rounded = np.round(points, decimals=decimals)
    unique_points, inverse, counts = np.unique(
        rounded, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse)
    splits = np.split(order, np.cumsum(counts[:-1]))
    return {
        "unique_points": unique_points,
        "inverse": inverse,
        "counts": counts.astype(np.int64),
        "groups": splits,
    }


def _aggregate_field(
    values: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(values.shape[0], -1)
    n_clusters = counts.shape[0]
    sums = np.zeros((n_clusters, flat.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, flat)
    means = sums / counts[:, None]
    diff_norm = np.linalg.norm(flat - means[inverse], axis=1)
    spread = np.zeros(n_clusters, dtype=np.float64)
    np.maximum.at(spread, inverse, diff_norm)
    means = means.reshape((n_clusters,) + values.shape[1:])
    return means, spread


def _recover_plastic_metrics(
    F_phys: np.ndarray,
    Be: np.ndarray,
) -> Dict[str, np.ndarray]:
    n_elem = F_phys.shape[0]
    det_Fp = np.zeros(n_elem, dtype=np.float64)
    eig1_Sp = np.zeros(n_elem, dtype=np.float64)
    eig_diff_Sp = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        Be_e = torch.from_numpy(Be[e])
        F_e = torch.from_numpy(F_phys[e])
        eigvals, eigvecs = torch.linalg.eigh(Be_e)
        eigvals = eigvals.clamp(min=1e-30)
        Fe_e = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        Fp_e = torch.linalg.solve(Fe_e, F_e)
        det_Fp[e] = torch.det(Fp_e).item()

        U, D, Vt = torch.linalg.svd(Fp_e)
        if torch.det(U @ Vt) < 0:
            U = U.clone()
            D = D.clone()
            U[:, -1] *= -1.0
            D[-1] *= -1.0
        Sp_e = Vt.T @ torch.diag(D) @ Vt
        eigs = np.sort(_to_numpy(torch.linalg.eigvalsh(Sp_e)))[::-1]
        eig1_Sp[e] = eigs[0]
        eig_diff_Sp[e] = eigs[0] - eigs[-1]

    return {
        "det_Fp": det_Fp,
        "eig1_Sp": eig1_Sp,
        "eig_diff_Sp": eig_diff_Sp,
    }


def _nearest_other_chart_spread(
    centroids: np.ndarray,
    chart_ids: np.ndarray,
    tensors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(centroids)
    k = min(8, centroids.shape[0])
    dists, idxs = tree.query(centroids, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    spread = np.zeros(centroids.shape[0], dtype=np.float64)
    other_dist = np.full(centroids.shape[0], np.inf, dtype=np.float64)
    tensors_flat = tensors.reshape(tensors.shape[0], -1)

    for i in range(centroids.shape[0]):
        for dist, j in zip(dists[i, 1:], idxs[i, 1:]):
            if chart_ids[j] == chart_ids[i]:
                continue
            other_dist[i] = dist
            denom = max(np.linalg.norm(tensors_flat[j]), 1e-12)
            spread[i] = np.linalg.norm(tensors_flat[i] - tensors_flat[j]) / denom
            break

    finite = np.isfinite(other_dist)
    if np.any(finite):
        distance_cutoff = np.percentile(other_dist[finite], 80.0)
        spread[~finite] = 0.0
        spread[other_dist > distance_cutoff] = 0.0
        other_dist[~finite] = 0.0
    else:
        other_dist[:] = 0.0

    return spread, other_dist


def reconstruct_checkpoint(
    checkpoint_path: str | Path,
    *,
    round_decimals: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ReconstructionResult:
    """Reconstruct raw chart fields and blended physical point fields."""
    context = build_torus_context_from_checkpoint(
        checkpoint_path, device=device, dtype=dtype
    )
    ckpt = context["checkpoint"]
    solvers = context["solvers"]
    decoders = context["decoders"]

    chart_entries: List[Dict[str, Any]] = []
    point_positions: List[np.ndarray] = []
    point_u: List[np.ndarray] = []
    point_boundary: List[np.ndarray] = []
    point_chart_ids: List[np.ndarray] = []

    cell_centroids: List[np.ndarray] = []
    cell_chart_ids: List[np.ndarray] = []
    cell_F_phys: List[np.ndarray] = []

    for ci, solver in enumerate(solvers):
        u = ckpt["u_charts"][ci].to(device=device, dtype=dtype)
        state = ckpt["states"][ci]
        Be = state["Be"].to(device=device, dtype=dtype)
        ep_bar = state["ep_bar"].to(device=device, dtype=dtype)
        beta = state["beta"].to(device=device, dtype=dtype)

        F_ref = solver.compute_F(u, physical=False)
        F_phys = solver.compute_F(u, physical=True)
        grad_u_ref = solver.compute_grad_u_ref(u)
        grad_u_phys = solver.compute_grad_u_phys(u)
        decoder_J = solver.decoder_jacobian(solver.elem_centroids_ref)

        F_ref_np = _to_numpy(F_ref)
        F_phys_np = _to_numpy(F_phys)
        grad_ref_np = _to_numpy(grad_u_ref)
        grad_phys_np = _to_numpy(grad_u_phys)
        Be_np = _to_numpy(Be)
        ep_np = _to_numpy(ep_bar)
        beta_np = _to_numpy(beta)
        nodes_ref = _to_numpy(solver.nodes)
        nodes_phys = _to_numpy(solver.nodes_phys)
        centroids_ref = _to_numpy(solver.elem_centroids_ref)
        centroids_phys = _to_numpy(solver.elem_centroids_phys)
        decoder_J_np = _to_numpy(decoder_J)
        decoder_detJ = np.linalg.det(decoder_J_np)
        geom_J_np = _to_numpy(solver.geom_J)
        geom_detJ = _to_numpy(solver.geom_detJ)

        rel = np.linalg.norm(
            F_ref_np.reshape(F_ref_np.shape[0], -1) - F_phys_np.reshape(F_phys_np.shape[0], -1),
            axis=1,
        ) / np.maximum(
            np.linalg.norm(F_phys_np.reshape(F_phys_np.shape[0], -1), axis=1),
            1e-12,
        )

        plastic = _recover_plastic_metrics(F_phys_np, Be_np)

        point_positions.append(nodes_phys)
        point_u.append(_to_numpy(u))
        point_boundary.append(_to_numpy(solver.boundary_mask))
        point_chart_ids.append(np.full(solver.n_nodes, ci, dtype=np.int64))

        cell_centroids.append(centroids_phys)
        cell_chart_ids.append(np.full(solver.n_elements, ci, dtype=np.int64))
        cell_F_phys.append(F_phys_np)

        chart_entries.append(
            {
                "chart_id": ci,
                "phi_center": context["phi_centers"][ci],
                "solver": solver,
                "decoder": decoders[ci],
                "elements": _to_numpy(solver.elements).astype(np.int64),
                "nodes_ref": nodes_ref,
                "nodes_phys": nodes_phys,
                "boundary_mask": _to_numpy(solver.boundary_mask).astype(bool),
                "u": _to_numpy(u),
                "grad_u_ref": grad_ref_np,
                "grad_u_phys": grad_phys_np,
                "F_ref": F_ref_np,
                "F_phys": F_phys_np,
                "Be": Be_np,
                "ep_bar": ep_np,
                "beta": beta_np,
                "centroids_ref": centroids_ref,
                "centroids_phys": centroids_phys,
                "decoder_J": decoder_J_np,
                "decoder_detJ": decoder_detJ,
                "geom_J": geom_J_np,
                "geom_detJ": geom_detJ,
                "det_F_ref": np.linalg.det(F_ref_np),
                "det_F_phys": np.linalg.det(F_phys_np),
                "F_ref_phys_rel": rel,
                "det_Fp": plastic["det_Fp"],
                "eig1_Sp": plastic["eig1_Sp"],
                "eig_diff_Sp": plastic["eig_diff_Sp"],
            }
        )

    all_points = np.vstack(point_positions)
    all_u = np.vstack(point_u)
    all_boundary = np.concatenate(point_boundary)
    all_point_chart_ids = np.concatenate(point_chart_ids)
    clusters = _cluster_points(all_points, decimals=round_decimals)
    unique_coords, point_spread = _aggregate_field(
        all_points, clusters["inverse"], clusters["counts"]
    )
    unique_displacement, u_spread = _aggregate_field(
        all_u, clusters["inverse"], clusters["counts"]
    )
    boundary_any = np.zeros(clusters["counts"].shape[0], dtype=bool)
    chart_support = np.zeros(clusters["counts"].shape[0], dtype=np.int64)
    for cid, idx in enumerate(clusters["groups"]):
        boundary_any[cid] = bool(np.any(all_boundary[idx]))
        chart_support[cid] = len(np.unique(all_point_chart_ids[idx]))

    unique_data = {
        "points": unique_coords,
        "point_overlap_spread": point_spread,
        "displacement": unique_displacement,
        "displacement_mag": np.linalg.norm(unique_displacement, axis=1),
        "u_overlap_spread": u_spread,
        "multiplicity": clusters["counts"],
        "chart_support": chart_support,
        "boundary_mask": boundary_any,
        "cluster_inverse": clusters["inverse"],
    }
    unique_surface = {
        key: value[boundary_any]
        for key, value in unique_data.items()
        if isinstance(value, np.ndarray) and value.shape[0] == boundary_any.shape[0]
    }

    # Push point overlap spreads back to each chart node for debugging exports.
    offset = 0
    for chart in chart_entries:
        n_local = chart["nodes_phys"].shape[0]
        cluster_ids = clusters["inverse"][offset:offset + n_local]
        chart["node_cluster_ids"] = cluster_ids
        chart["node_u_overlap_spread"] = u_spread[cluster_ids]
        chart["node_chart_support"] = chart_support[cluster_ids]
        offset += n_local

    all_centroids = np.vstack(cell_centroids)
    all_cell_chart_ids = np.concatenate(cell_chart_ids)
    all_F_phys = np.vstack(cell_F_phys)
    F_overlap_spread, nearest_other_dist = _nearest_other_chart_spread(
        all_centroids,
        all_cell_chart_ids,
        all_F_phys,
    )

    offset = 0
    for chart in chart_entries:
        n_local = chart["centroids_phys"].shape[0]
        chart["F_overlap_spread"] = F_overlap_spread[offset:offset + n_local]
        chart["nearest_other_chart_distance"] = nearest_other_dist[offset:offset + n_local]
        offset += n_local

    rel_all = np.concatenate([chart["F_ref_phys_rel"] for chart in chart_entries])
    det_phys_all = np.concatenate([chart["det_F_phys"] for chart in chart_entries])
    overlap_mask = unique_data["multiplicity"] > 1
    max_disp = max(float(unique_data["displacement_mag"].max(initial=0.0)), 1e-12)
    u_spread_ref = unique_data["u_overlap_spread"][overlap_mask] if np.any(overlap_mask) else np.zeros(1)
    F_spread_all = np.concatenate([chart["F_overlap_spread"] for chart in chart_entries])

    metrics = {
        "median_F_ref_vs_F_phys_rel": float(np.median(rel_all)),
        "p95_F_ref_vs_F_phys_rel": float(np.percentile(rel_all, 95.0)),
        "max_F_ref_vs_F_phys_rel": float(np.max(rel_all)),
        "p95_u_overlap_spread": float(np.percentile(u_spread_ref, 95.0)),
        "p95_u_overlap_spread_rel_to_max_u": float(np.percentile(u_spread_ref, 95.0) / max_disp),
        "p95_F_overlap_spread": float(np.percentile(F_spread_all, 95.0)),
        "negative_det_F_phys_fraction": float(np.mean(det_phys_all <= 0.0)),
        "max_displacement": float(max_disp),
        "overlap_cluster_fraction": float(np.mean(overlap_mask.astype(np.float64))),
    }

    solver_issue = (
        metrics["median_F_ref_vs_F_phys_rel"] > 0.05
        or metrics["p95_F_ref_vs_F_phys_rel"] > 0.20
        or metrics["p95_u_overlap_spread_rel_to_max_u"] > 0.05
        or metrics["negative_det_F_phys_fraction"] > 1e-3
    )
    classification = "solver_kinematics_issue" if solver_issue else "postprocessing_only"

    return ReconstructionResult(
        checkpoint_path=Path(checkpoint_path),
        config={
            "step": int(ckpt.get("step", -1)),
            "delta": float(ckpt.get("delta", 0.0)),
            "n_cells": context["n_cells"],
            "n_charts": context["n_charts"],
            "support_r": context["support_r"],
            "phi_halfwidth": context["phi_halfwidth"],
            "phi_centers": context["phi_centers"],
        },
        charts=chart_entries,
        unique_points=unique_data,
        unique_surface_points=unique_surface,
        metrics=metrics,
        classification=classification,
    )


def charts_to_multiblock(result: ReconstructionResult):
    """Convert raw chart fields to a PyVista MultiBlock for debugging."""
    import pyvista as pv

    blocks = pv.MultiBlock()
    for chart in result.charts:
        elements = chart["elements"]
        n_elem = elements.shape[0]
        cells = np.hstack([np.full((n_elem, 1), 4, dtype=np.int64), elements]).ravel()
        cell_types = np.full(n_elem, pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, cell_types, chart["nodes_phys"])
        grid.point_data["displacement"] = chart["u"]
        grid.point_data["displacement_mag"] = np.linalg.norm(chart["u"], axis=1)
        grid.point_data["u_overlap_spread"] = chart["node_u_overlap_spread"]
        grid.point_data["chart_support"] = chart["node_chart_support"]
        grid.cell_data["ep_bar"] = chart["ep_bar"]
        grid.cell_data["det_F_ref"] = chart["det_F_ref"]
        grid.cell_data["det_F_phys"] = chart["det_F_phys"]
        grid.cell_data["F_ref_phys_rel"] = chart["F_ref_phys_rel"]
        grid.cell_data["F_overlap_spread"] = chart["F_overlap_spread"]
        grid.cell_data["det_Fp"] = chart["det_Fp"]
        grid.cell_data["eig1_Sp"] = chart["eig1_Sp"]
        grid.cell_data["eig_diff_Sp"] = chart["eig_diff_Sp"]
        grid.cell_data["chart_id"] = np.full(n_elem, chart["chart_id"], dtype=np.int64)
        blocks.append(grid, f"chart_{chart['chart_id']}")
    return blocks


def unique_points_to_polydata(data: Dict[str, np.ndarray]):
    """Convert a unique-point dataset to PyVista PolyData."""
    import pyvista as pv

    poly = pv.PolyData(data["points"])
    for key, value in data.items():
        if key in {"points", "cluster_inverse"}:
            continue
        poly.point_data[key] = value
    return poly


def unique_points_to_unstructured_grid(data: Dict[str, np.ndarray]):
    """Convert a unique-point dataset to a vertex-based VTU grid."""
    import pyvista as pv

    n_points = data["points"].shape[0]
    cells = np.hstack(
        [np.ones((n_points, 1), dtype=np.int64), np.arange(n_points, dtype=np.int64)[:, None]]
    ).ravel()
    cell_types = np.full(n_points, pv.CellType.VERTEX, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, data["points"])
    for key, value in data.items():
        if key in {"points", "cluster_inverse"}:
            continue
        grid.point_data[key] = value
    return grid


def make_torus_surface(n_phi: int = 200, n_theta: int = 80):
    """Create a smooth torus surface for field visualization."""
    import pyvista as pv

    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    X = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.cos(PHI)
    Y = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.sin(PHI)
    Z = R_MINOR * np.sin(THETA)
    grid = pv.StructuredGrid(X, Y, Z)
    return grid.extract_surface(algorithm=None), PHI, THETA


def _polar_decompose(F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    U, D, Vt = torch.linalg.svd(F)
    if torch.det(U @ Vt) < 0:
        U = U.clone()
        D = D.clone()
        U[:, -1] *= -1.0
        D[-1] *= -1.0
    return U @ Vt, Vt.T @ torch.diag(D) @ Vt


def _log_rotation(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map SO(3) -> so(3) following Mota et al. (2013), Eq. (5.2).

    Handles three regimes:
      theta ≈ 0:  log R = 0  (identity rotation)
      theta in (0, pi):  log R = (theta / 2 sin theta) (R - R^T)
      theta ≈ pi: log R = ±pi * hat(v), where v is the eigenvector of R
                  for eigenvalue 1, sign chosen by continuity with (R - R^T).
    """
    tr = torch.trace(R)
    # cos(theta) = (tr - 1) / 2, clamp to valid range
    cos_theta = (0.5 * (tr - 1.0)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    # Case 1: theta ≈ 0 — near identity
    if theta.abs() < 1e-10:
        return torch.zeros(3, 3, dtype=R.dtype)

    # Case 2: theta ≈ pi — sinθ → 0, use eigenvector method
    if theta > math.pi - 1e-6:
        # R has eigenvalue 1 with eigenvector v.
        # Compute from R + I (which has rank ≥ 1 when θ = π):
        # pick the column of (R + I) with largest norm as v.
        B = R + torch.eye(3, dtype=R.dtype)
        norms = torch.norm(B, dim=0)
        col = torch.argmax(norms).item()
        v = B[:, col]
        v = v / torch.norm(v)
        # hat(v): skew-symmetric matrix such that hat(v) x = v × x
        W = torch.zeros(3, 3, dtype=R.dtype)
        W[0, 1] = -v[2]; W[0, 2] = v[1]
        W[1, 0] = v[2];  W[1, 2] = -v[0]
        W[2, 0] = -v[1]; W[2, 1] = v[0]
        # Choose sign consistent with (R - R^T)
        skew = R - R.T
        if torch.sum(skew * W) < 0:
            W = -W
        return math.pi * W

    # Case 3: generic — standard formula
    return (theta / (2.0 * torch.sin(theta))) * (R - R.T)


def _exp_rotation(W: torch.Tensor) -> torch.Tensor:
    """Exponential map so(3) -> SO(3), Rodrigues formula (Mota et al. Eq. 5.6).

    Input W is skew-symmetric (or approximately so after interpolation).
    """
    # Ensure skew-symmetry
    W = 0.5 * (W - W.T)
    theta = torch.sqrt(0.5 * torch.sum(W * W)).clamp(min=1e-30)
    if theta < 1e-10:
        return torch.eye(3, dtype=W.dtype)
    I = torch.eye(3, dtype=W.dtype)
    return I + (torch.sin(theta) / theta) * W + ((1.0 - torch.cos(theta)) / theta**2) * (W @ W)


def _log_spd(S: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = eigvals.clamp(min=1e-30)
    return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T


def _exp_spd(H: torch.Tensor) -> torch.Tensor:
    H = 0.5 * (H + H.T)
    if torch.any(torch.isnan(H)):
        return torch.eye(3, dtype=H.dtype)
    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = eigvals.clamp(-20.0, 20.0)
    return eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T


def _l2_project_to_nodes(
    elements: np.ndarray,
    volumes: np.ndarray,
    n_nodes: int,
    elem_values: np.ndarray,
) -> np.ndarray:
    """Global L2 projection of element-centroid values to P1 nodal values.

    Assembles the consistent P1 mass matrix M and right-hand side f,
    then solves M z_node = f via sparse direct solve.

    Parameters
    ----------
    elements : (n_elem, 4) int array — tetrahedral connectivity
    volumes : (n_elem,) float array — element volumes
    n_nodes : int — total number of nodes
    elem_values : (n_elem,) float array — one scalar field at centroids

    Returns
    -------
    nodal_values : (n_nodes,) float array
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    n_elem = elements.shape[0]

    # Consistent P1 tet mass matrix:
    #   M_{ab} = sum_e V_e * (1 + delta_{ab}) / 20   for nodes a,b in element e
    rows = []
    cols = []
    vals = []
    for a in range(4):
        for b in range(4):
            factor = 2.0 / 20.0 if a == b else 1.0 / 20.0
            rows.append(elements[:, a])
            cols.append(elements[:, b])
            vals.append(volumes * factor)

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    val_arr = np.concatenate(vals)
    M = coo_matrix((val_arr, (row_idx, col_idx)), shape=(n_nodes, n_nodes)).tocsr()

    # RHS: f_a = sum_{e containing a} V_e / 4 * z_e  (1-point centroid quadrature)
    f = np.zeros(n_nodes, dtype=np.float64)
    contrib = (volumes * elem_values) / 4.0
    for a in range(4):
        np.add.at(f, elements[:, a], contrib)

    return spsolve(M, f)


def _matrix_log_taylor(B: np.ndarray, n_terms: int = 12) -> np.ndarray:
    """Matrix logarithm via Taylor series (Ortiz 2002, Eq. 8.1.2).

    log(B) = sum_{k=1}^{n_terms} (-1)^{k-1}/k * (B - I)^k

    Convergent when ||B - I|| < 1 (i.e. B close to identity).
    """
    I = np.eye(3, dtype=B.dtype)
    X = B - I  # (B - I)
    result = np.zeros_like(B)
    power = I.copy()  # will accumulate (B - I)^k
    for k in range(1, n_terms + 1):
        power = power @ X
        result += ((-1.0) ** (k - 1) / k) * power
    return result


def _matrix_exp_taylor(A: np.ndarray, n_terms: int = 12) -> np.ndarray:
    """Matrix exponential via Taylor series (Ortiz 2002, Eq. 8.1.1).

    exp(A) = sum_{k=0}^{n_terms} (1/k!) * A^k
    """
    I = np.eye(3, dtype=A.dtype)
    result = I.copy()
    term = I.copy()  # A^k / k!
    for k in range(1, n_terms + 1):
        term = term @ A / k
        result += term
    return result


def _compute_chart_Fp_fields(chart: Dict[str, Any]):
    """Compute per-element F^p and its matrix logarithm via Taylor expansion.

    Uses Ortiz (2002) Eq. 8.1.2 for log(F^p) — no polar decomposition needed.
    The Taylor series converges when ||F^p - I|| < 1 (small plastic deformation).
    """
    F_phys = chart["F_phys"]
    Be = chart["Be"]
    n_elem = F_phys.shape[0]

    Fp_arr = np.zeros((n_elem, 3, 3), dtype=np.float64)
    log_Fp = np.zeros((n_elem, 3, 3), dtype=np.float64)

    for e in range(n_elem):
        Feig, Fvec = torch.linalg.eigh(torch.from_numpy(Be[e]))
        Feig = Feig.clamp(min=1e-30)
        Fe = Fvec @ torch.diag(torch.sqrt(Feig)) @ Fvec.T
        Fp = torch.linalg.solve(Fe, torch.from_numpy(F_phys[e]))
        Fp_np = _to_numpy(Fp)
        Fp_arr[e] = Fp_np
        log_Fp[e] = _matrix_log_taylor(Fp_np)

    return Fp_arr, log_Fp


def interpolate_plastic_surface_fields(
    result: ReconstructionResult,
    *,
    n_phi: int = 200,
    n_theta: int = 80,
):
    """Interpolate F^p fields to a smooth torus surface via global L2 projection.

    Uses Taylor series expansion (Ortiz 2002, Eq. 8.1.1-8.1.2) instead of
    polar decomposition + Lie algebra:
    - Compute F^p = Fe^{-1} F at each element centroid
    - Take log(F^p) via Taylor series (no polar decomposition needed)
    - L2-project log(F^p) components to nodes
    - Interpolate to visualization surface via griddata
    - Recover F^p = exp(log(F^p)) via Taylor series
    """
    from scipy.interpolate import griddata

    surf, _, _ = make_torus_surface(n_phi=n_phi, n_theta=n_theta)
    surface_points = np.asarray(surf.points)
    n_surface = surface_points.shape[0]

    # Step 1: Compute element-centroid F^p and log(F^p), L2-project to nodes
    all_node_xyz: List[np.ndarray] = []
    all_node_log_Fp: List[np.ndarray] = []
    all_node_ep_bar: List[np.ndarray] = []

    for chart in result.charts:
        Fp_elem, log_Fp_elem = _compute_chart_Fp_fields(chart)
        elements = chart["elements"]
        nodes_phys = chart["nodes_phys"]
        n_nodes = nodes_phys.shape[0]

        solver = chart["solver"]
        vol = _to_numpy(solver.vol_phys)

        # L2 project each component of log(F^p) to nodes
        log_Fp_nodes = np.zeros((n_nodes, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                log_Fp_nodes[:, i, j] = _l2_project_to_nodes(
                    elements, vol, n_nodes, log_Fp_elem[:, i, j])

        # L2 project ep_bar (accumulated plastic strain) to nodes
        ep_bar_nodes = _l2_project_to_nodes(
            elements, vol, n_nodes, chart["ep_bar"].ravel())

        all_node_xyz.append(nodes_phys)
        all_node_log_Fp.append(log_Fp_nodes)
        all_node_ep_bar.append(ep_bar_nodes)

    # Step 2: Pool all chart nodal values and interpolate to surface
    src_xyz = np.vstack(all_node_xyz)
    src_log_Fp = np.vstack(all_node_log_Fp)
    src_ep_bar = np.concatenate(all_node_ep_bar)

    log_Fp_surface = np.zeros((n_surface, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            vals_lin = griddata(
                src_xyz, src_log_Fp[:, i, j], surface_points,
                method="linear", fill_value=np.nan,
            )
            vals_nn = griddata(
                src_xyz, src_log_Fp[:, i, j], surface_points,
                method="nearest",
            )
            mask = np.isnan(vals_lin)
            log_Fp_surface[:, i, j] = np.where(mask, vals_nn, vals_lin)

    # Interpolate ep_bar
    ep_lin = griddata(src_xyz, src_ep_bar, surface_points,
                      method="linear", fill_value=np.nan)
    ep_nn = griddata(src_xyz, src_ep_bar, surface_points, method="nearest")
    ep_bar_surface = np.where(np.isnan(ep_lin), ep_nn, ep_lin)

    # Step 3: Recover F^p = exp(log(F^p)) via Taylor series, compute diagnostics
    det_Fp = np.zeros(n_surface, dtype=np.float64)
    eig1_Sp = np.zeros(n_surface, dtype=np.float64)
    eig_diff_Sp = np.zeros(n_surface, dtype=np.float64)
    iso_eig1 = np.zeros(n_surface, dtype=np.float64)
    iso_eig2 = np.zeros(n_surface, dtype=np.float64)
    iso_eig3 = np.zeros(n_surface, dtype=np.float64)

    for idx in range(n_surface):
        Fp = _matrix_exp_taylor(log_Fp_surface[idx])
        Jp = np.linalg.det(Fp)
        det_Fp[idx] = Jp

        # Right stretch via polar decomposition of recovered F^p for eigenvalue diagnostics
        C = Fp.T @ Fp  # right Cauchy-Green of F^p
        C = 0.5 * (C + C.T)
        eigs = np.sort(np.linalg.eigvalsh(C))[::-1]
        # eigenvalues of C = λ²; stretch eigenvalues = sqrt(λ²)
        stretch_eigs = np.sqrt(np.maximum(eigs, 0.0))
        eig1_Sp[idx] = stretch_eigs[0]
        eig_diff_Sp[idx] = stretch_eigs[0] - stretch_eigs[-1]

        Jp_abs = max(abs(Jp), 1e-30)
        stretch_bar = stretch_eigs / (Jp_abs ** (1.0 / 3.0))
        iso_eig1[idx] = stretch_bar[0]
        iso_eig2[idx] = stretch_bar[1]
        iso_eig3[idx] = stretch_bar[2]

    surf.point_data["det_Fp"] = det_Fp
    surf.point_data["eig1_Sp"] = eig1_Sp
    surf.point_data["eig_diff_Sp"] = eig_diff_Sp
    surf.point_data["iso_eig1"] = iso_eig1
    surf.point_data["iso_eig2"] = iso_eig2
    surf.point_data["iso_eig3"] = iso_eig3
    surf.point_data["ep_bar"] = ep_bar_surface

    # Attach per-chart projected data for VTU export
    surf._l2_chart_data = []
    for ci, chart in enumerate(result.charts):
        surf._l2_chart_data.append({
            "chart_id": ci,
            "elements": chart["elements"],
            "nodes_phys": all_node_xyz[ci],
            "log_Fp": all_node_log_Fp[ci],
            "ep_bar": all_node_ep_bar[ci],
        })

    return surf


def export_l2_projected_vtk(surf, out_dir):
    """Export L2-projected F^p fields as VTU files for ParaView inspection.

    Saves per-chart tetrahedral grids with L2-projected nodal fields.
    Returns a MultiBlock of the per-chart grids for rendering.
    """
    import pyvista as pv

    out_dir = Path(out_dir)
    l2_dir = out_dir / "vtu" / "l2_projected"
    l2_dir.mkdir(parents=True, exist_ok=True)

    chart_data_list = getattr(surf, "_l2_chart_data", [])
    blocks = pv.MultiBlock()

    for cdata in chart_data_list:
        ci = cdata["chart_id"]
        elements = cdata["elements"]
        nodes_phys = cdata["nodes_phys"]
        log_Fp = cdata["log_Fp"]
        n_elem = elements.shape[0]
        n_nodes = nodes_phys.shape[0]

        cells = np.hstack([np.full((n_elem, 1), 4, dtype=np.int64), elements]).ravel()
        cell_types = np.full(n_elem, pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, cell_types, nodes_phys)

        # Recover per-node F^p diagnostics from log(F^p) via Taylor exp
        det_Fp_node = np.zeros(n_nodes, dtype=np.float64)
        eig1_node = np.zeros(n_nodes, dtype=np.float64)
        eig_diff_node = np.zeros(n_nodes, dtype=np.float64)
        for n in range(n_nodes):
            Fp = _matrix_exp_taylor(log_Fp[n])
            det_Fp_node[n] = np.linalg.det(Fp)
            C = Fp.T @ Fp
            C = 0.5 * (C + C.T)
            eigs = np.sort(np.linalg.eigvalsh(C))[::-1]
            stretch = np.sqrt(np.maximum(eigs, 0.0))
            eig1_node[n] = stretch[0]
            eig_diff_node[n] = stretch[0] - stretch[-1]

        grid.point_data["det_Fp"] = det_Fp_node
        grid.point_data["eig1_Sp"] = eig1_node
        grid.point_data["eig_diff_Sp"] = eig_diff_node
        grid.point_data["log_Fp"] = log_Fp.reshape(n_nodes, 9)
        if "ep_bar" in cdata:
            grid.point_data["ep_bar"] = cdata["ep_bar"]
        grid.cell_data["chart_id"] = np.full(n_elem, ci, dtype=np.int64)

        grid.save(str(l2_dir / f"chart_{ci:02d}_l2_projected.vtu"))
        blocks.append(grid, f"chart_{ci}")

    print(f"  L2-projected VTU files saved to {l2_dir}")
    return blocks


def _write_report(result: ReconstructionResult, output_path: str | Path) -> None:
    payload = {
        "checkpoint": str(result.checkpoint_path),
        "config": result.config,
        "classification": result.classification,
        "metrics": result.metrics,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))


def find_checkpoints(run_dir: str | Path) -> List[Path]:
    return sorted(Path(run_dir).glob("checkpoint_step*.pt"))


def find_latest_checkpoint(run_dir: str | Path) -> Path:
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_step*.pt files found in {run_dir}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Reconstruct torus atlas fields from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint_step*.pt. If omitted, uses the latest checkpoint in --run-dir.")
    parser.add_argument("--run-dir", type=str, default="runs/torus_forward_bvp_debug_small_max25",
                        help="Run directory used when --checkpoint is omitted.")
    parser.add_argument("--report-json", type=str, default=None,
                        help="Optional path for a JSON diagnostic report.")
    parser.add_argument("--round-decimals", type=int, default=10)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(Path(args.run_dir))
    result = reconstruct_checkpoint(checkpoint, round_decimals=args.round_decimals)
    print(f"Checkpoint:      {result.checkpoint_path}")
    print(f"Classification:  {result.classification}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6e}")

    if args.report_json:
        _write_report(result, args.report_json)
        print(f"Report written to {args.report_json}")


if __name__ == "__main__":
    main()
