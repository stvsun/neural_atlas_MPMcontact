#!/usr/bin/env python3
"""
P1 tetrahedral FEM solver for mapped Poisson on one chart's reference domain.

Designed as a drop-in replacement for the PINN local solver in the Schwarz
alternating framework.  Uses the *same* frozen atlas (ChartDecoder, MaskNet)
and the *same* mapped PDE:

    -∇_ζ · (A_i(ζ) ∇_ζ û) = j_i(ζ) f(φ_i(ζ))

where  A_i = j_i J_i⁻¹ J_i⁻ᵀ ,  J_i = ∂φ_i/∂ζ ,  j_i = |det J_i|.

The reference domain [-r, r]³ is meshed with a structured hex grid split into
tetrahedra (Freudenthal decomposition), then SDF-filtered to retain only
elements inside the geometry.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch


# ── Freudenthal hex-to-tet splitting ─────────────────────────────────────────
# Each hex cell is split into 6 tets using a *consistent diagonal*
# orientation (the body diagonal connecting vertex 0 to vertex 7).  The
# 8 vertices of a hex cell are numbered:
#
#   v0 = (i,   j,   k  )    v4 = (i,   j,   k+1)
#   v1 = (i+1, j,   k  )    v5 = (i+1, j,   k+1)
#   v2 = (i,   j+1, k  )    v6 = (i,   j+1, k+1)
#   v3 = (i+1, j+1, k  )    v7 = (i+1, j+1, k+1)
#
# The 6 tets sharing the body diagonal (v0, v7):
_FREUDENTHAL_TETS = np.array([
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 4, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 6, 7],
], dtype=np.int64)


def _hex_vertex_offsets(n1: int) -> np.ndarray:
    """Return the 8 index offsets (into a flattened (n+1)³ grid) for a hex
    cell at grid position (i, j, k), where n1 = n_cells + 1."""
    # Vertex (di, dj, dk) has flat index di + dj*n1 + dk*n1*n1
    offsets = np.zeros(8, dtype=np.int64)
    for idx, (di, dj, dk) in enumerate(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    ):
        offsets[idx] = di + dj * n1 + dk * n1 * n1
    return offsets


class ChartFEMSolver:
    """P1 tetrahedral FEM solver for mapped Poisson on one chart's reference domain.

    Parameters
    ----------
    chart_id : int
        Index of this chart in the atlas.
    decoder : torch.nn.Module
        Frozen ChartDecoder φ_i : ζ → x.
    seed, t1, t2, n_vec : torch.Tensor
        Chart frame vectors (shape (3,)).
    support_r : torch.Tensor
        Scalar support radius r_i.
    n_cells : int
        Number of grid divisions per axis (default 16).
    sdf_net : torch.nn.Module or None
        Frozen SDF network; input is normalised x, output is signed distance.
    sdf_center : torch.Tensor or None
        Centre for SDF normalisation.
    sdf_scale : float
        Scale for SDF normalisation.
    sdf_threshold : float
        Centroid SDF threshold for element inclusion (< threshold ⇒ inside).
    device : torch.device
        Compute device for the decoder evaluation.
    dtype : torch.dtype
        Floating point precision for the decoder.
    """

    def __init__(
        self,
        chart_id: int,
        decoder: torch.nn.Module,
        seed: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        n_vec: torch.Tensor,
        support_r: torch.Tensor,
        n_cells: int = 16,
        sdf_net: Optional[torch.nn.Module] = None,
        sdf_center: Optional[torch.Tensor] = None,
        sdf_scale: float = 1.0,
        sdf_threshold: float = -0.005,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.chart_id = chart_id
        self.decoder = decoder
        self.seed = seed
        self.t1 = t1
        self.t2 = t2
        self.n_vec = n_vec
        self.support_r = support_r
        self.r = float(support_r.item()) if isinstance(support_r, torch.Tensor) else float(support_r)
        self.n_cells = n_cells
        self.sdf_net = sdf_net
        self.sdf_center = sdf_center
        self.sdf_scale = sdf_scale
        self.sdf_threshold = sdf_threshold
        self.device = device
        self.dtype = dtype

        # Computed by build_mesh / compute_diffusion_tensors / assemble
        self.nodes: Optional[np.ndarray] = None          # (N_nodes, 3)
        self.elements: Optional[np.ndarray] = None        # (N_tets, 4)
        self.h: float = 0.0
        self.n_nodes: int = 0

        self.phys_bc_nodes: Optional[np.ndarray] = None   # indices
        self.art_bc_nodes: Optional[np.ndarray] = None     # indices
        self.interior_nodes: Optional[np.ndarray] = None   # indices

        self.A_elem: Optional[np.ndarray] = None          # (N_el, 3, 3)
        self.det_elem: Optional[np.ndarray] = None        # (N_el,)
        self.x_centroids: Optional[np.ndarray] = None     # (N_el, 3) physical coords

        self.K: Optional[scipy.sparse.csr_matrix] = None
        self.F: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None               # DOF solution vector

        # For structured-grid element lookup during interpolation
        self._grid_n1: int = 0
        self._old_to_new: Optional[np.ndarray] = None
        self._all_nodes_grid: Optional[np.ndarray] = None  # full grid before compaction

        self._build_mesh()

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Structured mesh generation
    # ──────────────────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        """Generate a structured hex grid on [-r, r]³, split into tets,
        and SDF-filter to retain only interior elements."""
        n = self.n_cells
        r = self.r
        n1 = n + 1
        self._grid_n1 = n1
        self.h = 2.0 * r / n

        # Grid node coordinates
        lin = np.linspace(-r, r, n1)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        all_nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # ((n+1)³, 3)
        self._all_nodes_grid = all_nodes

        # Build hex cells → tet elements
        hex_offsets = _hex_vertex_offsets(n1)
        # Hex cell base indices: (i, j, k) ∈ [0, n)³
        ci, cj, ck = np.meshgrid(
            np.arange(n), np.arange(n), np.arange(n), indexing="ij"
        )
        base = ci.ravel() + cj.ravel() * n1 + ck.ravel() * n1 * n1  # (n³,)
        hex_verts = base[:, None] + hex_offsets[None, :]  # (n³, 8)

        # Each hex → 6 tets
        all_tets = hex_verts[:, _FREUDENTHAL_TETS].reshape(-1, 4)  # (6*n³, 4)

        # SDF filtering: evaluate SDF at tet centroids (mapped to physical space).
        # For volumetric atlases, the fill fraction of each reference cube is
        # often very low (<5%), so SDF filtering is disabled by default
        # (sdf_net=None).  When enabled, the fallback keeps all elements if
        # none pass the filter.
        if self.sdf_net is not None and self.sdf_threshold is not None:
            centroids_xi = all_nodes[all_tets].mean(axis=1)  # (N_tets, 3)
            keep_mask = self._sdf_filter(centroids_xi)
            n_keep = int(keep_mask.sum())
            if n_keep > 0:
                all_tets = all_tets[keep_mask]
            else:
                print(f"  [Chart {self.chart_id}] SDF filter kept 0/{len(keep_mask)} "
                      f"elements; keeping full grid")

        # Compact: remove orphan nodes
        used_nodes = np.unique(all_tets)
        old_to_new = np.full(len(all_nodes), -1, dtype=np.int64)
        old_to_new[used_nodes] = np.arange(len(used_nodes))
        self._old_to_new = old_to_new

        self.nodes = all_nodes[used_nodes]
        self.elements = old_to_new[all_tets]
        self.n_nodes = len(self.nodes)

        # Classify nodes
        self._classify_nodes()

    def _sdf_filter(self, centroids_xi: np.ndarray) -> np.ndarray:
        """Return boolean mask: True for tets whose centroid is inside the domain."""
        # Map ζ-centroids → physical x via decoder, then evaluate SDF.
        # IMPORTANT: atlas coordinates are already in SDF-normalised space.
        # Do NOT apply (x - center)/scale — that would be double-normalisation.
        batch = 4096
        sdf_vals = []
        for start in range(0, len(centroids_xi), batch):
            end = min(start + batch, len(centroids_xi))
            xi_t = torch.tensor(
                centroids_xi[start:end], device=self.device, dtype=self.dtype
            )
            with torch.no_grad():
                x_phys = self.decoder(
                    xi_t,
                    seed=self.seed,
                    t1=self.t1,
                    t2=self.t2,
                    n=self.n_vec,
                    chart_scale=self.support_r,
                )
                sv = self.sdf_net(x_phys)  # direct — no double-normalisation
            sdf_vals.append(sv.cpu().numpy())
        sdf_arr = np.concatenate(sdf_vals, axis=0)
        return sdf_arr < self.sdf_threshold

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Node classification
    # ──────────────────────────────────────────────────────────────────────

    def _classify_nodes(self) -> None:
        """Classify nodes into physical-BC, artificial-BC, and interior."""
        n_el = len(self.elements)
        n_nd = self.n_nodes

        # 1. Identify boundary faces of the mesh.
        # A face is a boundary face if it belongs to exactly one element.
        # For each tet, the 4 faces are (sorted node triples):
        face_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        all_faces = []
        face_elem = []
        for e_idx in range(n_el):
            nids = self.elements[e_idx]
            for fi in range(4):
                face = tuple(sorted(nids[face_indices[fi]]))
                all_faces.append(face)
                face_elem.append(e_idx)

        # Count face occurrences
        from collections import Counter
        face_count = Counter(all_faces)
        boundary_faces = {f for f, c in face_count.items() if c == 1}

        boundary_node_set = set()
        for f in boundary_faces:
            boundary_node_set.update(f)
        boundary_nodes = np.array(sorted(boundary_node_set), dtype=np.int64)

        # 2. Among boundary nodes, classify as physical-BC or artificial-BC.
        # Physical-BC nodes sit on ∂Ω (the geometry surface); artificial-BC
        # nodes sit on the boundary of the reference cube but are interior to Ω,
        # and receive Schwarz values from neighboring charts.
        if self.sdf_net is not None and len(boundary_nodes) > 0:
            # Use SDF to distinguish: |SDF(x)| < tol → on ∂Ω
            sdf_bc_tol = 2.0 * self.h
            xi_bnd = self.nodes[boundary_nodes]
            sdf_abs = self._eval_sdf_abs(xi_bnd)

            phys_mask = sdf_abs < sdf_bc_tol
            self.phys_bc_nodes = boundary_nodes[phys_mask]
            self.art_bc_nodes = boundary_nodes[~phys_mask]
        else:
            # Without SDF: treat ALL boundary nodes as artificial (Schwarz
            # interface).  The Schwarz BC routine will use manufactured-solution
            # fallback for nodes that no neighbor covers, which gives the
            # correct value both on ∂Ω and in the exterior.
            self.phys_bc_nodes = np.array([], dtype=np.int64)
            self.art_bc_nodes = boundary_nodes

        all_boundary = set(boundary_nodes.tolist())
        self.interior_nodes = np.array(
            [i for i in range(n_nd) if i not in all_boundary], dtype=np.int64
        )

    def _eval_sdf_abs(self, xi_coords: np.ndarray) -> np.ndarray:
        """Return |SDF(φ(ζ))| for an array of ζ-coordinates."""
        batch = 4096
        sdf_abs_all = []
        for start in range(0, len(xi_coords), batch):
            end = min(start + batch, len(xi_coords))
            xi_t = torch.tensor(
                xi_coords[start:end], device=self.device, dtype=self.dtype
            )
            with torch.no_grad():
                x_phys = self.decoder(
                    xi_t,
                    seed=self.seed,
                    t1=self.t1,
                    t2=self.t2,
                    n=self.n_vec,
                    chart_scale=self.support_r,
                )
                # Direct evaluation — atlas coords are already in SDF input space
                sv = self.sdf_net(x_phys)
            sdf_abs_all.append(np.abs(sv.cpu().numpy()))
        return np.concatenate(sdf_abs_all, axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Diffusion tensor evaluation
    # ──────────────────────────────────────────────────────────────────────

    def compute_diffusion_tensors(self) -> None:
        """Pre-compute A_i(ζ) = j_i · J_i⁻¹ J_i⁻ᵀ at each element centroid.

        Also stores: det |J| (= j_i) and physical centroids x = φ(ζ_c).
        """
        centroids = self.nodes[self.elements].mean(axis=1)  # (N_el, 3)
        n_el = len(centroids)

        batch = 2048
        A_list, det_list, x_list = [], [], []
        for start in range(0, n_el, batch):
            end = min(start + batch, n_el)
            xi_t = torch.tensor(
                centroids[start:end], device=self.device, dtype=self.dtype
            ).requires_grad_(True)

            x_phys = self.decoder(
                xi_t,
                seed=self.seed,
                t1=self.t1,
                t2=self.t2,
                n=self.n_vec,
                chart_scale=self.support_r,
            )
            # Jacobian ∂x/∂ζ component-by-component
            grads = []
            for comp in range(3):
                gi = torch.autograd.grad(
                    x_phys[:, comp],
                    xi_t,
                    grad_outputs=torch.ones(end - start, device=self.device, dtype=self.dtype),
                    create_graph=False,
                    retain_graph=(comp < 2),
                )[0]
                grads.append(gi)
            jac = torch.stack(grads, dim=1)  # (batch, 3, 3)

            # Stabilised inverse
            det_raw = torch.abs(torch.linalg.det(jac))
            det_abs = torch.clamp(det_raw, min=1e-10)
            inv_j = torch.linalg.inv(
                jac + 1e-6 * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
            )

            # A = det * J⁻¹ J⁻ᵀ
            A = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(inv_j, inv_j.transpose(1, 2))

            A_list.append(A.detach().cpu().numpy())
            det_list.append(det_abs.detach().cpu().numpy())
            x_list.append(x_phys.detach().cpu().numpy())

        self.A_elem = np.concatenate(A_list, axis=0)
        self.det_elem = np.concatenate(det_list, axis=0)
        self.x_centroids = np.concatenate(x_list, axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Vectorised FEM assembly
    # ──────────────────────────────────────────────────────────────────────

    def assemble(self, forcing_fn=None) -> None:
        """Assemble global stiffness K and load F.

        Parameters
        ----------
        forcing_fn : callable or None
            If None, uses the standard manufactured-solution forcing
            f(x) = 3π² sin(πx₁)sin(πx₂)sin(πx₃).
        """
        xe = self.nodes[self.elements]  # (N_el, 4, 3)
        n_el = len(self.elements)

        # Reference tet Jacobian: columns = edge vectors from node 0
        B = np.swapaxes(xe[:, 1:, :] - xe[:, 0:1, :], 1, 2)  # (N_el, 3, 3)
        detB = np.linalg.det(B)  # (N_el,)
        V = np.abs(detB) / 6.0   # (N_el,)

        # Skip degenerate tets
        good = V > 1e-20
        if not np.all(good):
            n_bad = int(np.sum(~good))
            print(f"  [Chart {self.chart_id}] Skipping {n_bad} degenerate elements")

        invB = np.linalg.inv(B)  # (N_el, 3, 3)

        # Shape function gradients: dN (N_el, 4, 3)
        # For tet mapping x = v0 + B λ, B = [v1-v0|v2-v0|v3-v0] (columns),
        # ∇_x N_k = B^{-T} e_k.  The j-th component of ∇_x N_k is (B^{-1})_{k-1,j},
        # i.e., the (k-1)-th row of B^{-1}.
        dN = np.zeros((n_el, 4, 3))
        dN[:, 1:, :] = invB                        # rows of invB
        dN[:, 0, :] = -dN[:, 1:, :].sum(axis=1)   # ∇N_0 = -sum of others

        # A @ dN^T for all elements at once
        # AdN[e, α, j] = Σ_k A[e,j,k] dN[e,α,k]
        AdN = np.einsum("eij, enj -> eni", self.A_elem, dN)  # (N_el, 4, 3)

        # Local stiffness: K_local[e, α, β] = V_e * dN[e,α]^T A dN[e,β]
        K_local = V[:, None, None] * np.einsum("eai, ebi -> eab", AdN, dN)  # (N_el, 4, 4)

        # Zero out degenerate elements
        K_local[~good] = 0.0

        # Scatter into COO sparse
        el = self.elements  # (N_el, 4)
        ii = np.repeat(el, 4, axis=1).ravel()  # each row node repeated 4 times
        jj = np.tile(el, (1, 4)).ravel()        # tiled columns
        self.K = scipy.sparse.coo_matrix(
            (K_local.ravel(), (ii, jj)),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()

        # Load vector: F[α] = Σ_e V_e * j_i(centroid) * f(x_centroid) * (1/4)
        if forcing_fn is not None:
            f_vals = forcing_fn(self.x_centroids)  # (N_el,) or (N_el, 1)
        else:
            f_vals = _manufactured_forcing_np(self.x_centroids)
        f_vals = np.asarray(f_vals).reshape(-1)  # (N_el,)

        f_contrib = V * self.det_elem * f_vals * 0.25  # (N_el,)
        f_contrib[~good] = 0.0
        self.F = np.zeros(self.n_nodes)
        np.add.at(self.F, el[:, 0], f_contrib)
        np.add.at(self.F, el[:, 1], f_contrib)
        np.add.at(self.F, el[:, 2], f_contrib)
        np.add.at(self.F, el[:, 3], f_contrib)

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Solve with Dirichlet BCs (row/column elimination)
    # ──────────────────────────────────────────────────────────────────────

    def solve(
        self,
        phys_bc_values: Dict[int, float],
        art_bc_values: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Solve K u = F with Dirichlet BCs via row/column elimination.

        Parameters
        ----------
        phys_bc_values : dict {node_idx: value}
            Values for nodes on the physical boundary ∂Ω.
        art_bc_values : dict {node_idx: value} or None
            Schwarz artificial boundary values from neighboring charts.

        Returns
        -------
        u : (N_nodes,) ndarray
            Solution vector.
        """
        bc = dict(phys_bc_values)
        if art_bc_values:
            bc.update(art_bc_values)

        if len(bc) == 0:
            # No BCs at all — just solve
            self.u = scipy.sparse.linalg.spsolve(self.K, self.F)
            return self.u

        bc_indices = np.array(list(bc.keys()), dtype=np.int64)
        bc_vals = np.array([bc[i] for i in bc_indices], dtype=np.float64)

        F_mod = self.F.copy()
        K_mod = self.K.tolil()

        # Adjust RHS for BC columns: F -= K[:, bc] * bc_val
        for idx, val in zip(bc_indices, bc_vals):
            col = np.asarray(self.K.getcol(int(idx)).todense()).ravel()
            F_mod -= col * val

        # Zero BC rows/cols, set diagonal to 1
        for idx, val in zip(bc_indices, bc_vals):
            K_mod[int(idx), :] = 0
            K_mod[:, int(idx)] = 0
            K_mod[int(idx), int(idx)] = 1.0
            F_mod[int(idx)] = val

        K_csr = K_mod.tocsr()
        self.u = scipy.sparse.linalg.spsolve(K_csr, F_mod)
        return self.u

    # ──────────────────────────────────────────────────────────────────────
    # Step 6: Solution interpolation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_at(self, xi_query: np.ndarray) -> np.ndarray:
        """Interpolate P1 solution at arbitrary ζ-coordinates.

        Uses structured grid lookup (O(1) per point) for the containing hex,
        then tries each of the 6 sub-tets for barycentric interpolation.
        Falls back to nearest-node value for points outside the mesh.

        Parameters
        ----------
        xi_query : (M, 3) ndarray  —  query points in chart-local coordinates.

        Returns
        -------
        u_query : (M,) ndarray  —  interpolated solution values.
        """
        if self.u is None:
            return np.zeros(len(xi_query))

        xi_query = np.asarray(xi_query, dtype=np.float64)
        if xi_query.ndim == 1:
            xi_query = xi_query.reshape(1, 3)

        M = len(xi_query)
        u_out = np.zeros(M)

        r = self.r
        h = self.h
        n = self.n_cells
        n1 = self._grid_n1
        old_to_new = self._old_to_new
        all_nodes_grid = self._all_nodes_grid

        for q in range(M):
            pt = xi_query[q]

            # Grid cell index
            fi = (pt[0] + r) / h
            fj = (pt[1] + r) / h
            fk = (pt[2] + r) / h

            gi = int(np.clip(np.floor(fi), 0, n - 1))
            gj = int(np.clip(np.floor(fj), 0, n - 1))
            gk = int(np.clip(np.floor(fk), 0, n - 1))

            # Hex cell vertex indices in original grid
            base_old = gi + gj * n1 + gk * n1 * n1
            hex_offsets = _hex_vertex_offsets(n1)
            hex_old = base_old + hex_offsets  # 8 original node indices

            # Map to compacted indices
            hex_new = old_to_new[hex_old]

            found = False
            # Try each of the 6 Freudenthal tets
            for ti in range(6):
                tet_local = _FREUDENTHAL_TETS[ti]  # 4 local hex-vertex indices
                tet_new = hex_new[tet_local]

                # Skip if any node was removed
                if np.any(tet_new < 0):
                    continue

                # Compute barycentric coordinates
                tet_coords = self.nodes[tet_new]  # (4, 3)
                lam = _barycentric_coords(pt, tet_coords)

                # Check if inside (with small tolerance)
                if np.all(lam >= -1e-8):
                    u_out[q] = np.dot(lam, self.u[tet_new])
                    found = True
                    break

            if not found:
                # Fallback: nearest-node interpolation
                dists = np.linalg.norm(self.nodes - pt, axis=1)
                nearest = np.argmin(dists)
                u_out[q] = self.u[nearest]

        return u_out

    def evaluate_at_batch(self, xi_query: np.ndarray) -> np.ndarray:
        """Vectorised version of evaluate_at — faster for large query sets.

        Falls back to nearest-node for out-of-mesh queries.
        """
        if self.u is None:
            return np.zeros(len(xi_query))

        xi_query = np.asarray(xi_query, dtype=np.float64)
        if xi_query.ndim == 1:
            xi_query = xi_query.reshape(1, 3)

        M = len(xi_query)
        u_out = np.full(M, np.nan)

        r = self.r
        h = self.h
        n = self.n_cells
        n1 = self._grid_n1
        old_to_new = self._old_to_new

        # Grid cell indices
        fi = np.clip(np.floor((xi_query[:, 0] + r) / h).astype(int), 0, n - 1)
        fj = np.clip(np.floor((xi_query[:, 1] + r) / h).astype(int), 0, n - 1)
        fk = np.clip(np.floor((xi_query[:, 2] + r) / h).astype(int), 0, n - 1)

        base_old = fi + fj * n1 + fk * n1 * n1
        hex_off = _hex_vertex_offsets(n1)

        for q in range(M):
            hex_old = base_old[q] + hex_off
            hex_new = old_to_new[hex_old]
            pt = xi_query[q]

            for ti in range(6):
                tet_local = _FREUDENTHAL_TETS[ti]
                tet_new = hex_new[tet_local]
                if np.any(tet_new < 0):
                    continue
                tet_coords = self.nodes[tet_new]
                lam = _barycentric_coords(pt, tet_coords)
                if np.all(lam >= -1e-8):
                    u_out[q] = np.dot(lam, self.u[tet_new])
                    break

        # Nearest-node fallback for remaining NaN entries
        nan_mask = np.isnan(u_out)
        if np.any(nan_mask):
            from scipy.spatial import cKDTree
            tree = cKDTree(self.nodes)
            _, nearest = tree.query(xi_query[nan_mask])
            u_out[nan_mask] = self.u[nearest]

        return u_out

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n_dofs(self) -> int:
        return self.n_nodes

    @property
    def n_elements(self) -> int:
        return len(self.elements) if self.elements is not None else 0

    def summary(self) -> str:
        n_phys = len(self.phys_bc_nodes) if self.phys_bc_nodes is not None else 0
        n_art = len(self.art_bc_nodes) if self.art_bc_nodes is not None else 0
        n_int = len(self.interior_nodes) if self.interior_nodes is not None else 0
        return (
            f"Chart {self.chart_id}: "
            f"{self.n_elements} tets, {self.n_nodes} nodes "
            f"(phys_bc={n_phys}, art_bc={n_art}, interior={n_int}), "
            f"h={self.h:.4f}"
        )

    def decoder_inverse(self, x_phys: np.ndarray, max_iters: int = 8, tol: float = 1e-8) -> np.ndarray:
        """Invert the decoder: given physical coordinates x, find ζ such that φ(ζ) ≈ x.

        Uses Newton iteration starting from ζ⁰ = T^T (x - seed)  (rigid TNB inverse).

        Parameters
        ----------
        x_phys : (M, 3) ndarray  —  physical coordinates.
        max_iters : int  —  max Newton iterations.
        tol : float  —  convergence tolerance in ζ-space.

        Returns
        -------
        xi : (M, 3) ndarray  —  reference-domain coordinates.
        """
        x_phys = np.asarray(x_phys, dtype=np.float64)
        if x_phys.ndim == 1:
            x_phys = x_phys.reshape(1, 3)
        M = len(x_phys)

        # Initial guess: rigid TNB inverse
        seed_np = self.seed.cpu().numpy()
        t1_np = self.t1.cpu().numpy()
        t2_np = self.t2.cpu().numpy()
        n_np = self.n_vec.cpu().numpy()

        d = x_phys - seed_np[np.newaxis, :]
        xi = np.stack([d @ t1_np, d @ t2_np, d @ n_np], axis=-1)  # (M, 3)

        # Newton iterations in batches
        batch = 2048
        for it in range(max_iters):
            max_update = 0.0
            for start in range(0, M, batch):
                end = min(start + batch, M)
                xi_t = torch.tensor(
                    xi[start:end], device=self.device, dtype=self.dtype
                ).requires_grad_(True)

                x_decoded = self.decoder(
                    xi_t,
                    seed=self.seed,
                    t1=self.t1,
                    t2=self.t2,
                    n=self.n_vec,
                    chart_scale=self.support_r,
                )

                # Jacobian (N, 3, 3) via autograd
                grads = []
                for comp in range(3):
                    gi = torch.autograd.grad(
                        x_decoded[:, comp],
                        xi_t,
                        grad_outputs=torch.ones(end - start, device=self.device, dtype=self.dtype),
                        create_graph=False,
                        retain_graph=(comp < 2),
                    )[0]
                    grads.append(gi)
                J = torch.stack(grads, dim=1).detach()  # (batch, 3, 3)

                residual = x_decoded.detach().cpu().numpy() - x_phys[start:end]  # (batch, 3)
                J_np = J.cpu().numpy()

                # Newton step: δζ = -J⁻¹ · residual
                try:
                    delta = np.linalg.solve(J_np, -residual[:, :, np.newaxis]).squeeze(-1)
                except np.linalg.LinAlgError:
                    delta = -residual  # fallback: gradient descent step

                # Clamp update to avoid overshooting
                delta = np.clip(delta, -self.r * 0.5, self.r * 0.5)

                xi[start:end] += delta
                max_update = max(max_update, np.max(np.abs(delta)))

            if max_update < tol:
                break

        # Clamp to reference domain
        xi = np.clip(xi, -self.r, self.r)
        return xi

    def evaluate_at_physical(self, x_phys: np.ndarray) -> np.ndarray:
        """Evaluate FEM solution at physical coordinates by inverting the decoder.

        This is the correct evaluation method when using the full mapped PDE:
          1. Invert decoder: x → ζ = φ⁻¹(x)  (Newton iteration)
          2. Interpolate FEM solution: û(ζ)

        Parameters
        ----------
        x_phys : (M, 3) ndarray  —  physical coordinates.

        Returns
        -------
        u : (M,) ndarray  —  interpolated solution values.
        """
        if self.u is None:
            return np.zeros(len(x_phys))
        xi = self.decoder_inverse(x_phys)
        return self.evaluate_at_batch(xi)

    def physical_coords_of_nodes(self) -> np.ndarray:
        """Return physical coordinates x = φ(ζ) for all mesh nodes."""
        batch = 4096
        x_all = []
        for start in range(0, self.n_nodes, batch):
            end = min(start + batch, self.n_nodes)
            xi_t = torch.tensor(
                self.nodes[start:end], device=self.device, dtype=self.dtype
            )
            with torch.no_grad():
                x_phys = self.decoder(
                    xi_t,
                    seed=self.seed,
                    t1=self.t1,
                    t2=self.t2,
                    n=self.n_vec,
                    chart_scale=self.support_r,
                )
            x_all.append(x_phys.cpu().numpy())
        return np.concatenate(x_all, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════

def _barycentric_coords(pt: np.ndarray, tet: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of *pt* in tet (4 vertices × 3)."""
    T = (tet[1:] - tet[0]).T  # (3, 3)
    try:
        lam_123 = np.linalg.solve(T, pt - tet[0])
    except np.linalg.LinAlgError:
        return np.array([-1.0, -1.0, -1.0, -1.0])
    lam0 = 1.0 - lam_123.sum()
    return np.array([lam0, lam_123[0], lam_123[1], lam_123[2]])


def _manufactured_forcing_np(x: np.ndarray) -> np.ndarray:
    """f(x) = 3π² sin(πx₁) sin(πx₂) sin(πx₃)."""
    pi = math.pi
    return 3.0 * pi**2 * np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * np.sin(pi * x[:, 2])


def manufactured_u_np(x: np.ndarray) -> np.ndarray:
    """u*(x) = sin(πx₁) sin(πx₂) sin(πx₃)."""
    pi = math.pi
    return np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * np.sin(pi * x[:, 2])
