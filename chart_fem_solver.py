#!/usr/bin/env python3
"""
P1 tetrahedral FEM solver for mapped Poisson on a single chart's reference domain.

Given a frozen chart decoder φ_i : ξ → x, solves the mapped Poisson equation:
    -div_ξ [ A_i(ξ) ∇_ξ u ] = j_i(ξ) f(φ_i(ξ))
where A_i = j_i J_i^{-1} J_i^{-T},  j_i = |det J_i|.

Mesh: structured hex grid → 6 tets/hex (Freudenthal), SDF-filtered.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch


# ---------------------------------------------------------------------------
# Freudenthal 6-tet decomposition of a unit cube
# Each tet defined by 4 corner offsets in the hex {0,1}^3.
# The 8 corners are indexed by (dz*4 + dy*2 + dx).
# ---------------------------------------------------------------------------
_FREUDENTHAL_TETS = np.array([
    # Tet 0: 000, 100, 110, 111
    [0, 1, 3, 7],
    # Tet 1: 000, 100, 101, 111
    [0, 1, 5, 7],
    # Tet 2: 000, 010, 110, 111
    [0, 2, 3, 7],
    # Tet 3: 000, 010, 011, 111
    [0, 2, 6, 7],
    # Tet 4: 000, 001, 101, 111
    [0, 4, 5, 7],
    # Tet 5: 000, 001, 011, 111
    [0, 4, 6, 7],
], dtype=np.int32)

# Corner offsets within a hex: index → (dx, dy, dz)
_HEX_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int32)


class ChartFEMSolver:
    """P1 tetrahedral FEM solver for mapped Poisson on one chart's reference domain."""

    def __init__(
        self,
        chart_id: int,
        decoder: "torch.nn.Module",
        seed: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        n_vec: torch.Tensor,
        support_r: torch.Tensor,
        n_cells: int = 16,
        sdf_oracle: Optional[object] = None,
        sdf_threshold: float = -0.005,
        mesh_extent: float = 1.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.chart_id = chart_id
        self.decoder = decoder
        self.seed = seed
        self.t1 = t1
        self.t2 = t2
        self.n_vec = n_vec
        self.support_r = support_r
        self.n_cells = n_cells
        self.sdf_oracle = sdf_oracle
        self.sdf_threshold = sdf_threshold
        self.mesh_extent = mesh_extent
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        r = float(support_r.item()) if isinstance(support_r, torch.Tensor) else float(support_r)
        self.r = r * mesh_extent  # extend mesh beyond support radius
        self.h = 2.0 * self.r / n_cells

        # Will be populated by _build_mesh
        self.nodes: np.ndarray = np.empty((0, 3))
        self.elements: np.ndarray = np.empty((0, 4), dtype=np.int64)
        self.n_nodes: int = 0

        # Boundary classification
        self.phys_bc_nodes: np.ndarray = np.empty(0, dtype=np.int64)
        self.art_bc_nodes: np.ndarray = np.empty(0, dtype=np.int64)
        self.interior_nodes: np.ndarray = np.empty(0, dtype=np.int64)

        # Pre-computed tensors (set by compute_diffusion_tensors)
        self.A_elem: Optional[np.ndarray] = None   # (N_el, 3, 3)
        self.det_elem: Optional[np.ndarray] = None  # (N_el,)
        self.x_centroids: Optional[np.ndarray] = None  # (N_el, 3) physical coords

        # Assembled system
        self.K: Optional[scipy.sparse.csr_matrix] = None
        self.F: Optional[np.ndarray] = None

        # Solution
        self.u: Optional[np.ndarray] = None

        # Grid → element lookup (for interpolation)
        self._grid_nodes_per_axis: int = n_cells + 1
        self._hex_to_tets: Optional[np.ndarray] = None  # mapping from hex cell to tet indices

        # Build mesh
        self._build_mesh()

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------
    def _build_mesh(self) -> None:
        """Generate structured hex grid, subdivide into tets, filter by SDF."""
        nc = self.n_cells
        r = self.r
        npa = nc + 1  # nodes per axis

        # Regular grid nodes
        lin = np.linspace(-r, r, npa)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        all_nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (npa^3, 3)

        def node_idx(ix, iy, iz):
            return ix * npa * npa + iy * npa + iz

        # Build hex cells → tet elements
        hex_indices = []
        for ix in range(nc):
            for iy in range(nc):
                for iz in range(nc):
                    hex_indices.append((ix, iy, iz))

        hex_indices = np.array(hex_indices, dtype=np.int32)  # (n_hex, 3)
        n_hex = len(hex_indices)

        # For each hex, get the 8 corner node indices
        hex_corners = np.zeros((n_hex, 8), dtype=np.int64)
        for c in range(8):
            dx, dy, dz = _HEX_CORNERS[c]
            hex_corners[:, c] = node_idx(
                hex_indices[:, 0] + dx,
                hex_indices[:, 1] + dy,
                hex_indices[:, 2] + dz,
            )

        # Subdivide each hex into 6 tets
        all_tets = np.zeros((n_hex * 6, 4), dtype=np.int64)
        for t in range(6):
            all_tets[t::6] = hex_corners[:, _FREUDENTHAL_TETS[t]]

        # Track which hex each tet belongs to
        hex_of_tet = np.repeat(np.arange(n_hex), 6)

        # SDF filtering: evaluate SDF at tet centroids in physical space
        if self.sdf_oracle is not None:
            centroids_xi = all_nodes[all_tets].mean(axis=1)  # (N_tet, 3)
            # Map to physical space
            x_phys = self._decode_points(centroids_xi)
            sdf_vals = self._eval_sdf(x_phys)
            keep = sdf_vals < self.sdf_threshold
            all_tets = all_tets[keep]
            hex_of_tet = hex_of_tet[keep]

        if len(all_tets) == 0:
            print(f"  [ChartFEM {self.chart_id}] WARNING: no elements survived SDF filtering!")
            self.nodes = all_nodes
            self.elements = np.empty((0, 4), dtype=np.int64)
            self.n_nodes = len(all_nodes)
            self._classify_nodes()
            return

        # Compact nodes: keep only nodes referenced by surviving tets
        used_nodes = np.unique(all_tets.ravel())
        old_to_new = np.full(len(all_nodes), -1, dtype=np.int64)
        old_to_new[used_nodes] = np.arange(len(used_nodes))

        self.nodes = all_nodes[used_nodes]
        self.elements = old_to_new[all_tets]
        self.n_nodes = len(self.nodes)

        # Build hex → tet index mapping for O(1) element lookup
        self._build_hex_to_tet_map(hex_of_tet, old_to_new, n_hex)

        # Classify boundary nodes
        self._classify_nodes()

        print(
            f"  [ChartFEM {self.chart_id}] mesh: {self.n_nodes} nodes, "
            f"{len(self.elements)} tets, "
            f"{len(self.phys_bc_nodes)} phys_bc, "
            f"{len(self.art_bc_nodes)} art_bc, "
            f"{len(self.interior_nodes)} interior"
        )

    def _build_hex_to_tet_map(self, hex_of_tet: np.ndarray, old_to_new: np.ndarray, n_hex: int) -> None:
        """Build mapping from hex cell index to list of tet indices (for interpolation)."""
        # Store as a list of arrays for variable-length (some hexes may have < 6 tets after filtering)
        self._hex_tet_lists: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_hex)]
        for tet_idx, hex_idx in enumerate(hex_of_tet):
            self._hex_tet_lists[hex_idx] = np.append(self._hex_tet_lists[hex_idx], tet_idx)

    def _decode_points(self, xi: np.ndarray) -> np.ndarray:
        """Map chart-local coords to physical coords using the decoder."""
        with torch.no_grad():
            xi_t = torch.tensor(xi, device=self.device, dtype=self.dtype)
            x_t = self.decoder(
                xi_t,
                seed=self.seed,
                t1=self.t1,
                t2=self.t2,
                n=self.n_vec,
                chart_scale=self.support_r,
            )
            return x_t.cpu().numpy()

    def _eval_sdf(self, x_phys: np.ndarray) -> np.ndarray:
        """Evaluate SDF at physical coordinates."""
        with torch.no_grad():
            x_t = torch.tensor(x_phys, device=self.device, dtype=self.dtype)
            sdf_vals = self.sdf_oracle.sdf(x_t)
            return sdf_vals.cpu().numpy()

    # ------------------------------------------------------------------
    # Node classification
    # ------------------------------------------------------------------
    def _classify_nodes(self) -> None:
        """Classify nodes into physical boundary, artificial boundary, and interior."""
        if len(self.elements) == 0:
            self.phys_bc_nodes = np.empty(0, dtype=np.int64)
            self.art_bc_nodes = np.empty(0, dtype=np.int64)
            self.interior_nodes = np.arange(self.n_nodes, dtype=np.int64)
            return

        # 1. Find boundary nodes: faces shared by exactly 1 element
        face_count: Dict[Tuple[int, ...], int] = {}
        face_to_nodes: Dict[Tuple[int, ...], Tuple[int, int, int]] = {}

        # Each tet has 4 triangular faces
        face_combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        for e_idx in range(len(self.elements)):
            nids = self.elements[e_idx]
            for combo in face_combos:
                face_nodes = tuple(sorted(nids[list(combo)]))
                face_count[face_nodes] = face_count.get(face_nodes, 0) + 1
                face_to_nodes[face_nodes] = face_nodes

        boundary_node_set = set()
        for face, count in face_count.items():
            if count == 1:
                boundary_node_set.update(face)

        boundary_nodes = np.array(sorted(boundary_node_set), dtype=np.int64)

        if len(boundary_nodes) == 0:
            self.phys_bc_nodes = np.empty(0, dtype=np.int64)
            self.art_bc_nodes = np.empty(0, dtype=np.int64)
            self.interior_nodes = np.arange(self.n_nodes, dtype=np.int64)
            return

        # 2. Among boundary nodes, classify as physical vs artificial
        if self.sdf_oracle is not None:
            x_phys = self._decode_points(self.nodes[boundary_nodes])
            sdf_vals = self._eval_sdf(x_phys)
            sdf_bc_tol = 2.0 * self.h
            is_phys = np.abs(sdf_vals) < sdf_bc_tol
            self.phys_bc_nodes = boundary_nodes[is_phys]
            self.art_bc_nodes = boundary_nodes[~is_phys]
        else:
            # Without SDF, all boundary nodes are artificial
            self.phys_bc_nodes = np.empty(0, dtype=np.int64)
            self.art_bc_nodes = boundary_nodes

        # 3. Interior = everything else
        all_boundary = set(boundary_nodes.tolist())
        self.interior_nodes = np.array(
            [i for i in range(self.n_nodes) if i not in all_boundary],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Diffusion tensor computation
    # ------------------------------------------------------------------
    def compute_diffusion_tensors(self) -> None:
        """Pre-compute A_i(ξ) = j_i J_i^{-1} J_i^{-T} at element centroids."""
        if len(self.elements) == 0:
            self.A_elem = np.empty((0, 3, 3))
            self.det_elem = np.empty(0)
            self.x_centroids = np.empty((0, 3))
            return

        centroids = self.nodes[self.elements].mean(axis=1)  # (N_el, 3)

        # Process in batches to avoid OOM
        batch_size = 2048
        n_el = len(centroids)
        A_all = np.zeros((n_el, 3, 3))
        det_all = np.zeros(n_el)
        x_all = np.zeros((n_el, 3))

        for start in range(0, n_el, batch_size):
            end = min(start + batch_size, n_el)
            xi_batch = centroids[start:end]

            xi_t = torch.tensor(xi_batch, device=self.device, dtype=self.dtype).requires_grad_(True)
            x_t = self.decoder(
                xi_t,
                seed=self.seed,
                t1=self.t1,
                t2=self.t2,
                n=self.n_vec,
                chart_scale=self.support_r,
            )

            # Compute Jacobian
            grads = []
            for i in range(3):
                gi = torch.autograd.grad(
                    x_t[:, i],
                    xi_t,
                    grad_outputs=torch.ones_like(x_t[:, i]),
                    create_graph=False,
                    retain_graph=True,
                )[0]
                grads.append(gi)
            jac = torch.stack(grads, dim=1)  # (batch, 3, 3)

            # Stabilized inverse
            u_svd, s_svd, vh = torch.linalg.svd(jac)
            s_safe = torch.clamp(s_svd, min=1e-3)
            inv_s = torch.diag_embed(1.0 / s_safe)
            inv_j = torch.bmm(vh.transpose(1, 2), torch.bmm(inv_s, u_svd.transpose(1, 2)))

            det_abs = torch.clamp(torch.abs(torch.det(jac)), min=1e-6)

            # A = det * J^{-1} J^{-T}
            A = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(inv_j, inv_j.transpose(1, 2))

            A_all[start:end] = A.detach().cpu().numpy()
            det_all[start:end] = det_abs.detach().cpu().numpy()
            x_all[start:end] = x_t.detach().cpu().numpy()

        self.A_elem = A_all
        self.det_elem = det_all
        self.x_centroids = x_all

    # ------------------------------------------------------------------
    # FEM assembly (vectorized)
    # ------------------------------------------------------------------
    def assemble(self, forcing_fn=None) -> None:
        """Assemble global stiffness matrix K and load vector F.

        Args:
            forcing_fn: callable(x_phys) -> (N,) array of forcing values.
                        If None, uses the standard manufactured forcing.
        """
        if len(self.elements) == 0:
            self.K = scipy.sparse.csr_matrix((self.n_nodes, self.n_nodes))
            self.F = np.zeros(self.n_nodes)
            return

        n_el = len(self.elements)
        xe = self.nodes[self.elements]  # (N_el, 4, 3)

        # Reference Jacobian: columns = edge vectors from node 0
        B = np.transpose(xe[:, 1:, :] - xe[:, 0:1, :], (0, 2, 1))  # (N_el, 3, 3)
        detB = np.linalg.det(B)
        V = np.abs(detB) / 6.0  # (N_el,)

        # Filter degenerate elements
        valid = V > 1e-20
        if not np.all(valid):
            print(f"  [ChartFEM {self.chart_id}] {np.sum(~valid)} degenerate tets skipped")

        invB = np.linalg.inv(B)  # (N_el, 3, 3)

        # Shape function gradients: dN (N_el, 4, 3)
        # ∇N_k = row (k-1) of B^{-1}  for k=1,2,3
        # because x(λ) = v0 + B λ  ⟹  λ = B^{-1}(x - v0)  ⟹  ∇λ_k = row k of B^{-1}
        dN = np.zeros((n_el, 4, 3))
        dN[:, 1:, :] = invB  # rows of B^{-1} are ∇N_1, ∇N_2, ∇N_3
        dN[:, 0, :] = -dN[:, 1:, :].sum(axis=1)  # ∇N_0

        # A @ dN for all elements: (N_el, 4, 3)
        # AdN[e, a, :] = A_e @ dN[e, a, :]
        AdN = np.einsum("eij, eaj -> eai", self.A_elem, dN)  # (N_el, 4, 3)

        # Local stiffness: K_local[e, a, b] = V[e] * AdN[e, a, :] · dN[e, b, :]
        K_local = V[:, None, None] * np.einsum("eai, ebi -> eab", AdN, dN)  # (N_el, 4, 4)

        # Zero out degenerate elements
        K_local[~valid] = 0.0

        # Scatter into COO sparse
        el = self.elements  # (N_el, 4)
        ii = np.repeat(el, 4, axis=1).ravel()
        jj = np.tile(el, (1, 4)).ravel()
        self.K = scipy.sparse.coo_matrix(
            (K_local.ravel(), (ii, jj)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()

        # Load vector
        if forcing_fn is not None:
            f_vals = forcing_fn(self.x_centroids)  # (N_el,)
        else:
            f_vals = self._default_forcing(self.x_centroids)

        f_vals = np.asarray(f_vals).ravel()
        f_contrib = V * self.det_elem * f_vals * 0.25  # (N_el,)
        f_contrib[~valid] = 0.0
        self.F = np.zeros(self.n_nodes)
        np.add.at(self.F, self.elements, f_contrib[:, None])

    @staticmethod
    def _default_forcing(x_phys: np.ndarray) -> np.ndarray:
        """f(x) = 3π² sin(πx₁) sin(πx₂) sin(πx₃)  (standard manufactured solution)."""
        pi = math.pi
        return (
            3.0 * pi**2
            * np.sin(pi * x_phys[:, 0])
            * np.sin(pi * x_phys[:, 1])
            * np.sin(pi * x_phys[:, 2])
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        phys_bc_values: Dict[int, float],
        art_bc_values: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Solve K u = F with Dirichlet BCs (row/column elimination).

        Args:
            phys_bc_values: {node_idx: value} for physical boundary nodes
            art_bc_values: {node_idx: value} for Schwarz artificial boundary nodes

        Returns:
            Solution vector u (N_nodes,)
        """
        if self.K is None or self.F is None:
            raise RuntimeError("Must call assemble() before solve()")

        if self.n_nodes == 0:
            self.u = np.zeros(0)
            return self.u

        bc = {}
        bc.update(phys_bc_values)
        if art_bc_values:
            bc.update(art_bc_values)

        if len(bc) == 0:
            # No BCs — solve directly (unusual)
            self.u = scipy.sparse.linalg.spsolve(self.K, self.F)
            return self.u

        bc_indices = np.array(list(bc.keys()), dtype=np.int64)
        bc_vals = np.array([bc[i] for i in bc_indices])

        # Efficient row/column elimination using CSR format
        K_mod = self.K.tolil()
        F_mod = self.F.copy()

        # Adjust RHS: subtract BC contribution from non-BC rows
        # F_mod -= K[:, bc_indices] @ bc_vals, then zero BC rows/cols
        for k in range(len(bc_indices)):
            idx = bc_indices[k]
            val = bc_vals[k]
            col = np.array(self.K.getcol(idx).todense()).ravel()
            F_mod -= col * val

        # Zero BC rows and columns, set diagonal
        for idx in bc_indices:
            K_mod[idx, :] = 0
            K_mod[:, idx] = 0
            K_mod[idx, idx] = 1.0

        # Set BC RHS
        for k in range(len(bc_indices)):
            F_mod[bc_indices[k]] = bc_vals[k]

        K_csr = K_mod.tocsr()
        self.u = scipy.sparse.linalg.spsolve(K_csr, F_mod)
        return self.u

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def evaluate_at(self, xi_query: np.ndarray) -> np.ndarray:
        """Interpolate P1 solution at arbitrary ξ-coordinates.

        Uses structured grid lookup (O(1) per point) + barycentric interpolation.
        Falls back to nearest-node for points outside the mesh.

        Args:
            xi_query: (N, 3) chart-local coordinates

        Returns:
            u_interp: (N,) interpolated solution values
        """
        if self.u is None:
            return np.zeros(len(xi_query))

        xi_query = np.atleast_2d(xi_query)
        n_query = len(xi_query)

        if self.n_nodes == 0 or len(self.elements) == 0:
            return np.zeros(n_query)

        r = self.r
        h = self.h
        nc = self.n_cells

        # Find containing hex cell for each query
        grid_idx = np.floor((xi_query + r) / h).astype(np.int64)
        grid_idx = np.clip(grid_idx, 0, nc - 1)
        hex_idx = grid_idx[:, 0] * nc * nc + grid_idx[:, 1] * nc + grid_idx[:, 2]

        result = np.full(n_query, np.nan)

        # Process point by point (needed due to variable tet count per hex)
        for q in range(n_query):
            hi = hex_idx[q]
            if hi < len(self._hex_tet_lists):
                for tet_idx in self._hex_tet_lists[hi]:
                    nids = self.elements[tet_idx]
                    xe = self.nodes[nids]
                    lam = self._barycentric(xi_query[q], xe)
                    if lam is not None and np.all(lam >= -1e-10):
                        result[q] = np.dot(lam, self.u[nids])
                        break

        # Fallback for unfound points: nearest node via KDTree
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            if not hasattr(self, '_kdtree') or self._kdtree is None:
                from scipy.spatial import cKDTree
                self._kdtree = cKDTree(self.nodes)
            _, nearest = self._kdtree.query(xi_query[nan_mask])
            result[nan_mask] = self.u[nearest]

        return result

    @staticmethod
    def _barycentric(p: np.ndarray, verts: np.ndarray) -> Optional[np.ndarray]:
        """Compute barycentric coordinates of point p in tet with given vertices.

        Returns (4,) array of barycentric coords, or None if singular.
        """
        # T = [v1-v0 | v2-v0 | v3-v0]^T
        T = (verts[1:] - verts[0]).T  # (3, 3)
        try:
            lam_123 = np.linalg.solve(T, p - verts[0])
        except np.linalg.LinAlgError:
            return None
        lam_0 = 1.0 - lam_123.sum()
        return np.array([lam_0, lam_123[0], lam_123[1], lam_123[2]])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_dofs(self) -> int:
        return self.n_nodes

    @property
    def n_elements(self) -> int:
        return len(self.elements)
