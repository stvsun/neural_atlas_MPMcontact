"""Multi-chart Schwarz solver for vector elasticity on mapped coordinate charts.

Extends the scalar SchwarzFEMSolver to 3D vector fields (displacement)
using ChartVectorFEMSolver per chart and multiplicative Schwarz iteration
with displacement exchange at chart overlaps.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from common.geometry import local_coords, invert_decoder
from common.models import ChartDecoder, MaskNet
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver


class SchwarzVectorFEMSolver:
    """Multi-chart Schwarz solver for vector elasticity.

    Parameters
    ----------
    chart_solvers : list of ChartVectorFEMSolver
        One per chart, already constructed with mesh/decoder.
    seeds : torch.Tensor (M, 3)
        Chart seed positions.
    decoders : list of ChartDecoder or None
        Chart decoders (None for identity).
    decoder_kwargs_list : list of dict
        Per-chart kwargs for decoder forward calls.
    neighbors : list of list of int
        Overlap neighbor graph.
    """

    def __init__(
        self,
        chart_solvers: List[ChartVectorFEMSolver],
        seeds: torch.Tensor,
        decoders: List[Optional[ChartDecoder]] = None,
        decoder_kwargs_list: List[dict] = None,
        neighbors: List[List[int]] = None,
    ):
        self.chart_solvers = chart_solvers
        self.n_charts = len(chart_solvers)
        self.seeds = seeds
        self.decoders = decoders or [None] * self.n_charts
        self.decoder_kwargs_list = decoder_kwargs_list or [{}] * self.n_charts

        if neighbors is None:
            # Default: all charts are neighbors
            neighbors = [[j for j in range(self.n_charts) if j != i]
                         for i in range(self.n_charts)]
        self.neighbors = neighbors

        # Per-chart solutions
        self.u_charts: List[Optional[torch.Tensor]] = [None] * self.n_charts

    def solve(
        self,
        stress_fn: Callable,
        tangent_fn: Callable,
        phys_bc_fn: Callable,
        f_ext_fn: Optional[Callable] = None,
        max_schwarz_iters: int = 20,
        tol: float = 1e-6,
        newton_max_iter: int = 20,
        newton_tol: float = 1e-8,
    ) -> List[torch.Tensor]:
        """Run multiplicative Schwarz iteration for vector elasticity.

        Parameters
        ----------
        stress_fn : callable
            F (M,3,3) -> P (M,3,3).
        tangent_fn : callable
            F (M,3,3) -> C (M,9,9).
        phys_bc_fn : callable
            (nodes_phys: ndarray (N,3)) -> (u_bc: ndarray (N,3), mask: ndarray (N,) bool)
            Returns prescribed displacement and boolean mask for physical boundary nodes.
        f_ext_fn : callable, optional
            (nodes_phys: ndarray (N,3)) -> ndarray (N,3) body forces.
        max_schwarz_iters : int
            Max outer Schwarz iterations.
        tol : float
            Convergence tolerance on displacement change.
        newton_max_iter, newton_tol : int, float
            Per-chart Newton parameters.

        Returns
        -------
        u_charts : list of torch.Tensor
            Per-chart displacement fields.
        """
        device = self.chart_solvers[0].device
        dtype = self.chart_solvers[0].dtype

        for schwarz_iter in range(max_schwarz_iters):
            max_change = 0.0

            for i in range(self.n_charts):
                solver = self.chart_solvers[i]
                if solver.n_nodes == 0:
                    continue

                nodes_ref = solver.nodes  # (N, 3) in chart coords
                nodes_phys = solver.nodes_phys  # (N, 3) in physical coords
                n_nodes = solver.n_nodes

                # Physical boundary conditions
                nodes_phys_np = nodes_phys.detach().cpu().numpy()
                u_bc_phys, phys_mask = phys_bc_fn(nodes_phys_np)

                # Convert to torch
                bc_mask = torch.tensor(phys_mask, dtype=torch.bool, device=device)
                u_bc = torch.tensor(u_bc_phys, dtype=dtype, device=device)

                # Artificial boundary: get displacement from neighbor charts
                art_mask = solver.boundary_mask & ~bc_mask
                if art_mask.any() and schwarz_iter > 0:
                    art_u = self._interpolate_from_neighbors(i, nodes_phys, art_mask)
                    if art_u is not None:
                        u_bc[art_mask] = art_u
                        bc_mask = bc_mask | art_mask

                # External forces
                if f_ext_fn is not None:
                    f_ext = torch.tensor(
                        f_ext_fn(nodes_phys_np), dtype=dtype, device=device,
                    )
                else:
                    f_ext = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

                # Solve this chart
                u_old = self.u_charts[i]
                u_new = solver.solve_nonlinear(
                    stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
                    u_init=u_old,
                    max_iter=newton_max_iter, tol=newton_tol,
                )

                # Track convergence
                if u_old is not None:
                    change = (u_new - u_old).norm().item()
                    scale = max(u_new.norm().item(), 1e-12)
                    max_change = max(max_change, change / scale)

                self.u_charts[i] = u_new

            if schwarz_iter > 0 and max_change < tol:
                print(f"  [Schwarz] converged at iter {schwarz_iter+1}: "
                      f"max_change={max_change:.2e}")
                break

            if schwarz_iter > 0:
                print(f"  [Schwarz] iter {schwarz_iter+1}: max_change={max_change:.2e}")

        return self.u_charts

    def _interpolate_from_neighbors(
        self, chart_i: int, nodes_phys: torch.Tensor, art_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Interpolate displacement at chart i's artificial boundary from neighbors.

        For each art BC node in chart i, find the nearest neighbor chart
        that has a solution, project the node into that chart's reference
        coordinates, and evaluate the displacement via linear interpolation.
        """
        art_indices = torch.where(art_mask)[0]
        if len(art_indices) == 0:
            return None

        art_nodes_phys = nodes_phys[art_indices]  # (N_art, 3)
        result = torch.zeros(len(art_indices), 3,
                             device=nodes_phys.device, dtype=nodes_phys.dtype)
        valid = torch.zeros(len(art_indices), dtype=torch.bool)

        for j in self.neighbors[chart_i]:
            if self.u_charts[j] is None:
                continue

            solver_j = self.chart_solvers[j]
            if solver_j.n_nodes == 0:
                continue

            # Project art nodes into chart j's reference coords
            if self.decoders[j] is not None:
                kw = self.decoder_kwargs_list[j]
                with torch.enable_grad():
                    xi_j = invert_decoder(
                        self.decoders[j], art_nodes_phys,
                        kw["seed"], kw["t1"], kw["t2"], kw["n"], kw["chart_scale"],
                        max_iter=10, tol=1e-6,
                    )
            else:
                xi_j = art_nodes_phys  # identity mapping

            # Check which nodes fall inside chart j's mesh domain
            r_j = solver_j.r
            in_mesh = torch.all(torch.abs(xi_j) <= r_j, dim=1)

            if not in_mesh.any():
                continue

            # Nearest-node interpolation from chart j's solution
            xi_j_np = xi_j.detach().cpu().numpy()
            u_j = self.u_charts[j].detach().cpu().numpy()
            nodes_j_np = solver_j.nodes.detach().cpu().numpy()

            for k_idx in range(len(art_indices)):
                if valid[k_idx] or not in_mesh[k_idx]:
                    continue
                # Find nearest node in chart j
                dists = np.linalg.norm(nodes_j_np - xi_j_np[k_idx], axis=1)
                nearest = np.argmin(dists)
                result[k_idx] = torch.tensor(u_j[nearest], dtype=result.dtype)
                valid[k_idx] = True

        if valid.any():
            return result
        return None

    def get_stress(self, stress_fn: Callable) -> List[np.ndarray]:
        """Compute stress at all elements in all charts.

        Returns list of (M_i, 3, 3) arrays.
        """
        stresses = []
        for i in range(self.n_charts):
            if self.u_charts[i] is None:
                stresses.append(np.zeros((0, 3, 3)))
                continue
            F = self.chart_solvers[i].compute_F(self.u_charts[i])
            P = stress_fn(F)
            stresses.append(P.detach().cpu().numpy())
        return stresses
