"""Multi-chart Schwarz solver for vector elasticity on mapped coordinate charts.

Extends the scalar SchwarzFEMSolver to 3D vector fields (displacement)
using ChartVectorFEMSolver per chart and multiplicative Schwarz iteration
with displacement exchange at chart overlaps.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

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
        parallel: bool = False,
        n_workers: int = 4,
    ):
        self.chart_solvers = chart_solvers
        self.n_charts = len(chart_solvers)
        self.seeds = seeds
        self.decoders = decoders or [None] * self.n_charts
        self.decoder_kwargs_list = decoder_kwargs_list or [{}] * self.n_charts
        self.parallel = parallel
        self.n_workers = n_workers

        if neighbors is None:
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

        def _solve_chart(i):
            """Solve a single chart — can run in a thread."""
            solver = self.chart_solvers[i]
            if solver.n_nodes == 0:
                return i, None, 0.0

            nodes_phys = solver.nodes_phys
            n_nodes = solver.n_nodes

            nodes_phys_np = nodes_phys.detach().cpu().numpy()
            u_bc_phys, phys_mask = phys_bc_fn(nodes_phys_np)

            bc_mask_i = torch.tensor(phys_mask, dtype=torch.bool, device=device)
            u_bc_i = torch.tensor(u_bc_phys, dtype=dtype, device=device)

            art_mask = solver.boundary_mask & ~bc_mask_i
            if art_mask.any() and schwarz_iter > 0:
                art_u = self._interpolate_from_neighbors(i, nodes_phys, art_mask)
                if art_u is not None:
                    u_bc_i[art_mask] = art_u
                    bc_mask_i = bc_mask_i | art_mask

            if f_ext_fn is not None:
                f_ext_i = torch.tensor(f_ext_fn(nodes_phys_np), dtype=dtype, device=device)
            else:
                f_ext_i = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

            u_old = self.u_charts[i]
            u_new = solver.solve_nonlinear(
                stress_fn, tangent_fn, f_ext_i, u_bc_i, bc_mask_i,
                u_init=u_old,
                max_iter=newton_max_iter, tol=newton_tol,
            )

            change = 0.0
            if u_old is not None:
                change = (u_new - u_old).norm().item() / max(u_new.norm().item(), 1e-12)

            return i, u_new, change

        for schwarz_iter in range(max_schwarz_iters):
            max_change = 0.0

            if self.parallel and self.n_charts > 1:
                # Parallel: solve all charts concurrently
                with ThreadPoolExecutor(max_workers=min(self.n_workers, self.n_charts)) as ex:
                    futures = [ex.submit(_solve_chart, i) for i in range(self.n_charts)]
                    for f in futures:
                        idx, u_new, change = f.result()
                        if u_new is not None:
                            self.u_charts[idx] = u_new
                            max_change = max(max_change, change)
            else:
                # Serial: solve charts sequentially
                for i in range(self.n_charts):
                    idx, u_new, change = _solve_chart(i)
                    if u_new is not None:
                        self.u_charts[idx] = u_new
                        max_change = max(max_change, change)

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

        For each art BC node in chart i, project into each neighbor chart's
        reference coordinates (using decoder.inverse() for analytical decoders,
        or invert_decoder() for neural decoders), then interpolate from the
        neighbor's solution via nearest-node lookup.
        """
        art_indices = torch.where(art_mask)[0]
        if len(art_indices) == 0:
            return None

        art_nodes_phys = nodes_phys[art_indices]  # (N_art, 3)
        n_art = len(art_indices)
        result = torch.zeros(n_art, 3, device=nodes_phys.device, dtype=nodes_phys.dtype)
        weight = torch.zeros(n_art, device=nodes_phys.device, dtype=nodes_phys.dtype)

        for j in self.neighbors[chart_i]:
            if self.u_charts[j] is None:
                continue

            solver_j = self.chart_solvers[j]
            if solver_j.n_nodes == 0:
                continue

            # Project art nodes into chart j's reference coords
            decoder_j = self.decoders[j]
            if decoder_j is not None and hasattr(decoder_j, 'inverse'):
                # Analytical decoder with closed-form inverse
                with torch.no_grad():
                    xi_j = decoder_j.inverse(art_nodes_phys)
            elif decoder_j is not None:
                # Neural decoder — use Newton inversion
                kw = self.decoder_kwargs_list[j]
                with torch.enable_grad():
                    xi_j = invert_decoder(
                        decoder_j, art_nodes_phys,
                        kw["seed"], kw["t1"], kw["t2"], kw["n"], kw["chart_scale"],
                        max_iter=10, tol=1e-6,
                    )
            else:
                xi_j = art_nodes_phys  # identity mapping

            # Check which nodes fall inside chart j's reference domain [-r, r]^3
            r_j = solver_j.r
            in_mesh = torch.all(torch.abs(xi_j) <= r_j, dim=1)

            if not in_mesh.any():
                continue

            # Nearest-node interpolation from chart j's solution
            xi_j_np = xi_j.detach().cpu().numpy()
            u_j = self.u_charts[j].detach().cpu().numpy()
            nodes_j_np = solver_j.nodes.detach().cpu().numpy()

            # Vectorized nearest-node lookup
            in_idx = torch.where(in_mesh)[0]
            for k in in_idx:
                k_int = k.item()
                dists = np.sum((nodes_j_np - xi_j_np[k_int])**2, axis=1)
                nearest = np.argmin(dists)
                u_interp = torch.tensor(u_j[nearest], dtype=result.dtype)

                # Distance-based weighting: closer to chart j center = higher weight
                d_center = torch.linalg.norm(xi_j[k_int]).item()
                w = max(1.0 - d_center / r_j, 0.01)

                result[k_int] += w * u_interp
                weight[k_int] += w

        # Normalize by total weight
        has_data = weight > 0
        if has_data.any():
            result[has_data] /= weight[has_data].unsqueeze(1)
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
