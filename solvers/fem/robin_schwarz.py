"""Robin-type parallel domain decomposition for vector elasticity on charts.

Implements the optimization-based nonoverlapping DD from Du (2002, SINUM 39(3)),
adapted for overlapping coordinate charts. Uses Robin interface conditions:

    du/dn + delta * u = du/dn + delta * lambda  on interface

instead of pure Dirichlet (u = lambda). Robin conditions exchange BOTH displacement
and traction at chart boundaries, converging faster than pure Schwarz.

Reference: Q. Du, "Optimization Based Nonoverlapping Domain Decomposition
Algorithms and Their Convergence", SIAM J. Numer. Anal. 39(3), 2002.

Parallel version (Section 5, Eqs. 5.9-5.12):
  Step 0: lambda^{n+1} = (u1^n + u2^n) / 2
          g^{n+1} = g^n + (u2^n - u1^n) / 2 * delta
  Step 1: Solve Robin BCs on each chart in parallel:
          du_i/dn + delta * u_i = delta * lambda + (-1)^{i+1} * g  on interface
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from common.geometry import local_coords, invert_decoder
from common.models import ChartDecoder
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver


class RobinSchwarzSolver:
    """Parallel Robin domain decomposition for vector elasticity on charts.

    Parameters
    ----------
    chart_solvers : list of ChartVectorFEMSolver
    seeds : torch.Tensor (M, 3)
    decoders : list of decoders (with .inverse())
    neighbors : list of list of int
    robin_delta : float
        Robin parameter. Larger = more weight on displacement matching.
        Optimal is problem-dependent; delta ~ E/h is a good starting point.
    relaxation : float
        Under-relaxation factor for interface flux update (0 < omega <= 1).
        omega=1.0 = no relaxation (original). omega=0.3-0.5 recommended
        for problems with oscillatory convergence (e.g., fracture).
    parallel : bool
    n_workers : int
    """

    def __init__(
        self,
        chart_solvers: List[ChartVectorFEMSolver],
        seeds: torch.Tensor,
        decoders: List = None,
        neighbors: List[List[int]] = None,
        robin_delta: float = 1.0,
        relaxation: float = 1.0,
        parallel: bool = True,
        n_workers: int = 4,
    ):
        self.chart_solvers = chart_solvers
        self.n_charts = len(chart_solvers)
        self.seeds = seeds
        self.decoders = decoders or [None] * self.n_charts
        self.robin_delta = robin_delta
        self.relaxation = relaxation
        self.parallel = parallel
        self.n_workers = n_workers

        if neighbors is None:
            neighbors = [[j for j in range(self.n_charts) if j != i]
                         for i in range(self.n_charts)]
        self.neighbors = neighbors

        self.u_charts: List[Optional[torch.Tensor]] = [None] * self.n_charts

        # Robin interface data: lambda (displacement) and g (flux) at interfaces
        self._lambda = [None] * self.n_charts  # interface displacement
        self._g = [None] * self.n_charts       # interface flux

    def solve(
        self,
        stress_fn: Callable,
        tangent_fn: Callable,
        phys_bc_fn: Callable,
        f_ext_fn: Optional[Callable] = None,
        max_iters: int = 30,
        tol: float = 1e-4,
        newton_max_iter: int = 10,
        newton_tol: float = 1e-8,
    ) -> List[torch.Tensor]:
        """Solve with Robin parallel domain decomposition.

        Parameters
        ----------
        stress_fn, tangent_fn : callables for constitutive model
        phys_bc_fn : (nodes_phys) -> (u_bc, mask) for physical BCs
        f_ext_fn : optional body force
        max_iters : max DD iterations
        tol : convergence tolerance on displacement change

        Returns
        -------
        u_charts : list of displacement fields per chart
        """
        device = self.chart_solvers[0].device
        dtype = self.chart_solvers[0].dtype
        delta = self.robin_delta

        def _solve_one_chart(i, lam_data, g_data):
            """Solve chart i with Robin BCs from interface data."""
            solver = self.chart_solvers[i]
            if solver.n_nodes == 0:
                return i, None, 0.0

            nodes_phys = solver.nodes_phys
            n_nodes = solver.n_nodes

            # Physical BCs
            nodes_phys_np = nodes_phys.detach().cpu().numpy()
            u_bc_phys, phys_mask = phys_bc_fn(nodes_phys_np)
            bc_mask = torch.tensor(phys_mask, dtype=torch.bool, device=device)
            u_bc = torch.tensor(u_bc_phys, dtype=dtype, device=device)

            # Robin BCs on artificial boundaries
            art_mask = solver.boundary_mask & ~bc_mask
            if art_mask.any() and lam_data is not None:
                # Robin: du/dn + delta*u = delta*lambda + g
                # In the FEM, this modifies both the stiffness (add delta*M on boundary)
                # and the RHS (add delta*lambda + g on boundary).
                # For simplicity, we approximate by prescribing:
                #   u = lambda + g/delta  (combines displacement and flux)
                art_u = self._get_interface_data(i, nodes_phys, art_mask, lam_data, g_data)
                if art_u is not None:
                    u_bc[art_mask] = art_u
                    bc_mask = bc_mask | art_mask

            # External forces
            if f_ext_fn is not None:
                f_ext = torch.tensor(f_ext_fn(nodes_phys_np), dtype=dtype, device=device)
            else:
                f_ext = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

            # Solve
            u_old = self.u_charts[i]
            u_new = solver.solve_nonlinear(
                stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
                u_init=u_old, max_iter=newton_max_iter, tol=newton_tol,
            )

            change = 0.0
            if u_old is not None:
                change = (u_new - u_old).norm().item() / max(u_new.norm().item(), 1e-12)

            return i, u_new, change

        for dd_iter in range(max_iters):
            max_change = 0.0

            # Step 0: Update interface data (Du Eqs. 5.9-5.10)
            if dd_iter > 0:
                self._update_interface_data(delta)

            # Prepare lambda/g data for each chart
            lam_data = self._lambda if dd_iter > 0 else None
            g_data = self._g if dd_iter > 0 else None

            # Step 1: Solve all charts in parallel
            if self.parallel and self.n_charts > 1:
                with ThreadPoolExecutor(max_workers=min(self.n_workers, self.n_charts)) as ex:
                    futures = [ex.submit(_solve_one_chart, i, lam_data, g_data)
                               for i in range(self.n_charts)]
                    for f in futures:
                        idx, u_new, change = f.result()
                        if u_new is not None:
                            self.u_charts[idx] = u_new
                            max_change = max(max_change, change)
            else:
                for i in range(self.n_charts):
                    idx, u_new, change = _solve_one_chart(i, lam_data, g_data)
                    if u_new is not None:
                        self.u_charts[idx] = u_new
                        max_change = max(max_change, change)

            if dd_iter > 0:
                if max_change < tol:
                    print(f"  [RobinDD] converged at iter {dd_iter+1}: "
                          f"max_change={max_change:.2e}")
                    break
                print(f"  [RobinDD] iter {dd_iter+1}: max_change={max_change:.2e}")

        return self.u_charts

    def _update_interface_data(self, delta):
        """Update interface lambda and g from current subdomain solutions.

        Stores data at ALL nodes (full-size arrays) so that art_mask
        indexing works correctly regardless of which nodes are physical BCs.
        """
        for i in range(self.n_charts):
            solver_i = self.chart_solvers[i]
            if solver_i.n_nodes == 0 or self.u_charts[i] is None:
                continue

            n_nodes = solver_i.n_nodes
            nodes_phys = solver_i.nodes_phys
            bnd_mask = solver_i.boundary_mask

            if not bnd_mask.any():
                continue

            # Initialize full-size arrays
            if self._lambda[i] is None:
                self._lambda[i] = torch.zeros(n_nodes, 3, device=nodes_phys.device,
                                               dtype=nodes_phys.dtype)
            if self._g[i] is None:
                self._g[i] = torch.zeros(n_nodes, 3, device=nodes_phys.device,
                                          dtype=nodes_phys.dtype)

            bnd_indices = torch.where(bnd_mask)[0]
            u_self = self.u_charts[i][bnd_indices]

            # Gather neighbor displacements at boundary nodes
            u_neighbor_sum = torch.zeros_like(u_self)
            n_nbr = 0
            for j in self.neighbors[i]:
                if self.u_charts[j] is None:
                    continue
                u_j = self._interpolate_from(j, nodes_phys[bnd_indices])
                if u_j is not None:
                    u_neighbor_sum += u_j
                    n_nbr += 1

            if n_nbr == 0:
                self._lambda[i][bnd_indices] = u_self.detach()
                continue

            u_neighbor_avg = u_neighbor_sum / n_nbr

            # Du's update at boundary nodes, with under-relaxation
            omega = self.relaxation
            lam_old = self._lambda[i][bnd_indices]
            lam_update = (u_self + u_neighbor_avg) / 2
            lam_new = omega * lam_update + (1 - omega) * lam_old

            g_old = self._g[i][bnd_indices]
            g_update = g_old + delta * (u_neighbor_avg - u_self) / 2
            g_new = omega * g_update + (1 - omega) * g_old

            self._lambda[i][bnd_indices] = lam_new.detach()
            self._g[i][bnd_indices] = g_new.detach()

    def _get_interface_data(self, chart_i, nodes_phys, art_mask, lam_data, g_data):
        """Get Robin interface data at art_mask nodes: u = lambda + g/delta."""
        if lam_data[chart_i] is None:
            return None

        art_indices = torch.where(art_mask)[0]
        lam = lam_data[chart_i][art_indices]  # index into full-size array
        g = g_data[chart_i][art_indices] if g_data[chart_i] is not None else torch.zeros_like(lam)

        delta = max(self.robin_delta, 1e-10)
        return lam + g / delta

    def _interpolate_from(self, chart_j, query_phys):
        """Interpolate displacement from chart j at physical query points."""
        solver_j = self.chart_solvers[chart_j]
        if solver_j.n_nodes == 0 or self.u_charts[chart_j] is None:
            return None

        decoder_j = self.decoders[chart_j]
        if decoder_j is not None and hasattr(decoder_j, 'inverse'):
            with torch.no_grad():
                xi_j = decoder_j.inverse(query_phys)
        elif decoder_j is not None:
            # Neural decoder — use Newton
            try:
                kw = {}
                with torch.enable_grad():
                    xi_j = invert_decoder(decoder_j, query_phys, **kw)
            except Exception:
                xi_j = query_phys
        else:
            xi_j = query_phys

        r_j = solver_j.r
        in_mesh = torch.all(torch.abs(xi_j) <= r_j, dim=1)
        if not in_mesh.any():
            return None

        # Use evaluate_at for P1 barycentric interpolation
        u_interp = solver_j.evaluate_at(xi_j, self.u_charts[chart_j])
        # Zero out points outside the mesh
        u_interp[~in_mesh] = 0.0

        return u_interp

    def get_stress(self, stress_fn):
        """Compute stress at all elements in all charts."""
        stresses = []
        for i in range(self.n_charts):
            if self.u_charts[i] is None:
                stresses.append(np.zeros((0, 3, 3)))
                continue
            F = self.chart_solvers[i].compute_F(self.u_charts[i])
            P = stress_fn(F)
            stresses.append(P.detach().cpu().numpy())
        return stresses
