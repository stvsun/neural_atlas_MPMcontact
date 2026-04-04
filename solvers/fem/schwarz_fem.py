"""Multi-chart Schwarz FEM solver on mapped coordinate charts.

Implements multiplicative Schwarz domain decomposition for solving
PDEs on overlapping atlas charts, each with a P1 tetrahedral FEM solver.

The solver is PDE-agnostic: forcing and boundary conditions are supplied
as callables, making it reusable across Poisson, elasticity, etc.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from common.geometry import invert_decoder, local_coords
from common.models import ChartDecoder, MaskNet
from common.schwarz import choose_color_groups, load_atlas_models
from common.utils import resolve_device, resolve_dtype
from solvers.fem.chart_fem_solver import ChartFEMSolver


class SchwarzFEMSolver:
    """Multi-chart Schwarz FEM solver for mapped coordinate charts.

    Parameters
    ----------
    atlas_data : dict or str
        Atlas data dictionary (from .npz) or path to .npz file.
        Must contain: seed_points, frame_t1, frame_t2, frame_n,
        support_radii, membership.
    decoders : list of ChartDecoder
        Trained chart decoders.
    masks : list of MaskNet
        Trained chart masks.
    sdf_oracle : object, optional
        SDF oracle with .sdf(x) method for mesh filtering.
    n_cells : int
        Grid resolution per axis per chart.
    sdf_threshold : float
        SDF threshold for element filtering.
    meta_json : str, optional
        Path to atlas meta.json with precomputed color_groups.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision.
    """

    def __init__(
        self,
        atlas_data,
        decoders: List[ChartDecoder],
        masks: List[MaskNet],
        sdf_oracle=None,
        n_cells: int = 16,
        sdf_threshold: float = -0.005,
        meta_json: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.device = device
        self.dtype = dtype

        # Load atlas data
        if isinstance(atlas_data, str):
            atlas_data = dict(np.load(atlas_data, allow_pickle=True))
        self.atlas_data = atlas_data

        self.decoders = decoders
        self.masks = masks
        self.n_charts = len(decoders)

        # Extract chart parameters as tensors
        seeds = atlas_data["seed_points"]  # (M, 3)
        t1s = atlas_data["frame_t1"]       # (M, 3)
        t2s = atlas_data["frame_t2"]       # (M, 3)
        ns = atlas_data["frame_n"]         # (M, 3)
        radii = atlas_data["support_radii"]  # (M,)

        self.seeds_t = torch.tensor(seeds, device=device, dtype=dtype)
        self.t1_t = torch.tensor(t1s, device=device, dtype=dtype)
        self.t2_t = torch.tensor(t2s, device=device, dtype=dtype)
        self.nvec_t = torch.tensor(ns, device=device, dtype=dtype)
        self.support_r_t = torch.tensor(radii, device=device, dtype=dtype)

        # Build overlap neighbor graph
        membership = atlas_data["membership"]  # (N, M)
        self.neighbors = self._build_neighbors(membership)

        # Determine color groups for multiplicative Schwarz
        membership_np = np.array(membership, dtype=np.float32)
        self.color_groups = choose_color_groups(meta_json, self.n_charts, membership_np)

        # Build per-chart FEM solvers
        self.fem_solvers: List[ChartFEMSolver] = []
        for i in range(self.n_charts):
            solver = ChartFEMSolver(
                chart_id=i,
                decoder=decoders[i],
                seed=self.seeds_t[i],
                t1=self.t1_t[i],
                t2=self.t2_t[i],
                n_vec=self.nvec_t[i],
                support_r=self.support_r_t[i],
                n_cells=n_cells,
                sdf_oracle=sdf_oracle,
                sdf_threshold=sdf_threshold,
                device=device,
                dtype=dtype,
            )
            self.fem_solvers.append(solver)

        # Precompute diffusion tensors
        for s in self.fem_solvers:
            s.compute_diffusion_tensors()

        # BC cache (populated during solve)
        self._bc_cache: Optional[Dict] = None

    def _build_neighbors(self, membership) -> List[List[int]]:
        """Build overlap neighbor graph from membership matrix."""
        membership = np.array(membership, dtype=bool)
        neighbors = []
        for i in range(self.n_charts):
            mi = membership[:, i]
            nbrs = []
            for j in range(self.n_charts):
                if i != j and np.any(mi & membership[:, j]):
                    nbrs.append(j)
            neighbors.append(nbrs)
        return neighbors

    def solve(
        self,
        forcing_fn: Callable[[np.ndarray], np.ndarray],
        phys_bc_fn: Callable[[np.ndarray], np.ndarray],
        max_iters: int = 50,
        tol: float = 0.01,
        eval_points: Optional[torch.Tensor] = None,
        exact_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Dict:
        """Run multiplicative Schwarz iteration.

        Parameters
        ----------
        forcing_fn : callable
            f(x_phys) -> (N,) forcing values at physical coordinates.
        phys_bc_fn : callable
            g(x_phys) -> (N,) Dirichlet BC values at physical boundary.
        max_iters : int
            Maximum Schwarz iterations.
        tol : float
            Convergence tolerance on relative L2 change between iterations.
        eval_points : torch.Tensor, optional
            (N, 3) evaluation points for convergence monitoring.
        exact_fn : callable, optional
            Exact solution for error computation.

        Returns
        -------
        result : dict
            Keys: n_iters, converged, history, u_pred (if eval_points given).
        """
        # Assemble per-chart systems
        for solver in self.fem_solvers:
            solver.assemble(forcing_fn=forcing_fn)

        # Precompute Schwarz BC geometry
        self._bc_cache = self._precompute_schwarz_bc(phys_bc_fn)

        history = {"rel_change": [], "rel_l2_error": []}
        prev_u = None

        for iteration in range(1, max_iters + 1):
            # Multiplicative Schwarz: update charts by color group
            for color_group in self.color_groups:
                for chart_i in color_group:
                    solver_i = self.fem_solvers[chart_i]
                    if solver_i.n_nodes == 0:
                        continue

                    # Physical BCs
                    phys_bc = self._compute_phys_bc(chart_i, phys_bc_fn)

                    # Artificial BCs from neighbors
                    art_bc = self._compute_schwarz_bc(chart_i)

                    solver_i.solve(phys_bc, art_bc)

            # Convergence check
            if eval_points is not None:
                u_pred = self.evaluate(eval_points)
                if prev_u is not None:
                    change = np.sqrt(np.mean((u_pred - prev_u) ** 2))
                    scale = max(np.sqrt(np.mean(u_pred ** 2)), 1e-12)
                    rel_change = change / scale
                    history["rel_change"].append(rel_change)

                    if exact_fn is not None:
                        u_true = exact_fn(eval_points.cpu().numpy())
                        err = u_pred - u_true
                        rel_l2 = float(np.sqrt(np.mean(err ** 2) / max(np.mean(u_true ** 2), 1e-12)))
                        history["rel_l2_error"].append(rel_l2)

                    if rel_change < tol:
                        return {
                            "n_iters": iteration,
                            "converged": True,
                            "history": history,
                            "u_pred": u_pred,
                        }
                prev_u = u_pred.copy()

        result = {"n_iters": max_iters, "converged": False, "history": history}
        if eval_points is not None:
            result["u_pred"] = self.evaluate(eval_points)
        return result

    def evaluate(self, eval_points: torch.Tensor) -> np.ndarray:
        """Compute PoU-blended FEM solution at evaluation points.

        Parameters
        ----------
        eval_points : torch.Tensor
            (N, 3) physical coordinates.

        Returns
        -------
        u_pred : np.ndarray
            (N,) blended solution values.
        """
        n_points = eval_points.shape[0]
        logits_all = np.full((self.n_charts, n_points), -1e10)
        vals_all = np.zeros((self.n_charts, n_points))
        valid_all = np.zeros((self.n_charts, n_points), dtype=bool)

        for i in range(self.n_charts):
            with torch.no_grad():
                xi_linear = local_coords(
                    eval_points, self.seeds_t[i], self.t1_t[i],
                    self.t2_t[i], self.nvec_t[i],
                )
                logits_all[i] = self.masks[i](
                    xi_linear, chart_scale=self.support_r_t[i]
                ).cpu().numpy()

            solver_i = self.fem_solvers[i]
            if solver_i.u is not None and solver_i.n_nodes > 0:
                with torch.enable_grad():
                    xi_star = invert_decoder(
                        decoder=self.decoders[i],
                        x_target=eval_points,
                        seed=self.seeds_t[i],
                        t1=self.t1_t[i],
                        t2=self.t2_t[i],
                        n_vec=self.nvec_t[i],
                        chart_scale=self.support_r_t[i],
                        xi_init=xi_linear,
                        max_iter=10,
                        tol=1e-8,
                    )
                xi_star_np = xi_star.cpu().numpy()
                r_i = solver_i.r
                in_mesh = np.all(np.abs(xi_star_np) <= r_i, axis=1)
                valid_all[i] = in_mesh
                if np.any(in_mesh):
                    vals_all[i, in_mesh] = solver_i.evaluate_at(xi_star_np[in_mesh])

        # Softmax PoU blending
        logits_all = logits_all.T  # (N, M)
        vals_all = vals_all.T
        valid_all = valid_all.T

        masked_logits = np.where(valid_all, logits_all, -1e30)
        logits_shifted = masked_logits - masked_logits.max(axis=1, keepdims=True)
        weights = np.exp(logits_shifted)
        weight_sum = weights.sum(axis=1, keepdims=True)

        no_valid = weight_sum.ravel() < 1e-20
        if np.any(no_valid):
            ls_full = logits_all[no_valid] - logits_all[no_valid].max(axis=1, keepdims=True)
            weights[no_valid] = np.exp(ls_full)
            weight_sum[no_valid] = weights[no_valid].sum(axis=1, keepdims=True)

        weights /= np.maximum(weight_sum, 1e-20)
        return np.sum(weights * vals_all, axis=1)

    def _compute_phys_bc(self, chart_i: int, phys_bc_fn) -> Dict[int, float]:
        """Physical boundary conditions for chart i."""
        solver = self.fem_solvers[chart_i]
        if len(solver.phys_bc_nodes) == 0:
            return {}
        xi = solver.nodes[solver.phys_bc_nodes]
        x_phys = solver._decode_points(xi)
        u_bc = phys_bc_fn(x_phys)
        return {int(idx): float(val) for idx, val in zip(solver.phys_bc_nodes, u_bc)}

    def _precompute_schwarz_bc(self, phys_bc_fn) -> Dict:
        """Precompute Newton inversions + PoU weights for Schwarz BCs."""
        cache = {}
        for chart_i in range(self.n_charts):
            solver_i = self.fem_solvers[chart_i]
            if len(solver_i.art_bc_nodes) == 0:
                cache[chart_i] = {"empty": True}
                continue

            xi_art = solver_i.nodes[solver_i.art_bc_nodes]
            x_phys = solver_i._decode_points(xi_art)
            x_phys_t = torch.tensor(x_phys, device=self.device, dtype=self.dtype)

            nbrs = self.neighbors[chart_i]
            if len(nbrs) == 0:
                u_fallback = phys_bc_fn(x_phys)
                cache[chart_i] = {
                    "empty": False,
                    "fallback_values": {int(idx): float(val) for idx, val in
                                        zip(solver_i.art_bc_nodes, u_fallback)},
                    "no_neighbors": True,
                }
                continue

            nbr_data = []
            for j in nbrs:
                xi_j_linear = local_coords(
                    x_phys_t, self.seeds_t[j], self.t1_t[j],
                    self.t2_t[j], self.nvec_t[j],
                )
                xi_j_lin_np = xi_j_linear.cpu().numpy()
                r_j = float(self.support_r_t[j].item())
                in_support = np.all(np.abs(xi_j_lin_np) <= 1.25 * r_j, axis=1)

                if not np.any(in_support):
                    continue

                with torch.no_grad():
                    logit_j = self.masks[j](
                        xi_j_linear, chart_scale=self.support_r_t[j]
                    ).cpu().numpy()

                xi_j_np = xi_j_lin_np.copy()
                if np.any(in_support):
                    with torch.enable_grad():
                        xi_j_star = invert_decoder(
                            decoder=self.decoders[j],
                            x_target=x_phys_t[in_support],
                            seed=self.seeds_t[j],
                            t1=self.t1_t[j],
                            t2=self.t2_t[j],
                            n_vec=self.nvec_t[j],
                            chart_scale=self.support_r_t[j],
                            xi_init=xi_j_linear[in_support],
                            max_iter=10,
                            tol=1e-8,
                        )
                    xi_j_np[in_support] = xi_j_star.cpu().numpy()

                r_j_mesh = self.fem_solvers[j].r
                in_mesh = in_support & np.all(np.abs(xi_j_np) <= r_j_mesh, axis=1)

                nbr_data.append({
                    "chart_j": j,
                    "xi_inverted": xi_j_np,
                    "logits": logit_j,
                    "in_mesh": in_mesh,
                })

            n_art = len(solver_i.art_bc_nodes)
            if len(nbr_data) == 0:
                u_fallback = phys_bc_fn(x_phys)
                cache[chart_i] = {
                    "empty": False,
                    "fallback_values": {int(idx): float(val) for idx, val in
                                        zip(solver_i.art_bc_nodes, u_fallback)},
                    "no_neighbors": True,
                }
                continue

            logits_arr = np.array([nd["logits"] for nd in nbr_data])
            in_mesh_arr = np.array([nd["in_mesh"] for nd in nbr_data])

            masked_logits = np.where(in_mesh_arr, logits_arr, -1e30)
            max_logits = masked_logits.max(axis=0, keepdims=True)
            weights = np.exp(masked_logits - max_logits)
            weight_sum = weights.sum(axis=0, keepdims=True)

            no_valid = weight_sum.ravel() < 1e-20
            if np.any(no_valid):
                ml_full = logits_arr[:, no_valid].max(axis=0, keepdims=True)
                weights[:, no_valid] = np.exp(logits_arr[:, no_valid] - ml_full)
                weight_sum[:, no_valid] = weights[:, no_valid].sum(axis=0, keepdims=True)
            weights /= np.maximum(weight_sum, 1e-20)

            any_valid = in_mesh_arr.any(axis=0)
            no_valid_mask = ~any_valid
            fallback_u = np.zeros(n_art)
            if np.any(no_valid_mask):
                fallback_u[no_valid_mask] = phys_bc_fn(x_phys[no_valid_mask])

            cache[chart_i] = {
                "empty": False,
                "no_neighbors": False,
                "nbr_data": nbr_data,
                "weights": weights,
                "no_valid_mask": no_valid_mask,
                "fallback_u": fallback_u,
                "x_phys": x_phys,
            }

        return cache

    def _compute_schwarz_bc(self, chart_i: int) -> Dict[int, float]:
        """Compute Schwarz BCs using precomputed geometry cache."""
        solver_i = self.fem_solvers[chart_i]
        entry = self._bc_cache[chart_i]

        if entry["empty"]:
            return {}
        if entry.get("no_neighbors", False):
            return entry["fallback_values"]

        nbr_data = entry["nbr_data"]
        weights = entry["weights"]
        n_art = len(solver_i.art_bc_nodes)

        vals_arr = np.zeros((len(nbr_data), n_art))
        for ni, nd in enumerate(nbr_data):
            j = nd["chart_j"]
            if self.fem_solvers[j].u is not None and np.any(nd["in_mesh"]):
                vals_arr[ni, nd["in_mesh"]] = self.fem_solvers[j].evaluate_at(
                    nd["xi_inverted"][nd["in_mesh"]]
                )

        u_blend = np.sum(weights * vals_arr, axis=0)

        no_valid_mask = entry.get("no_valid_mask")
        if no_valid_mask is not None and np.any(no_valid_mask):
            u_blend[no_valid_mask] = entry["fallback_u"][no_valid_mask]

        return {int(idx): float(val) for idx, val in zip(solver_i.art_bc_nodes, u_blend)}

    def add_charts(
        self,
        pairs,
        sdf_oracle=None,
        n_cells: Optional[int] = None,
        sdf_threshold: float = -0.005,
    ) -> int:
        """Add spawned chart pairs to the solver (Phase 3 dynamic spawning).

        Creates new ChartDecoder (warm-started from parent) and MaskNet
        for each side of each SpawnedChartPair, builds FEM solvers,
        rebuilds the neighbor graph and color groups.

        Parameters
        ----------
        pairs : list of SpawnedChartPair
            Chart pairs from ChartSpawner.spawn_from_event().
        sdf_oracle : object, optional
            SDF oracle for mesh filtering in new charts.
        n_cells : int, optional
            Grid resolution for new charts. Defaults to first chart's n_cells.
        sdf_threshold : float
            SDF filtering threshold.

        Returns
        -------
        n_added : int
            Number of new charts added (2 per pair).
        """
        if n_cells is None:
            n_cells = self.fem_solvers[0].n_cells if self.fem_solvers else 16

        n_added = 0
        for pair in pairs:
            for side in ("plus", "minus"):
                seed = getattr(pair, f"seed_{side}")
                frame = getattr(pair, f"frame_{side}")

                seed_t = torch.tensor(seed, device=self.device, dtype=self.dtype)
                t1_t = torch.tensor(frame[0], device=self.device, dtype=self.dtype)
                t2_t = torch.tensor(frame[1], device=self.device, dtype=self.dtype)
                n_t = torch.tensor(frame[2], device=self.device, dtype=self.dtype)
                r_t = torch.tensor(pair.radius, device=self.device, dtype=self.dtype)

                # Create warm-started decoder from parent
                parent_dec = self.decoders[pair.parent_chart]
                new_dec = ChartDecoder(
                    width=parent_dec.net.out.in_features,
                    depth=len(parent_dec.net.hidden),
                ).to(device=self.device, dtype=self.dtype)
                new_dec.warm_start_from(parent_dec)

                # Create fresh mask
                parent_mask = self.masks[pair.parent_chart]
                new_mask = MaskNet(
                    width=parent_mask.net.out.in_features,
                    depth=len(parent_mask.net.hidden),
                ).to(device=self.device, dtype=self.dtype)

                self.decoders.append(new_dec)
                self.masks.append(new_mask)

                # Extend parameter tensors
                self.seeds_t = torch.cat([self.seeds_t, seed_t.unsqueeze(0)])
                self.t1_t = torch.cat([self.t1_t, t1_t.unsqueeze(0)])
                self.t2_t = torch.cat([self.t2_t, t2_t.unsqueeze(0)])
                self.nvec_t = torch.cat([self.nvec_t, n_t.unsqueeze(0)])
                self.support_r_t = torch.cat([self.support_r_t, r_t.unsqueeze(0)])

                # Build FEM solver for new chart
                new_solver = ChartFEMSolver(
                    chart_id=self.n_charts + n_added,
                    decoder=new_dec,
                    seed=seed_t, t1=t1_t, t2=t2_t, n_vec=n_t,
                    support_r=r_t,
                    n_cells=n_cells,
                    sdf_oracle=sdf_oracle,
                    sdf_threshold=sdf_threshold,
                    device=self.device,
                    dtype=self.dtype,
                )
                new_solver.compute_diffusion_tensors()
                self.fem_solvers.append(new_solver)
                n_added += 1

        self.n_charts += n_added

        # Rebuild neighbor graph from scratch
        # Use simple distance-based overlap: charts overlap if seeds are within 2*max(radii)
        self.neighbors = []
        for i in range(self.n_charts):
            nbrs = []
            si = self.seeds_t[i]
            ri = self.support_r_t[i].item()
            for j in range(self.n_charts):
                if i == j:
                    continue
                sj = self.seeds_t[j]
                rj = self.support_r_t[j].item()
                dist = torch.linalg.norm(si - sj).item()
                if dist < 2.0 * (ri + rj):
                    nbrs.append(j)
            self.neighbors.append(nbrs)

        # Re-color
        membership_fake = np.zeros((1, self.n_charts), dtype=np.float32)
        self.color_groups = choose_color_groups(None, self.n_charts, membership_fake)

        # Invalidate BC cache
        self._bc_cache = None

        return n_added

    @classmethod
    def from_checkpoints(
        cls,
        atlas_data_path: str,
        atlas_checkpoint: str,
        sdf_checkpoint: Optional[str] = None,
        meta_json: Optional[str] = None,
        n_cells: int = 16,
        sdf_threshold: float = -0.005,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "SchwarzFEMSolver":
        """Convenience constructor from checkpoint file paths.

        Parameters
        ----------
        atlas_data_path : str
            Path to atlas .npz file.
        atlas_checkpoint : str
            Path to trained atlas .pt checkpoint.
        sdf_checkpoint : str, optional
            Path to SDF .pt checkpoint (enables mesh filtering).
        meta_json : str, optional
            Path to atlas meta.json.
        n_cells : int
            Grid cells per axis.
        """
        dev = resolve_device(device)
        dt = resolve_dtype(dtype, dev)

        decoders, masks, _ = load_atlas_models(atlas_checkpoint, dev, dt)
        atlas_data = dict(np.load(atlas_data_path, allow_pickle=True))

        sdf_oracle = None
        if sdf_checkpoint is not None:
            sdf_oracle = cls._load_sdf_oracle(sdf_checkpoint, dev, dt)

        return cls(
            atlas_data=atlas_data,
            decoders=decoders,
            masks=masks,
            sdf_oracle=sdf_oracle,
            n_cells=n_cells,
            sdf_threshold=sdf_threshold,
            meta_json=meta_json,
            device=dev,
            dtype=dt,
        )

    @staticmethod
    def _load_sdf_oracle(sdf_checkpoint: str, device, dtype):
        """Load SDF oracle from checkpoint."""
        from common.models import MLP

        ckpt = torch.load(sdf_checkpoint, map_location=device)
        model_kwargs = ckpt.get("model_kwargs", {"width": 128, "depth": 6})

        class SDFNet(torch.nn.Module):
            def __init__(self, width=128, depth=6):
                super().__init__()
                self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)
            def forward(self, x):
                return self.net(x).squeeze(-1)

        net = SDFNet(**model_kwargs).to(device=device, dtype=dtype)
        net.load_state_dict(ckpt["model_state"])
        net.eval()

        center = torch.tensor(ckpt["center"], device=device, dtype=dtype)
        scale = float(ckpt["scale"])

        class SDFOracle:
            def __init__(self, model, center, scale):
                self.model = model
                self.center = center
                self.scale_val = scale
            def sdf(self, x):
                with torch.no_grad():
                    x_norm = (x - self.center.unsqueeze(0)) / self.scale_val
                    return self.model(x_norm) * self.scale_val

        return SDFOracle(net, center, scale)
