"""Multi-chart Schwarz MPM solver on mapped coordinate charts.

Couples multiple single-chart MPM solvers via boundary velocity exchange
at chart overlaps. Supports dynamic topology monitoring and chart spawning.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from common.geometry import local_coords, invert_decoder
from common.models import ChartDecoder, MaskNet
from common.schwarz import choose_color_groups
from solvers.mpm.chart_mpm_solver import ChartMPMSolver
from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.constitutive import ConstitutiveModel, NeoHookeanModel


class SchwarzMPMSolver:
    """Multi-chart MPM solver with Schwarz-type boundary coupling.

    Each chart has its own ChartMPMSolver and particle cloud. At each step:
    1. Each chart runs P2G -> grid solve -> G2P independently
    2. Boundary velocities are exchanged between overlapping charts
    3. Particles that escape a chart's domain are transferred to neighbors
    4. Optionally, topology monitoring triggers chart spawning

    Parameters
    ----------
    atlas_data : dict or str
        Atlas data (.npz) with seed_points, frames, support_radii, membership.
    decoders : list of ChartDecoder
        Trained chart decoders.
    masks : list of MaskNet
        Trained chart masks.
    constitutive : ConstitutiveModel
        Material model for stress computation.
    n_cells : int
        MPM grid cells per axis per chart.
    gravity : tuple, optional
        Gravity vector (e.g., (0, 0, -9.81)).
    bc_type : str
        Boundary condition type for each chart grid.
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
        constitutive: Optional[ConstitutiveModel] = None,
        n_cells: int = 16,
        gravity: Optional[tuple] = None,
        bc_type: str = "fixed",
        meta_json: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.device = device
        self.dtype = dtype
        self.decoders = decoders
        self.masks = masks
        self.n_charts = len(decoders)
        self.constitutive = constitutive or NeoHookeanModel()

        if isinstance(atlas_data, str):
            atlas_data = dict(np.load(atlas_data, allow_pickle=True))
        self.atlas_data = atlas_data

        seeds = atlas_data["seed_points"]
        t1s = atlas_data["frame_t1"]
        t2s = atlas_data["frame_t2"]
        ns = atlas_data["frame_n"]
        radii = atlas_data["support_radii"]

        self.seeds_t = torch.tensor(seeds, device=device, dtype=dtype)
        self.t1_t = torch.tensor(t1s, device=device, dtype=dtype)
        self.t2_t = torch.tensor(t2s, device=device, dtype=dtype)
        self.nvec_t = torch.tensor(ns, device=device, dtype=dtype)
        self.support_r_t = torch.tensor(radii, device=device, dtype=dtype)

        # Build overlap neighbor graph
        membership = atlas_data["membership"]
        self.neighbors = self._build_neighbors(membership)

        membership_np = np.array(membership, dtype=np.float32)
        self.color_groups = choose_color_groups(meta_json, self.n_charts, membership_np)

        # Build per-chart MPM solvers
        self.solvers: List[ChartMPMSolver] = []
        for i in range(self.n_charts):
            extent = float(self.support_r_t[i].item()) * 1.5
            solver = ChartMPMSolver(
                n_cells=n_cells,
                extent=extent,
                constitutive=constitutive,
                gravity=gravity,
                bc_type=bc_type,
                device=device,
                dtype=dtype,
            )
            self.solvers.append(solver)

        # Per-chart particle clouds (initialized separately)
        self.particles: List[Optional[MaterialPointCloud]] = [None] * self.n_charts

        self.time = 0.0
        self.step_count = 0

        # Topology monitoring hooks (set externally for Phase 3)
        self._sdf_grid_fn = None      # callable() -> (N,N,N) grid
        self._chart_spawner = None    # ChartSpawner instance
        self._bbox_min = None         # (3,) domain lower bound
        self._bbox_max = None         # (3,) domain upper bound

    def setup_topology_monitoring(self, sdf_grid_fn, chart_spawner, bbox_min, bbox_max):
        """Configure live topology monitoring for dynamic chart spawning.

        Parameters
        ----------
        sdf_grid_fn : callable
            No-argument function returning (N,N,N) SDF grid at current state.
        chart_spawner : ChartSpawner
            Spawner that converts TopologyEvents to SpawnedChartPairs.
        bbox_min, bbox_max : array-like (3,)
            Domain bounding box for feature localization.
        """
        self._sdf_grid_fn = sdf_grid_fn
        self._chart_spawner = chart_spawner
        self._bbox_min = np.asarray(bbox_min, dtype=np.float64)
        self._bbox_max = np.asarray(bbox_max, dtype=np.float64)

    def _build_neighbors(self, membership) -> List[List[int]]:
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

    def initialize_particles(
        self,
        n_per_axis: int = 4,
        density: float = 1000.0,
        velocity_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize uniform particle distributions in each chart.

        Parameters
        ----------
        n_per_axis : int
            Particles per axis per chart.
        density : float
            Material density.
        velocity_fn : callable, optional
            v(x_phys) -> (N, 3) initial velocity field.
        """
        for i in range(self.n_charts):
            extent = self.solvers[i].grid.extent * 0.8  # stay inside grid
            cloud = MaterialPointCloud.create_uniform(
                n_per_axis=n_per_axis,
                extent=extent,
                density=density,
                device=self.device,
                dtype=self.dtype,
            )

            if velocity_fn is not None:
                x_phys = cloud.push_to_physical(
                    self.decoders[i], self.seeds_t[i], self.t1_t[i],
                    self.t2_t[i], self.nvec_t[i], self.support_r_t[i],
                )
                cloud.v = velocity_fn(x_phys)

            self.particles[i] = cloud

    def step(self, dt: float) -> Dict[str, float]:
        """Advance one time step across all charts.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        diagnostics : dict
            Aggregate diagnostics across all charts.
        """
        total_ke = 0.0
        max_v = 0.0
        total_particles = 0

        # Step each chart independently
        for i in range(self.n_charts):
            if self.particles[i] is None or self.particles[i].n_particles == 0:
                continue
            diag = self.solvers[i].step(self.particles[i], dt)
            total_ke += diag["kinetic_energy"]
            max_v = max(max_v, diag["max_velocity"])
            total_particles += self.particles[i].n_particles

        # Exchange boundary data between overlapping charts
        self._exchange_boundary_velocities()

        # Transfer escaped particles
        self._transfer_particles()

        self.time += dt
        self.step_count += 1

        return {
            "kinetic_energy": total_ke,
            "max_velocity": max_v,
            "total_particles": total_particles,
            "time": self.time,
            "step": self.step_count,
        }

    def _exchange_boundary_velocities(self) -> None:
        """Exchange boundary grid velocities between overlapping charts.

        For grid nodes near the chart boundary, blend in velocity from
        neighboring charts using mask-based PoU weights.
        """
        for i in range(self.n_charts):
            grid_i = self.solvers[i].grid
            if not grid_i.boundary_mask.any():
                continue

            boundary_idx = torch.where(grid_i.boundary_mask)[0]
            if len(boundary_idx) == 0:
                continue

            xi_boundary = grid_i.positions[boundary_idx]  # (N_bnd, 3)

            # Map to physical space
            with torch.no_grad():
                x_phys = self.decoders[i](
                    xi_boundary, seed=self.seeds_t[i], t1=self.t1_t[i],
                    t2=self.t2_t[i], n=self.nvec_t[i],
                    chart_scale=self.support_r_t[i],
                )

            # Gather velocity from neighbors
            for j in self.neighbors[i]:
                if self.particles[j] is None or self.particles[j].n_particles == 0:
                    continue

                xi_j = local_coords(
                    x_phys, self.seeds_t[j], self.t1_t[j],
                    self.t2_t[j], self.nvec_t[j],
                )

                # Check which boundary nodes fall within chart j's grid
                extent_j = self.solvers[j].grid.extent
                in_chart_j = torch.all(torch.abs(xi_j) <= extent_j, dim=1)

                if not in_chart_j.any():
                    continue

                # Blend: average velocity from chart j's grid at those nodes
                # (simplified — full implementation would interpolate from j's grid)
                with torch.no_grad():
                    logit_j = self.masks[j](xi_j[in_chart_j], chart_scale=self.support_r_t[j])
                    weight = torch.sigmoid(logit_j)

                # Apply weighted blend of chart j's grid velocity
                # This is a simplified coupling — enough for initial implementation
                v_j = self.solvers[j].grid.velocity
                # Nearest-node interpolation from chart j
                h_j = self.solvers[j].grid.h
                npa_j = self.solvers[j].grid.n_nodes_per_axis
                grid_idx_j = torch.floor((xi_j[in_chart_j] + extent_j) / h_j).long()
                grid_idx_j = torch.clamp(grid_idx_j, 0, npa_j - 1)
                flat_j = (grid_idx_j[:, 0] * npa_j * npa_j +
                          grid_idx_j[:, 1] * npa_j + grid_idx_j[:, 2])

                valid_flat = flat_j < v_j.shape[0]
                if valid_flat.any():
                    v_from_j = v_j[flat_j[valid_flat]]
                    # Blend into chart i's boundary velocity
                    blend_idx = boundary_idx[in_chart_j][valid_flat]
                    w = weight[valid_flat].unsqueeze(1) * 0.5  # 50% blend
                    grid_i.velocity[blend_idx] = (
                        (1.0 - w) * grid_i.velocity[blend_idx] + w * v_from_j
                    )

    def _transfer_particles(self) -> None:
        """Transfer particles that have left their chart domain to neighbors.

        A particle is considered escaped if its position exceeds the grid extent.
        It is transferred to the neighboring chart with highest mask logit.
        """
        for i in range(self.n_charts):
            if self.particles[i] is None or self.particles[i].n_particles == 0:
                continue

            cloud = self.particles[i]
            extent = self.solvers[i].grid.extent
            escaped = torch.any(torch.abs(cloud.xi) > extent * 0.95, dim=1)

            if not escaped.any():
                continue

            n_escaped = escaped.sum().item()
            escaped_idx = torch.where(escaped)[0]

            # Map escaped particles to physical space
            with torch.no_grad():
                xi_esc = cloud.xi[escaped_idx]
                x_phys = self.decoders[i](
                    xi_esc, seed=self.seeds_t[i], t1=self.t1_t[i],
                    t2=self.t2_t[i], n=self.nvec_t[i],
                    chart_scale=self.support_r_t[i],
                )

            # Find best neighbor for each escaped particle
            best_logit = torch.full((n_escaped,), -1e10, device=self.device, dtype=self.dtype)
            best_chart = torch.full((n_escaped,), -1, device=self.device, dtype=torch.long)

            for j in self.neighbors[i]:
                xi_j = local_coords(x_phys, self.seeds_t[j], self.t1_t[j],
                                    self.t2_t[j], self.nvec_t[j])
                extent_j = self.solvers[j].grid.extent
                in_j = torch.all(torch.abs(xi_j) <= extent_j * 0.9, dim=1)

                if not in_j.any():
                    continue

                with torch.no_grad():
                    logit = self.masks[j](xi_j, chart_scale=self.support_r_t[j])

                better = in_j & (logit > best_logit)
                best_logit[better] = logit[better]
                best_chart[better] = j

            # Transfer particles to their best neighbor
            transferred = best_chart >= 0
            if not transferred.any():
                continue

            for j in self.neighbors[i]:
                to_j = (best_chart == j) & transferred
                if not to_j.any():
                    continue

                src_idx = escaped_idx[to_j]
                xi_j = local_coords(
                    x_phys[to_j], self.seeds_t[j], self.t1_t[j],
                    self.t2_t[j], self.nvec_t[j],
                )

                # Add particles to chart j
                if self.particles[j] is None:
                    self.particles[j] = MaterialPointCloud(
                        positions=xi_j,
                        velocities=cloud.v[src_idx],
                        volumes=cloud.V0[src_idx],
                        masses=cloud.mass[src_idx],
                        device=self.device, dtype=self.dtype,
                    )
                    self.particles[j].F = cloud.F[src_idx].clone()
                    self.particles[j].stress = cloud.stress[src_idx].clone()
                else:
                    pj = self.particles[j]
                    pj.xi = torch.cat([pj.xi, xi_j], dim=0)
                    pj.v = torch.cat([pj.v, cloud.v[src_idx]], dim=0)
                    pj.V0 = torch.cat([pj.V0, cloud.V0[src_idx]], dim=0)
                    pj.mass = torch.cat([pj.mass, cloud.mass[src_idx]], dim=0)
                    pj.F = torch.cat([pj.F, cloud.F[src_idx]], dim=0)
                    pj.stress = torch.cat([pj.stress, cloud.stress[src_idx]], dim=0)

            # Remove transferred particles from chart i
            keep = ~escaped | ~transferred
            if keep.sum() < cloud.n_particles:
                keep_idx = torch.where(keep)[0]
                cloud.xi = cloud.xi[keep_idx]
                cloud.v = cloud.v[keep_idx]
                cloud.V0 = cloud.V0[keep_idx]
                cloud.mass = cloud.mass[keep_idx]
                cloud.F = cloud.F[keep_idx]
                cloud.stress = cloud.stress[keep_idx]

    def run(
        self,
        dt: float,
        n_steps: int,
        monitor=None,
        monitor_interval: int = 10,
        output_interval: int = 10,
    ) -> List[Dict[str, float]]:
        """Run multi-chart MPM simulation.

        Parameters
        ----------
        dt : float
            Time step.
        n_steps : int
            Number of steps.
        monitor : TopologyMonitor, optional
            Topology monitor for dynamic chart spawning.
        monitor_interval : int
            Steps between topology checks.
        output_interval : int
            Steps between diagnostic output.

        Returns
        -------
        history : list of dict
            Diagnostics at each step.
        """
        history = []
        for step_i in range(n_steps):
            diag = self.step(dt)
            history.append(diag)

            if (step_i + 1) % output_interval == 0:
                print(
                    f"  [SchwarzMPM] step {diag['step']:5d} | "
                    f"t={diag['time']:.4e} | "
                    f"KE={diag['kinetic_energy']:.4e} | "
                    f"v_max={diag['max_velocity']:.4e} | "
                    f"particles={diag['total_particles']}"
                )

            # Topology monitoring
            if monitor is not None and (step_i + 1) % monitor_interval == 0:
                events = self._check_topology(monitor, diag["step"])
                if events:
                    print(f"  [SchwarzMPM] {len(events)} topology event(s) at step {diag['step']}")

        return history

    def _check_topology(self, monitor, load_step: int) -> list:
        """Check for topology changes and spawn new charts if needed.

        Evaluates the SDF on a grid, runs the TopologyMonitor, and for
        each detected event, spawns new chart pairs via ChartSpawner.

        Parameters
        ----------
        monitor : TopologyMonitor
            Active topology monitor.
        load_step : int
            Current simulation step.

        Returns
        -------
        events : list of TopologyEvent
        """
        if self._sdf_grid_fn is None:
            return []

        from atlas.topo.filtration import clip_to_interior

        grid_vals = self._sdf_grid_fn()
        grid_clipped = clip_to_interior(grid_vals)
        events = monitor.update(grid_clipped, load_step=load_step)

        if events and self._chart_spawner is not None:
            existing_seeds = self.seeds_t.cpu().numpy()
            existing_frames = np.stack([
                np.stack([self.t1_t[i].cpu().numpy(),
                          self.t2_t[i].cpu().numpy(),
                          self.nvec_t[i].cpu().numpy()])
                for i in range(self.n_charts)
            ])

            for event in events:
                pair = self._chart_spawner.spawn_from_event(
                    event, existing_seeds, existing_frames,
                    grid_clipped, self._bbox_min, self._bbox_max,
                )
                self.add_charts([pair])

        return events

    def add_charts(self, pairs) -> int:
        """Add spawned chart pairs to the MPM solver.

        Creates new ChartMPMSolver instances with warm-started decoders
        and empty particle clouds for each side of each SpawnedChartPair.

        Parameters
        ----------
        pairs : list of SpawnedChartPair
            From ChartSpawner.spawn_from_event().

        Returns
        -------
        n_added : int
            Number of new charts added.
        """
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

                # Warm-start decoder from parent
                parent_dec = self.decoders[pair.parent_chart]
                new_dec = type(parent_dec)(
                    width=parent_dec.net.out.in_features,
                    depth=len(parent_dec.net.hidden),
                ).to(device=self.device, dtype=self.dtype)
                new_dec.warm_start_from(parent_dec)

                parent_mask = self.masks[pair.parent_chart]
                new_mask = type(parent_mask)(
                    width=parent_mask.net.out.in_features,
                    depth=len(parent_mask.net.hidden),
                ).to(device=self.device, dtype=self.dtype)

                self.decoders.append(new_dec)
                self.masks.append(new_mask)
                self.seeds_t = torch.cat([self.seeds_t, seed_t.unsqueeze(0)])
                self.t1_t = torch.cat([self.t1_t, t1_t.unsqueeze(0)])
                self.t2_t = torch.cat([self.t2_t, t2_t.unsqueeze(0)])
                self.nvec_t = torch.cat([self.nvec_t, n_t.unsqueeze(0)])
                self.support_r_t = torch.cat([self.support_r_t, r_t.unsqueeze(0)])

                extent = float(r_t.item()) * 1.5
                new_solver = ChartMPMSolver(
                    n_cells=self.solvers[0].grid.n_cells if self.solvers else 16,
                    extent=extent,
                    constitutive=self.constitutive,
                    bc_type="fixed",
                    device=self.device,
                    dtype=self.dtype,
                )
                self.solvers.append(new_solver)
                self.particles.append(None)  # empty cloud, filled by transfer
                n_added += 1

        self.n_charts += n_added

        # Rebuild neighbor graph
        self.neighbors = []
        for i in range(self.n_charts):
            nbrs = []
            si = self.seeds_t[i]
            ri = self.support_r_t[i].item()
            for j in range(self.n_charts):
                if i == j:
                    continue
                dist = torch.linalg.norm(si - self.seeds_t[j]).item()
                if dist < 2.0 * (ri + self.support_r_t[j].item()):
                    nbrs.append(j)
            self.neighbors.append(nbrs)

        membership_fake = np.zeros((1, self.n_charts), dtype=np.float32)
        self.color_groups = choose_color_groups(None, self.n_charts, membership_fake)

        return n_added
