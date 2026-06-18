"""
atlas/topo/chart_spawn.py
Automatic chart spawning when a topology change is detected.

When the TopologyMonitor fires a TopologyEvent (e.g., a new H_1 feature
indicating crack propagation), this module determines the seeding
parameters for the new chart pair and integrates them into the atlas.

The spawned charts differ from standard atlas charts in one key respect:
the Schwarz transmission condition on their shared artificial interface
is replaced by a prescribed displacement-jump condition encoding the
crack opening displacement (COD).  This is the atlas analog of XFEM
enrichment, but derived automatically from topology.

Chart spawning protocol
-----------------------
1. Localize the new topological feature in physical space using the
   SDF gradient field (features are born near SDF saddle points).
2. Seed two overlapping charts (chart_plus, chart_minus) on opposite
   sides of the detected feature's zero-level crossing.
3. Initialize decoder weights from the nearby existing chart with the
   smallest transition error to the new seed location.
4. Register the new chart pair in the atlas overlap graph G_A with a
   'crack' edge type, which the Schwarz solver treats differently from
   'continuity' edges.
5. Run quality gate checks on the new charts before activating them.

Reference: Fries & Belytschko (2010), "The extended/generalized finite
           element method: An overview..." IJNME 84:253-304.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from atlas.topo.monitor import TopologyEvent


@dataclass
class SpawnedChartPair:
    """
    Parameters for a pair of charts spawned across a topology boundary.

    Attributes
    ----------
    seed_plus      : [3] seed location on the +side of the feature
    seed_minus     : [3] seed location on the -side of the feature
    frame_plus     : [3,3] orthonormal frame at seed_plus
    frame_minus    : [3,3] orthonormal frame at seed_minus
    radius         : support radius for both charts
    parent_chart   : index of the existing chart whose decoder is warm-started
    edge_type      : 'crack' or 'interface'
    activation_step: load step at which this pair becomes active
    """
    seed_plus: np.ndarray
    seed_minus: np.ndarray
    frame_plus: np.ndarray
    frame_minus: np.ndarray
    radius: float
    parent_chart: int
    edge_type: str = "crack"
    activation_step: int = 0


class ChartSpawner:
    """
    Converts topology events into concrete chart spawn requests.

    Parameters
    ----------
    default_radius : float
        Support radius for newly spawned charts. Should be ~0.3 * (domain diameter)
        for Schwarz convergence — consistent with paper's seed placement strategy.
    sdf_net : callable or None
        Neural SDF used for feature localization via gradient ascent.
        If None, localization falls back to the event's birth_value position.
    verbose : bool
    """

    def __init__(
        self,
        default_radius: float = 0.3,
        sdf_net=None,
        verbose: bool = True,
    ) -> None:
        self.default_radius = default_radius
        self.sdf_net = sdf_net
        self.verbose = verbose
        self._spawn_log: List[SpawnedChartPair] = []

    def spawn_from_event(
        self,
        event: "TopologyEvent",
        existing_seeds: np.ndarray,
        existing_frames: np.ndarray,
        grid_vals: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> SpawnedChartPair:
        """
        Compute spawn parameters for a new chart pair from a TopologyEvent.

        Parameters
        ----------
        event           : TopologyEvent from TopologyMonitor.update()
        existing_seeds  : (M, 3) array of existing chart seed positions
        existing_frames : (M, 3, 3) array of existing chart frames
        grid_vals       : (N, N, N) SDF values at current step
        bbox_min, bbox_max : bounding box corners [3]

        Returns
        -------
        pair : SpawnedChartPair with all parameters needed by atlas.add_charts()
        """
        # Step 1: Localize feature in physical space
        center, normal = self._localize_feature(event, grid_vals, bbox_min, bbox_max)

        # Step 2: Place chart seeds on +/- sides of the feature
        r = self.default_radius
        seed_plus = center + 0.5 * r * normal
        seed_minus = center - 0.5 * r * normal

        # Step 3: Build orthonormal frames from the feature normal
        frame_plus = self._frame_from_normal(normal)
        frame_minus = self._frame_from_normal(-normal)

        # Step 4: Find nearest existing chart for warm-start
        dists = np.linalg.norm(existing_seeds - center[None, :], axis=-1)
        parent_chart = int(np.argmin(dists))

        pair = SpawnedChartPair(
            seed_plus=seed_plus,
            seed_minus=seed_minus,
            frame_plus=frame_plus,
            frame_minus=frame_minus,
            radius=r,
            parent_chart=parent_chart,
            edge_type="crack",
            activation_step=event.load_step,
        )
        self._spawn_log.append(pair)

        if self.verbose:
            print(
                f"[ChartSpawner] spawning pair at step={event.load_step} "
                f"center={center}  normal={normal}  parent_chart={parent_chart}"
            )

        return pair

    def _localize_feature(
        self,
        event: "TopologyEvent",
        grid_vals: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Localize a topological feature to a physical position and normal.

        The birth value of a new H_1 pair equals the SDF value at the
        saddle point that created the new loop.  We find grid cells with
        SDF value ~ birth_value and use the local SDF gradient as the
        normal to the feature.

        Returns: (center [3], normal [3])
        """
        N = grid_vals.shape[0]
        resolution = np.array(grid_vals.shape)

        # Find grid cells near the birth value
        birth = event.birth_value
        tolerance = 0.05 * (float(grid_vals.max()) - float(grid_vals.min()))
        mask = np.abs(grid_vals - birth) < tolerance

        if mask.any():
            idx = np.argwhere(mask)
            # Convert grid index to physical coordinate
            frac = idx.astype(float) / (resolution[None, :] - 1)
            phys = bbox_min[None, :] + frac * (bbox_max - bbox_min)[None, :]
            center = phys.mean(axis=0)
        else:
            # Fallback: use domain centroid
            center = 0.5 * (bbox_min + bbox_max)

        # Estimate normal from SDF gradient via finite differences
        normal = self._sdf_gradient_at(grid_vals, center, bbox_min, bbox_max)
        nrm = np.linalg.norm(normal)
        if nrm > 1e-8:
            normal /= nrm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        return center, normal

    def _sdf_gradient_at(
        self,
        grid_vals: np.ndarray,
        point: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        eps: float = 0.01,
    ) -> np.ndarray:
        """Estimate SDF gradient at a point using central finite differences on the grid."""
        N = grid_vals.shape[0]
        span = bbox_max - bbox_min

        def to_idx(x: np.ndarray) -> np.ndarray:
            frac = (x - bbox_min) / np.maximum(span, 1e-8)
            return np.clip(frac * (N - 1), 0, N - 1).astype(int)

        grad = np.zeros(3)
        for d in range(3):
            xp = point.copy(); xp[d] += eps * float(span[d])
            xm = point.copy(); xm[d] -= eps * float(span[d])
            ip, im = to_idx(xp), to_idx(xm)
            fp = grid_vals[ip[0], ip[1], ip[2]]
            fm = grid_vals[im[0], im[1], im[2]]
            grad[d] = (fp - fm) / (2.0 * eps * float(span[d]))

        return grad

    @staticmethod
    def _frame_from_normal(normal: np.ndarray) -> np.ndarray:
        """
        Build a right-handed orthonormal frame [t1, t2, n] from a normal vector.
        Uses the Gram-Schmidt process with a stable reference vector.
        """
        n = normal / np.linalg.norm(normal)
        # Choose reference vector not parallel to n
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        t1 = ref - np.dot(ref, n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return np.stack([t1, t2, n], axis=0)  # (3, 3), rows are basis vectors

    @property
    def spawn_log(self) -> List[SpawnedChartPair]:
        return list(self._spawn_log)
