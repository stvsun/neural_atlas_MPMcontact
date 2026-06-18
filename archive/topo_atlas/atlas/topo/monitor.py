"""
atlas/topo/monitor.py
Dynamic topology monitor for atlas-based simulations with evolving geometry.

During simulations involving crack propagation, phase-field fracture, or
solidification, the domain topology changes mid-simulation.  A propagating
crack introduces a new H_1 generator in the SDF of the deformed domain,
detectable as a new long-lived birth-death pair in Dgm_1 that was absent
at the previous load step.

This module monitors the persistence diagram at each load step and fires
TopologyEvent objects when changes are detected, enabling the atlas to
autonomously spawn new charts straddling the topological feature.

Coupling with atlas solver
--------------------------
At each load step:
    events = monitor.update(grid_vals, load_step)
    for event in events:
        new_charts = spawner.spawn(event, atlas)
        atlas.add_charts(new_charts)
        schwarz_solver.rebuild_overlap_graph(atlas)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TopologyEvent:
    """
    A detected topology change at a specific load step.

    Attributes
    ----------
    load_step     : simulation step index at which the event was detected
    event_type    : 'new_H1' | 'new_H2' | 'feature_died'
    dimension     : homological dimension of the changed feature
    birth_value   : filtration parameter at which the new feature was born
    lifetime      : persistence |d - b| of the new pair
    localization  : [3] approximate physical coordinates of the feature
                    (from SDF gradient localization; None if not computed)
    bottleneck_change : bottleneck distance change that triggered this event
    """
    load_step: int
    event_type: str
    dimension: int
    birth_value: float
    lifetime: float
    localization: Optional[np.ndarray] = None
    bottleneck_change: float = 0.0


@dataclass
class MonitorState:
    """Persistent state between monitor.update() calls."""
    step: int = -1
    diagrams: Dict[int, list] = field(default_factory=dict)
    betti: Dict[int, int] = field(default_factory=dict)
    event_count: int = 0


class TopologyMonitor:
    """
    Monitor topology changes in a neural SDF across simulation load steps.

    Parameters
    ----------
    lifetime_threshold : float
        Minimum |d - b| for a persistence pair to be considered a real
        topological feature (not SDF approximation noise).
        Rule of thumb: 0.05 * (domain diameter). Default 0.05.
    bottleneck_threshold : float
        Minimum bottleneck distance change between consecutive diagrams
        to fire a TopologyEvent. Default 0.02.
    monitor_dimensions : tuple
        Which homological dimensions to watch.
        (1,) watches for cracks/tunnels only.
        (1, 2) also watches for enclosed voids.
    relative_threshold : bool
        If True, lifetime_threshold is relative to the SDF value range.
    verbose : bool
        Print event messages to stdout.

    Verification contract
    ---------------------
    On a fixed domain (no crack, constant SDF across steps):
        monitor.update() should return [] at every step after the first.
    On a domain where a crack propagates and severs a cross-section:
        exactly one new_H1 event should fire within 1-2 steps of the
        crack reaching topological significance.
    """

    def __init__(
        self,
        lifetime_threshold: float = 0.05,
        bottleneck_threshold: float = 0.02,
        monitor_dimensions: Tuple[int, ...] = (1,),
        relative_threshold: bool = True,
        verbose: bool = True,
    ) -> None:
        self.lifetime_threshold = lifetime_threshold
        self.bottleneck_threshold = bottleneck_threshold
        self.monitor_dimensions = monitor_dimensions
        self.relative_threshold = relative_threshold
        self.verbose = verbose
        self._state = MonitorState()
        self._history: List[TopologyEvent] = []

    def update(
        self,
        grid_vals: np.ndarray,
        load_step: int,
    ) -> List[TopologyEvent]:
        """
        Process new SDF values and return any detected topology events.

        Parameters
        ----------
        grid_vals : (N, N, N) float array
                    SDF values at current load step, clipped at 0 (interior)
        load_step : current simulation step index

        Returns
        -------
        events : list of TopologyEvent (empty if no topology change)

        Notes
        -----
        The first call establishes a baseline; it returns [] even if the
        baseline diagram is non-trivial.  Subsequent calls detect *changes*
        relative to the previous step.
        """
        from atlas.topo.persistence import (
            compute_persistence_diagrams,
            filter_by_lifetime,
            betti_numbers_at,
            bottleneck_distance,
        )

        f_min, f_max = float(grid_vals.min()), float(grid_vals.max())

        # Compute and filter persistence diagrams
        raw = compute_persistence_diagrams(grid_vals, max_dimension=max(self.monitor_dimensions))
        filtered = filter_by_lifetime(
            raw,
            threshold=self.lifetime_threshold,
            relative=self.relative_threshold,
            filtration_range=(f_min, f_max),
        )
        betti = betti_numbers_at(filtered, t=-1e-6)

        events: List[TopologyEvent] = []

        if self._state.step >= 0:
            for dim in self.monitor_dimensions:
                prev_dgm = self._state.diagrams.get(dim, [])
                curr_dgm = filtered.get(dim, [])

                d_b = bottleneck_distance(prev_dgm, curr_dgm)
                if d_b < self.bottleneck_threshold:
                    continue  # no significant change

                new_pairs = self._find_new_pairs(prev_dgm, curr_dgm)
                for (b, d) in new_pairs:
                    lifetime = (d - b) if d != np.inf else np.inf
                    ev = TopologyEvent(
                        load_step=load_step,
                        event_type="new_H1" if dim == 1 else "new_H2",
                        dimension=dim,
                        birth_value=b,
                        lifetime=lifetime,
                        bottleneck_change=d_b,
                    )
                    events.append(ev)
                    self._history.append(ev)
                    self._state.event_count += 1

                    if self.verbose:
                        print(
                            f"[TopologyMonitor] step={load_step}  "
                            f"EVENT {ev.event_type}  dim={dim}  "
                            f"birth={b:.4f}  lifetime={lifetime:.4f}  "
                            f"d_bottleneck={d_b:.4f}"
                        )

        # Update state
        self._state.step = load_step
        self._state.diagrams = filtered
        self._state.betti = betti

        return events

    def _find_new_pairs(
        self,
        prev: list,
        curr: list,
    ) -> list:
        """
        Find persistence pairs in curr that have no close match in prev.

        Uses greedy matching by minimum L_inf distance between pairs,
        projecting unmatched pairs to the diagonal.

        Parameters
        ----------
        prev, curr : lists of (birth, death) persistence pairs

        Returns
        -------
        new_pairs : pairs in curr not matched to any pair in prev
        """
        if not prev:
            return list(curr)

        used = [False] * len(prev)
        new_pairs = []

        for (bc, dc) in curr:
            best_dist = self.bottleneck_threshold * 2.0
            best_idx = -1
            for i, (bp, dp) in enumerate(prev):
                if used[i]:
                    continue
                # L_inf distance; treat inf-death pairs separately
                if bc == np.inf and bp == np.inf:
                    dist = 0.0
                elif bc == np.inf or bp == np.inf:
                    dist = np.inf
                else:
                    dist = max(abs(bc - bp), abs(dc - dp) if dc != np.inf and dp != np.inf else np.inf)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                used[best_idx] = True
            else:
                new_pairs.append((bc, dc))

        return new_pairs

    @property
    def current_betti(self) -> Dict[int, int]:
        """Betti numbers at the most recently processed load step."""
        return dict(self._state.betti)

    @property
    def event_history(self) -> List[TopologyEvent]:
        """Immutable list of all events detected so far."""
        return list(self._history)

    @property
    def total_events(self) -> int:
        return self._state.event_count

    def reset(self) -> None:
        """Reset monitor state (e.g., when starting a new simulation run)."""
        self._state = MonitorState()
        self._history.clear()

    def summary(self) -> str:
        """Return a formatted summary of monitoring history."""
        lines = [
            f"TopologyMonitor summary",
            f"  Steps processed     : {self._state.step + 1}",
            f"  Total events        : {self._state.event_count}",
            f"  Current Betti       : {self._state.betti}",
        ]
        if self._history:
            lines.append("  Event log:")
            for ev in self._history:
                lines.append(
                    f"    step={ev.load_step:4d}  {ev.event_type}  "
                    f"dim={ev.dimension}  lifetime={ev.lifetime:.4f}"
                )
        return "\n".join(lines)
