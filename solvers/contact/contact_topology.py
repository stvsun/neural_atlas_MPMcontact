"""Topology-aware contact detection via persistent homology.

For two (or more) bodies with neural SDFs ``phi_i``, the **combined
SDF** is the pointwise minimum

    phi_AB(x) = min_i phi_i(x)

which describes the union of all the bodies' interior regions.  Running
persistent homology on the sublevel set ``{x : phi_AB < 0}`` gives a
direct geometry-agnostic readout of contact state:

* ``beta_0 == 2`` (two connected components) — bodies separated
* ``beta_0 == 1`` — bodies touching (first contact)
* ``beta_2`` increases by one — enclosure (one body inside another)

This module wraps the existing topology utilities at ``atlas/topo/`` in
a thin contact-specific monitor.  It does **not** replace or modify
``atlas/topo/monitor.py:TopologyMonitor``; the two are complementary —
``TopologyMonitor`` watches H_1 on a single body for crack detection,
while ``ContactTopologyMonitor`` watches H_0 on the combined field for
inter-body contact events.

Reference: ``contact_atlas/03_mathematical_theory.md`` Section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from solvers.contact.contact_pair import ContactBody
from atlas.topo.filtration import clip_to_interior, filtration_value_range
from atlas.topo.persistence import (
    compute_persistence_diagrams,
    filter_by_lifetime,
    betti_numbers_at,
)


# ── Dataclass ────────────────────────────────────────────────────────


@dataclass
class ContactTopologyEvent:
    """A detected contact state change from combined-SDF persistence.

    Attributes
    ----------
    event_type : str
        ``"first_contact"``, ``"separation"``, or ``"enclosure"``.
    load_step : int
        Simulation step index at which the event was detected.
    beta0_before, beta0_after : int
        Number of connected components of the combined sublevel set
        before and after this update.
    beta2_before, beta2_after : int
        Number of 3-dimensional voids (relevant only when
        ``track_enclosure=True``).
    location : np.ndarray or None
        (3,) approximate physical-space location of the contact region,
        or ``None`` for separation events (no unique location).
    """

    event_type: str
    load_step: int
    beta0_before: int
    beta0_after: int
    beta2_before: int = 0
    beta2_after: int = 0
    location: Optional[np.ndarray] = None


# ── Combined SDF grid sampling ───────────────────────────────────────


def _infer_sdf_device_dtype(body: ContactBody):
    """Detect the device and dtype of a body's SDF network.

    Uses the first trainable parameter (or buffer) of the module.
    Falls back to ``(cpu, float64)`` for SDF modules with no parameters
    (e.g., purely analytic SDFs built from ``register_buffer``).
    """
    try:
        p = next(body.sdf_net.parameters())
        return p.device, p.dtype
    except StopIteration:
        pass
    try:
        b = next(body.sdf_net.buffers())
        return b.device, b.dtype
    except StopIteration:
        pass
    return torch.device("cpu"), torch.float64


def combined_sdf_grid(
    bodies: List[ContactBody],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int = 16,
) -> np.ndarray:
    """Evaluate ``phi_AB(x) = min_i phi_i(x)`` on a regular cubical grid.

    The grid coordinates are built on the device and dtype of the
    **first** body's SDF.  If subsequent bodies live on a different
    device or dtype, their output is moved / cast to match before the
    pointwise minimum, so mixed-precision and multi-device setups are
    tolerated (though not recommended for performance).

    Parameters
    ----------
    bodies : list of ContactBody
        Bodies whose SDFs are combined with a pointwise minimum.
    bbox_min, bbox_max : array-like, shape (3,)
        Lower and upper corners of the monitoring bounding box.  The
        box must cover the union of the bodies' interiors.
    resolution : int
        Number of grid points per axis (total ``resolution ** 3``).

    Returns
    -------
    grid_vals : np.ndarray
        ``(resolution, resolution, resolution)`` float32 array of the
        combined SDF values on the grid, compatible with
        ``compute_persistence_diagrams`` and ``clip_to_interior``.
    """
    if not bodies:
        raise ValueError("combined_sdf_grid requires at least one body")

    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    device, dtype = _infer_sdf_device_dtype(bodies[0])

    # Build grid coordinates on the inferred device/dtype.  Use
    # indexing="ij" to silence the deprecation warning and guarantee
    # predictable axis ordering (first index = x, second = y, etc.).
    lin = [
        torch.linspace(
            float(bbox_min[i]), float(bbox_max[i]), resolution,
            dtype=dtype, device=device,
        )
        for i in range(3)
    ]
    try:
        gx, gy, gz = torch.meshgrid(*lin, indexing="ij")
    except TypeError:
        # Older torch (<1.10) doesn't support indexing=
        gx, gy, gz = torch.meshgrid(*lin)
    coords = torch.stack(
        [gx.flatten(), gy.flatten(), gz.flatten()], dim=1,
    )  # (N^3, 3)

    # Evaluate every body's SDF and take the pointwise minimum
    min_vals: Optional[torch.Tensor] = None
    with torch.no_grad():
        for body in bodies:
            body.sdf_net.eval()
            b_device, b_dtype = _infer_sdf_device_dtype(body)
            # Cast/move coords if this body lives on a different
            # device or dtype than the first.
            body_coords = coords
            if b_device != device or b_dtype != dtype:
                body_coords = coords.to(device=b_device, dtype=b_dtype)
            phi = body.sdf_net(body_coords)            # (N^3,) or (N^3, 1)
            if phi.dim() > 1:
                phi = phi.squeeze(-1)
            # Harmonize back to the reference dtype/device for min().
            phi = phi.to(device=device, dtype=dtype)
            min_vals = (
                phi if min_vals is None else torch.minimum(min_vals, phi)
            )

    grid = min_vals.reshape(resolution, resolution, resolution)
    return grid.cpu().numpy().astype(np.float32)


# ── Monitor ──────────────────────────────────────────────────────────


class ContactTopologyMonitor:
    """Detect inter-body contact events via persistent homology.

    At each ``update(load_step)`` call the monitor:

    1. Evaluates the combined SDF ``phi_AB = min_i phi_i`` on a cubical
       grid inside the bounding box.
    2. Runs the existing persistent-homology pipeline
       (``compute_persistence_diagrams`` + ``filter_by_lifetime`` +
       ``betti_numbers_at``) to extract Betti numbers beta_0 (and
       optionally beta_2).
    3. Compares with the previous update's Betti numbers and emits a
       :class:`ContactTopologyEvent` for each transition:

       * ``Delta beta_0 = -1`` -> ``"first_contact"``
       * ``Delta beta_0 = +1`` -> ``"separation"``
       * ``Delta beta_2 = +1`` -> ``"enclosure"`` (when track_enclosure)

    The **first** ``update`` call establishes the baseline and returns
    an empty list even if the configuration is non-trivial.

    Parameters
    ----------
    bodies : list of ContactBody
        Bodies to watch.  All participating SDFs are combined.
    bbox_min, bbox_max : array-like (3,)
        Monitoring bounding box.  Must cover the union of all bodies.
    resolution : int
        Grid points per axis.  16 is a reasonable default (coarse enough that
        the persistent-homology overhead stays ~1% for production monitoring).
    lifetime_threshold : float
        Relative lifetime threshold passed to ``filter_by_lifetime``.
        Pairs shorter than ``threshold * range`` are dropped as noise.
    track_enclosure : bool
        If True, also compute and watch beta_2 for enclosure events.
    """

    def __init__(
        self,
        bodies: List[ContactBody],
        bbox_min,
        bbox_max,
        resolution: int = 16,
        lifetime_threshold: float = 0.02,
        track_enclosure: bool = False,
    ):
        self.bodies = list(bodies)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float64)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64)
        self.resolution = resolution
        self.lifetime_threshold = lifetime_threshold
        self.track_enclosure = track_enclosure

        self.prev_beta0: Optional[int] = None
        self.prev_beta2: Optional[int] = None
        self.event_history: List[ContactTopologyEvent] = []

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def current_grid(self) -> np.ndarray:
        """Compute the combined-SDF grid at the current body state."""
        return combined_sdf_grid(
            self.bodies, self.bbox_min, self.bbox_max, self.resolution,
        )

    def current_betti(self) -> Dict[int, int]:
        """Compute Betti numbers of the current combined sublevel set.

        Returns a dict with ``0`` (and ``2`` when tracking enclosure)
        keys; ``beta_k = 0`` when no Betti numbers of dimension ``k``
        cross the boundary.
        """
        grid = self.current_grid()
        return self._betti_from_grid(grid)

    def _betti_from_grid(self, grid: np.ndarray) -> Dict[int, int]:
        """Shared helper: clip, compute diagrams, filter, read Betti."""
        max_dim = 2 if self.track_enclosure else 0
        clipped = clip_to_interior(grid)
        diagrams = compute_persistence_diagrams(
            clipped, max_dimension=max_dim,
        )
        f_range = filtration_value_range(clipped)
        diagrams_filt = filter_by_lifetime(
            diagrams,
            self.lifetime_threshold,
            relative=True,
            filtration_range=f_range,
        )
        betti = betti_numbers_at(diagrams_filt, t=-1e-6)
        # Make sure the expected keys are present
        if 0 not in betti:
            betti[0] = 0
        if self.track_enclosure and 2 not in betti:
            betti[2] = 0
        return betti

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def update(self, load_step: int) -> List[ContactTopologyEvent]:
        """Recompute Betti numbers and emit events for any transitions.

        Parameters
        ----------
        load_step : int
            Current simulation step index.  Stored in any emitted event.

        Returns
        -------
        events : list of ContactTopologyEvent
            All transitions since the last ``update`` call.  Empty on
            the first call (baseline only).
        """
        grid = self.current_grid()
        betti = self._betti_from_grid(grid)
        beta0 = betti.get(0, 0)
        beta2 = betti.get(2, 0)

        events: List[ContactTopologyEvent] = []
        if self.prev_beta0 is not None:
            if beta0 < self.prev_beta0:
                events.append(
                    ContactTopologyEvent(
                        event_type="first_contact",
                        load_step=load_step,
                        beta0_before=self.prev_beta0,
                        beta0_after=beta0,
                        beta2_before=self.prev_beta2 or 0,
                        beta2_after=beta2,
                        location=self._localize_contact(grid),
                    )
                )
            elif beta0 > self.prev_beta0:
                events.append(
                    ContactTopologyEvent(
                        event_type="separation",
                        load_step=load_step,
                        beta0_before=self.prev_beta0,
                        beta0_after=beta0,
                        beta2_before=self.prev_beta2 or 0,
                        beta2_after=beta2,
                        location=None,
                    )
                )
            if (
                self.track_enclosure
                and self.prev_beta2 is not None
                and beta2 > self.prev_beta2
            ):
                events.append(
                    ContactTopologyEvent(
                        event_type="enclosure",
                        load_step=load_step,
                        beta0_before=self.prev_beta0,
                        beta0_after=beta0,
                        beta2_before=self.prev_beta2,
                        beta2_after=beta2,
                        location=self._localize_contact(grid),
                    )
                )

        self.prev_beta0 = beta0
        self.prev_beta2 = beta2
        self.event_history.extend(events)
        return events

    def reset(self) -> None:
        """Forget the baseline so the next ``update`` re-establishes it."""
        self.prev_beta0 = None
        self.prev_beta2 = None
        self.event_history.clear()

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def _localize_contact(self, grid: np.ndarray) -> np.ndarray:
        """Find an approximate physical location of the contact region.

        Returns the grid point where the combined SDF is most negative.
        For a first-contact event this is the deepest overlap between
        the two bodies — a reasonable proxy for the contact point.
        """
        flat_idx = int(np.argmin(grid))
        ix, iy, iz = np.unravel_index(flat_idx, grid.shape)
        frac = np.array([ix, iy, iz], dtype=np.float64) / max(
            self.resolution - 1, 1,
        )
        return self.bbox_min + frac * (self.bbox_max - self.bbox_min)
