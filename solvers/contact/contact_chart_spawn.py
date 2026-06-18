"""Chart spawning for topology-aware contact events.

Bridges the :class:`ContactTopologyMonitor` output (a
``ContactTopologyEvent`` with a physical-space contact location) to the
existing atlas chart-spawning machinery (``SpawnedChartPair`` +
``SchwarzMPMSolver.add_charts``).

For a first-contact event, the helper:

1. Uses the autograd gradient of body A's SDF at the contact location
   to compute the outward contact normal ``n_A``.
2. Places two chart seeds symmetrically on opposite sides of the
   contact interface along that normal.
3. Builds an orthonormal frame at each seed via
   ``ChartSpawner._frame_from_normal`` (static helper reused verbatim).
4. Packages the result as a ``SpawnedChartPair`` with
   ``edge_type="contact"``.

Note on parent-chart semantics: the existing ``SpawnedChartPair`` carries
a single ``parent_chart`` index used for warm-starting both spawned
decoders.  For two-body contact the ideal behaviour would be different
parent charts for the + and - sides, but changing the dataclass is out
of scope for this helper — it sticks with the single-parent convention
and picks whichever existing seed is nearest to the contact location.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from atlas.topo.chart_spawn import ChartSpawner, SpawnedChartPair
from solvers.contact.contact_pair import ContactBody
from solvers.contact.contact_topology import ContactTopologyEvent


def _infer_sdf_device_dtype(body: ContactBody):
    """Detect the device and dtype of a body's SDF network (fallback
    to cpu/float64 when the network is parameter-free).
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


def compute_contact_normal(
    body: ContactBody,
    point: np.ndarray,
) -> np.ndarray:
    """Outward unit normal of ``body`` at a physical-space point.

    Uses torch autograd on the body's SDF (the same pattern as
    ``solvers/contact/gap.py:evaluate_gap``).  The query tensor is
    built on the SDF's own device and dtype so mixed-precision and
    non-CPU SDFs work without dtype-mismatch errors.

    Parameters
    ----------
    body : ContactBody
        Body whose SDF gradient we evaluate.
    point : np.ndarray, shape (3,)
        Physical-space query point.

    Returns
    -------
    normal : np.ndarray, shape (3,)
        Unit outward normal ``grad(phi) / |grad(phi)|``.  If the
        gradient vanishes (medial axis), returns ``[0, 0, 1]`` as a
        safe fallback.
    """
    device, dtype = _infer_sdf_device_dtype(body)
    x = torch.tensor(
        point, dtype=dtype, device=device,
    ).reshape(1, 3).requires_grad_(True)
    with torch.enable_grad():
        phi = body.sdf_net(x)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)
        grad = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=False,
        )[0]
    grad_np = grad.detach().squeeze(0).cpu().numpy().astype(np.float64)

    nrm = float(np.linalg.norm(grad_np))
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return grad_np / nrm


def spawn_contact_chart_pair(
    event: ContactTopologyEvent,
    bodies: List[ContactBody],
    existing_seeds: Optional[np.ndarray] = None,
    radius: float = 0.2,
    normal_from_body: int = 0,
) -> SpawnedChartPair:
    """Build a :class:`SpawnedChartPair` from a contact-topology event.

    Parameters
    ----------
    event : ContactTopologyEvent
        The detected event.  Must be of type ``"first_contact"`` or
        ``"enclosure"`` (separation events carry no location and are
        rejected).
    bodies : list of ContactBody
        The bodies participating in the contact; ``bodies[normal_from_body]``
        provides the SDF used to compute the contact normal.
    existing_seeds : np.ndarray, shape (M, 3), optional
        Seeds of the atlas the caller intends to extend.  If supplied,
        the ``parent_chart`` field of the returned pair is the index of
        the nearest seed.  Defaults to ``0`` when not supplied.
    radius : float
        Support radius for the spawned charts and the seed offset from
        the contact interface (seeds are placed at
        ``location ± 0.5 * radius * n``).
    normal_from_body : int
        Which body's SDF to use for the normal computation.  Defaults to
        body 0; body 1's normal is (up to SDF approximation error) the
        negative of this.

    Returns
    -------
    pair : SpawnedChartPair
        Tagged with ``edge_type="contact"`` and ``activation_step``
        copied from the event.

    Raises
    ------
    ValueError
        If the event has no location (e.g., a separation event) or if
        ``normal_from_body`` is out of range.
    """
    if event.location is None:
        raise ValueError(
            f"Cannot spawn a chart pair for event of type "
            f"{event.event_type!r}: no contact location available."
        )
    if not (0 <= normal_from_body < len(bodies)):
        raise ValueError(
            f"normal_from_body={normal_from_body} is out of range for "
            f"{len(bodies)} bodies."
        )

    center = np.asarray(event.location, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(
            f"event.location must have shape (3,), got {center.shape}"
        )

    # Step 1: contact normal from the selected body's SDF
    normal = compute_contact_normal(bodies[normal_from_body], center)

    # Step 2: seeds on opposite sides of the contact interface.  The
    # "plus" side follows the outward normal of body A (i.e., towards
    # body B in an overlapping configuration); the "minus" side is on
    # body A's interior.
    seed_plus = center + 0.5 * radius * normal
    seed_minus = center - 0.5 * radius * normal

    # Step 3: orthonormal frames via the existing ChartSpawner helper
    frame_plus = ChartSpawner._frame_from_normal(normal)
    frame_minus = ChartSpawner._frame_from_normal(-normal)

    # Step 4: nearest existing seed for warm-start
    if existing_seeds is not None and len(existing_seeds) > 0:
        seeds_arr = np.asarray(existing_seeds, dtype=np.float64)
        dists = np.linalg.norm(seeds_arr - center[None, :], axis=-1)
        parent_chart = int(np.argmin(dists))
    else:
        parent_chart = 0

    return SpawnedChartPair(
        seed_plus=seed_plus,
        seed_minus=seed_minus,
        frame_plus=frame_plus,
        frame_minus=frame_minus,
        radius=float(radius),
        parent_chart=parent_chart,
        edge_type="contact",
        activation_step=event.load_step,
    )
