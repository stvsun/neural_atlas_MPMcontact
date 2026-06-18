"""Topology-aware contact detection demo.

Sweeps two analytic ball SDFs from separated to overlapping (and back)
while a :class:`ContactTopologyMonitor` watches the combined SDF's
Betti numbers.  The monitor should emit a ``first_contact`` event at
the step where β₀ transitions from 2 → 1, and a ``separation`` event
when the bodies move apart again.

Run::

    PYTHONPATH=. python benchmarks/contact/contact_topology_demo.py

Outputs are written to ``runs/contact_topology_demo/``.
"""

import os
import json

import numpy as np
import torch

from solvers.contact.contact_pair import ContactBody
from solvers.contact.contact_topology import ContactTopologyMonitor


# ── Movable analytic sphere SDF ──────────────────────────────────────


class MovableSphereSDF(torch.nn.Module):
    """phi(x) = |x - c| - r   with a mutable centre buffer."""

    def __init__(self, center, radius):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float64),
        )
        self.radius = float(radius)

    def set_center(self, new_center):
        self.center.copy_(torch.as_tensor(new_center, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center.unsqueeze(0)).norm(dim=1) - self.radius


def make_body(body_id: int, center, radius: float) -> ContactBody:
    sdf = MovableSphereSDF(center=center, radius=radius).double()
    return ContactBody(
        body_id=body_id,
        sdf_net=sdf,
        seeds=torch.tensor([center], dtype=torch.float64),
        support_radii=torch.tensor([radius], dtype=torch.float64),
    )


# ── Main sweep ───────────────────────────────────────────────────────


def main():
    out_dir = os.path.join("runs", "contact_topology_demo")
    os.makedirs(out_dir, exist_ok=True)

    radius = 0.25
    body_a = make_body(0, center=(-0.6, 0.0, 0.0), radius=radius)
    body_b = make_body(1, center=(+0.6, 0.0, 0.0), radius=radius)

    monitor = ContactTopologyMonitor(
        bodies=[body_a, body_b],
        bbox_min=np.array([-1.2, -0.6, -0.6]),
        bbox_max=np.array([+1.2, +0.6, +0.6]),
        resolution=20,
        lifetime_threshold=0.02,
    )

    # Sweep: centres move from ±0.6 towards ±0.1 (approach), then back.
    approach = np.linspace(0.6, 0.1, 11)
    separation = np.linspace(0.1, 0.6, 11)[1:]    # skip repeated 0.1
    sweep = np.concatenate([approach, separation])

    print(f"radius           = {radius}")
    print(f"touching at      = d < {radius:.3f}")
    print(f"resolution       = {monitor.resolution}")
    print(f"lifetime thresh. = {monitor.lifetime_threshold}")
    print()
    print(f"{'step':>4} {'dist (one-sided)':>18} {'beta_0':>8} "
          f"{'events':>20}")
    print("-" * 54)

    history = []
    for step, d in enumerate(sweep):
        body_a.sdf_net.set_center((-d, 0.0, 0.0))
        body_b.sdf_net.set_center((+d, 0.0, 0.0))

        events = monitor.update(load_step=step)
        beta0 = monitor.prev_beta0

        event_str = ", ".join(e.event_type for e in events) or ""
        print(f"{step:>4} {d:>18.4f} {beta0:>8} {event_str:>20}")

        history.append({
            "step": step,
            "distance": float(d),
            "beta0": int(beta0) if beta0 is not None else -1,
            "events": [
                {
                    "type": e.event_type,
                    "beta0_before": e.beta0_before,
                    "beta0_after": e.beta0_after,
                    "location": (
                        e.location.tolist()
                        if e.location is not None else None
                    ),
                }
                for e in events
            ],
        })

    # Save history
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    all_events = monitor.event_history
    first_contact = [
        e for e in all_events if e.event_type == "first_contact"
    ]
    separations = [
        e for e in all_events if e.event_type == "separation"
    ]

    print()
    print("=" * 54)
    print("SUMMARY")
    print("=" * 54)
    print(f"total events              = {len(all_events)}")
    print(f"first_contact events      = {len(first_contact)}")
    print(f"separation events         = {len(separations)}")
    if first_contact:
        fc = first_contact[0]
        loc = fc.location
        print(
            f"first_contact at step {fc.load_step}, "
            f"location = ({loc[0]:+.4f}, {loc[1]:+.4f}, {loc[2]:+.4f})"
        )
    if separations:
        print(f"separation at step {separations[0].load_step}")

    expected = len(first_contact) == 1 and len(separations) == 1
    print()
    print(
        f"Expected exactly 1 first_contact and 1 separation: "
        f"{'OK' if expected else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
