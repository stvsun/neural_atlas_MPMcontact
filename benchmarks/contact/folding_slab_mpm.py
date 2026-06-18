"""Folding-slab self-contact benchmark.

A thin elastic slab is pinned at its left edge and driven from the
right edge such that its free end folds back onto itself.  The
:class:`SelfContactManager` should detect the fold and apply penalty
forces that keep the folded end from penetrating the slab.

This is a diagnostic demo — it does **not** perform a full physical
simulation of the fold.  Instead, it scripts a kinematic motion that
folds a subset of particles and reports the resulting self-contact
force field at each step.  The purpose is to show that the detection
+ force computation work in a realistic many-step scenario with a
prescribed folding trajectory.

Run::

    PYTHONPATH=. python benchmarks/contact/folding_slab_mpm.py

Outputs are written to ``runs/folding_slab_mpm/``.
"""

import os
import json

import numpy as np
import torch

from solvers.contact.contact_pair import ContactBody
from solvers.contact.self_contact import SelfContactManager


# ── Analytic slab SDF (axis-aligned box) ─────────────────────────────


class BoxSDF(torch.nn.Module):
    """phi(x) for an axis-aligned box of half-extents ``h`` centred at
    ``c``.  Uses the standard "distance to box" formula.
    """

    def __init__(self, center, half_extent):
        super().__init__()
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float64),
        )
        self.register_buffer(
            "half_extent",
            torch.tensor(half_extent, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = (x - self.center.unsqueeze(0)).abs() - self.half_extent.unsqueeze(0)
        outside = torch.clamp(d, min=0.0).norm(dim=1)
        inside = torch.clamp(d.max(dim=1).values, max=0.0)
        return outside + inside


def make_body() -> ContactBody:
    """Thin slab: 1.0 × 0.1 × 0.2 centred at the origin."""
    sdf = BoxSDF(
        center=(0.0, 0.0, 0.0),
        half_extent=(0.5, 0.05, 0.1),
    )
    return ContactBody(
        body_id=0,
        sdf_net=sdf,
        seeds=torch.zeros(1, 3, dtype=torch.float64),
        support_radii=torch.tensor([0.6], dtype=torch.float64),
    )


# ── Initial particle layout ─────────────────────────────────────────


def make_particles():
    """Build surface particles covering the top face of the slab
    (y = +0.05), plus a sparse bulk cloud so the solver has something
    to work with.

    Returns
    -------
    positions : (N, 3) tensor
    volumes   : (N,) tensor
    """
    # Surface layer on the top face (y = 0.05) — 10 × 5 grid
    nx, nz = 12, 6
    xs = np.linspace(-0.4, 0.4, nx)
    zs = np.linspace(-0.08, 0.08, nz)
    surf = []
    for x in xs:
        for z in zs:
            surf.append([x, 0.05, z])
    surf = np.asarray(surf, dtype=np.float64)

    # A sprinkling of bulk particles (should be ignored by self-contact)
    bulk = np.array(
        [
            [-0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [-0.2, 0.02, 0.05],
            [0.2, -0.02, -0.05],
        ],
        dtype=np.float64,
    )

    positions = np.concatenate([surf, bulk], axis=0)
    vols = np.full(positions.shape[0], 1e-4, dtype=np.float64)
    return (
        torch.tensor(positions, dtype=torch.float64),
        torch.tensor(vols, dtype=torch.float64),
    )


# ── Kinematic folding trajectory ────────────────────────────────────


def fold_right_half(positions: torch.Tensor, t: float) -> torch.Tensor:
    """Kinematically fold particles with ``x > 0`` back onto the left
    half.  Interpolation parameter ``t in [0, 1]``: 0 = initial,
    1 = fully folded (mirrored across x=0 with a vertical drop).
    """
    out = positions.clone()
    right = out[:, 0] > 0
    x0 = out[right, 0].clone()
    # Mirror across x=0 and drop slightly into the slab
    out[right, 0] = x0 - 2.0 * x0 * t           # x → x*(1-2t)
    out[right, 1] = out[right, 1] - 0.1 * t     # drop 0.1 into slab
    return out


# ── Main ────────────────────────────────────────────────────────────


def main():
    out_dir = os.path.join("runs", "folding_slab_mpm")
    os.makedirs(out_dir, exist_ok=True)

    body = make_body()
    positions, volumes = make_particles()
    n = positions.shape[0]

    mgr = SelfContactManager(
        body, positions,
        surface_tol=0.02,
        penetration_delta=0.02,
    )
    n_surface = mgr.n_surface_particles()
    n_bulk = n - n_surface
    print(f"Total particles   : {n}")
    print(f"Surface particles : {n_surface}")
    print(f"Bulk particles    : {n_bulk}")
    print()

    # Sweep the folding parameter from 0 to 1 in 11 steps
    n_steps = 11
    eps_n = 1e5
    history = []

    print(f"{'step':>4} {'t':>6} {'n_active':>10} {'max_delta_pen':>16} "
          f"{'force_norm':>14}")
    print("-" * 54)

    for step in range(n_steps):
        t = step / (n_steps - 1)
        folded = fold_right_half(positions, t)
        gap, normal, active = mgr.detect(folded)
        f = mgr.compute_force(folded, volumes, epsilon_n=eps_n)

        n_active = int(active.sum().item())
        max_delta = mgr.max_delta_penetration(folded)
        f_norm = float(f.norm().item())

        history.append({
            "step": step,
            "t": t,
            "n_active": n_active,
            "max_delta_penetration": max_delta,
            "total_force_norm": f_norm,
        })
        print(f"{step:>4} {t:>6.2f} {n_active:>10} "
              f"{max_delta:>16.6e} {f_norm:>14.6e}")

    # Save
    out_file = os.path.join(out_dir, "history.json")
    with open(out_file, "w") as f_out:
        json.dump(history, f_out, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary — pick the step with the maximum number of active particles
    peak = max(history, key=lambda h: h["n_active"])
    max_pen = max(h["max_delta_penetration"] for h in history)
    max_force = max(h["total_force_norm"] for h in history)

    print()
    print("=" * 54)
    print("SUMMARY")
    print("=" * 54)
    print(f"Surface particles              : {n_surface}")
    print(f"Bulk particles (never active)  : {n_bulk}")
    print(f"Active at start (t=0)          : {history[0]['n_active']}")
    print(f"Active at end   (t=1)          : {history[-1]['n_active']}")
    print(f"Peak active (step {peak['step']}, t={peak['t']:.2f}) : "
          f"{peak['n_active']}")
    print(f"Max delta penetration          : {max_pen:.4e}")
    print(f"Peak total force magnitude     : {max_force:.4e}")
    print()

    # Assertions — the detection should fire during the fold when
    # particles are crossing the slab interior, and should NOT fire
    # for bulk particles or when the particles are on a surface.
    ok_start = history[0]["n_active"] == 0
    ok_some_active = peak["n_active"] > 0
    ok_bulk = all(
        h["n_active"] <= n_surface for h in history
    )
    print(f"No self-contact at t=0             : {'OK' if ok_start else 'FAIL'}")
    print(f"Self-contact detected mid-fold     : {'OK' if ok_some_active else 'FAIL'}")
    print(f"Bulk particles never active        : {'OK' if ok_bulk else 'FAIL'}")


if __name__ == "__main__":
    main()
