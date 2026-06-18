"""CompactChartNet – Voronoi sub-atlas architecture for Schwarz-PINN.

Each chart's PINN is replaced by a mixture of M small Tanh-MLPs whose
contributions are blended via a softmax partition-of-unity (POU) keyed on
Euclidean distance from M sub-seeds arranged inside the chart's support ball.

Key properties
--------------
* C∞ everywhere   → autograd Laplacian is well-defined (unlike tri-linear grids)
* Spatially local → gradient of sub-net m is concentrated near sub-seed m
* Exclusive-zone fine-tuning: after the main Schwarz loop, only sub-nets
  whose seeds are *not* covered by any neighbouring chart's support ball
  receive gradient updates — the rest are frozen in place.

Architecture defaults (match parameter count ≈ LocalPoissonPINN width=64 depth=4)
-----------------
  n_subseed   = 9   (1 centre + 8 cube-corner sub-seeds at ±r/3 offsets)
  sub_width   = 32
  sub_depth   = 2
  tau_scale   = 0.125  (τ = tau_scale × support_r; controls POU locality)

Usage
-----
  from compact_chart_net import CompactChartNet, build_compact_u_nets

  u_nets = build_compact_u_nets(
      n_charts, seeds, support_r, t1, t2, nvec,
      device, dtype,
      n_subseed=9, sub_width=32, sub_depth=2, tau_scale=0.125,
  )

  # During fine-tuning – freeze sub-nets that overlap neighbours:
  for i, net in enumerate(u_nets):
      neighbours = [j for j in range(n_charts) if j != i and overlap(i, j)]
      net.freeze_overlapping_subnets(
          chart_seeds=[seeds[j] for j in neighbours],
          chart_radii=[support_r[j] for j in neighbours],
      )
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny MLP building block
# ---------------------------------------------------------------------------

class _SmallMLP(nn.Module):
    """width-×depth Tanh MLP: ℝ³ → ℝ¹, Xavier init."""

    def __init__(self, width: int = 32, depth: int = 2):
        super().__init__()
        assert depth >= 1, "depth must be ≥ 1"
        layers: List[nn.Linear] = [nn.Linear(3, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)

        for layer in self.hidden:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 3) → (N, 1)
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)


# ---------------------------------------------------------------------------
# Sub-seed layout helpers
# ---------------------------------------------------------------------------

def _make_subseed_offsets(n_subseed: int, support_r: float) -> torch.Tensor:
    """Return (n_subseed, 3) offsets in *local* ξ-coordinates.

    Layout:
      seed 0  → origin (chart centre)
      seeds 1…8 → ±r/3 cube corners  (up to 8)
      seeds 9… → random uniform in ball (for n_subseed > 9)

    All seeds lie within 0.6 × support_r of the origin so they stay
    well inside the chart's support ball.
    """
    offsets = [torch.zeros(3)]
    corner_scale = support_r / 3.0
    corners = [
        torch.tensor([sx, sy, sz], dtype=torch.float64) * corner_scale
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    for c in corners[: min(8, n_subseed - 1)]:
        offsets.append(c)

    # Fill remaining slots with uniform-ball random seeds.
    remaining = n_subseed - len(offsets)
    if remaining > 0:
        torch.manual_seed(0)
        pts = torch.randn(remaining * 4, 3, dtype=torch.float64)
        pts = pts / (pts.norm(dim=1, keepdim=True) + 1e-8)
        r_samples = torch.rand(remaining * 4, 1, dtype=torch.float64) ** (1.0 / 3.0)
        pts = pts * r_samples * (support_r * 0.55)
        offsets.extend(pts[:remaining].unbind(0))

    return torch.stack(offsets[:n_subseed], dim=0).to(torch.float64)  # (M, 3)


# ---------------------------------------------------------------------------
# CompactChartNet
# ---------------------------------------------------------------------------

class CompactChartNet(nn.Module):
    """Voronoi sub-atlas chart network.

    Given local ξ-coordinates (shape N×3), predicts u by blending M small
    MLPs via a softmax partition-of-unity centred on M sub-seeds.

    Parameters
    ----------
    support_r : float
        Support radius of this chart in *physical* (= ξ) space.  Used to
        place sub-seeds and to set the POU bandwidth τ.
    n_subseed  : int   Number of sub-seeds (default 9).
    sub_width  : int   Hidden width of each sub-net (default 32).
    sub_depth  : int   Hidden depth of each sub-net (default 2).
    tau_scale  : float τ = tau_scale × support_r  (default 0.125).
    """

    def __init__(
        self,
        support_r: float,
        n_subseed: int = 9,
        sub_width: int = 32,
        sub_depth: int = 2,
        tau_scale: float = 0.125,
    ):
        super().__init__()
        self.support_r = float(support_r)
        self.n_subseed = int(n_subseed)
        self.tau_scale = float(tau_scale)
        self.tau = self.tau_scale * self.support_r

        # Sub-seed offsets in ξ-space  (M, 3)
        offsets = _make_subseed_offsets(n_subseed, self.support_r)
        self.register_buffer("subseed_offsets", offsets.float())

        # One small MLP per sub-seed.
        self.subnets = nn.ModuleList(
            [_SmallMLP(width=sub_width, depth=sub_depth) for _ in range(n_subseed)]
        )

        # Mask: True ⟹ this sub-net's gradients are *frozen* during fine-tune.
        # Stored as a plain Python list so it is never serialised into the
        # state_dict, which keeps checkpoint compatibility simple.
        self._frozen_mask: List[bool] = [False] * n_subseed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """xi : (N, 3) in chart-local ξ-coordinates → u : (N, 1)"""
        # Distance from each sample to each sub-seed: (N, M)
        # subseed_offsets is (M, 3); xi is (N, 3)
        tau2 = self.tau ** 2 + 1e-12
        # (N, 1, 3) − (1, M, 3)  →  (N, M, 3)
        diff = xi.unsqueeze(1) - self.subseed_offsets.unsqueeze(0).to(xi)
        dist2 = (diff ** 2).sum(dim=2)  # (N, M)

        # Softmax partition-of-unity weights: (N, M)
        log_phi = -dist2 / (2.0 * tau2)
        weights = torch.softmax(log_phi, dim=1)  # (N, M)

        # Evaluate sub-nets, handling frozen masks via gradient hooks.
        vals = []
        for m, subnet in enumerate(self.subnets):
            # xi shifted so that sub-seed m is at the origin
            xi_m = xi - self.subseed_offsets[m].to(xi)
            v = subnet(xi_m)  # (N, 1)
            if self._frozen_mask[m]:
                v = v.detach()
            vals.append(v)

        # Weighted blend: sum_m φ_m(ξ) * u_m(ξ − s_m)
        vals_t = torch.cat(vals, dim=1)  # (N, M)
        out = (weights * vals_t).sum(dim=1, keepdim=True)  # (N, 1)
        return out

    # ------------------------------------------------------------------
    # Exclusive-zone masking
    # ------------------------------------------------------------------

    def freeze_overlapping_subnets(
        self,
        neighbour_seeds_xi: List[torch.Tensor],
        neighbour_radii: List[float],
        safety_factor: float = 4.0,
    ) -> int:
        """Mark sub-nets that lie inside a neighbour's support ball as frozen.

        A sub-seed s_m (in ξ-space, which equals physical-space here because
        the TNB-frame is rigid) is considered *overlapping* with neighbour j if

            ||s_m − seed_j_in_xi||  <  radius_j + safety_factor × τ

        where seed_j_in_xi is the neighbour seed expressed in THIS chart's
        local ξ-coordinate frame.

        Parameters
        ----------
        neighbour_seeds_xi : list of (3,) tensors
            Each neighbour chart's seed expressed in *this* chart's local
            ξ-coordinates (i.e. already transformed by local_coords).
        neighbour_radii : list of float
            Each neighbour's support radius.
        safety_factor : float
            Extra margin in units of τ (default 4.0).

        Returns
        -------
        n_frozen : int  Number of sub-nets newly frozen.
        """
        margin = safety_factor * self.tau
        offsets = self.subseed_offsets  # (M, 3)
        n_frozen = 0
        for m in range(self.n_subseed):
            s_m = offsets[m]  # (3,)
            for seed_xi, r_j in zip(neighbour_seeds_xi, neighbour_radii):
                seed_xi_t = torch.as_tensor(seed_xi, dtype=offsets.dtype, device=offsets.device)
                dist = (s_m - seed_xi_t).norm().item()
                if dist < float(r_j) + margin:
                    if not self._frozen_mask[m]:
                        self._frozen_mask[m] = True
                        n_frozen += 1
                    break
        return n_frozen

    def unfreeze_all_subnets(self) -> None:
        """Remove all frozen flags (call before main Schwarz loop)."""
        self._frozen_mask = [False] * self.n_subseed

    def n_frozen(self) -> int:
        return sum(self._frozen_mask)

    def n_exclusive(self) -> int:
        return self.n_subseed - self.n_frozen()

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function that mirrors how LocalPoissonPINN is built in the solver
# ---------------------------------------------------------------------------

def build_compact_u_nets(
    n_charts: int,
    support_r: "torch.Tensor",          # (n_charts,) physical support radii
    device: torch.device,
    dtype: torch.dtype,
    n_subseed: int = 9,
    sub_width: int = 32,
    sub_depth: int = 2,
    tau_scale: float = 0.125,
) -> List[CompactChartNet]:
    """Build one CompactChartNet per chart, move to device/dtype, return list."""
    nets = []
    for i in range(n_charts):
        r = float(support_r[i].item())
        net = CompactChartNet(
            support_r=r,
            n_subseed=n_subseed,
            sub_width=sub_width,
            sub_depth=sub_depth,
            tau_scale=tau_scale,
        ).to(device=device, dtype=dtype)
        nets.append(net)
    return nets


# ---------------------------------------------------------------------------
# Exclusive-zone fine-tuning helper
# ---------------------------------------------------------------------------

def apply_exclusive_zone_freeze(
    u_nets: List[CompactChartNet],
    seeds: "torch.Tensor",          # (n_charts, 3) physical seeds
    support_r: "torch.Tensor",      # (n_charts,) physical support radii
    t1: "torch.Tensor",             # (n_charts, 3) tangent vectors
    t2: "torch.Tensor",             # (n_charts, 3)
    nvec: "torch.Tensor",           # (n_charts, 3)
    overlap_pairs: List[tuple],     # list of (i, j) pairs with overlap
) -> dict:
    """Freeze overlapping sub-nets in every CompactChartNet.

    For each chart i, iterates over neighbour charts j (from overlap_pairs)
    and calls net_i.freeze_overlapping_subnets(neighbour_seed_in_xi_i, r_j).

    The neighbour seed is expressed in chart i's ξ-frame via:
        ξ_j = local_coords(seed_j, seed_i, t1_i, t2_i, n_i)

    Returns a dict  {i: (n_frozen, n_exclusive)} for logging.
    """
    # Build neighbour map
    from collections import defaultdict
    neighbours: dict = defaultdict(list)
    for (a, b) in overlap_pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)

    stats = {}
    for i, net in enumerate(u_nets):
        if not isinstance(net, CompactChartNet):
            continue
        net.unfreeze_all_subnets()
        nbr_seeds_xi: List[torch.Tensor] = []
        nbr_radii: List[float] = []
        for j in neighbours[i]:
            # Transform neighbour seed into chart i's ξ-frame.
            s_j = seeds[j].unsqueeze(0)  # (1, 3)
            d = s_j - seeds[i].unsqueeze(0)
            xi_j = torch.stack([
                (d * t1[i]).sum(dim=-1),
                (d * t2[i]).sum(dim=-1),
                (d * nvec[i]).sum(dim=-1),
            ], dim=-1).squeeze(0)  # (3,)
            nbr_seeds_xi.append(xi_j)
            nbr_radii.append(float(support_r[j].item()))

        n_frz = net.freeze_overlapping_subnets(nbr_seeds_xi, nbr_radii)
        stats[i] = (n_frz, net.n_exclusive())

    return stats


# ---------------------------------------------------------------------------
# Parameter-count comparison
# ---------------------------------------------------------------------------

def print_param_comparison(
    compact_net: CompactChartNet,
    mlp_width: int = 64,
    mlp_depth: int = 4,
) -> None:
    """Print parameter counts for CompactChartNet vs dense MLP."""
    compact_params = compact_net.count_parameters()

    # Reproduce the MLP parameter count for comparison.
    def _mlp_params(in_d: int, out_d: int, w: int, d: int) -> int:
        p = in_d * w + w          # first layer
        for _ in range(d - 1):   # subsequent hidden
            p += w * w + w
        p += w * out_d + out_d   # output layer
        return p

    dense_params = _mlp_params(3, 1, mlp_width, mlp_depth)
    print(
        f"CompactChartNet params : {compact_params:,d}  "
        f"(M={compact_net.n_subseed}, w={compact_net.subnets[0].out.in_features}, "
        f"d={len(compact_net.subnets[0].hidden) + 1})"
    )
    print(
        f"Dense MLP params       : {dense_params:,d}  "
        f"(width={mlp_width}, depth={mlp_depth})"
    )


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(42)
    net = CompactChartNet(support_r=0.5, n_subseed=9, sub_width=32, sub_depth=2, tau_scale=0.125)
    xi = torch.randn(64, 3) * 0.3
    xi.requires_grad_(True)
    u = net(xi)
    assert u.shape == (64, 1), f"Unexpected shape: {u.shape}"

    # Check that second-order autograd works (needed for Laplacian)
    grads = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
    lap_terms = []
    for j in range(3):
        d2 = torch.autograd.grad(
            grads[:, j].sum(), xi, create_graph=False, retain_graph=True
        )[0][:, j]
        lap_terms.append(d2)
    lap = sum(lap_terms)
    assert lap.shape == (64,), f"Laplacian shape mismatch: {lap.shape}"
    print("Smoke test passed.")
    print_param_comparison(net)
    print(f"Sub-seed offsets (first 3):\n{net.subseed_offsets[:3]}")
