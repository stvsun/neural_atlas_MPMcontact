#!/usr/bin/env python3
"""
Rabbit Poisson solver: PINN with FEM-solution initialization and decaying guidance.

Phases:
  1. BC warm-start (300 epochs): train each chart network on g=0
  2. FEM pretraining (500 epochs): fit chart networks to FEM solution
  3. Schwarz iterations (120 sweeps) with decaying FEM guidance loss

Uses CompactChartNet (9 sub-seeds, 32x2 tanh subnets, tau_scale=0.125) and
direct-coord Poisson residual (rigid TNB frame, no decoder needed).
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compact_chart_net import CompactChartNet, build_compact_u_nets

# ---------------------------------------------------------------------------
# Device / seed helpers
# ---------------------------------------------------------------------------

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------------

def manufactured_u(x: torch.Tensor) -> torch.Tensor:
    """u*(x) = sin(pi x1) sin(pi x2) sin(pi x3)"""
    return (
        torch.sin(math.pi * x[:, 0:1])
        * torch.sin(math.pi * x[:, 1:2])
        * torch.sin(math.pi * x[:, 2:3])
    )


def manufactured_f(x: torch.Tensor) -> torch.Tensor:
    """f = 3 pi^2 u*"""
    return 3.0 * (math.pi ** 2) * manufactured_u(x)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def local_coords(
    x: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Map physical points to chart-local TNB coordinates."""
    d = x - seed.unsqueeze(0)
    return torch.stack(
        [
            torch.sum(d * t1.unsqueeze(0), dim=1),
            torch.sum(d * t2.unsqueeze(0), dim=1),
            torch.sum(d * n.unsqueeze(0), dim=1),
        ],
        dim=1,
    )


def xi_to_physical(
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Map chart-local coordinates back to physical space via rigid TNB."""
    return (
        seed.unsqueeze(0)
        + xi[:, 0:1] * t1.unsqueeze(0)
        + xi[:, 1:2] * t2.unsqueeze(0)
        + xi[:, 2:3] * n.unsqueeze(0)
    )


# ---------------------------------------------------------------------------
# Direct-coord Poisson residual via rigid TNB
# ---------------------------------------------------------------------------

def direct_poisson_residual(
    u_net: torch.nn.Module,
    x_phys: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Compute -Delta_xi u_hat - f(x_phys).

    Since TNB is orthonormal, Laplacian_x = Laplacian_xi exactly.
    Returns residual of shape (N, 1).
    """
    x = x_phys.clone().detach().requires_grad_(True)
    xi = local_coords(x, seed, t1, t2, n)
    u = u_net(xi)  # (N, 1)

    grad_u = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3)

    lap = torch.zeros_like(u)
    for j in range(3):
        d2 = torch.autograd.grad(
            grad_u[:, j], x,
            grad_outputs=torch.ones(grad_u[:, j].shape, device=x.device, dtype=x.dtype),
            create_graph=True,
            retain_graph=True,
        )[0][:, j:j+1]
        lap = lap + d2

    residual = -lap - manufactured_f(x)
    return residual


def grad_u_physical_tnb(
    u_net: torch.nn.Module,
    x_phys: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Return nabla_x u via rigid TNB frame. Shape (N, 3).

    Computes grad w.r.t. xi directly (xi as leaf), then rotates to physical.
    """
    xi = local_coords(x_phys.detach(), seed, t1, t2, n)
    xi = xi.clone().detach().requires_grad_(True)
    u = u_net(xi)
    grad_xi = torch.autograd.grad(
        u, xi,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=False,
    )[0]  # (N, 3)
    # Rotate from local TNB to physical frame
    grad_x = (
        grad_xi[:, 0:1] * t1.unsqueeze(0)
        + grad_xi[:, 1:2] * t2.unsqueeze(0)
        + grad_xi[:, 2:3] * n.unsqueeze(0)
    )
    return grad_x


# ---------------------------------------------------------------------------
# Atlas loader (minimal: decoders + masks from checkpoint)
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(width, out_dim)
        for layer in self.hidden:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)


class MaskNet(torch.nn.Module):
    def __init__(self, width: int = 48, depth: int = 3):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, xi: torch.Tensor, chart_scale: torch.Tensor) -> torch.Tensor:
        xi_n = xi / torch.clamp(chart_scale, min=1e-6)
        return self.net(xi_n).squeeze(-1)


def load_mask_nets(
    atlas_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> List[MaskNet]:
    """Load only the mask networks from the atlas checkpoint."""
    ckpt = torch.load(atlas_checkpoint, map_location=torch.device("cpu"))
    mask_kw = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    mask_states = ckpt["mask_states"]
    masks: List[MaskNet] = []
    for ms in mask_states:
        m = MaskNet(width=mask_kw["width"], depth=mask_kw["depth"]).to(device=device, dtype=dtype)
        m.load_state_dict(ms)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        masks.append(m)
    return masks


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_chart_points(
    chart_idx: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Randomly pick n_samples indices from chart_idx (with replacement if needed)."""
    n = chart_idx.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    replace = n_samples > n
    if device.type == "mps":
        sel = torch.randint(0, n, (n_samples,), device=device)
    else:
        sel = torch.randint(0, n, (n_samples,), device=device)
    return chart_idx[sel]


def sample_local_xi(
    points: torch.Tensor,
    chart_idx: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n_vec: torch.Tensor,
    support_r: float,
    n_samples: int,
    device: torch.device,
    noise_scale: float = 0.05,
) -> torch.Tensor:
    """Sample PDE collocation points for a chart from atlas membership."""
    pick = sample_chart_points(chart_idx, n_samples, device)
    if pick.numel() == 0:
        return torch.zeros((n_samples, 3), device=device, dtype=points.dtype)
    x = points[pick]
    xi = local_coords(x, seed, t1, t2, n_vec)
    noise = noise_scale * support_r * torch.randn_like(xi)
    xi = xi + noise
    max_abs = 1.25 * support_r
    xi = torch.clamp(xi, min=-max_abs, max=max_abs)
    return xi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rabbit Poisson with FEM-init guidance")
    parser.add_argument("--atlas-data", required=True, help="atlas_data.npz path")
    parser.add_argument("--atlas-checkpoint", required=True, help="atlas .pt checkpoint")
    parser.add_argument("--fem-solution", required=True, help="FEM solution .npz path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="fem_init")
    parser.add_argument("--max-schwarz-iters", type=int, default=120)
    parser.add_argument("--local-steps", type=int, default=30)
    parser.add_argument("--pde-batch", type=int, default=384)
    parser.add_argument("--bc-batch", type=int, default=256)
    parser.add_argument("--if-batch", type=int, default=192)
    parser.add_argument("--bc-pretrain-epochs", type=int, default=300)
    parser.add_argument("--fem-pretrain-epochs", type=int, default=500)
    parser.add_argument("--fem-guidance-start", type=float, default=1.0)
    parser.add_argument("--fem-guidance-end", type=float, default=0.0)
    parser.add_argument("--fem-guidance-decay-iters", type=int, default=60)
    parser.add_argument("--high-error-charts", default="4,6,0")
    parser.add_argument("--high-error-pde-batch", type=int, default=768)
    parser.add_argument("--w-pde", type=float, default=5.0)
    parser.add_argument("--w-bc", type=float, default=1.0)
    parser.add_argument("--w-interface-value", type=float, default=2.0)
    parser.add_argument("--w-interface-flux", type=float, default=2.0)
    parser.add_argument("--pde-warmup-iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plateau-patience", type=int, default=20)
    parser.add_argument("--trust-region-margin", type=float, default=0.5,
                        help="Reject sweep if rel-L2 increases by more than this fraction")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--xi-noise-scale", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device()
    dtype = torch.float32  # MPS-safe default
    print(f"Device: {device}, dtype: {dtype}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_stem = f"rabbit_poisson_{args.run_tag}"

    # ------------------------------------------------------------------
    # Load atlas
    # ------------------------------------------------------------------
    atlas_np = np.load(args.atlas_data)
    points = torch.tensor(atlas_np["points"], device=device, dtype=dtype)
    seeds = torch.tensor(atlas_np["seed_points"], device=device, dtype=dtype)
    t1 = torch.tensor(atlas_np["frame_t1"], device=device, dtype=dtype)
    t2 = torch.tensor(atlas_np["frame_t2"], device=device, dtype=dtype)
    nvec = torch.tensor(atlas_np["frame_n"], device=device, dtype=dtype)
    membership = torch.tensor(atlas_np["membership"].astype(np.int64), device=device, dtype=torch.int64)
    support_r = torch.tensor(atlas_np["support_radii"], device=device, dtype=dtype)

    n_points, n_charts = membership.shape
    print(f"Atlas: {n_charts} charts, {n_points} points")

    # Mask nets for blending at evaluation
    masks = load_mask_nets(args.atlas_checkpoint, device, dtype)

    # Chart membership indices
    point_idx_by_chart: List[torch.Tensor] = []
    for i in range(n_charts):
        idx = torch.where(membership[:, i] > 0)[0]
        point_idx_by_chart.append(idx)

    # Overlap / neighbor structure
    neighbors: List[List[int]] = [[] for _ in range(n_charts)]
    overlap_idx: Dict[Tuple[int, int], torch.Tensor] = {}
    for i in range(n_charts):
        mi = membership[:, i] > 0
        for j in range(i + 1, n_charts):
            mj = membership[:, j] > 0
            shared = torch.where(mi & mj)[0]
            if shared.numel() > 0:
                overlap_idx[(i, j)] = shared
                neighbors[i].append(j)
                neighbors[j].append(i)

    # Color groups (greedy graph coloring for Schwarz)
    adj = {i: set() for i in range(n_charts)}
    for (i, j) in overlap_idx:
        adj[i].add(j)
        adj[j].add(i)
    color: Dict[int, int] = {}
    for i in range(n_charts):
        used = {color[j] for j in adj[i] if j in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    n_colors = max(color.values()) + 1 if color else 1
    color_groups: List[List[int]] = [[] for _ in range(n_colors)]
    for i in range(n_charts):
        color_groups[color[i]].append(i)
    print(f"Color groups ({n_colors}): {[len(g) for g in color_groups]}")

    # High-error chart set
    high_error_set = set()
    if args.high_error_charts:
        for tok in args.high_error_charts.split(","):
            tok = tok.strip()
            if tok.isdigit():
                high_error_set.add(int(tok))
    print(f"High-error charts: {sorted(high_error_set)}")

    # ------------------------------------------------------------------
    # Load FEM solution & assign to charts
    # ------------------------------------------------------------------
    fem_np = np.load(args.fem_solution)
    fem_points_np = fem_np["points"]  # (N_fem, 3)
    fem_u_np = fem_np["u_pred"]       # (N_fem,)
    fem_points = torch.tensor(fem_points_np, device=device, dtype=dtype)
    fem_u = torch.tensor(fem_u_np, device=device, dtype=dtype)
    print(f"FEM solution: {fem_points.shape[0]} points")

    # Assign FEM points to charts by distance to seed within support radius
    fem_xi_by_chart: List[torch.Tensor] = []   # local coords of FEM points in chart i
    fem_u_by_chart: List[torch.Tensor] = []     # FEM u values for those points
    for i in range(n_charts):
        dist = torch.linalg.norm(fem_points - seeds[i].unsqueeze(0), dim=1)
        mask = dist < float(support_r[i].item())
        idx = torch.where(mask)[0]
        if idx.numel() > 0:
            x_chart = fem_points[idx]
            xi = local_coords(x_chart, seeds[i], t1[i], t2[i], nvec[i])
            fem_xi_by_chart.append(xi.detach())
            fem_u_by_chart.append(fem_u[idx].unsqueeze(-1).detach())
        else:
            fem_xi_by_chart.append(torch.zeros((0, 3), device=device, dtype=dtype))
            fem_u_by_chart.append(torch.zeros((0, 1), device=device, dtype=dtype))
        if idx.numel() > 0:
            print(f"  Chart {i}: {idx.numel()} FEM points")

    # ------------------------------------------------------------------
    # Build CompactChartNet per chart
    # ------------------------------------------------------------------
    u_nets = build_compact_u_nets(
        n_charts=n_charts,
        support_r=support_r,
        device=device,
        dtype=dtype,
        n_subseed=9,
        sub_width=32,
        sub_depth=2,
        tau_scale=0.125,
    )
    params_per_chart = u_nets[0].count_parameters()
    print(f"CompactChartNet: {params_per_chart:,d} params/chart")

    opts = [torch.optim.Adam(u_nets[i].parameters(), lr=args.lr) for i in range(n_charts)]
    scheds = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=args.plateau_patience, min_lr=1e-6,
        )
        for opt in opts
    ]

    # ------------------------------------------------------------------
    # Helper: sample BC batch for chart i
    # ------------------------------------------------------------------
    def local_bc_batch(i: int, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pick = sample_chart_points(point_idx_by_chart[i], n_samples, device)
        if pick.numel() == 0:
            z = torch.zeros((n_samples, 3), device=device, dtype=dtype)
            return z, torch.zeros((n_samples, 1), device=device, dtype=dtype), z
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        target = manufactured_u(x).detach()
        return xi, target, x

    # ------------------------------------------------------------------
    # Helper: interface batch
    # ------------------------------------------------------------------
    def interface_batch(
        i: int, j: int, n_samples: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        key = (i, j) if i < j else (j, i)
        shared = overlap_idx.get(key)
        if shared is None or shared.numel() == 0:
            return None
        take = min(n_samples, int(shared.numel()))
        sel = torch.randint(0, shared.numel(), (take,), device=device)
        pick = shared[sel]
        x = points[pick]
        xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j])
        n_seed = (seeds[j] - seeds[i]).unsqueeze(0).repeat(take, 1)
        n_norm = torch.linalg.norm(n_seed, dim=1, keepdim=True)
        n_if = n_seed / torch.clamp(n_norm, min=1e-12)
        return x, xi_i, xi_j, n_if

    # ------------------------------------------------------------------
    # Helper: sample FEM points for chart i
    # ------------------------------------------------------------------
    def sample_fem_batch(i: int, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (xi, u_fem) batch from FEM data assigned to chart i."""
        xi_all = fem_xi_by_chart[i]
        u_all = fem_u_by_chart[i]
        n_avail = xi_all.shape[0]
        if n_avail == 0:
            return (
                torch.zeros((0, 3), device=device, dtype=dtype),
                torch.zeros((0, 1), device=device, dtype=dtype),
            )
        take = min(n_samples, n_avail)
        sel = torch.randint(0, n_avail, (take,), device=device)
        return xi_all[sel], u_all[sel]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # Global eval subset (fixed 2000 random points)
    n_eval = min(2000, n_points)
    eval_perm = torch.randperm(n_points, device=device)[:n_eval]

    def eval_rel_l2() -> float:
        with torch.no_grad():
            x = points[eval_perm]
            logits = []
            vals = []
            for i in range(n_charts):
                xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
                logits.append(masks[i](xi, chart_scale=support_r[i]))
                vals.append(u_nets[i](xi).squeeze(-1))
            logits_t = torch.stack(logits, dim=1)
            weights = torch.softmax(logits_t, dim=1)
            vals_t = torch.stack(vals, dim=1)
            u_pred = torch.sum(weights * vals_t, dim=1, keepdim=True)
            u_true = manufactured_u(x)
            num = torch.mean((u_pred - u_true) ** 2)
            den = torch.mean(u_true ** 2)
            rel = torch.sqrt(num / torch.clamp(den, min=1e-12))
        return float(rel.item())

    def copy_states() -> List[Dict[str, torch.Tensor]]:
        return [{k: v.detach().clone() for k, v in net.state_dict().items()} for net in u_nets]

    def load_states(states: List[Dict[str, torch.Tensor]]) -> None:
        for i in range(n_charts):
            u_nets[i].load_state_dict(states[i])

    # ==================================================================
    # Phase 1: BC warm-start
    # ==================================================================
    print(f"\n=== Phase 1: BC warm-start ({args.bc_pretrain_epochs} epochs) ===")
    t0 = time.time()
    for ep in range(1, args.bc_pretrain_epochs + 1):
        losses = []
        for i in range(n_charts):
            if point_idx_by_chart[i].numel() == 0:
                continue
            u_nets[i].train()
            opts[i].zero_grad()
            xi_bc, u_bc, x_bc = local_bc_batch(i, args.bc_batch)
            if xi_bc.shape[0] == 0:
                continue
            u_hat = u_nets[i](xi_bc)
            loss = torch.mean((u_hat - u_bc) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=args.grad_clip)
            opts[i].step()
            losses.append(float(loss.item()))
        if ep % 50 == 0 and losses:
            rel = eval_rel_l2()
            print(f"  [BC] epoch={ep}/{args.bc_pretrain_epochs}  loss={np.mean(losses):.3e}  rel_L2={rel:.4e}")
    print(f"  Phase 1 done in {time.time() - t0:.1f}s  rel_L2={eval_rel_l2():.4e}")

    # ==================================================================
    # Phase 2: FEM pretraining
    # ==================================================================
    print(f"\n=== Phase 2: FEM pretraining ({args.fem_pretrain_epochs} epochs) ===")
    t0 = time.time()
    fem_batch_size = 512
    for ep in range(1, args.fem_pretrain_epochs + 1):
        losses = []
        for i in range(n_charts):
            xi_fem, u_fem = sample_fem_batch(i, fem_batch_size)
            if xi_fem.shape[0] == 0:
                continue
            u_nets[i].train()
            opts[i].zero_grad()
            u_hat = u_nets[i](xi_fem)
            loss_fem = torch.mean((u_hat - u_fem) ** 2)
            # BC retention to prevent catastrophic forgetting
            xi_bc, u_bc, _ = local_bc_batch(i, max(64, args.bc_batch // 4))
            if xi_bc.shape[0] > 0:
                loss_bc = torch.mean((u_nets[i](xi_bc) - u_bc) ** 2)
                loss = loss_fem + 0.2 * loss_bc
            else:
                loss = loss_fem
            loss.backward()
            torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=args.grad_clip)
            opts[i].step()
            losses.append(float(loss.item()))
        if ep % 100 == 0 and losses:
            rel = eval_rel_l2()
            print(f"  [FEM] epoch={ep}/{args.fem_pretrain_epochs}  loss={np.mean(losses):.3e}  rel_L2={rel:.4e}")
    print(f"  Phase 2 done in {time.time() - t0:.1f}s  rel_L2={eval_rel_l2():.4e}")

    # ==================================================================
    # Phase 3: Schwarz iterations with decaying FEM guidance
    # ==================================================================
    print(f"\n=== Phase 3: Schwarz iterations ({args.max_schwarz_iters} sweeps) ===")
    t0 = time.time()

    history: Dict[str, List[float]] = {
        "rel_l2": [],
        "pde_loss": [],
        "bc_loss": [],
        "iv_loss": [],
        "fem_weight": [],
        "rejected": [],
    }

    best_rel_l2 = float("inf")
    best_states: Optional[List[Dict[str, torch.Tensor]]] = None
    current_rel_l2 = eval_rel_l2()
    print(f"  Initial rel_L2 = {current_rel_l2:.4e}")

    for sweep in range(1, args.max_schwarz_iters + 1):
        # PDE weight warmup
        warm = min(1.0, float(sweep) / max(1.0, float(args.pde_warmup_iters)))
        w_pde_eff = args.w_pde * warm

        # FEM guidance weight (linear decay)
        if args.fem_guidance_decay_iters > 0:
            alpha = max(0.0, 1.0 - float(sweep) / float(args.fem_guidance_decay_iters))
        else:
            alpha = 0.0
        w_fem = args.fem_guidance_start * alpha + args.fem_guidance_end * (1.0 - alpha)

        # Save state for trust-region rollback
        states_before = copy_states()
        rel_l2_before = current_rel_l2

        sweep_pde_losses = []
        sweep_bc_losses = []
        sweep_iv_losses = []

        # Schwarz: iterate over color groups
        for group in color_groups:
            for i in group:
                if point_idx_by_chart[i].numel() == 0:
                    continue

                pde_batch_i = args.high_error_pde_batch if i in high_error_set else args.pde_batch
                u_nets[i].train()

                for _ in range(args.local_steps):
                    opts[i].zero_grad()

                    # --- PDE loss ---
                    if w_pde_eff > 0.0:
                        xi_pde = sample_local_xi(
                            points, point_idx_by_chart[i],
                            seeds[i], t1[i], t2[i], nvec[i],
                            float(support_r[i].item()), pde_batch_i, device,
                            noise_scale=args.xi_noise_scale,
                        )
                        x_phys = xi_to_physical(xi_pde, seeds[i], t1[i], t2[i], nvec[i])
                        res = direct_poisson_residual(
                            u_nets[i], x_phys, seeds[i], t1[i], t2[i], nvec[i],
                        )
                        loss_pde = torch.mean(res ** 2)
                    else:
                        loss_pde = torch.tensor(0.0, device=device, dtype=dtype)

                    # --- BC loss ---
                    xi_bc, u_bc, x_bc = local_bc_batch(i, args.bc_batch)
                    if xi_bc.shape[0] > 0:
                        loss_bc = torch.mean((u_nets[i](xi_bc) - u_bc) ** 2)
                    else:
                        loss_bc = torch.tensor(0.0, device=device, dtype=dtype)

                    # --- Interface losses ---
                    iv_terms: List[torch.Tensor] = []
                    if_terms: List[torch.Tensor] = []
                    for j in neighbors[i]:
                        ib = interface_batch(i, j, args.if_batch)
                        if ib is None:
                            continue
                        x_if, xi_i, xi_j, n_if = ib

                        # Value matching
                        ui = u_nets[i](xi_i)
                        with torch.no_grad():
                            uj = u_nets[j](xi_j)
                        iv_terms.append(torch.mean((ui - uj) ** 2))

                        # Flux matching (projected normal flux)
                        # Active chart: needs grad for backprop
                        grad_i = grad_u_physical_tnb(
                            u_nets[i], x_if, seeds[i], t1[i], t2[i], nvec[i],
                        )
                        # Frozen neighbor: use finite-diff approx or skip grad
                        # (autograd.grad doesn't work inside no_grad context)
                        xi_j_g = local_coords(x_if.detach(), seeds[j], t1[j], t2[j], nvec[j])
                        xi_j_g = xi_j_g.clone().detach().requires_grad_(True)
                        uj_g = u_nets[j](xi_j_g)
                        grad_xi_j = torch.autograd.grad(
                            uj_g, xi_j_g,
                            grad_outputs=torch.ones_like(uj_g),
                            create_graph=False,
                        )[0]
                        grad_j = (
                            grad_xi_j[:, 0:1] * t1[j].unsqueeze(0)
                            + grad_xi_j[:, 1:2] * t2[j].unsqueeze(0)
                            + grad_xi_j[:, 2:3] * nvec[j].unsqueeze(0)
                        ).detach()
                        flux_i = torch.sum(grad_i * n_if, dim=1, keepdim=True)
                        flux_j = torch.sum(grad_j * n_if, dim=1, keepdim=True)
                        if_terms.append(torch.mean((flux_i - flux_j) ** 2))

                    loss_iv = torch.mean(torch.stack(iv_terms)) if iv_terms else torch.tensor(0.0, device=device, dtype=dtype)
                    loss_if = torch.mean(torch.stack(if_terms)) if if_terms else torch.tensor(0.0, device=device, dtype=dtype)

                    # --- FEM guidance loss ---
                    if w_fem > 0.0:
                        xi_fem, u_fem = sample_fem_batch(i, 256)
                        if xi_fem.shape[0] > 0:
                            loss_fem = torch.mean((u_nets[i](xi_fem) - u_fem) ** 2)
                        else:
                            loss_fem = torch.tensor(0.0, device=device, dtype=dtype)
                    else:
                        loss_fem = torch.tensor(0.0, device=device, dtype=dtype)

                    # --- Total loss ---
                    loss = (
                        w_pde_eff * loss_pde
                        + args.w_bc * loss_bc
                        + args.w_interface_value * loss_iv
                        + args.w_interface_flux * loss_if
                        + w_fem * loss_fem
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=args.grad_clip)
                    opts[i].step()

                sweep_pde_losses.append(float(loss_pde.item()))
                sweep_bc_losses.append(float(loss_bc.item()))
                sweep_iv_losses.append(float(loss_iv.item()))

        # Evaluate after sweep
        current_rel_l2 = eval_rel_l2()
        rejected = 0

        # Trust-region rejection
        if rel_l2_before > 0 and (current_rel_l2 - rel_l2_before) / max(rel_l2_before, 1e-12) > args.trust_region_margin:
            # Reject: restore previous state
            load_states(states_before)
            current_rel_l2 = rel_l2_before
            rejected = 1
            print(f"  [REJECT] sweep={sweep}  proposed_rel_L2={eval_rel_l2():.4e} > threshold")

        # Update schedulers
        for sched in scheds:
            sched.step(current_rel_l2)

        # Track best
        if current_rel_l2 < best_rel_l2:
            best_rel_l2 = current_rel_l2
            best_states = copy_states()

        # Record history
        pde_m = np.mean(sweep_pde_losses) if sweep_pde_losses else 0.0
        bc_m = np.mean(sweep_bc_losses) if sweep_bc_losses else 0.0
        iv_m = np.mean(sweep_iv_losses) if sweep_iv_losses else 0.0
        history["rel_l2"].append(current_rel_l2)
        history["pde_loss"].append(pde_m)
        history["bc_loss"].append(bc_m)
        history["iv_loss"].append(iv_m)
        history["fem_weight"].append(w_fem)
        history["rejected"].append(rejected)

        if sweep % 5 == 0 or sweep <= 3:
            lr_now = opts[0].param_groups[0]["lr"]
            print(
                f"  sweep={sweep:3d}/{args.max_schwarz_iters}  "
                f"rel_L2={current_rel_l2:.4e}  best={best_rel_l2:.4e}  "
                f"pde={pde_m:.3e}  bc={bc_m:.3e}  iv={iv_m:.3e}  "
                f"w_fem={w_fem:.3f}  lr={lr_now:.2e}  rej={rejected}"
            )

    elapsed = time.time() - t0
    print(f"\n  Phase 3 done in {elapsed:.1f}s")
    print(f"  Best rel_L2 = {best_rel_l2:.4e}")

    # Restore best
    if best_states is not None:
        load_states(best_states)
    final_rel_l2 = eval_rel_l2()
    print(f"  Final rel_L2 (best restored) = {final_rel_l2:.4e}")

    # ==================================================================
    # Full evaluation on all atlas points
    # ==================================================================
    print("\n=== Final evaluation ===")
    with torch.no_grad():
        logits_all = []
        vals_all = []
        for i in range(n_charts):
            xi = local_coords(points, seeds[i], t1[i], t2[i], nvec[i])
            logits_all.append(masks[i](xi, chart_scale=support_r[i]))
            vals_all.append(u_nets[i](xi).squeeze(-1))
        logits_t = torch.stack(logits_all, dim=1)
        weights = torch.softmax(logits_t, dim=1)
        vals_t = torch.stack(vals_all, dim=1)
        u_pred_all = torch.sum(weights * vals_t, dim=1).cpu().numpy()
        u_true_all = manufactured_u(points).squeeze(-1).cpu().numpy()

    err = u_pred_all - u_true_all
    l2_err = float(np.sqrt(np.mean(err ** 2)))
    rel_l2 = float(np.sqrt(np.mean(err ** 2) / max(np.mean(u_true_all ** 2), 1e-12)))
    max_err = float(np.max(np.abs(err)))
    print(f"  L2 error:     {l2_err:.6e}")
    print(f"  Relative L2:  {rel_l2:.6e}")
    print(f"  Max error:    {max_err:.6e}")

    # ==================================================================
    # Save outputs
    # ==================================================================
    # Metrics JSON
    metrics = {
        "run_tag": args.run_tag,
        "n_charts": n_charts,
        "n_points": n_points,
        "l2_error": l2_err,
        "relative_l2_error": rel_l2,
        "max_error": max_err,
        "best_sweep_rel_l2": best_rel_l2,
        "total_schwarz_sweeps": args.max_schwarz_iters,
        "bc_pretrain_epochs": args.bc_pretrain_epochs,
        "fem_pretrain_epochs": args.fem_pretrain_epochs,
        "history": history,
    }
    metrics_path = os.path.join(args.output_dir, f"{run_stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    # Checkpoint
    ckpt_path = os.path.join(args.output_dir, f"{run_stem}_checkpoint.pt")
    torch.save(
        {
            "u_states": [net.state_dict() for net in u_nets],
            "u_kwargs": {
                "arch": "compact",
                "n_subseed": 9,
                "sub_width": 32,
                "sub_depth": 2,
                "tau_scale": 0.125,
            },
            "best_rel_l2": best_rel_l2,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")

    # Solution NPZ
    sol_path = os.path.join(args.output_dir, f"{run_stem}_solution.npz")
    np.savez(
        sol_path,
        points=points.cpu().numpy(),
        u_pred=u_pred_all,
        u_true=u_true_all,
        u_error=err,
    )
    print(f"  Saved solution: {sol_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
