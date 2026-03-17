#!/usr/bin/env python3
"""
Run meshfree Poisson solve on a rabbit atlas with multiplicative alternating Schwarz.

Inputs:
- atlas build output (.npz + meta.json)
- atlas training checkpoint (chart decoders + mask nets)

Outputs:
- Schwarz checkpoint and metrics
- solution fields on canonical rabbit points
- training curve figure
"""

import argparse
import contextlib
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Requested --device cuda but CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested --device mps but MPS is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_arg}")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type in ("cuda", "mps"):
            return torch.float32
        return torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS backend does not support float64 well; use --dtype float32 or auto.")
        return torch.float64
    raise ValueError(f"Unsupported dtype option: {dtype_arg}")


def build_run_stem(run_tag: str) -> str:
    if run_tag is None:
        run_tag = ""
    run_tag = run_tag.strip()
    if len(run_tag) == 0:
        return "rabbit_poisson_schwarz"
    return f"rabbit_poisson_schwarz_{run_tag}"


def normalize_rows_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / torch.clamp(n, min=eps)


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


class ChartDecoder(torch.nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)
        self.raw_scale = torch.nn.Parameter(torch.tensor(-1.8, dtype=torch.float64))

    def forward(
        self,
        xi: torch.Tensor,
        seed: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        n: torch.Tensor,
        chart_scale: torch.Tensor,
    ) -> torch.Tensor:
        base = (
            seed.unsqueeze(0)
            + xi[:, 0:1] * t1.unsqueeze(0)
            + xi[:, 1:2] * t2.unsqueeze(0)
            + xi[:, 2:3] * n.unsqueeze(0)
        )
        xi_n = xi / torch.clamp(chart_scale, min=1e-6)
        amp = 0.20 * torch.tanh(self.raw_scale)
        res = amp * torch.clamp(chart_scale, min=1e-6) * self.net(xi_n)
        return base + res


class MaskNet(torch.nn.Module):
    def __init__(self, width: int = 48, depth: int = 3):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, xi: torch.Tensor, chart_scale: torch.Tensor) -> torch.Tensor:
        xi_n = xi / torch.clamp(chart_scale, min=1e-6)
        return self.net(xi_n).squeeze(-1)


class LocalPoissonPINN(torch.nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.net(xi)


def local_coords(
    x: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    d = x - seed.unsqueeze(0)
    return torch.stack(
        [
            torch.sum(d * t1.unsqueeze(0), dim=1),
            torch.sum(d * t2.unsqueeze(0), dim=1),
            torch.sum(d * n.unsqueeze(0), dim=1),
        ],
        dim=1,
    )


def chart_map_and_jacobian(
    decoder: ChartDecoder,
    xi_in: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xi = xi_in.clone().detach().requires_grad_(True)
    x = decoder(xi, seed=seed, t1=t1, t2=t2, n=n, chart_scale=chart_scale)
    grads = []
    for i in range(3):
        gi = torch.autograd.grad(
            x[:, i],
            xi,
            grad_outputs=torch.ones_like(x[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(gi)
    jac = torch.stack(grads, dim=1)
    return x, xi, jac


def manufactured_u(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(math.pi * x[:, 0:1])
        * torch.sin(math.pi * x[:, 1:2])
        * torch.sin(math.pi * x[:, 2:3])
    )


def manufactured_grad_u(x: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    du1 = pi * torch.cos(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    du2 = pi * torch.sin(pi * x1) * torch.cos(pi * x2) * torch.sin(pi * x3)
    du3 = pi * torch.sin(pi * x1) * torch.sin(pi * x2) * torch.cos(pi * x3)
    return torch.cat([du1, du2, du3], dim=1)


def forcing_f(x: torch.Tensor) -> torch.Tensor:
    return 3.0 * (math.pi**2) * manufactured_u(x)


def stabilized_jacobian_ops(
    jac: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(jac)
    s_safe = torch.clamp(s, min=sigma_floor)
    inv_s = torch.diag_embed(1.0 / s_safe)
    inv_j = torch.bmm(vh.transpose(1, 2), torch.bmm(inv_s, u.transpose(1, 2)))

    raw_det_abs = torch.abs(torch.det(jac))
    det_abs = torch.clamp(raw_det_abs, min=det_floor)
    kappa = s_safe[:, 0] / torch.clamp(s_safe[:, -1], min=sigma_floor)
    valid = raw_det_abs > det_floor
    valid = valid & torch.isfinite(kappa) & torch.isfinite(det_abs)
    return inv_j, det_abs, kappa, valid


def mapped_poisson_residual(
    u_model: LocalPoissonPINN,
    decoder: ChartDecoder,
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
    jac_kappa_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, xi_var, jac = chart_map_and_jacobian(
        decoder,
        xi,
        seed=seed,
        t1=t1,
        t2=t2,
        n=n,
        chart_scale=chart_scale,
    )
    inv_j, det_abs, kappa, valid = stabilized_jacobian_ops(
        jac=jac,
        sigma_floor=sigma_floor,
        det_floor=det_floor,
    )
    valid = valid & (kappa <= jac_kappa_max)
    a = det_abs.unsqueeze(-1).unsqueeze(-1) * (inv_j @ inv_j.transpose(1, 2))

    u = u_model(xi_var)
    grad_u = torch.autograd.grad(
        u,
        xi_var,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    flux = torch.bmm(a, grad_u.unsqueeze(-1)).squeeze(-1)
    div_flux = torch.zeros_like(u)
    for j in range(3):
        dflux_j = torch.autograd.grad(
            flux[:, j],
            xi_var,
            grad_outputs=torch.ones_like(flux[:, j]),
            create_graph=True,
            retain_graph=True,
        )[0][:, j : j + 1]
        div_flux = div_flux + dflux_j

    rhs = det_abs.unsqueeze(-1) * forcing_f(x)
    residual = -div_flux - rhs
    return residual, x, valid


def grad_u_in_physical(
    u_model: LocalPoissonPINN,
    decoder: ChartDecoder,
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
) -> torch.Tensor:
    x, xi_var, jac = chart_map_and_jacobian(
        decoder,
        xi,
        seed=seed,
        t1=t1,
        t2=t2,
        n=n,
        chart_scale=chart_scale,
    )
    _ = x
    u = u_model(xi_var)
    grad_xi = torch.autograd.grad(
        u,
        xi_var,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]
    inv_j, _, _, _ = stabilized_jacobian_ops(
        jac=jac,
        sigma_floor=sigma_floor,
        det_floor=det_floor,
    )
    grad_x = torch.bmm(inv_j.transpose(1, 2), grad_xi.unsqueeze(-1)).squeeze(-1)
    return grad_x


def copy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def blend_model_with_old(model: torch.nn.Module, old_state: Dict[str, torch.Tensor], omega: float) -> None:
    new_state = model.state_dict()
    blended = {}
    for k, v in new_state.items():
        old_v = old_state[k]
        blended[k] = (1.0 - omega) * old_v + omega * v
    model.load_state_dict(blended)


def choose_color_groups(meta_json: Optional[str], n_charts: int, membership_np: np.ndarray) -> List[List[int]]:
    if meta_json is not None and os.path.isfile(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        groups = meta.get("color_groups")
        if isinstance(groups, list) and len(groups) > 0:
            out = []
            seen = set()
            for g in groups:
                gi = []
                for x in g:
                    ix = int(x)
                    if 0 <= ix < n_charts:
                        gi.append(ix)
                        seen.add(ix)
                if gi:
                    out.append(sorted(set(gi)))
            missing = [i for i in range(n_charts) if i not in seen]
            if missing:
                out.append(missing)
            return out

    # Fallback greedy coloring from overlap membership.
    adj = {i: set() for i in range(n_charts)}
    for i in range(n_charts):
        mi = membership_np[:, i].astype(bool)
        for j in range(i + 1, n_charts):
            mj = membership_np[:, j].astype(bool)
            shared = int(np.sum(mi & mj))
            if shared > 0:
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
    groups = [[] for _ in range(n_colors)]
    for i in range(n_charts):
        groups[color[i]].append(i)
    return groups


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    iters = np.arange(1, len(history["global_residual"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.semilogy(iters, np.maximum(history["global_residual"], 1e-16), label="global_residual")
    ax.semilogy(iters, np.maximum(history["interface_value"], 1e-16), label="interface_value")
    ax.semilogy(iters, np.maximum(history["interface_flux"], 1e-16), label="interface_flux")
    ax.semilogy(iters, np.maximum(history["bc_loss"], 1e-16), label="bc_loss")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Metric")
    ax.set_title("Atlas Schwarz Poisson convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def load_atlas_models(
    atlas_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[ChartDecoder], List[MaskNet], Dict[str, object]]:
    ckpt = torch.load(atlas_checkpoint, map_location=device)

    dec_kw = ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kw = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    dec_states = ckpt["decoder_states"]
    mask_states = ckpt["mask_states"]

    decoders: List[ChartDecoder] = []
    masks: List[MaskNet] = []
    for ds, ms in zip(dec_states, mask_states):
        d = ChartDecoder(width=dec_kw["width"], depth=dec_kw["depth"]).to(device=device, dtype=dtype)
        m = MaskNet(width=mask_kw["width"], depth=mask_kw["depth"]).to(device=device, dtype=dtype)
        d.load_state_dict(ds)
        m.load_state_dict(ms)
        d.eval()
        m.eval()
        for p in d.parameters():
            p.requires_grad_(False)
        for p in m.parameters():
            p.requires_grad_(False)
        decoders.append(d)
        masks.append(m)

    return decoders, masks, ckpt


def metric_l2(u_pred: np.ndarray, u_true: np.ndarray) -> Dict[str, float]:
    err = u_pred - u_true
    l2 = float(np.sqrt(np.mean(err**2)))
    rel = float(np.sqrt(np.mean(err**2) / max(np.mean(u_true**2), 1e-12)))
    max_e = float(np.max(np.abs(err)))
    return {
        "l2_error": l2,
        "relative_l2_error": rel,
        "max_error": max_e,
    }


def train_schwarz(args: argparse.Namespace) -> Dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    elif args.tf32:
        print("TF32 requested but CUDA is unavailable; ignoring --tf32.")

    use_amp = bool(args.amp and device.type == "cuda" and dtype == torch.float32)
    if args.amp and not use_amp:
        print(
            "AMP requested but unavailable for this device/dtype; "
            "continuing without AMP."
        )

    use_cuda_stream_parallel = bool(
        args.parallel_color_updates and device.type == "cuda" and torch.cuda.device_count() > 0
    )
    if args.parallel_color_updates and not use_cuda_stream_parallel:
        print(
            "parallel_color_updates requested but CUDA streams unavailable on this host; "
            "falling back to sequential color sweeps."
        )

    print(
        f"Device={device.type} dtype={dtype} amp={use_amp} tf32={bool(args.tf32 and device.type == 'cuda')} "
        f"parallel_color_updates={use_cuda_stream_parallel}"
    )

    atlas_np = np.load(args.atlas_data)
    points = torch.tensor(atlas_np["points"], device=device, dtype=dtype)
    normals = torch.tensor(atlas_np["normals"], device=device, dtype=dtype)
    seeds = torch.tensor(atlas_np["seed_points"], device=device, dtype=dtype)
    t1 = torch.tensor(atlas_np["frame_t1"], device=device, dtype=dtype)
    t2 = torch.tensor(atlas_np["frame_t2"], device=device, dtype=dtype)
    nvec = torch.tensor(atlas_np["frame_n"], device=device, dtype=dtype)
    membership = torch.tensor(atlas_np["membership"].astype(np.int64), device=device, dtype=torch.int64)
    support_r = torch.tensor(atlas_np["support_radii"], device=device, dtype=dtype)

    n_points, n_charts = membership.shape

    decoders, masks, atlas_ckpt = load_atlas_models(
        atlas_checkpoint=args.atlas_checkpoint,
        device=device,
        dtype=dtype,
    )

    gate = atlas_ckpt.get("gate")
    if isinstance(gate, dict) and (not args.allow_failed_gate) and not bool(gate.get("passed", False)):
        raise RuntimeError(
            "Atlas gate check failed. Refusing to run Poisson Schwarz solve. "
            f"Checkpoint: {args.atlas_checkpoint}"
        )

    color_groups = choose_color_groups(
        meta_json=args.atlas_meta,
        n_charts=n_charts,
        membership_np=atlas_np["membership"],
    )

    point_idx_by_chart: List[torch.Tensor] = []
    for i in range(n_charts):
        idx = torch.where(membership[:, i] > 0)[0]
        point_idx_by_chart.append(idx)

    overlap_idx_pairs: Dict[Tuple[int, int], torch.Tensor] = {}
    neighbors: List[List[int]] = [[] for _ in range(n_charts)]
    for i in range(n_charts):
        mi = membership[:, i] > 0
        for j in range(i + 1, n_charts):
            mj = membership[:, j] > 0
            shared = torch.where(mi & mj)[0]
            if shared.numel() > 0:
                overlap_idx_pairs[(i, j)] = shared
                neighbors[i].append(j)
                neighbors[j].append(i)

    def mask_interface_normals(i: int, j: int, x: torch.Tensor) -> torch.Tensor:
        x_var = x.clone().detach().requires_grad_(True)
        xi_i = local_coords(x_var, seeds[i], t1[i], t2[i], nvec[i])
        xi_j = local_coords(x_var, seeds[j], t1[j], t2[j], nvec[j])
        li = masks[i](xi_i, chart_scale=support_r[i])
        lj = masks[j](xi_j, chart_scale=support_r[j])
        phi = li - lj
        g = torch.autograd.grad(
            phi,
            x_var,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=False,
        )[0]
        gnorm = torch.linalg.norm(g, dim=1, keepdim=True)
        seed_dir = (seeds[j] - seeds[i]).unsqueeze(0).repeat(x.shape[0], 1)
        seed_dir = normalize_rows_tensor(seed_dir)
        n = g / torch.clamp(gnorm, min=args.interface_normal_eps)
        if args.interface_normal_blend > 0.0:
            n = normalize_rows_tensor(
                (1.0 - args.interface_normal_blend) * n + args.interface_normal_blend * seed_dir,
                eps=args.interface_normal_eps,
            )
        bad = gnorm.squeeze(-1) < args.interface_normal_eps
        if torch.any(bad):
            n[bad] = seed_dir[bad]
        return n.detach()

    overlap_normals: Dict[Tuple[int, int], torch.Tensor] = {}
    if args.interface_flux_mode == "projected" and args.interface_normal_mode == "mask_levelset":
        print("Precomputing interface normals from mask level-set gradients")
        chunk = max(64, int(args.interface_normal_cache_batch))
        with torch.enable_grad():
            for (i, j), shared in overlap_idx_pairs.items():
                x_all = points[shared]
                n_parts = []
                for s in range(0, x_all.shape[0], chunk):
                    e = min(x_all.shape[0], s + chunk)
                    n_parts.append(mask_interface_normals(i, j, x_all[s:e]))
                overlap_normals[(i, j)] = torch.cat(n_parts, dim=0)

    u_nets = [
        LocalPoissonPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype)
        for _ in range(n_charts)
    ]
    if args.init_u_checkpoint is not None:
        init_ckpt = torch.load(args.init_u_checkpoint, map_location=device)
        init_states = init_ckpt.get("u_states")
        if not isinstance(init_states, list) or len(init_states) != n_charts:
            raise RuntimeError(
                f"Invalid u_states in init checkpoint: {args.init_u_checkpoint}. "
                f"Expected {n_charts} chart states."
            )
        for i in range(n_charts):
            u_nets[i].load_state_dict(init_states[i])
        print(f"Loaded initial chart PINNs from: {args.init_u_checkpoint}")

    opts = [torch.optim.Adam(u.parameters(), lr=args.lr) for u in u_nets]
    scalers = [torch.cuda.amp.GradScaler(enabled=use_amp) for _ in range(n_charts)]
    amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext
    stream_pool: List[torch.cuda.Stream] = []
    if use_cuda_stream_parallel:
        n_streams = max(1, min(args.stream_pool_size, args.max_parallel_charts))
        stream_pool = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
        print(f"Using persistent CUDA stream pool with {len(stream_pool)} streams")

    history: Dict[str, List[float]] = {
        "global_residual": [],
        "interface_value": [],
        "interface_flux": [],
        "bc_loss": [],
        "rel_l2_eval": [],
        "w_interface_flux_eff": [],
    }

    def sample_local_xi(i: int, n_samples: int) -> torch.Tensor:
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            return torch.zeros((n_samples, 3), device=device, dtype=dtype)
        pick = idx[torch.randint(0, idx.numel(), (n_samples,), device=device)]
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        noise = args.xi_noise_scale * support_r[i] * torch.randn_like(xi)
        xi = xi + noise
        max_abs = 1.25 * support_r[i]
        xi = torch.clamp(xi, min=-max_abs, max=max_abs)
        return xi

    def local_bc_batch(i: int, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            z = torch.zeros((n_samples, 3), device=device, dtype=dtype)
            return z, torch.zeros((n_samples, 1), device=device, dtype=dtype), z
        pick = idx[torch.randint(0, idx.numel(), (n_samples,), device=device)]
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        target = manufactured_u(x).detach()
        return xi, target, x

    def interface_batch(
        i: int,
        j: int,
        n_samples: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        key = (i, j) if i < j else (j, i)
        shared = overlap_idx_pairs.get(key)
        if shared is None or shared.numel() == 0:
            return None
        take = min(n_samples, int(shared.numel()))
        sel = torch.randint(0, shared.numel(), (take,), device=device)
        pick = shared[sel]
        x = points[pick]
        xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j])
        if args.interface_flux_mode == "projected" and args.interface_normal_mode == "mask_levelset":
            n_if = overlap_normals[key][sel]
        else:
            n_seed = (seeds[j] - seeds[i]).unsqueeze(0).repeat(take, 1)
            n_if = normalize_rows_tensor(n_seed, eps=args.interface_normal_eps)
        return x, xi_i, xi_j, n_if

    # Deterministic eval cache for stable model selection and stopping.
    eval_rng = np.random.default_rng(args.eval_cache_seed)
    eval_cache_pde: Dict[int, torch.Tensor] = {}
    eval_cache_bc: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    eval_cache_if: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def fixed_pick_from_idx(idx: torch.Tensor, n_samples: int) -> Optional[torch.Tensor]:
        if idx.numel() == 0 or n_samples <= 0:
            return None
        n_take = min(int(n_samples), int(idx.numel()))
        base = idx.detach().cpu().numpy()
        sel_np = eval_rng.choice(base, size=n_take, replace=False)
        sel = torch.tensor(sel_np, device=device, dtype=torch.int64)
        return sel

    if args.eval_fixed_cache:
        for i in range(n_charts):
            idx = point_idx_by_chart[i]
            pick = fixed_pick_from_idx(idx, max(16, int(args.eval_cache_per_chart)))
            if pick is None:
                continue
            x = points[pick]
            xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
            if args.xi_noise_scale > 0.0:
                noise_np = eval_rng.standard_normal(size=tuple(xi.shape))
                noise = torch.tensor(noise_np, device=device, dtype=dtype)
                xi = xi + args.xi_noise_scale * support_r[i] * noise
                max_abs = 1.25 * support_r[i]
                xi = torch.clamp(xi, min=-max_abs, max=max_abs)
            eval_cache_pde[i] = xi.detach()
            eval_cache_bc[i] = (local_coords(x, seeds[i], t1[i], t2[i], nvec[i]).detach(), manufactured_u(x).detach())

        for (i, j), shared in overlap_idx_pairs.items():
            n_take = min(int(shared.numel()), max(8, int(args.eval_cache_per_overlap)))
            if n_take <= 0:
                continue
            sel_np = eval_rng.choice(np.arange(int(shared.numel())), size=n_take, replace=False)
            sel = torch.tensor(sel_np, device=device, dtype=torch.int64)
            pick = shared[sel]
            x = points[pick]
            xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i]).detach()
            xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j]).detach()
            if args.interface_flux_mode == "projected" and args.interface_normal_mode == "mask_levelset":
                n_if = overlap_normals[(i, j)][sel].detach()
            else:
                n_seed = (seeds[j] - seeds[i]).unsqueeze(0).repeat(n_take, 1)
                n_if = normalize_rows_tensor(n_seed, eps=args.interface_normal_eps).detach()
            eval_cache_if[(i, j)] = (xi_i, xi_j, n_if)

    global_eval_count = min(n_points, max(1024, int(args.eval_cache_per_chart) * n_charts * 4))
    global_eval_idx_np = eval_rng.choice(np.arange(n_points), size=global_eval_count, replace=False)
    global_eval_idx = torch.tensor(global_eval_idx_np, device=device, dtype=torch.int64)

    def select_residual_samples(residual: torch.Tensor, valid: torch.Tensor, with_clip: bool) -> torch.Tensor:
        if torch.any(valid):
            res = residual[valid]
        else:
            res = residual
        if with_clip and args.pde_clip_quantile < 1.0 and res.numel() >= 8:
            q = float(torch.quantile(torch.abs(res.detach()).reshape(-1), args.pde_clip_quantile).item())
            q = max(q, 1e-8)
            res = torch.clamp(res, min=-q, max=q)
        return res

    def pde_loss_fn(residual_values: torch.Tensor) -> torch.Tensor:
        if args.pde_huber_delta > 0.0:
            return torch.nn.functional.huber_loss(
                residual_values,
                torch.zeros_like(residual_values),
                delta=args.pde_huber_delta,
            )
        return torch.mean(residual_values**2)

    def interface_coupling_terms(
        i: int,
        j: int,
        xi_i: torch.Tensor,
        xi_j: torch.Tensor,
        n_if: torch.Tensor,
        detach_neighbor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ui = u_nets[i](xi_i)
        if detach_neighbor:
            with torch.no_grad():
                uj = u_nets[j](xi_j)
        else:
            uj = u_nets[j](xi_j)
        du = ui - uj
        loss_iv = torch.mean(du**2)

        gxi = grad_u_in_physical(
            u_nets[i],
            decoders[i],
            xi_i,
            seed=seeds[i],
            t1=t1[i],
            t2=t2[i],
            n=nvec[i],
            chart_scale=support_r[i],
            sigma_floor=args.sigma_floor,
            det_floor=args.det_floor,
        )
        gxj = grad_u_in_physical(
            u_nets[j],
            decoders[j],
            xi_j,
            seed=seeds[j],
            t1=t1[j],
            t2=t2[j],
            n=nvec[j],
            chart_scale=support_r[j],
            sigma_floor=args.sigma_floor,
            det_floor=args.det_floor,
        )
        if detach_neighbor:
            gxj = gxj.detach()

        if args.interface_flux_mode == "vector":
            dq_vec = gxi - gxj
            loss_if_metric = torch.mean(dq_vec**2)
            fi = torch.sum(gxi * n_if, dim=1, keepdim=True)
            fj = torch.sum(gxj * n_if, dim=1, keepdim=True)
        else:
            fi = torch.sum(gxi * n_if, dim=1, keepdim=True)
            fj = torch.sum(gxj * n_if, dim=1, keepdim=True)
            loss_if_metric = torch.mean((fi - fj) ** 2)

        if args.interface_transmission_mode == "robin":
            robin_res = args.robin_lambda * du + (fi - fj)
            loss_if_train = torch.mean(robin_res**2)
        else:
            loss_if_train = loss_if_metric

        return loss_iv, loss_if_metric, loss_if_train

    def eval_rel_l2_subset() -> float:
        with torch.no_grad():
            x = points[global_eval_idx]
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
            den = torch.mean(u_true**2)
            rel = torch.sqrt(num / torch.clamp(den, min=1e-12))
        return float(rel.item())

    def eval_global_metrics() -> Tuple[float, float, float, float]:
        with torch.enable_grad():
            pde_terms = []
            bc_terms = []
            iv_terms = []
            if_terms = []
            for i in range(n_charts):
                if args.eval_fixed_cache and i in eval_cache_pde:
                    xi_int = eval_cache_pde[i]
                else:
                    ni = max(16, args.eval_pde_samples_per_chart)
                    xi_int = sample_local_xi(i, ni)
                res, _, valid = mapped_poisson_residual(
                    u_nets[i],
                    decoders[i],
                    xi_int,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                    sigma_floor=args.sigma_floor,
                    det_floor=args.det_floor,
                    jac_kappa_max=args.jac_kappa_max,
                )
                res_eval = select_residual_samples(res, valid, with_clip=False)
                pde_terms.append(torch.mean(res_eval**2))

                if args.eval_fixed_cache and i in eval_cache_bc:
                    xi_bc, u_bc = eval_cache_bc[i]
                else:
                    xi_bc, u_bc, _ = local_bc_batch(i, max(16, args.eval_bc_samples_per_chart))
                u_hat = u_nets[i](xi_bc)
                bc_terms.append(torch.mean((u_hat - u_bc) ** 2))

                for j in neighbors[i]:
                    if j <= i:
                        continue
                    if args.eval_fixed_cache and (i, j) in eval_cache_if:
                        xi_i, xi_j, n_if = eval_cache_if[(i, j)]
                    else:
                        ib = interface_batch(i, j, max(8, args.eval_if_samples))
                        if ib is None:
                            continue
                        _, xi_i, xi_j, n_if = ib
                    liv, lif_metric, _ = interface_coupling_terms(i, j, xi_i, xi_j, n_if, detach_neighbor=False)
                    iv_terms.append(liv)
                    if_terms.append(lif_metric)

            pde = float(torch.mean(torch.stack(pde_terms)).item()) if pde_terms else 0.0
            bc = float(torch.mean(torch.stack(bc_terms)).item()) if bc_terms else 0.0
            iv = float(torch.mean(torch.stack(iv_terms)).item()) if iv_terms else 0.0
            iflux = float(torch.mean(torch.stack(if_terms)).item()) if if_terms else 0.0
        return pde, bc, iv, iflux

    if args.bc_pretrain_epochs > 0:
        print(f"Starting BC warm-start pretraining for {args.bc_pretrain_epochs} epochs")
        for ep in range(1, args.bc_pretrain_epochs + 1):
            losses_ep = []
            for i in range(n_charts):
                if point_idx_by_chart[i].numel() == 0:
                    continue
                u_nets[i].train()
                opts[i].zero_grad()

                with amp_ctx():
                    xi_bc, u_bc, x_bc = local_bc_batch(i, args.bc_pretrain_batch)
                    u_hat = u_nets[i](xi_bc)
                    loss = torch.mean((u_hat - u_bc) ** 2)

                    if args.bc_pretrain_grad_weight > 0.0:
                        grad_pred = grad_u_in_physical(
                            u_nets[i],
                            decoders[i],
                            xi_bc,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                            sigma_floor=args.sigma_floor,
                            det_floor=args.det_floor,
                        )
                        grad_true = manufactured_grad_u(x_bc)
                        loss = loss + args.bc_pretrain_grad_weight * torch.mean((grad_pred - grad_true) ** 2)

                    if args.bc_pretrain_interface_weight > 0.0:
                        iv_terms = []
                        for j in neighbors[i]:
                            ib = interface_batch(i, j, max(16, args.if_batch // 2))
                            if ib is None:
                                continue
                            _, xi_i, xi_j, n_if = ib
                            liv, _, _ = interface_coupling_terms(i, j, xi_i, xi_j, n_if, detach_neighbor=True)
                            iv_terms.append(liv)
                        if iv_terms:
                            loss = loss + args.bc_pretrain_interface_weight * torch.mean(torch.stack(iv_terms))

                if use_amp:
                    scalers[i].scale(loss).backward()
                    scalers[i].unscale_(opts[i])
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                    scalers[i].step(opts[i])
                    scalers[i].update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                    opts[i].step()
                losses_ep.append(float(loss.item()))

            if ep % max(1, args.bc_pretrain_log_every) == 0 and losses_ep:
                print(
                    f"[Pretrain] epoch={ep}/{args.bc_pretrain_epochs} "
                    f"loss={np.mean(losses_ep):.3e}"
                )

    if args.interior_pretrain_epochs > 0:
        print(f"Starting interior supervised pretraining for {args.interior_pretrain_epochs} epochs")
        for ep in range(1, args.interior_pretrain_epochs + 1):
            losses_ep = []
            for i in range(n_charts):
                if point_idx_by_chart[i].numel() == 0:
                    continue

                u_nets[i].train()
                opts[i].zero_grad()

                with amp_ctx():
                    xi_sup = sample_local_xi(i, args.interior_pretrain_batch)
                    x_sup = decoders[i](
                        xi_sup,
                        seed=seeds[i],
                        t1=t1[i],
                        t2=t2[i],
                        n=nvec[i],
                        chart_scale=support_r[i],
                    )
                    u_true_sup = manufactured_u(x_sup).detach()
                    u_pred_sup = u_nets[i](xi_sup)
                    loss = torch.mean((u_pred_sup - u_true_sup) ** 2)

                    if args.interior_pretrain_grad_weight > 0.0:
                        grad_pred_sup = grad_u_in_physical(
                            u_nets[i],
                            decoders[i],
                            xi_sup,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                            sigma_floor=args.sigma_floor,
                            det_floor=args.det_floor,
                        )
                        grad_true_sup = manufactured_grad_u(x_sup).detach()
                        loss = loss + args.interior_pretrain_grad_weight * torch.mean((grad_pred_sup - grad_true_sup) ** 2)

                if use_amp:
                    scalers[i].scale(loss).backward()
                    scalers[i].unscale_(opts[i])
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                    scalers[i].step(opts[i])
                    scalers[i].update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                    opts[i].step()
                losses_ep.append(float(loss.item()))

            if ep % max(1, args.interior_pretrain_log_every) == 0 and losses_ep:
                print(
                    f"[InteriorPre] epoch={ep}/{args.interior_pretrain_epochs} "
                    f"loss={np.mean(losses_ep):.3e}"
                )

    def optimize_chart(i: int, w_pde_eff: float, w_if_flux_eff: float) -> None:
        if point_idx_by_chart[i].numel() == 0:
            return
        u_nets[i].train()
        old_state = copy_state_dict(u_nets[i].state_dict())

        for _ in range(args.local_steps):
            opts[i].zero_grad()

            with amp_ctx():
                xi_int = sample_local_xi(i, args.pde_batch)
                res, _, valid = mapped_poisson_residual(
                    u_nets[i],
                    decoders[i],
                    xi_int,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                    sigma_floor=args.sigma_floor,
                    det_floor=args.det_floor,
                    jac_kappa_max=args.jac_kappa_max,
                )
                res_use = select_residual_samples(res, valid, with_clip=True)
                loss_pde = pde_loss_fn(res_use)

                xi_bc, u_bc, _ = local_bc_batch(i, args.bc_batch)
                u_hat_bc = u_nets[i](xi_bc)
                loss_bc = torch.mean((u_hat_bc - u_bc) ** 2)

                loss_sup = torch.tensor(0.0, device=device, dtype=dtype)
                loss_sup_grad = torch.tensor(0.0, device=device, dtype=dtype)
                if args.w_manufactured_supervision > 0.0 or args.w_manufactured_grad_supervision > 0.0:
                    xi_sup = sample_local_xi(i, args.manufactured_supervision_batch)
                    x_sup = decoders[i](
                        xi_sup,
                        seed=seeds[i],
                        t1=t1[i],
                        t2=t2[i],
                        n=nvec[i],
                        chart_scale=support_r[i],
                    )
                    u_sup_true = manufactured_u(x_sup).detach()
                    u_sup_pred = u_nets[i](xi_sup)
                    loss_sup = torch.mean((u_sup_pred - u_sup_true) ** 2)

                    if args.w_manufactured_grad_supervision > 0.0:
                        grad_sup_pred = grad_u_in_physical(
                            u_nets[i],
                            decoders[i],
                            xi_sup,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                            sigma_floor=args.sigma_floor,
                            det_floor=args.det_floor,
                        )
                        grad_sup_true = manufactured_grad_u(x_sup).detach()
                        loss_sup_grad = torch.mean((grad_sup_pred - grad_sup_true) ** 2)

                iv_terms: List[torch.Tensor] = []
                if_terms: List[torch.Tensor] = []
                for j in neighbors[i]:
                    ib = interface_batch(i, j, args.if_batch)
                    if ib is None:
                        continue
                    _, xi_i, xi_j, n_if = ib
                    liv, _, lif_train = interface_coupling_terms(i, j, xi_i, xi_j, n_if, detach_neighbor=True)
                    iv_terms.append(liv)
                    if_terms.append(lif_train)

                loss_iv = torch.mean(torch.stack(iv_terms)) if iv_terms else torch.tensor(0.0, device=device, dtype=dtype)
                loss_if = torch.mean(torch.stack(if_terms)) if if_terms else torch.tensor(0.0, device=device, dtype=dtype)

                loss = (
                    w_pde_eff * loss_pde
                    + args.w_bc * loss_bc
                    + args.w_interface_value * loss_iv
                    + w_if_flux_eff * loss_if
                    + args.w_manufactured_supervision * loss_sup
                    + args.w_manufactured_grad_supervision * loss_sup_grad
                )

            if use_amp:
                scalers[i].scale(loss).backward()
                scalers[i].unscale_(opts[i])
                torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                scalers[i].step(opts[i])
                scalers[i].update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=5.0)
                opts[i].step()

        if args.omega < 1.0:
            blend_model_with_old(u_nets[i], old_state, omega=args.omega)

    def snapshot_u_states() -> List[Dict[str, torch.Tensor]]:
        return [copy_state_dict(u.state_dict()) for u in u_nets]

    def load_snapshot_u_states(states: List[Dict[str, torch.Tensor]]) -> None:
        for i in range(n_charts):
            u_nets[i].load_state_dict(states[i])

    def current_lr() -> float:
        if not opts or not opts[0].param_groups:
            return 0.0
        return float(opts[0].param_groups[0]["lr"])

    def set_lr(new_lr: float) -> None:
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = float(new_lr)

    def effective_flux_weight(it: int) -> float:
        if args.w_interface_flux_start is None and args.w_interface_flux_end is None:
            return float(args.w_interface_flux)
        if args.w_interface_flux_start is None:
            w_start = float(args.w_interface_flux)
        else:
            w_start = float(args.w_interface_flux_start)
        if args.w_interface_flux_end is None:
            w_end = float(args.w_interface_flux)
        else:
            w_end = float(args.w_interface_flux_end)
        if args.flux_ramp_iters <= 0:
            return w_end
        alpha = min(1.0, float(it) / max(1.0, float(args.flux_ramp_iters)))
        return w_start + (w_end - w_start) * alpha

    snapshots: Dict[str, Optional[Dict[str, object]]] = {
        "best_score": None,
        "best_rel_l2": None,
        "best_target": None,
        "best_flux": None,
    }
    best_score = float("inf")
    best_rel_l2 = float("inf")
    best_target_obj = float("inf")
    best_flux = float("inf")
    stale = 0
    guard_stale = 0
    guard_limit = float(args.guard_rel_l2 if args.guard_rel_l2 > 0.0 else args.target_rel_l2)

    def maybe_record_snapshot(
        name: str,
        it: int,
        pde_m: float,
        bc_m: float,
        iv_m: float,
        if_m: float,
        rel_l2_eval: float,
        score: float,
    ) -> None:
        snapshots[name] = {
            "iter": int(it),
            "pde": float(pde_m),
            "bc": float(bc_m),
            "if_val": float(iv_m),
            "if_flux": float(if_m),
            "rel_l2_eval": float(rel_l2_eval),
            "score": float(score),
            "u_states": snapshot_u_states(),
            "lr": current_lr(),
        }

    start = time.time()

    for it in range(1, args.max_schwarz_iters + 1):
        warm = min(1.0, float(it) / max(1.0, float(args.pde_warmup_iters)))
        w_pde_eff = args.w_pde * warm
        w_if_flux_eff = effective_flux_weight(it)

        for group in color_groups:
            active = [i for i in group if point_idx_by_chart[i].numel() > 0]
            if not active:
                continue

            if use_cuda_stream_parallel and len(active) > 1:
                chunk_size = max(1, min(args.max_parallel_charts, len(active), len(stream_pool)))
                for s in range(0, len(active), chunk_size):
                    chunk = active[s : s + chunk_size]
                    streams = stream_pool[: len(chunk)]
                    for stream, i in zip(streams, chunk):
                        with torch.cuda.stream(stream):
                            optimize_chart(i, w_pde_eff=w_pde_eff, w_if_flux_eff=w_if_flux_eff)
                    for stream in streams:
                        stream.synchronize()
            else:
                for i in active:
                    optimize_chart(i, w_pde_eff=w_pde_eff, w_if_flux_eff=w_if_flux_eff)

        pde_m, bc_m, iv_m, if_m = eval_global_metrics()
        rel_l2_eval = eval_rel_l2_subset()
        history["global_residual"].append(pde_m)
        history["bc_loss"].append(bc_m)
        history["interface_value"].append(iv_m)
        history["interface_flux"].append(if_m)
        history["rel_l2_eval"].append(rel_l2_eval)
        history["w_interface_flux_eff"].append(w_if_flux_eff)

        # Plateau tracking should reflect the effective training objective weights.
        score = w_pde_eff * pde_m + args.w_interface_value * iv_m + w_if_flux_eff * if_m
        if score + args.plateau_tol < best_score:
            best_score = score
            stale = 0
            maybe_record_snapshot("best_score", it, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)
        else:
            stale += 1

        if rel_l2_eval + 1e-14 < best_rel_l2:
            best_rel_l2 = rel_l2_eval
            maybe_record_snapshot("best_rel_l2", it, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)

        if rel_l2_eval <= args.target_rel_l2:
            target_obj = iv_m + if_m
            if target_obj + 1e-14 < best_target_obj:
                best_target_obj = target_obj
                maybe_record_snapshot("best_target", it, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)

        if rel_l2_eval <= guard_limit and if_m + 1e-14 < best_flux:
            best_flux = if_m
            maybe_record_snapshot("best_flux", it, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)

        if it % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            print(
                f"[Schwarz] iter={it}/{args.max_schwarz_iters} "
                f"pde={pde_m:.3e} bc={bc_m:.3e} if_val={iv_m:.3e} if_flux={if_m:.3e} "
                f"rel_l2_eval={rel_l2_eval:.3e} w_if={w_if_flux_eff:.3e} "
                f"score={score:.3e} stale={stale} t={elapsed:.1f}s"
            )

        pde_ok = (args.w_pde <= 0.0) or (pde_m <= args.residual_tol)
        converged = pde_ok and (iv_m <= args.interface_tol) and (if_m <= args.interface_flux_tol)
        if converged:
            print(f"Converged at iteration {it}")
            break

        if args.guard_patience > 0:
            if rel_l2_eval > guard_limit:
                guard_stale += 1
            else:
                guard_stale = 0
            if guard_stale >= args.guard_patience:
                fallback = snapshots.get("best_target") or snapshots.get("best_rel_l2")
                if fallback is not None:
                    load_snapshot_u_states(fallback["u_states"])  # type: ignore[index]
                    new_lr = max(1e-7, 0.5 * current_lr())
                    set_lr(new_lr)
                    stale = 0
                    guard_stale = 0
                    print(
                        f"[Guard] L2 exceeded {guard_limit:.3e} for {args.guard_patience} evals; "
                        f"restored iter={fallback['iter']} and reduced lr to {new_lr:.3e}"
                    )
        if stale >= args.plateau_patience:
            print(f"Stopped by plateau patience at iteration {it}")
            break

    if len(history["global_residual"]) == 0:
        # Pretrain-only runs still produce one deterministic metric snapshot.
        pde_m, bc_m, iv_m, if_m = eval_global_metrics()
        rel_l2_eval = eval_rel_l2_subset()
        w_if_flux_eff = effective_flux_weight(0)
        score = args.w_pde * pde_m + args.w_interface_value * iv_m + w_if_flux_eff * if_m
        history["global_residual"].append(pde_m)
        history["bc_loss"].append(bc_m)
        history["interface_value"].append(iv_m)
        history["interface_flux"].append(if_m)
        history["rel_l2_eval"].append(rel_l2_eval)
        history["w_interface_flux_eff"].append(w_if_flux_eff)
        maybe_record_snapshot("best_score", 0, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)
        maybe_record_snapshot("best_rel_l2", 0, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)
        if rel_l2_eval <= args.target_rel_l2:
            maybe_record_snapshot("best_target", 0, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)
        if rel_l2_eval <= guard_limit:
            maybe_record_snapshot("best_flux", 0, pde_m, bc_m, iv_m, if_m, rel_l2_eval, score)

    def choose_state_label(policy: str) -> Optional[str]:
        if policy == "last":
            return None
        if policy == "best_score":
            return "best_score"
        if policy == "best_target":
            if snapshots.get("best_target") is not None:
                return "best_target"
            return "best_rel_l2"
        if policy == "best_rel_l2":
            return "best_rel_l2"
        if policy == "best_flux":
            return "best_flux"
        # Pareto fallback: exact-threshold feasible first, then min normalized violation.
        if snapshots.get("best_target") is not None:
            return "best_target"
        candidates = [k for k in ["best_rel_l2", "best_flux", "best_score"] if snapshots.get(k) is not None]
        if not candidates:
            return None
        best_name = candidates[0]
        best_val = float("inf")
        for c in candidates:
            m = snapshots[c]
            reln = float(m["rel_l2_eval"]) / max(1e-12, float(args.target_rel_l2))
            ivn = float(m["if_val"]) / max(1e-12, float(args.interface_tol))
            ifn = float(m["if_flux"]) / max(1e-12, float(args.interface_flux_tol))
            v = max(reln, ivn, ifn)
            if v < best_val:
                best_val = v
                best_name = c
        return best_name

    selected_label = choose_state_label(args.checkpoint_policy)
    if selected_label is not None and snapshots.get(selected_label) is not None:
        load_snapshot_u_states(snapshots[selected_label]["u_states"])  # type: ignore[index]
        print(f"Selected checkpoint-policy state: {selected_label}")

    run_stem = build_run_stem(args.run_tag)

    # Global assembly on canonical rabbit points.
    with torch.no_grad():
        logits = []
        u_chart = []
        for i in range(n_charts):
            xi = local_coords(points, seeds[i], t1[i], t2[i], nvec[i])
            logits.append(masks[i](xi, chart_scale=support_r[i]))
            u_chart.append(u_nets[i](xi).squeeze(-1))
        logits_t = torch.stack(logits, dim=1)
        weights = torch.softmax(logits_t, dim=1)
        u_chart_t = torch.stack(u_chart, dim=1)
        u_pred = torch.sum(weights * u_chart_t, dim=1, keepdim=True)
        u_true = manufactured_u(points)
        u_err = u_pred - u_true
        e_mag = torch.abs(u_err).squeeze(-1)
        chart_id = torch.argmax(weights, dim=1)
        blend_weight = torch.max(weights, dim=1).values

        interface_residual = torch.zeros((n_points,), device=device, dtype=dtype)
        mem_bool = membership > 0
        for r in range(n_points):
            ids = torch.where(mem_bool[r])[0]
            if ids.numel() > 1:
                vals = u_chart_t[r, ids]
                interface_residual[r] = torch.std(vals)

    # Per-chart error summaries.
    per_chart = []
    for i in range(n_charts):
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            per_chart.append(
                {
                    "chart_id": i,
                    "n_points": 0,
                    "l2_error": None,
                    "relative_l2_error": None,
                    "max_error": None,
                }
            )
            continue
        stats = metric_l2(
            u_pred[idx].cpu().numpy().reshape(-1),
            u_true[idx].cpu().numpy().reshape(-1),
        )
        per_chart.append(
            {
                "chart_id": i,
                "n_points": int(idx.numel()),
                **stats,
            }
        )

    global_stats = metric_l2(
        u_pred.cpu().numpy().reshape(-1),
        u_true.cpu().numpy().reshape(-1),
    )

    out = {
        "global": global_stats,
        "per_chart": per_chart,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "amp_used": bool(use_amp),
        "tf32_used": bool(args.tf32 and device.type == "cuda"),
        "parallel_color_updates_used": bool(use_cuda_stream_parallel),
        "interface_transmission_mode": args.interface_transmission_mode,
        "robin_lambda": float(args.robin_lambda),
        "interface_flux_mode": args.interface_flux_mode,
        "interface_normal_mode": args.interface_normal_mode,
        "color_groups": [[int(x) for x in g] for g in color_groups],
        "n_charts": int(n_charts),
        "n_points": int(n_points),
        "mean_interface_residual": float(torch.mean(interface_residual).item()),
        "max_interface_residual": float(torch.max(interface_residual).item()),
        "final_global_residual": float(history["global_residual"][-1]) if history["global_residual"] else None,
        "final_interface_value": float(history["interface_value"][-1]) if history["interface_value"] else None,
        "final_interface_flux": float(history["interface_flux"][-1]) if history["interface_flux"] else None,
        "final_rel_l2_eval": float(history["rel_l2_eval"][-1]) if history["rel_l2_eval"] else None,
        "interface_target_met": bool(
            (history["interface_value"][-1] <= args.interface_tol if history["interface_value"] else False)
            and (history["interface_flux"][-1] <= args.interface_flux_tol if history["interface_flux"] else False)
        ),
        "target_relative_l2": float(args.target_rel_l2),
        "target_met": bool(global_stats["relative_l2_error"] <= args.target_rel_l2),
        "checkpoint_policy": args.checkpoint_policy,
        "selected_state": "last" if selected_label is None else selected_label,
        "runtime_seconds": float(time.time() - start),
    }

    def snap_summary(name: str) -> Optional[Dict[str, object]]:
        s = snapshots.get(name)
        if s is None:
            return None
        return {
            "iter": int(s["iter"]),
            "rel_l2_eval": float(s["rel_l2_eval"]),
            "if_val": float(s["if_val"]),
            "if_flux": float(s["if_flux"]),
            "score": float(s["score"]),
            "lr": float(s["lr"]),
        }

    out["checkpoint_triplet"] = {
        "best_rel_l2": snap_summary("best_rel_l2"),
        "best_target": snap_summary("best_target"),
        "best_flux_under_guard": snap_summary("best_flux"),
        "best_score": snap_summary("best_score"),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    solution_npz = os.path.join(args.output_dir, f"{run_stem}_solution.npz")
    np.savez_compressed(
        solution_npz,
        points=points.cpu().numpy(),
        normals=normals.cpu().numpy(),
        u_pred=u_pred.cpu().numpy().reshape(-1),
        u_true=u_true.cpu().numpy().reshape(-1),
        u_error=u_err.cpu().numpy().reshape(-1),
        u_error_mag=e_mag.cpu().numpy().reshape(-1),
        chart_id=chart_id.cpu().numpy().astype(np.int32),
        blend_weight=blend_weight.cpu().numpy().reshape(-1),
        interface_residual=interface_residual.cpu().numpy().reshape(-1),
        chart_weights=weights.cpu().numpy(),
        chart_values=u_chart_t.cpu().numpy(),
    )

    def save_ckpt(path: str, u_states: List[Dict[str, torch.Tensor]], label: str, fallback_from: Optional[str] = None) -> None:
        payload = {
            "u_states": u_states,
            "u_kwargs": {"width": args.pinn_width, "depth": args.pinn_depth},
            "history": history,
            "metrics": out,
            "atlas_data_path": args.atlas_data,
            "atlas_checkpoint_path": args.atlas_checkpoint,
            "snapshot_label": label,
            "fallback_from": fallback_from,
        }
        torch.save(payload, path)

    ckpt_path = os.path.join(args.output_dir, f"{run_stem}_checkpoint.pt")
    save_ckpt(ckpt_path, snapshot_u_states(), label="selected_state", fallback_from=None)

    def pick_snapshot_for_save(name: str, backup_order: Sequence[str]) -> Tuple[List[Dict[str, torch.Tensor]], Optional[str]]:
        s = snapshots.get(name)
        if s is not None:
            return s["u_states"], None  # type: ignore[return-value]
        for b in backup_order:
            sb = snapshots.get(b)
            if sb is not None:
                return sb["u_states"], b  # type: ignore[return-value]
        return snapshot_u_states(), "last"

    best_rel_l2_states, best_rel_fallback = pick_snapshot_for_save("best_rel_l2", ["best_score"])
    best_target_states, best_target_fallback = pick_snapshot_for_save("best_target", ["best_rel_l2", "best_score"])
    best_flux_states, best_flux_fallback = pick_snapshot_for_save("best_flux", ["best_target", "best_rel_l2", "best_score"])
    best_score_states, best_score_fallback = pick_snapshot_for_save("best_score", ["best_rel_l2"])

    ckpt_best_rel = os.path.join(args.output_dir, f"{run_stem}_best_rel_l2.pt")
    ckpt_best_target = os.path.join(args.output_dir, f"{run_stem}_best_target.pt")
    ckpt_best_flux = os.path.join(args.output_dir, f"{run_stem}_best_flux.pt")
    ckpt_best_score = os.path.join(args.output_dir, f"{run_stem}_best_score.pt")
    save_ckpt(ckpt_best_rel, best_rel_l2_states, "best_rel_l2", fallback_from=best_rel_fallback)
    save_ckpt(ckpt_best_target, best_target_states, "best_target", fallback_from=best_target_fallback)
    save_ckpt(ckpt_best_flux, best_flux_states, "best_flux", fallback_from=best_flux_fallback)
    save_ckpt(ckpt_best_score, best_score_states, "best_score", fallback_from=best_score_fallback)

    out["checkpoint_paths"] = {
        "selected": ckpt_path,
        "best_rel_l2": ckpt_best_rel,
        "best_target": ckpt_best_target,
        "best_flux": ckpt_best_flux,
        "best_score": ckpt_best_score,
    }

    metrics_path = os.path.join(args.output_dir, f"{run_stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    history_path = os.path.join(args.output_dir, f"{run_stem}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    curve_path = os.path.join(args.output_dir, f"{run_stem}_curves.png")
    plot_history(history, curve_path)

    print("Schwarz Poisson run complete")
    print(f"  solution_npz: {solution_npz}")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  best_rel_l2:  {ckpt_best_rel}")
    print(f"  best_target:  {ckpt_best_target}")
    print(f"  best_flux:    {ckpt_best_flux}")
    print(f"  best_score:   {ckpt_best_score}")
    print(f"  metrics:      {metrics_path}")
    print(f"  curves:       {curve_path}")
    print(f"  rel_l2:       {out['global']['relative_l2_error']:.6e}")
    print(f"  max_error:    {out['global']['max_error']:.6e}")
    print(f"  target_met:   {out['target_met']}")

    return {
        "solution_npz": solution_npz,
        "checkpoint": ckpt_path,
        "best_rel_l2_checkpoint": ckpt_best_rel,
        "best_target_checkpoint": ckpt_best_target,
        "best_flux_checkpoint": ckpt_best_flux,
        "best_score_checkpoint": ckpt_best_score,
        "metrics": metrics_path,
        "history": history_path,
        "curves": curve_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atlas Schwarz Poisson solver on rabbit point cloud")
    parser.add_argument("--atlas-data", required=True, help="Path to rabbit_atlas_data.npz")
    parser.add_argument("--atlas-checkpoint", required=True, help="Path to rabbit_atlas_trained.pt")
    parser.add_argument("--atlas-meta", default=None, help="Path to rabbit_atlas_meta.json")
    parser.add_argument("--init-u-checkpoint", default=None, help="Optional warm-start checkpoint with chart PINN states.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="", help="Suffix tag for stage-specific artifacts.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float64"],
        help="Tensor dtype. 'auto' chooses float32 on GPU backends and float64 on CPU.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA.")

    parser.add_argument("--pinn-width", type=int, default=64)
    parser.add_argument("--pinn-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--max-schwarz-iters", type=int, default=60)
    parser.add_argument("--local-steps", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.8)

    parser.add_argument("--pde-batch", type=int, default=192)
    parser.add_argument("--bc-batch", type=int, default=192)
    parser.add_argument("--if-batch", type=int, default=128)
    parser.add_argument(
        "--xi-noise-scale",
        type=float,
        default=0.30,
        help="Relative Gaussian perturbation scale for interior chart sampling in xi coordinates.",
    )

    parser.add_argument("--eval-pde-samples-per-chart", type=int, default=96)
    parser.add_argument("--eval-bc-samples-per-chart", type=int, default=96)
    parser.add_argument("--eval-if-samples", type=int, default=64)
    parser.add_argument("--eval-fixed-cache", action="store_true", help="Use fixed eval sample caches across iterations.")
    parser.add_argument("--eval-cache-seed", type=int, default=1234)
    parser.add_argument("--eval-cache-per-chart", type=int, default=128)
    parser.add_argument("--eval-cache-per-overlap", type=int, default=96)
    parser.add_argument("--sigma-floor", type=float, default=1e-3, help="SVD floor for stable Jacobian inversion.")
    parser.add_argument("--det-floor", type=float, default=1e-6, help="Lower bound for |det(J)| stabilization.")
    parser.add_argument("--jac-kappa-max", type=float, default=1e3, help="Discard PDE samples with kappa(J) above this.")
    parser.add_argument(
        "--pde-clip-quantile",
        type=float,
        default=0.98,
        help="Clip PDE residual magnitude at this quantile during training (set 1.0 to disable).",
    )
    parser.add_argument(
        "--pde-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta for PDE residual loss (set <=0 for plain MSE).",
    )
    parser.add_argument("--pde-warmup-iters", type=int, default=10, help="Ramp PDE weight over this many Schwarz iterations.")

    parser.add_argument("--bc-pretrain-epochs", type=int, default=300)
    parser.add_argument("--bc-pretrain-batch", type=int, default=256)
    parser.add_argument("--bc-pretrain-grad-weight", type=float, default=0.05)
    parser.add_argument("--bc-pretrain-interface-weight", type=float, default=0.2)
    parser.add_argument("--bc-pretrain-log-every", type=int, default=50)
    parser.add_argument("--manufactured-supervision-batch", type=int, default=128)
    parser.add_argument("--w-manufactured-supervision", type=float, default=0.0)
    parser.add_argument("--w-manufactured-grad-supervision", type=float, default=0.0)

    parser.add_argument(
        "--parallel-color-updates",
        action="store_true",
        help="On CUDA, update non-overlapping charts in a color group concurrently via CUDA streams.",
    )
    parser.add_argument("--max-parallel-charts", type=int, default=4)
    parser.add_argument("--stream-pool-size", type=int, default=4, help="Persistent CUDA stream pool size.")
    parser.add_argument(
        "--interface-flux-mode",
        choices=["projected", "vector"],
        default="projected",
        help="Flux continuity metric: projected normal flux or full gradient-vector matching.",
    )
    parser.add_argument(
        "--interface-transmission-mode",
        choices=["penalty", "robin"],
        default="penalty",
        help="Interface transmission training condition.",
    )
    parser.add_argument(
        "--robin-lambda",
        type=float,
        default=10.0,
        help="Robin coupling coefficient lambda in lambda*(u_i-u_j)+(q_i-q_j).",
    )
    parser.add_argument(
        "--interface-normal-mode",
        choices=["mask_levelset", "seed"],
        default="mask_levelset",
        help="Interface normal construction for projected flux continuity.",
    )
    parser.add_argument("--interface-normal-eps", type=float, default=1e-6)
    parser.add_argument(
        "--interface-normal-blend",
        type=float,
        default=0.15,
        help="Blend ratio toward seed-direction normal for robustness.",
    )
    parser.add_argument(
        "--interface-normal-cache-batch",
        type=int,
        default=2048,
        help="Batch size for precomputing mask-levelset interface normals.",
    )

    parser.add_argument("--interior-pretrain-epochs", type=int, default=0)
    parser.add_argument("--interior-pretrain-batch", type=int, default=256)
    parser.add_argument("--interior-pretrain-grad-weight", type=float, default=0.5)
    parser.add_argument("--interior-pretrain-log-every", type=int, default=50)

    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=2.0)
    parser.add_argument("--w-interface-value", type=float, default=0.8)
    parser.add_argument("--w-interface-flux", type=float, default=0.2)
    parser.add_argument("--w-interface-flux-start", type=float, default=None)
    parser.add_argument("--w-interface-flux-end", type=float, default=None)
    parser.add_argument("--flux-ramp-iters", type=int, default=0)

    parser.add_argument("--residual-tol", type=float, default=2e-3)
    parser.add_argument("--interface-tol", type=float, default=8e-3)
    parser.add_argument("--interface-flux-tol", type=float, default=1.5e-2)
    parser.add_argument("--plateau-patience", type=int, default=15)
    parser.add_argument("--plateau-tol", type=float, default=5e-5)
    parser.add_argument("--target-rel-l2", type=float, default=1.5e-1)
    parser.add_argument("--guard-rel-l2", type=float, default=0.0, help="L2 guard threshold (<=0 uses target-rel-l2).")
    parser.add_argument("--guard-patience", type=int, default=0, help="Rollback after this many consecutive guard violations.")
    parser.add_argument(
        "--checkpoint-policy",
        type=str,
        default="last",
        choices=["last", "best_score", "best_target", "best_pareto", "best_rel_l2", "best_flux"],
    )

    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Debug only: bypass atlas-gate prerequisite and run solver anyway.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_schwarz(args)


if __name__ == "__main__":
    main()
