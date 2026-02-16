#!/usr/bin/env python3
"""
Rabbit TI Arruda-Boyce inverse benchmark with single normal-traction ramp family
and strict Schwarz interface control.

Key features:
- Fixed rabbit atlas (meshfree coordinate charts).
- One load family: full-surface normal traction with linear z-ramp and 6 amplitudes.
- Stage T (teacher): solve forward chart fields using true parameters.
- Stage A: inverse on (mu_m, k_f) with N_m/N_f fixed.
- Stage B: inverse on all four parameters.
- Strong Schwarz controls: color-sweep multiplicative updates, relaxation,
  interface-weight ramp, hard-pair oversampling, and interface acceptance gate.
- Exports: metrics/history/checkpoints and ParaView VTU files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fnn


torch.set_default_dtype(torch.float64)


# ----------------------------- Utilities -----------------------------


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
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Requested CUDA but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        print("Requested MPS but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            print("Requested float64 on MPS; using float32.")
            return torch.float32
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def scalar(x: torch.Tensor, device: torch.device) -> float:
    sync_device(device)
    return float(x.detach().to(device=torch.device("cpu")).reshape(-1)[0].item())


def parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{v}'")


def parse_simple_yaml(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def cfg_get_str(cfg: Dict[str, str], key: str, default: str) -> str:
    return str(cfg.get(key, default))


def cfg_get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    if key not in cfg:
        return int(default)
    return int(float(cfg[key]))


def cfg_get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    if key not in cfg:
        return float(default)
    return float(cfg[key])


def cfg_get_bool(cfg: Dict[str, str], key: str, default: bool) -> bool:
    if key not in cfg:
        return bool(default)
    return parse_bool(cfg[key])


def parse_float_list(v: str) -> List[float]:
    toks = [t.strip() for t in str(v).replace(";", ",").split(",") if t.strip()]
    return [float(t) for t in toks]


def softplus_inv(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


def build_run_stem(run_tag: str) -> str:
    run_tag = str(run_tag).strip()
    if run_tag:
        return f"rabbit_inverse_ti_arruda_boyce_atlas_schwarz_normal_ramp_{run_tag}"
    return "rabbit_inverse_ti_arruda_boyce_atlas_schwarz_normal_ramp"


def robust_huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    # Smooth L1 over full tensor.
    return Fnn.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        reduction="mean",
        beta=float(delta),
    )


# ----------------------------- Models -----------------------------


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
        self.raw_scale = torch.nn.Parameter(torch.tensor(-1.8))

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


class LocalVectorPINN(torch.nn.Module):
    def __init__(self, width: int = 96, depth: int = 5):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.net(xi)


class TIParams(torch.nn.Module):
    def __init__(
        self,
        mu_m_init: float,
        N_m_init: float,
        k_f_init: float,
        N_f_init: float,
        mu_m_min: float,
        N_m_min: float,
        k_f_min: float,
        N_f_min: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.mu_m_min = float(mu_m_min)
        self.N_m_min = float(N_m_min)
        self.k_f_min = float(k_f_min)
        self.N_f_min = float(N_f_min)

        self.mu_m_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(mu_m_init - self.mu_m_min, 1e-6)), device=device, dtype=dtype)
        )
        self.N_m_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(N_m_init - self.N_m_min, 1e-6)), device=device, dtype=dtype)
        )
        self.k_f_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(k_f_init - self.k_f_min, 1e-6)), device=device, dtype=dtype)
        )
        self.N_f_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(N_f_init - self.N_f_min, 1e-6)), device=device, dtype=dtype)
        )

    def values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_m = torch.nn.functional.softplus(self.mu_m_raw) + self.mu_m_min
        N_m = torch.nn.functional.softplus(self.N_m_raw) + self.N_m_min
        k_f = torch.nn.functional.softplus(self.k_f_raw) + self.k_f_min
        N_f = torch.nn.functional.softplus(self.N_f_raw) + self.N_f_min
        return mu_m, N_m, k_f, N_f


# ----------------------------- Tensor ops -----------------------------


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps)


def gradient_tensor(v: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    grads = []
    for i in range(v.shape[1]):
        gi = torch.autograd.grad(
            v[:, i],
            x,
            grad_outputs=torch.ones_like(v[:, i]),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        grads.append(gi.unsqueeze(1))
    return torch.cat(grads, dim=1)


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


def det_3x3(a: torch.Tensor) -> torch.Tensor:
    a11 = a[:, 0, 0]
    a12 = a[:, 0, 1]
    a13 = a[:, 0, 2]
    a21 = a[:, 1, 0]
    a22 = a[:, 1, 1]
    a23 = a[:, 1, 2]
    a31 = a[:, 2, 0]
    a32 = a[:, 2, 1]
    a33 = a[:, 2, 2]
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)


def inv_det_3x3_stable(jac: torch.Tensor, det_floor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = jac[:, 0, 0]
    b = jac[:, 0, 1]
    c = jac[:, 0, 2]
    d = jac[:, 1, 0]
    e = jac[:, 1, 1]
    f = jac[:, 1, 2]
    g = jac[:, 2, 0]
    h = jac[:, 2, 1]
    i = jac[:, 2, 2]

    c11 = e * i - f * h
    c12 = c * h - b * i
    c13 = b * f - c * e
    c21 = f * g - d * i
    c22 = a * i - c * g
    c23 = c * d - a * f
    c31 = d * h - e * g
    c32 = b * g - a * h
    c33 = a * e - b * d

    det_raw = a * c11 + b * c21 + c * c31
    det_safe = torch.where(
        det_raw >= 0.0,
        torch.clamp(det_raw, min=det_floor),
        torch.clamp(det_raw, max=-det_floor),
    )

    adj = torch.stack(
        [
            torch.stack([c11, c12, c13], dim=1),
            torch.stack([c21, c22, c23], dim=1),
            torch.stack([c31, c32, c33], dim=1),
        ],
        dim=1,
    )
    inv_j = adj / det_safe.unsqueeze(-1).unsqueeze(-1)
    det_abs = torch.clamp(torch.abs(det_raw), min=det_floor)
    return inv_j, det_abs, det_raw


def stabilized_jacobian_ops(
    jac: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
    jac_kappa_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sf = torch.as_tensor(sigma_floor, device=jac.device, dtype=jac.dtype)
    df = torch.as_tensor(det_floor, device=jac.device, dtype=jac.dtype)
    eye = torch.eye(3, device=jac.device, dtype=jac.dtype).unsqueeze(0)
    jac_reg = jac + torch.clamp(sf, min=torch.as_tensor(1e-8, device=jac.device, dtype=jac.dtype)) * eye
    inv_j, det_abs, det_raw = inv_det_3x3_stable(jac_reg, det_floor=df)

    n_j = torch.sqrt(torch.clamp(torch.sum(jac_reg * jac_reg, dim=(1, 2)), min=1e-24))
    n_inv = torch.sqrt(torch.clamp(torch.sum(inv_j * inv_j, dim=(1, 2)), min=1e-24))
    kappa = n_j * n_inv

    valid = torch.isfinite(det_abs) & torch.isfinite(kappa) & torch.isfinite(det_raw)
    valid = valid & (torch.abs(det_raw) > df) & (kappa <= torch.as_tensor(jac_kappa_max, device=jac.device, dtype=jac.dtype))
    return inv_j, det_abs, kappa, valid


def divergence_mapped(mapped_flux: torch.Tensor, xi_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
    # mapped_flux: [N,3,3], divergence over eta-dim -> [N,3]
    div = torch.zeros((mapped_flux.shape[0], 3), device=xi_var.device, dtype=xi_var.dtype)
    for c in range(3):
        comp = torch.zeros((mapped_flux.shape[0], 1), device=xi_var.device, dtype=xi_var.dtype)
        for j in range(3):
            dcomp = torch.autograd.grad(
                mapped_flux[:, c, j],
                xi_var,
                grad_outputs=torch.ones_like(mapped_flux[:, c, j]),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, j : j + 1]
            comp = comp + dcomp
        div[:, c : c + 1] = comp
    return div


# ----------------------------- Material model -----------------------------


def ti_arruda_boyce_energy(
    F: torch.Tensor,
    mu_m: torch.Tensor,
    N_m: torch.Tensor,
    k_f: torch.Tensor,
    N_f: torch.Tensor,
    kappa: float,
    a0: torch.Tensor,
) -> torch.Tensor:
    C = torch.bmm(F.transpose(1, 2), F)
    J = torch.clamp(det_3x3(F), min=torch.as_tensor(1e-8, device=F.device, dtype=F.dtype))
    I1 = torch.einsum("bii->b", C)
    Jm23 = J.pow(-2.0 / 3.0)
    I1_bar = Jm23 * I1

    a0b = a0.unsqueeze(0).expand(F.shape[0], -1)
    Ca = torch.bmm(C, a0b.unsqueeze(-1)).squeeze(-1)
    I4 = torch.sum(a0b * Ca, dim=1)
    I4_bar = Jm23 * I4

    N_m_safe = torch.clamp(N_m, min=torch.as_tensor(1.01, device=F.device, dtype=F.dtype))
    N_f_safe = torch.clamp(N_f, min=torch.as_tensor(1.01, device=F.device, dtype=F.dtype))

    # Polynomial Arruda-Boyce approximation around reference state.
    t1m = 0.5 * (I1_bar - 3.0)
    t2m = (I1_bar * I1_bar - 9.0) / (20.0 * N_m_safe)
    t3m = 11.0 * (I1_bar**3 - 27.0) / (1050.0 * (N_m_safe**2))
    Wm = mu_m * (t1m + t2m + t3m)

    I4_eff = 1.0 + torch.nn.functional.relu(I4_bar - 1.0)
    t1f = 0.5 * (I4_eff - 1.0)
    t2f = (I4_eff * I4_eff - 1.0) / (20.0 * N_f_safe)
    t3f = 11.0 * (I4_eff**3 - 1.0) / (1050.0 * (N_f_safe**2))
    Wf = k_f * (t1f + t2f + t3f)

    Wvol = 0.5 * torch.as_tensor(kappa, device=F.device, dtype=F.dtype) * (J - 1.0) ** 2
    return Wm + Wf + Wvol


def ti_arruda_boyce_p(
    F: torch.Tensor,
    mu_m: torch.Tensor,
    N_m: torch.Tensor,
    k_f: torch.Tensor,
    N_f: torch.Tensor,
    kappa: float,
    a0: torch.Tensor,
) -> torch.Tensor:
    W = ti_arruda_boyce_energy(F=F, mu_m=mu_m, N_m=N_m, k_f=k_f, N_f=N_f, kappa=kappa, a0=a0)
    P = torch.autograd.grad(W.sum(), F, create_graph=True, retain_graph=True)[0]
    return P


# ----------------------------- Loads and caches -----------------------------


@dataclass
class LoadLevel:
    name: str
    lam: float


@dataclass
class ChartCache:
    xi_int: torch.Tensor
    xi_surf: torch.Tensor
    x_surf: torch.Tensor
    n_surf: torch.Tensor
    alpha_surf: torch.Tensor
    u_teacher_by_level: List[torch.Tensor]


@dataclass
class PairCache:
    xi_i: torch.Tensor
    xi_j: torch.Tensor


def make_load_levels(lambdas: Sequence[float]) -> List[LoadLevel]:
    out = []
    for i, v in enumerate(lambdas):
        out.append(LoadLevel(name=f"L{i+1:02d}", lam=float(v)))
    return out


def sample_indices(n_total: int, n_take: int, device: torch.device) -> torch.Tensor:
    if n_total <= 0:
        return torch.zeros((0,), device=device, dtype=torch.int64)
    if n_take >= n_total:
        return torch.randint(0, n_total, (n_take,), device=device)
    perm = torch.randperm(n_total, device=device)
    return perm[:n_take]


def parse_neighbors(meta_json: Optional[str], n_charts: int) -> Tuple[Dict[int, List[int]], List[List[int]]]:
    default_neighbors = {i: [] for i in range(n_charts)}
    default_groups = [[i] for i in range(n_charts)]
    if meta_json is None or (not os.path.isfile(meta_json)):
        return default_neighbors, default_groups

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    neighbors = {i: [] for i in range(n_charts)}
    og = meta.get("overlap_graph")
    if isinstance(og, dict):
        for k, v in og.items():
            i = int(k)
            if not (0 <= i < n_charts):
                continue
            neigh = []
            for x in v:
                j = int(x)
                if 0 <= j < n_charts and j != i:
                    neigh.append(j)
            neighbors[i] = sorted(set(neigh))

    groups = []
    graw = meta.get("color_groups")
    if isinstance(graw, list) and len(graw) > 0:
        for g in graw:
            if not isinstance(g, list):
                continue
            gi = []
            for x in g:
                ix = int(x)
                if 0 <= ix < n_charts:
                    gi.append(ix)
            if gi:
                groups.append(sorted(set(gi)))

    if not groups:
        groups = default_groups

    return neighbors, groups


def load_atlas_models(
    atlas_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[ChartDecoder], List[MaskNet], Dict[str, object]]:
    ckpt = torch.load(atlas_checkpoint, map_location=torch.device("cpu"))
    dec_kw = ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kw = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    dec_states = ckpt["decoder_states"]
    mask_states = ckpt["mask_states"]

    def cast_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            out[k] = v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v
        return out

    decoders: List[ChartDecoder] = []
    masks: List[MaskNet] = []
    for ds, ms in zip(dec_states, mask_states):
        d = ChartDecoder(width=dec_kw["width"], depth=dec_kw["depth"]).to(device=device, dtype=dtype)
        m = MaskNet(width=mask_kw["width"], depth=mask_kw["depth"]).to(device=device, dtype=dtype)
        d.load_state_dict(cast_state(ds))
        m.load_state_dict(cast_state(ms))
        d.eval()
        m.eval()
        for p in d.parameters():
            p.requires_grad_(False)
        for p in m.parameters():
            p.requires_grad_(False)
        decoders.append(d)
        masks.append(m)
    return decoders, masks, ckpt


def compute_alpha_centered(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    zmin = torch.min(z)
    zmax = torch.max(z)
    a = (z - zmin) / torch.clamp(zmax - zmin, min=torch.as_tensor(eps, device=z.device, dtype=z.dtype))
    return a - torch.mean(a)


def prepare_chart_caches(
    args: argparse.Namespace,
    points: torch.Tensor,
    normals: torch.Tensor,
    membership: torch.Tensor,
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    n_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ChartCache]:
    n_charts = membership.shape[1]
    caches: List[ChartCache] = []

    alpha_global = compute_alpha_centered(points[:, 2])

    for i in range(n_charts):
        idx_all = torch.where(membership[:, i] > 0.5)[0]
        if idx_all.numel() < 32:
            raise RuntimeError(f"Chart {i} has too few support points: {idx_all.numel()}")

        pick_int = idx_all[torch.randint(0, idx_all.numel(), (args.int_cache_per_chart,), device=device)]
        x_int_seed = points[pick_int]
        xi_int = local_coords(x_int_seed, seeds[i], t1[i], t2[i], nvec[i])

        r = torch.clamp(chart_scales[i], min=torch.as_tensor(1e-3, device=device, dtype=dtype))
        noise = torch.randn_like(xi_int) * (args.xi_noise_scale * r)
        noise[:, 2] = noise[:, 2] * args.normal_noise_boost
        xi_int = xi_int + noise

        pick_surf = idx_all[torch.randint(0, idx_all.numel(), (args.n_obs_per_chart,), device=device)]
        x_surf = points[pick_surf]
        xi_surf = local_coords(x_surf, seeds[i], t1[i], t2[i], nvec[i])
        n_surf = normalize_rows(normals[pick_surf])
        alpha_surf = alpha_global[pick_surf]

        caches.append(
            ChartCache(
                xi_int=xi_int.detach(),
                xi_surf=xi_surf.detach(),
                x_surf=x_surf.detach(),
                n_surf=n_surf.detach(),
                alpha_surf=alpha_surf.detach(),
                u_teacher_by_level=[torch.zeros_like(x_surf) for _ in range(n_levels)],
            )
        )

    return caches


def prepare_pair_caches(
    args: argparse.Namespace,
    membership: torch.Tensor,
    points: torch.Tensor,
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    neighbors: Dict[int, List[int]],
    device: torch.device,
) -> Dict[Tuple[int, int], PairCache]:
    n_charts = membership.shape[1]
    out: Dict[Tuple[int, int], PairCache] = {}

    for i in range(n_charts):
        for j in neighbors.get(i, []):
            if j <= i:
                continue
            shared = torch.where((membership[:, i] > 0.5) & (membership[:, j] > 0.5))[0]
            if shared.numel() < 24:
                continue
            n_take = min(args.if_cache_per_pair, int(shared.numel()))
            sel = shared[torch.randint(0, shared.numel(), (n_take,), device=device)]
            x = points[sel]
            xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i]).detach()
            xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j]).detach()
            out[(i, j)] = PairCache(xi_i=xi_i, xi_j=xi_j)

    return out


def rigid_body_penalty(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    xc = x - torch.mean(x, dim=0, keepdim=True)
    eps = torch.as_tensor(1e-12, device=x.device, dtype=x.dtype)

    mode_tx = torch.stack([torch.ones_like(xc[:, 0]), torch.zeros_like(xc[:, 0]), torch.zeros_like(xc[:, 0])], dim=1)
    mode_ty = torch.stack([torch.zeros_like(xc[:, 0]), torch.ones_like(xc[:, 0]), torch.zeros_like(xc[:, 0])], dim=1)
    mode_tz = torch.stack([torch.zeros_like(xc[:, 0]), torch.zeros_like(xc[:, 0]), torch.ones_like(xc[:, 0])], dim=1)

    mode_rx = torch.stack([torch.zeros_like(xc[:, 0]), -xc[:, 2], xc[:, 1]], dim=1)
    mode_ry = torch.stack([xc[:, 2], torch.zeros_like(xc[:, 0]), -xc[:, 0]], dim=1)
    mode_rz = torch.stack([-xc[:, 1], xc[:, 0], torch.zeros_like(xc[:, 0])], dim=1)

    modes = [mode_tx, mode_ty, mode_tz, mode_rx, mode_ry, mode_rz]
    pen = torch.zeros((), device=x.device, dtype=x.dtype)
    for m in modes:
        num = torch.mean(torch.sum(m * u, dim=1))
        den = torch.mean(torch.sum(m * m, dim=1)) + eps
        c = num / den
        pen = pen + c * c
    return pen


# ----------------------------- Evaluation helpers -----------------------------


def chart_u_states(u_nets: Sequence[Sequence[LocalVectorPINN]]) -> List[List[Dict[str, torch.Tensor]]]:
    return [[{k: v.detach().clone() for k, v in m.state_dict().items()} for m in chart] for chart in u_nets]


def load_chart_u_states(u_nets: Sequence[Sequence[LocalVectorPINN]], states: Sequence[Sequence[Dict[str, torch.Tensor]]]) -> None:
    for i in range(len(u_nets)):
        for l in range(len(u_nets[i])):
            u_nets[i][l].load_state_dict(states[i][l])


def lerp(a: float, b: float, t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return float(a + (b - a) * t)


def make_theta_optimizer(
    ti_params: TIParams,
    stage: str,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if stage == "A":
        ti_params.N_m_raw.requires_grad_(False)
        ti_params.N_f_raw.requires_grad_(False)
        ti_params.mu_m_raw.requires_grad_(True)
        ti_params.k_f_raw.requires_grad_(True)
        return torch.optim.Adam(
            [ti_params.mu_m_raw, ti_params.k_f_raw],
            lr=args.lr_theta_stage_a,
        )

    ti_params.N_m_raw.requires_grad_(True)
    ti_params.N_f_raw.requires_grad_(True)
    ti_params.mu_m_raw.requires_grad_(True)
    ti_params.k_f_raw.requires_grad_(True)
    return torch.optim.Adam(
        [
            {"params": [ti_params.mu_m_raw, ti_params.k_f_raw], "lr": args.lr_theta_stage_b_fast},
            {"params": [ti_params.N_m_raw, ti_params.N_f_raw], "lr": args.lr_theta_stage_b_slow},
        ]
    )


def adjust_optimizer_lr(opt: torch.optim.Optimizer, factor: float, min_lr: float = 1e-8) -> None:
    for g in opt.param_groups:
        g["lr"] = max(min_lr, float(g["lr"]) * float(factor))


def compute_pair_scores_norm(pair_scores: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not pair_scores:
        return {}
    vals = np.array(list(pair_scores.values()), dtype=float)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo + 1e-12:
        return {k: 0.0 for k in pair_scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in pair_scores.items()}


def compute_interface_metrics(
    u_nets: Sequence[Sequence[LocalVectorPINN]],
    decoders: Sequence[ChartDecoder],
    pair_caches: Dict[Tuple[int, int], PairCache],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    levels: Sequence[LoadLevel],
    args: argparse.Namespace,
) -> Tuple[float, float, Dict[Tuple[int, int], float]]:
    if not pair_caches:
        return 0.0, 0.0, {}

    pair_scores: Dict[Tuple[int, int], float] = {}
    if_val_all: List[float] = []
    if_flux_all: List[float] = []

    for key, pc in pair_caches.items():
        i, j = key
        sel = sample_indices(pc.xi_i.shape[0], min(args.eval_if_samples, pc.xi_i.shape[0]), device=pc.xi_i.device)
        xi_i = pc.xi_i[sel]
        xi_j = pc.xi_j[sel]

        pair_if_val = 0.0
        pair_if_flux = 0.0
        for lidx in range(len(levels)):
            _, xi_i_var, jac_i = chart_map_and_jacobian(
                decoder=decoders[i],
                xi_in=xi_i,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=chart_scales[i],
            )
            inv_i, _, _, _ = stabilized_jacobian_ops(
                jac=jac_i,
                sigma_floor=args.sigma_floor,
                det_floor=args.detJ_floor,
                jac_kappa_max=args.jac_kappa_max,
            )

            _, xi_j_var, jac_j = chart_map_and_jacobian(
                decoder=decoders[j],
                xi_in=xi_j,
                seed=seeds[j],
                t1=t1[j],
                t2=t2[j],
                n=nvec[j],
                chart_scale=chart_scales[j],
            )
            inv_jj, _, _, _ = stabilized_jacobian_ops(
                jac=jac_j,
                sigma_floor=args.sigma_floor,
                det_floor=args.detJ_floor,
                jac_kappa_max=args.jac_kappa_max,
            )

            ui = u_nets[i][lidx](xi_i_var)
            uj = u_nets[j][lidx](xi_j_var)
            gi = torch.bmm(gradient_tensor(ui, xi_i_var, create_graph=False), inv_i)
            gj = torch.bmm(gradient_tensor(uj, xi_j_var, create_graph=False), inv_jj)

            v = torch.mean((ui - uj) ** 2)
            f = torch.mean((gi - gj) ** 2)
            pair_if_val += float(v.detach().item())
            pair_if_flux += float(f.detach().item())

        pair_if_val /= float(len(levels))
        pair_if_flux /= float(len(levels))
        score = pair_if_val + args.if_score_flux_factor * pair_if_flux
        pair_scores[key] = score
        if_val_all.append(pair_if_val)
        if_flux_all.append(pair_if_flux)

    mean_if_val = float(np.mean(if_val_all)) if if_val_all else 0.0
    mean_if_flux = float(np.mean(if_flux_all)) if if_flux_all else 0.0
    return mean_if_val, mean_if_flux, pair_scores


def compute_surface_rel_l2(
    u_nets: Sequence[Sequence[LocalVectorPINN]],
    caches: Sequence[ChartCache],
    levels: Sequence[LoadLevel],
) -> float:
    vals = []
    for lidx in range(len(levels)):
        pred_all = []
        true_all = []
        for i in range(len(caches)):
            with torch.no_grad():
                pred_all.append(u_nets[i][lidx](caches[i].xi_surf))
            true_all.append(caches[i].u_teacher_by_level[lidx])
        up = torch.cat(pred_all, dim=0)
        ut = torch.cat(true_all, dim=0)
        rel = torch.sqrt(torch.mean((up - ut) ** 2) / torch.clamp(torch.mean(ut * ut), min=torch.as_tensor(1e-12, device=ut.device, dtype=ut.dtype)))
        vals.append(float(rel.detach().item()))
    return float(np.mean(vals)) if vals else 0.0


def compute_level_energy(caches: Sequence[ChartCache], levels: Sequence[LoadLevel]) -> List[float]:
    energies = []
    for lidx in range(len(levels)):
        arr = []
        for i in range(len(caches)):
            u = caches[i].u_teacher_by_level[lidx]
            arr.append(torch.mean(u * u).detach().cpu().item())
        energies.append(max(float(np.mean(arr)), 1e-12))
    return energies


def effective_sensitivity_report(
    u_nets: Sequence[Sequence[LocalVectorPINN]],
    decoders: Sequence[ChartDecoder],
    caches: Sequence[ChartCache],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    levels: Sequence[LoadLevel],
    a0: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    params_center: Sequence[float],
) -> Dict[str, object]:
    # Effective identifiability proxy: d traction / d theta with teacher u fixed.
    theta0 = np.array(params_center, dtype=float)
    eps_rel = np.array([args.sens_eps_rel] * 4, dtype=float)

    # Build fixed F samples from current chart fields on surface points.
    samples = []
    for i in range(len(caches)):
        n_take = min(args.sensitivity_surface_per_chart, caches[i].xi_surf.shape[0])
        sel = sample_indices(caches[i].xi_surf.shape[0], n_take, device=device)
        xi = caches[i].xi_surf[sel]
        n_s = caches[i].n_surf[sel]
        for lidx in range(len(levels)):
            _, xi_var, jac = chart_map_and_jacobian(
                decoder=decoders[i],
                xi_in=xi,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=chart_scales[i],
            )
            inv_j, _, _, _ = stabilized_jacobian_ops(
                jac=jac,
                sigma_floor=args.sigma_floor,
                det_floor=args.detJ_floor,
                jac_kappa_max=args.jac_kappa_max,
            )
            u = u_nets[i][lidx](xi_var)
            grad_u_xi = gradient_tensor(u, xi_var, create_graph=False)
            grad_u_x = torch.bmm(grad_u_xi, inv_j)
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            Fm = eye + grad_u_x
            samples.append((Fm.detach(), n_s.detach()))

    cols = []
    for pidx in range(4):
        d = eps_rel[pidx] * max(abs(theta0[pidx]), 1e-4)
        tp = theta0.copy()
        tm = theta0.copy()
        tp[pidx] += d
        tm[pidx] -= d

        vec_p = []
        vec_m = []
        for Fm, n_s in samples:
            Pp = ti_arruda_boyce_p(
                F=Fm.clone().detach().requires_grad_(True),
                mu_m=torch.as_tensor(tp[0], device=device, dtype=dtype),
                N_m=torch.as_tensor(tp[1], device=device, dtype=dtype),
                k_f=torch.as_tensor(tp[2], device=device, dtype=dtype),
                N_f=torch.as_tensor(tp[3], device=device, dtype=dtype),
                kappa=args.kappa_fixed,
                a0=a0,
            )
            Pm = ti_arruda_boyce_p(
                F=Fm.clone().detach().requires_grad_(True),
                mu_m=torch.as_tensor(tm[0], device=device, dtype=dtype),
                N_m=torch.as_tensor(tm[1], device=device, dtype=dtype),
                k_f=torch.as_tensor(tm[2], device=device, dtype=dtype),
                N_f=torch.as_tensor(tm[3], device=device, dtype=dtype),
                kappa=args.kappa_fixed,
                a0=a0,
            )
            t_p = torch.bmm(Pp, n_s.unsqueeze(-1)).squeeze(-1)
            t_m = torch.bmm(Pm, n_s.unsqueeze(-1)).squeeze(-1)
            vec_p.append(t_p.detach().cpu().numpy().reshape(-1))
            vec_m.append(t_m.detach().cpu().numpy().reshape(-1))

        col = (np.concatenate(vec_p) - np.concatenate(vec_m)) / (2.0 * d)
        cols.append(col)

    J = np.stack(cols, axis=1)
    _, svals, _ = np.linalg.svd(J, full_matrices=False)
    rank = int(np.sum(svals > (svals[0] * 1e-10 if svals.size else 0.0)))
    cond = float(svals[0] / max(svals[-1], 1e-16)) if svals.size else float("inf")

    report = {
        "sensitivity_type": "effective_dtraction_dtheta_at_current_u",
        "rank": rank,
        "condition_number": cond,
        "singular_values": [float(x) for x in svals.tolist()],
        "passes_rank4": bool(rank >= 4),
        "passes_cond": bool(cond <= args.sensitivity_cond_max),
        "pass": bool(rank >= 4 and cond <= args.sensitivity_cond_max),
    }
    return report


# ----------------------------- Export -----------------------------


def write_vtu_points(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    n0 = int(points.shape[0])
    if n0 <= 0:
        raise ValueError("points must be non-empty")

    valid = np.all(np.isfinite(points), axis=1)
    cleaned: Dict[str, np.ndarray] = {}
    for name, arr in point_data.items():
        a = np.asarray(arr)
        if a.shape[0] != n0:
            raise ValueError(f"point_data '{name}' first dimension must be {n0}, got {a.shape[0]}")
        if a.ndim == 1:
            finite = np.isfinite(a)
        elif a.ndim == 2:
            finite = np.all(np.isfinite(a), axis=1)
        else:
            raise ValueError(f"Unsupported point_data shape for '{name}': {a.shape}")
        valid &= finite
        cleaned[name] = a

    if int(np.count_nonzero(valid)) <= 0:
        raise ValueError(f"No finite rows for VTU export: {path}")

    points = points[valid].astype(np.float32, copy=False)
    for k in list(cleaned.keys()):
        a = cleaned[k][valid]
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32, copy=False)
        cleaned[k] = a
    n = int(points.shape[0])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)

    def vtk_type(arr: np.ndarray) -> str:
        if np.issubdtype(arr.dtype, np.integer):
            return "Int32" if arr.dtype.itemsize <= 4 else "Int64"
        return "Float32" if arr.dtype == np.float32 else "Float64"

    def write_arr(fh, arr: np.ndarray) -> None:
        flat = np.asarray(arr).reshape(-1)
        chunk = 24
        for i in range(0, flat.size, chunk):
            fh.write("          " + " ".join(map(str, flat[i : i + chunk])) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write(f'        <DataArray type="{vtk_type(points)}" NumberOfComponents="3" format="ascii">\n')
        write_arr(f, points)
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        write_arr(f, connectivity)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        write_arr(f, offsets)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        write_arr(f, cell_types)
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in cleaned.items():
            ncomp = 1 if arr.ndim == 1 else int(arr.shape[1])
            f.write(
                f'        <DataArray type="{vtk_type(arr)}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            write_arr(f, arr)
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def compute_blended_and_dominant(
    x: torch.Tensor,
    nets_level: Sequence[LocalVectorPINN],
    masks: Sequence[MaskNet],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    blend_temp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = []
    ups = []
    for i in range(len(nets_level)):
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        li = masks[i](xi, chart_scale=chart_scales[i])
        ui = nets_level[i](xi)
        logits.append(li)
        ups.append(ui)

    L = torch.stack(logits, dim=1)
    W = torch.softmax(L / max(blend_temp, 1e-6), dim=1)
    U = torch.stack(ups, dim=1)

    u_blend = torch.sum(W.unsqueeze(-1) * U, dim=1)
    chart_id = torch.argmax(W, dim=1)
    wmax = torch.max(W, dim=1).values

    # dominant chart prediction for seam diagnostics
    npts = x.shape[0]
    u_dom = torch.zeros((npts, 3), device=x.device, dtype=x.dtype)
    for i in range(len(nets_level)):
        m = chart_id == i
        if bool(torch.any(m).item()):
            u_dom[m] = U[m, i, :]

    return u_blend, u_dom, chart_id, wmax


def make_plot(history: Dict[str, List[float]], out_png: str) -> None:
    ep = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].semilogy(ep, np.maximum(history["total"], 1e-16), label="total")
    axes[0, 0].semilogy(ep, np.maximum(history["eq"], 1e-16), label="eq")
    axes[0, 0].semilogy(ep, np.maximum(history["trac_bc"], 1e-16), label="trac_bc")
    axes[0, 0].semilogy(ep, np.maximum(history["data"], 1e-16), label="data")
    axes[0, 0].set_title("Core Losses")
    axes[0, 0].set_xlabel("Iter")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(ep, np.maximum(history["if_val"], 1e-16), label="if_val")
    axes[0, 1].semilogy(ep, np.maximum(history["if_flux"], 1e-16), label="if_flux")
    axes[0, 1].semilogy(ep, np.maximum(history["rbm"], 1e-16), label="rbm")
    axes[0, 1].semilogy(ep, np.maximum(history["reg"], 1e-16), label="reg")
    axes[0, 1].plot(ep, history["if_jump"], label="if_jump")
    axes[0, 1].set_title("Interface / Regularization")
    axes[0, 1].set_xlabel("Iter")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ep, history["mu_m"], label="mu_m")
    axes[1, 0].plot(ep, history["N_m"], label="N_m")
    axes[1, 0].plot(ep, history["k_f"], label="k_f")
    axes[1, 0].plot(ep, history["N_f"], label="N_f")
    axes[1, 0].set_title("Estimated Parameters")
    axes[1, 0].set_xlabel("Iter")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ep, history["surf_rel_l2"], label="surf_rel_l2")
    axes[1, 1].plot(ep, history["err_mu_m_pct"], label="mu_m%")
    axes[1, 1].plot(ep, history["err_N_m_pct"], label="N_m%")
    axes[1, 1].plot(ep, history["err_k_f_pct"], label="k_f%")
    axes[1, 1].plot(ep, history["err_N_f_pct"], label="N_f%")
    axes[1, 1].set_title("Fit/Error Diagnostics")
    axes[1, 1].set_xlabel("Iter")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Stage solver -----------------------------


def run_stage(
    stage_name: str,
    n_iters: int,
    u_nets: List[List[LocalVectorPINN]],
    chart_opts: List[torch.optim.Optimizer],
    ti_params: Optional[TIParams],
    theta_opt: Optional[torch.optim.Optimizer],
    fixed_params: Optional[Tuple[float, float, float, float]],
    caches: List[ChartCache],
    pair_caches: Dict[Tuple[int, int], PairCache],
    pair_scores_seed: Dict[Tuple[int, int], float],
    decoders: Sequence[ChartDecoder],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    levels: Sequence[LoadLevel],
    a0: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    include_data: bool,
    data_level_energy: Optional[Sequence[float]],
    ramp_w_if_val: Tuple[float, float],
    ramp_w_if_flux: Tuple[float, float],
    ramp_w_eq: Tuple[float, float],
    w_trac_bc: float,
    w_data: float,
    w_rbm: float,
    w_reg: float,
    omega: float,
    do_interface_gate: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, object], Dict[Tuple[int, int], float]]:
    n_charts = len(caches)

    history: Dict[str, List[float]] = {
        "total": [],
        "eq": [],
        "trac_bc": [],
        "data": [],
        "if_val": [],
        "if_flux": [],
        "rbm": [],
        "reg": [],
        "if_jump": [],
        "surf_rel_l2": [],
        "mu_m": [],
        "N_m": [],
        "k_f": [],
        "N_f": [],
        "err_mu_m_pct": [],
        "err_N_m_pct": [],
        "err_k_f_pct": [],
        "err_N_f_pct": [],
        "iter_rejected": [],
        "lr_u": [],
        "lr_theta_fast": [],
    }

    pair_scores = dict(pair_scores_seed)
    pair_scores_norm = compute_pair_scores_norm(pair_scores)

    best_data = {"score": float("inf"), "iter": 0, "u": chart_u_states(u_nets), "ti": None}
    best_if = {"score": float("inf"), "iter": 0, "u": chart_u_states(u_nets), "ti": None}
    best_combo = {"score": float("inf"), "iter": 0, "u": chart_u_states(u_nets), "ti": None}

    def ti_snapshot() -> Optional[Dict[str, torch.Tensor]]:
        if ti_params is None:
            return None
        return {k: v.detach().clone() for k, v in ti_params.state_dict().items()}

    def ti_restore(st: Optional[Dict[str, torch.Tensor]]) -> None:
        if ti_params is not None and st is not None:
            ti_params.load_state_dict(st)

    best_data["ti"] = ti_snapshot()
    best_if["ti"] = ti_snapshot()
    best_combo["ti"] = ti_snapshot()

    if_jump_ref = float("inf")

    t_start = time.time()

    for it in range(1, n_iters + 1):
        frac = float(it) / float(max(n_iters, 1))
        w_if_val_eff = lerp(ramp_w_if_val[0], ramp_w_if_val[1], frac)
        w_if_flux_eff = lerp(ramp_w_if_flux[0], ramp_w_if_flux[1], frac)
        w_eq_eff = lerp(ramp_w_eq[0], ramp_w_eq[1], frac)

        full_u_snapshot = chart_u_states(u_nets)
        full_ti_snapshot = ti_snapshot()

        eq_acc = 0.0
        trac_acc = 0.0
        data_acc = 0.0
        ifv_acc = 0.0
        iff_acc = 0.0
        rbm_acc = 0.0
        reg_acc = 0.0
        n_updates = 0

        for group in args.color_groups_runtime:
            for i in group:
                old_chart_states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in u_nets[i]]

                for _ in range(args.local_steps):
                    chart_opts[i].zero_grad()
                    if theta_opt is not None:
                        theta_opt.zero_grad()

                    if ti_params is None:
                        assert fixed_params is not None
                        mu_m = torch.as_tensor(fixed_params[0], device=device, dtype=dtype)
                        N_m = torch.as_tensor(fixed_params[1], device=device, dtype=dtype)
                        k_f = torch.as_tensor(fixed_params[2], device=device, dtype=dtype)
                        N_f = torch.as_tensor(fixed_params[3], device=device, dtype=dtype)
                    else:
                        mu_m, N_m, k_f, N_f = ti_params.values()

                    loss_eq = torch.zeros((), device=device, dtype=dtype)
                    loss_trac = torch.zeros((), device=device, dtype=dtype)
                    loss_data = torch.zeros((), device=device, dtype=dtype)
                    loss_if_val = torch.zeros((), device=device, dtype=dtype)
                    loss_if_flux = torch.zeros((), device=device, dtype=dtype)
                    loss_rbm = torch.zeros((), device=device, dtype=dtype)

                    c_i = caches[i]

                    for lidx, lev in enumerate(levels):
                        # interior PDE residual (no body force)
                        sel_int = sample_indices(c_i.xi_int.shape[0], args.pde_batch, device=device)
                        xi_int = c_i.xi_int[sel_int]
                        _, xi_var, jac = chart_map_and_jacobian(
                            decoder=decoders[i],
                            xi_in=xi_int,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=chart_scales[i],
                        )
                        inv_j, det_abs, _, valid = stabilized_jacobian_ops(
                            jac=jac,
                            sigma_floor=args.sigma_floor,
                            det_floor=args.detJ_floor,
                            jac_kappa_max=args.jac_kappa_max,
                        )

                        u_int = u_nets[i][lidx](xi_var)
                        grad_u_xi = gradient_tensor(u_int, xi_var, create_graph=True)
                        grad_u_x = torch.bmm(grad_u_xi, inv_j)
                        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
                        Fm = eye + grad_u_x
                        Pm = ti_arruda_boyce_p(F=Fm, mu_m=mu_m, N_m=N_m, k_f=k_f, N_f=N_f, kappa=args.kappa_fixed, a0=a0)
                        mapped_flux = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(Pm, inv_j.transpose(1, 2))
                        div_flux = divergence_mapped(mapped_flux, xi_var, create_graph=True)
                        res = torch.nan_to_num(div_flux, nan=0.0, posinf=0.0, neginf=0.0)
                        if args.eq_residual_clip > 0.0:
                            clip_t = torch.as_tensor(args.eq_residual_clip, device=device, dtype=dtype)
                            res = torch.clamp(res, min=-clip_t, max=clip_t)

                        if bool(torch.any(valid).item()):
                            loss_eq = loss_eq + torch.mean(res[valid] ** 2)
                        else:
                            loss_eq = loss_eq + torch.mean(res**2)

                        # surface traction BC
                        sel_s = sample_indices(c_i.xi_surf.shape[0], args.bc_batch, device=device)
                        xi_s = c_i.xi_surf[sel_s]
                        n_s = c_i.n_surf[sel_s]
                        a_s = c_i.alpha_surf[sel_s].unsqueeze(1)

                        _, xi_s_var, jac_s = chart_map_and_jacobian(
                            decoder=decoders[i],
                            xi_in=xi_s,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=chart_scales[i],
                        )
                        inv_s, _, _, _ = stabilized_jacobian_ops(
                            jac=jac_s,
                            sigma_floor=args.sigma_floor,
                            det_floor=args.detJ_floor,
                            jac_kappa_max=args.jac_kappa_max,
                        )

                        u_s = u_nets[i][lidx](xi_s_var)
                        grad_s_xi = gradient_tensor(u_s, xi_s_var, create_graph=True)
                        grad_s_x = torch.bmm(grad_s_xi, inv_s)
                        Fs = eye + grad_s_x
                        Ps = ti_arruda_boyce_p(F=Fs, mu_m=mu_m, N_m=N_m, k_f=k_f, N_f=N_f, kappa=args.kappa_fixed, a0=a0)
                        t_pred = torch.bmm(Ps, n_s.unsqueeze(-1)).squeeze(-1)
                        t_tar = (lev.lam * args.t0) * a_s * n_s
                        loss_trac = loss_trac + torch.mean((t_pred - t_tar) ** 2)

                        rbm_i = rigid_body_penalty(x=c_i.x_surf[sel_s], u=u_s)
                        loss_rbm = loss_rbm + rbm_i

                        if include_data:
                            u_true = c_i.u_teacher_by_level[lidx][sel_s]
                            d = u_s - u_true
                            d_loss = robust_huber(d, delta=args.huber_delta)
                            if data_level_energy is not None:
                                d_loss = d_loss / max(float(data_level_energy[lidx]), 1e-12)
                            loss_data = loss_data + d_loss

                        # interface terms for neighbor pairs touching chart i
                        for j in args.neighbors_runtime.get(i, []):
                            key = (i, j) if (i, j) in pair_caches else ((j, i) if (j, i) in pair_caches else None)
                            if key is None:
                                continue
                            pc = pair_caches[key]
                            if key[0] == i:
                                xi_i_all = pc.xi_i
                                xi_j_all = pc.xi_j
                                j_idx = j
                            else:
                                xi_i_all = pc.xi_j
                                xi_j_all = pc.xi_i
                                j_idx = j

                            s_norm = pair_scores_norm.get(key, 0.0)
                            n_if_eff = int(round(args.if_batch * (1.0 + args.if_hard_oversample * s_norm)))
                            n_if_eff = max(8, n_if_eff)

                            sel_if = sample_indices(xi_i_all.shape[0], min(n_if_eff, xi_i_all.shape[0]), device=device)
                            xi_i = xi_i_all[sel_if]
                            xi_j = xi_j_all[sel_if]

                            _, xi_i_var, jac_i = chart_map_and_jacobian(
                                decoder=decoders[i],
                                xi_in=xi_i,
                                seed=seeds[i],
                                t1=t1[i],
                                t2=t2[i],
                                n=nvec[i],
                                chart_scale=chart_scales[i],
                            )
                            inv_i, _, _, _ = stabilized_jacobian_ops(
                                jac=jac_i,
                                sigma_floor=args.sigma_floor,
                                det_floor=args.detJ_floor,
                                jac_kappa_max=args.jac_kappa_max,
                            )

                            _, xi_j_var, jac_j = chart_map_and_jacobian(
                                decoder=decoders[j_idx],
                                xi_in=xi_j,
                                seed=seeds[j_idx],
                                t1=t1[j_idx],
                                t2=t2[j_idx],
                                n=nvec[j_idx],
                                chart_scale=chart_scales[j_idx],
                            )
                            inv_jj, _, _, _ = stabilized_jacobian_ops(
                                jac=jac_j,
                                sigma_floor=args.sigma_floor,
                                det_floor=args.detJ_floor,
                                jac_kappa_max=args.jac_kappa_max,
                            )

                            ui = u_nets[i][lidx](xi_i_var)
                            gi = torch.bmm(gradient_tensor(ui, xi_i_var, create_graph=True), inv_i)
                            uj = u_nets[j_idx][lidx](xi_j_var)
                            gj = torch.bmm(gradient_tensor(uj, xi_j_var, create_graph=False), inv_jj)
                            loss_if_val = loss_if_val + torch.mean((ui - uj.detach()) ** 2)
                            loss_if_flux = loss_if_flux + torch.mean((gi - gj.detach()) ** 2)

                    denom_levels = float(max(len(levels), 1))
                    loss_eq = loss_eq / denom_levels
                    loss_trac = loss_trac / denom_levels
                    loss_data = loss_data / denom_levels
                    loss_if_val = loss_if_val / denom_levels
                    loss_if_flux = loss_if_flux / denom_levels
                    loss_rbm = loss_rbm / denom_levels

                    if ti_params is None:
                        loss_reg = torch.zeros((), device=device, dtype=dtype)
                    else:
                        mu_m_log = torch.log(torch.clamp(mu_m, min=torch.as_tensor(1e-8, device=device, dtype=dtype)))
                        N_m_log = torch.log(torch.clamp(N_m, min=torch.as_tensor(1e-8, device=device, dtype=dtype)))
                        k_f_log = torch.log(torch.clamp(k_f, min=torch.as_tensor(1e-8, device=device, dtype=dtype)))
                        N_f_log = torch.log(torch.clamp(N_f, min=torch.as_tensor(1e-8, device=device, dtype=dtype)))
                        loss_reg = (
                            (mu_m_log - math.log(args.mu_m_prior)) ** 2
                            + (N_m_log - math.log(args.N_m_prior)) ** 2
                            + (k_f_log - math.log(args.k_f_prior)) ** 2
                            + (N_f_log - math.log(args.N_f_prior)) ** 2
                        )

                    loss = (
                        w_eq_eff * loss_eq
                        + w_trac_bc * loss_trac
                        + w_data * loss_data
                        + w_if_val_eff * loss_if_val
                        + w_if_flux_eff * loss_if_flux
                        + w_rbm * loss_rbm
                        + w_reg * loss_reg
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u_nets[i][0].parameters(), max_norm=args.grad_clip)
                    if ti_params is not None:
                        torch.nn.utils.clip_grad_norm_(ti_params.parameters(), max_norm=args.grad_clip_theta)

                    chart_opts[i].step()
                    if theta_opt is not None:
                        theta_opt.step()

                    eq_acc += float(loss_eq.detach().item())
                    trac_acc += float(loss_trac.detach().item())
                    data_acc += float(loss_data.detach().item())
                    ifv_acc += float(loss_if_val.detach().item())
                    iff_acc += float(loss_if_flux.detach().item())
                    rbm_acc += float(loss_rbm.detach().item())
                    reg_acc += float(loss_reg.detach().item())
                    n_updates += 1

                # multiplicative Schwarz relaxation on updated chart
                with torch.no_grad():
                    for lidx in range(len(levels)):
                        curr = u_nets[i][lidx].state_dict()
                        blended = {}
                        for k, v in curr.items():
                            blended[k] = (1.0 - omega) * old_chart_states[lidx][k] + omega * v
                        u_nets[i][lidx].load_state_dict(blended)

        mean_if_val, mean_if_flux, pair_scores_now = compute_interface_metrics(
            u_nets=u_nets,
            decoders=decoders,
            pair_caches=pair_caches,
            seeds=seeds,
            t1=t1,
            t2=t2,
            nvec=nvec,
            chart_scales=chart_scales,
            levels=levels,
            args=args,
        )
        if_jump = mean_if_val + args.if_score_flux_factor * mean_if_flux

        rejected = 0
        if do_interface_gate and if_jump_ref < float("inf"):
            if if_jump > if_jump_ref * (1.0 + args.if_gate_rel_tol):
                load_chart_u_states(u_nets, full_u_snapshot)
                ti_restore(full_ti_snapshot)
                for opt in chart_opts:
                    adjust_optimizer_lr(opt, args.if_gate_lr_decay)
                if theta_opt is not None:
                    adjust_optimizer_lr(theta_opt, args.if_gate_lr_decay)
                rejected = 1
            else:
                if_jump_ref = min(if_jump_ref, if_jump)
        else:
            if_jump_ref = if_jump

        if rejected == 0:
            pair_scores = pair_scores_now
            pair_scores_norm = compute_pair_scores_norm(pair_scores)

        if ti_params is None:
            mu_m_v, N_m_v, k_f_v, N_f_v = fixed_params if fixed_params is not None else (0.0, 0.0, 0.0, 0.0)
        else:
            mu_t, Nm_t, kf_t, Nf_t = ti_params.values()
            mu_m_v = scalar(mu_t, device)
            N_m_v = scalar(Nm_t, device)
            k_f_v = scalar(kf_t, device)
            N_f_v = scalar(Nf_t, device)

        if include_data:
            surf_rel = compute_surface_rel_l2(u_nets=u_nets, caches=caches, levels=levels)
        else:
            surf_rel = 0.0

        eq_v = eq_acc / max(n_updates, 1)
        trac_v = trac_acc / max(n_updates, 1)
        data_v = data_acc / max(n_updates, 1)
        ifv_v = ifv_acc / max(n_updates, 1)
        iff_v = iff_acc / max(n_updates, 1)
        rbm_v = rbm_acc / max(n_updates, 1)
        reg_v = reg_acc / max(n_updates, 1)
        total_v = (
            w_eq_eff * eq_v
            + w_trac_bc * trac_v
            + w_data * data_v
            + w_if_val_eff * ifv_v
            + w_if_flux_eff * iff_v
            + w_rbm * rbm_v
            + w_reg * reg_v
        )

        err_mu = 100.0 * abs(mu_m_v - args.mu_m_true) / max(args.mu_m_true, 1e-12)
        err_Nm = 100.0 * abs(N_m_v - args.N_m_true) / max(args.N_m_true, 1e-12)
        err_kf = 100.0 * abs(k_f_v - args.k_f_true) / max(args.k_f_true, 1e-12)
        err_Nf = 100.0 * abs(N_f_v - args.N_f_true) / max(args.N_f_true, 1e-12)

        history["total"].append(total_v)
        history["eq"].append(eq_v)
        history["trac_bc"].append(trac_v)
        history["data"].append(data_v)
        history["if_val"].append(ifv_v)
        history["if_flux"].append(iff_v)
        history["rbm"].append(rbm_v)
        history["reg"].append(reg_v)
        history["if_jump"].append(if_jump)
        history["surf_rel_l2"].append(surf_rel)
        history["mu_m"].append(mu_m_v)
        history["N_m"].append(N_m_v)
        history["k_f"].append(k_f_v)
        history["N_f"].append(N_f_v)
        history["err_mu_m_pct"].append(err_mu)
        history["err_N_m_pct"].append(err_Nm)
        history["err_k_f_pct"].append(err_kf)
        history["err_N_f_pct"].append(err_Nf)
        history["iter_rejected"].append(float(rejected))
        history["lr_u"].append(float(chart_opts[0].param_groups[0]["lr"]))
        if theta_opt is not None:
            history["lr_theta_fast"].append(float(theta_opt.param_groups[0]["lr"]))
        else:
            history["lr_theta_fast"].append(0.0)

        # checkpoint criteria
        data_score = surf_rel if include_data else total_v
        if data_score < best_data["score"]:
            best_data["score"] = data_score
            best_data["iter"] = it
            best_data["u"] = chart_u_states(u_nets)
            best_data["ti"] = ti_snapshot()

        if if_jump < best_if["score"]:
            best_if["score"] = if_jump
            best_if["iter"] = it
            best_if["u"] = chart_u_states(u_nets)
            best_if["ti"] = ti_snapshot()

        combo = data_score + args.combo_if_weight * if_jump + args.combo_param_weight * (err_mu + err_Nm + err_kf + err_Nf)
        if combo < best_combo["score"]:
            best_combo["score"] = combo
            best_combo["iter"] = it
            best_combo["u"] = chart_u_states(u_nets)
            best_combo["ti"] = ti_snapshot()

        elapsed = time.time() - t_start
        print(
            f"[{stage_name}] iter={it:03d}/{n_iters} total={total_v:.3e} eq={eq_v:.3e} trac={trac_v:.3e} "
            f"data={data_v:.3e} ifV={ifv_v:.3e} ifF={iff_v:.3e} ifJump={if_jump:.3e} "
            f"mu={mu_m_v:.6f} ({err_mu:.2f}%) Nm={N_m_v:.6f} ({err_Nm:.2f}%) "
            f"kf={k_f_v:.6f} ({err_kf:.2f}%) Nf={N_f_v:.6f} ({err_Nf:.2f}%) "
            f"surf_relL2={surf_rel:.3e} rej={rejected} t={elapsed:.1f}s"
        )

    # Restore best composite state at stage end.
    load_chart_u_states(u_nets, best_combo["u"])
    ti_restore(best_combo["ti"])

    summary = {
        "stage": stage_name,
        "iters": n_iters,
        "best_data_iter": int(best_data["iter"]),
        "best_interface_iter": int(best_if["iter"]),
        "best_combo_iter": int(best_combo["iter"]),
        "best_data_score": float(best_data["score"]),
        "best_interface_score": float(best_if["score"]),
        "best_combo_score": float(best_combo["score"]),
        "final_if_jump": float(history["if_jump"][-1]) if history["if_jump"] else None,
        "final_surf_rel_l2": float(history["surf_rel_l2"][-1]) if history["surf_rel_l2"] else None,
    }

    best_states = {
        "best_data": {"u": best_data["u"], "ti": best_data["ti"]},
        "best_interface": {"u": best_if["u"], "ti": best_if["ti"]},
        "best_combo": {"u": best_combo["u"], "ti": best_combo["ti"]},
    }

    return history, {"summary": summary, "best_states": best_states}, pair_scores


# ----------------------------- Main -----------------------------


def run(args: argparse.Namespace) -> Dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.set_default_dtype(dtype)
    set_seed(args.seed)

    if args.amp and device.type == "mps":
        print("Requested AMP on MPS; disabling AMP for stability.")
        args.amp = False

    print(f"Device={device.type} dtype={dtype} amp={args.amp}")

    atlas = np.load(args.atlas_data)
    points = torch.from_numpy(atlas["points"]).to(device=device, dtype=dtype)
    normals = torch.from_numpy(atlas["normals"]).to(device=device, dtype=dtype)
    seeds = torch.from_numpy(atlas["seed_points"]).to(device=device, dtype=dtype)
    t1 = torch.from_numpy(atlas["frame_t1"]).to(device=device, dtype=dtype)
    t2 = torch.from_numpy(atlas["frame_t2"]).to(device=device, dtype=dtype)
    nvec = torch.from_numpy(atlas["frame_n"]).to(device=device, dtype=dtype)
    membership = torch.from_numpy(atlas["membership"].astype(np.float32)).to(device=device, dtype=dtype)
    chart_scales = torch.from_numpy(atlas["support_radii"]).to(device=device, dtype=dtype).unsqueeze(1)

    n_charts = int(seeds.shape[0])
    neighbors, color_groups = parse_neighbors(args.atlas_meta, n_charts=n_charts)
    args.neighbors_runtime = neighbors
    args.color_groups_runtime = color_groups

    decoders, masks, _ = load_atlas_models(args.atlas_checkpoint, device=device, dtype=dtype)

    levels = make_load_levels(args.load_lambdas)
    if len(levels) < 1:
        raise RuntimeError("No load levels configured.")

    points_n = points
    a0 = normalize_rows(torch.tensor([[args.fiber_a0_x, args.fiber_a0_y, args.fiber_a0_z]], device=device, dtype=dtype))[0]

    caches = prepare_chart_caches(
        args=args,
        points=points_n,
        normals=normals,
        membership=membership,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        n_levels=len(levels),
        device=device,
        dtype=dtype,
    )

    pair_caches = prepare_pair_caches(
        args=args,
        membership=membership,
        points=points_n,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        neighbors=neighbors,
        device=device,
    )

    # Build per-chart/per-level networks.
    u_nets: List[List[LocalVectorPINN]] = []
    chart_opts: List[torch.optim.Optimizer] = []
    for i in range(n_charts):
        nets_i = [LocalVectorPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in levels]
        u_nets.append(nets_i)
        params_i: List[torch.nn.Parameter] = []
        for m in nets_i:
            params_i.extend(list(m.parameters()))
        chart_opts.append(torch.optim.Adam(params_i, lr=args.lr_u))

    # ---------------- Stage T: teacher forward ----------------
    print(f"Teacher stage: levels={len(levels)} iters={args.teacher_iters}")

    teacher_hist, teacher_pack, pair_scores = run_stage(
        stage_name="Teacher",
        n_iters=args.teacher_iters,
        u_nets=u_nets,
        chart_opts=chart_opts,
        ti_params=None,
        theta_opt=None,
        fixed_params=(args.mu_m_true, args.N_m_true, args.k_f_true, args.N_f_true),
        caches=caches,
        pair_caches=pair_caches,
        pair_scores_seed={},
        decoders=decoders,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        a0=a0,
        args=args,
        device=device,
        dtype=dtype,
        include_data=False,
        data_level_energy=None,
        ramp_w_if_val=(args.w_teacher_if_val_start, args.w_teacher_if_val_end),
        ramp_w_if_flux=(args.w_teacher_if_flux_start, args.w_teacher_if_flux_end),
        ramp_w_eq=(args.w_teacher_eq_start, args.w_teacher_eq_end),
        w_trac_bc=args.w_teacher_trac_bc,
        w_data=0.0,
        w_rbm=args.w_teacher_rbm,
        w_reg=0.0,
        omega=args.omega_teacher,
        do_interface_gate=False,
    )

    # Freeze teacher networks for later truth-field export and diagnostics.
    teacher_nets: List[List[LocalVectorPINN]] = []
    for i in range(n_charts):
        row: List[LocalVectorPINN] = []
        for lidx in range(len(levels)):
            m = copy.deepcopy(u_nets[i][lidx]).to(device=device, dtype=dtype)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            row.append(m)
        teacher_nets.append(row)

    # collect teacher surface displacement observations
    for i in range(n_charts):
        for lidx in range(len(levels)):
            with torch.no_grad():
                caches[i].u_teacher_by_level[lidx] = u_nets[i][lidx](caches[i].xi_surf).detach()

    # ---------------- Sensitivity gate at init ----------------
    sens_init = effective_sensitivity_report(
        u_nets=u_nets,
        decoders=decoders,
        caches=caches,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        a0=a0,
        args=args,
        device=device,
        dtype=dtype,
        params_center=[args.mu_m_init, args.N_m_init, args.k_f_init, args.N_f_init],
    )
    print(
        f"Sensitivity(init): rank={sens_init['rank']} cond={sens_init['condition_number']:.3e} "
        f"pass={sens_init['pass']}"
    )
    if args.strict_sensitivity_gate and (not sens_init["pass"]):
        raise RuntimeError("Sensitivity gate failed at initialization.")

    # ---------------- Inverse parameters ----------------
    ti_params = TIParams(
        mu_m_init=args.mu_m_init,
        N_m_init=args.N_m_init,
        k_f_init=args.k_f_init,
        N_f_init=args.N_f_init,
        mu_m_min=args.mu_m_min,
        N_m_min=args.N_m_min,
        k_f_min=args.k_f_min,
        N_f_min=args.N_f_min,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    level_energy = compute_level_energy(caches, levels)

    # ---------------- Stage A: identify mu_m + k_f ----------------
    theta_opt_A = make_theta_optimizer(ti_params, stage="A", args=args)
    print(f"Stage A: iters={args.stage_a_iters}, unknowns=(mu_m,k_f)")
    hist_A, pack_A, pair_scores = run_stage(
        stage_name="StageA",
        n_iters=args.stage_a_iters,
        u_nets=u_nets,
        chart_opts=chart_opts,
        ti_params=ti_params,
        theta_opt=theta_opt_A,
        fixed_params=None,
        caches=caches,
        pair_caches=pair_caches,
        pair_scores_seed=pair_scores,
        decoders=decoders,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        a0=a0,
        args=args,
        device=device,
        dtype=dtype,
        include_data=True,
        data_level_energy=level_energy,
        ramp_w_if_val=(args.w_if_val_start, args.w_if_val_end),
        ramp_w_if_flux=(args.w_if_flux_start, args.w_if_flux_end),
        ramp_w_eq=(args.w_eq_start, args.w_eq_end),
        w_trac_bc=args.w_trac_bc,
        w_data=args.w_data,
        w_rbm=args.w_rbm,
        w_reg=args.w_reg,
        omega=args.omega,
        do_interface_gate=True,
    )

    # ---------------- Sensitivity gate after Stage A ----------------
    mu_a, Nm_a, kf_a, Nf_a = ti_params.values()
    sens_after_A = effective_sensitivity_report(
        u_nets=u_nets,
        decoders=decoders,
        caches=caches,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        a0=a0,
        args=args,
        device=device,
        dtype=dtype,
        params_center=[scalar(mu_a, device), scalar(Nm_a, device), scalar(kf_a, device), scalar(Nf_a, device)],
    )
    print(
        f"Sensitivity(after A): rank={sens_after_A['rank']} cond={sens_after_A['condition_number']:.3e} "
        f"pass={sens_after_A['pass']}"
    )
    if args.strict_sensitivity_gate and (not sens_after_A["pass"]):
        raise RuntimeError("Sensitivity gate failed after Stage A.")

    # ---------------- Stage B: unlock all 4 ----------------
    theta_opt_B = make_theta_optimizer(ti_params, stage="B", args=args)
    print(f"Stage B: iters={args.stage_b_iters}, unknowns=(mu_m,N_m,k_f,N_f)")
    hist_B, pack_B, pair_scores = run_stage(
        stage_name="StageB",
        n_iters=args.stage_b_iters,
        u_nets=u_nets,
        chart_opts=chart_opts,
        ti_params=ti_params,
        theta_opt=theta_opt_B,
        fixed_params=None,
        caches=caches,
        pair_caches=pair_caches,
        pair_scores_seed=pair_scores,
        decoders=decoders,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        a0=a0,
        args=args,
        device=device,
        dtype=dtype,
        include_data=True,
        data_level_energy=level_energy,
        ramp_w_if_val=(args.w_if_val_stage_b_start, args.w_if_val_stage_b_end),
        ramp_w_if_flux=(args.w_if_flux_stage_b_start, args.w_if_flux_stage_b_end),
        ramp_w_eq=(args.w_eq_stage_b_start, args.w_eq_stage_b_end),
        w_trac_bc=args.w_trac_bc,
        w_data=args.w_data,
        w_rbm=args.w_rbm,
        w_reg=args.w_reg,
        omega=args.omega,
        do_interface_gate=True,
    )

    # Optional Stage C polish
    hist_C = None
    pack_C = None
    if args.stage_c_iters > 0:
        print(f"Stage C polish: iters={args.stage_c_iters}")
        theta_opt_C = make_theta_optimizer(ti_params, stage="B", args=args)
        hist_C, pack_C, pair_scores = run_stage(
            stage_name="StageC",
            n_iters=args.stage_c_iters,
            u_nets=u_nets,
            chart_opts=chart_opts,
            ti_params=ti_params,
            theta_opt=theta_opt_C,
            fixed_params=None,
            caches=caches,
            pair_caches=pair_caches,
            pair_scores_seed=pair_scores,
            decoders=decoders,
            seeds=seeds,
            t1=t1,
            t2=t2,
            nvec=nvec,
            chart_scales=chart_scales,
            levels=levels,
            a0=a0,
            args=args,
            device=device,
            dtype=dtype,
            include_data=True,
            data_level_energy=level_energy,
            ramp_w_if_val=(args.w_if_val_stage_c_start, args.w_if_val_stage_c_end),
            ramp_w_if_flux=(args.w_if_flux_stage_c_start, args.w_if_flux_stage_c_end),
            ramp_w_eq=(args.w_eq_stage_c_start, args.w_eq_stage_c_end),
            w_trac_bc=args.w_trac_bc,
            w_data=args.w_data,
            w_rbm=args.w_rbm,
            w_reg=args.w_reg,
            omega=args.omega,
            do_interface_gate=True,
        )

    # Final parameter values
    mu_f, Nm_f, kf_f, Nf_f = ti_params.values()
    mu_v = scalar(mu_f, device)
    Nm_v = scalar(Nm_f, device)
    kf_v = scalar(kf_f, device)
    Nf_v = scalar(Nf_f, device)

    err_mu = 100.0 * abs(mu_v - args.mu_m_true) / max(args.mu_m_true, 1e-12)
    err_Nm = 100.0 * abs(Nm_v - args.N_m_true) / max(args.N_m_true, 1e-12)
    err_kf = 100.0 * abs(kf_v - args.k_f_true) / max(args.k_f_true, 1e-12)
    err_Nf = 100.0 * abs(Nf_v - args.N_f_true) / max(args.N_f_true, 1e-12)

    # final interface score
    mean_if_val_f, mean_if_flux_f, pair_scores_f = compute_interface_metrics(
        u_nets=u_nets,
        decoders=decoders,
        pair_caches=pair_caches,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        levels=levels,
        args=args,
    )
    if_jump_final = mean_if_val_f + args.if_score_flux_factor * mean_if_flux_f

    surf_rel_final = compute_surface_rel_l2(u_nets=u_nets, caches=caches, levels=levels)

    # ----------- Exports -----------
    os.makedirs(args.output_dir, exist_ok=True)
    stem = build_run_stem(args.run_tag)

    # For per-point if_jump_score, assign chart-level average pair score.
    chart_pair_score = np.zeros((n_charts,), dtype=np.float32)
    chart_pair_count = np.zeros((n_charts,), dtype=np.float32)
    for (i, j), s in pair_scores_f.items():
        chart_pair_score[i] += float(s)
        chart_pair_score[j] += float(s)
        chart_pair_count[i] += 1.0
        chart_pair_count[j] += 1.0
    chart_pair_count = np.where(chart_pair_count > 0.0, chart_pair_count, 1.0)
    chart_pair_score = chart_pair_score / chart_pair_count

    # Build domain points from cached interior points.
    x_dom_parts = []
    for i in range(n_charts):
        n_take = min(args.export_points_per_chart, caches[i].xi_int.shape[0])
        sel = sample_indices(caches[i].xi_int.shape[0], n_take, device=device)
        xi = caches[i].xi_int[sel]
        with torch.no_grad():
            xx = decoders[i](xi, seed=seeds[i], t1=t1[i], t2=t2[i], n=nvec[i], chart_scale=chart_scales[i])
        x_dom_parts.append(xx)
    x_dom = torch.cat(x_dom_parts, dim=0)

    merged_points = []
    merged_data: Dict[str, List[np.ndarray]] = {
        "u_pred": [],
        "u_pred_dom": [],
        "u_true": [],
        "u_error": [],
        "u_error_mag": [],
        "chart_id": [],
        "chart_weight_max": [],
        "if_jump_score": [],
        "load_lambda": [],
        "load_id": [],
    }

    case_metrics = []

    for lidx, lev in enumerate(levels):
        # surface export
        x_surf = points_n
        with torch.no_grad():
            u_s_blend, u_s_dom, cid_s, wmax_s = compute_blended_and_dominant(
                x=x_surf,
                nets_level=[u_nets[i][lidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )
            u_s_true, u_s_true_dom, _, _ = compute_blended_and_dominant(
                x=x_surf,
                nets_level=[teacher_nets[i][lidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )

        # surface metrics
        u_s_err = u_s_blend - u_s_true
        surf_rel = torch.sqrt(
            torch.mean(u_s_err * u_s_err)
            / torch.clamp(torch.mean(u_s_true * u_s_true), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
        )
        case_metrics.append(
            {
                "name": lev.name,
                "lambda": float(lev.lam),
                "surface_rel_l2": float(surf_rel.detach().item()),
                "surface_max_abs": float(torch.max(torch.abs(u_s_err)).detach().item()),
                "n_surface": int(x_surf.shape[0]),
            }
        )

        if_score_s = torch.from_numpy(chart_pair_score).to(device=device, dtype=dtype)[cid_s]

        surf_vtu = os.path.join(args.output_dir, f"{stem}_{lev.name}_surface.vtu")
        write_vtu_points(
            surf_vtu,
            points=x_surf.detach().cpu().numpy(),
            point_data={
                "u_pred": u_s_blend.detach().cpu().numpy(),
                "u_pred_dom": u_s_dom.detach().cpu().numpy(),
                "u_true": u_s_true.detach().cpu().numpy(),
                "u_error": u_s_err.detach().cpu().numpy(),
                "u_error_mag": torch.linalg.norm(u_s_err, dim=1).detach().cpu().numpy(),
                "chart_id": cid_s.to(dtype=torch.float32).detach().cpu().numpy(),
                "chart_weight_max": wmax_s.detach().cpu().numpy(),
                "if_jump_score": if_score_s.detach().cpu().numpy(),
                "load_lambda": np.full((x_surf.shape[0],), float(lev.lam), dtype=np.float32),
                "load_id": np.full((x_surf.shape[0],), float(lidx), dtype=np.float32),
            },
        )

        # domain export for this level and merged-domain accumulation
        with torch.no_grad():
            u_d_blend, u_d_dom, cid_d, wmax_d = compute_blended_and_dominant(
                x=x_dom,
                nets_level=[u_nets[i][lidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )
            u_d_true, _, _, _ = compute_blended_and_dominant(
                x=x_dom,
                nets_level=[teacher_nets[i][lidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )

        u_d_err = u_d_blend - u_d_true
        if_score_d = torch.from_numpy(chart_pair_score).to(device=device, dtype=dtype)[cid_d]

        dom_vtu = os.path.join(args.output_dir, f"{stem}_{lev.name}_domain.vtu")
        write_vtu_points(
            dom_vtu,
            points=x_dom.detach().cpu().numpy(),
            point_data={
                "u_pred": u_d_blend.detach().cpu().numpy(),
                "u_pred_dom": u_d_dom.detach().cpu().numpy(),
                "u_true": u_d_true.detach().cpu().numpy(),
                "u_error": u_d_err.detach().cpu().numpy(),
                "u_error_mag": torch.linalg.norm(u_d_err, dim=1).detach().cpu().numpy(),
                "chart_id": cid_d.to(dtype=torch.float32).detach().cpu().numpy(),
                "chart_weight_max": wmax_d.detach().cpu().numpy(),
                "if_jump_score": if_score_d.detach().cpu().numpy(),
                "load_lambda": np.full((x_dom.shape[0],), float(lev.lam), dtype=np.float32),
                "load_id": np.full((x_dom.shape[0],), float(lidx), dtype=np.float32),
            },
        )

        merged_points.append(x_dom.detach().cpu().numpy())
        merged_data["u_pred"].append(u_d_blend.detach().cpu().numpy())
        merged_data["u_pred_dom"].append(u_d_dom.detach().cpu().numpy())
        merged_data["u_true"].append(u_d_true.detach().cpu().numpy())
        merged_data["u_error"].append(u_d_err.detach().cpu().numpy())
        merged_data["u_error_mag"].append(torch.linalg.norm(u_d_err, dim=1).detach().cpu().numpy())
        merged_data["chart_id"].append(cid_d.to(dtype=torch.float32).detach().cpu().numpy())
        merged_data["chart_weight_max"].append(wmax_d.detach().cpu().numpy())
        merged_data["if_jump_score"].append(if_score_d.detach().cpu().numpy())
        merged_data["load_lambda"].append(np.full((x_dom.shape[0],), float(lev.lam), dtype=np.float32))
        merged_data["load_id"].append(np.full((x_dom.shape[0],), float(lidx), dtype=np.float32))

    merged_points_np = np.concatenate(merged_points, axis=0)
    merged_arrays = {k: np.concatenate(v, axis=0) for k, v in merged_data.items()}
    merged_vtu = os.path.join(args.output_dir, f"{stem}_domain_merged.vtu")
    write_vtu_points(merged_vtu, points=merged_points_np, point_data=merged_arrays)

    # Save histories and checkpoints.
    hist_path = os.path.join(args.output_dir, f"{stem}_history.json")
    met_path = os.path.join(args.output_dir, f"{stem}_metrics.json")
    curve_path = os.path.join(args.output_dir, f"{stem}_curves.png")
    sens_path = os.path.join(args.output_dir, f"{stem}_sensitivity.json")
    ckpt_best_data = os.path.join(args.output_dir, f"{stem}_best_datafit.pt")
    ckpt_best_if = os.path.join(args.output_dir, f"{stem}_best_interface.pt")
    ckpt_best_combo = os.path.join(args.output_dir, f"{stem}_best_composite.pt")
    ckpt_final = os.path.join(args.output_dir, f"{stem}_final.pt")

    # Use Stage B best states for final checkpoint triplet exports.
    best_states_B = pack_B["best_states"]
    torch.save(
        {
            "u_states": best_states_B["best_data"]["u"],
            "ti_state": best_states_B["best_data"]["ti"],
            "args": vars(args),
            "stage": "B",
            "kind": "best_datafit",
        },
        ckpt_best_data,
    )
    torch.save(
        {
            "u_states": best_states_B["best_interface"]["u"],
            "ti_state": best_states_B["best_interface"]["ti"],
            "args": vars(args),
            "stage": "B",
            "kind": "best_interface",
        },
        ckpt_best_if,
    )
    torch.save(
        {
            "u_states": best_states_B["best_combo"]["u"],
            "ti_state": best_states_B["best_combo"]["ti"],
            "args": vars(args),
            "stage": "B",
            "kind": "best_composite",
        },
        ckpt_best_combo,
    )
    torch.save(
        {
            "u_states": chart_u_states(u_nets),
            "ti_state": ti_params.state_dict(),
            "args": vars(args),
            "stage": "final",
        },
        ckpt_final,
    )

    # Concatenate A/B/C histories for plotting.
    plot_hist: Dict[str, List[float]] = {}
    keys = list(hist_A.keys())
    for k in keys:
        vals = []
        vals.extend(hist_A.get(k, []))
        vals.extend(hist_B.get(k, []))
        if hist_C is not None:
            vals.extend(hist_C.get(k, []))
        plot_hist[k] = vals
    make_plot(plot_hist, curve_path)

    stageA_final_mu = float(hist_A["err_mu_m_pct"][-1]) if hist_A["err_mu_m_pct"] else float("inf")
    stageA_final_kf = float(hist_A["err_k_f_pct"][-1]) if hist_A["err_k_f_pct"] else float("inf")
    stageA_if_jump = float(hist_A["if_jump"][-1]) if hist_A["if_jump"] else float("inf")
    if_reduction = (
        (stageA_if_jump - if_jump_final) / max(stageA_if_jump, 1e-12)
        if math.isfinite(stageA_if_jump)
        else 0.0
    )

    metrics = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_charts": n_charts,
        "n_load_levels": len(levels),
        "load_lambdas": [float(l.lam) for l in levels],
        "mu_m_true": float(args.mu_m_true),
        "N_m_true": float(args.N_m_true),
        "k_f_true": float(args.k_f_true),
        "N_f_true": float(args.N_f_true),
        "mu_m_est": mu_v,
        "N_m_est": Nm_v,
        "k_f_est": kf_v,
        "N_f_est": Nf_v,
        "mu_m_rel_error_percent": err_mu,
        "N_m_rel_error_percent": err_Nm,
        "k_f_rel_error_percent": err_kf,
        "N_f_rel_error_percent": err_Nf,
        "surface_rel_l2_mean": surf_rel_final,
        "interface": {
            "final_if_val": mean_if_val_f,
            "final_if_flux": mean_if_flux_f,
            "final_if_jump": if_jump_final,
            "stageA_if_jump": stageA_if_jump,
            "if_jump_reduction_vs_stageA": if_reduction,
        },
        "case_metrics": case_metrics,
        "sensitivity": {
            "init": sens_init,
            "after_stage_A": sens_after_A,
        },
        "stages": {
            "teacher": teacher_pack["summary"],
            "stage_A": pack_A["summary"],
            "stage_B": pack_B["summary"],
            "stage_C": pack_C["summary"] if pack_C is not None else None,
        },
        "acceptance": {
            "stage_A_mu_m_k_f_2pct": bool(stageA_final_mu <= 2.0 and stageA_final_kf <= 2.0),
            "stage_B_all_5pct": bool(err_mu <= 5.0 and err_Nm <= 5.0 and err_kf <= 5.0 and err_Nf <= 5.0),
            "surface_rel_l2_1p5pct": bool(surf_rel_final <= 0.015),
            "if_jump_reduction_40pct": bool(if_reduction >= 0.40),
        },
        "target_met": bool(
            (stageA_final_mu <= 2.0 and stageA_final_kf <= 2.0)
            and (err_mu <= 5.0 and err_Nm <= 5.0 and err_kf <= 5.0 and err_Nf <= 5.0)
            and (surf_rel_final <= 0.015)
            and (if_reduction >= 0.40)
        ),
        "paths": {
            "history": hist_path,
            "metrics": met_path,
            "curves": curve_path,
            "sensitivity": sens_path,
            "best_datafit": ckpt_best_data,
            "best_interface": ckpt_best_if,
            "best_composite": ckpt_best_combo,
            "final": ckpt_final,
            "merged_domain_vtu": merged_vtu,
        },
    }

    history_out = {
        "teacher": teacher_hist,
        "stage_A": hist_A,
        "stage_B": hist_B,
        "stage_C": hist_C,
        "combined_for_plot": plot_hist,
    }

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_out, f, indent=2)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump({"init": sens_init, "after_stage_A": sens_after_A}, f, indent=2)

    print("Rabbit TI Arruda-Boyce normal-ramp inverse run complete")
    print(f"  metrics:    {met_path}")
    print(f"  history:    {hist_path}")
    print(f"  curves:     {curve_path}")
    print(f"  merged_vtu: {merged_vtu}")
    print(
        f"  final params: mu_m={mu_v:.6f} ({err_mu:.2f}%), N_m={Nm_v:.6f} ({err_Nm:.2f}%), "
        f"k_f={kf_v:.6f} ({err_kf:.2f}%), N_f={Nf_v:.6f} ({err_Nf:.2f}%)"
    )
    print(
        f"  surface_rel_l2={surf_rel_final:.3e}, if_jump={if_jump_final:.3e}, "
        f"if_reduction_vs_stageA={if_reduction:.2%}, target_met={metrics['target_met']}"
    )

    return metrics


# ----------------------------- CLI -----------------------------


def parse_args() -> argparse.Namespace:
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", default="")
    cfg_args, rest = cfg_parser.parse_known_args()
    cfg: Dict[str, str] = {}
    if cfg_args.config:
        cfg = parse_simple_yaml(cfg_args.config)

    p = argparse.ArgumentParser(
        description="Rabbit TI Arruda-Boyce inverse with single normal-traction ramp and strong Schwarz interface control."
    )
    p.add_argument("--config", default=cfg_args.config)

    p.add_argument(
        "--atlas-data",
        default=cfg_get_str(
            cfg,
            "atlas-data",
            "/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/atlas_schwarz_20260213_005232/rabbit_atlas_data.npz",
        ),
    )
    p.add_argument(
        "--atlas-checkpoint",
        default=cfg_get_str(
            cfg,
            "atlas-checkpoint",
            "/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/atlas_schwarz_20260213_005232/rabbit_atlas_trained.pt",
        ),
    )
    p.add_argument(
        "--atlas-meta",
        default=cfg_get_str(
            cfg,
            "atlas-meta",
            "/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/atlas_schwarz_20260213_005232/rabbit_atlas_meta.json",
        ),
    )
    p.add_argument(
        "--output-dir",
        default=cfg_get_str(
            cfg,
            "output-dir",
            "/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/rabbit_inverse_ti_arruda_boyce_normal_ramp_main",
        ),
    )
    p.add_argument("--run-tag", default=cfg_get_str(cfg, "run-tag", "main"))

    p.add_argument("--seed", type=int, default=cfg_get_int(cfg, "seed", 42))
    p.add_argument("--device", default=cfg_get_str(cfg, "device", "auto"), choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default=cfg_get_str(cfg, "dtype", "auto"), choices=["auto", "float32", "float64"])
    p.add_argument("--amp", type=parse_bool, default=cfg_get_bool(cfg, "amp", False))
    p.add_argument("--no-amp", action="store_true", help="Disable AMP (alias for --amp false).")

    p.add_argument("--pinn-width", type=int, default=cfg_get_int(cfg, "pinn-width", 96))
    p.add_argument("--pinn-depth", type=int, default=cfg_get_int(cfg, "pinn-depth", 5))

    # Load family
    lambdas_default = cfg_get_str(cfg, "load-lambdas", "0.2,0.4,0.6,0.8,1.0,1.2")
    p.add_argument("--load-lambdas", type=parse_float_list, default=parse_float_list(lambdas_default))
    p.add_argument("--t0", type=float, default=cfg_get_float(cfg, "t0", 0.12))

    # Iteration schedules
    p.add_argument("--teacher-iters", type=int, default=cfg_get_int(cfg, "teacher-iters", 50))
    p.add_argument("--stage-a-iters", type=int, default=cfg_get_int(cfg, "stage-a-iters", 80))
    p.add_argument("--stage-b-iters", type=int, default=cfg_get_int(cfg, "stage-b-iters", 80))
    p.add_argument("--stage-c-iters", type=int, default=cfg_get_int(cfg, "stage-c-iters", 20))
    p.add_argument("--local-steps", type=int, default=cfg_get_int(cfg, "local-steps", 2))
    p.add_argument("--omega", type=float, default=cfg_get_float(cfg, "omega", 0.85))
    p.add_argument("--omega-teacher", type=float, default=cfg_get_float(cfg, "omega-teacher", 0.85))

    # Learning rates
    p.add_argument("--lr-u", type=float, default=cfg_get_float(cfg, "lr-u", 2e-4))
    p.add_argument("--lr-theta-stage-a", type=float, default=cfg_get_float(cfg, "lr-theta-stage-a", 1e-3))
    p.add_argument("--lr-theta-stage-b-fast", type=float, default=cfg_get_float(cfg, "lr-theta-stage-b-fast", 4e-4))
    p.add_argument("--lr-theta-stage-b-slow", type=float, default=cfg_get_float(cfg, "lr-theta-stage-b-slow", 2e-4))
    p.add_argument("--grad-clip", type=float, default=cfg_get_float(cfg, "grad-clip", 5.0))
    p.add_argument("--grad-clip-theta", type=float, default=cfg_get_float(cfg, "grad-clip-theta", 2.0))

    # Sampling
    p.add_argument("--int-cache-per-chart", type=int, default=cfg_get_int(cfg, "int-cache-per-chart", 1800))
    p.add_argument("--if-cache-per-pair", type=int, default=cfg_get_int(cfg, "if-cache-per-pair", 1000))
    p.add_argument("--n-obs-per-chart", type=int, default=cfg_get_int(cfg, "n-obs-per-chart", 700))
    p.add_argument("--pde-batch", type=int, default=cfg_get_int(cfg, "pde-batch", 96))
    p.add_argument("--bc-batch", type=int, default=cfg_get_int(cfg, "bc-batch", 160))
    p.add_argument("--if-batch", type=int, default=cfg_get_int(cfg, "if-batch", 96))
    p.add_argument("--xi-noise-scale", type=float, default=cfg_get_float(cfg, "xi-noise-scale", 0.06))
    p.add_argument("--normal-noise-boost", type=float, default=cfg_get_float(cfg, "normal-noise-boost", 1.2))

    # Weights teacher
    p.add_argument("--w-teacher-eq-start", type=float, default=cfg_get_float(cfg, "w-teacher-eq-start", 0.6))
    p.add_argument("--w-teacher-eq-end", type=float, default=cfg_get_float(cfg, "w-teacher-eq-end", 1.0))
    p.add_argument("--w-teacher-trac-bc", type=float, default=cfg_get_float(cfg, "w-teacher-trac-bc", 35.0))
    p.add_argument("--w-teacher-if-val-start", type=float, default=cfg_get_float(cfg, "w-teacher-if-val-start", 8.0))
    p.add_argument("--w-teacher-if-val-end", type=float, default=cfg_get_float(cfg, "w-teacher-if-val-end", 20.0))
    p.add_argument("--w-teacher-if-flux-start", type=float, default=cfg_get_float(cfg, "w-teacher-if-flux-start", 20.0))
    p.add_argument("--w-teacher-if-flux-end", type=float, default=cfg_get_float(cfg, "w-teacher-if-flux-end", 60.0))
    p.add_argument("--w-teacher-rbm", type=float, default=cfg_get_float(cfg, "w-teacher-rbm", 30.0))

    # Weights inverse
    p.add_argument("--w-eq-start", type=float, default=cfg_get_float(cfg, "w-eq-start", 0.5))
    p.add_argument("--w-eq-end", type=float, default=cfg_get_float(cfg, "w-eq-end", 1.0))
    p.add_argument("--w-if-val-start", type=float, default=cfg_get_float(cfg, "w-if-val-start", 8.0))
    p.add_argument("--w-if-val-end", type=float, default=cfg_get_float(cfg, "w-if-val-end", 20.0))
    p.add_argument("--w-if-flux-start", type=float, default=cfg_get_float(cfg, "w-if-flux-start", 20.0))
    p.add_argument("--w-if-flux-end", type=float, default=cfg_get_float(cfg, "w-if-flux-end", 60.0))
    p.add_argument("--w-if-val-stage-b-start", type=float, default=cfg_get_float(cfg, "w-if-val-stage-b-start", 10.0))
    p.add_argument("--w-if-val-stage-b-end", type=float, default=cfg_get_float(cfg, "w-if-val-stage-b-end", 24.0))
    p.add_argument("--w-if-flux-stage-b-start", type=float, default=cfg_get_float(cfg, "w-if-flux-stage-b-start", 24.0))
    p.add_argument("--w-if-flux-stage-b-end", type=float, default=cfg_get_float(cfg, "w-if-flux-stage-b-end", 70.0))
    p.add_argument("--w-eq-stage-b-start", type=float, default=cfg_get_float(cfg, "w-eq-stage-b-start", 0.7))
    p.add_argument("--w-eq-stage-b-end", type=float, default=cfg_get_float(cfg, "w-eq-stage-b-end", 1.0))
    p.add_argument("--w-if-val-stage-c-start", type=float, default=cfg_get_float(cfg, "w-if-val-stage-c-start", 14.0))
    p.add_argument("--w-if-val-stage-c-end", type=float, default=cfg_get_float(cfg, "w-if-val-stage-c-end", 28.0))
    p.add_argument("--w-if-flux-stage-c-start", type=float, default=cfg_get_float(cfg, "w-if-flux-stage-c-start", 32.0))
    p.add_argument("--w-if-flux-stage-c-end", type=float, default=cfg_get_float(cfg, "w-if-flux-stage-c-end", 80.0))
    p.add_argument("--w-eq-stage-c-start", type=float, default=cfg_get_float(cfg, "w-eq-stage-c-start", 0.8))
    p.add_argument("--w-eq-stage-c-end", type=float, default=cfg_get_float(cfg, "w-eq-stage-c-end", 1.0))

    p.add_argument("--w-trac-bc", type=float, default=cfg_get_float(cfg, "w-trac-bc", 40.0))
    p.add_argument("--w-data", type=float, default=cfg_get_float(cfg, "w-data", 100.0))
    p.add_argument("--w-rbm", type=float, default=cfg_get_float(cfg, "w-rbm", 30.0))
    p.add_argument("--w-reg", type=float, default=cfg_get_float(cfg, "w-reg", 1e-2))
    p.add_argument("--huber-delta", type=float, default=cfg_get_float(cfg, "huber-delta", 0.02))

    # interface gate and hard pairs
    p.add_argument("--if-hard-oversample", type=float, default=cfg_get_float(cfg, "if-hard-oversample", 2.0))
    p.add_argument("--if-score-flux-factor", type=float, default=cfg_get_float(cfg, "if-score-flux-factor", 1.0))
    p.add_argument("--if-gate-rel-tol", type=float, default=cfg_get_float(cfg, "if-gate-rel-tol", 0.08))
    p.add_argument("--if-gate-lr-decay", type=float, default=cfg_get_float(cfg, "if-gate-lr-decay", 0.7))
    p.add_argument("--eval-if-samples", type=int, default=cfg_get_int(cfg, "eval-if-samples", 96))

    # score weights
    p.add_argument("--combo-if-weight", type=float, default=cfg_get_float(cfg, "combo-if-weight", 0.2))
    p.add_argument("--combo-param-weight", type=float, default=cfg_get_float(cfg, "combo-param-weight", 0.01))

    # Jacobian stabilization
    p.add_argument("--sigma-floor", type=float, default=cfg_get_float(cfg, "sigma-floor", 1e-6))
    p.add_argument("--detJ-floor", type=float, default=cfg_get_float(cfg, "detJ-floor", 1e-6))
    p.add_argument("--jac-kappa-max", type=float, default=cfg_get_float(cfg, "jac-kappa-max", 1e4))
    p.add_argument("--eq-residual-clip", type=float, default=cfg_get_float(cfg, "eq-residual-clip", 200.0))

    # Material and priors
    p.add_argument("--kappa-fixed", type=float, default=cfg_get_float(cfg, "kappa-fixed", 80.0))

    p.add_argument("--mu-m-true", type=float, default=cfg_get_float(cfg, "mu-m-true", 1.80))
    p.add_argument("--N-m-true", type=float, default=cfg_get_float(cfg, "N-m-true", 4.80))
    p.add_argument("--k-f-true", type=float, default=cfg_get_float(cfg, "k-f-true", 2.20))
    p.add_argument("--N-f-true", type=float, default=cfg_get_float(cfg, "N-f-true", 5.20))

    p.add_argument("--mu-m-init", type=float, default=cfg_get_float(cfg, "mu-m-init", 1.782))
    p.add_argument("--N-m-init", type=float, default=cfg_get_float(cfg, "N-m-init", 4.752))
    p.add_argument("--k-f-init", type=float, default=cfg_get_float(cfg, "k-f-init", 2.178))
    p.add_argument("--N-f-init", type=float, default=cfg_get_float(cfg, "N-f-init", 5.148))

    p.add_argument("--mu-m-min", type=float, default=cfg_get_float(cfg, "mu-m-min", 1e-4))
    p.add_argument("--N-m-min", type=float, default=cfg_get_float(cfg, "N-m-min", 1.01))
    p.add_argument("--k-f-min", type=float, default=cfg_get_float(cfg, "k-f-min", 1e-4))
    p.add_argument("--N-f-min", type=float, default=cfg_get_float(cfg, "N-f-min", 1.01))

    p.add_argument("--mu-m-prior", type=float, default=cfg_get_float(cfg, "mu-m-prior", 1.80))
    p.add_argument("--N-m-prior", type=float, default=cfg_get_float(cfg, "N-m-prior", 4.80))
    p.add_argument("--k-f-prior", type=float, default=cfg_get_float(cfg, "k-f-prior", 2.20))
    p.add_argument("--N-f-prior", type=float, default=cfg_get_float(cfg, "N-f-prior", 5.20))

    p.add_argument("--fiber-a0-x", type=float, default=cfg_get_float(cfg, "fiber-a0-x", 0.62))
    p.add_argument("--fiber-a0-y", type=float, default=cfg_get_float(cfg, "fiber-a0-y", 0.49))
    p.add_argument("--fiber-a0-z", type=float, default=cfg_get_float(cfg, "fiber-a0-z", 0.61))

    # Sensitivity gate
    p.add_argument("--sensitivity-surface-per-chart", type=int, default=cfg_get_int(cfg, "sensitivity-surface-per-chart", 120))
    p.add_argument("--sensitivity-cond-max", type=float, default=cfg_get_float(cfg, "sensitivity-cond-max", 1e5))
    p.add_argument("--sens-eps-rel", type=float, default=cfg_get_float(cfg, "sens-eps-rel", 1e-2))
    p.add_argument("--strict-sensitivity-gate", type=parse_bool, default=cfg_get_bool(cfg, "strict-sensitivity-gate", False))

    # Export
    p.add_argument("--blend-temp", type=float, default=cfg_get_float(cfg, "blend-temp", 1.0))
    p.add_argument("--export-points-per-chart", type=int, default=cfg_get_int(cfg, "export-points-per-chart", 2400))

    args = p.parse_args(rest)
    if args.no_amp:
        args.amp = False
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
