#!/usr/bin/env python3
"""
Inverse steady Elder-like buoyancy flow on rabbit atlas charts.

This script reuses the rabbit atlas chart/mask parameterization and Schwarz-style
chart coupling, and solves a teacher-student inverse problem for homogeneous
rotated SPD permeability:

    K = k0 * R(q) * diag(exp(a1), exp(a2), exp(-a1-a2)) * R(q)^T

Stages:
1) Teacher forward solve with true K (same mapped PDE + BC + interface coupling)
2) Sparse observation extraction from teacher fields
3) Inverse solve for p/c chart fields + permeability params from close init

Notes:
- First milestone is steady snapshot inversion (not transient Elder).
- Defaults are tuned for feasibility and parameter recovery first.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)


# -------------------------- Utilities --------------------------

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
            print("Requested --device cuda but CUDA unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested --device mps but MPS unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type in ("cuda", "mps"):
            return torch.float32
        return torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS backend does not support float64 reliably; use float32 or auto.")
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def normalize_rows_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / torch.clamp(n, min=eps)


def copy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def blend_model_with_old(model: torch.nn.Module, old_state: Dict[str, torch.Tensor], omega: float) -> None:
    new_state = model.state_dict()
    blended = {}
    for k, v in new_state.items():
        blended[k] = (1.0 - omega) * old_state[k] + omega * v
    model.load_state_dict(blended)


def parse_simple_yaml(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def build_run_stem(run_tag: str) -> str:
    if run_tag is None:
        run_tag = ""
    run_tag = run_tag.strip()
    if len(run_tag) == 0:
        return "rabbit_inverse_elder_atlas_schwarz"
    return f"rabbit_inverse_elder_atlas_schwarz_{run_tag}"


def robust_unit_interval(values: torch.Tensor, q_lo: float = 0.05, q_hi: float = 0.95) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros_like(values)
    # MPS does not support float64 tensors; always move to CPU first, then cast.
    v_cpu = values.detach().to(device=torch.device("cpu")).reshape(-1)
    if not torch.is_floating_point(v_cpu):
        v_cpu = v_cpu.to(dtype=torch.float32)
    elif v_cpu.dtype in (torch.float16, torch.bfloat16):
        v_cpu = v_cpu.to(dtype=torch.float32)
    if not torch.isfinite(v_cpu).all():
        v_cpu = torch.nan_to_num(v_cpu, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(torch.quantile(v_cpu, q_lo).item())
    hi = float(torch.quantile(v_cpu, q_hi).item())
    if hi <= lo + 1e-12:
        return torch.zeros_like(values)
    out = (values - lo) / (hi - lo)
    return torch.clamp(out, min=0.0, max=1.0)


def sample_indices_from_weights(weights: torch.Tensor, n_take: int) -> torch.Tensor:
    n = int(weights.numel())
    if n <= 0 or n_take <= 0:
        return torch.zeros((0,), device=weights.device, dtype=torch.int64)
    replace = n_take > n
    w = torch.clamp(weights, min=1e-12)
    w_sum = torch.sum(w)
    if (not torch.isfinite(w_sum)) or float(w_sum.item()) <= 0.0:
        return torch.randint(0, n, (n_take,), device=weights.device)
    w = w / w_sum
    if weights.device.type == "mps":
        sel = torch.multinomial(w.detach().cpu(), n_take, replacement=replace)
        return sel.to(device=weights.device, dtype=torch.int64)
    return torch.multinomial(w, n_take, replacement=replace)


# -------------------------- Models --------------------------

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


class LocalScalarPINN(torch.nn.Module):
    def __init__(self, width: int = 96, depth: int = 5):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.net(xi)


def softplus_inv(y: float) -> float:
    y = max(y, 1e-8)
    return math.log(math.expm1(y))


def euler_deg_to_quat_xyz(ax_deg: float, ay_deg: float, az_deg: float, dtype: torch.dtype) -> torch.Tensor:
    ax = math.radians(ax_deg)
    ay = math.radians(ay_deg)
    az = math.radians(az_deg)
    cx, sx = math.cos(ax / 2.0), math.sin(ax / 2.0)
    cy, sy = math.cos(ay / 2.0), math.sin(ay / 2.0)
    cz, sz = math.cos(az / 2.0), math.sin(az / 2.0)
    # q = qz * qy * qx
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    q = torch.tensor([w, x, y, z], dtype=dtype)
    q = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
    return q


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    qn = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
    w, x, y, z = qn[0], qn[1], qn[2], qn[3]
    two = torch.as_tensor(2.0, device=q.device, dtype=q.dtype)
    r00 = 1 - two * (y * y + z * z)
    r01 = two * (x * y - z * w)
    r02 = two * (x * z + y * w)
    r10 = two * (x * y + z * w)
    r11 = 1 - two * (x * x + z * z)
    r12 = two * (y * z - x * w)
    r20 = two * (x * z - y * w)
    r21 = two * (y * z + x * w)
    r22 = 1 - two * (x * x + y * y)
    return torch.stack(
        [
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ],
        dim=0,
    )


class PermeabilityParam(torch.nn.Module):
    def __init__(
        self,
        k0_init: float,
        k0_min: float,
        a1_init: float,
        a2_init: float,
        euler_init_deg: Tuple[float, float, float],
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.k0_min = float(k0_min)
        k0_adj = max(k0_init - self.k0_min, 1e-6)
        self.k0_raw = torch.nn.Parameter(torch.tensor(softplus_inv(k0_adj), device=device, dtype=dtype))
        self.a1 = torch.nn.Parameter(torch.tensor(a1_init, device=device, dtype=dtype))
        self.a2 = torch.nn.Parameter(torch.tensor(a2_init, device=device, dtype=dtype))
        q0 = euler_deg_to_quat_xyz(*euler_init_deg, dtype=dtype).to(device=device, dtype=dtype)
        self.q_raw = torch.nn.Parameter(q0)

    def k0(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.k0_raw) + self.k0_min

    def quaternion(self) -> torch.Tensor:
        return self.q_raw / torch.clamp(torch.linalg.norm(self.q_raw), min=1e-12)

    def eigvals(self) -> torch.Tensor:
        e1 = torch.exp(self.a1)
        e2 = torch.exp(self.a2)
        e3 = torch.exp(-self.a1 - self.a2)
        return torch.stack([e1, e2, e3], dim=0)

    def K_tensor(self) -> torch.Tensor:
        qn = self.quaternion()
        R = quat_to_rotmat(qn)
        lam = self.eigvals()
        D = torch.diag(lam)
        K = self.k0() * (R @ D @ R.transpose(0, 1))
        return K


# -------------------------- Atlas Ops --------------------------

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


def inv_det_3x3_stable(jac: torch.Tensor, det_floor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Cofactor-based 3x3 inverse/determinant to avoid backend-specific linalg kernels.
    a = jac[:, 0, 0]
    b = jac[:, 0, 1]
    c = jac[:, 0, 2]
    d = jac[:, 1, 0]
    e = jac[:, 1, 1]
    f = jac[:, 1, 2]
    g = jac[:, 2, 0]
    h = jac[:, 2, 1]
    i = jac[:, 2, 2]

    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02
    det_abs = torch.abs(det)
    sign = torch.where(det >= 0.0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = torch.where(det_abs > det_floor, det, sign * det_floor)

    inv = torch.stack(
        [
            torch.stack([c00, c10, c20], dim=1),
            torch.stack([c01, c11, c21], dim=1),
            torch.stack([c02, c12, c22], dim=1),
        ],
        dim=1,
    ) / det_safe.unsqueeze(-1).unsqueeze(-1)

    return inv, det_abs, det_safe


def stabilized_jacobian_ops(
    jac: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sf = torch.as_tensor(sigma_floor, device=jac.device, dtype=jac.dtype)
    df = torch.as_tensor(det_floor, device=jac.device, dtype=jac.dtype)

    # MPS currently has unstable/unsupported autograd behavior for linalg_svd.
    # Use a regularized inverse + condition surrogate there.
    def inverse_fallback() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eye = torch.eye(3, device=jac.device, dtype=jac.dtype).unsqueeze(0)
        reg = torch.clamp(sf, min=torch.as_tensor(1e-6, device=jac.device, dtype=jac.dtype))
        jac_reg = jac + reg * eye
        inv_j_loc, raw_det_abs_loc, _ = inv_det_3x3_stable(jac_reg, det_floor=df)
        det_abs_loc = torch.clamp(raw_det_abs_loc, min=df)
        # Frobenius-norm condition surrogate: ||J||_F * ||J^{-1}||_F
        jn = torch.sqrt(torch.clamp(torch.sum(jac_reg * jac_reg, dim=(1, 2)), min=1e-24))
        inorm = torch.sqrt(torch.clamp(torch.sum(inv_j_loc * inv_j_loc, dim=(1, 2)), min=1e-24))
        kappa_loc = jn * inorm
        valid_loc = raw_det_abs_loc > df
        valid_loc = valid_loc & torch.isfinite(kappa_loc) & torch.isfinite(det_abs_loc)
        return inv_j_loc, det_abs_loc, kappa_loc, valid_loc

    if jac.device.type == "mps":
        return inverse_fallback()

    try:
        u, s, vh = torch.linalg.svd(jac)
        s_safe = torch.clamp(s, min=sf)
        inv_s = torch.diag_embed(1.0 / s_safe)
        inv_j = torch.bmm(vh.transpose(1, 2), torch.bmm(inv_s, u.transpose(1, 2)))
        raw_det_abs = torch.abs(torch.det(jac))
        det_abs = torch.clamp(raw_det_abs, min=df)
        kappa = s_safe[:, 0] / torch.clamp(s_safe[:, -1], min=sf)
        valid = raw_det_abs > df
        valid = valid & torch.isfinite(kappa) & torch.isfinite(det_abs)
        return inv_j, det_abs, kappa, valid
    except RuntimeError:
        # Keep training robust on backends/builds where SVD fails.
        return inverse_fallback()


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


# -------------------------- Physics --------------------------

@dataclass
class LocalState:
    x: torch.Tensor
    xi_var: torch.Tensor
    jac: torch.Tensor
    inv_j: torch.Tensor
    det_abs: torch.Tensor
    valid: torch.Tensor
    p: torch.Tensor
    c: torch.Tensor
    grad_p_x: torch.Tensor
    grad_c_x: torch.Tensor
    u: torch.Tensor
    flow_flux_mapped: torch.Tensor
    trans_flux_mapped: torch.Tensor
    flow_residual: torch.Tensor
    trans_residual: torch.Tensor


def divergence_mapped(mapped_flux: torch.Tensor, xi_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
    div = torch.zeros((mapped_flux.shape[0], 1), device=xi_var.device, dtype=xi_var.dtype)
    for j in range(3):
        dcomp = torch.autograd.grad(
            mapped_flux[:, j],
            xi_var,
            grad_outputs=torch.ones_like(mapped_flux[:, j]),
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, j : j + 1]
        div = div + dcomp
    return div


def compute_local_state(
    p_model: LocalScalarPINN,
    c_model: LocalScalarPINN,
    decoder: ChartDecoder,
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
    K_tensor: torch.Tensor,
    Ra: float,
    Dm: float,
    gravity: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
    jac_kappa_max: float,
    create_graph: bool,
) -> LocalState:
    x, xi_var, jac = chart_map_and_jacobian(
        decoder,
        xi,
        seed=seed,
        t1=t1,
        t2=t2,
        n=n,
        chart_scale=chart_scale,
    )
    inv_j, det_abs, kappa, valid = stabilized_jacobian_ops(jac=jac, sigma_floor=sigma_floor, det_floor=det_floor)
    valid = valid & (kappa <= jac_kappa_max)

    p = p_model(xi_var)
    c = c_model(xi_var)

    grad_p_xi = torch.autograd.grad(
        p,
        xi_var,
        grad_outputs=torch.ones_like(p),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    grad_c_xi = torch.autograd.grad(
        c,
        xi_var,
        grad_outputs=torch.ones_like(c),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    grad_p_x = torch.bmm(inv_j.transpose(1, 2), grad_p_xi.unsqueeze(-1)).squeeze(-1)
    grad_c_x = torch.bmm(inv_j.transpose(1, 2), grad_c_xi.unsqueeze(-1)).squeeze(-1)

    v = grad_p_x - float(Ra) * c * gravity.unsqueeze(0)
    Kbat = K_tensor.unsqueeze(0).repeat(xi.shape[0], 1, 1)
    u = -torch.bmm(Kbat, v.unsqueeze(-1)).squeeze(-1)

    flow_flux_mapped = det_abs.unsqueeze(-1) * torch.bmm(inv_j, u.unsqueeze(-1)).squeeze(-1)
    flow_residual = divergence_mapped(flow_flux_mapped, xi_var, create_graph=create_graph)

    q_diff = float(Dm) * grad_c_x
    trans_flux_mapped = det_abs.unsqueeze(-1) * torch.bmm(inv_j, q_diff.unsqueeze(-1)).squeeze(-1)
    div_trans_flux = divergence_mapped(trans_flux_mapped, xi_var, create_graph=create_graph)
    adv = torch.sum(u * grad_c_x, dim=1, keepdim=True)
    trans_residual = det_abs.unsqueeze(-1) * adv - div_trans_flux

    return LocalState(
        x=x,
        xi_var=xi_var,
        jac=jac,
        inv_j=inv_j,
        det_abs=det_abs,
        valid=valid,
        p=p,
        c=c,
        grad_p_x=grad_p_x,
        grad_c_x=grad_c_x,
        u=u,
        flow_flux_mapped=flow_flux_mapped,
        trans_flux_mapped=trans_flux_mapped,
        flow_residual=flow_residual,
        trans_residual=trans_residual,
    )


def select_residual_samples(residual: torch.Tensor, valid: torch.Tensor, clip_q: float) -> torch.Tensor:
    if torch.any(valid):
        res = residual[valid]
    else:
        res = residual
    if clip_q < 1.0 and res.numel() >= 8:
        rabs_cpu = torch.abs(res.detach()).reshape(-1).to(device=torch.device("cpu"))
        q = float(torch.quantile(rabs_cpu, clip_q).item())
        q = max(q, 1e-8)
        res = torch.clamp(res, min=-q, max=q)
    return res


# -------------------------- Boundaries --------------------------

def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


@dataclass
class BoundaryMasks:
    inlet: torch.Tensor
    fresh: torch.Tensor
    gauge: torch.Tensor
    noflow: torch.Tensor
    nodiff: torch.Tensor


def build_boundary_masks(
    points: torch.Tensor,
    inlet_z_quantile: float,
    top_z_quantile: float,
    inlet_phi_center_deg: float,
    inlet_phi_halfwidth_deg: float,
    gauge_z_quantile: float,
    gauge_phi_center_deg: float,
    gauge_phi_halfwidth_deg: float,
) -> BoundaryMasks:
    z = points[:, 2]
    phi = torch.atan2(points[:, 1], points[:, 0])

    z_in_thr = torch.quantile(z.detach().cpu(), float(np.clip(inlet_z_quantile, 0.0, 1.0))).to(device=points.device, dtype=points.dtype)
    z_top_thr = torch.quantile(z.detach().cpu(), float(np.clip(top_z_quantile, 0.0, 1.0))).to(device=points.device, dtype=points.dtype)
    z_gauge_thr = torch.quantile(z.detach().cpu(), float(np.clip(gauge_z_quantile, 0.0, 1.0))).to(device=points.device, dtype=points.dtype)

    in_phi_c = torch.as_tensor(math.radians(inlet_phi_center_deg), device=points.device, dtype=points.dtype)
    in_phi_hw = torch.as_tensor(abs(math.radians(inlet_phi_halfwidth_deg)), device=points.device, dtype=points.dtype)
    g_phi_c = torch.as_tensor(math.radians(gauge_phi_center_deg), device=points.device, dtype=points.dtype)
    g_phi_hw = torch.as_tensor(abs(math.radians(gauge_phi_halfwidth_deg)), device=points.device, dtype=points.dtype)

    inlet_phi_mask = torch.abs(wrap_to_pi(phi - in_phi_c)) <= in_phi_hw
    gauge_phi_mask = torch.abs(wrap_to_pi(phi - g_phi_c)) <= g_phi_hw

    top_mask = z >= z_top_thr
    inlet = (z >= z_in_thr) & inlet_phi_mask
    fresh = top_mask & (~inlet)
    gauge = (z <= z_gauge_thr) & gauge_phi_mask
    noflow = ~gauge
    nodiff = ~(inlet | fresh)

    # Safety fallback to keep each BC non-empty.
    def ensure_non_empty(mask: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
        if int(torch.sum(mask).item()) > 0:
            return mask
        idx = torch.argsort(ref, descending=True)[:k]
        out = torch.zeros_like(mask)
        out[idx] = True
        return out

    k_small = max(16, points.shape[0] // 500)
    inlet = ensure_non_empty(inlet, z, k_small)
    fresh = ensure_non_empty(fresh, z, k_small)
    gauge = ensure_non_empty(gauge, -z, k_small)
    noflow = ~gauge
    nodiff = ~(inlet | fresh)

    return BoundaryMasks(inlet=inlet, fresh=fresh, gauge=gauge, noflow=noflow, nodiff=nodiff)


# -------------------------- Metrics --------------------------

def sorted_eigh_desc(K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(K)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs


def axis_angle_errors_deg(evec_pred: torch.Tensor, evec_true: torch.Tensor) -> torch.Tensor:
    dots = torch.sum(evec_pred * evec_true, dim=0)
    dots = torch.clamp(torch.abs(dots), min=0.0, max=1.0)
    return torch.rad2deg(torch.arccos(dots))


# -------------------------- IO helpers --------------------------

def load_atlas_models(
    atlas_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[ChartDecoder], List[MaskNet], Dict[str, object]]:
    # Always load atlas weights on CPU first.
    # This avoids MPS failures when legacy checkpoints contain float64 tensors.
    ckpt = torch.load(atlas_checkpoint, map_location=torch.device("cpu"))
    dec_kw = ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kw = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    dec_states = ckpt["decoder_states"]
    mask_states = ckpt["mask_states"]

    decoders: List[ChartDecoder] = []
    masks: List[MaskNet] = []
    def cast_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v
        return out

    for ds, ms in zip(dec_states, mask_states):
        d = ChartDecoder(width=dec_kw["width"], depth=dec_kw["depth"]).to(device=device, dtype=dtype)
        m = MaskNet(width=mask_kw["width"], depth=mask_kw["depth"]).to(device=device, dtype=dtype)
        d.load_state_dict(cast_state_dict(ds))
        m.load_state_dict(cast_state_dict(ms))
        d.eval()
        m.eval()
        for p in d.parameters():
            p.requires_grad_(False)
        for p in m.parameters():
            p.requires_grad_(False)
        decoders.append(d)
        masks.append(m)
    return decoders, masks, ckpt


def save_ckpt(
    path: str,
    p_states: List[Dict[str, torch.Tensor]],
    c_states: List[Dict[str, torch.Tensor]],
    perm_state: Dict[str, torch.Tensor],
    label: str,
    args: argparse.Namespace,
    history: Dict[str, List[float]],
    metrics: Dict[str, object],
    fallback_from: Optional[str] = None,
) -> None:
    payload = {
        "p_states": p_states,
        "c_states": c_states,
        "perm_state": perm_state,
        "p_kwargs": {"width": args.pinn_width, "depth": args.pinn_depth},
        "c_kwargs": {"width": args.pinn_width, "depth": args.pinn_depth},
        "history": history,
        "metrics": metrics,
        "snapshot_label": label,
        "fallback_from": fallback_from,
        "args": vars(args),
    }
    torch.save(payload, path)


# -------------------------- Main --------------------------

def train(args: argparse.Namespace) -> Dict[str, object]:
    run_start = time.time()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.set_default_dtype(dtype)

    # AMP support policy mirrors existing atlas script.
    mps_amp_supported = False
    if device.type == "mps" and hasattr(torch, "autocast"):
        try:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                pass
            mps_amp_supported = True
        except Exception:
            mps_amp_supported = False

    use_cuda_amp = bool(args.amp and device.type == "cuda" and dtype == torch.float32)
    use_mps_amp = bool(args.amp and device.type == "mps" and dtype == torch.float32 and mps_amp_supported)
    if device.type == "mps" and use_mps_amp:
        print("Requested AMP on MPS; disabling AMP for stability on this solver.")
        use_mps_amp = False
    use_amp = bool(use_cuda_amp or use_mps_amp)
    amp_backend = "cuda" if use_cuda_amp else ("mps" if use_mps_amp else "none")

    def amp_ctx() -> contextlib.AbstractContextManager:
        if use_cuda_amp:
            return torch.cuda.amp.autocast()
        if use_mps_amp:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return contextlib.nullcontext()

    print(f"Device={device.type} dtype={dtype} amp={use_amp} amp_backend={amp_backend}")

    atlas_np = np.load(args.atlas_data)
    points = torch.tensor(atlas_np["points"], device=device, dtype=dtype)
    normals = normalize_rows_tensor(torch.tensor(atlas_np["normals"], device=device, dtype=dtype))
    seeds = torch.tensor(atlas_np["seed_points"], device=device, dtype=dtype)
    t1 = torch.tensor(atlas_np["frame_t1"], device=device, dtype=dtype)
    t2 = torch.tensor(atlas_np["frame_t2"], device=device, dtype=dtype)
    nvec = torch.tensor(atlas_np["frame_n"], device=device, dtype=dtype)
    membership = torch.tensor(atlas_np["membership"].astype(np.int64), device=device, dtype=torch.int64)
    support_r = torch.tensor(atlas_np["support_radii"], device=device, dtype=dtype)

    n_points, n_charts = membership.shape

    decoders, masks, atlas_ckpt = load_atlas_models(args.atlas_checkpoint, device=device, dtype=dtype)
    gate = atlas_ckpt.get("gate")
    if isinstance(gate, dict) and (not args.allow_failed_gate) and not bool(gate.get("passed", False)):
        raise RuntimeError(
            "Atlas gate check failed. Refusing inverse Elder run. "
            f"Checkpoint: {args.atlas_checkpoint}"
        )

    color_groups = choose_color_groups(args.atlas_meta, n_charts=n_charts, membership_np=atlas_np["membership"])

    point_idx_by_chart: List[torch.Tensor] = []
    for i in range(n_charts):
        idx = torch.where(membership[:, i] > 0)[0]
        point_idx_by_chart.append(idx)

    primary_chart_np = atlas_np["primary_chart"] if "primary_chart" in atlas_np.files else np.argmax(atlas_np["membership"], axis=1)
    primary_chart = torch.tensor(primary_chart_np.astype(np.int64), device=device, dtype=torch.int64)

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

    boundary_masks = build_boundary_masks(
        points=points,
        inlet_z_quantile=args.inlet_z_quantile,
        top_z_quantile=args.top_z_quantile,
        inlet_phi_center_deg=args.inlet_phi_center_deg,
        inlet_phi_halfwidth_deg=args.inlet_phi_halfwidth_deg,
        gauge_z_quantile=args.gauge_z_quantile,
        gauge_phi_center_deg=args.gauge_phi_center_deg,
        gauge_phi_halfwidth_deg=args.gauge_phi_halfwidth_deg,
    )

    # Detail proxy and adaptive chart sampling.
    normal_dev = torch.linalg.norm(normals - nvec[primary_chart], dim=1)
    centroid = torch.mean(points, dim=0, keepdim=True)
    extremity = torch.linalg.norm(points - centroid, dim=1)
    overlap_count = torch.sum(membership.to(dtype=dtype), dim=1)

    n_w = max(0.0, float(args.detail_normal_weight))
    e_w = max(0.0, float(args.detail_extremity_weight))
    o_w = max(0.0, float(args.detail_overlap_weight))
    w_sum = max(1e-12, n_w + e_w + o_w)
    detail_score = (
        n_w * robust_unit_interval(normal_dev)
        + e_w * robust_unit_interval(extremity)
        + o_w * robust_unit_interval(overlap_count)
    ) / w_sum
    detail_score = torch.clamp(detail_score, min=0.0, max=1.0)

    q_detail = float(np.clip(args.detail_quantile, 0.0, 1.0))
    if q_detail <= 0.0:
        detail_cut = torch.as_tensor(-1e9, device=device, dtype=dtype)
    elif q_detail >= 1.0:
        detail_cut = torch.max(detail_score.detach())
    else:
        detail_cut = torch.quantile(detail_score.detach().to(device=torch.device("cpu")), q_detail).to(device=device, dtype=dtype)
    high_detail_mask = detail_score >= detail_cut
    n_high_detail_points = int(torch.sum(high_detail_mask).item())

    chart_detail_strength: List[float] = []
    detail_idx_by_chart: List[torch.Tensor] = []
    chart_sampling_weights: List[torch.Tensor] = []
    detail_sampling_weights: List[torch.Tensor] = []
    for i in range(n_charts):
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            chart_detail_strength.append(0.0)
            detail_idx_by_chart.append(idx)
            chart_sampling_weights.append(torch.zeros((0,), device=device, dtype=dtype))
            detail_sampling_weights.append(torch.zeros((0,), device=device, dtype=dtype))
            continue
        s = detail_score[idx]
        hi = high_detail_mask[idx]
        chart_detail_strength.append(float(torch.mean(s).item()))

        if args.curvature_adaptive_sampling:
            w_all = 1.0 + float(args.curvature_sample_weight) * s + 0.5 * float(args.curvature_sample_weight) * hi.to(dtype=dtype)
        else:
            w_all = torch.ones_like(s)
        chart_sampling_weights.append(torch.clamp(w_all, min=1e-8))

        didx = idx[hi]
        detail_idx_by_chart.append(didx)
        if didx.numel() > 0:
            ds = detail_score[didx]
            w_detail = 1.0 + float(args.curvature_sample_weight) * ds
            detail_sampling_weights.append(torch.clamp(w_detail, min=1e-8))
        else:
            detail_sampling_weights.append(torch.zeros((0,), device=device, dtype=dtype))

    detail_chart_mask: List[bool] = [False for _ in range(n_charts)]
    detail_chart_topk = max(0, min(int(args.detail_chart_topk), int(n_charts)))
    if args.curvature_adaptive_sampling and detail_chart_topk > 0:
        order = np.argsort(np.asarray(chart_detail_strength))[::-1]
        picked = 0
        for chart_i in order:
            i = int(chart_i)
            if point_idx_by_chart[i].numel() == 0:
                continue
            detail_chart_mask[i] = True
            picked += 1
            if picked >= detail_chart_topk:
                break

    overlap_sampling_weights: Dict[Tuple[int, int], torch.Tensor] = {}
    overlap_detail_strength: Dict[Tuple[int, int], float] = {}
    for key, shared in overlap_idx_pairs.items():
        if shared.numel() == 0:
            overlap_sampling_weights[key] = torch.zeros((0,), device=device, dtype=dtype)
            overlap_detail_strength[key] = 0.0
            continue
        s = detail_score[shared]
        overlap_detail_strength[key] = float(torch.mean(s).item())
        if args.curvature_adaptive_sampling:
            hi = high_detail_mask[shared].to(dtype=dtype)
            w_pair = 1.0 + float(args.curvature_sample_weight) * s + 0.5 * float(args.curvature_sample_weight) * hi
        else:
            w_pair = torch.ones_like(s)
        overlap_sampling_weights[key] = torch.clamp(w_pair, min=1e-8)

    if args.curvature_adaptive_sampling:
        detail_charts = [i for i, f in enumerate(detail_chart_mask) if f]
        print(
            f"Curvature-adaptive sampling enabled | high-detail points={n_high_detail_points}/{n_points} "
            f"(q={q_detail:.2f}) | detail charts={detail_charts}"
        )

    # Interface normals from mask level-set gradients.
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
    if args.interface_normal_mode == "mask_levelset":
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

    # Helper sampling utilities.
    def sample_chart_point_indices(i: int, n_samples: int, prefer_detail: bool = True) -> Optional[torch.Tensor]:
        idx = point_idx_by_chart[i]
        if idx.numel() == 0 or n_samples <= 0:
            return None
        n_take = int(n_samples)
        if not args.curvature_adaptive_sampling:
            sel = torch.randint(0, idx.numel(), (n_take,), device=device)
            return idx[sel]

        detail_ratio = float(np.clip(args.detail_region_ratio, 0.0, 1.0)) if prefer_detail else 0.0
        detail_idx = detail_idx_by_chart[i]
        picks: List[torch.Tensor] = []

        n_detail = 0
        if detail_idx.numel() > 0 and detail_ratio > 0.0:
            n_detail = int(round(n_take * detail_ratio))
            n_detail = max(0, min(n_take, n_detail))
            if n_detail > 0:
                sel_d = sample_indices_from_weights(detail_sampling_weights[i], n_detail)
                picks.append(detail_idx[sel_d])

        n_base = n_take - n_detail
        if n_base > 0:
            sel_b = sample_indices_from_weights(chart_sampling_weights[i], n_base)
            picks.append(idx[sel_b])

        if not picks:
            sel = torch.randint(0, idx.numel(), (n_take,), device=device)
            return idx[sel]
        out = torch.cat(picks, dim=0)
        perm = torch.randperm(out.numel(), device=device)
        return out[perm]

    def sample_local_xi(i: int, n_samples: int) -> torch.Tensor:
        pick = sample_chart_point_indices(i, n_samples=n_samples, prefer_detail=True)
        if pick is None:
            return torch.zeros((n_samples, 3), device=device, dtype=dtype)
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        noise = args.xi_noise_scale * support_r[i] * torch.randn_like(xi)
        xi = xi + noise
        max_abs = 1.25 * support_r[i]
        xi = torch.clamp(xi, min=-max_abs, max=max_abs)
        return xi

    # Boundary indices per chart for each mask.
    chart_boundary_idx: List[Dict[str, torch.Tensor]] = []
    for i in range(n_charts):
        idx = point_idx_by_chart[i]
        cdict: Dict[str, torch.Tensor] = {}
        for key, mask in [
            ("inlet", boundary_masks.inlet),
            ("fresh", boundary_masks.fresh),
            ("gauge", boundary_masks.gauge),
            ("noflow", boundary_masks.noflow),
            ("nodiff", boundary_masks.nodiff),
        ]:
            if idx.numel() == 0:
                cdict[key] = idx
            else:
                cdict[key] = idx[mask[idx]]
        chart_boundary_idx.append(cdict)

    def sample_chart_boundary(i: int, key: str, n_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = chart_boundary_idx[i][key]
        if idx.numel() == 0 or n_samples <= 0:
            return None
        take = min(int(n_samples), int(idx.numel()))
        sel = torch.randint(0, idx.numel(), (take,), device=device)
        pick = idx[sel]
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        n_phys = normals[pick]
        return xi, x, n_phys

    def interface_batch(i: int, j: int, n_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        key = (i, j) if i < j else (j, i)
        shared = overlap_idx_pairs.get(key)
        if shared is None or shared.numel() == 0:
            return None
        take = min(n_samples, int(shared.numel()))
        if args.curvature_adaptive_sampling and args.curvature_adaptive_interface_sampling:
            sel = sample_indices_from_weights(overlap_sampling_weights[key], take)
        else:
            sel = torch.randint(0, shared.numel(), (take,), device=device)
        pick = shared[sel]
        x = points[pick]
        xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j])
        if args.interface_normal_mode == "mask_levelset":
            n_if = overlap_normals[key][sel]
        else:
            n_seed = (seeds[j] - seeds[i]).unsqueeze(0).repeat(take, 1)
            n_if = normalize_rows_tensor(n_seed, eps=args.interface_normal_eps)
        return x, xi_i, xi_j, n_if

    # Fixed eval caches.
    eval_rng = np.random.default_rng(args.eval_cache_seed)
    eval_cache_pde: Dict[int, torch.Tensor] = {}
    eval_cache_if: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    eval_cache_bc: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def fixed_pick(idx: torch.Tensor, n_samples: int) -> Optional[torch.Tensor]:
        if idx.numel() == 0 or n_samples <= 0:
            return None
        n_take = min(int(n_samples), int(idx.numel()))
        base = idx.detach().cpu().numpy()
        sel_np = eval_rng.choice(base, size=n_take, replace=False)
        return torch.tensor(sel_np, device=device, dtype=torch.int64)

    if args.eval_fixed_cache:
        for i in range(n_charts):
            idx = point_idx_by_chart[i]
            pick = fixed_pick(idx, max(16, int(args.eval_cache_per_chart)))
            if pick is not None:
                x = points[pick]
                xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
                eval_cache_pde[i] = xi.detach()

            eval_cache_bc[i] = {}
            for key in ["inlet", "fresh", "gauge", "noflow", "nodiff"]:
                pick_bc = fixed_pick(chart_boundary_idx[i][key], max(8, int(args.eval_bc_samples_per_chart)))
                if pick_bc is None:
                    continue
                x = points[pick_bc]
                xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i]).detach()
                n_phys = normals[pick_bc].detach()
                eval_cache_bc[i][key] = (xi, n_phys)

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
            if args.interface_normal_mode == "mask_levelset":
                n_if = overlap_normals[(i, j)][sel].detach()
            else:
                n_seed = (seeds[j] - seeds[i]).unsqueeze(0).repeat(n_take, 1)
                n_if = normalize_rows_tensor(n_seed, eps=args.interface_normal_eps).detach()
            eval_cache_if[(i, j)] = (xi_i, xi_j, n_if)

    gravity = normalize_rows_tensor(torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=dtype))[0]

    # True and init permeability setup.
    true_q = euler_deg_to_quat_xyz(args.true_euler_x_deg, args.true_euler_y_deg, args.true_euler_z_deg, dtype=dtype).to(device=device)
    true_k0 = torch.as_tensor(args.k0_true, device=device, dtype=dtype)
    true_lam = torch.tensor([args.lam1_true, args.lam2_true, args.lam3_true], device=device, dtype=dtype)
    true_R = quat_to_rotmat(true_q)
    K_true = true_k0 * (true_R @ torch.diag(true_lam) @ true_R.transpose(0, 1))

    perm = PermeabilityParam(
        k0_init=args.k0_init,
        k0_min=args.k0_min,
        a1_init=args.a1_init,
        a2_init=args.a2_init,
        euler_init_deg=(args.init_euler_x_deg, args.init_euler_y_deg, args.init_euler_z_deg),
        device=device,
        dtype=dtype,
    ).to(device=device)

    # ---------------- Teacher stage ----------------
    p_teacher = [LocalScalarPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in range(n_charts)]
    c_teacher = [LocalScalarPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in range(n_charts)]
    opt_teacher = [torch.optim.Adam(list(p_teacher[i].parameters()) + list(c_teacher[i].parameters()), lr=args.teacher_lr) for i in range(n_charts)]

    def interface_terms(
        p_models: List[LocalScalarPINN],
        c_models: List[LocalScalarPINN],
        i: int,
        j: int,
        xi_i: torch.Tensor,
        xi_j: torch.Tensor,
        n_if: torch.Tensor,
        K_tensor: torch.Tensor,
        detach_neighbor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_i = compute_local_state(
            p_model=p_models[i],
            c_model=c_models[i],
            decoder=decoders[i],
            xi=xi_i,
            seed=seeds[i],
            t1=t1[i],
            t2=t2[i],
            n=nvec[i],
            chart_scale=support_r[i],
            K_tensor=K_tensor,
            Ra=args.Ra,
            Dm=args.Dm,
            gravity=gravity,
            sigma_floor=args.sigma_floor,
            det_floor=args.det_floor,
            jac_kappa_max=args.jac_kappa_max,
            create_graph=True,
        )
        if detach_neighbor:
            state_j = compute_local_state(
                p_model=p_models[j],
                c_model=c_models[j],
                decoder=decoders[j],
                xi=xi_j,
                seed=seeds[j],
                t1=t1[j],
                t2=t2[j],
                n=nvec[j],
                chart_scale=support_r[j],
                K_tensor=K_tensor,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=False,
            )
        else:
            state_j = compute_local_state(
                p_model=p_models[j],
                c_model=c_models[j],
                decoder=decoders[j],
                xi=xi_j,
                seed=seeds[j],
                t1=t1[j],
                t2=t2[j],
                n=nvec[j],
                chart_scale=support_r[j],
                K_tensor=K_tensor,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=True,
            )

        p_j = state_j.p.detach() if detach_neighbor else state_j.p
        c_j = state_j.c.detach() if detach_neighbor else state_j.c
        u_j = state_j.u.detach() if detach_neighbor else state_j.u
        grad_c_j = state_j.grad_c_x.detach() if detach_neighbor else state_j.grad_c_x

        dp = state_i.p - p_j
        dc = state_i.c - c_j
        loss_if_val = 0.5 * (torch.mean(dp**2) + torch.mean(dc**2))

        fi = torch.sum(state_i.u * n_if, dim=1, keepdim=True)
        fj = torch.sum(u_j * n_if, dim=1, keepdim=True)

        ji = state_i.u * state_i.c - args.Dm * state_i.grad_c_x
        jj = u_j * c_j - args.Dm * grad_c_j
        gi = torch.sum(ji * n_if, dim=1, keepdim=True)
        gj = torch.sum(jj * n_if, dim=1, keepdim=True)

        loss_flux_metric = 0.5 * (torch.mean((fi - fj) ** 2) + torch.mean((gi - gj) ** 2))
        if args.interface_transmission_mode == "robin":
            robin_p = args.robin_lambda * dp + (fi - fj)
            robin_c = args.robin_lambda * dc + (gi - gj)
            loss_flux_train = 0.5 * (torch.mean(robin_p**2) + torch.mean(robin_c**2))
        else:
            loss_flux_train = loss_flux_metric

        return loss_if_val, loss_flux_metric, loss_flux_train

    def boundary_losses(
        p_models: List[LocalScalarPINN],
        c_models: List[LocalScalarPINN],
        i: int,
        K_tensor: torch.Tensor,
        n_each: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.tensor(0.0, device=device, dtype=dtype)
        loss_p = z.clone()
        loss_c = z.clone()
        loss_flow = z.clone()
        loss_diff = z.clone()

        ib = sample_chart_boundary(i, "gauge", n_each)
        if ib is not None:
            xi, _, _ = ib
            p_hat = p_models[i](xi)
            loss_p = loss_p + torch.mean(p_hat**2)

        ib = sample_chart_boundary(i, "inlet", n_each)
        if ib is not None:
            xi, _, _ = ib
            c_hat = c_models[i](xi)
            loss_c = loss_c + torch.mean((c_hat - 1.0) ** 2)

        ib = sample_chart_boundary(i, "fresh", n_each)
        if ib is not None:
            xi, _, _ = ib
            c_hat = c_models[i](xi)
            loss_c = loss_c + torch.mean(c_hat**2)

        ib = sample_chart_boundary(i, "noflow", n_each)
        if ib is not None:
            xi, _, n_phys = ib
            st = compute_local_state(
                p_model=p_models[i],
                c_model=c_models[i],
                decoder=decoders[i],
                xi=xi,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
                K_tensor=K_tensor,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=True,
            )
            un = torch.sum(st.u * n_phys, dim=1, keepdim=True)
            loss_flow = loss_flow + torch.mean(un**2)

        ib = sample_chart_boundary(i, "nodiff", n_each)
        if ib is not None:
            xi, _, n_phys = ib
            st = compute_local_state(
                p_model=p_models[i],
                c_model=c_models[i],
                decoder=decoders[i],
                xi=xi,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
                K_tensor=K_tensor,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=True,
            )
            qn = torch.sum((-args.Dm * st.grad_c_x) * n_phys, dim=1, keepdim=True)
            loss_diff = loss_diff + torch.mean(qn**2)

        return loss_p, loss_c, loss_flow, loss_diff

    print(f"Teacher stage: iters={args.teacher_iters}")
    for it in range(1, args.teacher_iters + 1):
        losses_it: List[float] = []
        for group in color_groups:
            for i in group:
                if point_idx_by_chart[i].numel() == 0:
                    continue
                old_p = copy_state_dict(p_teacher[i].state_dict())
                old_c = copy_state_dict(c_teacher[i].state_dict())
                local_steps_i = max(1, int(round(args.teacher_local_steps * (args.detail_chart_step_boost if detail_chart_mask[i] else 1.0))))
                for _ in range(local_steps_i):
                    opt_teacher[i].zero_grad()
                    with amp_ctx():
                        xi_int = sample_local_xi(i, args.teacher_pde_batch)
                        st = compute_local_state(
                            p_model=p_teacher[i],
                            c_model=c_teacher[i],
                            decoder=decoders[i],
                            xi=xi_int,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                            K_tensor=K_true,
                            Ra=args.Ra,
                            Dm=args.Dm,
                            gravity=gravity,
                            sigma_floor=args.sigma_floor,
                            det_floor=args.det_floor,
                            jac_kappa_max=args.jac_kappa_max,
                            create_graph=True,
                        )
                        flow_res = select_residual_samples(st.flow_residual, st.valid, clip_q=args.pde_clip_quantile)
                        trans_res = select_residual_samples(st.trans_residual, st.valid, clip_q=args.pde_clip_quantile)
                        loss_flow = torch.mean(flow_res**2)
                        loss_trans = torch.mean(trans_res**2)

                        lp, lc, lnf, lnd = boundary_losses(
                            p_models=p_teacher,
                            c_models=c_teacher,
                            i=i,
                            K_tensor=K_true,
                            n_each=args.teacher_bc_batch,
                        )
                        loss_bc = (
                            args.w_bc_p * lp
                            + args.w_bc_c * lc
                            + args.w_bc_flow * lnf
                            + args.w_bc_diff * lnd
                        )

                        iv_terms: List[torch.Tensor] = []
                        if_terms: List[torch.Tensor] = []
                        for j in neighbors[i]:
                            ib = interface_batch(i, j, args.teacher_if_batch)
                            if ib is None:
                                continue
                            _, xi_i, xi_j, n_if = ib
                            liv, _, lif_train = interface_terms(
                                p_teacher,
                                c_teacher,
                                i=i,
                                j=j,
                                xi_i=xi_i,
                                xi_j=xi_j,
                                n_if=n_if,
                                K_tensor=K_true,
                                detach_neighbor=True,
                            )
                            pair_key = (i, j) if i < j else (j, i)
                            pair_boost = 1.0 + (args.interface_detail_boost - 1.0) * overlap_detail_strength.get(pair_key, 0.0)
                            iv_terms.append(pair_boost * liv)
                            if_terms.append(pair_boost * lif_train)

                        loss_if_val = torch.mean(torch.stack(iv_terms)) if iv_terms else torch.tensor(0.0, device=device, dtype=dtype)
                        loss_if_flux = torch.mean(torch.stack(if_terms)) if if_terms else torch.tensor(0.0, device=device, dtype=dtype)

                        loss_total = (
                            args.w_flow * loss_flow
                            + args.w_trans * loss_trans
                            + args.w_bc * loss_bc
                            + args.w_if_val * loss_if_val
                            + args.w_if_flux_teacher * loss_if_flux
                        )

                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(list(p_teacher[i].parameters()) + list(c_teacher[i].parameters()), max_norm=5.0)
                    opt_teacher[i].step()

                    losses_it.append(float(loss_total.item()))

                if args.teacher_omega < 1.0:
                    blend_model_with_old(p_teacher[i], old_p, omega=args.teacher_omega)
                    blend_model_with_old(c_teacher[i], old_c, omega=args.teacher_omega)

        if it % max(1, args.teacher_log_every) == 0 and losses_it:
            print(f"[Teacher] iter={it}/{args.teacher_iters} loss={np.mean(losses_it):.3e}")

    # ---------------- Observation stage ----------------
    print("Building sparse teacher observations")
    obs_by_chart: List[Dict[str, torch.Tensor]] = []
    for i in range(n_charts):
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            obs_by_chart.append({
                "xi": torch.zeros((0, 3), device=device, dtype=dtype),
                "p": torch.zeros((0, 1), device=device, dtype=dtype),
                "c": torch.zeros((0, 1), device=device, dtype=dtype),
                "u": torch.zeros((0, 3), device=device, dtype=dtype),
            })
            continue

        n_take = min(int(idx.numel()), int(args.n_obs_per_chart))
        if args.curvature_adaptive_sampling:
            local_w = chart_sampling_weights[i]
            sel = sample_indices_from_weights(local_w, n_take)
            pick = idx[sel]
        else:
            sel = torch.randint(0, idx.numel(), (n_take,), device=device)
            pick = idx[sel]

        x_obs = points[pick]
        xi_obs = local_coords(x_obs, seeds[i], t1[i], t2[i], nvec[i])
        with torch.enable_grad():
            st = compute_local_state(
                p_model=p_teacher[i],
                c_model=c_teacher[i],
                decoder=decoders[i],
                xi=xi_obs,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
                K_tensor=K_true,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=False,
            )
        obs = {
            "xi": xi_obs.detach(),
            "p": st.p.detach(),
            "c": st.c.detach(),
            "u": st.u.detach(),
        }
        obs_by_chart.append(obs)

    # ---------------- Inverse stage ----------------
    p_inv = [LocalScalarPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in range(n_charts)]
    c_inv = [LocalScalarPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in range(n_charts)]
    if args.warm_start_teacher:
        for i in range(n_charts):
            p_inv[i].load_state_dict(copy.deepcopy(p_teacher[i].state_dict()))
            c_inv[i].load_state_dict(copy.deepcopy(c_teacher[i].state_dict()))

    chart_opts = [
        torch.optim.Adam(list(p_inv[i].parameters()) + list(c_inv[i].parameters()), lr=args.lr)
        for i in range(n_charts)
    ]
    perm_opt = torch.optim.Adam(list(perm.parameters()), lr=args.lr_perm)

    def current_lr() -> float:
        if not chart_opts or not chart_opts[0].param_groups:
            return 0.0
        return float(chart_opts[0].param_groups[0]["lr"])

    def current_lr_perm() -> float:
        if not perm_opt.param_groups:
            return 0.0
        return float(perm_opt.param_groups[0]["lr"])

    def set_lr_all(new_lr_chart: float, new_lr_perm: Optional[float] = None) -> None:
        for opt in chart_opts:
            for g in opt.param_groups:
                g["lr"] = float(new_lr_chart)
        if new_lr_perm is None:
            new_lr_perm = new_lr_chart
        for g in perm_opt.param_groups:
            g["lr"] = float(new_lr_perm)

    def snapshot_state() -> Dict[str, object]:
        return {
            "p_states": [copy_state_dict(m.state_dict()) for m in p_inv],
            "c_states": [copy_state_dict(m.state_dict()) for m in c_inv],
            "perm_state": copy_state_dict(perm.state_dict()),
        }

    def load_snapshot(s: Dict[str, object]) -> None:
        p_states = s["p_states"]  # type: ignore[index]
        c_states = s["c_states"]  # type: ignore[index]
        perm_state = s["perm_state"]  # type: ignore[index]
        for i in range(n_charts):
            p_inv[i].load_state_dict(p_states[i])
            c_inv[i].load_state_dict(c_states[i])
        perm.load_state_dict(perm_state)

    def eval_metrics() -> Dict[str, float]:
        K_est = perm.K_tensor()
        eval_flow_terms: List[torch.Tensor] = []
        eval_trans_terms: List[torch.Tensor] = []
        eval_if_val_terms: List[torch.Tensor] = []
        eval_if_flux_terms: List[torch.Tensor] = []
        eval_data_p_num = 0.0
        eval_data_p_den = 0.0
        eval_data_c_num = 0.0
        eval_data_c_den = 0.0

        with torch.enable_grad():
            for i in range(n_charts):
                if point_idx_by_chart[i].numel() == 0:
                    continue
                if args.eval_fixed_cache and i in eval_cache_pde:
                    xi_int = eval_cache_pde[i]
                else:
                    xi_int = sample_local_xi(i, max(16, args.eval_pde_samples_per_chart))

                st = compute_local_state(
                    p_model=p_inv[i],
                    c_model=c_inv[i],
                    decoder=decoders[i],
                    xi=xi_int,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                    K_tensor=K_est,
                    Ra=args.Ra,
                    Dm=args.Dm,
                    gravity=gravity,
                    sigma_floor=args.sigma_floor,
                    det_floor=args.det_floor,
                    jac_kappa_max=args.jac_kappa_max,
                    create_graph=True,
                )
                fr = select_residual_samples(st.flow_residual, st.valid, clip_q=1.0)
                tr = select_residual_samples(st.trans_residual, st.valid, clip_q=1.0)
                eval_flow_terms.append(torch.mean(fr**2))
                eval_trans_terms.append(torch.mean(tr**2))

                obs_i = obs_by_chart[i]
                if obs_i["xi"].numel() > 0:
                    p_hat = p_inv[i](obs_i["xi"])
                    c_hat = c_inv[i](obs_i["xi"])
                    eval_data_p_num += float(torch.mean((p_hat - obs_i["p"]) ** 2).item())
                    eval_data_p_den += float(torch.mean(obs_i["p"] ** 2).item()) + 1e-12
                    eval_data_c_num += float(torch.mean((c_hat - obs_i["c"]) ** 2).item())
                    eval_data_c_den += float(torch.mean(obs_i["c"] ** 2).item()) + 1e-12

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
                    liv, lif_metric, _ = interface_terms(
                        p_inv,
                        c_inv,
                        i=i,
                        j=j,
                        xi_i=xi_i,
                        xi_j=xi_j,
                        n_if=n_if,
                        K_tensor=K_est,
                        detach_neighbor=False,
                    )
                    eval_if_val_terms.append(liv)
                    eval_if_flux_terms.append(lif_metric)

        flow_m = float(torch.mean(torch.stack(eval_flow_terms)).item()) if eval_flow_terms else 0.0
        trans_m = float(torch.mean(torch.stack(eval_trans_terms)).item()) if eval_trans_terms else 0.0
        if_val_m = float(torch.mean(torch.stack(eval_if_val_terms)).item()) if eval_if_val_terms else 0.0
        if_flux_m = float(torch.mean(torch.stack(eval_if_flux_terms)).item()) if eval_if_flux_terms else 0.0

        rel_p = math.sqrt(max(eval_data_p_num, 0.0) / max(eval_data_p_den, 1e-12)) if eval_data_p_den > 0.0 else 0.0
        rel_c = math.sqrt(max(eval_data_c_num, 0.0) / max(eval_data_c_den, 1e-12)) if eval_data_c_den > 0.0 else 0.0
        field_rel = 0.5 * (rel_p + rel_c)

        k0_est = float(perm.k0().item())
        evals_est, evecs_est = sorted_eigh_desc(K_est.detach().cpu())
        evals_true, evecs_true = sorted_eigh_desc(K_true.detach().cpu())
        eig_rel = torch.abs(evals_est - evals_true) / torch.clamp(evals_true, min=1e-12)
        axis_deg = axis_angle_errors_deg(evecs_est, evecs_true)

        k0_rel = abs(k0_est - args.k0_true) / max(args.k0_true, 1e-12)
        eig_rel_mean = float(torch.mean(eig_rel).item())
        eig_rel_max = float(torch.max(eig_rel).item())
        axis_mean = float(torch.mean(axis_deg).item())
        axis_max = float(torch.max(axis_deg).item())
        param_score = k0_rel + eig_rel_mean + axis_mean / 90.0

        return {
            "flow": flow_m,
            "trans": trans_m,
            "if_val": if_val_m,
            "if_flux": if_flux_m,
            "rel_p": rel_p,
            "rel_c": rel_c,
            "field_rel": field_rel,
            "k0_est": k0_est,
            "k0_rel": k0_rel,
            "eig1_est": float(evals_est[0].item()),
            "eig2_est": float(evals_est[1].item()),
            "eig3_est": float(evals_est[2].item()),
            "eig_rel_mean": eig_rel_mean,
            "eig_rel_max": eig_rel_max,
            "axis_deg_mean": axis_mean,
            "axis_deg_max": axis_max,
            "param_score": param_score,
            "lambda_min": float(torch.min(evals_est).item()),
        }

    history: Dict[str, List[float]] = {
        "flow": [],
        "trans": [],
        "if_val": [],
        "if_flux": [],
        "field_rel": [],
        "k0": [],
        "eig1": [],
        "eig2": [],
        "eig3": [],
        "k0_rel": [],
        "eig_rel_mean": [],
        "axis_deg_mean": [],
        "param_score": [],
        "w_if_flux_eff": [],
        "adaptive_flux_multiplier": [],
        "lr_chart": [],
        "lr_perm": [],
        "iter_rejected": [],
    }

    def effective_flux_weight(it: int) -> float:
        if args.flux_ramp_iters <= 0:
            return float(args.w_if_flux_end)
        alpha = min(1.0, float(it) / max(1.0, float(args.flux_ramp_iters)))
        return float(args.w_if_flux_start + (args.w_if_flux_end - args.w_if_flux_start) * alpha)

    snapshots: Dict[str, Optional[Dict[str, object]]] = {
        "best_score": None,
        "best_field": None,
        "best_target": None,
        "best_flux": None,
    }
    best_score = float("inf")
    best_field = float("inf")
    best_target_obj = float("inf")
    best_flux = float("inf")
    stale = 0
    guard_stale = 0
    reject_count = 0
    adaptive_flux_mult = 1.0

    def maybe_record_snapshot(name: str, it: int, m: Dict[str, float], score: float) -> None:
        snapshots[name] = {
            "iter": int(it),
            "metrics": dict(m),
            "score": float(score),
            "state": snapshot_state(),
            "lr_chart": current_lr(),
            "lr_perm": current_lr_perm(),
        }

    print(f"Inverse stage: max_iters={args.max_schwarz_iters}")
    schwarz_start = time.time()
    for it in range(1, args.max_schwarz_iters + 1):
        flux_base = effective_flux_weight(it)
        if args.adaptive_flux_weight:
            w_if_flux_eff = float(np.clip(flux_base * adaptive_flux_mult, args.w_if_flux_start, args.w_if_flux_end))
        else:
            w_if_flux_eff = flux_base

        iter_before = snapshot_state()
        lr_chart_before = current_lr()
        lr_perm_before = current_lr_perm()
        iter_rejected = 0

        for group in color_groups:
            for i in group:
                if point_idx_by_chart[i].numel() == 0:
                    continue
                old_p = copy_state_dict(p_inv[i].state_dict())
                old_c = copy_state_dict(c_inv[i].state_dict())
                chart_is_detail = bool(detail_chart_mask[i]) if args.curvature_adaptive_sampling else False
                step_boost = float(args.detail_chart_step_boost) if chart_is_detail else 1.0
                batch_boost = float(args.detail_chart_batch_boost) if chart_is_detail else 1.0
                local_steps_i = max(1, int(round(args.local_steps * max(1.0, step_boost))))
                pde_batch_i = max(8, int(round(args.pde_batch * max(1.0, batch_boost))))
                bc_batch_i = max(8, int(round(args.bc_batch * max(1.0, batch_boost))))
                if_batch_i = max(8, int(round(args.if_batch * max(1.0, batch_boost))))

                for _ in range(local_steps_i):
                    chart_opts[i].zero_grad()
                    perm_opt.zero_grad()
                    K_est = perm.K_tensor()

                    with amp_ctx():
                        xi_int = sample_local_xi(i, pde_batch_i)
                        st = compute_local_state(
                            p_model=p_inv[i],
                            c_model=c_inv[i],
                            decoder=decoders[i],
                            xi=xi_int,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                            K_tensor=K_est,
                            Ra=args.Ra,
                            Dm=args.Dm,
                            gravity=gravity,
                            sigma_floor=args.sigma_floor,
                            det_floor=args.det_floor,
                            jac_kappa_max=args.jac_kappa_max,
                            create_graph=True,
                        )
                        flow_res = select_residual_samples(st.flow_residual, st.valid, clip_q=args.pde_clip_quantile)
                        trans_res = select_residual_samples(st.trans_residual, st.valid, clip_q=args.pde_clip_quantile)
                        loss_flow = torch.mean(flow_res**2)
                        loss_trans = torch.mean(trans_res**2)

                        lp, lc, lnf, lnd = boundary_losses(
                            p_models=p_inv,
                            c_models=c_inv,
                            i=i,
                            K_tensor=K_est,
                            n_each=bc_batch_i,
                        )
                        loss_bc = (
                            args.w_bc_p * lp
                            + args.w_bc_c * lc
                            + args.w_bc_flow * lnf
                            + args.w_bc_diff * lnd
                        )

                        obs_i = obs_by_chart[i]
                        loss_data_p = torch.tensor(0.0, device=device, dtype=dtype)
                        loss_data_c = torch.tensor(0.0, device=device, dtype=dtype)
                        loss_data_u = torch.tensor(0.0, device=device, dtype=dtype)
                        if obs_i["xi"].numel() > 0:
                            p_hat = p_inv[i](obs_i["xi"])
                            c_hat = c_inv[i](obs_i["xi"])
                            loss_data_p = torch.mean((p_hat - obs_i["p"]) ** 2)
                            loss_data_c = torch.mean((c_hat - obs_i["c"]) ** 2)
                            if args.w_data_u > 0.0 and obs_i["u"].numel() > 0:
                                st_obs = compute_local_state(
                                    p_model=p_inv[i],
                                    c_model=c_inv[i],
                                    decoder=decoders[i],
                                    xi=obs_i["xi"],
                                    seed=seeds[i],
                                    t1=t1[i],
                                    t2=t2[i],
                                    n=nvec[i],
                                    chart_scale=support_r[i],
                                    K_tensor=K_est,
                                    Ra=args.Ra,
                                    Dm=args.Dm,
                                    gravity=gravity,
                                    sigma_floor=args.sigma_floor,
                                    det_floor=args.det_floor,
                                    jac_kappa_max=args.jac_kappa_max,
                                    create_graph=True,
                                )
                                loss_data_u = torch.mean((st_obs.u - obs_i["u"]) ** 2)
                        loss_data = args.w_data_p * loss_data_p + args.w_data_c * loss_data_c + args.w_data_u * loss_data_u

                        iv_terms: List[torch.Tensor] = []
                        if_terms: List[torch.Tensor] = []
                        for j in neighbors[i]:
                            ib = interface_batch(i, j, if_batch_i)
                            if ib is None:
                                continue
                            _, xi_i, xi_j, n_if = ib
                            liv, _, lif_train = interface_terms(
                                p_inv,
                                c_inv,
                                i=i,
                                j=j,
                                xi_i=xi_i,
                                xi_j=xi_j,
                                n_if=n_if,
                                K_tensor=K_est,
                                detach_neighbor=True,
                            )
                            pair_key = (i, j) if i < j else (j, i)
                            pair_boost = 1.0
                            if args.interface_detail_boost > 1.0:
                                d_pair = overlap_detail_strength.get(pair_key, 0.0)
                                pair_boost = 1.0 + (args.interface_detail_boost - 1.0) * d_pair
                            iv_terms.append(pair_boost * liv)
                            if_terms.append(pair_boost * lif_train)

                        loss_if_val = torch.mean(torch.stack(iv_terms)) if iv_terms else torch.tensor(0.0, device=device, dtype=dtype)
                        loss_if_flux = torch.mean(torch.stack(if_terms)) if if_terms else torch.tensor(0.0, device=device, dtype=dtype)

                        # Parameter regularization.
                        k0_est = perm.k0()
                        loss_reg = (
                            args.w_reg_k0 * ((k0_est - args.k0_prior) / max(args.k0_prior, 1e-8)) ** 2
                            + args.w_reg_aniso * (perm.a1**2 + perm.a2**2)
                            + args.w_reg_quat * (torch.linalg.norm(perm.q_raw) - 1.0) ** 2
                        )

                        loss_total = (
                            args.w_flow * loss_flow
                            + args.w_trans * loss_trans
                            + args.w_bc * loss_bc
                            + args.w_if_val * loss_if_val
                            + w_if_flux_eff * loss_if_flux
                            + args.w_data * loss_data
                            + args.w_reg * loss_reg
                        )

                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(list(p_inv[i].parameters()) + list(c_inv[i].parameters()), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(list(perm.parameters()), max_norm=5.0)
                    chart_opts[i].step()
                    perm_opt.step()

                if args.omega < 1.0:
                    blend_model_with_old(p_inv[i], old_p, omega=args.omega)
                    blend_model_with_old(c_inv[i], old_c, omega=args.omega)

        m = eval_metrics()
        field_rel_proposed = m["field_rel"]

        if args.iter_accept_field_rel > 0.0 and field_rel_proposed > args.iter_accept_field_rel:
            load_snapshot(iter_before)
            set_lr_all(
                new_lr_chart=max(1e-7, lr_chart_before * args.iter_reject_lr_decay),
                new_lr_perm=max(1e-7, lr_perm_before * args.iter_reject_lr_decay),
            )
            m = eval_metrics()
            iter_rejected = 1
            reject_count += 1
            print(
                f"[TrustRegion] iter={it} rejected: field_rel={field_rel_proposed:.3e} > "
                f"{args.iter_accept_field_rel:.3e}; restored state and lr={current_lr():.3e}"
            )

        if args.adaptive_flux_weight:
            if m["field_rel"] <= args.adaptive_flux_field_thresh:
                adaptive_flux_mult *= float(args.adaptive_flux_up)
            else:
                adaptive_flux_mult *= float(args.adaptive_flux_down)
            adaptive_flux_mult = max(1e-6, adaptive_flux_mult)

        history["flow"].append(m["flow"])
        history["trans"].append(m["trans"])
        history["if_val"].append(m["if_val"])
        history["if_flux"].append(m["if_flux"])
        history["field_rel"].append(m["field_rel"])
        history["k0"].append(m["k0_est"])
        history["eig1"].append(m["eig1_est"])
        history["eig2"].append(m["eig2_est"])
        history["eig3"].append(m["eig3_est"])
        history["k0_rel"].append(m["k0_rel"])
        history["eig_rel_mean"].append(m["eig_rel_mean"])
        history["axis_deg_mean"].append(m["axis_deg_mean"])
        history["param_score"].append(m["param_score"])
        history["w_if_flux_eff"].append(w_if_flux_eff)
        history["adaptive_flux_multiplier"].append(float(adaptive_flux_mult))
        history["lr_chart"].append(current_lr())
        history["lr_perm"].append(current_lr_perm())
        history["iter_rejected"].append(float(iter_rejected))

        score = (
            args.w_flow * m["flow"]
            + args.w_trans * m["trans"]
            + args.w_if_val * m["if_val"]
            + w_if_flux_eff * m["if_flux"]
            + args.w_data * (m["rel_p"] + m["rel_c"])
        )

        if score + args.plateau_tol < best_score:
            best_score = score
            stale = 0
            maybe_record_snapshot("best_score", it, m, score)
        else:
            stale += 1

        if m["field_rel"] + 1e-14 < best_field:
            best_field = m["field_rel"]
            maybe_record_snapshot("best_field", it, m, score)

        target_cond = (
            m["k0_rel"] <= args.acc_k0_rel
            and m["eig_rel_max"] <= args.acc_eig_rel
            and m["axis_deg_max"] <= args.acc_axis_deg
            and m["field_rel"] <= args.target_field_rel
        )
        if target_cond and m["param_score"] + 1e-14 < best_target_obj:
            best_target_obj = m["param_score"]
            maybe_record_snapshot("best_target", it, m, score)

        if m["field_rel"] <= args.guard_field_rel and m["if_flux"] + 1e-14 < best_flux:
            best_flux = m["if_flux"]
            maybe_record_snapshot("best_flux", it, m, score)

        if m["field_rel"] > args.guard_field_rel:
            guard_stale += 1
        else:
            guard_stale = 0

        if args.guard_patience > 0 and guard_stale >= args.guard_patience:
            snap = snapshots.get("best_target") or snapshots.get("best_field") or snapshots.get("best_score")
            if snap is not None:
                load_snapshot(snap["state"])  # type: ignore[index]
                set_lr_all(
                    new_lr_chart=max(1e-7, current_lr() * args.guard_lr_decay),
                    new_lr_perm=max(1e-7, current_lr_perm() * args.guard_lr_decay),
                )
                print(
                    f"[Guard] field_rel exceeded {args.guard_field_rel:.3e} for {args.guard_patience} evals; "
                    f"restored iter={snap['iter']} and reduced lr to {current_lr():.3e}"
                )
            guard_stale = 0

        if it % max(1, args.log_every) == 0:
            elapsed = time.time() - schwarz_start
            print(
                f"[Inverse] iter={it}/{args.max_schwarz_iters} "
                f"flow={m['flow']:.3e} trans={m['trans']:.3e} if_val={m['if_val']:.3e} if_flux={m['if_flux']:.3e} "
                f"field_rel={m['field_rel']:.3e} k0={m['k0_est']:.6f} eig=({m['eig1_est']:.3f},{m['eig2_est']:.3f},{m['eig3_est']:.3f}) "
                f"axis_mean={m['axis_deg_mean']:.2f}deg w_if={w_if_flux_eff:.2e} stale={stale} rej={iter_rejected} "
                f"lr={current_lr():.3e}/{current_lr_perm():.3e} t={elapsed:.1f}s"
            )

        if stale >= args.plateau_patience:
            print(f"Stopped by plateau patience at iteration {it}")
            break

    # Select checkpoint state.
    def pick_snapshot(name: str, backup_order: Sequence[str]) -> Tuple[Dict[str, object], Optional[str]]:
        s = snapshots.get(name)
        if s is not None:
            return s, None
        for b in backup_order:
            sb = snapshots.get(b)
            if sb is not None:
                return sb, b
        # fallback to current
        return {
            "iter": len(history["flow"]),
            "metrics": eval_metrics(),
            "score": float("nan"),
            "state": snapshot_state(),
            "lr_chart": current_lr(),
            "lr_perm": current_lr_perm(),
        }, "last"

    if args.checkpoint_policy == "best_target":
        selected, selected_fallback = pick_snapshot("best_target", ["best_field", "best_score", "best_flux"])
    elif args.checkpoint_policy == "best_field":
        selected, selected_fallback = pick_snapshot("best_field", ["best_target", "best_score", "best_flux"])
    elif args.checkpoint_policy == "best_flux":
        selected, selected_fallback = pick_snapshot("best_flux", ["best_target", "best_field", "best_score"])
    elif args.checkpoint_policy == "best_score":
        selected, selected_fallback = pick_snapshot("best_score", ["best_field", "best_target", "best_flux"])
    else:
        selected = {
            "iter": len(history["flow"]),
            "metrics": eval_metrics(),
            "score": float("nan"),
            "state": snapshot_state(),
            "lr_chart": current_lr(),
            "lr_perm": current_lr_perm(),
        }
        selected_fallback = None

    load_snapshot(selected["state"])  # type: ignore[index]
    final_m = eval_metrics()

    # Build full-point outputs with primary chart assignment.
    p_true = torch.zeros((n_points, 1), device=device, dtype=dtype)
    c_true = torch.zeros((n_points, 1), device=device, dtype=dtype)
    u_true = torch.zeros((n_points, 3), device=device, dtype=dtype)
    p_pred = torch.zeros((n_points, 1), device=device, dtype=dtype)
    c_pred = torch.zeros((n_points, 1), device=device, dtype=dtype)
    u_pred = torch.zeros((n_points, 3), device=device, dtype=dtype)

    with torch.enable_grad():
        for i in range(n_charts):
            idx = torch.where(primary_chart == i)[0]
            if idx.numel() == 0:
                continue
            x_i = points[idx]
            xi_i = local_coords(x_i, seeds[i], t1[i], t2[i], nvec[i])

            st_true = compute_local_state(
                p_model=p_teacher[i],
                c_model=c_teacher[i],
                decoder=decoders[i],
                xi=xi_i,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
                K_tensor=K_true,
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=False,
            )
            st_pred = compute_local_state(
                p_model=p_inv[i],
                c_model=c_inv[i],
                decoder=decoders[i],
                xi=xi_i,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
                K_tensor=perm.K_tensor(),
                Ra=args.Ra,
                Dm=args.Dm,
                gravity=gravity,
                sigma_floor=args.sigma_floor,
                det_floor=args.det_floor,
                jac_kappa_max=args.jac_kappa_max,
                create_graph=False,
            )

            p_true[idx] = st_true.p.detach()
            c_true[idx] = st_true.c.detach()
            u_true[idx] = st_true.u.detach()
            p_pred[idx] = st_pred.p.detach()
            c_pred[idx] = st_pred.c.detach()
            u_pred[idx] = st_pred.u.detach()

    p_err = p_pred - p_true
    c_err = c_pred - c_true
    u_err = u_pred - u_true
    u_err_mag = torch.linalg.norm(u_err, dim=1, keepdim=True)

    rel_p_global = float(torch.sqrt(torch.mean(p_err**2) / torch.clamp(torch.mean(p_true**2), min=1e-12)).item())
    rel_c_global = float(torch.sqrt(torch.mean(c_err**2) / torch.clamp(torch.mean(c_true**2), min=1e-12)).item())

    evals_est, evecs_est = sorted_eigh_desc(perm.K_tensor().detach().cpu())
    evals_true, evecs_true = sorted_eigh_desc(K_true.detach().cpu())
    eig_rel = torch.abs(evals_est - evals_true) / torch.clamp(evals_true, min=1e-12)
    axis_deg = axis_angle_errors_deg(evecs_est, evecs_true)

    k0_est = float(perm.k0().item())
    k0_rel = abs(k0_est - args.k0_true) / max(args.k0_true, 1e-12)
    target_met = bool(
        (k0_rel <= args.acc_k0_rel)
        and (float(torch.max(eig_rel).item()) <= args.acc_eig_rel)
        and (float(torch.max(axis_deg).item()) <= args.acc_axis_deg)
    )

    out = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "amp_used": bool(use_amp),
        "amp_backend": amp_backend,
        "n_charts": int(n_charts),
        "n_points": int(n_points),
        "teacher_iters": int(args.teacher_iters),
        "inverse_iters_ran": int(len(history["flow"])),
        "checkpoint_policy": args.checkpoint_policy,
        "selected_state_iter": int(selected["iter"]),
        "selected_state_fallback": selected_fallback,
        "curvature_adaptive_sampling": bool(args.curvature_adaptive_sampling),
        "curvature_adaptive_interface_sampling": bool(args.curvature_adaptive_interface_sampling),
        "detail_chart_ids": [int(i) for i, flag in enumerate(detail_chart_mask) if flag],
        "n_high_detail_points": int(n_high_detail_points),
        "high_detail_fraction": float(n_high_detail_points / max(1, n_points)),
        "k0_true": float(args.k0_true),
        "k0_init": float(args.k0_init),
        "k0_est": float(k0_est),
        "k0_rel_error": float(k0_rel),
        "eig_true": [float(v) for v in evals_true.tolist()],
        "eig_est": [float(v) for v in evals_est.tolist()],
        "eig_rel_error": [float(v) for v in eig_rel.tolist()],
        "eig_rel_error_mean": float(torch.mean(eig_rel).item()),
        "eig_rel_error_max": float(torch.max(eig_rel).item()),
        "axis_angle_error_deg": [float(v) for v in axis_deg.tolist()],
        "axis_angle_error_deg_mean": float(torch.mean(axis_deg).item()),
        "axis_angle_error_deg_max": float(torch.max(axis_deg).item()),
        "rel_p_global": rel_p_global,
        "rel_c_global": rel_c_global,
        "field_rel_global": 0.5 * (rel_p_global + rel_c_global),
        "final_flow_residual": float(final_m["flow"]),
        "final_trans_residual": float(final_m["trans"]),
        "final_interface_value": float(final_m["if_val"]),
        "final_interface_flux": float(final_m["if_flux"]),
        "lambda_min_est": float(torch.min(evals_est).item()),
        "spd_ok": bool(float(torch.min(evals_est).item()) > 0.0),
        "target_met": target_met,
        "acc_thresholds": {
            "k0_rel": float(args.acc_k0_rel),
            "eig_rel": float(args.acc_eig_rel),
            "axis_deg": float(args.acc_axis_deg),
        },
        "runtime_seconds": float(time.time() - run_start),
        "schwarz_runtime_seconds": float(time.time() - schwarz_start),
        "trust_region": {
            "iter_accept_field_rel": float(args.iter_accept_field_rel),
            "iter_reject_lr_decay": float(args.iter_reject_lr_decay),
            "rejected_iterations": int(reject_count),
        },
        "adaptive_flux": {
            "enabled": bool(args.adaptive_flux_weight),
            "up": float(args.adaptive_flux_up),
            "down": float(args.adaptive_flux_down),
            "field_thresh": float(args.adaptive_flux_field_thresh),
            "start": float(args.w_if_flux_start),
            "end": float(args.w_if_flux_end),
        },
    }

    def snap_summary(name: str) -> Optional[Dict[str, object]]:
        s = snapshots.get(name)
        if s is None:
            return None
        m = s["metrics"]  # type: ignore[index]
        return {
            "iter": int(s["iter"]),
            "k0_rel": float(m["k0_rel"]),
            "eig_rel_max": float(m["eig_rel_max"]),
            "axis_deg_max": float(m["axis_deg_max"]),
            "field_rel": float(m["field_rel"]),
            "if_flux": float(m["if_flux"]),
            "param_score": float(m["param_score"]),
            "score": float(s["score"]),
        }

    out["checkpoint_triplet"] = {
        "best_field": snap_summary("best_field"),
        "best_target": snap_summary("best_target"),
        "best_flux_under_guard": snap_summary("best_flux"),
        "best_score": snap_summary("best_score"),
    }

    # Persist artifacts.
    os.makedirs(args.output_dir, exist_ok=True)
    run_stem = build_run_stem(args.run_tag)

    solution_npz = os.path.join(args.output_dir, f"{run_stem}_solution.npz")
    pressure_velocity_npz = os.path.join(args.output_dir, f"{run_stem}_pressure_velocity_fields.npz")
    np.savez_compressed(
        solution_npz,
        points=points.detach().cpu().numpy(),
        normals=normals.detach().cpu().numpy(),
        chart_id=primary_chart.detach().cpu().numpy().astype(np.int32),
        detail_score=detail_score.detach().cpu().numpy(),
        high_detail_mask=high_detail_mask.detach().cpu().numpy().astype(np.uint8),
        p_true=p_true.detach().cpu().numpy().reshape(-1),
        p_pred=p_pred.detach().cpu().numpy().reshape(-1),
        p_error=p_err.detach().cpu().numpy().reshape(-1),
        p_error_mag=torch.abs(p_err).detach().cpu().numpy().reshape(-1),
        c_true=c_true.detach().cpu().numpy().reshape(-1),
        c_pred=c_pred.detach().cpu().numpy().reshape(-1),
        c_error=c_err.detach().cpu().numpy().reshape(-1),
        c_error_mag=torch.abs(c_err).detach().cpu().numpy().reshape(-1),
        u_true=u_true.detach().cpu().numpy(),
        u_pred=u_pred.detach().cpu().numpy(),
        u_error=u_err.detach().cpu().numpy(),
        u_error_mag=u_err_mag.detach().cpu().numpy().reshape(-1),
    )
    np.savez_compressed(
        pressure_velocity_npz,
        points=points.detach().cpu().numpy(),
        chart_id=primary_chart.detach().cpu().numpy().astype(np.int32),
        pressure_true=p_true.detach().cpu().numpy().reshape(-1),
        pressure_pred=p_pred.detach().cpu().numpy().reshape(-1),
        pressure_error=p_err.detach().cpu().numpy().reshape(-1),
        pressure_error_mag=torch.abs(p_err).detach().cpu().numpy().reshape(-1),
        velocity_true=u_true.detach().cpu().numpy(),
        velocity_pred=u_pred.detach().cpu().numpy(),
        velocity_error=u_err.detach().cpu().numpy(),
        velocity_error_mag=u_err_mag.detach().cpu().numpy().reshape(-1),
    )

    metrics_path = os.path.join(args.output_dir, f"{run_stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    history_path = os.path.join(args.output_dir, f"{run_stem}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig_path = os.path.join(args.output_dir, f"{run_stem}_curves.png")
    iters = np.arange(1, len(history["flow"]) + 1, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if iters.size == 0:
        titles = ["Residuals", "Permeability params", "Inverse errors", "Controls"]
        for ax, ttl in zip(axes.reshape(-1), titles):
            ax.set_title(ttl)
            ax.grid(True, alpha=0.3)
            ax.text(
                0.5,
                0.5,
                "No inverse iterations ran\n(max_schwarz_iters=0 or early stop/failure)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
    else:
        ax = axes[0, 0]
        ax.semilogy(iters, np.maximum(history["flow"], 1e-16), label="flow")
        ax.semilogy(iters, np.maximum(history["trans"], 1e-16), label="trans")
        ax.semilogy(iters, np.maximum(history["if_val"], 1e-16), label="if_val")
        ax.semilogy(iters, np.maximum(history["if_flux"], 1e-16), label="if_flux")
        ax.grid(True, alpha=0.3)
        ax.set_title("Residuals")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(iters, history["k0"], label="k0")
        ax.plot(iters, history["eig1"], label="eig1")
        ax.plot(iters, history["eig2"], label="eig2")
        ax.plot(iters, history["eig3"], label="eig3")
        ax.grid(True, alpha=0.3)
        ax.set_title("Permeability params")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(iters, history["k0_rel"], label="k0_rel")
        ax.plot(iters, history["eig_rel_mean"], label="eig_rel_mean")
        ax.plot(iters, history["axis_deg_mean"], label="axis_deg_mean")
        ax.plot(iters, history["field_rel"], label="field_rel")
        ax.grid(True, alpha=0.3)
        ax.set_title("Inverse errors")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(iters, history["w_if_flux_eff"], label="w_if_flux_eff")
        ax.plot(iters, history["adaptive_flux_multiplier"], label="adaptive_flux_multiplier")
        ax.plot(iters, history["lr_chart"], label="lr_chart")
        ax.plot(iters, history["iter_rejected"], label="iter_rejected")
        ax.grid(True, alpha=0.3)
        ax.set_title("Controls")
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Checkpoints.
    ckpt_selected = os.path.join(args.output_dir, f"{run_stem}_checkpoint.pt")
    s = selected["state"]  # type: ignore[index]
    save_ckpt(
        path=ckpt_selected,
        p_states=s["p_states"],  # type: ignore[index]
        c_states=s["c_states"],  # type: ignore[index]
        perm_state=s["perm_state"],  # type: ignore[index]
        label="selected_state",
        args=args,
        history=history,
        metrics=out,
        fallback_from=selected_fallback,
    )

    def pick_for_save(name: str, backup_order: Sequence[str]) -> Tuple[Dict[str, object], Optional[str]]:
        s0 = snapshots.get(name)
        if s0 is not None:
            return s0, None
        for b in backup_order:
            sb = snapshots.get(b)
            if sb is not None:
                return sb, b
        return selected, "selected"

    s_rel, fb_rel = pick_for_save("best_field", ["best_target", "best_score", "best_flux"])
    s_tgt, fb_tgt = pick_for_save("best_target", ["best_field", "best_score", "best_flux"])
    s_flux, fb_flux = pick_for_save("best_flux", ["best_target", "best_field", "best_score"])

    ckpt_best_rel = os.path.join(args.output_dir, f"{run_stem}_best_rel_l2.pt")
    ckpt_best_tgt = os.path.join(args.output_dir, f"{run_stem}_best_target.pt")
    ckpt_best_flux = os.path.join(args.output_dir, f"{run_stem}_best_flux.pt")

    save_ckpt(
        path=ckpt_best_rel,
        p_states=s_rel["state"]["p_states"],  # type: ignore[index]
        c_states=s_rel["state"]["c_states"],  # type: ignore[index]
        perm_state=s_rel["state"]["perm_state"],  # type: ignore[index]
        label="best_field",
        args=args,
        history=history,
        metrics=out,
        fallback_from=fb_rel,
    )
    save_ckpt(
        path=ckpt_best_tgt,
        p_states=s_tgt["state"]["p_states"],  # type: ignore[index]
        c_states=s_tgt["state"]["c_states"],  # type: ignore[index]
        perm_state=s_tgt["state"]["perm_state"],  # type: ignore[index]
        label="best_target",
        args=args,
        history=history,
        metrics=out,
        fallback_from=fb_tgt,
    )
    save_ckpt(
        path=ckpt_best_flux,
        p_states=s_flux["state"]["p_states"],  # type: ignore[index]
        c_states=s_flux["state"]["c_states"],  # type: ignore[index]
        perm_state=s_flux["state"]["perm_state"],  # type: ignore[index]
        label="best_flux",
        args=args,
        history=history,
        metrics=out,
        fallback_from=fb_flux,
    )

    out["checkpoint_paths"] = {
        "selected": ckpt_selected,
        "best_rel_l2": ckpt_best_rel,
        "best_target": ckpt_best_tgt,
        "best_flux": ckpt_best_flux,
    }
    out["field_outputs"] = {
        "solution_npz": solution_npz,
        "pressure_velocity_npz": pressure_velocity_npz,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Inverse Elder atlas run complete")
    print(f"  solution_npz: {solution_npz}")
    print(f"  p/v fields:   {pressure_velocity_npz}")
    print(f"  checkpoint:   {ckpt_selected}")
    print(f"  best_rel_l2:  {ckpt_best_rel}")
    print(f"  best_target:  {ckpt_best_tgt}")
    print(f"  best_flux:    {ckpt_best_flux}")
    print(f"  metrics:      {metrics_path}")
    print(f"  curves:       {fig_path}")
    print(f"  target_met:   {out['target_met']}")
    print(
        "  param errors: "
        f"k0_rel={out['k0_rel_error']:.3e}, eig_rel_max={out['eig_rel_error_max']:.3e}, "
        f"axis_deg_max={out['axis_angle_error_deg_max']:.3e}"
    )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse steady Elder-like flow on rabbit atlas charts")
    parser.add_argument("--config", default=None, help="Optional simple YAML key:value config")

    parser.add_argument("--atlas-data", required=False, default=None)
    parser.add_argument("--atlas-checkpoint", required=False, default=None)
    parser.add_argument("--atlas-meta", default=None)
    parser.add_argument("--output-dir", required=False, default="runs/rabbit_inverse_elder_atlas")
    parser.add_argument("--run-tag", default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--mps-fallback-cpu-on-error", dest="mps_fallback_cpu_on_error", action="store_true", default=True)
    parser.add_argument("--no-mps-fallback-cpu-on-error", dest="mps_fallback_cpu_on_error", action="store_false")

    parser.add_argument("--pinn-width", type=int, default=96)
    parser.add_argument("--pinn-depth", type=int, default=5)

    parser.add_argument("--teacher-iters", type=int, default=40)
    parser.add_argument("--teacher-lr", type=float, default=8e-4)
    parser.add_argument("--teacher-local-steps", type=int, default=1)
    parser.add_argument("--teacher-omega", type=float, default=0.9)
    parser.add_argument("--teacher-pde-batch", type=int, default=160)
    parser.add_argument("--teacher-bc-batch", type=int, default=128)
    parser.add_argument("--teacher-if-batch", type=int, default=128)
    parser.add_argument("--teacher-log-every", type=int, default=5)

    parser.add_argument("--max-schwarz-iters", type=int, default=120)
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--omega", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lr-perm", type=float, default=1.2e-4)

    parser.add_argument("--pde-batch", type=int, default=192)
    parser.add_argument("--bc-batch", type=int, default=160)
    parser.add_argument("--if-batch", type=int, default=160)
    parser.add_argument("--xi-noise-scale", type=float, default=0.08)

    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--det-floor", type=float, default=1e-6)
    parser.add_argument("--jac-kappa-max", type=float, default=1e3)
    parser.add_argument("--pde-clip-quantile", type=float, default=0.98)

    parser.add_argument("--Ra", type=float, default=6.0)
    parser.add_argument("--Dm", type=float, default=0.02)

    parser.add_argument("--w-flow", type=float, default=1.0)
    parser.add_argument("--w-trans", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=2.0)
    parser.add_argument("--w-if-val", type=float, default=1.0)
    parser.add_argument("--w-if-flux-teacher", type=float, default=0.8)
    parser.add_argument("--w-if-flux-start", type=float, default=8.0)
    parser.add_argument("--w-if-flux-end", type=float, default=80.0)
    parser.add_argument("--flux-ramp-iters", type=int, default=20)
    parser.add_argument("--adaptive-flux-weight", action="store_true")
    parser.add_argument("--adaptive-flux-up", type=float, default=1.02)
    parser.add_argument("--adaptive-flux-down", type=float, default=0.85)
    parser.add_argument("--adaptive-flux-field-thresh", type=float, default=0.14)

    parser.add_argument("--w-bc-p", type=float, default=1.0)
    parser.add_argument("--w-bc-c", type=float, default=1.0)
    parser.add_argument("--w-bc-flow", type=float, default=1.0)
    parser.add_argument("--w-bc-diff", type=float, default=1.0)

    parser.add_argument("--w-data", type=float, default=8.0)
    parser.add_argument("--w-data-p", type=float, default=1.0)
    parser.add_argument("--w-data-c", type=float, default=1.0)
    parser.add_argument("--w-data-u", type=float, default=0.0)

    parser.add_argument("--w-reg", type=float, default=0.5)
    parser.add_argument("--w-reg-k0", type=float, default=1.0)
    parser.add_argument("--w-reg-aniso", type=float, default=1e-3)
    parser.add_argument("--w-reg-quat", type=float, default=1e-3)
    parser.add_argument("--k0-prior", type=float, default=1.0)

    parser.add_argument("--interface-transmission-mode", choices=["penalty", "robin"], default="robin")
    parser.add_argument("--robin-lambda", type=float, default=12.0)
    parser.add_argument("--interface-normal-mode", choices=["mask_levelset", "seed"], default="mask_levelset")
    parser.add_argument("--interface-normal-eps", type=float, default=1e-6)
    parser.add_argument("--interface-normal-blend", type=float, default=0.15)
    parser.add_argument("--interface-normal-cache-batch", type=int, default=2048)
    parser.add_argument("--interface-detail-boost", type=float, default=1.6)

    parser.add_argument("--inlet-z-quantile", type=float, default=0.90)
    parser.add_argument("--top-z-quantile", type=float, default=0.78)
    parser.add_argument("--inlet-phi-center-deg", type=float, default=45.0)
    parser.add_argument("--inlet-phi-halfwidth-deg", type=float, default=40.0)
    parser.add_argument("--gauge-z-quantile", type=float, default=0.08)
    parser.add_argument("--gauge-phi-center-deg", type=float, default=-120.0)
    parser.add_argument("--gauge-phi-halfwidth-deg", type=float, default=18.0)

    parser.add_argument("--curvature-adaptive-sampling", action="store_true")
    parser.add_argument("--curvature-adaptive-interface-sampling", action="store_true")
    parser.add_argument("--detail-quantile", type=float, default=0.82)
    parser.add_argument("--detail-region-ratio", type=float, default=0.55)
    parser.add_argument("--curvature-sample-weight", type=float, default=3.0)
    parser.add_argument("--detail-normal-weight", type=float, default=0.75)
    parser.add_argument("--detail-extremity-weight", type=float, default=0.20)
    parser.add_argument("--detail-overlap-weight", type=float, default=0.05)
    parser.add_argument("--detail-chart-topk", type=int, default=4)
    parser.add_argument("--detail-chart-step-boost", type=float, default=1.6)
    parser.add_argument("--detail-chart-batch-boost", type=float, default=1.35)

    parser.add_argument("--n-obs-per-chart", type=int, default=256)

    parser.add_argument("--k0-true", type=float, default=1.0)
    parser.add_argument("--k0-init", type=float, default=0.97)
    parser.add_argument("--k0-min", type=float, default=1e-4)
    parser.add_argument("--lam1-true", type=float, default=1.80)
    parser.add_argument("--lam2-true", type=float, default=1.00)
    parser.add_argument("--lam3-true", type=float, default=0.5556)

    parser.add_argument("--a1-init", type=float, default=0.554)
    parser.add_argument("--a2-init", type=float, default=0.02956)
    parser.add_argument("--true-euler-x-deg", type=float, default=22.0)
    parser.add_argument("--true-euler-y-deg", type=float, default=-17.0)
    parser.add_argument("--true-euler-z-deg", type=float, default=31.0)
    parser.add_argument("--init-euler-x-deg", type=float, default=20.0)
    parser.add_argument("--init-euler-y-deg", type=float, default=-16.0)
    parser.add_argument("--init-euler-z-deg", type=float, default=29.0)

    parser.add_argument("--acc-k0-rel", type=float, default=0.05)
    parser.add_argument("--acc-eig-rel", type=float, default=0.10)
    parser.add_argument("--acc-axis-deg", type=float, default=12.0)

    parser.add_argument("--eval-fixed-cache", action="store_true")
    parser.add_argument("--eval-cache-seed", type=int, default=1234)
    parser.add_argument("--eval-cache-per-chart", type=int, default=128)
    parser.add_argument("--eval-cache-per-overlap", type=int, default=96)
    parser.add_argument("--eval-bc-samples-per-chart", type=int, default=64)
    parser.add_argument("--eval-pde-samples-per-chart", type=int, default=96)
    parser.add_argument("--eval-if-samples", type=int, default=64)

    parser.add_argument("--iter-accept-field-rel", type=float, default=0.155)
    parser.add_argument("--iter-reject-lr-decay", type=float, default=0.8)
    parser.add_argument("--guard-field-rel", type=float, default=0.151)
    parser.add_argument("--guard-patience", type=int, default=4)
    parser.add_argument("--guard-lr-decay", type=float, default=0.5)
    parser.add_argument("--target-field-rel", type=float, default=0.15)

    parser.add_argument("--plateau-patience", type=int, default=16)
    parser.add_argument("--plateau-tol", type=float, default=5e-5)
    parser.add_argument("--checkpoint-policy", choices=["last", "best_score", "best_field", "best_target", "best_flux"], default="best_target")

    parser.add_argument("--warm-start-teacher", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)

    argv_flags = set()
    for tok in sys.argv[1:]:
        if not tok.startswith("--"):
            continue
        name = tok[2:]
        if "=" in name:
            name = name.split("=", 1)[0]
        argv_flags.add(name.replace("-", "_"))

    args = parser.parse_args()

    if args.config is not None:
        cfg = parse_simple_yaml(args.config)
        for k, v in cfg.items():
            attr = k.replace("-", "_")
            if not hasattr(args, attr):
                continue
            if attr in argv_flags:
                continue
            cur = getattr(args, attr)
            if isinstance(cur, bool):
                val = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(v)
            elif isinstance(cur, float):
                val = float(v)
            else:
                val = v
            setattr(args, attr, val)

    if args.no_amp:
        args.amp = False

    if args.atlas_data is None or args.atlas_checkpoint is None:
        raise RuntimeError("--atlas-data and --atlas-checkpoint are required (or provided via --config).")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    try:
        train(args)
    except Exception as exc:
        err_msg = str(exc)
        is_mps_error = isinstance(exc, torch.AcceleratorError) or ("mps" in err_msg.lower())
        if not (getattr(args, "mps_fallback_cpu_on_error", True) and is_mps_error):
            raise
        if args.device not in ("mps", "auto"):
            raise
        print(
            "MPS runtime failure detected; restarting this run on CPU with AMP disabled.\n"
            f"Original error: {type(exc).__name__}: {exc}"
        )
        args.device = "cpu"
        args.amp = False
        # Reset seed so fallback run is reproducible.
        set_seed(args.seed)
        train(args)


if __name__ == "__main__":
    main()
