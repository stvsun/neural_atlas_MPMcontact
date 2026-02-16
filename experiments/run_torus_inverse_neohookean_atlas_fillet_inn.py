#!/usr/bin/env python3
"""
Inverse Neo-Hookean on 3D torus using 8 filleted-cube charts and mapped Cartesian PDE.

This is a full field inverse PINN variant:
- One displacement PINN per chart u_i(z), z in filleted-cube local coordinates.
- Shared invertible neural map z->eta learned before PDE training.
- Analytic eta->torus chart map with 20% overlap in major-angle coordinate.
- Joint training with PDE, BC, traction-data, and interface continuity penalties.
- Unknowns: global mu and K (both via softplus).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float32)
_FILLET_P = 6.0
_FILLET_INN_MAP: Optional["InvertibleFilletMap"] = None


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
            print("Requested CUDA but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested MPS but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            print("Requested float64 on MPS; using float32 instead.")
            return torch.float32
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def scalar_from_tensor(t: torch.Tensor, device: torch.device) -> float:
    sync_device(device)
    t_cpu = t.detach().to(device=torch.device("cpu"))
    if t_cpu.numel() == 0:
        raise ValueError("Cannot extract scalar from empty tensor")
    return float(t_cpu.reshape(-1)[0].item())


def sanitize_tensor(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    fill_t = torch.as_tensor(fill, device=x.device, dtype=x.dtype)
    return torch.nan_to_num(x, nan=fill_t, posinf=fill_t, neginf=fill_t)


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eps_t = torch.as_tensor(eps, device=x.device, dtype=x.dtype)
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps_t)


def rand_uniform_cpu(
    n: int,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r = torch.rand((n,), device=torch.device("cpu"), dtype=torch.float64)
    out = low + (high - low) * r
    return out.to(device=device, dtype=dtype)


def rand_int_cpu(
    n: int,
    low: int,
    high: int,
    device: torch.device,
) -> torch.Tensor:
    r = torch.randint(low, high, (n,), device=torch.device("cpu"), dtype=torch.int64)
    return r.to(device=device)


def parse_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in {"1", "true", "yes", "on", "y"}:
        return True
    if v in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {s}")


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


class LocalVectorPINN(torch.nn.Module):
    def __init__(self, width: int = 96, depth: int = 5):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        return self.net(eta)


class AdditiveCoupling(torch.nn.Module):
    def __init__(self, mask: Sequence[float], width: int, depth: int):
        super().__init__()
        if len(mask) != 3:
            raise ValueError("mask must have length 3")
        m = torch.as_tensor(mask, dtype=torch.float32).reshape(1, 3)
        self.register_buffer("mask", m)
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)

    def _masked_shift(self, x_masked: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(device=x_masked.device, dtype=x_masked.dtype)
        return self.net(x_masked) * (1.0 - m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(device=x.device, dtype=x.dtype)
        x_m = x * m
        shift = self._masked_shift(x_m)
        return x_m + (1.0 - m) * (x + shift)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(device=y.device, dtype=y.dtype)
        y_m = y * m
        shift = self._masked_shift(y_m)
        return y_m + (1.0 - m) * (y - shift)


class InvertibleFilletMap(torch.nn.Module):
    def __init__(self, width: int = 64, depth: int = 2, n_layers: int = 8):
        super().__init__()
        base_masks = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        layers = []
        for i in range(max(1, int(n_layers))):
            layers.append(AdditiveCoupling(mask=base_masks[i % len(base_masks)], width=width, depth=depth))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        x = eta
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def set_fillet_context(fillet_p: float, inn_map: Optional[InvertibleFilletMap]) -> None:
    global _FILLET_P
    global _FILLET_INN_MAP
    _FILLET_P = float(fillet_p)
    _FILLET_INN_MAP = inn_map


def cube_to_filleted_cube(eta: torch.Tensor, fillet_p: float) -> torch.Tensor:
    # Superellipsoid-like rounded cube (filleted cube proxy): ||z||_p <= 1.
    # Closed-form bijection from cube to this rounded domain.
    p_t = torch.as_tensor(float(fillet_p), device=eta.device, dtype=eta.dtype)
    eps = torch.as_tensor(1e-12, device=eta.device, dtype=eta.dtype)
    abs_eta = torch.abs(eta)
    n_inf = torch.max(abs_eta, dim=1, keepdim=True).values
    n_p = torch.sum(abs_eta**p_t, dim=1, keepdim=True) ** (1.0 / p_t)
    scale = torch.where(n_inf > eps, n_inf / torch.clamp(n_p, min=eps), torch.ones_like(n_inf))
    return eta * scale


def filleted_cube_to_cube(z: torch.Tensor, fillet_p: float) -> torch.Tensor:
    # Inverse of cube_to_filleted_cube.
    p_t = torch.as_tensor(float(fillet_p), device=z.device, dtype=z.dtype)
    eps = torch.as_tensor(1e-12, device=z.device, dtype=z.dtype)
    abs_z = torch.abs(z)
    n_inf = torch.max(abs_z, dim=1, keepdim=True).values
    n_p = torch.sum(abs_z**p_t, dim=1, keepdim=True) ** (1.0 / p_t)
    scale = torch.where(n_inf > eps, n_p / torch.clamp(n_inf, min=eps), torch.ones_like(n_inf))
    return z * scale


def train_invertible_fillet_map(
    device: torch.device,
    dtype: torch.dtype,
    fillet_p: float,
    width: int,
    depth: int,
    n_layers: int,
    steps: int,
    batch_size: int,
    lr: float,
    cycle_weight: float,
    range_weight: float,
    eval_n: int,
    seed: int,
) -> Tuple[InvertibleFilletMap, Dict[str, float]]:
    set_seed(seed)
    inn = InvertibleFilletMap(width=width, depth=depth, n_layers=n_layers).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(inn.parameters(), lr=lr)
    last_loss = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        eta = sample_eta_cube_raw(batch_size, device=device, dtype=dtype)
        z = cube_to_filleted_cube(eta, fillet_p=fillet_p)
        eta_pred = inn(z)
        z_back = inn.inverse(eta_pred)
        eta_back = inn(inn.inverse(eta))
        loss_fit = torch.mean((eta_pred - eta) ** 2)
        loss_cycle = torch.mean((z_back - z) ** 2) + torch.mean((eta_back - eta) ** 2)
        loss_range = torch.mean(torch.nn.functional.relu(torch.abs(eta_pred) - 1.02) ** 2)
        loss = loss_fit + cycle_weight * loss_cycle + range_weight * loss_range
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=5.0)
        opt.step()
        last_loss = float(loss.detach().item())
        if step % max(1, steps // 5) == 0 or step == 1 or step == steps:
            print(
                f"[FilletINN] step={step}/{steps} loss={last_loss:.3e} "
                f"fit={float(loss_fit.detach().item()):.3e} cycle={float(loss_cycle.detach().item()):.3e}"
            )

    with torch.no_grad():
        eta_eval = sample_eta_cube_raw(eval_n, device=device, dtype=dtype)
        z_eval = cube_to_filleted_cube(eta_eval, fillet_p=fillet_p)
        eta_hat = inn(z_eval)
        z_hat = inn.inverse(eta_eval)
        fit_rmse = float(torch.sqrt(torch.mean((eta_hat - eta_eval) ** 2)).item())
        fit_max = float(torch.max(torch.abs(eta_hat - eta_eval)).item())
        inv_rmse = float(torch.sqrt(torch.mean((z_hat - z_eval) ** 2)).item())
        z_rec = cube_to_filleted_cube(eta_hat, fillet_p=fillet_p)
        geom_rmse = float(torch.sqrt(torch.mean((z_rec - z_eval) ** 2)).item())
    for p in inn.parameters():
        p.requires_grad_(False)
    inn.eval()
    stats = {
        "train_loss_last": last_loss,
        "fit_rmse": fit_rmse,
        "fit_max_abs": fit_max,
        "inv_rmse": inv_rmse,
        "geom_rmse": geom_rmse,
        "steps": float(steps),
    }
    return inn, stats


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


def square_to_disk_concentric(s: torch.Tensor) -> torch.Tensor:
    # Shirley-Chiu concentric map from [-1,1]^2 to unit disk.
    sx = s[:, 0]
    sy = s[:, 1]
    absx = torch.abs(sx)
    absy = torch.abs(sy)
    eps = torch.as_tensor(1e-12, device=s.device, dtype=s.dtype)
    pi_t = torch.as_tensor(math.pi, device=s.device, dtype=s.dtype)

    cond = absx > absy
    r = torch.where(cond, sx, sy)
    theta = torch.where(
        cond,
        (pi_t / 4.0) * (sy / torch.where(torch.abs(sx) > eps, sx, torch.sign(sx) * eps + eps)),
        (pi_t / 2.0) - (pi_t / 4.0) * (sx / torch.where(torch.abs(sy) > eps, sy, torch.sign(sy) * eps + eps)),
    )
    qx = r * torch.cos(theta)
    qy = r * torch.sin(theta)

    zero = (absx <= eps) & (absy <= eps)
    qx = torch.where(zero, torch.zeros_like(qx), qx)
    qy = torch.where(zero, torch.zeros_like(qy), qy)
    return torch.stack([qx, qy], dim=1)


def disk_to_square_concentric_newton(q: torch.Tensor, n_iters: int = 8, step: float = 0.9) -> torch.Tensor:
    # Numerical inverse utility for concentric map; used only where inverse chart lookup is needed.
    s = torch.clamp(q.clone(), min=-1.0, max=1.0)
    eye = torch.eye(2, device=q.device, dtype=q.dtype).unsqueeze(0)
    eps = torch.as_tensor(1e-8, device=q.device, dtype=q.dtype)
    step_t = torch.as_tensor(step, device=q.device, dtype=q.dtype)
    for _ in range(n_iters):
        s_var = s.clone().detach().requires_grad_(True)
        q_pred = square_to_disk_concentric(s_var)
        r = q_pred - q

        g1 = torch.autograd.grad(
            q_pred[:, 0],
            s_var,
            grad_outputs=torch.ones_like(q_pred[:, 0]),
            create_graph=False,
            retain_graph=True,
        )[0]
        g2 = torch.autograd.grad(
            q_pred[:, 1],
            s_var,
            grad_outputs=torch.ones_like(q_pred[:, 1]),
            create_graph=False,
            retain_graph=False,
        )[0]
        J = torch.stack([g1, g2], dim=1)  # [N,2,2]
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det_safe = torch.where(
            det >= 0.0,
            torch.clamp(det, min=eps),
            torch.clamp(det, max=-eps),
        )
        invJ = torch.zeros_like(J)
        invJ[:, 0, 0] = J[:, 1, 1] / det_safe
        invJ[:, 0, 1] = -J[:, 0, 1] / det_safe
        invJ[:, 1, 0] = -J[:, 1, 0] / det_safe
        invJ[:, 1, 1] = J[:, 0, 0] / det_safe
        ds = torch.bmm(invJ, r.unsqueeze(-1)).squeeze(-1)
        s = torch.clamp(s - step_t * ds, min=-1.0, max=1.0)
    return s


def chart_centers(n_charts: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    two_pi = torch.as_tensor(2.0 * math.pi, device=device, dtype=dtype)
    return torch.linspace(0.0, float(two_pi.item()), steps=n_charts + 1, device=device, dtype=dtype)[:-1]


def chart_eta_from_global(phi: torch.Tensor, s: torch.Tensor, center: torch.Tensor, h_phi: torch.Tensor, eta1_clip: float) -> torch.Tensor:
    eta1 = wrap_to_pi(phi - center) / h_phi
    if eta1_clip > 0:
        c = torch.as_tensor(float(eta1_clip), device=eta1.device, dtype=eta1.dtype)
        eta1 = torch.clamp(eta1, min=-c, max=c)
    eta = torch.stack([eta1, s[:, 0], s[:, 1]], dim=1)
    if _FILLET_INN_MAP is not None:
        return _FILLET_INN_MAP.inverse(eta)
    return cube_to_filleted_cube(eta, fillet_p=_FILLET_P)


def torus_xyz_from_phi_q(phi: torch.Tensor, q: torch.Tensor, major_radius: float, minor_radius: float) -> torch.Tensor:
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    q1 = q[:, 0]
    q2 = q[:, 1]
    rr = torch.as_tensor(major_radius, device=phi.device, dtype=phi.dtype) + torch.as_tensor(minor_radius, device=phi.device, dtype=phi.dtype) * q1
    x = rr * cphi
    y = rr * sphi
    z = torch.as_tensor(minor_radius, device=phi.device, dtype=phi.dtype) * q2
    return torch.stack([x, y, z], dim=1)


def torus_boundary_normals(phi: torch.Tensor, q_boundary: torch.Tensor) -> torch.Tensor:
    qn = normalize_rows(q_boundary)
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    q1 = qn[:, 0]
    q2 = qn[:, 1]
    n = torch.stack([q1 * cphi, q1 * sphi, q2], dim=1)
    return normalize_rows(n)


def chart_map_eta_to_x(
    eta: torch.Tensor,
    center: torch.Tensor,
    h_phi: torch.Tensor,
    major_radius: float,
    minor_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi = center + h_phi * eta[:, 0]
    s = eta[:, 1:3]
    q = square_to_disk_concentric(s)
    x = torus_xyz_from_phi_q(phi=phi, q=q, major_radius=major_radius, minor_radius=minor_radius)
    return x, q, phi


def global_map_from_phi_s(phi: torch.Tensor, s: torch.Tensor, major_radius: float, minor_radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    q = square_to_disk_concentric(s)
    x = torus_xyz_from_phi_q(phi=phi, q=q, major_radius=major_radius, minor_radius=minor_radius)
    return x, q


def global_boundary_from_phi_s(
    phi: torch.Tensor,
    s_boundary: torch.Tensor,
    major_radius: float,
    minor_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = square_to_disk_concentric(s_boundary)
    qn = normalize_rows(q)
    x = torus_xyz_from_phi_q(phi=phi, q=qn, major_radius=major_radius, minor_radius=minor_radius)
    n = torus_boundary_normals(phi=phi, q_boundary=qn)
    return x, n, qn


def chart_map_and_jacobian(
    eta_in: torch.Tensor,
    center: torch.Tensor,
    h_phi: torch.Tensor,
    major_radius: float,
    minor_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local = eta_in.clone().detach().requires_grad_(True)
    if _FILLET_INN_MAP is None:
        eta = local
    else:
        eta = _FILLET_INN_MAP(local)
    x, q, phi = chart_map_eta_to_x(
        eta=eta,
        center=center,
        h_phi=h_phi,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    grads = []
    for i in range(3):
        gi = torch.autograd.grad(
            x[:, i],
            local,
            grad_outputs=torch.ones_like(x[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(gi)
    jac = torch.stack(grads, dim=1)
    return x, local, jac, q, phi


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


def divergence_mapped(mapped_flux: torch.Tensor, eta_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
    # mapped_flux: [N,3,3], component k and eta-dir j.
    div = torch.zeros((mapped_flux.shape[0], 3), device=eta_var.device, dtype=eta_var.dtype)
    for k in range(3):
        comp = torch.zeros((mapped_flux.shape[0], 1), device=eta_var.device, dtype=eta_var.dtype)
        for j in range(3):
            dcomp = torch.autograd.grad(
                mapped_flux[:, k, j],
                eta_var,
                grad_outputs=torch.ones_like(mapped_flux[:, k, j]),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, j : j + 1]
            comp = comp + dcomp
        div[:, k : k + 1] = comp
    return div


def neo_hookean_p(F: torch.Tensor, mu: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    det_f = det_3x3(F)
    det_floor = torch.as_tensor(1e-8, device=det_f.device, dtype=det_f.dtype)
    det_safe = torch.clamp(det_f, min=det_floor)
    finv_t = torch.linalg.inv(F).transpose(1, 2)
    log_j = torch.log(det_safe).unsqueeze(-1).unsqueeze(-1)
    return mu * (F - finv_t) + K * log_j * finv_t


def torsion_window(phi: torch.Tensor, phi_center: float, phi_halfwidth: float) -> torch.Tensor:
    phi_c = torch.as_tensor(phi_center, device=phi.device, dtype=phi.dtype)
    phi_hw = torch.as_tensor(phi_halfwidth, device=phi.device, dtype=phi.dtype)
    eps = torch.as_tensor(1e-8, device=phi.device, dtype=phi.dtype)
    half = torch.as_tensor(0.5, device=phi.device, dtype=phi.dtype)
    one = torch.as_tensor(1.0, device=phi.device, dtype=phi.dtype)
    pi_t = torch.as_tensor(math.pi, device=phi.device, dtype=phi.dtype)
    d = wrap_to_pi(phi - phi_c)
    m = (torch.abs(d) <= phi_hw).to(phi.dtype)
    w = half * (one + torch.cos(pi_t * d / torch.clamp(phi_hw, min=eps)))
    return w * m


def manufactured_displacement(
    x: torch.Tensor,
    major_radius: float,
    minor_radius: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    phi = torch.atan2(x2, x1)
    w = torsion_window(phi, phi_center=phi_center, phi_halfwidth=phi_halfwidth)
    rr = torch.sqrt(torch.clamp(x1 * x1 + x2 * x2, min=1e-12))
    q2 = x3 / torch.as_tensor(minor_radius, device=x.device, dtype=x.dtype)
    rot = torch.stack([-x2, x1, torch.zeros_like(x3)], dim=1)
    one = torch.as_tensor(1.0, device=x.device, dtype=x.dtype)
    tau_t = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
    z_s = torch.as_tensor(z_scale, device=x.device, dtype=x.dtype)
    # Mild centerline scaling keeps field smooth near torus interior.
    center_fac = rr / torch.as_tensor(major_radius + minor_radius, device=x.device, dtype=x.dtype)
    amp = tau_t * (one + z_s * q2) * center_fac
    return amp.unsqueeze(1) * w.unsqueeze(1) * rot


def manufactured_body_force(
    x: torch.Tensor,
    mu_true: float,
    K_true: float,
    major_radius: float,
    minor_radius: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
) -> torch.Tensor:
    with torch.enable_grad():
        x_var = x.clone().detach().requires_grad_(True)
        u_true = manufactured_displacement(
            x=x_var,
            major_radius=major_radius,
            minor_radius=minor_radius,
            tau=tau,
            phi_center=phi_center,
            phi_halfwidth=phi_halfwidth,
            z_scale=z_scale,
        )
        grad_u = gradient_tensor(u_true, x_var, create_graph=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        F = eye + grad_u
        mu_t = torch.as_tensor(mu_true, device=x.device, dtype=x.dtype)
        K_t = torch.as_tensor(K_true, device=x.device, dtype=x.dtype)
        P = neo_hookean_p(F, mu_t, K_t)

        div_P = []
        for i in range(3):
            comp = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            for j in range(3):
                dP_ij = torch.autograd.grad(
                    P[:, i, j],
                    x_var,
                    grad_outputs=torch.ones_like(P[:, i, j]),
                    create_graph=False,
                    retain_graph=True,
                )[0][:, j : j + 1]
                comp = comp + dP_ij
            div_P.append(comp)
        div_P = torch.cat(div_P, dim=1)
        b = -div_P
    return b.detach()


def manufactured_traction(
    x: torch.Tensor,
    n_phys: torch.Tensor,
    mu_true: float,
    K_true: float,
    major_radius: float,
    minor_radius: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
) -> torch.Tensor:
    with torch.enable_grad():
        x_var = x.clone().detach().requires_grad_(True)
        u_true = manufactured_displacement(
            x=x_var,
            major_radius=major_radius,
            minor_radius=minor_radius,
            tau=tau,
            phi_center=phi_center,
            phi_halfwidth=phi_halfwidth,
            z_scale=z_scale,
        )
        grad_u = gradient_tensor(u_true, x_var, create_graph=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        F = eye + grad_u
        mu_t = torch.as_tensor(mu_true, device=x.device, dtype=x.dtype)
        K_t = torch.as_tensor(K_true, device=x.device, dtype=x.dtype)
        P = neo_hookean_p(F, mu_t, K_t)
        t = torch.bmm(P, n_phys.unsqueeze(-1)).squeeze(-1)
    return t.detach()


def in_phi_patch(phi: torch.Tensor, center: float, halfwidth: float) -> torch.Tensor:
    c = torch.as_tensor(center, device=phi.device, dtype=phi.dtype)
    hw = torch.as_tensor(halfwidth, device=phi.device, dtype=phi.dtype)
    return torch.abs(wrap_to_pi(phi - c)) <= hw


def sample_phi_interval(
    n: int,
    center: float,
    halfwidth: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return rand_uniform_cpu(n=n, low=center - halfwidth, high=center + halfwidth, device=device, dtype=dtype)


def sample_phi_free(
    n: int,
    load_center: float,
    load_halfwidth: float,
    anchor_center: float,
    anchor_halfwidth: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = []
    two_pi = 2.0 * math.pi
    need = n
    while need > 0:
        m = max(need * 2, 512)
        cand = rand_uniform_cpu(m, 0.0, two_pi, device=device, dtype=dtype)
        bad = in_phi_patch(cand, load_center, load_halfwidth) | in_phi_patch(cand, anchor_center, anchor_halfwidth)
        keep = cand[~bad]
        if keep.numel() > 0:
            take = min(need, int(keep.numel()))
            out.append(keep[:take])
            need -= take
    return torch.cat(out, dim=0)


def sample_square(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sx = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    sy = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    return torch.stack([sx, sy], dim=1)


def sample_eta_cube_raw(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    e1 = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    e2 = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    e3 = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    return torch.stack([e1, e2, e3], dim=1)


def sample_eta_cube(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Local coordinates are sampled in filleted-cube space.
    eta = sample_eta_cube_raw(n=n, device=device, dtype=dtype)
    return cube_to_filleted_cube(eta, fillet_p=_FILLET_P)


def sample_square_boundary(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    edge = rand_int_cpu(n=n, low=0, high=4, device=device)
    t = rand_uniform_cpu(n, -1.0, 1.0, device=device, dtype=dtype)
    sx = torch.zeros((n,), device=device, dtype=dtype)
    sy = torch.zeros((n,), device=device, dtype=dtype)
    sx = torch.where(edge == 0, -torch.ones_like(sx), sx)
    sy = torch.where(edge == 0, t, sy)
    sx = torch.where(edge == 1, torch.ones_like(sx), sx)
    sy = torch.where(edge == 1, t, sy)
    sx = torch.where(edge == 2, t, sx)
    sy = torch.where(edge == 2, -torch.ones_like(sy), sy)
    sx = torch.where(edge == 3, t, sx)
    sy = torch.where(edge == 3, torch.ones_like(sy), sy)
    return torch.stack([sx, sy], dim=1)


def build_chart_weights(phi: torch.Tensor, centers: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    d = wrap_to_pi(phi.unsqueeze(1) - centers.unsqueeze(0))
    logits = -torch.as_tensor(0.5, device=phi.device, dtype=phi.dtype) * (d / sigma) ** 2
    return torch.softmax(logits, dim=1)


def neighbor_pairs_ring(n_charts: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n_charts) for i in range(n_charts)]


def interface_pair_mid_and_halfwidth(
    center_i: torch.Tensor,
    center_j: torch.Tensor,
    h_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = wrap_to_pi(center_j - center_i)
    mid = center_i + 0.5 * d
    mid = torch.remainder(mid, torch.as_tensor(2.0 * math.pi, device=mid.device, dtype=mid.dtype))
    half = h_phi - 0.5 * torch.abs(d)
    half = torch.clamp(half, min=torch.as_tensor(1e-4, device=half.device, dtype=half.dtype))
    return mid, half


def softplus_inv(y: float) -> float:
    y = max(y, 1e-8)
    return math.log(math.expm1(y))


def clamp_init(val: float, target: float, frac: float, name: str) -> float:
    lo = target * (1.0 - frac)
    hi = target * (1.0 + frac)
    if val < lo or val > hi:
        clamped = min(max(val, lo), hi)
        print(
            f"[Init] {name}={val:.6f} outside +/-{100.0*frac:.1f}% of target {target:.6f}; "
            f"clamped to {clamped:.6f}"
        )
        return clamped
    return val


def compute_vm_stress(P: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    det_f = det_3x3(F)
    det_safe = torch.clamp(det_f, min=torch.as_tensor(1e-8, device=det_f.device, dtype=det_f.dtype))
    sigma = torch.bmm(P, F.transpose(1, 2)) / det_safe.unsqueeze(-1).unsqueeze(-1)
    tr = torch.einsum("bii->b", sigma).unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
    s = sigma - (tr / 3.0) * eye
    vm = torch.sqrt(torch.clamp(1.5 * torch.sum(s * s, dim=(1, 2)), min=torch.as_tensor(0.0, device=sigma.device, dtype=sigma.dtype)))
    return vm


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

    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        raise ValueError(f"All points are invalid for VTU export: {path}")
    dropped = n0 - valid_count
    if dropped > 0:
        print(f"[VTU] Dropping {dropped}/{n0} non-finite rows for {os.path.basename(path)}")

    points = points[valid]
    if np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float32, copy=False)
    n = int(points.shape[0])

    for name, arr in cleaned.items():
        a = arr[valid]
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32, copy=False)
        cleaned[name] = a

    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)

    def vtk_type(arr: np.ndarray) -> str:
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            if a.dtype.itemsize <= 4:
                return "Int32"
            return "Int64"
        if a.dtype == np.float32:
            return "Float32"
        return "Float64"

    def write_ascii_dataarray_lines(fh, arr: np.ndarray, indent: str = "          ", vals_per_line: int = 24) -> None:
        flat = np.asarray(arr).reshape(-1)
        nvals = int(flat.shape[0])
        for s in range(0, nvals, vals_per_line):
            e = min(nvals, s + vals_per_line)
            fh.write(indent + " ".join(map(str, flat[s:e])) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write(f'        <DataArray type="{vtk_type(points)}" NumberOfComponents="3" format="ascii">\n')
        write_ascii_dataarray_lines(f, points)
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        write_ascii_dataarray_lines(f, connectivity)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        write_ascii_dataarray_lines(f, offsets)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        write_ascii_dataarray_lines(f, cell_types)
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in cleaned.items():
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = int(arr.shape[1])
            else:
                raise ValueError(f"Unsupported point_data shape for '{name}': {arr.shape}")
            f.write(
                f'        <DataArray type="{vtk_type(arr)}" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            write_ascii_dataarray_lines(f, arr)
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def build_run_stem(run_tag: str) -> str:
    run_tag = str(run_tag).strip()
    if run_tag:
        return f"torus_inverse_neohookean_atlas_fillet_inn_{run_tag}"
    return "torus_inverse_neohookean_atlas_fillet_inn"


def evaluate_fixed_cache(
    u_nets: Sequence[LocalVectorPINN],
    centers: torch.Tensor,
    h_phi: torch.Tensor,
    args: argparse.Namespace,
    mu_est: torch.Tensor,
    K_est: torch.Tensor,
    eval_cache: Dict[str, object],
) -> Dict[str, float]:
    device = centers.device
    dtype = centers.dtype
    n_charts = len(u_nets)

    eq_vals: List[float] = []
    valid_ratios: List[float] = []
    detJ_min: List[float] = []
    detF_min: List[float] = []
    if_val_vals: List[float] = []
    if_grad_vals: List[float] = []

    for i in range(n_charts):
        eta = eval_cache["eta_int"][i]  # type: ignore[index]
        x, eta_var, jac, _, _ = chart_map_and_jacobian(
            eta_in=eta,
            center=centers[i],
            h_phi=h_phi,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        inv_j, det_abs, _, valid = stabilized_jacobian_ops(
            jac=jac,
            sigma_floor=args.sigma_floor,
            det_floor=args.detJ_floor,
            jac_kappa_max=args.jac_kappa_max,
        )
        u = u_nets[i](eta_var)
        grad_u_eta = gradient_tensor(u, eta_var, create_graph=False)
        grad_u_x = torch.bmm(grad_u_eta, inv_j)
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        F = eye + grad_u_x
        P = neo_hookean_p(F, mu_est, K_est)
        mapped_flux = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(P, inv_j.transpose(1, 2))
        div_flux = divergence_mapped(mapped_flux, eta_var, create_graph=False)
        b_true = manufactured_body_force(
            x=x.detach(),
            mu_true=args.mu_true,
            K_true=args.K_true,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        res = div_flux + det_abs.unsqueeze(-1) * b_true
        if bool(torch.any(valid).item()):
            eq = torch.mean(res[valid] ** 2)
        else:
            eq = torch.mean(res**2)
        eq_vals.append(float(eq.detach().item()))
        valid_ratios.append(float(valid.to(dtype).mean().detach().item()))
        detJ_min.append(float(torch.min(det_abs).detach().item()))
        detF_min.append(float(torch.min(det_3x3(F)).detach().item()))

    # Interface cache.
    for (i, j), payload in eval_cache["if_pairs"].items():  # type: ignore[union-attr]
        phi = payload["phi"]
        s = payload["s"]
        eta_i = chart_eta_from_global(phi, s, centers[i], h_phi, args.eta1_clip_eval)
        eta_j = chart_eta_from_global(phi, s, centers[j], h_phi, args.eta1_clip_eval)

        _, ei, ji, _, _ = chart_map_and_jacobian(
            eta_in=eta_i,
            center=centers[i],
            h_phi=h_phi,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        _, ej, jj, _, _ = chart_map_and_jacobian(
            eta_in=eta_j,
            center=centers[j],
            h_phi=h_phi,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        inv_i, _, _, _ = stabilized_jacobian_ops(ji, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
        inv_j, _, _, _ = stabilized_jacobian_ops(jj, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)

        ui = u_nets[i](ei)
        uj = u_nets[j](ej)
        gu_i = torch.bmm(gradient_tensor(ui, ei, create_graph=False), inv_i)
        gu_j = torch.bmm(gradient_tensor(uj, ej, create_graph=False), inv_j)
        if_val_vals.append(float(torch.mean((ui - uj) ** 2).detach().item()))
        if_grad_vals.append(float(torch.mean((gu_i - gu_j) ** 2).detach().item()))

    # Traction eval on fixed observation subset.
    obs = eval_cache["obs_eval"]
    phi_o = obs["phi"]
    s_o = obs["s"]
    n_o = obs["n"]
    t_true = obs["t_true"]
    w_o = build_chart_weights(
        phi=phi_o,
        centers=centers,
        sigma=torch.as_tensor(args.chart_sigma_scale * (2.0 * math.pi / args.n_charts), device=device, dtype=dtype),
    )
    t_pred_all = []
    for i in range(n_charts):
        eta_i = chart_eta_from_global(phi_o, s_o, centers[i], h_phi, args.eta1_clip_eval)
        _, ev, jv, _, _ = chart_map_and_jacobian(
            eta_in=eta_i,
            center=centers[i],
            h_phi=h_phi,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        inv_j, _, _, _ = stabilized_jacobian_ops(jv, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
        u_i = u_nets[i](ev)
        grad_u = torch.bmm(gradient_tensor(u_i, ev, create_graph=False), inv_j)
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        F = eye + grad_u
        P = neo_hookean_p(F, mu_est, K_est)
        t_i = torch.bmm(P, n_o.unsqueeze(-1)).squeeze(-1)
        t_pred_all.append(t_i)
    t_stack = torch.stack(t_pred_all, dim=1)
    t_pred = torch.sum(w_o.unsqueeze(-1) * t_stack, dim=1)
    traction_rel_l2_t = torch.sqrt(
        torch.mean((t_pred - t_true) ** 2)
        / torch.clamp(torch.mean(t_true**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
    )

    metrics = {
        "eq_eval": float(np.mean(eq_vals)) if eq_vals else float("nan"),
        "if_val_eval": float(np.mean(if_val_vals)) if if_val_vals else float("nan"),
        "if_grad_eval": float(np.mean(if_grad_vals)) if if_grad_vals else float("nan"),
        "map_valid_ratio_eval": float(np.mean(valid_ratios)) if valid_ratios else float("nan"),
        "detJ_min_eval": float(np.min(detJ_min)) if detJ_min else float("nan"),
        "detF_min_eval": float(np.min(detF_min)) if detF_min else float("nan"),
        "traction_rel_l2_eval": float(traction_rel_l2_t.detach().item()),
    }
    return metrics


def make_plot(history: Dict[str, List[float]], out_path: str) -> None:
    ep = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].semilogy(ep, np.maximum(history["total"], 1e-16), label="total")
    axes[0, 0].semilogy(ep, np.maximum(history["eq"], 1e-16), label="eq")
    axes[0, 0].semilogy(ep, np.maximum(history["bc_load"], 1e-16), label="bc_load")
    axes[0, 0].semilogy(ep, np.maximum(history["bc_anchor"], 1e-16), label="bc_anchor")
    axes[0, 0].semilogy(ep, np.maximum(history["bc_free"], 1e-16), label="bc_free")
    axes[0, 0].set_title("Loss Components")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(ep, np.maximum(history["data_trac"], 1e-16), label="data_trac")
    axes[0, 1].semilogy(ep, np.maximum(history["if_val"], 1e-16), label="if_val")
    axes[0, 1].semilogy(ep, np.maximum(history["if_grad"], 1e-16), label="if_grad")
    axes[0, 1].semilogy(ep, np.maximum(history["detF_barrier"], 1e-16), label="detF")
    axes[0, 1].semilogy(ep, np.maximum(history["reg"], 1e-16), label="reg")
    axes[0, 1].set_title("Coupling/Regularization Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ep, history["mu_guess"], label="mu")
    axes[1, 0].plot(ep, history["K_guess"], label="K")
    axes[1, 0].set_title("Estimated Moduli")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(ep, history["traction_rel_l2_eval"], label="traction_rel_l2_eval")
    axes[1, 1].plot(ep, history["map_valid_ratio_eval"], label="map_valid_ratio_eval")
    axes[1, 1].plot(ep, history["if_val_eval"], label="if_val_eval")
    axes[1, 1].plot(ep, history["if_grad_eval"], label="if_grad_eval")
    axes[1, 1].set_title("Eval Diagnostics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def snapshot_u_states(u_nets: Sequence[LocalVectorPINN]) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in u_nets]


def load_u_states(u_nets: Sequence[LocalVectorPINN], states: Sequence[Dict[str, torch.Tensor]]) -> None:
    for i, m in enumerate(u_nets):
        m.load_state_dict(states[i])


def run(args: argparse.Namespace) -> Dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.set_default_dtype(dtype)

    if args.n_charts != 8:
        print(f"n_charts={args.n_charts} requested; forcing n_charts=8 per benchmark plan.")
        args.n_charts = 8

    print(f"Device={device.type} dtype={dtype} n_charts={args.n_charts}")

    centers = chart_centers(args.n_charts, device=device, dtype=dtype)
    delta = torch.as_tensor(2.0 * math.pi / args.n_charts, device=device, dtype=dtype)
    h_phi = torch.as_tensor(0.5 * float(delta.item()) * (1.0 + args.overlap_ratio), device=device, dtype=dtype)
    sigma = torch.as_tensor(args.chart_sigma_scale * float(delta.item()), device=device, dtype=dtype)
    set_fillet_context(args.fillet_p, None)
    print(
        f"Filleted cube map: p={args.fillet_p:.3f} | "
        f"INN layers={args.inn_layers} width={args.inn_width} depth={args.inn_depth}"
    )
    inn_start = time.time()
    inn_map, inn_stats = train_invertible_fillet_map(
        device=device,
        dtype=dtype,
        fillet_p=args.fillet_p,
        width=args.inn_width,
        depth=args.inn_depth,
        n_layers=args.inn_layers,
        steps=args.inn_steps,
        batch_size=args.inn_batch,
        lr=args.inn_lr,
        cycle_weight=args.inn_cycle_weight,
        range_weight=args.inn_range_weight,
        eval_n=args.inn_eval_n,
        seed=args.seed + 101,
    )
    set_fillet_context(args.fillet_p, inn_map)
    inn_train_seconds = time.time() - inn_start
    print(
        f"[FilletINN] trained in {inn_train_seconds:.1f}s | "
        f"fit_rmse={inn_stats['fit_rmse']:.3e} inv_rmse={inn_stats['inv_rmse']:.3e}"
    )

    mu_init = args.mu_init if args.mu_init is not None else args.mu_true * 0.97
    K_init = args.K_init if args.K_init is not None else args.K_true * 1.04
    mu_init = clamp_init(mu_init, args.mu_true, 0.05, "mu_init")
    K_init = clamp_init(K_init, args.K_true, 0.05, "K_init")

    mu_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(mu_init - args.mu_min), device=device, dtype=dtype)
    )
    K_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(K_init - args.K_min), device=device, dtype=dtype)
    )

    u_nets = [LocalVectorPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in range(args.n_charts)]
    params: List[torch.nn.Parameter] = [mu_raw, K_raw]
    for m in u_nets:
        params.extend(list(m.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Fixed traction observations in loaded boundary subsection.
    phi_obs = sample_phi_interval(
        n=args.n_obs,
        center=args.load_phi_center,
        halfwidth=args.load_phi_halfwidth,
        device=device,
        dtype=dtype,
    )
    s_obs = sample_square_boundary(args.n_obs, device=device, dtype=dtype)
    x_obs, n_obs, _ = global_boundary_from_phi_s(
        phi=phi_obs,
        s_boundary=s_obs,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
    )
    t_obs_true = manufactured_traction(
        x=x_obs,
        n_phys=n_obs,
        mu_true=args.mu_true,
        K_true=args.K_true,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    if args.traction_noise_std > 0.0:
        t_obs_true = t_obs_true + args.traction_noise_std * torch.std(t_obs_true) * torch.randn_like(t_obs_true)
    t_obs_true = t_obs_true.detach()

    anchor_center = (args.load_phi_center + math.pi) % (2.0 * math.pi)

    pairs = neighbor_pairs_ring(args.n_charts)

    # Deterministic eval cache.
    eval_cache: Dict[str, object] = {}
    set_seed(args.eval_cache_seed)
    eval_cache["eta_int"] = [sample_eta_cube(args.eval_n_int_per_chart, device=device, dtype=dtype) for _ in range(args.n_charts)]
    pair_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for i, j in pairs:
        mid, half = interface_pair_mid_and_halfwidth(centers[i], centers[j], h_phi)
        phi_if = rand_uniform_cpu(
            args.eval_n_if_pair,
            float(mid.item() - half.item()),
            float(mid.item() + half.item()),
            device=device,
            dtype=dtype,
        )
        s_if = sample_square(args.eval_n_if_pair, device=device, dtype=dtype)
        pair_cache[(i, j)] = {"phi": phi_if, "s": s_if}
    eval_cache["if_pairs"] = pair_cache

    n_obs_eval = min(args.eval_n_obs, args.n_obs)
    eval_idx = rand_int_cpu(n=n_obs_eval, low=0, high=args.n_obs, device=device)
    eval_cache["obs_eval"] = {
        "phi": phi_obs[eval_idx],
        "s": s_obs[eval_idx],
        "n": n_obs[eval_idx],
        "t_true": t_obs_true[eval_idx],
    }
    set_seed(args.seed)

    history: Dict[str, List[float]] = {
        "total": [],
        "eq": [],
        "bc_load": [],
        "bc_anchor": [],
        "bc_free": [],
        "data_trac": [],
        "if_val": [],
        "if_grad": [],
        "detF_barrier": [],
        "reg": [],
        "mu_guess": [],
        "K_guess": [],
        "traction_rel_l2_eval": [],
        "if_val_eval": [],
        "if_grad_eval": [],
        "map_valid_ratio_eval": [],
        "detJ_min_eval": [],
        "detF_min_eval": [],
        "eq_eval": [],
    }

    start = time.time()
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_states = snapshot_u_states(u_nets)
    best_mu_raw = mu_raw.detach().clone()
    best_K_raw = K_raw.detach().clone()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        mu_est = torch.nn.functional.softplus(mu_raw) + args.mu_min
        K_est = torch.nn.functional.softplus(K_raw) + args.K_min

        # PDE interior loss per chart.
        eq_terms = []
        detF_terms = []
        valid_fracs = []
        for i in range(args.n_charts):
            eta = sample_eta_cube(args.n_int_per_chart, device=device, dtype=dtype)
            x, eta_var, jac, _, _ = chart_map_and_jacobian(
                eta_in=eta,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            inv_j, det_abs, _, valid = stabilized_jacobian_ops(
                jac=jac,
                sigma_floor=args.sigma_floor,
                det_floor=args.detJ_floor,
                jac_kappa_max=args.jac_kappa_max,
            )
            u = u_nets[i](eta_var)
            grad_u_eta = gradient_tensor(u, eta_var, create_graph=True)
            grad_u_x = torch.bmm(grad_u_eta, inv_j)
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            F = eye + grad_u_x
            P = neo_hookean_p(F, mu_est, K_est)
            flux = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(P, inv_j.transpose(1, 2))
            div_flux = divergence_mapped(flux, eta_var, create_graph=True)
            b_true = manufactured_body_force(
                x=x.detach(),
                mu_true=args.mu_true,
                K_true=args.K_true,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
                tau=args.torsion_tau,
                phi_center=args.load_phi_center,
                phi_halfwidth=args.load_phi_halfwidth,
                z_scale=args.torsion_z_scale,
            )
            res = div_flux + det_abs.unsqueeze(-1) * b_true
            if bool(torch.any(valid).item()):
                eq_i = torch.mean(res[valid] ** 2)
            else:
                eq_i = torch.mean(res**2)
            detF = det_3x3(F)
            detF_i = torch.mean(torch.nn.functional.softplus(torch.as_tensor(args.detF_floor, device=device, dtype=dtype) - detF) ** 2)
            eq_terms.append(eq_i)
            detF_terms.append(detF_i)
            valid_fracs.append(torch.mean(valid.to(dtype)))

        loss_eq = torch.mean(torch.stack(eq_terms))
        loss_detF = torch.mean(torch.stack(detF_terms))

        # Boundary/load data samples in global latent form.
        phi_load = sample_phi_interval(args.n_bc_load, args.load_phi_center, args.load_phi_halfwidth, device=device, dtype=dtype)
        s_load = sample_square_boundary(args.n_bc_load, device=device, dtype=dtype)
        x_load, n_load, _ = global_boundary_from_phi_s(
            phi=phi_load,
            s_boundary=s_load,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        u_load_true = manufactured_displacement(
            x=x_load,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        w_load = build_chart_weights(phi_load, centers=centers, sigma=sigma)

        phi_anchor = sample_phi_interval(args.n_bc_anchor, anchor_center, args.anchor_phi_halfwidth, device=device, dtype=dtype)
        s_anchor = sample_square_boundary(args.n_bc_anchor, device=device, dtype=dtype)
        x_anchor, n_anchor, _ = global_boundary_from_phi_s(
            phi=phi_anchor,
            s_boundary=s_anchor,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        u_anchor_true = torch.zeros_like(x_anchor)
        w_anchor = build_chart_weights(phi_anchor, centers=centers, sigma=sigma)

        phi_free = sample_phi_free(
            args.n_bc_free,
            load_center=args.load_phi_center,
            load_halfwidth=args.load_phi_halfwidth,
            anchor_center=anchor_center,
            anchor_halfwidth=args.anchor_phi_halfwidth,
            device=device,
            dtype=dtype,
        )
        s_free = sample_square_boundary(args.n_bc_free, device=device, dtype=dtype)
        x_free, n_free, _ = global_boundary_from_phi_s(
            phi=phi_free,
            s_boundary=s_free,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
        )
        w_free = build_chart_weights(phi_free, centers=centers, sigma=sigma)

        obs_idx = rand_int_cpu(n=min(args.n_data_batch, args.n_obs), low=0, high=args.n_obs, device=device)
        phi_data = phi_obs[obs_idx]
        s_data = s_obs[obs_idx]
        n_data = n_obs[obs_idx]
        t_data_true = t_obs_true[obs_idx]
        w_data = build_chart_weights(phi_data, centers=centers, sigma=sigma)

        # Weighted per-chart BC/data accumulators.
        num_load = torch.as_tensor(0.0, device=device, dtype=dtype)
        den_load = torch.as_tensor(0.0, device=device, dtype=dtype)
        num_anchor = torch.as_tensor(0.0, device=device, dtype=dtype)
        den_anchor = torch.as_tensor(0.0, device=device, dtype=dtype)
        num_free = torch.as_tensor(0.0, device=device, dtype=dtype)
        den_free = torch.as_tensor(0.0, device=device, dtype=dtype)
        num_data = torch.as_tensor(0.0, device=device, dtype=dtype)
        den_data = torch.as_tensor(0.0, device=device, dtype=dtype)

        for i in range(args.n_charts):
            # Load BC (Dirichlet to manufactured u*).
            eta_load = chart_eta_from_global(phi_load, s_load, centers[i], h_phi, args.eta1_clip_train)
            _, e_load, j_load, _, _ = chart_map_and_jacobian(
                eta_in=eta_load,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            inv_load, _, _, _ = stabilized_jacobian_ops(j_load, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
            u_load_pred = u_nets[i](e_load)
            diff_load = torch.mean((u_load_pred - u_load_true) ** 2, dim=1)
            wi_load = w_load[:, i]
            num_load = num_load + torch.sum(wi_load * diff_load)
            den_load = den_load + torch.sum(wi_load)

            # Anchor BC (Dirichlet zero).
            eta_anchor = chart_eta_from_global(phi_anchor, s_anchor, centers[i], h_phi, args.eta1_clip_train)
            _, e_anchor, _, _, _ = chart_map_and_jacobian(
                eta_in=eta_anchor,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            u_anchor_pred = u_nets[i](e_anchor)
            diff_anchor = torch.mean((u_anchor_pred - u_anchor_true) ** 2, dim=1)
            wi_anchor = w_anchor[:, i]
            num_anchor = num_anchor + torch.sum(wi_anchor * diff_anchor)
            den_anchor = den_anchor + torch.sum(wi_anchor)

            # Free BC traction.
            eta_free = chart_eta_from_global(phi_free, s_free, centers[i], h_phi, args.eta1_clip_train)
            _, e_free, j_free, _, _ = chart_map_and_jacobian(
                eta_in=eta_free,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            inv_free, _, _, _ = stabilized_jacobian_ops(j_free, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
            u_free = u_nets[i](e_free)
            grad_u_free = torch.bmm(gradient_tensor(u_free, e_free, create_graph=True), inv_free)
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            F_free = eye + grad_u_free
            P_free = neo_hookean_p(F_free, mu_est, K_est)
            t_free = torch.bmm(P_free, n_free.unsqueeze(-1)).squeeze(-1)
            t_free_mse = torch.mean(t_free * t_free, dim=1)
            wi_free = w_free[:, i]
            num_free = num_free + torch.sum(wi_free * t_free_mse)
            den_free = den_free + torch.sum(wi_free)

            # Traction data mismatch.
            eta_data = chart_eta_from_global(phi_data, s_data, centers[i], h_phi, args.eta1_clip_train)
            _, e_data, j_data, _, _ = chart_map_and_jacobian(
                eta_in=eta_data,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            inv_data, _, _, _ = stabilized_jacobian_ops(j_data, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
            u_data = u_nets[i](e_data)
            grad_u_data = torch.bmm(gradient_tensor(u_data, e_data, create_graph=True), inv_data)
            F_data = eye + grad_u_data
            P_data = neo_hookean_p(F_data, mu_est, K_est)
            t_data = torch.bmm(P_data, n_data.unsqueeze(-1)).squeeze(-1)
            t_data_mse = torch.mean((t_data - t_data_true) ** 2, dim=1)
            wi_data = w_data[:, i]
            num_data = num_data + torch.sum(wi_data * t_data_mse)
            den_data = den_data + torch.sum(wi_data)

        eps = torch.as_tensor(1e-12, device=device, dtype=dtype)
        loss_bc_load = num_load / torch.clamp(den_load, min=eps)
        loss_bc_anchor = num_anchor / torch.clamp(den_anchor, min=eps)
        loss_bc_free = num_free / torch.clamp(den_free, min=eps)
        loss_data = num_data / torch.clamp(den_data, min=eps)

        # Interface continuity.
        if_val_terms = []
        if_grad_terms = []
        for i, j in pairs:
            mid, half = interface_pair_mid_and_halfwidth(centers[i], centers[j], h_phi)
            phi_if = rand_uniform_cpu(
                args.n_if_pair,
                float(mid.item() - half.item()),
                float(mid.item() + half.item()),
                device=device,
                dtype=dtype,
            )
            s_if = sample_square(args.n_if_pair, device=device, dtype=dtype)

            eta_i = chart_eta_from_global(phi_if, s_if, centers[i], h_phi, args.eta1_clip_train)
            eta_j = chart_eta_from_global(phi_if, s_if, centers[j], h_phi, args.eta1_clip_train)

            _, ei, ji, _, _ = chart_map_and_jacobian(
                eta_in=eta_i,
                center=centers[i],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            _, ej, jj, _, _ = chart_map_and_jacobian(
                eta_in=eta_j,
                center=centers[j],
                h_phi=h_phi,
                major_radius=args.major_radius,
                minor_radius=args.minor_radius,
            )
            inv_i, _, _, _ = stabilized_jacobian_ops(ji, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
            inv_j, _, _, _ = stabilized_jacobian_ops(jj, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)

            ui = u_nets[i](ei)
            uj = u_nets[j](ej)
            gui = torch.bmm(gradient_tensor(ui, ei, create_graph=True), inv_i)
            guj = torch.bmm(gradient_tensor(uj, ej, create_graph=True), inv_j)

            if_val_terms.append(torch.mean((ui - uj) ** 2))
            if_grad_terms.append(torch.mean((gui - guj) ** 2))

        loss_if_val = torch.mean(torch.stack(if_val_terms))
        loss_if_grad = torch.mean(torch.stack(if_grad_terms))

        mu_prior_t = torch.as_tensor(args.mu_prior, device=device, dtype=dtype)
        K_prior_t = torch.as_tensor(args.K_prior, device=device, dtype=dtype)
        loss_reg = ((mu_est - mu_prior_t) / torch.clamp(mu_prior_t, min=eps)) ** 2 + (
            (K_est - K_prior_t) / torch.clamp(K_prior_t, min=eps)
        ) ** 2

        loss_total = (
            args.w_eq * loss_eq
            + args.w_bc_load * loss_bc_load
            + args.w_bc_anchor * loss_bc_anchor
            + args.w_bc_free * loss_bc_free
            + args.w_data_trac * loss_data
            + args.w_if_val * loss_if_val
            + args.w_if_grad * loss_if_grad
            + args.w_detF * loss_detF
            + args.w_reg * loss_reg
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_max_norm))
        optimizer.step()

        total_v = float(loss_total.detach().item())
        eq_v = float(loss_eq.detach().item())
        bc_load_v = float(loss_bc_load.detach().item())
        bc_anchor_v = float(loss_bc_anchor.detach().item())
        bc_free_v = float(loss_bc_free.detach().item())
        data_v = float(loss_data.detach().item())
        if_val_v = float(loss_if_val.detach().item())
        if_grad_v = float(loss_if_grad.detach().item())
        detF_v = float(loss_detF.detach().item())
        reg_v = float(loss_reg.detach().item())

        mu_guess = float((torch.nn.functional.softplus(mu_raw) + args.mu_min).detach().item())
        K_guess = float((torch.nn.functional.softplus(K_raw) + args.K_min).detach().item())
        mu_err = 100.0 * abs(mu_guess - args.mu_true) / max(args.mu_true, 1e-12)
        K_err = 100.0 * abs(K_guess - args.K_true) / max(args.K_true, 1e-12)

        eval_metrics = evaluate_fixed_cache(
            u_nets=u_nets,
            centers=centers,
            h_phi=h_phi,
            args=args,
            mu_est=torch.nn.functional.softplus(mu_raw) + args.mu_min,
            K_est=torch.nn.functional.softplus(K_raw) + args.K_min,
            eval_cache=eval_cache,
        )

        history["total"].append(total_v)
        history["eq"].append(eq_v)
        history["bc_load"].append(bc_load_v)
        history["bc_anchor"].append(bc_anchor_v)
        history["bc_free"].append(bc_free_v)
        history["data_trac"].append(data_v)
        history["if_val"].append(if_val_v)
        history["if_grad"].append(if_grad_v)
        history["detF_barrier"].append(detF_v)
        history["reg"].append(reg_v)
        history["mu_guess"].append(mu_guess)
        history["K_guess"].append(K_guess)
        history["traction_rel_l2_eval"].append(eval_metrics["traction_rel_l2_eval"])
        history["if_val_eval"].append(eval_metrics["if_val_eval"])
        history["if_grad_eval"].append(eval_metrics["if_grad_eval"])
        history["map_valid_ratio_eval"].append(eval_metrics["map_valid_ratio_eval"])
        history["detJ_min_eval"].append(eval_metrics["detJ_min_eval"])
        history["detF_min_eval"].append(eval_metrics["detF_min_eval"])
        history["eq_eval"].append(eval_metrics["eq_eval"])

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:04d}/{args.epochs} | total={total_v:.4e} eq={eq_v:.3e} bcL={bc_load_v:.3e} "
            f"bcA={bc_anchor_v:.3e} bcF={bc_free_v:.3e} data={data_v:.3e} ifV={if_val_v:.3e} ifG={if_grad_v:.3e} "
            f"detF={detF_v:.3e} mu_guess={mu_guess:.6f} ({mu_err:.3f}%) K_guess={K_guess:.6f} ({K_err:.3f}%) "
            f"trac_relL2={eval_metrics['traction_rel_l2_eval']:.3e} t={elapsed:.1f}s"
        )

        if total_v + args.min_delta < best_loss:
            best_loss = total_v
            best_epoch = epoch
            stale = 0
            best_states = snapshot_u_states(u_nets)
            best_mu_raw = mu_raw.detach().clone()
            best_K_raw = K_raw.detach().clone()
        else:
            stale += 1

        if args.target_loss is not None and total_v <= args.target_loss:
            print(f"Stopping early: target_loss={args.target_loss:.3e} reached at epoch {epoch}")
            break
        if stale >= args.patience:
            print(f"Stopping early: plateau patience={args.patience} reached at epoch {epoch}")
            break

    # Restore best model snapshot.
    load_u_states(u_nets, best_states)
    with torch.no_grad():
        mu_raw.copy_(best_mu_raw)
        K_raw.copy_(best_K_raw)

    # Optional short LBFGS polish on fixed mini-cache.
    if args.use_lbfgs and args.lbfgs_iters > 0:
        print(f"Starting optional LBFGS polish: iters={args.lbfgs_iters}")
        lbfgs = torch.optim.LBFGS(params, lr=args.lbfgs_lr, max_iter=args.lbfgs_iters, line_search_fn="strong_wolfe")

        # Fixed polish cache.
        polish_eta = [sample_eta_cube(args.lbfgs_n_int_per_chart, device=device, dtype=dtype) for _ in range(args.n_charts)]
        phi_pol = sample_phi_interval(args.lbfgs_n_bc_load, args.load_phi_center, args.load_phi_halfwidth, device=device, dtype=dtype)
        s_pol = sample_square_boundary(args.lbfgs_n_bc_load, device=device, dtype=dtype)
        x_pol, n_pol, _ = global_boundary_from_phi_s(phi_pol, s_pol, args.major_radius, args.minor_radius)
        t_pol_true = manufactured_traction(
            x=x_pol,
            n_phys=n_pol,
            mu_true=args.mu_true,
            K_true=args.K_true,
            major_radius=args.major_radius,
            minor_radius=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        w_pol = build_chart_weights(phi_pol, centers, sigma)

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            mu_est = torch.nn.functional.softplus(mu_raw) + args.mu_min
            K_est = torch.nn.functional.softplus(K_raw) + args.K_min
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

            eq_terms = []
            detF_terms = []
            for i in range(args.n_charts):
                x, e, j, _, _ = chart_map_and_jacobian(polish_eta[i], centers[i], h_phi, args.major_radius, args.minor_radius)
                inv_j, det_abs, _, valid = stabilized_jacobian_ops(j, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
                u = u_nets[i](e)
                grad_u = torch.bmm(gradient_tensor(u, e, create_graph=True), inv_j)
                F = eye + grad_u
                P = neo_hookean_p(F, mu_est, K_est)
                flux = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(P, inv_j.transpose(1, 2))
                div_flux = divergence_mapped(flux, e, create_graph=True)
                b_true = manufactured_body_force(
                    x=x.detach(),
                    mu_true=args.mu_true,
                    K_true=args.K_true,
                    major_radius=args.major_radius,
                    minor_radius=args.minor_radius,
                    tau=args.torsion_tau,
                    phi_center=args.load_phi_center,
                    phi_halfwidth=args.load_phi_halfwidth,
                    z_scale=args.torsion_z_scale,
                )
                res = div_flux + det_abs.unsqueeze(-1) * b_true
                if bool(torch.any(valid).item()):
                    eq_terms.append(torch.mean(res[valid] ** 2))
                else:
                    eq_terms.append(torch.mean(res**2))
                detF = det_3x3(F)
                detF_terms.append(torch.mean(torch.nn.functional.softplus(torch.as_tensor(args.detF_floor, device=device, dtype=dtype) - detF) ** 2))

            num_data = torch.as_tensor(0.0, device=device, dtype=dtype)
            den_data = torch.as_tensor(0.0, device=device, dtype=dtype)
            for i in range(args.n_charts):
                eta_i = chart_eta_from_global(phi_pol, s_pol, centers[i], h_phi, args.eta1_clip_train)
                _, e, j, _, _ = chart_map_and_jacobian(eta_i, centers[i], h_phi, args.major_radius, args.minor_radius)
                inv_j, _, _, _ = stabilized_jacobian_ops(j, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
                u = u_nets[i](e)
                grad_u = torch.bmm(gradient_tensor(u, e, create_graph=True), inv_j)
                F = eye + grad_u
                P = neo_hookean_p(F, mu_est, K_est)
                t_pred = torch.bmm(P, n_pol.unsqueeze(-1)).squeeze(-1)
                mse = torch.mean((t_pred - t_pol_true) ** 2, dim=1)
                wi = w_pol[:, i]
                num_data = num_data + torch.sum(wi * mse)
                den_data = den_data + torch.sum(wi)

            loss_eq = torch.mean(torch.stack(eq_terms))
            loss_detF = torch.mean(torch.stack(detF_terms))
            loss_data = num_data / torch.clamp(den_data, min=torch.as_tensor(1e-12, device=device, dtype=dtype))
            loss_reg = ((mu_est - args.mu_prior) / max(args.mu_prior, 1e-12)) ** 2 + (
                (K_est - args.K_prior) / max(args.K_prior, 1e-12)
            ) ** 2
            loss = args.w_eq * loss_eq + args.w_data_trac * loss_data + args.w_detF * loss_detF + args.w_reg * loss_reg
            loss.backward()
            return loss

        lbfgs.step(closure)

    # Final evaluation from best (and optional LBFGS-polished) state.
    mu_final = torch.nn.functional.softplus(mu_raw.detach()) + args.mu_min
    K_final = torch.nn.functional.softplus(K_raw.detach()) + args.K_min
    mu_final_val = scalar_from_tensor(mu_final, device=device)
    K_final_val = scalar_from_tensor(K_final, device=device)

    eval_final = evaluate_fixed_cache(
        u_nets=u_nets,
        centers=centers,
        h_phi=h_phi,
        args=args,
        mu_est=mu_final,
        K_est=K_final,
        eval_cache=eval_cache,
    )

    # Dense domain export.
    n_domain = args.n_export_domain
    phi_dom = rand_uniform_cpu(n_domain, 0.0, 2.0 * math.pi, device=device, dtype=dtype)
    s_dom = sample_square(n_domain, device=device, dtype=dtype)
    x_dom, q_dom = global_map_from_phi_s(phi_dom, s_dom, args.major_radius, args.minor_radius)
    w_dom = build_chart_weights(phi_dom, centers, sigma)
    chart_id_dom_i = torch.argmax(w_dom, dim=1)
    chart_id_dom = chart_id_dom_i.to(dtype=torch.float32)
    chart_wmax_dom = torch.max(w_dom, dim=1).values

    u_list = []
    grad_list = []
    detF_list = []
    for i in range(args.n_charts):
        eta_i = chart_eta_from_global(phi_dom, s_dom, centers[i], h_phi, args.eta1_clip_eval)
        _, e, j, _, _ = chart_map_and_jacobian(eta_i, centers[i], h_phi, args.major_radius, args.minor_radius)
        inv_j, _, _, _ = stabilized_jacobian_ops(j, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
        u_i = u_nets[i](e)
        grad_u_i = torch.bmm(gradient_tensor(u_i, e, create_graph=False), inv_j)
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        F_i = eye + grad_u_i
        detF_i = det_3x3(F_i)
        u_list.append(u_i)
        grad_list.append(grad_u_i)
        detF_list.append(detF_i)

    u_stack = torch.stack(u_list, dim=1)
    grad_stack = torch.stack(grad_list, dim=1)
    detF_stack = torch.stack(detF_list, dim=1)

    u_pred = torch.sum(w_dom.unsqueeze(-1) * u_stack, dim=1)
    grad_pred = torch.sum(w_dom.unsqueeze(-1).unsqueeze(-1) * grad_stack, dim=1)
    detF_pred = torch.sum(w_dom * detF_stack, dim=1)

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    F_pred = eye + grad_pred
    P_pred = neo_hookean_p(F_pred, mu_final, K_final)
    sigma_vm = compute_vm_stress(P=P_pred, F=F_pred)

    u_true_dom = manufactured_displacement(
        x_dom,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    u_err = u_pred - u_true_dom
    u_err_mag = torch.linalg.norm(u_err, dim=1)

    # Interface residual proxy from chart prediction spread.
    diff = torch.linalg.norm(u_stack - u_pred.unsqueeze(1), dim=2)
    interface_residual = torch.sum(w_dom * diff, dim=1)

    mu_field = torch.full((n_domain,), mu_final_val, device=device, dtype=dtype)
    K_field = torch.full((n_domain,), K_final_val, device=device, dtype=dtype)

    # Boundary observation export.
    n_bnd = args.n_export_boundary
    phi_bnd = rand_uniform_cpu(n_bnd, 0.0, 2.0 * math.pi, device=device, dtype=dtype)
    s_bnd = sample_square_boundary(n_bnd, device=device, dtype=dtype)
    x_bnd, n_bnd_vec, _ = global_boundary_from_phi_s(phi_bnd, s_bnd, args.major_radius, args.minor_radius)
    w_bnd = build_chart_weights(phi_bnd, centers, sigma)
    chart_id_bnd_i = torch.argmax(w_bnd, dim=1)
    chart_id_bnd = chart_id_bnd_i.to(dtype=torch.float32)
    chart_wmax_bnd = torch.max(w_bnd, dim=1).values

    t_bnd_all = []
    u_bnd_all = []
    grad_bnd_all = []
    for i in range(args.n_charts):
        eta_i = chart_eta_from_global(phi_bnd, s_bnd, centers[i], h_phi, args.eta1_clip_eval)
        _, e, j, _, _ = chart_map_and_jacobian(eta_i, centers[i], h_phi, args.major_radius, args.minor_radius)
        inv_j, _, _, _ = stabilized_jacobian_ops(j, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
        u_i = u_nets[i](e)
        grad_u_i = torch.bmm(gradient_tensor(u_i, e, create_graph=False), inv_j)
        F_i = eye + grad_u_i
        P_i = neo_hookean_p(F_i, mu_final, K_final)
        t_i = torch.bmm(P_i, n_bnd_vec.unsqueeze(-1)).squeeze(-1)
        t_bnd_all.append(t_i)
        u_bnd_all.append(u_i)
        grad_bnd_all.append(grad_u_i)

    t_bnd_stack = torch.stack(t_bnd_all, dim=1)
    u_bnd_stack = torch.stack(u_bnd_all, dim=1)
    grad_bnd_stack = torch.stack(grad_bnd_all, dim=1)

    t_bnd_pred = torch.sum(w_bnd.unsqueeze(-1) * t_bnd_stack, dim=1)
    u_bnd_pred = torch.sum(w_bnd.unsqueeze(-1) * u_bnd_stack, dim=1)
    grad_bnd_pred = torch.sum(w_bnd.unsqueeze(-1).unsqueeze(-1) * grad_bnd_stack, dim=1)
    F_bnd_pred = eye + grad_bnd_pred
    detF_bnd_pred = det_3x3(F_bnd_pred)

    t_bnd_true = manufactured_traction(
        x=x_bnd,
        n_phys=n_bnd_vec,
        mu_true=args.mu_true,
        K_true=args.K_true,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    t_bnd_err = t_bnd_pred - t_bnd_true
    t_bnd_err_mag = torch.linalg.norm(t_bnd_err, dim=1)

    u_bnd_true = manufactured_displacement(
        x=x_bnd,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    u_bnd_err = u_bnd_pred - u_bnd_true
    u_bnd_err_mag = torch.linalg.norm(u_bnd_err, dim=1)

    # Observation subset predictions.
    w_obs = build_chart_weights(phi_obs, centers, sigma)
    t_obs_pred_all = []
    for i in range(args.n_charts):
        eta_i = chart_eta_from_global(phi_obs, s_obs, centers[i], h_phi, args.eta1_clip_eval)
        _, e, j, _, _ = chart_map_and_jacobian(eta_i, centers[i], h_phi, args.major_radius, args.minor_radius)
        inv_j, _, _, _ = stabilized_jacobian_ops(j, args.sigma_floor, args.detJ_floor, args.jac_kappa_max)
        u_i = u_nets[i](e)
        grad_u_i = torch.bmm(gradient_tensor(u_i, e, create_graph=False), inv_j)
        F_i = eye + grad_u_i
        P_i = neo_hookean_p(F_i, mu_final, K_final)
        t_i = torch.bmm(P_i, n_obs.unsqueeze(-1)).squeeze(-1)
        t_obs_pred_all.append(t_i)
    t_obs_pred = torch.sum(w_obs.unsqueeze(-1) * torch.stack(t_obs_pred_all, dim=1), dim=1)
    t_obs_err = t_obs_pred - t_obs_true
    t_obs_err_mag = torch.linalg.norm(t_obs_err, dim=1)

    traction_rel_l2_obs_t = torch.sqrt(
        torch.mean((t_obs_pred - t_obs_true) ** 2)
        / torch.clamp(torch.mean(t_obs_true**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
    )

    # Output paths.
    os.makedirs(args.output_dir, exist_ok=True)
    stem = build_run_stem(args.run_tag)
    metrics_path = os.path.join(args.output_dir, f"{stem}_metrics.json")
    history_path = os.path.join(args.output_dir, f"{stem}_history.json")
    state_path = os.path.join(args.output_dir, f"{stem}_state.pt")
    obs_npz_path = os.path.join(args.output_dir, f"{stem}_obs.npz")
    curve_path = os.path.join(args.output_dir, f"{stem}_curves.png")
    domain_vtu_path = os.path.join(args.output_dir, f"{stem}_domain_dense.vtu")
    boundary_vtu_path = os.path.join(args.output_dir, f"{stem}_boundary_full.vtu")
    obs_vtu_path = os.path.join(args.output_dir, f"{stem}_boundary_obs_error.vtu")

    # Domain VTU.
    write_vtu_points(
        domain_vtu_path,
        points=x_dom.detach().cpu().numpy(),
        point_data={
            "phi": phi_dom.detach().cpu().numpy(),
            "q": q_dom.detach().cpu().numpy(),
            "chart_id": chart_id_dom.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_dom.detach().cpu().numpy(),
            "u_pred": u_pred.detach().cpu().numpy(),
            "u_true": u_true_dom.detach().cpu().numpy(),
            "u_error": u_err.detach().cpu().numpy(),
            "u_error_mag": u_err_mag.detach().cpu().numpy(),
            "sigma_vm": sigma_vm.detach().cpu().numpy(),
            "detF": detF_pred.detach().cpu().numpy(),
            "interface_residual": interface_residual.detach().cpu().numpy(),
            "mu_est": mu_field.detach().cpu().numpy(),
            "K_est": K_field.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_dom[:, i].detach().cpu().numpy() for i in range(args.n_charts)},
        },
    )

    # Per-chart domain VTUs.
    per_chart_paths: List[str] = []
    for i in range(args.n_charts):
        mask = w_dom[:, i] >= torch.as_tensor(args.per_chart_export_weight_threshold, device=device, dtype=dtype)
        if int(torch.sum(mask).item()) == 0:
            mask = chart_id_dom_i == i
        if int(torch.sum(mask).item()) == 0:
            continue
        p = os.path.join(args.output_dir, f"{stem}_domain_chart_{i:02d}.vtu")
        eta_i = chart_eta_from_global(phi_dom[mask], s_dom[mask], centers[i], h_phi, args.eta1_clip_eval)
        write_vtu_points(
            p,
            points=x_dom[mask].detach().cpu().numpy(),
            point_data={
                "phi": phi_dom[mask].detach().cpu().numpy(),
                "chart_id": chart_id_dom[mask].detach().cpu().numpy(),
                "chart_weight_local": w_dom[mask, i].detach().cpu().numpy(),
                "eta_local": eta_i.detach().cpu().numpy(),
                "u_pred": u_pred[mask].detach().cpu().numpy(),
                "u_true": u_true_dom[mask].detach().cpu().numpy(),
                "u_error_mag": u_err_mag[mask].detach().cpu().numpy(),
                "sigma_vm": sigma_vm[mask].detach().cpu().numpy(),
                "detF": detF_pred[mask].detach().cpu().numpy(),
            },
        )
        per_chart_paths.append(p)

    # Full boundary VTU.
    write_vtu_points(
        boundary_vtu_path,
        points=x_bnd.detach().cpu().numpy(),
        point_data={
            "phi": phi_bnd.detach().cpu().numpy(),
            "chart_id": chart_id_bnd.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_bnd.detach().cpu().numpy(),
            "normal": n_bnd_vec.detach().cpu().numpy(),
            "traction_true": t_bnd_true.detach().cpu().numpy(),
            "traction_pred": t_bnd_pred.detach().cpu().numpy(),
            "traction_error": t_bnd_err.detach().cpu().numpy(),
            "traction_error_mag": t_bnd_err_mag.detach().cpu().numpy(),
            "u_pred": u_bnd_pred.detach().cpu().numpy(),
            "u_true": u_bnd_true.detach().cpu().numpy(),
            "u_error": u_bnd_err.detach().cpu().numpy(),
            "u_error_mag": u_bnd_err_mag.detach().cpu().numpy(),
            "detF": detF_bnd_pred.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_bnd[:, i].detach().cpu().numpy() for i in range(args.n_charts)},
        },
    )

    # Observation error VTU.
    chart_id_obs = torch.argmax(w_obs, dim=1).to(dtype=torch.float32)
    chart_wmax_obs = torch.max(w_obs, dim=1).values
    write_vtu_points(
        obs_vtu_path,
        points=x_obs.detach().cpu().numpy(),
        point_data={
            "phi": phi_obs.detach().cpu().numpy(),
            "chart_id": chart_id_obs.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_obs.detach().cpu().numpy(),
            "normal": n_obs.detach().cpu().numpy(),
            "traction_true": t_obs_true.detach().cpu().numpy(),
            "traction_pred": t_obs_pred.detach().cpu().numpy(),
            "traction_error": t_obs_err.detach().cpu().numpy(),
            "traction_error_mag": t_obs_err_mag.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_obs[:, i].detach().cpu().numpy() for i in range(args.n_charts)},
        },
    )

    metrics = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "chart_domain": "filleted_cube",
        "fillet_p": float(args.fillet_p),
        "inn_train_seconds": float(inn_train_seconds),
        "inn_stats": inn_stats,
        "n_charts": int(args.n_charts),
        "overlap_ratio": float(args.overlap_ratio),
        "h_phi": float(h_phi.item()),
        "mu_true": float(args.mu_true),
        "K_true": float(args.K_true),
        "mu_init": float(mu_init),
        "K_init": float(K_init),
        "mu_final": float(mu_final_val),
        "K_final": float(K_final_val),
        "mu_rel_error_percent": float(100.0 * abs(mu_final_val - args.mu_true) / max(args.mu_true, 1e-12)),
        "K_rel_error_percent": float(100.0 * abs(K_final_val - args.K_true) / max(args.K_true, 1e-12)),
        "traction_rel_l2_obs": float(traction_rel_l2_obs_t.detach().item()),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history["total"])),
        "runtime_seconds": float(time.time() - start),
        "final_losses": {
            "total": float(history["total"][-1]) if history["total"] else None,
            "eq": float(history["eq"][-1]) if history["eq"] else None,
            "bc_load": float(history["bc_load"][-1]) if history["bc_load"] else None,
            "bc_anchor": float(history["bc_anchor"][-1]) if history["bc_anchor"] else None,
            "bc_free": float(history["bc_free"][-1]) if history["bc_free"] else None,
            "data_trac": float(history["data_trac"][-1]) if history["data_trac"] else None,
            "if_val": float(history["if_val"][-1]) if history["if_val"] else None,
            "if_grad": float(history["if_grad"][-1]) if history["if_grad"] else None,
            "detF_barrier": float(history["detF_barrier"][-1]) if history["detF_barrier"] else None,
            "reg": float(history["reg"][-1]) if history["reg"] else None,
        },
        "eval_final": eval_final,
        "target_met": bool(
            (100.0 * abs(mu_final_val - args.mu_true) / max(args.mu_true, 1e-12) <= args.acc_mu_rel_percent)
            and (100.0 * abs(K_final_val - args.K_true) / max(args.K_true, 1e-12) <= args.acc_K_rel_percent)
            and (float(traction_rel_l2_obs_t.detach().item()) <= args.acc_traction_rel_l2)
            and (float(eval_final["detF_min_eval"]) > args.detF_floor)
        ),
        "acceptance_thresholds": {
            "mu_rel_error_percent": float(args.acc_mu_rel_percent),
            "K_rel_error_percent": float(args.acc_K_rel_percent),
            "traction_rel_l2": float(args.acc_traction_rel_l2),
            "detF_floor": float(args.detF_floor),
        },
        "paths": {
            "metrics": metrics_path,
            "history": history_path,
            "state": state_path,
            "obs_npz": obs_npz_path,
            "curves": curve_path,
            "domain_vtu": domain_vtu_path,
            "boundary_vtu": boundary_vtu_path,
            "obs_vtu": obs_vtu_path,
            "per_chart_domain_vtu": per_chart_paths,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "u_states": snapshot_u_states(u_nets),
            "u_kwargs": {
                "width": args.pinn_width,
                "depth": args.pinn_depth,
            },
            "inn_state": {k: v.detach().to(device=torch.device("cpu")) for k, v in inn_map.state_dict().items()},
            "inn_kwargs": {
                "width": int(args.inn_width),
                "depth": int(args.inn_depth),
                "n_layers": int(args.inn_layers),
                "fillet_p": float(args.fillet_p),
            },
            "mu_raw": mu_raw.detach().to(device=torch.device("cpu")),
            "K_raw": K_raw.detach().to(device=torch.device("cpu")),
            "mu_final": float(mu_final_val),
            "K_final": float(K_final_val),
            "centers": centers.detach().to(device=torch.device("cpu")),
            "h_phi": float(h_phi.detach().cpu().item()),
            "args": vars(args),
            "metrics": metrics,
            "history": history,
        },
        state_path,
    )

    np.savez_compressed(
        obs_npz_path,
        phi_obs=phi_obs.detach().cpu().numpy(),
        s_obs=s_obs.detach().cpu().numpy(),
        x_obs=x_obs.detach().cpu().numpy(),
        n_obs=n_obs.detach().cpu().numpy(),
        t_obs_true=t_obs_true.detach().cpu().numpy(),
        t_obs_pred=t_obs_pred.detach().cpu().numpy(),
        chart_weights_obs=w_obs.detach().cpu().numpy(),
        phi_domain=phi_dom.detach().cpu().numpy(),
        s_domain=s_dom.detach().cpu().numpy(),
        x_domain=x_dom.detach().cpu().numpy(),
        u_pred_domain=u_pred.detach().cpu().numpy(),
        u_true_domain=u_true_dom.detach().cpu().numpy(),
        u_err_domain=u_err.detach().cpu().numpy(),
        sigma_vm_domain=sigma_vm.detach().cpu().numpy(),
        detF_domain=detF_pred.detach().cpu().numpy(),
        chart_weights_domain=w_dom.detach().cpu().numpy(),
    )

    make_plot(history, curve_path)

    print("Inverse torus filleted-cube INN run complete")
    print(f"  curves:   {curve_path}")
    print(f"  metrics:  {metrics_path}")
    print(f"  history:  {history_path}")
    print(f"  state:    {state_path}")
    print(f"  obs:      {obs_npz_path}")
    print(f"  domain:   {domain_vtu_path}")
    print(f"  boundary: {boundary_vtu_path}")
    print(f"  obs_vtu:  {obs_vtu_path}")
    print(f"  per_chart_domain_vtu_files: {len(per_chart_paths)}")
    print(f"  mu_true={args.mu_true:.6f}, mu_final={mu_final_val:.6f}")
    print(f"  K_true={args.K_true:.6f}, K_final={K_final_val:.6f}")
    print(f"  traction_rel_l2_obs={float(traction_rel_l2_obs_t.detach().item()):.6e}")
    print(f"  target_met={metrics['target_met']}")

    return metrics


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = parse_simple_yaml(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(
        description="Inverse Neo-Hookean on torus with 8 filleted-cube charts and learned invertible map."
    )
    parser.add_argument("--config", default=pre_args.config)

    out_default = cfg.get("output-dir", None)
    parser.add_argument("--output-dir", required=(out_default is None), default=out_default)
    parser.add_argument("--run-tag", default=cfg_get_str(cfg, "run-tag", ""))

    parser.add_argument("--device", default=cfg_get_str(cfg, "device", "auto"), choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default=cfg_get_str(cfg, "dtype", "auto"), choices=["auto", "float32", "float64"])

    parser.add_argument("--n-charts", type=int, default=cfg_get_int(cfg, "n-charts", 8))
    parser.add_argument("--overlap-ratio", type=float, default=cfg_get_float(cfg, "overlap-ratio", 0.20))
    parser.add_argument("--chart-sigma-scale", type=float, default=cfg_get_float(cfg, "chart-sigma-scale", 0.65))
    parser.add_argument("--fillet-p", type=float, default=cfg_get_float(cfg, "fillet-p", 6.0))
    parser.add_argument("--inn-width", type=int, default=cfg_get_int(cfg, "inn-width", 64))
    parser.add_argument("--inn-depth", type=int, default=cfg_get_int(cfg, "inn-depth", 2))
    parser.add_argument("--inn-layers", type=int, default=cfg_get_int(cfg, "inn-layers", 8))
    parser.add_argument("--inn-steps", type=int, default=cfg_get_int(cfg, "inn-steps", 800))
    parser.add_argument("--inn-batch", type=int, default=cfg_get_int(cfg, "inn-batch", 2048))
    parser.add_argument("--inn-lr", type=float, default=cfg_get_float(cfg, "inn-lr", 1e-3))
    parser.add_argument("--inn-cycle-weight", type=float, default=cfg_get_float(cfg, "inn-cycle-weight", 0.3))
    parser.add_argument("--inn-range-weight", type=float, default=cfg_get_float(cfg, "inn-range-weight", 0.1))
    parser.add_argument("--inn-eval-n", type=int, default=cfg_get_int(cfg, "inn-eval-n", 8192))

    parser.add_argument("--major-radius", type=float, default=cfg_get_float(cfg, "major-radius", 1.0))
    parser.add_argument("--minor-radius", type=float, default=cfg_get_float(cfg, "minor-radius", 0.35))

    parser.add_argument("--load-phi-center", type=float, default=cfg_get_float(cfg, "load-phi-center", 0.5 * math.pi))
    parser.add_argument("--load-phi-halfwidth", type=float, default=cfg_get_float(cfg, "load-phi-halfwidth", 0.35 * math.pi))
    parser.add_argument("--anchor-phi-halfwidth", type=float, default=cfg_get_float(cfg, "anchor-phi-halfwidth", 0.20 * math.pi))

    parser.add_argument("--torsion-tau", type=float, default=cfg_get_float(cfg, "torsion-tau", 0.085))
    parser.add_argument("--torsion-z-scale", type=float, default=cfg_get_float(cfg, "torsion-z-scale", 0.75))

    parser.add_argument("--mu-true", type=float, default=cfg_get_float(cfg, "mu-true", 1.8))
    parser.add_argument("--K-true", type=float, default=cfg_get_float(cfg, "K-true", 25.0))
    parser.add_argument("--mu-init", type=float, default=float(cfg["mu-init"]) if "mu-init" in cfg else None)
    parser.add_argument("--K-init", type=float, default=float(cfg["K-init"]) if "K-init" in cfg else None)
    parser.add_argument("--mu-min", type=float, default=cfg_get_float(cfg, "mu-min", 1e-4))
    parser.add_argument("--K-min", type=float, default=cfg_get_float(cfg, "K-min", 1e-4))
    parser.add_argument("--mu-prior", type=float, default=cfg_get_float(cfg, "mu-prior", 1.8))
    parser.add_argument("--K-prior", type=float, default=cfg_get_float(cfg, "K-prior", 25.0))

    parser.add_argument("--n-obs", type=int, default=cfg_get_int(cfg, "n-obs", 6000))
    parser.add_argument("--n-int-per-chart", type=int, default=cfg_get_int(cfg, "n-int-per-chart", 96))
    parser.add_argument("--n-bc-load", type=int, default=cfg_get_int(cfg, "n-bc-load", 768))
    parser.add_argument("--n-bc-anchor", type=int, default=cfg_get_int(cfg, "n-bc-anchor", 384))
    parser.add_argument("--n-bc-free", type=int, default=cfg_get_int(cfg, "n-bc-free", 768))
    parser.add_argument("--n-if-pair", type=int, default=cfg_get_int(cfg, "n-if-pair", 192))
    parser.add_argument("--n-data-batch", type=int, default=cfg_get_int(cfg, "n-data-batch", 1024))

    parser.add_argument("--eval-cache-seed", type=int, default=cfg_get_int(cfg, "eval-cache-seed", 1234))
    parser.add_argument("--eval-n-int-per-chart", type=int, default=cfg_get_int(cfg, "eval-n-int-per-chart", 96))
    parser.add_argument("--eval-n-if-pair", type=int, default=cfg_get_int(cfg, "eval-n-if-pair", 128))
    parser.add_argument("--eval-n-obs", type=int, default=cfg_get_int(cfg, "eval-n-obs", 2000))

    parser.add_argument("--pinn-width", type=int, default=cfg_get_int(cfg, "pinn-width", 96))
    parser.add_argument("--pinn-depth", type=int, default=cfg_get_int(cfg, "pinn-depth", 5))

    parser.add_argument("--epochs", type=int, default=cfg_get_int(cfg, "epochs", 250))
    parser.add_argument("--lr", type=float, default=cfg_get_float(cfg, "lr", 8e-4))
    parser.add_argument("--grad-clip-max-norm", type=float, default=cfg_get_float(cfg, "grad-clip-max-norm", 5.0))

    parser.add_argument("--w-eq", type=float, default=cfg_get_float(cfg, "w-eq", 1.0))
    parser.add_argument("--w-bc-load", type=float, default=cfg_get_float(cfg, "w-bc-load", 2.0))
    parser.add_argument("--w-bc-anchor", type=float, default=cfg_get_float(cfg, "w-bc-anchor", 2.0))
    parser.add_argument("--w-bc-free", type=float, default=cfg_get_float(cfg, "w-bc-free", 1.0))
    parser.add_argument("--w-data-trac", type=float, default=cfg_get_float(cfg, "w-data-trac", 3.0))
    parser.add_argument("--w-if-val", type=float, default=cfg_get_float(cfg, "w-if-val", 1.0))
    parser.add_argument("--w-if-grad", type=float, default=cfg_get_float(cfg, "w-if-grad", 0.5))
    parser.add_argument("--w-detF", type=float, default=cfg_get_float(cfg, "w-detF", 1e-2))
    parser.add_argument("--w-reg", type=float, default=cfg_get_float(cfg, "w-reg", 1e-3))

    parser.add_argument("--sigma-floor", type=float, default=cfg_get_float(cfg, "sigma-floor", 1e-6))
    parser.add_argument("--detJ-floor", type=float, default=cfg_get_float(cfg, "detJ-floor", 1e-6))
    parser.add_argument("--jac-kappa-max", type=float, default=cfg_get_float(cfg, "jac-kappa-max", 1e4))
    parser.add_argument("--detF-floor", type=float, default=cfg_get_float(cfg, "detF-floor", 1e-4))

    parser.add_argument("--traction-noise-std", type=float, default=cfg_get_float(cfg, "traction-noise-std", 0.0))

    parser.add_argument("--target-loss", type=float, default=float(cfg["target-loss"]) if "target-loss" in cfg else None)
    parser.add_argument("--patience", type=int, default=cfg_get_int(cfg, "patience", 120))
    parser.add_argument("--min-delta", type=float, default=cfg_get_float(cfg, "min-delta", 1e-8))

    parser.add_argument("--use-lbfgs", action="store_true", default=cfg_get_bool(cfg, "use-lbfgs", False))
    parser.add_argument("--lbfgs-iters", type=int, default=cfg_get_int(cfg, "lbfgs-iters", 10))
    parser.add_argument("--lbfgs-lr", type=float, default=cfg_get_float(cfg, "lbfgs-lr", 0.8))
    parser.add_argument("--lbfgs-n-int-per-chart", type=int, default=cfg_get_int(cfg, "lbfgs-n-int-per-chart", 48))
    parser.add_argument("--lbfgs-n-bc-load", type=int, default=cfg_get_int(cfg, "lbfgs-n-bc-load", 256))

    parser.add_argument("--eta1-clip-train", type=float, default=cfg_get_float(cfg, "eta1-clip-train", 1.5))
    parser.add_argument("--eta1-clip-eval", type=float, default=cfg_get_float(cfg, "eta1-clip-eval", 1.5))

    parser.add_argument("--n-export-domain", type=int, default=cfg_get_int(cfg, "n-export-domain", 80000))
    parser.add_argument("--n-export-boundary", type=int, default=cfg_get_int(cfg, "n-export-boundary", 50000))
    parser.add_argument(
        "--per-chart-export-weight-threshold",
        type=float,
        default=cfg_get_float(cfg, "per-chart-export-weight-threshold", 1.0 / 16.0),
    )

    parser.add_argument("--acc-mu-rel-percent", type=float, default=cfg_get_float(cfg, "acc-mu-rel-percent", 5.0))
    parser.add_argument("--acc-K-rel-percent", type=float, default=cfg_get_float(cfg, "acc-K-rel-percent", 5.0))
    parser.add_argument("--acc-traction-rel-l2", type=float, default=cfg_get_float(cfg, "acc-traction-rel-l2", 0.08))

    parser.add_argument("--seed", type=int, default=cfg_get_int(cfg, "seed", 42))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
