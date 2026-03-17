#!/usr/bin/env python3
"""
Rabbit inverse benchmark: transverse-isotropic Arruda-Boyce material ID from
surface displacement data on a fixed coordinate-chart atlas.

Pipeline:
1) Load fixed rabbit atlas (12 charts by default).
2) Build synthetic teacher data for 3 independent load cases.
3) Inverse train chart PINNs + 4 material parameters (mu_m, N_m, k_f, N_f)
   with equilibrium, interface, data, rigid-body penalty, and regularization.
4) Export metrics/history/checkpoint + ParaView VTU fields + sensitivity report.
"""

from __future__ import annotations

import argparse
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


def softplus_inv(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


def build_run_stem(run_tag: str) -> str:
    run_tag = str(run_tag).strip()
    if run_tag:
        return f"rabbit_inverse_ti_arruda_boyce_atlas_schwarz_{run_tag}"
    return "rabbit_inverse_ti_arruda_boyce_atlas_schwarz"


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


# ----------------------------- Synthetic loads -----------------------------


@dataclass
class LoadCase:
    name: str
    amp: float
    kind: str


def make_load_cases() -> List[LoadCase]:
    return [
        LoadCase(name="shear_xz", amp=0.070, kind="shear"),
        LoadCase(name="vertical_comp", amp=0.060, kind="compress"),
        LoadCase(name="torsion_like", amp=0.050, kind="torsion"),
    ]


def manufactured_u_case(x: torch.Tensor, case: LoadCase) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    if case.kind == "shear":
        ux = x3 + 0.20 * x1 * x2
        uy = 0.15 * x1 * x3
        uz = -0.10 * x2 * x3
        u = torch.stack([ux, uy, uz], dim=1)
    elif case.kind == "compress":
        ux = -0.20 * x1 * x3
        uy = -0.22 * x2 * x3
        uz = 0.62 * x3 + 0.10 * (x1 * x1 + x2 * x2)
        u = torch.stack([ux, uy, uz], dim=1)
    else:
        w = 0.5 + 0.5 * torch.tanh(2.0 * x3)
        ux = -x2
        uy = x1
        uz = 0.20 * x3 * (x1 - x2)
        u = w.unsqueeze(1) * torch.stack([ux, uy, uz], dim=1)

    return torch.as_tensor(case.amp, device=x.device, dtype=x.dtype) * u


def body_force_from_teacher(
    x: torch.Tensor,
    case: LoadCase,
    mu_m_true: float,
    N_m_true: float,
    k_f_true: float,
    N_f_true: float,
    kappa: float,
    a0: torch.Tensor,
) -> torch.Tensor:
    with torch.enable_grad():
        xv = x.clone().detach().requires_grad_(True)
        u = manufactured_u_case(xv, case)
        grad_u = gradient_tensor(u, xv, create_graph=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        F = eye + grad_u
        P = ti_arruda_boyce_p(
            F=F,
            mu_m=torch.as_tensor(mu_m_true, device=x.device, dtype=x.dtype),
            N_m=torch.as_tensor(N_m_true, device=x.device, dtype=x.dtype),
            k_f=torch.as_tensor(k_f_true, device=x.device, dtype=x.dtype),
            N_f=torch.as_tensor(N_f_true, device=x.device, dtype=x.dtype),
            kappa=kappa,
            a0=a0,
        )

        div_p = []
        for i in range(3):
            comp = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            for j in range(3):
                d = torch.autograd.grad(
                    P[:, i, j],
                    xv,
                    grad_outputs=torch.ones_like(P[:, i, j]),
                    create_graph=False,
                    retain_graph=True,
                )[0][:, j : j + 1]
                comp = comp + d
            div_p.append(comp)
        div_p = torch.cat(div_p, dim=1)
        b = -div_p
    return b.detach()


# ----------------------------- IO helpers -----------------------------


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


# ----------------------------- Training data cache -----------------------------


@dataclass
class ChartCaches:
    xi_int: torch.Tensor
    b_true_by_case: List[torch.Tensor]
    x_obs: torch.Tensor
    xi_obs: torch.Tensor
    u_obs_true_by_case: List[torch.Tensor]


@dataclass
class PairCache:
    xi_i: torch.Tensor
    xi_j: torch.Tensor


def sample_indices(n_total: int, n_take: int, device: torch.device) -> torch.Tensor:
    if n_take >= n_total:
        return torch.randint(0, n_total, (n_take,), device=device)
    perm = torch.randperm(n_total, device=device)
    return perm[:n_take]


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
    decoders: Sequence[ChartDecoder],
    load_cases: Sequence[LoadCase],
    a0: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ChartCaches]:
    n_charts = membership.shape[1]
    caches: List[ChartCaches] = []

    for i in range(n_charts):
        idx_all = torch.where(membership[:, i] > 0.5)[0]
        if idx_all.numel() < 32:
            raise RuntimeError(f"Chart {i} has too few support points: {idx_all.numel()}")

        # Interior cache: local coords around support cloud with isotropic jitter.
        pick_int = idx_all[torch.randint(0, idx_all.numel(), (args.int_cache_per_chart,), device=device)]
        x_int_seed = points[pick_int]
        xi_int = local_coords(x_int_seed, seeds[i], t1[i], t2[i], nvec[i])

        # Expand into local volume to improve interior coverage.
        r = torch.clamp(chart_scales[i], min=torch.as_tensor(1e-3, device=device, dtype=dtype))
        noise = torch.randn_like(xi_int) * (args.xi_noise_scale * r)
        noise[:, 2] = noise[:, 2] * args.normal_noise_boost
        xi_int = xi_int + noise

        with torch.no_grad():
            x_int = decoders[i](xi_int, seed=seeds[i], t1=t1[i], t2=t2[i], n=nvec[i], chart_scale=chart_scales[i])

        b_true_by_case: List[torch.Tensor] = []
        for case in load_cases:
            b_case = body_force_from_teacher(
                x=x_int,
                case=case,
                mu_m_true=args.mu_m_true,
                N_m_true=args.N_m_true,
                k_f_true=args.k_f_true,
                N_f_true=args.N_f_true,
                kappa=args.kappa_fixed,
                a0=a0,
            )
            if args.b_true_clip > 0.0:
                clip_t = torch.as_tensor(args.b_true_clip, device=device, dtype=dtype)
                b_case = torch.clamp(b_case, min=-clip_t, max=clip_t)
            b_true_by_case.append(b_case)

        # Surface observation cache from atlas points.
        pick_obs = idx_all[torch.randint(0, idx_all.numel(), (args.n_obs_per_chart,), device=device)]
        x_obs = points[pick_obs]
        xi_obs = local_coords(x_obs, seeds[i], t1[i], t2[i], nvec[i])

        u_obs_by_case: List[torch.Tensor] = []
        for case in load_cases:
            u_obs_by_case.append(manufactured_u_case(x_obs, case).detach())

        caches.append(
            ChartCaches(
                xi_int=xi_int.detach(),
                b_true_by_case=[b.detach() for b in b_true_by_case],
                x_obs=x_obs.detach(),
                xi_obs=xi_obs.detach(),
                u_obs_true_by_case=u_obs_by_case,
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
            if shared.numel() < 16:
                continue
            n_take = min(args.if_cache_per_pair, int(shared.numel()))
            sel = shared[torch.randint(0, shared.numel(), (n_take,), device=device)]
            x = points[sel]
            xi_i = local_coords(x, seeds[i], t1[i], t2[i], nvec[i]).detach()
            xi_j = local_coords(x, seeds[j], t1[j], t2[j], nvec[j]).detach()
            out[(i, j)] = PairCache(xi_i=xi_i, xi_j=xi_j)

    return out


def rigid_body_penalty(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Penalty on projection onto 6 rigid-body modes (3 translations + 3 rotations).
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


# ----------------------------- Sensitivity gate -----------------------------


def traction_from_u(
    x: torch.Tensor,
    n: torch.Tensor,
    u_fn,
    params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    kappa: float,
    a0: torch.Tensor,
) -> torch.Tensor:
    xv = x.clone().detach().requires_grad_(True)
    u = u_fn(xv)
    grad_u = gradient_tensor(u, xv, create_graph=True)
    eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
    F = eye + grad_u
    mu_m, N_m, k_f, N_f = params
    P = ti_arruda_boyce_p(F=F, mu_m=mu_m, N_m=N_m, k_f=k_f, N_f=N_f, kappa=kappa, a0=a0)
    t = torch.bmm(P, n.unsqueeze(-1)).squeeze(-1)
    return t


def sensitivity_report(
    x: torch.Tensor,
    n: torch.Tensor,
    load_cases: Sequence[LoadCase],
    args: argparse.Namespace,
    a0: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    base = np.array([args.mu_m_init, args.N_m_init, args.k_f_init, args.N_f_init], dtype=np.float64)
    eps_rel = np.array([1e-2, 1e-2, 1e-2, 1e-2], dtype=np.float64)

    cols = []
    for pidx in range(4):
        plus = base.copy()
        minus = base.copy()
        d = eps_rel[pidx] * max(abs(base[pidx]), 1e-3)
        plus[pidx] += d
        minus[pidx] -= d

        t_plus_all = []
        t_minus_all = []
        for case in load_cases:
            ufn = lambda xx, c=case: manufactured_u_case(xx, c)
            tp = traction_from_u(
                x=x,
                n=n,
                u_fn=ufn,
                params=(
                    torch.as_tensor(plus[0], device=device, dtype=dtype),
                    torch.as_tensor(plus[1], device=device, dtype=dtype),
                    torch.as_tensor(plus[2], device=device, dtype=dtype),
                    torch.as_tensor(plus[3], device=device, dtype=dtype),
                ),
                kappa=args.kappa_fixed,
                a0=a0,
            )
            tm = traction_from_u(
                x=x,
                n=n,
                u_fn=ufn,
                params=(
                    torch.as_tensor(minus[0], device=device, dtype=dtype),
                    torch.as_tensor(minus[1], device=device, dtype=dtype),
                    torch.as_tensor(minus[2], device=device, dtype=dtype),
                    torch.as_tensor(minus[3], device=device, dtype=dtype),
                ),
                kappa=args.kappa_fixed,
                a0=a0,
            )
            t_plus_all.append(tp.detach().to(device=torch.device("cpu")).numpy().reshape(-1))
            t_minus_all.append(tm.detach().to(device=torch.device("cpu")).numpy().reshape(-1))

        vec_plus = np.concatenate(t_plus_all, axis=0)
        vec_minus = np.concatenate(t_minus_all, axis=0)
        dcol = (vec_plus - vec_minus) / (2.0 * d)
        cols.append(dcol)

    J = np.stack(cols, axis=1)  # [M,4]
    _, svals, _ = np.linalg.svd(J, full_matrices=False)
    rank = int(np.sum(svals > (svals[0] * 1e-10 if svals.size > 0 else 0.0)))
    cond = float(svals[0] / max(svals[-1], 1e-16)) if svals.size > 0 else float("inf")

    report = {
        "rank": rank,
        "condition_number": cond,
        "singular_values": [float(s) for s in svals.tolist()],
        "passes_rank4": bool(rank >= 4),
        "passes_cond": bool(cond <= args.sensitivity_cond_max),
        "pass": bool(rank >= 4 and cond <= args.sensitivity_cond_max),
    }
    return report


# ----------------------------- Evaluation/export -----------------------------


def compute_blended_prediction(
    x: torch.Tensor,
    chart_nets: Sequence[LocalVectorPINN],
    masks: Sequence[MaskNet],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    chart_scales: torch.Tensor,
    blend_temp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = []
    ups = []
    for i in range(len(chart_nets)):
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        li = masks[i](xi, chart_scale=chart_scales[i])
        ui = chart_nets[i](xi)
        logits.append(li)
        ups.append(ui)

    L = torch.stack(logits, dim=1)
    W = torch.softmax(L / max(blend_temp, 1e-6), dim=1)
    U = torch.stack(ups, dim=1)
    u_blend = torch.sum(W.unsqueeze(-1) * U, dim=1)
    chart_id = torch.argmax(W, dim=1)
    wmax = torch.max(W, dim=1).values
    return u_blend, chart_id, wmax


def export_case_vtu(
    out_path: str,
    x: torch.Tensor,
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    chart_id: torch.Tensor,
    wmax: torch.Tensor,
    mu_m: float,
    N_m: float,
    k_f: float,
    N_f: float,
) -> None:
    u_err = u_pred - u_true
    u_err_mag = torch.linalg.norm(u_err, dim=1)
    write_vtu_points(
        out_path,
        points=x.detach().cpu().numpy(),
        point_data={
            "u_pred": u_pred.detach().cpu().numpy(),
            "u_true": u_true.detach().cpu().numpy(),
            "u_error": u_err.detach().cpu().numpy(),
            "u_error_mag": u_err_mag.detach().cpu().numpy(),
            "chart_id": chart_id.to(dtype=torch.float32).detach().cpu().numpy(),
            "chart_weight_max": wmax.detach().cpu().numpy(),
            "mu_m_est": np.full((x.shape[0],), mu_m, dtype=np.float32),
            "N_m_est": np.full((x.shape[0],), N_m, dtype=np.float32),
            "k_f_est": np.full((x.shape[0],), k_f, dtype=np.float32),
            "N_f_est": np.full((x.shape[0],), N_f, dtype=np.float32),
        },
    )


def make_plot(history: Dict[str, List[float]], out_png: str) -> None:
    ep = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].semilogy(ep, np.maximum(history["total"], 1e-16), label="total")
    axes[0, 0].semilogy(ep, np.maximum(history["eq"], 1e-16), label="eq")
    axes[0, 0].semilogy(ep, np.maximum(history["data"], 1e-16), label="data")
    axes[0, 0].semilogy(ep, np.maximum(history["if_val"], 1e-16), label="if_val")
    axes[0, 0].semilogy(ep, np.maximum(history["if_flux"], 1e-16), label="if_flux")
    axes[0, 0].set_title("Loss Terms")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(ep, np.maximum(history["rbm"], 1e-16), label="rbm")
    axes[0, 1].semilogy(ep, np.maximum(history["reg"], 1e-16), label="reg")
    axes[0, 1].plot(ep, history["surf_rel_l2"], label="surf_rel_l2")
    axes[0, 1].set_title("Regularization / Fit")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ep, history["mu_m"], label="mu_m")
    axes[1, 0].plot(ep, history["N_m"], label="N_m")
    axes[1, 0].plot(ep, history["k_f"], label="k_f")
    axes[1, 0].plot(ep, history["N_f"], label="N_f")
    axes[1, 0].set_title("Estimated Parameters")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ep, history["err_mu_m_pct"], label="mu_m %")
    axes[1, 1].plot(ep, history["err_N_m_pct"], label="N_m %")
    axes[1, 1].plot(ep, history["err_k_f_pct"], label="k_f %")
    axes[1, 1].plot(ep, history["err_N_f_pct"], label="N_f %")
    axes[1, 1].set_title("Parameter Relative Errors (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Main training -----------------------------


def run(args: argparse.Namespace) -> Dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.set_default_dtype(dtype)
    set_seed(args.seed)

    if args.device == "mps" and dtype == torch.float64:
        dtype = torch.float32

    if args.amp and device.type == "mps":
        # Keep solver deterministic/stable on MPS for high-order autograd.
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

    decoders, masks, _ = load_atlas_models(args.atlas_checkpoint, device=device, dtype=dtype)

    load_cases = make_load_cases()[: args.n_load_cases]
    if len(load_cases) < 1:
        raise RuntimeError("n_load_cases must be >= 1")

    # Atlas points in this repository are already in the canonical normalized frame.
    # Keep them unchanged to stay consistent with decoder geometry.
    points_n = points
    # Keep a0 fixed and non-axis-aligned.
    a0 = normalize_rows(torch.tensor([[args.fiber_a0_x, args.fiber_a0_y, args.fiber_a0_z]], device=device, dtype=dtype))[0]

    chart_caches = prepare_chart_caches(
        args=args,
        points=points_n,
        normals=normals,
        membership=membership,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        chart_scales=chart_scales,
        decoders=decoders,
        load_cases=load_cases,
        a0=a0,
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

    # Observability/sensitivity gate on a dense surface subset.
    obs_idx = torch.randint(0, points_n.shape[0], (min(args.sensitivity_n_surface, points_n.shape[0]),), device=device)
    x_sens = points_n[obs_idx]
    n_sens = normalize_rows(normals[obs_idx])
    sens = sensitivity_report(x=x_sens, n=n_sens, load_cases=load_cases, args=args, a0=a0, device=device, dtype=dtype)

    # Teacher stage: generate synthetic surface observations.
    print(f"Teacher stage: generating synthetic displacement observations for {len(load_cases)} load cases")

    u_nets: List[List[LocalVectorPINN]] = []
    chart_opts: List[torch.optim.Optimizer] = []
    for i in range(n_charts):
        case_nets = [LocalVectorPINN(width=args.pinn_width, depth=args.pinn_depth).to(device=device, dtype=dtype) for _ in load_cases]
        u_nets.append(case_nets)
        params_i: List[torch.nn.Parameter] = []
        for m in case_nets:
            params_i.extend(list(m.parameters()))
        chart_opts.append(torch.optim.Adam(params_i, lr=args.lr_u))

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
    theta_opt = torch.optim.Adam(list(ti_params.parameters()), lr=args.lr_theta)

    history: Dict[str, List[float]] = {
        "total": [],
        "eq": [],
        "data": [],
        "if_val": [],
        "if_flux": [],
        "rbm": [],
        "reg": [],
        "surf_rel_l2": [],
        "mu_m": [],
        "N_m": [],
        "k_f": [],
        "N_f": [],
        "err_mu_m_pct": [],
        "err_N_m_pct": [],
        "err_k_f_pct": [],
        "err_N_f_pct": [],
    }

    best = {
        "score": float("inf"),
        "epoch": 0,
        "ti_state": {k: v.detach().clone() for k, v in ti_params.state_dict().items()},
        "u_states": [[{k: v.detach().clone() for k, v in m.state_dict().items()} for m in nets] for nets in u_nets],
    }

    start = time.time()

    def snapshot_u() -> List[List[Dict[str, torch.Tensor]]]:
        return [[{k: v.detach().clone() for k, v in m.state_dict().items()} for m in nets] for nets in u_nets]

    def load_u(states: List[List[Dict[str, torch.Tensor]]]) -> None:
        for i in range(n_charts):
            for c in range(len(load_cases)):
                u_nets[i][c].load_state_dict(states[i][c])

    # Main inverse stage.
    for it in range(1, args.max_schwarz_iters + 1):
        loss_eq_acc = 0.0
        loss_data_acc = 0.0
        loss_if_val_acc = 0.0
        loss_if_flux_acc = 0.0
        loss_rbm_acc = 0.0
        loss_reg_acc = 0.0
        n_updates = 0

        for group in color_groups:
            for i in group:
                chart_opts[i].zero_grad()
                theta_opt.zero_grad()

                mu_m, N_m, k_f, N_f = ti_params.values()

                loss_eq_i = torch.zeros((), device=device, dtype=dtype)
                loss_data_i = torch.zeros((), device=device, dtype=dtype)
                loss_if_val_i = torch.zeros((), device=device, dtype=dtype)
                loss_if_flux_i = torch.zeros((), device=device, dtype=dtype)
                loss_rbm_i = torch.zeros((), device=device, dtype=dtype)

                cache_i = chart_caches[i]

                # Per-load terms.
                for cidx, case in enumerate(load_cases):
                    # Interior equilibrium residual.
                    sel_int = sample_indices(cache_i.xi_int.shape[0], args.pde_batch, device=device)
                    xi_int = cache_i.xi_int[sel_int]
                    b_true = cache_i.b_true_by_case[cidx][sel_int]

                    x_int, xi_var, jac = chart_map_and_jacobian(
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

                    u_pred = u_nets[i][cidx](xi_var)
                    grad_u_xi = gradient_tensor(u_pred, xi_var, create_graph=True)
                    grad_u_x = torch.bmm(grad_u_xi, inv_j)
                    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
                    F = eye + grad_u_x
                    P = ti_arruda_boyce_p(F=F, mu_m=mu_m, N_m=N_m, k_f=k_f, N_f=N_f, kappa=args.kappa_fixed, a0=a0)
                    mapped_flux = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(P, inv_j.transpose(1, 2))
                    div_flux = divergence_mapped(mapped_flux, xi_var, create_graph=True)
                    res = div_flux + det_abs.unsqueeze(-1) * b_true
                    if args.eq_residual_clip > 0.0:
                        clip_t = torch.as_tensor(args.eq_residual_clip, device=device, dtype=dtype)
                        res = torch.clamp(res, min=-clip_t, max=clip_t)
                    res = torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

                    if bool(torch.any(valid).item()):
                        eq_i = torch.mean(res[valid] ** 2)
                    else:
                        eq_i = torch.mean(res**2)
                    loss_eq_i = loss_eq_i + eq_i

                    # Surface displacement data.
                    sel_obs = sample_indices(cache_i.x_obs.shape[0], args.bc_batch, device=device)
                    xi_obs = cache_i.xi_obs[sel_obs]
                    x_obs = cache_i.x_obs[sel_obs]
                    u_true_obs = cache_i.u_obs_true_by_case[cidx][sel_obs]
                    u_obs_pred = u_nets[i][cidx](xi_obs)
                    data_i = torch.mean((u_obs_pred - u_true_obs) ** 2)
                    loss_data_i = loss_data_i + data_i

                    # RBM penalty on observed surface prediction.
                    rbm_i = rigid_body_penalty(x=x_obs, u=u_obs_pred)
                    loss_rbm_i = loss_rbm_i + rbm_i

                    # Interface coupling to neighbors for this case.
                    for j in neighbors.get(i, []):
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

                        sel_if = sample_indices(xi_i_all.shape[0], args.if_batch, device=device)
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

                        ui = u_nets[i][cidx](xi_i_var)
                        gi = torch.bmm(gradient_tensor(ui, xi_i_var, create_graph=True), inv_i)
                        uj = u_nets[j_idx][cidx](xi_j_var)
                        gj = torch.bmm(gradient_tensor(uj, xi_j_var, create_graph=False), inv_jj)
                        # Schwarz-style one-sided update: treat neighbor chart as fixed target.
                        uj_t = uj.detach()
                        gj_t = gj.detach()

                        loss_if_val_i = loss_if_val_i + torch.mean((ui - uj_t) ** 2)
                        loss_if_flux_i = loss_if_flux_i + torch.mean((gi - gj_t) ** 2)

                # Average over load cases.
                denom_cases = float(len(load_cases))
                loss_eq_i = loss_eq_i / denom_cases
                loss_data_i = loss_data_i / denom_cases
                loss_rbm_i = loss_rbm_i / denom_cases
                loss_if_val_i = loss_if_val_i / max(denom_cases, 1.0)
                loss_if_flux_i = loss_if_flux_i / max(denom_cases, 1.0)

                # Weak priors on log-params.
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
                    args.w_data * loss_data_i
                    + args.w_eq * loss_eq_i
                    + args.w_if_val * loss_if_val_i
                    + args.w_if_flux * loss_if_flux_i
                    + args.w_rbm * loss_rbm_i
                    + args.w_reg * loss_reg
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(u_nets[i][0].parameters()) + list(ti_params.parameters()), max_norm=args.grad_clip)
                chart_opts[i].step()
                theta_opt.step()

                loss_eq_acc += float(loss_eq_i.detach().item())
                loss_data_acc += float(loss_data_i.detach().item())
                loss_if_val_acc += float(loss_if_val_i.detach().item())
                loss_if_flux_acc += float(loss_if_flux_i.detach().item())
                loss_rbm_acc += float(loss_rbm_i.detach().item())
                loss_reg_acc += float(loss_reg.detach().item())
                n_updates += 1

        # Eval on observation caches.
        mu_m_t, N_m_t, k_f_t, N_f_t = ti_params.values()

        case_rel_l2 = []
        for cidx in range(len(load_cases)):
            pred_all = []
            true_all = []
            for i in range(n_charts):
                with torch.no_grad():
                    u_pred = u_nets[i][cidx](chart_caches[i].xi_obs)
                pred_all.append(u_pred)
                true_all.append(chart_caches[i].u_obs_true_by_case[cidx])
            up = torch.cat(pred_all, dim=0)
            ut = torch.cat(true_all, dim=0)
            rel = torch.sqrt(torch.mean((up - ut) ** 2) / torch.clamp(torch.mean(ut**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype)))
            case_rel_l2.append(float(rel.detach().item()))

        surf_rel = float(np.mean(case_rel_l2))

        eq_v = loss_eq_acc / max(n_updates, 1)
        data_v = loss_data_acc / max(n_updates, 1)
        ifv_v = loss_if_val_acc / max(n_updates, 1)
        iff_v = loss_if_flux_acc / max(n_updates, 1)
        rbm_v = loss_rbm_acc / max(n_updates, 1)
        reg_v = loss_reg_acc / max(n_updates, 1)
        total_v = args.w_data * data_v + args.w_eq * eq_v + args.w_if_val * ifv_v + args.w_if_flux * iff_v + args.w_rbm * rbm_v + args.w_reg * reg_v

        mu_m_val = scalar(mu_m_t, device)
        N_m_val = scalar(N_m_t, device)
        k_f_val = scalar(k_f_t, device)
        N_f_val = scalar(N_f_t, device)

        err_mu = 100.0 * abs(mu_m_val - args.mu_m_true) / max(args.mu_m_true, 1e-12)
        err_Nm = 100.0 * abs(N_m_val - args.N_m_true) / max(args.N_m_true, 1e-12)
        err_kf = 100.0 * abs(k_f_val - args.k_f_true) / max(args.k_f_true, 1e-12)
        err_Nf = 100.0 * abs(N_f_val - args.N_f_true) / max(args.N_f_true, 1e-12)

        history["total"].append(total_v)
        history["eq"].append(eq_v)
        history["data"].append(data_v)
        history["if_val"].append(ifv_v)
        history["if_flux"].append(iff_v)
        history["rbm"].append(rbm_v)
        history["reg"].append(reg_v)
        history["surf_rel_l2"].append(surf_rel)
        history["mu_m"].append(mu_m_val)
        history["N_m"].append(N_m_val)
        history["k_f"].append(k_f_val)
        history["N_f"].append(N_f_val)
        history["err_mu_m_pct"].append(err_mu)
        history["err_N_m_pct"].append(err_Nm)
        history["err_k_f_pct"].append(err_kf)
        history["err_N_f_pct"].append(err_Nf)

        score = surf_rel + 0.002 * (err_mu + err_Nm + err_kf + err_Nf)
        if score < best["score"]:
            best["score"] = score
            best["epoch"] = it
            best["ti_state"] = {k: v.detach().clone() for k, v in ti_params.state_dict().items()}
            best["u_states"] = snapshot_u()

        elapsed = time.time() - start
        print(
            f"Epoch {it:04d}/{args.max_schwarz_iters} | total={total_v:.4e} eq={eq_v:.3e} data={data_v:.3e} "
            f"ifV={ifv_v:.3e} ifF={iff_v:.3e} rbm={rbm_v:.3e} "
            f"mu_m={mu_m_val:.6f} ({err_mu:.3f}%) N_m={N_m_val:.6f} ({err_Nm:.3f}%) "
            f"k_f={k_f_val:.6f} ({err_kf:.3f}%) N_f={N_f_val:.6f} ({err_Nf:.3f}%) "
            f"surf_relL2={surf_rel:.3e} t={elapsed:.1f}s"
        )

        if (
            err_mu <= args.acc_param_rel_percent
            and err_Nm <= args.acc_param_rel_percent
            and err_kf <= args.acc_param_rel_percent
            and err_Nf <= args.acc_param_rel_percent
            and surf_rel <= args.acc_surface_rel_l2
        ):
            print("Stopping early: acceptance criteria reached.")
            break

    # Restore best state.
    ti_params.load_state_dict(best["ti_state"])
    load_u(best["u_states"])  # type: ignore[arg-type]
    mu_m_t, N_m_t, k_f_t, N_f_t = ti_params.values()

    # Final metrics and exports.
    os.makedirs(args.output_dir, exist_ok=True)
    stem = build_run_stem(args.run_tag)

    # Aggregate surface metrics by load case.
    case_metrics = []
    for cidx, case in enumerate(load_cases):
        pred_all = []
        true_all = []
        x_all = []
        for i in range(n_charts):
            with torch.no_grad():
                up = u_nets[i][cidx](chart_caches[i].xi_obs)
            ut = chart_caches[i].u_obs_true_by_case[cidx]
            pred_all.append(up)
            true_all.append(ut)
            x_all.append(chart_caches[i].x_obs)
        up = torch.cat(pred_all, dim=0)
        ut = torch.cat(true_all, dim=0)
        xx = torch.cat(x_all, dim=0)
        rel = torch.sqrt(torch.mean((up - ut) ** 2) / torch.clamp(torch.mean(ut**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype)))
        max_err = torch.max(torch.abs(up - ut))
        case_metrics.append(
            {
                "name": case.name,
                "surface_rel_l2": float(rel.detach().item()),
                "surface_max_abs": float(max_err.detach().item()),
                "n_surface": int(xx.shape[0]),
            }
        )

    surf_rel_mean = float(np.mean([m["surface_rel_l2"] for m in case_metrics]))

    # Domain export points from decoder-mapped interior caches.
    domain_chunks = []
    for i in range(n_charts):
        n_take = min(args.export_points_per_chart, chart_caches[i].xi_int.shape[0])
        sel = sample_indices(chart_caches[i].xi_int.shape[0], n_take, device=device)
        xi = chart_caches[i].xi_int[sel]
        with torch.no_grad():
            x = decoders[i](xi, seed=seeds[i], t1=t1[i], t2=t2[i], n=nvec[i], chart_scale=chart_scales[i])
        domain_chunks.append(x)
    x_dom = torch.cat(domain_chunks, dim=0)

    for cidx, case in enumerate(load_cases):
        # Domain blended field.
        with torch.no_grad():
            u_dom_pred, cid_dom, wmax_dom = compute_blended_prediction(
                x=x_dom,
                chart_nets=[u_nets[i][cidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )
            u_dom_true = manufactured_u_case(x_dom, case)

        dom_vtu = os.path.join(args.output_dir, f"{stem}_{case.name}_domain.vtu")
        export_case_vtu(
            out_path=dom_vtu,
            x=x_dom,
            u_pred=u_dom_pred,
            u_true=u_dom_true,
            chart_id=cid_dom,
            wmax=wmax_dom,
            mu_m=scalar(mu_m_t, device),
            N_m=scalar(N_m_t, device),
            k_f=scalar(k_f_t, device),
            N_f=scalar(N_f_t, device),
        )

        # Surface blended field.
        x_surf = points_n
        with torch.no_grad():
            u_s_pred, cid_s, wmax_s = compute_blended_prediction(
                x=x_surf,
                chart_nets=[u_nets[i][cidx] for i in range(n_charts)],
                masks=masks,
                seeds=seeds,
                t1=t1,
                t2=t2,
                nvec=nvec,
                chart_scales=chart_scales,
                blend_temp=args.blend_temp,
            )
            u_s_true = manufactured_u_case(x_surf, case)

        surf_vtu = os.path.join(args.output_dir, f"{stem}_{case.name}_surface.vtu")
        export_case_vtu(
            out_path=surf_vtu,
            x=x_surf,
            u_pred=u_s_pred,
            u_true=u_s_true,
            chart_id=cid_s,
            wmax=wmax_s,
            mu_m=scalar(mu_m_t, device),
            N_m=scalar(N_m_t, device),
            k_f=scalar(k_f_t, device),
            N_f=scalar(N_f_t, device),
        )

    # Save artifacts.
    hist_path = os.path.join(args.output_dir, f"{stem}_history.json")
    met_path = os.path.join(args.output_dir, f"{stem}_metrics.json")
    sens_path = os.path.join(args.output_dir, f"{stem}_sensitivity.json")
    ckpt_path = os.path.join(args.output_dir, f"{stem}_checkpoint.pt")
    curve_path = os.path.join(args.output_dir, f"{stem}_curves.png")

    metrics = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_charts": n_charts,
        "n_load_cases": len(load_cases),
        "best_epoch": int(best["epoch"]),
        "epochs_completed": int(len(history["total"])),
        "mu_m_true": float(args.mu_m_true),
        "N_m_true": float(args.N_m_true),
        "k_f_true": float(args.k_f_true),
        "N_f_true": float(args.N_f_true),
        "mu_m_est": float(scalar(mu_m_t, device)),
        "N_m_est": float(scalar(N_m_t, device)),
        "k_f_est": float(scalar(k_f_t, device)),
        "N_f_est": float(scalar(N_f_t, device)),
        "mu_m_rel_error_percent": float(100.0 * abs(scalar(mu_m_t, device) - args.mu_m_true) / max(args.mu_m_true, 1e-12)),
        "N_m_rel_error_percent": float(100.0 * abs(scalar(N_m_t, device) - args.N_m_true) / max(args.N_m_true, 1e-12)),
        "k_f_rel_error_percent": float(100.0 * abs(scalar(k_f_t, device) - args.k_f_true) / max(args.k_f_true, 1e-12)),
        "N_f_rel_error_percent": float(100.0 * abs(scalar(N_f_t, device) - args.N_f_true) / max(args.N_f_true, 1e-12)),
        "surface_rel_l2_mean": float(surf_rel_mean),
        "case_metrics": case_metrics,
        "sensitivity": sens,
        "target_met": bool(
            (100.0 * abs(scalar(mu_m_t, device) - args.mu_m_true) / max(args.mu_m_true, 1e-12) <= args.acc_param_rel_percent)
            and (100.0 * abs(scalar(N_m_t, device) - args.N_m_true) / max(args.N_m_true, 1e-12) <= args.acc_param_rel_percent)
            and (100.0 * abs(scalar(k_f_t, device) - args.k_f_true) / max(args.k_f_true, 1e-12) <= args.acc_param_rel_percent)
            and (100.0 * abs(scalar(N_f_t, device) - args.N_f_true) / max(args.N_f_true, 1e-12) <= args.acc_param_rel_percent)
            and (surf_rel_mean <= args.acc_surface_rel_l2)
        ),
        "runtime_seconds": float(time.time() - start),
        "paths": {
            "metrics": met_path,
            "history": hist_path,
            "checkpoint": ckpt_path,
            "curves": curve_path,
            "sensitivity": sens_path,
        },
    }

    checkpoint = {
        "u_states": [[m.state_dict() for m in nets] for nets in u_nets],
        "ti_state": ti_params.state_dict(),
        "args": vars(args),
        "load_cases": [c.__dict__ for c in load_cases],
        "metrics": metrics,
    }

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)
    torch.save(checkpoint, ckpt_path)
    make_plot(history, curve_path)

    print("Rabbit inverse TI Arruda-Boyce run complete")
    print(f"  metrics:    {met_path}")
    print(f"  history:    {hist_path}")
    print(f"  sensitivity:{sens_path}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  curves:     {curve_path}")

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
        description="Rabbit surface-displacement inverse TI Arruda-Boyce benchmark on fixed atlas charts."
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
            "/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/rabbit_inverse_ti_arruda_boyce_main",
        ),
    )
    p.add_argument("--run-tag", default=cfg_get_str(cfg, "run-tag", "main"))

    p.add_argument("--seed", type=int, default=cfg_get_int(cfg, "seed", 42))
    p.add_argument("--device", default=cfg_get_str(cfg, "device", "auto"), choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default=cfg_get_str(cfg, "dtype", "auto"), choices=["auto", "float32", "float64"])
    p.add_argument("--amp", type=parse_bool, default=cfg_get_bool(cfg, "amp", False))

    p.add_argument("--pinn-width", type=int, default=cfg_get_int(cfg, "pinn-width", 96))
    p.add_argument("--pinn-depth", type=int, default=cfg_get_int(cfg, "pinn-depth", 5))

    p.add_argument("--n-load-cases", type=int, default=cfg_get_int(cfg, "n-load-cases", 3))
    p.add_argument("--max-schwarz-iters", type=int, default=cfg_get_int(cfg, "max-schwarz-iters", 60))
    p.add_argument("--local-steps", type=int, default=cfg_get_int(cfg, "local-steps", 2))
    p.add_argument("--lr-u", type=float, default=cfg_get_float(cfg, "lr-u", 2e-4))
    p.add_argument("--lr-theta", type=float, default=cfg_get_float(cfg, "lr-theta", 2e-3))
    p.add_argument("--grad-clip", type=float, default=cfg_get_float(cfg, "grad-clip", 5.0))

    p.add_argument("--int-cache-per-chart", type=int, default=cfg_get_int(cfg, "int-cache-per-chart", 1600))
    p.add_argument("--if-cache-per-pair", type=int, default=cfg_get_int(cfg, "if-cache-per-pair", 800))
    p.add_argument("--n-obs-per-chart", type=int, default=cfg_get_int(cfg, "n-obs-per-chart", 512))
    p.add_argument("--pde-batch", type=int, default=cfg_get_int(cfg, "pde-batch", 96))
    p.add_argument("--bc-batch", type=int, default=cfg_get_int(cfg, "bc-batch", 128))
    p.add_argument("--if-batch", type=int, default=cfg_get_int(cfg, "if-batch", 96))
    p.add_argument("--xi-noise-scale", type=float, default=cfg_get_float(cfg, "xi-noise-scale", 0.08))
    p.add_argument("--normal-noise-boost", type=float, default=cfg_get_float(cfg, "normal-noise-boost", 1.4))

    p.add_argument("--w-data", type=float, default=cfg_get_float(cfg, "w-data", 80.0))
    p.add_argument("--w-eq", type=float, default=cfg_get_float(cfg, "w-eq", 1.0))
    p.add_argument("--w-if-val", type=float, default=cfg_get_float(cfg, "w-if-val", 5.0))
    p.add_argument("--w-if-flux", type=float, default=cfg_get_float(cfg, "w-if-flux", 20.0))
    p.add_argument("--w-rbm", type=float, default=cfg_get_float(cfg, "w-rbm", 20.0))
    p.add_argument("--w-reg", type=float, default=cfg_get_float(cfg, "w-reg", 1e-2))

    p.add_argument("--sigma-floor", type=float, default=cfg_get_float(cfg, "sigma-floor", 1e-6))
    p.add_argument("--detJ-floor", type=float, default=cfg_get_float(cfg, "detJ-floor", 1e-6))
    p.add_argument("--jac-kappa-max", type=float, default=cfg_get_float(cfg, "jac-kappa-max", 1e4))
    p.add_argument("--eq-residual-clip", type=float, default=cfg_get_float(cfg, "eq-residual-clip", 200.0))
    p.add_argument("--b-true-clip", type=float, default=cfg_get_float(cfg, "b-true-clip", 200.0))

    p.add_argument("--kappa-fixed", type=float, default=cfg_get_float(cfg, "kappa-fixed", 80.0))

    p.add_argument("--mu-m-true", type=float, default=cfg_get_float(cfg, "mu-m-true", 1.80))
    p.add_argument("--N-m-true", type=float, default=cfg_get_float(cfg, "N-m-true", 4.80))
    p.add_argument("--k-f-true", type=float, default=cfg_get_float(cfg, "k-f-true", 2.20))
    p.add_argument("--N-f-true", type=float, default=cfg_get_float(cfg, "N-f-true", 5.20))

    p.add_argument("--mu-m-init", type=float, default=cfg_get_float(cfg, "mu-m-init", 1.74))
    p.add_argument("--N-m-init", type=float, default=cfg_get_float(cfg, "N-m-init", 4.60))
    p.add_argument("--k-f-init", type=float, default=cfg_get_float(cfg, "k-f-init", 2.12))
    p.add_argument("--N-f-init", type=float, default=cfg_get_float(cfg, "N-f-init", 5.05))

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

    p.add_argument("--acc-param-rel-percent", type=float, default=cfg_get_float(cfg, "acc-param-rel-percent", 5.0))
    p.add_argument("--acc-surface-rel-l2", type=float, default=cfg_get_float(cfg, "acc-surface-rel-l2", 1e-2))

    p.add_argument("--sensitivity-n-surface", type=int, default=cfg_get_int(cfg, "sensitivity-n-surface", 3000))
    p.add_argument("--sensitivity-cond-max", type=float, default=cfg_get_float(cfg, "sensitivity-cond-max", 1e5))

    p.add_argument("--blend-temp", type=float, default=cfg_get_float(cfg, "blend-temp", 1.0))
    p.add_argument("--export-points-per-chart", type=int, default=cfg_get_int(cfg, "export-points-per-chart", 2200))

    return p.parse_args(rest)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
