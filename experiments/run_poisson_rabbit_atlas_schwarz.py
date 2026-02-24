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


def robust_unit_interval(values: torch.Tensor, q_lo: float = 0.05, q_hi: float = 0.95) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros_like(values)
    # MPS does not support float64 tensors; move to CPU first, then cast if needed.
    v_cpu = values.detach().to(device=torch.device("cpu")).reshape(-1)
    if not torch.is_floating_point(v_cpu):
        v_cpu = v_cpu.to(dtype=torch.float32)
    elif v_cpu.dtype in (torch.float16, torch.bfloat16):
        v_cpu = v_cpu.to(dtype=torch.float32)
    if not torch.isfinite(v_cpu).all():
        v_cpu = torch.nan_to_num(v_cpu, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(torch.quantile(v_cpu, max(0.0, min(1.0, q_lo))).item())
    hi = float(torch.quantile(v_cpu, max(0.0, min(1.0, q_hi))).item())
    if hi <= lo + 1e-12:
        return torch.zeros_like(values)
    out = (values - lo) / (hi - lo)
    return torch.clamp(out, min=0.0, max=1.0)


def sample_indices_from_weights(weights: torch.Tensor, n_take: int) -> torch.Tensor:
    n = int(weights.numel())
    if n <= 0:
        return torch.zeros((0,), device=weights.device, dtype=torch.int64)
    if n_take <= 0:
        return torch.zeros((0,), device=weights.device, dtype=torch.int64)
    replace = n_take > n

    # On MPS, keep this entire path on CPU to avoid sporadic Metal backend index errors.
    if weights.device.type == "mps":
        w_cpu = weights.detach().to(device=torch.device("cpu"))
        if not torch.is_floating_point(w_cpu):
            w_cpu = w_cpu.to(dtype=torch.float32)
        elif w_cpu.dtype in (torch.float16, torch.bfloat16):
            w_cpu = w_cpu.to(dtype=torch.float32)
        w_cpu = torch.nan_to_num(torch.clamp(w_cpu, min=1e-12), nan=0.0, posinf=0.0, neginf=0.0)
        w_sum = float(torch.sum(w_cpu).item())
        if (not np.isfinite(w_sum)) or w_sum <= 0.0:
            sel = torch.randint(0, n, (n_take,), device=torch.device("cpu"))
            return sel.to(device=weights.device, dtype=torch.int64)
        w_cpu = w_cpu / w_sum
        sel = torch.multinomial(w_cpu, n_take, replacement=replace)
        return sel.to(device=weights.device, dtype=torch.int64)

    w = torch.clamp(weights, min=1e-12)
    w_sum = torch.sum(w)
    if (not bool(torch.isfinite(w_sum).item())) or float(w_sum.item()) <= 0.0:
        return torch.randint(0, n, (n_take,), device=weights.device)
    w = w / w_sum
    return torch.multinomial(w, n_take, replacement=replace)


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
        self.raw_scale = torch.nn.Parameter(torch.tensor(-1.8, dtype=torch.get_default_dtype()))

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
    def __init__(
        self,
        width: int = 64,
        depth: int = 4,
        arch: str = "mlp",
        residual_scale: float = 0.1,
        residual_zero_init: bool = True,
    ):
        super().__init__()
        arch = str(arch).lower().strip()
        if arch not in {"mlp", "resnet"}:
            raise ValueError(f"Unsupported LocalPoissonPINN arch: {arch}")
        self.arch = arch
        self.residual_scale = float(residual_scale)
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)
        if self.arch == "resnet":
            self.res_net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)
            if residual_zero_init:
                # Start from the base branch when warm-starting from an MLP checkpoint.
                torch.nn.init.zeros_(self.res_net.out.weight)
                torch.nn.init.zeros_(self.res_net.out.bias)
        else:
            self.res_net = None

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        out = self.net(xi)
        if self.res_net is not None:
            out = out + self.residual_scale * self.res_net(xi)
        return out


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
    # MPS currently has unstable behavior for linalg.svd/linalg.inv in this workflow.
    # Use an adjugate-based 3x3 inverse branch on MPS to avoid those kernels.
    if jac.device.type == "mps":
        eye = torch.eye(3, device=jac.device, dtype=jac.dtype).unsqueeze(0).expand(jac.shape[0], 3, 3)
        jac_reg = jac + float(sigma_floor) * eye

        a = jac_reg[:, 0, 0]
        b = jac_reg[:, 0, 1]
        c = jac_reg[:, 0, 2]
        d = jac_reg[:, 1, 0]
        e = jac_reg[:, 1, 1]
        f = jac_reg[:, 1, 2]
        g = jac_reg[:, 2, 0]
        h = jac_reg[:, 2, 1]
        i = jac_reg[:, 2, 2]

        c11 = e * i - f * h
        c12 = c * h - b * i
        c13 = b * f - c * e
        c21 = f * g - d * i
        c22 = a * i - c * g
        c23 = c * d - a * f
        c31 = d * h - e * g
        c32 = b * g - a * h
        c33 = a * e - b * d

        det_reg = a * c11 + b * c21 + c * c31
        det_reg_safe = torch.where(
            det_reg >= 0.0,
            torch.clamp(det_reg, min=det_floor),
            torch.clamp(det_reg, max=-det_floor),
        )
        adj = torch.stack(
            [
                torch.stack([c11, c12, c13], dim=1),
                torch.stack([c21, c22, c23], dim=1),
                torch.stack([c31, c32, c33], dim=1),
            ],
            dim=1,
        )
        inv_j = adj / det_reg_safe.unsqueeze(-1).unsqueeze(-1)

        # Avoid torch.det on MPS (unstable in some builds) with explicit 3x3 determinant.
        qa = jac[:, 0, 0]
        qb = jac[:, 0, 1]
        qc = jac[:, 0, 2]
        qd = jac[:, 1, 0]
        qe = jac[:, 1, 1]
        qf = jac[:, 1, 2]
        qg = jac[:, 2, 0]
        qh = jac[:, 2, 1]
        qi = jac[:, 2, 2]
        det_raw = qa * (qe * qi - qf * qh) - qb * (qd * qi - qf * qg) + qc * (qd * qh - qe * qg)
        raw_det_abs = torch.abs(det_raw)
        det_abs = torch.clamp(raw_det_abs, min=det_floor)
        # Condition surrogate: ||J||_F * ||J^{-1}||_F
        n_j = torch.sqrt(torch.clamp(torch.sum(jac_reg * jac_reg, dim=(1, 2)), min=1e-18))
        n_inv = torch.sqrt(torch.clamp(torch.sum(inv_j * inv_j, dim=(1, 2)), min=1e-18))
        kappa = n_j * n_inv
        valid = raw_det_abs > det_floor
        valid = valid & torch.isfinite(kappa) & torch.isfinite(det_abs) & torch.isfinite(det_reg_safe)
        return inv_j, det_abs, kappa, valid

    # M3: Use SVD only in no_grad for masking/kappa; invert via LU (torch.linalg.inv).
    # SVD singular-value backward is unstable when singular values cluster (common for
    # near-isometric maps). LU inversion has well-defined, numerically stable gradients.
    with torch.no_grad():
        s_chk = torch.linalg.svdvals(jac)
        s_safe_chk = torch.clamp(s_chk, min=sigma_floor)
        raw_det_abs = torch.abs(torch.linalg.det(jac))
        kappa = s_safe_chk[:, 0] / torch.clamp(s_safe_chk[:, -1], min=sigma_floor)
        valid = raw_det_abs > det_floor
        valid = valid & torch.isfinite(kappa) & torch.isfinite(raw_det_abs)

    det_abs = torch.clamp(torch.abs(torch.linalg.det(jac)), min=det_floor)
    inv_j = torch.linalg.inv(jac)
    eye3 = torch.eye(3, device=jac.device, dtype=jac.dtype).unsqueeze(0).expand_as(inv_j)
    inv_j = torch.where(valid.view(-1, 1, 1), inv_j, eye3)
    return inv_j, det_abs, kappa, valid


# ---------------------------------------------------------------------------
# M1 – Direct-coordinate PDE helpers (bypass learned decoder Jacobian)
# ---------------------------------------------------------------------------

def grad_u_in_physical_tnb(
    u_model: "LocalPoissonPINN",
    x_phys: torch.Tensor,
    seed: torch.Tensor,
    t1_vec: torch.Tensor,
    t2_vec: torch.Tensor,
    n_vec: torch.Tensor,
) -> torch.Tensor:
    """Return ∇u in physical (x,y,z) coordinates via the rigid TNB frame.

    The chart coordinate ξ = local_coords(x) is an orthogonal rigid map, so
    ∂u/∂x = J_TNB^T · ∂u/∂ξ where J_TNB = [t1 | t2 | n] (orthonormal columns).
    """
    x = x_phys.clone().detach().requires_grad_(True)
    xi = local_coords(x, seed, t1_vec, t2_vec, n_vec)
    u = u_model(xi)
    grad_xi = torch.autograd.grad(
        u, xi,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3)
    # J_TNB columns are t1, t2, n (all unit vectors, orthonormal)
    t1e = t1_vec.unsqueeze(0)   # (1, 3)
    t2e = t2_vec.unsqueeze(0)
    ne  = n_vec.unsqueeze(0)
    grad_x = (
        grad_xi[:, 0:1] * t1e
        + grad_xi[:, 1:2] * t2e
        + grad_xi[:, 2:3] * ne
    )  # (N, 3)
    return grad_x


def direct_poisson_residual_tnb(
    u_model: "LocalPoissonPINN",
    x_in: torch.Tensor,
    seed: torch.Tensor,
    t1_vec: torch.Tensor,
    t2_vec: torch.Tensor,
    n_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute –Δu – f(x) in physical space via the rigid TNB frame.

    Since the TNB frame is orthogonal (det J = 1, κ = 1), Δ_x u = Δ_ξ u exactly;
    no learned decoder Jacobian is involved so there is no SVD instability.

    Returns:
        residual : (N, 1)
        x_phys   : (N, 3) with grad enabled
        valid    : bool tensor (N,) – all True for the direct path
    """
    x = x_in.clone().detach().requires_grad_(True)
    xi = local_coords(x, seed, t1_vec, t2_vec, n_vec)
    u = u_model(xi)

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

    residual = -lap - forcing_f(x)
    valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    return residual, x, valid


# ---------------------------------------------------------------------------
# M7 – SDF-based hard BC weighting
# ---------------------------------------------------------------------------

def sdf_hard_bc_scale(
    x_phys: torch.Tensor,
    sdf_net: torch.nn.Module,
    sdf_center: torch.Tensor,
    sdf_scale_val: float,
    hard_bc_scale: float,
) -> torch.Tensor:
    """Return tanh(max(0, –SDF(x)) / scale) ∈ [0,1].

    This is ≈ 0 near the boundary (SDF ≈ 0) and ≈ 1 deep in the interior,
    providing a smooth, SDF-driven multiplier for hard BC enforcement.
    """
    x_norm = (x_phys - sdf_center.unsqueeze(0)) / sdf_scale_val
    sdf_val = sdf_net(x_norm)
    depth = torch.clamp(-sdf_val, min=0.0)
    return torch.tanh(depth / max(hard_bc_scale, 1e-8)).unsqueeze(-1)


# ---------------------------------------------------------------------------
# M2 – SDF-guided volumetric interior sampling
# ---------------------------------------------------------------------------

class _SDFNetSchwarz(torch.nn.Module):
    """Thin wrapper: MLP(in=3, out=1) that returns a signed distance value."""
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_sdf_for_schwarz(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, torch.Tensor, float]:
    """Load a trained SDF network from *ckpt_path*.

    Returns:
        sdf_net    : eval-mode SDF network (x_norm → signed distance)
        sdf_center : (3,) tensor – domain centre used during SDF training
        sdf_scale  : float – domain scale used during SDF training
    """
    # Always load checkpoint to CPU first to avoid dtype/device mismatches
    # (e.g. MPS does not support float64; loading float64 weights on MPS raises an error).
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support two checkpoint formats:
    #   Format A (train_sdf_rabbit.py): keys "model_state", "model_kwargs", "center", "scale"
    #   Format B (simplified):         keys "model", "width", "depth", "center", "scale"
    if "model_state" in ckpt:
        kw = ckpt.get("model_kwargs", {})
        width = int(kw.get("width", 128))
        depth = int(kw.get("depth", 6))
        state_key = "model_state"
    else:
        width = int(ckpt.get("width", 128))
        depth = int(ckpt.get("depth", 6))
        state_key = "model"
    # Build on CPU first, load weights, then move to target device+dtype (handles float64→float32)
    sdf_net = _SDFNetSchwarz(width=width, depth=depth)
    sdf_net.load_state_dict(ckpt[state_key])
    sdf_net = sdf_net.to(device=device, dtype=dtype)
    sdf_net.eval()
    sdf_net.requires_grad_(False)
    center = torch.tensor(ckpt.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
    scale  = float(ckpt.get("scale", 1.0))
    return sdf_net, center, scale


# ---------------------------------------------------------------------------
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


def load_chart_id_set(path: Optional[str], n_charts: int, key: str = "chart_ids") -> Optional[set]:
    if path is None or (not os.path.isfile(path)):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vals: List[int] = []
    if isinstance(payload, list):
        vals = [int(x) for x in payload]
    elif isinstance(payload, dict):
        if isinstance(payload.get(key), list):
            vals = [int(x) for x in payload.get(key, [])]
        elif isinstance(payload.get("chart_ids"), list):
            vals = [int(x) for x in payload.get("chart_ids", [])]
        elif isinstance(payload.get("ids"), list):
            vals = [int(x) for x in payload.get("ids", [])]
        elif isinstance(payload.get("trainable_charts"), list):
            vals = [int(x) for x in payload.get("trainable_charts", [])]
        elif isinstance(payload.get("freeze_charts"), list):
            vals = [int(x) for x in payload.get("freeze_charts", [])]
    out = {i for i in vals if 0 <= i < n_charts}
    return out


def load_new_parent_map(path: Optional[str], n_charts: int) -> Optional[List[int]]:
    if path is None or (not os.path.isfile(path)):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    arr = payload.get("new_parent")
    if not isinstance(arr, list):
        return None
    out = [int(x) for x in arr]
    if len(out) != n_charts:
        return None
    return out


def blend_model_with_old(model: torch.nn.Module, old_state: Dict[str, torch.Tensor], omega: float) -> None:
    new_state = model.state_dict()
    blended = {}
    for k, v in new_state.items():
        old_v = old_state[k]
        blended[k] = (1.0 - omega) * old_v + omega * v
    model.load_state_dict(blended)


def average_state_dicts(states_list: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(states_list) == 0:
        return {}
    out: Dict[str, torch.Tensor] = {}
    keys = list(states_list[0].keys())
    inv_n = 1.0 / float(len(states_list))
    for k in keys:
        acc = states_list[0][k].detach().clone()
        for s in states_list[1:]:
            acc = acc + s[k]
        out[k] = acc * inv_n
    return out


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
    iters = np.arange(1, len(history["global_residual"]) + 1, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogy(iters, np.maximum(history["global_residual"], 1e-16), label="global_residual")
    ax.semilogy(iters, np.maximum(history["interface_value"], 1e-16), label="interface_value")
    ax.semilogy(iters, np.maximum(history["interface_flux"], 1e-16), label="interface_flux")
    ax.semilogy(iters, np.maximum(history["bc_loss"], 1e-16), label="bc_loss")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Loss-like metric")
    ax.set_title("Primary metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if len(history.get("rel_l2_eval", [])) > 0:
        ax.semilogy(iters, np.maximum(history["rel_l2_eval"], 1e-16), label="rel_l2_eval")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Relative L2")
    ax.set_title("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if len(history.get("w_interface_flux_eff", [])) > 0:
        ax.plot(iters, history["w_interface_flux_eff"], label="w_interface_flux_eff")
    if len(history.get("adaptive_flux_multiplier", [])) > 0:
        ax.plot(iters, history["adaptive_flux_multiplier"], label="adaptive_flux_multiplier")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Weight")
    ax.set_title("Flux schedule")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if len(history.get("lr", [])) > 0:
        ax.plot(iters, history["lr"], label="lr")
    if len(history.get("iter_rejected", [])) > 0:
        ax.plot(iters, history["iter_rejected"], label="iter_rejected")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Control")
    ax.set_title("Trust-region control")
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
    # Always deserialize on CPU first so float64 checkpoints can be loaded on MPS.
    ckpt = torch.load(atlas_checkpoint, map_location=torch.device("cpu"))

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
    run_start = time.time()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    if dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    elif args.tf32:
        print("TF32 requested but CUDA is unavailable; ignoring --tf32.")

    # AMP support:
    # - CUDA: autocast + GradScaler
    # - MPS: autocast only (GradScaler is CUDA-specific)
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
    use_amp = bool(use_cuda_amp or use_mps_amp)
    if args.amp and not use_amp:
        if device.type == "mps":
            print(
                "AMP requested on MPS but unsupported by this PyTorch/MPS setup; "
                "continuing without AMP."
            )
        else:
            print(
                "AMP requested but unavailable for this device/dtype; "
                "continuing without AMP."
            )

    if device.type == "mps" and (
        bool(args.curvature_adaptive_sampling) or bool(args.curvature_adaptive_interface_sampling)
    ):
        print(
            "Disabling curvature-adaptive sampling on MPS due backend instability in weighted sampling/index ops."
        )
        args.curvature_adaptive_sampling = False
        args.curvature_adaptive_interface_sampling = False
    if device.type == "mps" and args.interface_normal_mode == "mask_levelset":
        print("Switching interface normal mode to 'seed' on MPS for stability/speed.")
        args.interface_normal_mode = "seed"

    use_cuda_stream_parallel = bool(
        args.parallel_color_updates and device.type == "cuda" and torch.cuda.device_count() > 0
    )
    if args.parallel_color_updates and not use_cuda_stream_parallel:
        print(
            "parallel_color_updates requested but CUDA streams unavailable on this host; "
            "falling back to sequential color sweeps."
        )

    amp_backend = "cuda" if use_cuda_amp else ("mps" if use_mps_amp else "none")
    print(
        f"Device={device.type} dtype={dtype} amp={use_amp} amp_backend={amp_backend} "
        f"tf32={bool(args.tf32 and device.type == 'cuda')} parallel_color_updates={use_cuda_stream_parallel}"
    )
    if device.type == "mps":
        print(
            "MPS backend selected. For unsupported ops fallback, run with "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 in your shell."
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

    trainable_include = load_chart_id_set(args.trainable_charts_json, n_charts=n_charts, key="trainable_charts")
    freeze_set = load_chart_id_set(args.freeze_charts_json, n_charts=n_charts, key="freeze_charts")
    trainable_chart_mask: List[bool] = []
    for i in range(n_charts):
        has_points = point_idx_by_chart[i].numel() > 0
        if trainable_include is not None:
            trainable = has_points and (i in trainable_include)
        elif args.chart_train_mode == "all":
            trainable = has_points
        else:
            trainable = has_points and (len(neighbors[i]) >= args.overlap_min_neighbors)
        if freeze_set is not None and i in freeze_set:
            trainable = False
        trainable_chart_mask.append(bool(trainable))
    n_trainable = int(sum(1 for x in trainable_chart_mask if x))
    print(f"Chart train mode={args.chart_train_mode} trainable_charts={n_trainable}/{n_charts}")
    if trainable_include is not None:
        print(f"  explicit trainable chart list size={len(trainable_include)}")
    if freeze_set is not None:
        print(f"  explicit frozen chart list size={len(freeze_set)}")
    if n_trainable <= 0:
        raise RuntimeError(
            "No trainable charts after applying chart train/freeze masks. "
            "Check --trainable-charts-json/--freeze-charts-json contents; "
            "supported keys include chart_ids, ids, trainable_charts, freeze_charts."
        )

    # Curvature/detail proxy for adaptive sampling and chart refinement.
    # This emphasizes high-normal-deviation + extremity regions (ears/paws-like zones).
    primary_chart = torch.argmax(membership, dim=1)
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
        detail_cut = torch.quantile(detail_score.detach().to(device=torch.device("cpu")), q_detail).to(
            device=device, dtype=dtype
        )
    high_detail_mask = detail_score >= detail_cut
    n_high_detail_points = int(torch.sum(high_detail_mask).item())

    chart_detail_strength: List[float] = []
    chart_high_detail_fraction: List[float] = []
    detail_idx_by_chart: List[torch.Tensor] = []
    chart_sampling_weights: List[torch.Tensor] = []
    detail_sampling_weights: List[torch.Tensor] = []
    for i in range(n_charts):
        idx = point_idx_by_chart[i]
        if idx.numel() == 0:
            chart_detail_strength.append(0.0)
            chart_high_detail_fraction.append(0.0)
            detail_idx_by_chart.append(idx)
            chart_sampling_weights.append(torch.zeros((0,), device=device, dtype=dtype))
            detail_sampling_weights.append(torch.zeros((0,), device=device, dtype=dtype))
            continue
        s = detail_score[idx]
        hi = high_detail_mask[idx]
        chart_detail_strength.append(float(torch.mean(s).item()))
        chart_high_detail_fraction.append(float(torch.mean(hi.to(dtype=dtype)).item()))

        # Weighting for adaptive sampler.
        if args.curvature_adaptive_sampling:
            w_all = 1.0 + float(args.curvature_sample_weight) * s
            w_all = w_all + float(args.curvature_sample_weight) * 0.5 * hi.to(dtype=dtype)
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
        detail_order = np.argsort(np.asarray(chart_detail_strength))[::-1]
        picked = 0
        for chart_i in detail_order:
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
            overlap_detail_strength[key] = 0.0
            overlap_sampling_weights[key] = torch.zeros((0,), device=device, dtype=dtype)
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
        detail_charts = [i for i, flag in enumerate(detail_chart_mask) if flag]
        print(
            f"Curvature-adaptive sampling enabled | high-detail points={n_high_detail_points}/{n_points} "
            f"(q={q_detail:.2f}) | detail charts={detail_charts}"
        )

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
    # M6: volumetric atlas uses seed-direction normals — skip mask_levelset precomputation
    if (args.interface_flux_mode == "projected"
            and args.interface_normal_mode == "mask_levelset"
            and not getattr(args, "volumetric_atlas", False)):
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
        LocalPoissonPINN(
            width=args.pinn_width,
            depth=args.pinn_depth,
            arch=args.pinn_arch,
            residual_scale=args.pinn_residual_scale,
            residual_zero_init=bool(args.pinn_residual_zero_init),
        ).to(device=device, dtype=dtype)
        for _ in range(n_charts)
    ]
    if args.init_u_checkpoint is not None:
        # Keep checkpoint deserialization on CPU to avoid MPS float64 load failures.
        init_ckpt = torch.load(args.init_u_checkpoint, map_location=torch.device("cpu"))
        init_states = init_ckpt.get("u_states")
        init_u_kwargs = init_ckpt.get("u_kwargs", {})
        if isinstance(init_u_kwargs, dict):
            init_arch = str(init_u_kwargs.get("arch", "mlp")).lower().strip()
            if init_arch != str(args.pinn_arch).lower().strip():
                print(
                    "Init checkpoint u-arch differs from requested architecture; "
                    f"loading with strict=False (init={init_arch}, requested={args.pinn_arch})."
                )
        if not isinstance(init_states, list):
            raise RuntimeError(
                f"Invalid u_states in init checkpoint: {args.init_u_checkpoint}. "
                "Expected list of chart states."
            )
        if len(init_states) == n_charts:
            for i in range(n_charts):
                u_nets[i].load_state_dict(init_states[i], strict=False)
        else:
            new_parent = load_new_parent_map(args.u_remap_json, n_charts=n_charts)
            if new_parent is None:
                raise RuntimeError(
                    f"Init checkpoint chart-count mismatch: got {len(init_states)} states, expected {n_charts}. "
                    "Provide --u-remap-json with new_parent mapping for split-child warmstart."
                )
            copied = 0
            for i in range(n_charts):
                src = int(new_parent[i])
                if 0 <= src < len(init_states):
                    u_nets[i].load_state_dict(init_states[src], strict=False)
                    copied += 1
            print(f"Loaded remapped initial chart PINNs: copied {copied}/{n_charts} from {len(init_states)} source states")
        print(f"Loaded initial chart PINNs from: {args.init_u_checkpoint}")

    # M2: load optional SDF network for interior sampling / M7 hard-BC
    _sdf_net: Optional[torch.nn.Module] = None
    _sdf_center: Optional[torch.Tensor] = None
    _sdf_scale: float = 1.0
    if getattr(args, "sdf_checkpoint", None):
        try:
            _sdf_net, _sdf_center, _sdf_scale = load_sdf_for_schwarz(
                args.sdf_checkpoint, device=device, dtype=dtype
            )
            print(f"Loaded SDF network from: {args.sdf_checkpoint}")
        except Exception as _sdf_err:
            print(f"Warning: failed to load SDF checkpoint ({_sdf_err}). Disabling SDF features.")

    opts = [torch.optim.Adam(u.parameters(), lr=args.lr) for u in u_nets]
    if use_cuda_amp:
        scalers: List[Optional[torch.cuda.amp.GradScaler]] = [torch.cuda.amp.GradScaler(enabled=True) for _ in range(n_charts)]
    else:
        scalers = [None for _ in range(n_charts)]

    def amp_ctx() -> contextlib.AbstractContextManager:
        if use_cuda_amp:
            return torch.cuda.amp.autocast()
        if use_mps_amp:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return contextlib.nullcontext()
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
        "adaptive_flux_multiplier": [],
        "lr": [],
        "iter_rejected": [],
        "accepted_progress_iter": [],
    }

    def sample_chart_point_indices(i: int, n_samples: int, prefer_detail: bool = True) -> Optional[torch.Tensor]:
        idx = point_idx_by_chart[i]
        if idx.numel() == 0 or n_samples <= 0:
            return None
        n_take = int(n_samples)
        if device.type == "mps":
            # MPS-safe path: avoid weighted sampling/index kernels that can fail nondeterministically.
            sel = torch.randint(0, idx.numel(), (n_take,), device=device)
            return idx[sel]
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
        if len(picks) == 1:
            return picks[0]
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

    # M2: SDF-guided interior sampling -----------------------------------------
    def sample_interior_xi_sdf(i: int, n_samples: int) -> torch.Tensor:
        """Sample PDE collocation points inside the domain using SDF rejection.

        Candidates are drawn uniformly in the chart bounding cube, mapped to
        physical coordinates via the rigid TNB frame, then accepted only when
        SDF < sdf_interior_threshold (i.e. inside the domain). Falls back to
        surface-noise sampling when acceptance rate is too low.
        """
        # M6: also activate when --volumetric-atlas is set
        _want_sdf = getattr(args, "use_sdf_sampling", False) or getattr(args, "volumetric_atlas", False)
        if _sdf_net is None or not _want_sdf:
            return sample_local_xi(i, n_samples)
        r = float(support_r[i]) * 1.0
        factor = max(1, int(getattr(args, "sdf_rejection_factor", 6)))
        thresh = float(getattr(args, "sdf_interior_threshold", 0.0))
        n_cand = n_samples * factor
        seed_i = seeds[i]
        t1_i   = t1[i]
        t2_i   = t2[i]
        n_i    = nvec[i]
        xi_cand = (2.0 * torch.rand(n_cand, 3, device=device, dtype=dtype) - 1.0) * r
        x_cand = (
            seed_i.unsqueeze(0)
            + xi_cand[:, 0:1] * t1_i.unsqueeze(0)
            + xi_cand[:, 1:2] * t2_i.unsqueeze(0)
            + xi_cand[:, 2:3] * n_i.unsqueeze(0)
        )
        with torch.no_grad():
            x_norm = (x_cand - _sdf_center.unsqueeze(0)) / _sdf_scale
            sdf_vals = _sdf_net(x_norm)
            inside = sdf_vals < thresh
        if inside.sum() >= n_samples:
            xi_ok = xi_cand[inside][:n_samples]
            return xi_ok
        # fallback: surface-noise sampling
        return sample_local_xi(i, n_samples)

    def sample_pde_xi(i: int, n_samples: int) -> torch.Tensor:
        """Choose PDE sampling strategy: SDF interior or surface-noise."""
        # M6: --volumetric-atlas forces SDF interior sampling
        use_sdf = (
            getattr(args, "use_sdf_sampling", False)
            or getattr(args, "volumetric_atlas", False)
        )
        if _sdf_net is not None and use_sdf:
            return sample_interior_xi_sdf(i, n_samples)
        return sample_local_xi(i, n_samples)

    # M8: Residual-based Adaptive Refinement (RAR) pools ----------------------
    rar_pool: List[Optional[torch.Tensor]] = [None] * n_charts

    def maybe_update_rar_pools(it: int, w_pde_eff: float) -> None:
        """Every rar_period iters, refill each chart's RAR pool with top-k residual points."""
        rar_period = int(getattr(args, "rar_period", 0))
        if rar_period <= 0 or it % rar_period != 0:
            return
        n_cand  = int(getattr(args, "rar_candidates", 512))
        top_k   = int(getattr(args, "rar_top_k", 64))
        pool_max = int(getattr(args, "rar_pool_max", 256))
        direct  = bool(getattr(args, "direct_coord_pde", False))
        for i in range(n_charts):
            if not trainable_chart_mask[i]:
                continue
            xi_cand = sample_pde_xi(i, n_cand)
            with torch.no_grad():
                if direct:
                    x_phys = (
                        seeds[i].unsqueeze(0)
                        + xi_cand[:, 0:1] * t1[i].unsqueeze(0)
                        + xi_cand[:, 1:2] * t2[i].unsqueeze(0)
                        + xi_cand[:, 2:3] * nvec[i].unsqueeze(0)
                    )
                    res, _, _ = direct_poisson_residual_tnb(
                        u_nets[i], x_phys, seeds[i], t1[i], t2[i], nvec[i]
                    )
                else:
                    res, _, _ = mapped_poisson_residual(
                        u_nets[i], decoders[i], xi_cand,
                        seeds[i], t1[i], t2[i], nvec[i],
                        support_r[i],
                        sigma_floor=args.sigma_floor,
                        det_floor=args.det_floor,
                        jac_kappa_max=args.jac_kappa_max,
                    )
                scores = res.squeeze(-1).abs()
            _, top_idx = torch.topk(scores, min(top_k, scores.numel()))
            new_pts = xi_cand[top_idx].detach()
            if rar_pool[i] is not None:
                combined = torch.cat([rar_pool[i], new_pts], dim=0)
                if combined.shape[0] > pool_max:
                    perm = torch.randperm(combined.shape[0], device=device)
                    combined = combined[perm[:pool_max]]
                rar_pool[i] = combined
            else:
                rar_pool[i] = new_pts

    def sample_pde_xi_with_rar(i: int, n_samples: int) -> torch.Tensor:
        """Mix n_samples from sample_pde_xi with up to rar_mix_n RAR-pool points."""
        mix_n = int(getattr(args, "rar_mix_n", 32))
        base = sample_pde_xi(i, n_samples)
        if rar_pool[i] is None or mix_n <= 0:
            return base
        pool = rar_pool[i]
        n_mix = min(mix_n, pool.shape[0])
        perm = torch.randperm(pool.shape[0], device=device)[:n_mix]
        return torch.cat([base, pool[perm]], dim=0)

    # M6: Volumetric interface sampling ----------------------------------------
    def sample_interface_volumetric(
        i: int,
        j: int,
        n_samples: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """M6: Sample interface points from the 3-D intersection of chart balls.

        Generates candidate points around the midpoint of seeds[i] and seeds[j],
        accepts those inside BOTH support balls (and optionally inside the SDF
        domain), then converts to per-chart local coordinates.  Falls back to a
        seed-segment fan when the geometric intersection is too sparse.
        """
        key = (i, j) if i < j else (j, i)
        # Check whether i and j are actually neighbours
        if j not in neighbors[i]:
            return None
        r_i = float(support_r[i])
        r_j = float(support_r[j])
        s_i = seeds[i]
        s_j = seeds[j]
        midpt = 0.5 * (s_i + s_j)
        r_samp = 0.5 * min(r_i, r_j)
        factor = max(4, int(getattr(args, "sdf_rejection_factor", 6)))
        n_cand = n_samples * factor

        # Uniform-in-ball sampling around midpoint (Muller method)
        u = torch.randn(n_cand, 3, device=device, dtype=dtype)
        u_norm = torch.linalg.norm(u, dim=1, keepdim=True)
        u = u / torch.clamp(u_norm, min=1e-12)
        r_rand = r_samp * torch.rand(n_cand, 1, device=device, dtype=dtype) ** (1.0 / 3.0)
        x_cand = midpt.unsqueeze(0) + u * r_rand

        with torch.no_grad():
            d_i = torch.linalg.norm(x_cand - s_i.unsqueeze(0), dim=1)
            d_j = torch.linalg.norm(x_cand - s_j.unsqueeze(0), dim=1)
            mask = (d_i <= r_i) & (d_j <= r_j)
            if _sdf_net is not None:
                x_norm_cand = (x_cand - _sdf_center.unsqueeze(0)) / _sdf_scale
                sdf_vals = _sdf_net(x_norm_cand)
                mask = mask & (sdf_vals < float(getattr(args, "sdf_interior_threshold", 0.0)))

        x_ok = x_cand[mask]
        if x_ok.shape[0] >= n_samples:
            perm = torch.randperm(x_ok.shape[0], device=device)[:n_samples]
            x_if = x_ok[perm]
        elif x_ok.shape[0] > 0:
            # Repeat with small noise to reach n_samples
            reps = math.ceil(n_samples / x_ok.shape[0])
            rep = x_ok.repeat(reps, 1)[:n_samples]
            x_if = rep + 0.02 * r_samp * torch.randn_like(rep)
        else:
            # Fallback: linear segment between seeds with noise
            tv = torch.linspace(0.2, 0.8, n_samples, device=device, dtype=dtype).unsqueeze(1)
            x_if = s_i.unsqueeze(0) + tv * (s_j - s_i).unsqueeze(0)
            x_if = x_if + 0.05 * r_samp * torch.randn_like(x_if)

        x_if = x_if.detach()
        xi_i = local_coords(x_if, seeds[i], t1[i], t2[i], nvec[i])
        xi_j = local_coords(x_if, seeds[j], t1[j], t2[j], nvec[j])

        # Interface normal = direction from seed i to seed j (seed-direction mode)
        sd = s_j - s_i
        sd_len = float(torch.linalg.norm(sd).item())
        n_if = (sd / max(sd_len, 1e-12)).unsqueeze(0).expand(x_if.shape[0], -1).detach()

        return x_if, xi_i, xi_j, n_if

    # --------------------------------------------------------------------------
    def _sample_bc_surface_sdf(i: int, n_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """M6/volumetric: sample points near the domain surface (|SDF| < eps_surface)
        within chart i's support ball.  Returns (xi, target, x_phys) or None on failure.
        """
        if _sdf_net is None:
            return None
        r_i = float(support_r[i])
        # Use higher rejection factor: surface is a thin shell
        factor = max(12, int(getattr(args, "sdf_rejection_factor", 6)) * 3)
        n_cand = n_samples * factor

        # Uniform-in-ball sampling around seed i
        u_vec = torch.randn(n_cand, 3, device=device, dtype=dtype)
        u_vec = u_vec / torch.clamp(torch.linalg.norm(u_vec, dim=1, keepdim=True), min=1e-12)
        r_rand = r_i * torch.rand(n_cand, 1, device=device, dtype=dtype) ** (1.0 / 3.0)
        x_cand = seeds[i].unsqueeze(0) + u_vec * r_rand  # (n_cand, 3) in atlas-normalized coords

        with torch.no_grad():
            x_norm_cand = (x_cand - _sdf_center.unsqueeze(0)) / _sdf_scale
            sdf_vals = _sdf_net(x_norm_cand)  # (n_cand,)
            # Accept: near surface |SDF| < eps_surface (SDF=0 on surface)
            eps_surface = float(getattr(args, "bc_surface_eps", 0.04))
            mask = sdf_vals.abs() < eps_surface
            x_ok = x_cand[mask]

        if x_ok.shape[0] == 0:
            return None  # caller will fall back to interior supervision

        if x_ok.shape[0] >= n_samples:
            x_bc = x_ok[torch.randperm(x_ok.shape[0], device=device)[:n_samples]]
        else:
            rep = x_ok.repeat(math.ceil(n_samples / x_ok.shape[0]), 1)[:n_samples]
            x_bc = rep + 0.005 * r_i * torch.randn_like(rep)

        xi_bc = local_coords(x_bc, seeds[i], t1[i], t2[i], nvec[i]).detach()
        target_bc = manufactured_u(x_bc).detach()
        return xi_bc, target_bc, x_bc.detach()

    # --------------------------------------------------------------------------
    def local_bc_batch(i: int, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # M6: in volumetric mode, BC points should be on the domain surface (SDF ≈ 0),
        # not interior atlas points. Use SDF rejection sampling near the surface.
        if getattr(args, "volumetric_atlas", False):
            result = _sample_bc_surface_sdf(i, n_samples)
            if result is not None:
                return result
            # Deep interior chart: no surface contact → return EMPTY tensors so callers
            # produce zero BC loss for this chart (correct: interior charts don't touch ∂Ω).
            empty_xi = torch.zeros((0, 3), device=device, dtype=dtype)
            empty_u  = torch.zeros((0, 1), device=device, dtype=dtype)
            return empty_xi, empty_u, empty_xi

        pick = sample_chart_point_indices(i, n_samples=n_samples, prefer_detail=True)
        if pick is None:
            z = torch.zeros((n_samples, 3), device=device, dtype=dtype)
            return z, torch.zeros((n_samples, 1), device=device, dtype=dtype), z
        x = points[pick]
        xi = local_coords(x, seeds[i], t1[i], t2[i], nvec[i])
        target = manufactured_u(x).detach()
        return xi, target, x

    def interface_batch(
        i: int,
        j: int,
        n_samples: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # M6: volumetric atlas uses 3-D ball-intersection interface sampling
        if getattr(args, "volumetric_atlas", False):
            return sample_interface_volumetric(i, j, n_samples)
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
            if args.skip_pde_if_no_valid:
                # Keep shape non-empty and differentiable while effectively disabling PDE term for this batch.
                return residual[:1] * 0.0
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
                    # M6: use SDF-guided interior sampling for eval in volumetric mode
                    xi_int = sample_pde_xi(i, ni)
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
                res_eval = select_residual_samples(res, valid, with_clip=True)
                pde_terms.append(torch.mean(res_eval**2))

                if args.eval_fixed_cache and i in eval_cache_bc:
                    xi_bc, u_bc = eval_cache_bc[i]
                else:
                    xi_bc, u_bc, _ = local_bc_batch(i, max(16, args.eval_bc_samples_per_chart))
                # Guard: interior charts return empty tensors — skip their BC term
                if xi_bc.shape[0] > 0:
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

    def pre_snapshot_u_states() -> List[Dict[str, torch.Tensor]]:
        return [copy_state_dict(u.state_dict()) for u in u_nets]

    def pre_load_snapshot_u_states(states: List[Dict[str, torch.Tensor]]) -> None:
        for i in range(n_charts):
            u_nets[i].load_state_dict(states[i])

    def pre_current_lr() -> float:
        if not opts or not opts[0].param_groups:
            return 0.0
        return float(opts[0].param_groups[0]["lr"])

    def pre_set_lr(new_lr: float) -> None:
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = float(new_lr)

    swa_candidates: List[Dict[str, object]] = []

    def maybe_add_swa_candidate(rel_l2_eval: float, label: str, states: Optional[List[Dict[str, torch.Tensor]]] = None) -> None:
        topk = max(0, int(args.swa_topk))
        if topk <= 0:
            return
        use_states = states if states is not None else pre_snapshot_u_states()
        swa_candidates.append(
            {
                "rel_l2_eval": float(rel_l2_eval),
                "label": str(label),
                "u_states": use_states,
            }
        )
        swa_candidates.sort(key=lambda x: float(x["rel_l2_eval"]))
        if len(swa_candidates) > topk:
            del swa_candidates[topk:]

    pretrain_guard_events: List[Dict[str, object]] = []

    def run_pretrain_guard(stage: str, ep: int) -> None:
        if not bool(args.pretrain_guard_enable):
            return
        eval_every = max(1, int(args.pretrain_guard_eval_every))
        if (ep % eval_every) != 0 and ep != 1:
            return
        nonlocal pretrain_best_rel_l2, pretrain_best_states, pretrain_guard_bad_count
        rel_eval = eval_rel_l2_subset()
        maybe_add_swa_candidate(rel_eval, label=f"{stage}_ep{ep}")
        if rel_eval + 1e-14 < pretrain_best_rel_l2:
            pretrain_best_rel_l2 = rel_eval
            pretrain_best_states = pre_snapshot_u_states()
            pretrain_guard_bad_count = 0
            pretrain_guard_events.append(
                {
                    "stage": stage,
                    "epoch": int(ep),
                    "action": "best_update",
                    "rel_l2_eval": float(rel_eval),
                    "lr": float(pre_current_lr()),
                }
            )
            return
        if rel_eval > pretrain_best_rel_l2 + float(args.pretrain_guard_rel_l2_margin):
            pretrain_guard_bad_count += 1
        else:
            pretrain_guard_bad_count = 0

        if pretrain_guard_bad_count >= max(1, int(args.pretrain_guard_patience)):
            if pretrain_best_states is not None:
                pre_load_snapshot_u_states(pretrain_best_states)
            new_lr = max(float(args.pretrain_guard_min_lr), pre_current_lr() * float(args.pretrain_guard_lr_decay))
            pre_set_lr(new_lr)
            pretrain_guard_events.append(
                {
                    "stage": stage,
                    "epoch": int(ep),
                    "action": "restore_and_decay",
                    "rel_l2_eval": float(rel_eval),
                    "best_rel_l2": float(pretrain_best_rel_l2),
                    "lr": float(new_lr),
                }
            )
            print(
                f"[PretrainGuard] stage={stage} epoch={ep} restore best_rel_l2={pretrain_best_rel_l2:.3e}; "
                f"rel_l2_eval={rel_eval:.3e} lr={new_lr:.3e}"
            )
            pretrain_guard_bad_count = 0

    pretrain_best_rel_l2 = float("inf")
    pretrain_best_states: Optional[List[Dict[str, torch.Tensor]]] = None
    pretrain_guard_bad_count = 0

    if args.bc_pretrain_epochs > 0:
        print(f"Starting BC warm-start pretraining for {args.bc_pretrain_epochs} epochs")
        for ep in range(1, args.bc_pretrain_epochs + 1):
            losses_ep = []
            for i in range(n_charts):
                if point_idx_by_chart[i].numel() == 0 or (not trainable_chart_mask[i]):
                    continue
                u_nets[i].train()
                opts[i].zero_grad()

                with amp_ctx():
                    xi_bc, u_bc, x_bc = local_bc_batch(i, args.bc_pretrain_batch)
                    # Guard: interior charts return empty tensors (no surface BC to enforce)
                    if xi_bc.shape[0] > 0:
                        u_hat = u_nets[i](xi_bc)
                        loss = torch.mean((u_hat - u_bc) ** 2)
                    else:
                        loss = torch.tensor(0.0, device=device, dtype=dtype)

                    if args.bc_pretrain_grad_weight > 0.0 and xi_bc.shape[0] > 0:
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

                    # Joint interior supervised pretrain: fit manufactured_u at interior SDF points
                    # simultaneously with BC so neither overwrites the other.
                    bc_pretrain_sup_weight = float(getattr(args, "bc_pretrain_sup_weight", 0.0))
                    if bc_pretrain_sup_weight > 0.0:
                        xi_sup_p = sample_pde_xi(i, int(args.bc_pretrain_batch))
                        x_sup_p = decoders[i](
                            xi_sup_p,
                            seed=seeds[i],
                            t1=t1[i],
                            t2=t2[i],
                            n=nvec[i],
                            chart_scale=support_r[i],
                        )
                        u_sup_p = manufactured_u(x_sup_p).detach()
                        loss_sup_p = torch.mean((u_nets[i](xi_sup_p) - u_sup_p) ** 2)
                        loss = loss + bc_pretrain_sup_weight * loss_sup_p

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

                # Skip backward if loss is zero (e.g. interior charts with no BC samples)
                if loss.item() != 0.0:
                    if use_cuda_amp:
                        assert scalers[i] is not None
                        scalers[i].scale(loss).backward()
                        scalers[i].unscale_(opts[i])
                        torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
                        scalers[i].step(opts[i])
                        scalers[i].update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
                        opts[i].step()
                losses_ep.append(float(loss.item()))

            if ep % max(1, args.bc_pretrain_log_every) == 0 and losses_ep:
                print(
                    f"[Pretrain] epoch={ep}/{args.bc_pretrain_epochs} "
                    f"loss={np.mean(losses_ep):.3e}"
                )
            run_pretrain_guard("bc", ep)

    if args.interior_pretrain_epochs > 0:
        print(f"Starting interior supervised pretraining for {args.interior_pretrain_epochs} epochs")
        for ep in range(1, args.interior_pretrain_epochs + 1):
            losses_ep = []
            for i in range(n_charts):
                if point_idx_by_chart[i].numel() == 0 or (not trainable_chart_mask[i]):
                    continue

                u_nets[i].train()
                opts[i].zero_grad()

                with amp_ctx():
                    # M6: use SDF-guided interior sampling in volumetric mode
                    xi_sup = sample_pde_xi(i, args.interior_pretrain_batch)
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

                if use_cuda_amp:
                    assert scalers[i] is not None
                    scalers[i].scale(loss).backward()
                    scalers[i].unscale_(opts[i])
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
                    scalers[i].step(opts[i])
                    scalers[i].update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
                    opts[i].step()
                losses_ep.append(float(loss.item()))

            if ep % max(1, args.interior_pretrain_log_every) == 0 and losses_ep:
                print(
                    f"[InteriorPre] epoch={ep}/{args.interior_pretrain_epochs} "
                    f"loss={np.mean(losses_ep):.3e}"
                )
            run_pretrain_guard("interior", ep)

    if bool(args.pretrain_guard_enable) and bool(args.pretrain_guard_restore_best_end) and pretrain_best_states is not None:
        pre_load_snapshot_u_states(pretrain_best_states)
        print(
            f"[PretrainGuard] restored best pretrain state at end with rel_l2_eval={pretrain_best_rel_l2:.3e} "
            f"and lr={pre_current_lr():.3e}"
        )

    def optimize_chart(i: int, w_pde_eff: float, w_if_flux_eff: float) -> None:
        if point_idx_by_chart[i].numel() == 0 or (not trainable_chart_mask[i]):
            return
        u_nets[i].train()
        old_state = copy_state_dict(u_nets[i].state_dict())
        chart_is_detail = bool(detail_chart_mask[i]) if args.curvature_adaptive_sampling else False
        step_boost = float(args.detail_chart_step_boost) if chart_is_detail else 1.0
        batch_boost = float(args.detail_chart_batch_boost) if chart_is_detail else 1.0
        local_steps_i = max(1, int(round(args.local_steps * max(1.0, step_boost))))
        pde_batch_i = max(8, int(round(args.pde_batch * max(1.0, batch_boost))))
        bc_batch_i = max(8, int(round(args.bc_batch * max(1.0, batch_boost))))
        if_batch_i = max(8, int(round(args.if_batch * max(1.0, batch_boost))))
        sup_batch_i = max(8, int(round(args.manufactured_supervision_batch * max(1.0, batch_boost))))

        for _ in range(local_steps_i):
            opts[i].zero_grad()

            with amp_ctx():
                # M1/M8: Use RAR-augmented sampling; route through direct or mapped residual.
                # Guard: skip expensive double-autodiff when PDE weight is zero.
                if w_pde_eff > 0.0:
                    xi_int = sample_pde_xi_with_rar(i, pde_batch_i)
                    _direct = bool(getattr(args, "direct_coord_pde", False))
                    if _direct:
                        x_phys_int = (
                            seeds[i].unsqueeze(0)
                            + xi_int[:, 0:1] * t1[i].unsqueeze(0)
                            + xi_int[:, 1:2] * t2[i].unsqueeze(0)
                            + xi_int[:, 2:3] * nvec[i].unsqueeze(0)
                        )
                        res, _, valid = direct_poisson_residual_tnb(
                            u_nets[i], x_phys_int, seeds[i], t1[i], t2[i], nvec[i]
                        )
                    else:
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
                else:
                    loss_pde = torch.tensor(0.0, device=device, dtype=dtype)

                xi_bc, u_bc, x_bc = local_bc_batch(i, bc_batch_i)
                # Guard: interior charts (volumetric mode) may return empty BC tensors
                if xi_bc.shape[0] > 0:
                    u_hat_bc = u_nets[i](xi_bc)
                    # M7: weight BC loss by SDF depth so near-boundary points dominate less
                    if getattr(args, "hard_bc", False) and _sdf_net is not None:
                        _bc_w = sdf_hard_bc_scale(
                            x_bc, _sdf_net, _sdf_center, _sdf_scale,
                            float(getattr(args, "hard_bc_scale", 0.05)),
                        )
                        loss_bc = torch.mean(_bc_w * (u_hat_bc - u_bc) ** 2)
                    else:
                        loss_bc = torch.mean((u_hat_bc - u_bc) ** 2)
                else:
                    loss_bc = torch.tensor(0.0, device=device, dtype=dtype)

                loss_sup = torch.tensor(0.0, device=device, dtype=dtype)
                loss_sup_grad = torch.tensor(0.0, device=device, dtype=dtype)
                if args.w_manufactured_supervision > 0.0 or args.w_manufactured_grad_supervision > 0.0:
                    # M6: use SDF-guided interior sampling in volumetric mode
                    xi_sup = sample_pde_xi(i, sup_batch_i)
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
                _need_flux = w_if_flux_eff > 0.0
                for j in neighbors[i]:
                    ib = interface_batch(i, j, if_batch_i)
                    if ib is None:
                        continue
                    _, xi_i, xi_j, n_if = ib
                    pair_key = (i, j) if i < j else (j, i)
                    pair_boost = 1.0
                    if args.curvature_adaptive_sampling and args.interface_detail_boost > 1.0:
                        d_pair = overlap_detail_strength.get(pair_key, 0.0)
                        pair_boost = 1.0 + (float(args.interface_detail_boost) - 1.0) * float(d_pair)
                    if _need_flux:
                        # Full coupling: value + flux (requires decoder Jacobian via grad_u_in_physical)
                        liv, _, lif_train = interface_coupling_terms(i, j, xi_i, xi_j, n_if, detach_neighbor=True)
                        if_terms.append(pair_boost * lif_train)
                    else:
                        # Fast path: value-only coupling (no decoder Jacobian needed)
                        ui = u_nets[i](xi_i)
                        with torch.no_grad():
                            uj = u_nets[j](xi_j)
                        liv = torch.mean((ui - uj) ** 2)
                    iv_terms.append(pair_boost * liv)

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

            if use_cuda_amp:
                assert scalers[i] is not None
                scalers[i].scale(loss).backward()
                scalers[i].unscale_(opts[i])
                torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
                scalers[i].step(opts[i])
                scalers[i].update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(u_nets[i].parameters(), max_norm=float(args.grad_clip_max_norm))
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

    if args.w_interface_flux_start is None:
        flux_w_start = float(args.w_interface_flux)
    else:
        flux_w_start = float(args.w_interface_flux_start)
    if args.w_interface_flux_end is None:
        flux_w_end = float(args.w_interface_flux)
    else:
        flux_w_end = float(args.w_interface_flux_end)
    flux_w_min = min(flux_w_start, flux_w_end)
    flux_w_max = max(flux_w_start, flux_w_end)
    reject_lr_decay = min(1.0, max(1e-4, float(args.iter_reject_lr_decay)))
    rel_accept_margin = max(0.0, float(args.iter_accept_rel_l2_margin))
    pde_drop_override = max(0.0, float(args.iter_accept_pde_drop))
    rel_l2_hardcap = max(0.0, float(args.iter_accept_rel_l2_hardcap))

    def effective_flux_weight(it: int) -> float:
        if args.flux_ramp_iters <= 0:
            return flux_w_end
        alpha = min(1.0, float(it) / max(1.0, float(args.flux_ramp_iters)))
        return flux_w_start + (flux_w_end - flux_w_start) * alpha

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
    reject_count = 0
    accepted_iters = 0
    guard_limit = float(args.guard_rel_l2 if args.guard_rel_l2 > 0.0 else args.target_rel_l2)
    adaptive_flux_mult = 1.0

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
        states_now = snapshot_u_states()
        snapshots[name] = {
            "iter": int(it),
            "pde": float(pde_m),
            "bc": float(bc_m),
            "if_val": float(iv_m),
            "if_flux": float(if_m),
            "rel_l2_eval": float(rel_l2_eval),
            "score": float(score),
            "u_states": states_now,
            "lr": current_lr(),
        }
        maybe_add_swa_candidate(rel_l2_eval, label=f"{name}_iter{it}", states=states_now)

    schwarz_start = time.time()
    current_pde_m, _, _, _ = eval_global_metrics()
    current_rel_l2 = eval_rel_l2_subset()

    for it in range(1, args.max_schwarz_iters + 1):
        progress_it = accepted_iters + 1
        warm = min(1.0, float(progress_it) / max(1.0, float(args.pde_warmup_iters)))
        w_pde_eff = args.w_pde * warm
        flux_base = effective_flux_weight(progress_it)
        if args.adaptive_flux_weight:
            w_if_flux_eff = float(np.clip(flux_base * adaptive_flux_mult, flux_w_min, flux_w_max))
        else:
            w_if_flux_eff = flux_base

        iter_state_before = snapshot_u_states()
        iter_lr_before = current_lr()
        iter_rejected = 0

        # M8: update RAR pools every rar_period Schwarz iterations
        maybe_update_rar_pools(it, w_pde_eff)

        for group in color_groups:
            active = [i for i in group if point_idx_by_chart[i].numel() > 0 and trainable_chart_mask[i]]
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
        pde_proposed = pde_m
        rel_l2_proposed = rel_l2_eval
        rel_accept_limit = float(args.iter_accept_rel_l2)
        if rel_accept_limit > 0.0:
            rel_accept_limit = max(rel_accept_limit, current_rel_l2 + rel_accept_margin)
        should_reject = rel_accept_limit > 0.0 and rel_l2_proposed > rel_accept_limit
        if should_reject and pde_drop_override > 0.0 and rel_l2_hardcap > 0.0:
            pde_drop = (current_pde_m - pde_proposed) / max(current_pde_m, 1e-12)
            if (pde_drop >= pde_drop_override) and (rel_l2_proposed <= rel_l2_hardcap):
                should_reject = False
                print(
                    f"[TrustRegion] iter={it} accepted by PDE override: "
                    f"pde_drop={pde_drop:.3e} (thr={pde_drop_override:.3e}), "
                    f"rel_l2_eval={rel_l2_proposed:.3e} (hardcap={rel_l2_hardcap:.3e})"
                )
        if should_reject:
            load_snapshot_u_states(iter_state_before)
            new_lr = max(1e-7, iter_lr_before * reject_lr_decay)
            set_lr(new_lr)
            pde_m, bc_m, iv_m, if_m = eval_global_metrics()
            rel_l2_eval = eval_rel_l2_subset()
            iter_rejected = 1
            reject_count += 1
            print(
                f"[TrustRegion] iter={it} rejected: rel_l2_eval={rel_l2_proposed:.3e} "
                f"> {rel_accept_limit:.3e}; pde_proposed={pde_proposed:.3e} "
                f"pde_restored={pde_m:.3e}; restored state and lr={new_lr:.3e}"
            )
        current_pde_m = pde_m
        current_rel_l2 = rel_l2_eval

        if args.adaptive_flux_weight and iter_rejected == 0:
            if rel_l2_eval <= args.adaptive_flux_rel_l2_thresh:
                adaptive_flux_mult *= float(args.adaptive_flux_up)
            else:
                adaptive_flux_mult *= float(args.adaptive_flux_down)
            adaptive_flux_mult = max(1e-6, adaptive_flux_mult)

        if iter_rejected == 0:
            accepted_iters += 1

        history["global_residual"].append(pde_m)
        history["bc_loss"].append(bc_m)
        history["interface_value"].append(iv_m)
        history["interface_flux"].append(if_m)
        history["rel_l2_eval"].append(rel_l2_eval)
        history["w_interface_flux_eff"].append(w_if_flux_eff)
        history["adaptive_flux_multiplier"].append(float(adaptive_flux_mult))
        history["lr"].append(current_lr())
        history["iter_rejected"].append(float(iter_rejected))
        history["accepted_progress_iter"].append(float(accepted_iters))

        # Plateau tracking should reflect the effective training objective weights.
        # Rejected iterations restore the previous state; do not count them as stale progress.
        score = w_pde_eff * pde_m + args.w_interface_value * iv_m + w_if_flux_eff * if_m
        if iter_rejected == 0:
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
            elapsed = time.time() - schwarz_start
            print(
                f"[Schwarz] iter={it}/{args.max_schwarz_iters} "
                f"pde={pde_m:.3e} bc={bc_m:.3e} if_val={iv_m:.3e} if_flux={if_m:.3e} "
                f"rel_l2_eval={rel_l2_eval:.3e} w_if={w_if_flux_eff:.3e} "
                f"score={score:.3e} stale={stale} rejected={iter_rejected} "
                f"accepted={accepted_iters} "
                f"lr={current_lr():.3e} flux_mult={adaptive_flux_mult:.3e} t={elapsed:.1f}s"
            )

        pde_ok = (args.w_pde <= 0.0) or (pde_m <= args.residual_tol)
        converged = pde_ok and (iv_m <= args.interface_tol) and (if_m <= args.interface_flux_tol)
        if converged:
            print(f"Converged at iteration {it}")
            break

        if args.guard_patience > 0 and iter_rejected == 0:
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
        history["adaptive_flux_multiplier"].append(float(adaptive_flux_mult))
        history["lr"].append(current_lr())
        history["iter_rejected"].append(0.0)
        history["accepted_progress_iter"].append(0.0)
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

    selected_rel_before_swa = eval_rel_l2_subset()
    selected_states_before_swa = snapshot_u_states()
    swa_info: Dict[str, object] = {
        "enabled": bool(args.swa_topk > 0),
        "topk": int(max(0, args.swa_topk)),
        "n_candidates": int(len(swa_candidates)),
        "selected_rel_before_swa": float(selected_rel_before_swa),
    }
    swa_states_for_save: Optional[List[Dict[str, torch.Tensor]]] = None
    if args.swa_topk > 1 and len(swa_candidates) >= 2:
        n_use = min(int(args.swa_topk), len(swa_candidates))
        topk_rel = [float(x["rel_l2_eval"]) for x in swa_candidates[:n_use]]
        avg_states: List[Dict[str, torch.Tensor]] = []
        for i in range(n_charts):
            avg_states.append(average_state_dicts([swa_candidates[k]["u_states"][i] for k in range(n_use)]))  # type: ignore[index]
        swa_states_for_save = avg_states
        load_snapshot_u_states(avg_states)
        swa_rel_eval = eval_rel_l2_subset()
        swa_info.update(
            {
                "used_topk": int(n_use),
                "topk_rel_l2": topk_rel,
                "swa_rel_l2_eval": float(swa_rel_eval),
            }
        )
        if bool(args.swa_select_if_better) and swa_rel_eval + 1e-14 < selected_rel_before_swa:
            selected_label = "swa_topk"
            print(
                f"Selected SWA state: rel_l2_eval improved from {selected_rel_before_swa:.3e} "
                f"to {swa_rel_eval:.3e} with topk={n_use}"
            )
        else:
            load_snapshot_u_states(selected_states_before_swa)
            swa_info["kept_selected_state"] = True
            print(
                f"SWA candidate evaluated (topk={n_use}) rel_l2_eval={swa_rel_eval:.3e}; "
                f"kept selected state rel_l2_eval={selected_rel_before_swa:.3e}"
            )

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
        "pinn_arch": args.pinn_arch,
        "pinn_width": int(args.pinn_width),
        "pinn_depth": int(args.pinn_depth),
        "pinn_residual_scale": float(args.pinn_residual_scale),
        "pinn_residual_zero_init": bool(args.pinn_residual_zero_init),
        "grad_clip_max_norm": float(args.grad_clip_max_norm),
        "amp_used": bool(use_amp),
        "amp_backend": amp_backend,
        "tf32_used": bool(args.tf32 and device.type == "cuda"),
        "parallel_color_updates_used": bool(use_cuda_stream_parallel),
        "interface_transmission_mode": args.interface_transmission_mode,
        "robin_lambda": float(args.robin_lambda),
        "interface_flux_mode": args.interface_flux_mode,
        "interface_normal_mode": args.interface_normal_mode,
        "curvature_adaptive_sampling": bool(args.curvature_adaptive_sampling),
        "curvature_adaptive_interface_sampling": bool(args.curvature_adaptive_interface_sampling),
        "detail_quantile": float(q_detail),
        "n_high_detail_points": int(n_high_detail_points),
        "high_detail_fraction": float(n_high_detail_points / max(1, n_points)),
        "detail_chart_ids": [int(i) for i, flag in enumerate(detail_chart_mask) if flag],
        "chart_detail_strength": [float(v) for v in chart_detail_strength],
        "chart_high_detail_fraction": [float(v) for v in chart_high_detail_fraction],
        "detail_chart_step_boost": float(args.detail_chart_step_boost),
        "detail_chart_batch_boost": float(args.detail_chart_batch_boost),
        "interface_detail_boost": float(args.interface_detail_boost),
        "chart_train_mode": args.chart_train_mode,
        "overlap_min_neighbors": int(args.overlap_min_neighbors),
        "n_trainable_charts": int(n_trainable),
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
        "runtime_seconds": float(time.time() - run_start),
        "schwarz_runtime_seconds": float(time.time() - schwarz_start),
        "trust_region": {
            "iter_accept_rel_l2": float(args.iter_accept_rel_l2),
            "iter_accept_rel_l2_margin": float(args.iter_accept_rel_l2_margin),
            "iter_accept_pde_drop": float(args.iter_accept_pde_drop),
            "iter_accept_rel_l2_hardcap": float(args.iter_accept_rel_l2_hardcap),
            "iter_reject_lr_decay": float(reject_lr_decay),
            "rejected_iterations": int(reject_count),
        },
        "adaptive_flux": {
            "enabled": bool(args.adaptive_flux_weight),
            "up": float(args.adaptive_flux_up),
            "down": float(args.adaptive_flux_down),
            "rel_l2_thresh": float(args.adaptive_flux_rel_l2_thresh),
            "start": float(flux_w_start),
            "end": float(flux_w_end),
        },
        "pretrain_guard": {
            "enabled": bool(args.pretrain_guard_enable),
            "eval_every": int(args.pretrain_guard_eval_every),
            "patience": int(args.pretrain_guard_patience),
            "rel_l2_margin": float(args.pretrain_guard_rel_l2_margin),
            "lr_decay": float(args.pretrain_guard_lr_decay),
            "min_lr": float(args.pretrain_guard_min_lr),
            "restore_best_end": bool(args.pretrain_guard_restore_best_end),
            "best_rel_l2_eval": (None if not math.isfinite(pretrain_best_rel_l2) else float(pretrain_best_rel_l2)),
            "events": pretrain_guard_events,
        },
        "swa": swa_info,
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
        detail_score=detail_score.cpu().numpy().reshape(-1),
        high_detail_mask=high_detail_mask.cpu().numpy().astype(np.uint8).reshape(-1),
        chart_weights=weights.cpu().numpy(),
        chart_values=u_chart_t.cpu().numpy(),
    )

    def save_ckpt(path: str, u_states: List[Dict[str, torch.Tensor]], label: str, fallback_from: Optional[str] = None) -> None:
        payload = {
            "u_states": u_states,
            "u_kwargs": {
                "width": args.pinn_width,
                "depth": args.pinn_depth,
                "arch": args.pinn_arch,
                "residual_scale": args.pinn_residual_scale,
                "residual_zero_init": bool(args.pinn_residual_zero_init),
            },
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
    ckpt_swa = None
    if swa_states_for_save is not None:
        ckpt_swa = os.path.join(args.output_dir, f"{run_stem}_swa_topk.pt")
        save_ckpt(ckpt_swa, swa_states_for_save, "swa_topk", fallback_from=None)

    out["checkpoint_paths"] = {
        "selected": ckpt_path,
        "best_rel_l2": ckpt_best_rel,
        "best_target": ckpt_best_target,
        "best_flux": ckpt_best_flux,
        "best_score": ckpt_best_score,
    }
    if ckpt_swa is not None:
        out["checkpoint_paths"]["swa_topk"] = ckpt_swa

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
    if ckpt_swa is not None:
        print(f"  swa_topk:     {ckpt_swa}")
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
        "swa_topk_checkpoint": ckpt_swa,
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
    parser.add_argument("--u-remap-json", default=None, help="Optional JSON with new_parent mapping for split-child warmstart.")
    parser.add_argument("--trainable-charts-json", default=None, help="Optional JSON/list specifying trainable chart ids.")
    parser.add_argument("--freeze-charts-json", default=None, help="Optional JSON/list specifying frozen chart ids.")
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision when supported (CUDA autocast+GradScaler, MPS autocast).",
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA.")

    parser.add_argument("--pinn-width", type=int, default=64)
    parser.add_argument("--pinn-depth", type=int, default=4)
    parser.add_argument("--pinn-arch", type=str, choices=["mlp", "resnet"], default="mlp")
    parser.add_argument("--pinn-residual-scale", type=float, default=0.1)
    parser.add_argument("--pinn-residual-zero-init", dest="pinn_residual_zero_init", action="store_true")
    parser.add_argument("--no-pinn-residual-zero-init", dest="pinn_residual_zero_init", action="store_false")
    parser.set_defaults(pinn_residual_zero_init=True)
    parser.add_argument("--grad-clip-max-norm", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument(
        "--chart-train-mode",
        type=str,
        default="all",
        choices=["all", "overlap_only"],
        help="Train all charts or only charts with sufficient overlap neighbors.",
    )
    parser.add_argument(
        "--overlap-min-neighbors",
        type=int,
        default=1,
        help="Minimum overlap-neighbor count for trainable charts in overlap_only mode.",
    )

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
    parser.add_argument(
        "--curvature-adaptive-sampling",
        action="store_true",
        help="Enable curvature/extremity adaptive chart sampling and detail-chart refinement.",
    )
    parser.add_argument(
        "--curvature-adaptive-interface-sampling",
        action="store_true",
        help="Use detail-weighted sampling on chart-overlap interfaces.",
    )
    parser.add_argument("--detail-quantile", type=float, default=0.82, help="High-detail threshold quantile in [0,1].")
    parser.add_argument(
        "--detail-region-ratio",
        type=float,
        default=0.55,
        help="Target fraction of detail samples in adaptive chart batches.",
    )
    parser.add_argument(
        "--curvature-sample-weight",
        type=float,
        default=3.0,
        help="Weight multiplier applied to detail score for adaptive sampling.",
    )
    parser.add_argument(
        "--detail-normal-weight",
        type=float,
        default=0.75,
        help="Detail-score weight for normal-deviation proxy.",
    )
    parser.add_argument(
        "--detail-extremity-weight",
        type=float,
        default=0.20,
        help="Detail-score weight for extremity-distance proxy.",
    )
    parser.add_argument(
        "--detail-overlap-weight",
        type=float,
        default=0.05,
        help="Detail-score weight for overlap-count proxy.",
    )
    parser.add_argument("--detail-chart-topk", type=int, default=4, help="Number of highest-detail charts to refine.")
    parser.add_argument(
        "--detail-chart-step-boost",
        type=float,
        default=1.6,
        help="Local-steps multiplier for selected high-detail charts.",
    )
    parser.add_argument(
        "--detail-chart-batch-boost",
        type=float,
        default=1.35,
        help="Batch-size multiplier for selected high-detail charts.",
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
    parser.add_argument(
        "--skip-pde-if-no-valid",
        dest="skip_pde_if_no_valid",
        action="store_true",
        help="If enabled, PDE term is skipped for batches with no valid Jacobian samples.",
    )
    parser.add_argument(
        "--no-skip-pde-if-no-valid",
        dest="skip_pde_if_no_valid",
        action="store_false",
        help="Disable the no-valid Jacobian PDE skip safeguard.",
    )
    parser.set_defaults(skip_pde_if_no_valid=True)

    parser.add_argument("--bc-pretrain-epochs", type=int, default=300)
    parser.add_argument("--bc-pretrain-batch", type=int, default=256)
    parser.add_argument("--bc-pretrain-grad-weight", type=float, default=0.05)
    parser.add_argument("--bc-pretrain-interface-weight", type=float, default=0.2)
    parser.add_argument(
        "--bc-pretrain-sup-weight", type=float, default=0.0, dest="bc_pretrain_sup_weight",
        help="M6/volumetric: weight for joint interior supervision during BC pretrain. "
             "Simultaneously fits manufactured_u at SDF interior points while BC is fitted "
             "at surface, preventing the two from overwriting each other.",
    )
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
    parser.add_argument(
        "--interface-detail-boost",
        type=float,
        default=1.0,
        help="Multiplicative boost (>=1) for interface terms on high-detail overlap pairs.",
    )

    parser.add_argument("--interior-pretrain-epochs", type=int, default=0)
    parser.add_argument("--interior-pretrain-batch", type=int, default=256)
    parser.add_argument("--interior-pretrain-grad-weight", type=float, default=0.5)
    parser.add_argument("--interior-pretrain-log-every", type=int, default=50)
    parser.add_argument("--pretrain-guard-enable", action="store_true")
    parser.add_argument("--pretrain-guard-eval-every", type=int, default=50)
    parser.add_argument("--pretrain-guard-patience", type=int, default=3)
    parser.add_argument("--pretrain-guard-rel-l2-margin", type=float, default=0.002)
    parser.add_argument("--pretrain-guard-lr-decay", type=float, default=0.5)
    parser.add_argument("--pretrain-guard-min-lr", type=float, default=1e-7)
    parser.add_argument("--pretrain-guard-restore-best-end", action="store_true")
    parser.add_argument("--swa-topk", type=int, default=0)
    parser.add_argument("--swa-select-if-better", action="store_true")

    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=2.0)
    parser.add_argument("--w-interface-value", type=float, default=0.8)
    parser.add_argument("--w-interface-flux", type=float, default=0.2)
    parser.add_argument("--w-interface-flux-start", type=float, default=None)
    parser.add_argument("--w-interface-flux-end", type=float, default=None)
    parser.add_argument("--flux-ramp-iters", type=int, default=0)
    parser.add_argument("--adaptive-flux-weight", action="store_true")
    parser.add_argument("--adaptive-flux-up", type=float, default=1.05)
    parser.add_argument("--adaptive-flux-down", type=float, default=0.8)
    parser.add_argument("--adaptive-flux-rel-l2-thresh", type=float, default=0.145)

    parser.add_argument("--residual-tol", type=float, default=2e-3)
    parser.add_argument("--interface-tol", type=float, default=8e-3)
    parser.add_argument("--interface-flux-tol", type=float, default=1.5e-2)
    parser.add_argument("--plateau-patience", type=int, default=15)
    parser.add_argument("--plateau-tol", type=float, default=5e-5)
    parser.add_argument("--target-rel-l2", type=float, default=1.5e-1)
    parser.add_argument("--guard-rel-l2", type=float, default=0.0, help="L2 guard threshold (<=0 uses target-rel-l2).")
    parser.add_argument("--guard-patience", type=int, default=0, help="Rollback after this many consecutive guard violations.")
    parser.add_argument(
        "--iter-accept-rel-l2",
        type=float,
        default=0.155,
        help="Reject Schwarz iteration if eval rel-L2 exceeds this value; <=0 disables rejection.",
    )
    parser.add_argument(
        "--iter-reject-lr-decay",
        type=float,
        default=0.7,
        help="Learning-rate multiplier after trust-region rejection.",
    )
    parser.add_argument(
        "--iter-accept-rel-l2-margin",
        type=float,
        default=0.03,
        help="Allow temporary rel-L2 increase up to current_rel_l2 + margin before rejecting an iteration.",
    )
    parser.add_argument(
        "--iter-accept-pde-drop",
        type=float,
        default=0.10,
        help="Trust-region override: accept a high-rel-L2 step if PDE drops by at least this relative amount; <=0 disables.",
    )
    parser.add_argument(
        "--iter-accept-rel-l2-hardcap",
        type=float,
        default=0.35,
        help="Maximum rel-L2 allowed when PDE-drop override is active; <=0 disables override.",
    )
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

    # M1: direct-coordinate PDE mode (bypasses learned decoder Jacobian)
    parser.add_argument(
        "--direct-coord-pde",
        dest="direct_coord_pde",
        action="store_true",
        default=False,
        help="M1: compute Poisson residual directly in physical coords via rigid TNB frame, "
             "bypassing the learned decoder Jacobian entirely.",
    )

    # M2: SDF-guided interior sampling
    parser.add_argument(
        "--sdf-checkpoint",
        default=None,
        help="M2/M7: path to a trained SDF network checkpoint (.pt). "
             "Enables sdf_hard_bc_scale (M7) and optionally SDF interior sampling (M2).",
    )
    parser.add_argument(
        "--use-sdf-sampling",
        dest="use_sdf_sampling",
        action="store_true",
        default=False,
        help="M2: sample PDE collocation points in the volumetric interior via SDF rejection.",
    )
    parser.add_argument(
        "--sdf-interior-threshold",
        type=float,
        default=0.0,
        help="M2: accept candidate points with SDF < this value (0 = exactly inside).",
    )
    parser.add_argument(
        "--sdf-rejection-factor",
        type=int,
        default=6,
        help="M2: how many candidate points to draw per required sample for rejection.",
    )

    # M7: SDF-based hard BC weighting
    parser.add_argument(
        "--hard-bc",
        dest="hard_bc",
        action="store_true",
        default=False,
        help="M7: weight BC loss by tanh(-SDF(x)/scale) so near-boundary points dominate less.",
    )
    parser.add_argument(
        "--hard-bc-scale",
        type=float,
        default=0.05,
        help="M7: length scale for the SDF-based BC weight (tanh argument divisor).",
    )

    # M8: Residual-based Adaptive Refinement (RAR)
    parser.add_argument(
        "--rar-period",
        type=int,
        default=0,
        help="M8: refill RAR pool every N Schwarz iterations (0 = disabled).",
    )
    parser.add_argument("--rar-candidates", type=int, default=512,
                        help="M8: candidate points to evaluate residual on per RAR update.")
    parser.add_argument("--rar-top-k", type=int, default=64,
                        help="M8: top-k highest-residual points added to pool per RAR update.")
    parser.add_argument("--rar-pool-max", type=int, default=256,
                        help="M8: maximum pool size per chart (older points evicted randomly).")
    parser.add_argument("--rar-mix-n", type=int, default=32,
                        help="M8: number of RAR pool points mixed into each PDE batch.")

    # M6: volumetric atlas mode
    parser.add_argument(
        "--volumetric-atlas",
        dest="volumetric_atlas",
        action="store_true",
        default=False,
        help="M6: enable volumetric atlas mode — uses 3-D ball-intersection interface "
             "sampling and forces SDF interior PDE sampling (requires --sdf-checkpoint).",
    )
    parser.add_argument(
        "--bc-surface-eps",
        type=float,
        default=0.04,
        dest="bc_surface_eps",
        help="M6: SDF band half-width for surface BC sampling in volumetric mode "
             "(accept points with |SDF| < bc_surface_eps as boundary points).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_schwarz(args)


if __name__ == "__main__":
    main()
