#!/usr/bin/env python3
"""
Dense postprocessing for rabbit Poisson PINN:
- Samples dense points in chart neighborhoods mapped to physical rabbit coordinates.
- Reconstructs pressure (u) and velocity (-grad u) in the full domain.
- Exports ParaView-compatible VTU files for domain and boundary points.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


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
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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
            return torch.float32
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(width, out_dim)

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


def manufactured_u_np(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.sin(math.pi * x) * np.sin(math.pi * y) * np.sin(math.pi * z)


def manufactured_grad_u_np(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    pi = math.pi
    dux = pi * np.cos(pi * x) * np.sin(pi * y) * np.sin(pi * z)
    duy = pi * np.sin(pi * x) * np.cos(pi * y) * np.sin(pi * z)
    duz = pi * np.sin(pi * x) * np.sin(pi * y) * np.cos(pi * z)
    return np.stack([dux, duy, duz], axis=1)


def attach_reference_and_error_fields(points: np.ndarray, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pressure_true = manufactured_u_np(points)
    grad_true = manufactured_grad_u_np(points)
    velocity_true = -grad_true
    pressure_pred = fields["pressure"]
    velocity_pred = fields["velocity"]

    pressure_error = pressure_pred - pressure_true
    velocity_error = velocity_pred - velocity_true

    out = dict(fields)
    out["pressure_true"] = pressure_true
    out["pressure_error"] = pressure_error
    out["pressure_error_abs"] = np.abs(pressure_error)
    out["velocity_true"] = velocity_true
    out["velocity_error"] = velocity_error
    out["velocity_error_mag"] = np.linalg.norm(velocity_error, axis=1)
    return out


def write_vtu_points(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    n = int(points.shape[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int64)
    offsets = np.arange(1, n + 1, dtype=np.int64)
    types = np.full((n,), 1, dtype=np.uint8)  # VTK_VERTEX

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in points:
            f.write(f"          {float(p[0]):.12e} {float(p[1]):.12e} {float(p[2]):.12e}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in connectivity) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in types) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")

        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            a = np.asarray(arr)
            if a.ndim == 1:
                f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="ascii">\n')
                f.write("          " + " ".join(f"{float(v):.12e}" for v in a) + "\n")
                f.write("        </DataArray>\n")
            elif a.ndim == 2 and a.shape[1] == 3:
                f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="ascii">\n')
                for row in a:
                    f.write(f"          {float(row[0]):.12e} {float(row[1]):.12e} {float(row[2]):.12e}\n")
                f.write("        </DataArray>\n")
            else:
                raise ValueError(f"Unsupported shape for PointData '{name}': {a.shape}")
        f.write("      </PointData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def velocity_quantiles(velocity_mag: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(velocity_mag, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"q95": 0.0, "q99": 0.0, "q999": 0.0, "max": 0.0}
    return {
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "q999": float(np.quantile(arr, 0.999)),
        "max": float(np.max(arr)),
    }


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points
    mins = np.min(points, axis=0, keepdims=True)
    q = np.floor((points - mins) / voxel_size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx]


def orient_normals_outward(surface_points: np.ndarray, surface_normals: np.ndarray) -> Tuple[np.ndarray, float]:
    center = np.mean(surface_points, axis=0, keepdims=True)
    score = float(np.mean(np.sum(surface_normals * (surface_points - center), axis=1)))
    if score < 0.0:
        return -surface_normals, -score
    return surface_normals, score


def signed_score_chunked(
    query_points: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    tree: cKDTree,
    k: int,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(query_points.shape[0])
    signed_mean = np.zeros((n,), dtype=np.float64)
    nearest_dist = np.zeros((n,), dtype=np.float64)
    nearest_signed = np.zeros((n,), dtype=np.float64)
    k_eff = int(max(1, min(k, surface_points.shape[0])))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        q = query_points[s:e]
        d, idx = tree.query(q, k=k_eff)
        if k_eff == 1:
            d = d.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        nb_p = surface_points[idx]
        nb_n = surface_normals[idx]
        vec = q[:, None, :] - nb_p
        signed = np.sum(vec * nb_n, axis=2)
        w = 1.0 / np.maximum(d, 1e-12)
        signed_mean[s:e] = np.sum(w * signed, axis=1) / np.sum(w, axis=1)
        nearest_dist[s:e] = d[:, 0]
        nearest_signed[s:e] = signed[:, 0]
    return signed_mean, nearest_dist, nearest_signed


def apply_inside_filter(
    *,
    points: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    tree: cKDTree,
    k: int,
    signed_margin: float,
    bbox_margin_ratio: float,
    signed_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bb_min = np.min(surface_points, axis=0, keepdims=True)
    bb_max = np.max(surface_points, axis=0, keepdims=True)
    diag = float(np.linalg.norm((bb_max - bb_min).reshape(-1)))
    margin = max(0.0, bbox_margin_ratio) * diag
    in_box = np.all((points >= (bb_min - margin)) & (points <= (bb_max + margin)), axis=1)

    signed_mean, nearest_dist, nearest_signed = signed_score_chunked(
        query_points=points,
        surface_points=surface_points,
        surface_normals=surface_normals,
        tree=tree,
        k=k,
        chunk=signed_chunk,
    )
    keep = in_box & (signed_mean <= float(signed_margin)) & (nearest_signed <= float(signed_margin))
    return keep, signed_mean, nearest_dist, nearest_signed


def compute_surface_quality(
    *,
    points: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    tree: cKDTree,
    k: int,
    chunk: int,
) -> Dict[str, np.ndarray]:
    signed_mean, nearest_dist, nearest_signed = signed_score_chunked(
        query_points=points,
        surface_points=surface_points,
        surface_normals=surface_normals,
        tree=tree,
        k=k,
        chunk=chunk,
    )
    return {
        "inside_signed_score": signed_mean.astype(np.float64),
        "nearest_surface_dist": nearest_dist.astype(np.float64),
        "nearest_signed_score": nearest_signed.astype(np.float64),
    }


def sample_dense_candidates(
    *,
    decoders: List[ChartDecoder],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    support_r: torch.Tensor,
    atlas_points: torch.Tensor,
    membership_np: Optional[np.ndarray],
    n_samples_per_chart: int,
    sampling_mode: str,
    xi_noise_scale: float,
    xi_scale: float,
    shell_fraction: float,
    shell_w_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    chunks: List[np.ndarray] = [atlas_points.detach().cpu().numpy()]
    n_charts = len(decoders)
    n_shell = int(max(0, min(n_samples_per_chart, round(shell_fraction * n_samples_per_chart))))
    n_bulk = int(max(0, n_samples_per_chart - n_shell))

    for i in range(n_charts):
        with torch.no_grad():
            r = float(support_r[i].item())
            if sampling_mode == "training_like" and membership_np is not None:
                chart_members = np.where(membership_np[:, i] > 0.05)[0]
                if chart_members.size > 0:
                    pick = np.random.choice(chart_members, size=n_samples_per_chart, replace=True)
                    xb = atlas_points[pick]
                    xi_base = local_coords(xb, seeds[i], t1[i], t2[i], nvec[i])
                    noise = xi_noise_scale * support_r[i] * torch.randn_like(xi_base)
                    xi = xi_base + noise
                    max_abs = 1.25 * support_r[i]
                    xi = torch.clamp(xi, min=-max_abs, max=max_abs)
                else:
                    xi = (2.0 * torch.rand((n_samples_per_chart, 3), device=device, dtype=dtype) - 1.0) * (xi_scale * r)
            else:
                xi_parts: List[torch.Tensor] = []
                if n_bulk > 0:
                    xi_bulk = (2.0 * torch.rand((n_bulk, 3), device=device, dtype=dtype) - 1.0) * (xi_scale * r)
                    xi_parts.append(xi_bulk)
                if n_shell > 0:
                    xi_shell = (2.0 * torch.rand((n_shell, 3), device=device, dtype=dtype) - 1.0) * (xi_scale * r)
                    xi_shell[:, 2] = (2.0 * torch.rand((n_shell,), device=device, dtype=dtype) - 1.0) * (shell_w_scale * r)
                    xi_parts.append(xi_shell)
                if not xi_parts:
                    continue
                xi = torch.cat(xi_parts, dim=0)
            x = decoders[i](
                xi,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
            )
            chunks.append(x.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def sample_boundary_candidates(
    *,
    decoders: List[ChartDecoder],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    support_r: torch.Tensor,
    atlas_points: torch.Tensor,
    n_samples_per_chart: int,
    xi_scale: float,
    boundary_w_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    chunks: List[np.ndarray] = [atlas_points.detach().cpu().numpy()]
    n_charts = len(decoders)
    for i in range(n_charts):
        with torch.no_grad():
            r = float(support_r[i].item())
            if n_samples_per_chart <= 0:
                continue
            xi = (2.0 * torch.rand((n_samples_per_chart, 3), device=device, dtype=dtype) - 1.0) * (xi_scale * r)
            # Constrain local normal offset near chart mid-surface to create a dense boundary shell.
            xi[:, 2] = (2.0 * torch.rand((n_samples_per_chart,), device=device, dtype=dtype) - 1.0) * (
                boundary_w_scale * r
            )
            x = decoders[i](
                xi,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
            )
            chunks.append(x.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_global_fields(
    *,
    points_np: np.ndarray,
    u_nets: List[LocalPoissonPINN],
    masks: List[MaskNet],
    seeds: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    nvec: torch.Tensor,
    support_r: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    velocity_mode: str,
) -> Dict[str, np.ndarray]:
    n = int(points_np.shape[0])
    pressure = np.zeros((n,), dtype=np.float64)
    velocity = np.zeros((n, 3), dtype=np.float64)
    vel_mag = np.zeros((n,), dtype=np.float64)
    max_weight = np.zeros((n,), dtype=np.float64)
    chart_id = np.zeros((n,), dtype=np.int32)
    entropy = np.zeros((n,), dtype=np.float64)

    n_charts = len(u_nets)
    if velocity_mode not in {"strict_blended", "weighted_detached", "dominant_chart"}:
        raise ValueError(f"Unsupported velocity mode: {velocity_mode}")

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.tensor(points_np[s:e], device=device, dtype=dtype, requires_grad=True)
        logits: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        ui_full: List[torch.Tensor] = []
        for i in range(n_charts):
            xi = local_coords(xb, seeds[i], t1[i], t2[i], nvec[i])
            logits.append(masks[i](xi, chart_scale=support_r[i]))
            ui = u_nets[i](xi)
            vals.append(ui.squeeze(-1))
            ui_full.append(ui)

        logits_t = torch.stack(logits, dim=1)
        weights = torch.softmax(logits_t, dim=1)
        vals_t = torch.stack(vals, dim=1)
        u_pred = torch.sum(weights * vals_t, dim=1, keepdim=True)

        if velocity_mode == "strict_blended":
            grad_u = torch.autograd.grad(
                u_pred,
                xb,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=False,
                retain_graph=False,
            )[0]
            vel = -grad_u
        else:
            grad_each: List[torch.Tensor] = []
            for i, ui in enumerate(ui_full):
                gi = torch.autograd.grad(
                    ui,
                    xb,
                    grad_outputs=torch.ones_like(ui),
                    create_graph=False,
                    retain_graph=(i < n_charts - 1),
                )[0]
                grad_each.append(gi)
            grad_stack = torch.stack(grad_each, dim=1)  # [B, C, 3]
            if velocity_mode == "weighted_detached":
                vel = -torch.sum(weights.detach().unsqueeze(-1) * grad_stack, dim=1)
            else:  # dominant_chart
                cid_local = torch.argmax(weights, dim=1)
                gather_idx = cid_local.view(-1, 1, 1).expand(-1, 1, 3)
                vel = -torch.gather(grad_stack, dim=1, index=gather_idx).squeeze(1)

        maxw, cid = torch.max(weights, dim=1)
        ent = -torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)), dim=1)

        pressure[s:e] = u_pred.detach().cpu().numpy().reshape(-1)
        vb = vel.detach().cpu().numpy()
        velocity[s:e, :] = vb
        vel_mag[s:e] = np.linalg.norm(vb, axis=1)
        max_weight[s:e] = maxw.detach().cpu().numpy().reshape(-1)
        chart_id[s:e] = cid.detach().cpu().numpy().astype(np.int32)
        entropy[s:e] = ent.detach().cpu().numpy().reshape(-1)

    return {
        "pressure": pressure,
        "velocity": velocity,
        "velocity_mag": vel_mag,
        "max_weight": max_weight,
        "chart_id": chart_id,
        "chart_entropy": entropy,
    }


def build_models(
    *,
    atlas_checkpoint: str,
    solver_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[ChartDecoder], List[MaskNet], List[LocalPoissonPINN]]:
    atlas_ckpt = torch.load(atlas_checkpoint, map_location=torch.device("cpu"))
    dec_kw = atlas_ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kw = atlas_ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    dec_states = atlas_ckpt.get("decoder_states", [])
    mask_states = atlas_ckpt.get("mask_states", [])
    if not isinstance(dec_states, list) or not isinstance(mask_states, list) or len(dec_states) != len(mask_states):
        raise RuntimeError("Invalid decoder/mask states in atlas checkpoint.")

    decoders: List[ChartDecoder] = []
    masks: List[MaskNet] = []
    for ds, ms in zip(dec_states, mask_states):
        d = ChartDecoder(**dec_kw).to(device=device, dtype=dtype)
        m = MaskNet(**mask_kw).to(device=device, dtype=dtype)
        d.load_state_dict(ds)
        m.load_state_dict(ms)
        d.eval()
        m.eval()
        decoders.append(d)
        masks.append(m)

    solver_ckpt_obj = torch.load(solver_checkpoint, map_location=torch.device("cpu"))
    u_states = solver_ckpt_obj.get("u_states")
    u_kw = solver_ckpt_obj.get("u_kwargs", {"width": 64, "depth": 4})
    u_width = int(u_kw.get("width", 64)) if isinstance(u_kw, dict) else 64
    u_depth = int(u_kw.get("depth", 4)) if isinstance(u_kw, dict) else 4
    u_arch = str(u_kw.get("arch", "mlp")).lower().strip() if isinstance(u_kw, dict) else "mlp"
    u_res_scale = float(u_kw.get("residual_scale", 0.1)) if isinstance(u_kw, dict) else 0.1
    u_res_zero = bool(u_kw.get("residual_zero_init", True)) if isinstance(u_kw, dict) else True
    if not isinstance(u_states, list):
        raise RuntimeError("Invalid u_states in solver checkpoint.")
    if len(u_states) != len(decoders):
        raise RuntimeError(
            f"Solver checkpoint chart count mismatch: u_states={len(u_states)} vs decoders={len(decoders)}"
        )
    u_nets: List[LocalPoissonPINN] = []
    for us in u_states:
        u = LocalPoissonPINN(
            width=u_width,
            depth=u_depth,
            arch=u_arch,
            residual_scale=u_res_scale,
            residual_zero_init=u_res_zero,
        ).to(device=device, dtype=dtype)
        u.load_state_dict(us, strict=False)
        u.eval()
        u_nets.append(u)
    return decoders, masks, u_nets


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    parser = argparse.ArgumentParser(description="Dense pressure/velocity postprocessing for rabbit Poisson PINN")
    parser.add_argument(
        "--atlas-data",
        default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_data.npz"),
    )
    parser.add_argument(
        "--atlas-checkpoint",
        default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_trained.pt"),
    )
    parser.add_argument("--solver-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="dense_fields")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float64"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--velocity-mode",
        choices=["strict_blended", "weighted_detached", "dominant_chart"],
        default="weighted_detached",
        help="Velocity reconstruction mode for exported fields.",
    )
    parser.add_argument("--viz-min-blend-weight", type=float, default=0.90)
    parser.add_argument("--viz-velocity-clip-quantile", type=float, default=0.995)
    parser.add_argument("--compare-strict-baseline", dest="compare_strict_baseline", action="store_true")
    parser.add_argument("--no-compare-strict-baseline", dest="compare_strict_baseline", action="store_false")
    parser.set_defaults(compare_strict_baseline=True)
    parser.add_argument("--export-raw-vtu", dest="export_raw_vtu", action="store_true")
    parser.add_argument("--no-export-raw-vtu", dest="export_raw_vtu", action="store_false")
    parser.set_defaults(export_raw_vtu=True)
    parser.add_argument("--export-viz-vtu", dest="export_viz_vtu", action="store_true")
    parser.add_argument("--no-export-viz-vtu", dest="export_viz_vtu", action="store_false")
    parser.set_defaults(export_viz_vtu=True)
    parser.add_argument("--save-point-quality", dest="save_point_quality", action="store_true")
    parser.add_argument("--no-save-point-quality", dest="save_point_quality", action="store_false")
    parser.set_defaults(save_point_quality=True)

    parser.add_argument("--n-samples-per-chart", type=int, default=20000)
    parser.add_argument("--n-boundary-samples-per-chart", type=int, default=10000)
    parser.add_argument(
        "--sampling-mode",
        choices=["training_like", "uniform"],
        default="training_like",
        help="Dense interior sampling strategy. training_like is usually more stable on rabbit.",
    )
    parser.add_argument("--xi-noise-scale", type=float, default=0.08)
    parser.add_argument("--xi-scale", type=float, default=0.95)
    parser.add_argument("--shell-fraction", type=float, default=0.35)
    parser.add_argument("--shell-w-scale", type=float, default=0.15)
    parser.add_argument("--boundary-w-scale", type=float, default=0.04)
    parser.add_argument("--voxel-size", type=float, default=0.0012)
    parser.add_argument("--boundary-voxel-size", type=float, default=0.0010)
    parser.add_argument("--max-points", type=int, default=350000)
    parser.add_argument("--max-boundary-points", type=int, default=220000)
    parser.add_argument("--domain-min-max-weight", type=float, default=0.0)
    parser.add_argument("--min-max-weight", type=float, default=None, help="Deprecated alias for --domain-min-max-weight.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--boundary-subsample", type=int, default=0, help="Optional cap before boundary densification; 0 keeps all atlas points.")
    parser.add_argument(
        "--inside-filter",
        dest="inside_filter",
        action="store_true",
        help="Enable strict inside-domain filtering using boundary normals.",
    )
    parser.add_argument(
        "--no-inside-filter",
        dest="inside_filter",
        action="store_false",
        help="Disable strict inside-domain filtering.",
    )
    parser.set_defaults(inside_filter=True)
    parser.add_argument("--inside-k", type=int, default=8, help="kNN count for signed-distance voting.")
    parser.add_argument("--inside-signed-margin", type=float, default=-1e-4, help="Keep points with signed score <= margin.")
    parser.add_argument("--inside-bbox-margin-ratio", type=float, default=0.02, help="Bounding-box margin ratio for coarse rejection.")
    parser.add_argument("--inside-signed-chunk", type=int, default=50000, help="Chunk size for signed-score evaluation.")
    parser.add_argument("--boundary-band", type=float, default=0.0022, help="Boundary shell half-band in signed score.")
    parser.add_argument(
        "--boundary-outside-margin",
        type=float,
        default=-1e-4,
        help="Boundary points must satisfy signed score <= this to avoid outside points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    if dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
    print(f"Device={device} dtype={dtype} n_samples_per_chart={args.n_samples_per_chart}")

    atlas_np = np.load(args.atlas_data)
    points = torch.tensor(atlas_np["points"], device=device, dtype=dtype)
    membership_np = atlas_np["membership"] if "membership" in atlas_np else None
    boundary_points_ref = atlas_np["points"].astype(np.float64)
    boundary_normals_ref = atlas_np["normals"].astype(np.float64)
    boundary_normals_ref, normal_orientation_score = orient_normals_outward(boundary_points_ref, boundary_normals_ref)
    boundary_tree = cKDTree(boundary_points_ref)
    seeds = torch.tensor(atlas_np["seed_points"], device=device, dtype=dtype)
    t1 = torch.tensor(atlas_np["frame_t1"], device=device, dtype=dtype)
    t2 = torch.tensor(atlas_np["frame_t2"], device=device, dtype=dtype)
    nvec = torch.tensor(atlas_np["frame_n"], device=device, dtype=dtype)
    support_r = torch.tensor(atlas_np["support_radii"], device=device, dtype=dtype)
    n_charts = int(support_r.shape[0])
    print(f"Loaded atlas: points={points.shape[0]} charts={n_charts}")

    decoders, masks, u_nets = build_models(
        atlas_checkpoint=args.atlas_checkpoint,
        solver_checkpoint=args.solver_checkpoint,
        device=device,
        dtype=dtype,
    )

    dense_candidates = sample_dense_candidates(
        decoders=decoders,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        support_r=support_r,
        atlas_points=points,
        membership_np=membership_np,
        n_samples_per_chart=args.n_samples_per_chart,
        sampling_mode=args.sampling_mode,
        xi_noise_scale=args.xi_noise_scale,
        xi_scale=args.xi_scale,
        shell_fraction=args.shell_fraction,
        shell_w_scale=args.shell_w_scale,
        device=device,
        dtype=dtype,
    )
    dense_candidates = dense_candidates[np.isfinite(dense_candidates).all(axis=1)]
    dense_candidates = voxel_downsample(dense_candidates, voxel_size=args.voxel_size)
    if args.max_points > 0 and dense_candidates.shape[0] > args.max_points:
        pick = np.random.choice(dense_candidates.shape[0], size=args.max_points, replace=False)
        dense_candidates = dense_candidates[pick]
    domain_filter_stats: Dict[str, float] = {}
    if args.inside_filter:
        keep_in, signed_score, nearest_dist, nearest_signed = apply_inside_filter(
            points=dense_candidates,
            surface_points=boundary_points_ref,
            surface_normals=boundary_normals_ref,
            tree=boundary_tree,
            k=args.inside_k,
            signed_margin=args.inside_signed_margin,
            bbox_margin_ratio=args.inside_bbox_margin_ratio,
            signed_chunk=args.inside_signed_chunk,
        )
        dense_candidates = dense_candidates[keep_in]
        domain_filter_stats = {
            "enabled": 1.0,
            "before": float(keep_in.shape[0]),
            "after": float(np.sum(keep_in)),
            "kept_fraction": float(np.mean(keep_in)),
            "signed_mean": float(np.mean(signed_score)),
            "signed_q95": float(np.quantile(signed_score, 0.95)),
            "nearest_signed_q95": float(np.quantile(nearest_signed, 0.95)),
            "nearest_dist_q95": float(np.quantile(nearest_dist, 0.95)),
        }
    else:
        domain_filter_stats = {"enabled": 0.0, "before": float(dense_candidates.shape[0]), "after": float(dense_candidates.shape[0])}
    print(f"Dense candidates after downsample: {dense_candidates.shape[0]}")

    domain_fields_all = evaluate_global_fields(
        points_np=dense_candidates,
        u_nets=u_nets,
        masks=masks,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        support_r=support_r,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        velocity_mode=args.velocity_mode,
    )
    min_w = float(args.domain_min_max_weight if args.min_max_weight is None else args.min_max_weight)
    keep_raw = np.ones((dense_candidates.shape[0],), dtype=bool)
    if min_w > 0.0:
        keep_raw = domain_fields_all["max_weight"] >= min_w
        if np.sum(keep_raw) == 0:
            keep_raw = np.ones((dense_candidates.shape[0],), dtype=bool)
    domain_points_raw = dense_candidates[keep_raw]
    domain_data_raw = {
        "pressure": domain_fields_all["pressure"][keep_raw],
        "velocity": domain_fields_all["velocity"][keep_raw],
        "velocity_mag": domain_fields_all["velocity_mag"][keep_raw],
        "blend_weight": domain_fields_all["max_weight"][keep_raw],
        "chart_id": domain_fields_all["chart_id"][keep_raw].astype(np.float64),
        "chart_entropy": domain_fields_all["chart_entropy"][keep_raw],
    }
    if args.save_point_quality:
        q_domain = compute_surface_quality(
            points=domain_points_raw,
            surface_points=boundary_points_ref,
            surface_normals=boundary_normals_ref,
            tree=boundary_tree,
            k=args.inside_k,
            chunk=args.inside_signed_chunk,
        )
        domain_data_raw.update(q_domain)
    domain_data_raw = attach_reference_and_error_fields(domain_points_raw, domain_data_raw)
    print(f"Domain raw points kept: {domain_points_raw.shape[0]} (min_max_weight={min_w})")

    keep_viz_domain = domain_data_raw["blend_weight"] >= float(args.viz_min_blend_weight)
    if np.sum(keep_viz_domain) == 0:
        keep_viz_domain = np.ones_like(keep_viz_domain, dtype=bool)
    domain_points_viz = domain_points_raw[keep_viz_domain]
    domain_data_viz = {k: v[keep_viz_domain] for k, v in domain_data_raw.items()}

    atlas_boundary_points = atlas_np["points"].copy()
    if args.boundary_subsample > 0 and atlas_boundary_points.shape[0] > args.boundary_subsample:
        pick = np.random.choice(atlas_boundary_points.shape[0], size=args.boundary_subsample, replace=False)
        atlas_boundary_points = atlas_boundary_points[pick]
    boundary_candidates = sample_boundary_candidates(
        decoders=decoders,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        support_r=support_r,
        atlas_points=torch.tensor(atlas_boundary_points, device=device, dtype=dtype),
        n_samples_per_chart=args.n_boundary_samples_per_chart,
        xi_scale=args.xi_scale,
        boundary_w_scale=args.boundary_w_scale,
        device=device,
        dtype=dtype,
    )
    boundary_candidates = boundary_candidates[np.isfinite(boundary_candidates).all(axis=1)]
    boundary_filter_stats: Dict[str, float] = {}
    if args.inside_filter:
        keep_b, signed_b, nearest_dist_b, nearest_signed_b = apply_inside_filter(
            points=boundary_candidates,
            surface_points=boundary_points_ref,
            surface_normals=boundary_normals_ref,
            tree=boundary_tree,
            k=args.inside_k,
            signed_margin=args.boundary_outside_margin,
            bbox_margin_ratio=args.inside_bbox_margin_ratio,
            signed_chunk=args.inside_signed_chunk,
        )
        band = np.abs(signed_b) <= float(args.boundary_band)
        keep_b = keep_b & band
        boundary_candidates = boundary_candidates[keep_b]
        boundary_filter_stats = {
            "enabled": 1.0,
            "before": float(keep_b.shape[0]),
            "after": float(np.sum(keep_b)),
            "kept_fraction": float(np.mean(keep_b)),
            "signed_mean": float(np.mean(signed_b)),
            "signed_q95_abs": float(np.quantile(np.abs(signed_b), 0.95)),
            "nearest_signed_q95_abs": float(np.quantile(np.abs(nearest_signed_b), 0.95)),
            "nearest_dist_q95": float(np.quantile(nearest_dist_b, 0.95)),
        }
        if boundary_candidates.shape[0] < 5000:
            boundary_candidates = atlas_boundary_points.astype(np.float64)
    else:
        boundary_filter_stats = {"enabled": 0.0, "before": float(boundary_candidates.shape[0]), "after": float(boundary_candidates.shape[0])}
    boundary_points = voxel_downsample(boundary_candidates, voxel_size=args.boundary_voxel_size)
    if args.max_boundary_points > 0 and boundary_points.shape[0] > args.max_boundary_points:
        pick = np.random.choice(boundary_points.shape[0], size=args.max_boundary_points, replace=False)
        boundary_points = boundary_points[pick]
    boundary_fields_all = evaluate_global_fields(
        points_np=boundary_points,
        u_nets=u_nets,
        masks=masks,
        seeds=seeds,
        t1=t1,
        t2=t2,
        nvec=nvec,
        support_r=support_r,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        velocity_mode=args.velocity_mode,
    )
    boundary_points_raw = boundary_points
    boundary_data_raw = {
        "pressure": boundary_fields_all["pressure"],
        "velocity": boundary_fields_all["velocity"],
        "velocity_mag": boundary_fields_all["velocity_mag"],
        "blend_weight": boundary_fields_all["max_weight"],
        "chart_id": boundary_fields_all["chart_id"].astype(np.float64),
        "chart_entropy": boundary_fields_all["chart_entropy"],
    }
    if args.save_point_quality:
        q_boundary = compute_surface_quality(
            points=boundary_points_raw,
            surface_points=boundary_points_ref,
            surface_normals=boundary_normals_ref,
            tree=boundary_tree,
            k=args.inside_k,
            chunk=args.inside_signed_chunk,
        )
        boundary_data_raw.update(q_boundary)
    boundary_data_raw = attach_reference_and_error_fields(boundary_points_raw, boundary_data_raw)

    keep_viz_boundary = boundary_data_raw["blend_weight"] >= float(args.viz_min_blend_weight)
    if np.sum(keep_viz_boundary) == 0:
        keep_viz_boundary = np.ones_like(keep_viz_boundary, dtype=bool)
    boundary_points_viz = boundary_points_raw[keep_viz_boundary]
    boundary_data_viz = {k: v[keep_viz_boundary] for k, v in boundary_data_raw.items()}

    q_clip = float(np.quantile(domain_data_viz["velocity_mag"], args.viz_velocity_clip_quantile))
    q_clip = max(q_clip, 1e-12)

    def add_viz_clip_fields(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(data)
        vm = out["velocity_mag"]
        out["velocity_mag_raw"] = vm.copy()
        out["velocity_mag_viz_clipped"] = np.minimum(vm, q_clip)
        scale = np.minimum(1.0, q_clip / np.maximum(vm, 1e-12))
        out["velocity_viz_clipped"] = out["velocity"] * scale[:, None]
        return out

    domain_data_raw = add_viz_clip_fields(domain_data_raw)
    domain_data_viz = add_viz_clip_fields(domain_data_viz)
    boundary_data_raw = add_viz_clip_fields(boundary_data_raw)
    boundary_data_viz = add_viz_clip_fields(boundary_data_viz)

    stem = f"rabbit_poisson_{args.run_tag}"
    npz_path = os.path.join(args.output_dir, f"{stem}_dense_fields.npz")
    np.savez_compressed(
        npz_path,
        domain_points=domain_points_raw,
        domain_pressure=domain_data_raw["pressure"],
        domain_velocity=domain_data_raw["velocity"],
        domain_velocity_mag=domain_data_raw["velocity_mag"],
        domain_blend_weight=domain_data_raw["blend_weight"],
        domain_chart_id=domain_data_raw["chart_id"],
        domain_pressure_true=domain_data_raw["pressure_true"],
        domain_pressure_error=domain_data_raw["pressure_error"],
        domain_velocity_true=domain_data_raw["velocity_true"],
        domain_velocity_error=domain_data_raw["velocity_error"],
        domain_velocity_error_mag=domain_data_raw["velocity_error_mag"],
        domain_inside_signed_score=domain_data_raw.get("inside_signed_score", np.full((domain_points_raw.shape[0],), np.nan)),
        domain_nearest_surface_dist=domain_data_raw.get("nearest_surface_dist", np.full((domain_points_raw.shape[0],), np.nan)),
        domain_nearest_signed_score=domain_data_raw.get("nearest_signed_score", np.full((domain_points_raw.shape[0],), np.nan)),
        domain_viz_points=domain_points_viz,
        domain_viz_velocity=domain_data_viz["velocity"],
        domain_viz_velocity_mag_raw=domain_data_viz["velocity_mag_raw"],
        domain_viz_velocity_mag_clipped=domain_data_viz["velocity_mag_viz_clipped"],
        boundary_points=boundary_points_raw,
        boundary_pressure=boundary_data_raw["pressure"],
        boundary_velocity=boundary_data_raw["velocity"],
        boundary_velocity_mag=boundary_data_raw["velocity_mag"],
        boundary_blend_weight=boundary_data_raw["blend_weight"],
        boundary_chart_id=boundary_data_raw["chart_id"],
        boundary_pressure_true=boundary_data_raw["pressure_true"],
        boundary_pressure_error=boundary_data_raw["pressure_error"],
        boundary_velocity_true=boundary_data_raw["velocity_true"],
        boundary_velocity_error=boundary_data_raw["velocity_error"],
        boundary_velocity_error_mag=boundary_data_raw["velocity_error_mag"],
        boundary_inside_signed_score=boundary_data_raw.get("inside_signed_score", np.full((boundary_points_raw.shape[0],), np.nan)),
        boundary_nearest_surface_dist=boundary_data_raw.get("nearest_surface_dist", np.full((boundary_points_raw.shape[0],), np.nan)),
        boundary_nearest_signed_score=boundary_data_raw.get("nearest_signed_score", np.full((boundary_points_raw.shape[0],), np.nan)),
        boundary_viz_points=boundary_points_viz,
        boundary_viz_velocity=boundary_data_viz["velocity"],
        boundary_viz_velocity_mag_raw=boundary_data_viz["velocity_mag_raw"],
        boundary_viz_velocity_mag_clipped=boundary_data_viz["velocity_mag_viz_clipped"],
    )
    domain_raw_vtu = os.path.join(args.output_dir, f"{stem}_domain_raw.vtu")
    boundary_raw_vtu = os.path.join(args.output_dir, f"{stem}_boundary_raw.vtu")
    domain_viz_vtu = os.path.join(args.output_dir, f"{stem}_domain_viz.vtu")
    boundary_viz_vtu = os.path.join(args.output_dir, f"{stem}_boundary_viz.vtu")
    if args.export_raw_vtu:
        write_vtu_points(domain_raw_vtu, domain_points_raw, domain_data_raw)
        write_vtu_points(boundary_raw_vtu, boundary_points_raw, boundary_data_raw)
    if args.export_viz_vtu:
        write_vtu_points(domain_viz_vtu, domain_points_viz, domain_data_viz)
        write_vtu_points(boundary_viz_vtu, boundary_points_viz, boundary_data_viz)

    strict_compare = None
    if args.compare_strict_baseline and args.velocity_mode != "strict_blended":
        strict_fields = evaluate_global_fields(
            points_np=domain_points_raw,
            u_nets=u_nets,
            masks=masks,
            seeds=seeds,
            t1=t1,
            t2=t2,
            nvec=nvec,
            support_r=support_r,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            velocity_mode="strict_blended",
        )
        q_sel = velocity_quantiles(domain_data_raw["velocity_mag"])
        q_strict = velocity_quantiles(strict_fields["velocity_mag"])
        reduction_q99 = 1.0 - (q_sel["q99"] / max(q_strict["q99"], 1e-12))
        strict_compare = {
            "selected_mode": args.velocity_mode,
            "selected_quantiles": q_sel,
            "strict_quantiles": q_strict,
            "q99_reduction_fraction": float(reduction_q99),
        }

    solver_eval_metrics = {}
    try:
        solver_ckpt_obj = torch.load(args.solver_checkpoint, map_location=torch.device("cpu"))
        if isinstance(solver_ckpt_obj, dict) and isinstance(solver_ckpt_obj.get("metrics"), dict):
            g = solver_ckpt_obj["metrics"].get("global", {})
            solver_eval_metrics = {
                "relative_l2_error": float(g.get("relative_l2_error", float("nan"))),
                "l2_error": float(g.get("l2_error", float("nan"))),
                "max_error": float(g.get("max_error", float("nan"))),
            }
    except Exception:
        solver_eval_metrics = {}

    summary = {
        "atlas_data": args.atlas_data,
        "atlas_checkpoint": args.atlas_checkpoint,
        "solver_checkpoint": args.solver_checkpoint,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "velocity_mode": args.velocity_mode,
        "n_charts": n_charts,
        "n_domain_points_raw": int(domain_points_raw.shape[0]),
        "n_domain_points_viz": int(domain_points_viz.shape[0]),
        "n_boundary_points_raw": int(boundary_points_raw.shape[0]),
        "n_boundary_points_viz": int(boundary_points_viz.shape[0]),
        "viz_min_blend_weight": float(args.viz_min_blend_weight),
        "viz_velocity_clip_quantile": float(args.viz_velocity_clip_quantile),
        "viz_velocity_clip_value": float(q_clip),
        "inside_filter": {
            "enabled": bool(args.inside_filter),
            "normal_orientation_score": float(normal_orientation_score),
            "inside_k": int(args.inside_k),
            "inside_signed_margin": float(args.inside_signed_margin),
            "boundary_band": float(args.boundary_band),
            "domain_stats": domain_filter_stats,
            "boundary_stats": boundary_filter_stats,
        },
        "solver_eval_metrics": solver_eval_metrics,
        "velocity_quantiles_raw": velocity_quantiles(domain_data_raw["velocity_mag"]),
        "velocity_quantiles_viz": velocity_quantiles(domain_data_viz["velocity_mag"]),
        "strict_compare": strict_compare,
        "no_outside_in_viz": bool(
            np.all(domain_data_viz.get("inside_signed_score", np.array([-1.0], dtype=np.float64)) <= 0.0)
        ),
        "domain_rel_l2_pressure": float(
            np.sqrt(np.mean(domain_data_raw["pressure_error"] ** 2) / max(np.mean(domain_data_raw["pressure_true"] ** 2), 1e-12))
        ),
        "domain_l2_velocity_error": float(np.sqrt(np.mean(domain_data_raw["velocity_error_mag"] ** 2))),
        "npz": npz_path,
        "domain_raw_vtu": domain_raw_vtu if args.export_raw_vtu else None,
        "boundary_raw_vtu": boundary_raw_vtu if args.export_raw_vtu else None,
        "domain_viz_vtu": domain_viz_vtu if args.export_viz_vtu else None,
        "boundary_viz_vtu": boundary_viz_vtu if args.export_viz_vtu else None,
    }
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)

    note_path = os.path.join(args.output_dir, f"{stem}_publication_note.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("# Rabbit Poisson Publication Note\n")
        f.write("\n")
        f.write(f"- Velocity reconstruction: `{args.velocity_mode}` to reduce seam spikes from `u_i * grad(w_i)` terms.\n")
        f.write(
            f"- Confidence masking: visualization VTU uses `blend_weight >= {args.viz_min_blend_weight:.2f}`.\n"
        )
        f.write(
            f"- Outlier control: visualization includes `velocity_mag_viz_clipped` at quantile {args.viz_velocity_clip_quantile:.3f} (value `{q_clip:.6g}`).\n"
        )
        if strict_compare is not None:
            f.write(
                f"- q99 reduction vs strict_blended on raw domain points: `{100.0 * strict_compare['q99_reduction_fraction']:.2f}%`.\n"
            )
        f.write("- Geometry validity: inside-filter diagnostics are recorded in the summary JSON.\n")

    if args.export_raw_vtu:
        print(f"Saved raw domain VTU: {domain_raw_vtu}")
        print(f"Saved raw boundary VTU: {boundary_raw_vtu}")
    if args.export_viz_vtu:
        print(f"Saved viz domain VTU: {domain_viz_vtu}")
        print(f"Saved viz boundary VTU: {boundary_viz_vtu}")
    print(f"Saved NPZ: {npz_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved publication note: {note_path}")


if __name__ == "__main__":
    main()
