#!/usr/bin/env python3
"""
Export 3D rabbit-geometry displacement error fields to ParaView-compatible VTU.

The script loads:
- inverse PINN checkpoint (displacement network and load scales)
- mapping checkpoint (sphere -> physical rabbit geometry)

Then it samples interior and boundary points in reference sphere, maps them to
physical coordinates, computes predicted/true displacement and error, and saves
VTU point-cloud files (VTK_VERTEX cells).
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch


torch.set_default_dtype(torch.float64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class MappingNet(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6, disp_cap: float = 0.45):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)
        self.log_scale = torch.nn.Parameter(torch.zeros(3))
        self.shift = torch.nn.Parameter(torch.zeros(3))
        self.raw_disp = torch.nn.Parameter(torch.tensor(0.0))
        self.disp_cap = disp_cap

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        disp = torch.tanh(self.net(y))
        disp_scale = self.disp_cap * torch.tanh(self.raw_disp)
        base = y + disp_scale * disp
        scale = torch.exp(self.log_scale).unsqueeze(0)
        return base * scale + self.shift.unsqueeze(0)


class VectorPINN(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def sample_interior_unit_ball(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    r = torch.rand((n, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
    return r * d


def sample_boundary_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    return d


def manufactured_displacement(x: torch.Tensor, load_scale: float) -> torch.Tensor:
    pi = math.pi
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    u1 = 0.055 * torch.sin(pi * x1) * torch.cos(0.5 * pi * x2)
    u2 = -0.047 * torch.sin(pi * x2) * torch.cos(0.5 * pi * x3)
    u3 = 0.039 * torch.sin(pi * x3) * torch.cos(0.5 * pi * x1)
    return load_scale * torch.cat([u1, u2, u3], dim=1)


def load_mapping(mapping_checkpoint: str, device: torch.device, dtype: torch.dtype) -> MappingNet:
    ckpt = torch.load(mapping_checkpoint, map_location=device)
    kwargs = ckpt["model_kwargs"]
    model = MappingNet(
        width=kwargs["width"],
        depth=kwargs["depth"],
        disp_cap=kwargs["disp_cap"],
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_inverse(inverse_checkpoint: str, device: torch.device, dtype: torch.dtype) -> Dict[str, object]:
    ckpt = torch.load(inverse_checkpoint, map_location=device)
    kwargs = ckpt["u_model_kwargs"]
    model = VectorPINN(width=kwargs["width"], depth=kwargs["depth"]).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["u_model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {
        "model": model,
        "checkpoint": ckpt,
    }


def load_metrics_mu_true(metrics_json: Optional[str], default_mu_true: float) -> float:
    if metrics_json is None:
        return default_mu_true
    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    return float(m.get("mu_true", default_mu_true))


def write_vtu_points(
    path: str,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray],
) -> None:
    n = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)  # VTK_VERTEX

    def array_to_ascii(arr: np.ndarray) -> str:
        arr2 = np.asarray(arr)
        return " ".join(map(str, arr2.reshape(-1)))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + array_to_ascii(points) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + array_to_ascii(connectivity) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + array_to_ascii(offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + array_to_ascii(cell_types) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = arr.shape[1]
            else:
                raise ValueError(f"PointData '{name}' has unsupported shape {arr.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + array_to_ascii(arr) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def export_for_load_scale(
    mapping: MappingNet,
    u_model: VectorPINN,
    load_scale: float,
    mu_true: float,
    n_interior: int,
    n_boundary: int,
    out_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    _ = mu_true  # kept for metadata compatibility; displacement truth uses load scale only.

    with torch.no_grad():
        y_int = sample_interior_unit_ball(n_interior, device=device, dtype=dtype)
        y_bc = sample_boundary_unit_sphere(n_boundary, device=device, dtype=dtype)

        x_int = mapping(y_int)
        x_bc = mapping(y_bc)

        u_pred_int = u_model(y_int)
        u_pred_bc = u_model(y_bc)

        u_true_int = manufactured_displacement(x_int, load_scale=load_scale)
        u_true_bc = manufactured_displacement(x_bc, load_scale=load_scale)

        u_err_int = u_pred_int - u_true_int
        u_err_bc = u_pred_bc - u_true_bc
        e_mag_int = torch.linalg.norm(u_err_int, dim=1)
        e_mag_bc = torch.linalg.norm(u_err_bc, dim=1)

        rel_l2 = torch.sqrt(
            torch.mean((u_err_int) ** 2) / torch.clamp(torch.mean(u_true_int**2), min=1e-12)
        ).item()

    suffix = f"load_{str(load_scale).replace('.', 'p')}"

    int_path = os.path.join(out_dir, f"rabbit_error_interior_{suffix}.vtu")
    bc_path = os.path.join(out_dir, f"rabbit_error_boundary_{suffix}.vtu")

    write_vtu_points(
        path=int_path,
        points=x_int.cpu().numpy(),
        point_data={
            "y_ref": y_int.cpu().numpy(),
            "u_pred": u_pred_int.cpu().numpy(),
            "u_true": u_true_int.cpu().numpy(),
            "u_error": u_err_int.cpu().numpy(),
            "u_error_mag": e_mag_int.cpu().numpy(),
        },
    )

    write_vtu_points(
        path=bc_path,
        points=x_bc.cpu().numpy(),
        point_data={
            "y_ref": y_bc.cpu().numpy(),
            "u_pred": u_pred_bc.cpu().numpy(),
            "u_true": u_true_bc.cpu().numpy(),
            "u_error": u_err_bc.cpu().numpy(),
            "u_error_mag": e_mag_bc.cpu().numpy(),
        },
    )

    return {
        "load_scale": float(load_scale),
        "interior_vtu": int_path,
        "boundary_vtu": bc_path,
        "interior_rel_l2": float(rel_l2),
        "interior_l2": float(torch.sqrt(torch.mean((u_err_int) ** 2)).item()),
        "interior_max_error": float(torch.max(e_mag_int).item()),
        "boundary_max_error": float(torch.max(e_mag_bc).item()),
        "boundary_mean_error": float(torch.mean(e_mag_bc).item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rabbit 3D error fields to ParaView VTU")
    parser.add_argument("--inverse-checkpoint", required=True)
    parser.add_argument("--mapping-checkpoint", default=None, help="Override mapping path from inverse checkpoint")
    parser.add_argument("--metrics-json", default=None, help="Optional inverse metrics JSON to read mu_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-interior", type=int, default=60000)
    parser.add_argument("--n-boundary", type=int, default=20000)
    parser.add_argument("--load-scales", default=None, help="Comma-separated list; default from inverse checkpoint")
    parser.add_argument("--mu-true", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    dtype = torch.float64

    inv = load_inverse(args.inverse_checkpoint, device=device, dtype=dtype)
    inv_ckpt = inv["checkpoint"]
    u_model = inv["model"]

    mapping_path = args.mapping_checkpoint or inv_ckpt.get("mapping_checkpoint")
    if mapping_path is None:
        raise ValueError("Mapping checkpoint path is missing. Provide --mapping-checkpoint.")
    mapping = load_mapping(mapping_path, device=device, dtype=dtype)

    if args.load_scales is None:
        scales = inv_ckpt.get("load_scales", [1.0])
    else:
        scales = [float(x.strip()) for x in args.load_scales.split(",") if x.strip()]
    if not scales:
        raise ValueError("No load scales available to export.")

    mu_true = load_metrics_mu_true(args.metrics_json, args.mu_true)

    os.makedirs(args.output_dir, exist_ok=True)
    summary: Dict[str, object] = {
        "inverse_checkpoint": args.inverse_checkpoint,
        "mapping_checkpoint": mapping_path,
        "metrics_json": args.metrics_json,
        "mu_true": mu_true,
        "n_interior": args.n_interior,
        "n_boundary": args.n_boundary,
        "exports": [],
    }

    for s in scales:
        stats = export_for_load_scale(
            mapping=mapping,
            u_model=u_model,
            load_scale=float(s),
            mu_true=mu_true,
            n_interior=args.n_interior,
            n_boundary=args.n_boundary,
            out_dir=args.output_dir,
            device=device,
            dtype=dtype,
        )
        summary["exports"].append(stats)
        print(
            f"load={s:.4g} | rel_l2={stats['interior_rel_l2']:.4e} | "
            f"max_err_int={stats['interior_max_error']:.4e} | "
            f"interior_vtu={stats['interior_vtu']}"
        )

    summary_path = os.path.join(args.output_dir, "rabbit_error_paraview_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
