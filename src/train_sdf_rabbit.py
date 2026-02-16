#!/usr/bin/env python3
"""
Train a meshfree neural SDF for rabbit-like geometry.

Workflow:
1) Load point cloud + normals from file (optional).
2) Fallback to a procedural rabbit-like point cloud if no file is provided.
3) Train neural SDF with surface, Eikonal, normal, and sign-anchor losses.
4) Save checkpoint usable by mapping and PINN scripts.
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

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


class SDFNet(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norm, eps)


def random_unit_vectors(n: int) -> np.ndarray:
    v = np.random.randn(n, 3)
    return normalize(v)


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown axis: {axis}")


def sample_ellipsoid_surface(
    n: int,
    center: np.ndarray,
    radii: np.ndarray,
    rot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dirs = random_unit_vectors(n)
    local = dirs * radii[None, :]
    pts = local @ rot.T + center[None, :]

    grad_local = local / (radii[None, :] ** 2)
    grad_world = grad_local @ rot.T
    normals = normalize(grad_world)
    return pts, normals


def generate_procedural_rabbit(n_surface: int) -> Tuple[np.ndarray, np.ndarray]:
    components = [
        {
            "center": np.array([0.00, 0.00, 0.00]),
            "radii": np.array([0.58, 0.36, 0.34]),
            "rot": rotation_matrix("z", 0.0),
            "weight": 0.50,
        },
        {
            "center": np.array([0.56, 0.09, 0.00]),
            "radii": np.array([0.27, 0.23, 0.22]),
            "rot": rotation_matrix("z", 0.10),
            "weight": 0.22,
        },
        {
            "center": np.array([0.75, 0.42, 0.12]),
            "radii": np.array([0.09, 0.33, 0.09]),
            "rot": rotation_matrix("z", 0.32) @ rotation_matrix("x", -0.25),
            "weight": 0.11,
        },
        {
            "center": np.array([0.74, 0.40, -0.12]),
            "radii": np.array([0.09, 0.31, 0.09]),
            "rot": rotation_matrix("z", 0.26) @ rotation_matrix("x", 0.25),
            "weight": 0.11,
        },
        {
            "center": np.array([-0.64, 0.09, 0.00]),
            "radii": np.array([0.12, 0.12, 0.12]),
            "rot": rotation_matrix("y", 0.0),
            "weight": 0.06,
        },
    ]

    weights = np.array([c["weight"] for c in components], dtype=float)
    weights = weights / np.sum(weights)
    counts = np.random.multinomial(n_surface, weights)

    all_pts = []
    all_nrm = []
    for c, n in zip(components, counts):
        if n == 0:
            continue
        pts, nrm = sample_ellipsoid_surface(
            n=n,
            center=c["center"],
            radii=c["radii"],
            rot=c["rot"],
        )
        all_pts.append(pts)
        all_nrm.append(nrm)

    pts = np.concatenate(all_pts, axis=0)
    nrm = np.concatenate(all_nrm, axis=0)

    centroid = np.mean(pts, axis=0, keepdims=True)
    orient = np.sum((pts - centroid) * nrm, axis=1) < 0
    nrm[orient] *= -1.0

    return pts, normalize(nrm)


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if path.endswith(".npz"):
        data = np.load(path)
        if "points" not in data:
            raise ValueError("NPZ must include key 'points'")
        points = np.asarray(data["points"], dtype=float)
        normals = np.asarray(data["normals"], dtype=float) if "normals" in data else None
    elif path.endswith(".npy"):
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("NPY must be shape [N,3] or [N,6]")
        points = arr[:, :3]
        normals = arr[:, 3:6] if arr.shape[1] == 6 else None
    else:
        arr = np.loadtxt(path, delimiter=",")
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("Text/CSV must be shape [N,3] or [N,6]")
        points = arr[:, :3]
        normals = arr[:, 3:6] if arr.shape[1] == 6 else None

    if normals is not None:
        normals = normalize(normals)
        centroid = np.mean(points, axis=0, keepdims=True)
        flip = np.sum((points - centroid) * normals, axis=1) < 0
        normals[flip] *= -1.0

    return points, normals


def save_slice_plot(
    model: SDFNet,
    center: np.ndarray,
    scale: float,
    out_path: str,
    grid_n: int = 220,
) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def eval_slice(axis: int, value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        line = np.linspace(-1.2, 1.2, grid_n)
        A, B = np.meshgrid(line, line)
        pts = np.zeros((grid_n * grid_n, 3), dtype=float)
        idx = [0, 1, 2]
        free = [i for i in idx if i != axis]
        pts[:, free[0]] = A.reshape(-1)
        pts[:, free[1]] = B.reshape(-1)
        pts[:, axis] = value

        pts_phys = pts * scale + center[None, :]
        pts_norm = torch.tensor(pts, device=device, dtype=dtype)
        with torch.no_grad():
            phi = model(pts_norm).cpu().numpy().reshape(grid_n, grid_n)
        return A, B, phi

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    settings = [(2, 0.0, "z=0"), (1, 0.0, "y=0"), (0, 0.0, "x=0")]
    for ax, (axis, val, title) in zip(axes, settings):
        A, B, phi = eval_slice(axis=axis, value=val)
        cf = ax.contourf(A, B, phi, levels=40, cmap="coolwarm")
        ax.contour(A, B, phi, levels=[0.0], colors="black", linewidths=1.5)
        ax.set_title(f"SDF slice ({title})")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, shrink=0.85)

    fig.suptitle("Learned Rabbit SDF (normalized coordinates)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_sdf(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cpu")
    dtype = torch.float64

    if args.points_file is not None:
        points, normals = load_point_cloud(args.points_file)
        source = f"file:{args.points_file}"
    else:
        points, normals = generate_procedural_rabbit(args.n_surface)
        source = "procedural_rabbit"

    if normals is None:
        centroid = np.mean(points, axis=0, keepdims=True)
        normals = normalize(points - centroid)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    scale = float(np.max(maxs - mins))
    scale = max(scale, 1e-6)

    points_norm = (points - center[None, :]) / scale
    normals_norm = normalize(normals)

    pts_t = torch.tensor(points_norm, device=device, dtype=dtype)
    nrm_t = torch.tensor(normals_norm, device=device, dtype=dtype)

    model = SDFNet(width=args.width, depth=args.depth).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "total": [],
        "surface": [],
        "eikonal": [],
        "normal": [],
        "sign": [],
    }

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        idx = torch.randint(0, pts_t.shape[0], (args.batch_surface,), device=device)
        p_surf = pts_t[idx]
        n_surf = nrm_t[idx]

        p_surf_req = p_surf.clone().detach().requires_grad_(True)
        phi_surf = model(p_surf_req)
        grad_phi_surf = torch.autograd.grad(
            phi_surf,
            p_surf_req,
            grad_outputs=torch.ones_like(phi_surf),
            create_graph=True,
        )[0]

        loss_surface = torch.mean(torch.abs(phi_surf))

        grad_norm = torch.linalg.norm(grad_phi_surf, dim=1, keepdim=True)
        grad_dir = grad_phi_surf / torch.clamp(grad_norm, min=1e-12)
        loss_normal = torch.mean(1.0 - torch.sum(grad_dir * n_surf, dim=1))

        p_eik = (torch.rand((args.batch_eikonal, 3), device=device, dtype=dtype) * 2.8) - 1.4
        p_eik_req = p_eik.clone().detach().requires_grad_(True)
        phi_eik = model(p_eik_req)
        grad_phi_eik = torch.autograd.grad(
            phi_eik,
            p_eik_req,
            grad_outputs=torch.ones_like(phi_eik),
            create_graph=True,
        )[0]
        loss_eik = torch.mean((torch.linalg.norm(grad_phi_eik, dim=1) - 1.0) ** 2)

        offset = args.anchor_offset
        p_in = p_surf - offset * n_surf
        p_out = p_surf + offset * n_surf
        target_in = -torch.ones((p_in.shape[0],), device=device, dtype=dtype)
        target_out = torch.ones((p_out.shape[0],), device=device, dtype=dtype)

        phi_in = model(p_in)
        phi_out = model(p_out)
        loss_sign_in = torch.mean(torch.nn.functional.softplus(-target_in * phi_in))
        loss_sign_out = torch.mean(torch.nn.functional.softplus(-target_out * phi_out))

        p_far = (torch.rand((args.batch_far, 3), device=device, dtype=dtype) * 6.0) - 3.0
        r_far = torch.linalg.norm(p_far, dim=1)
        mask_far = r_far > 1.65
        if torch.any(mask_far):
            phi_far = model(p_far[mask_far])
            loss_far = torch.mean(torch.nn.functional.softplus(-phi_far))
        else:
            loss_far = torch.tensor(0.0, device=device, dtype=dtype)

        loss_sign = 0.5 * (loss_sign_in + loss_sign_out) + 0.5 * loss_far

        loss_total = (
            args.w_surface * loss_surface
            + args.w_eikonal * loss_eik
            + args.w_normal * loss_normal
            + args.w_sign * loss_sign
        )

        loss_total.backward()
        optimizer.step()

        history["total"].append(float(loss_total.item()))
        history["surface"].append(float(loss_surface.item()))
        history["eikonal"].append(float(loss_eik.item()))
        history["normal"].append(float(loss_normal.item()))
        history["sign"].append(float(loss_sign.item()))

        if epoch % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"total={loss_total.item():.4e} "
                f"surface={loss_surface.item():.4e} "
                f"eikonal={loss_eik.item():.4e} "
                f"normal={loss_normal.item():.4e} "
                f"sign={loss_sign.item():.4e} "
                f"time={elapsed:.1f}s"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "rabbit_sdf.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {"width": args.width, "depth": args.depth},
            "center": center.tolist(),
            "scale": scale,
            "source": source,
            "history": history,
        },
        ckpt_path,
    )

    hist_path = os.path.join(args.output_dir, "rabbit_sdf_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join(args.output_dir, "rabbit_sdf_slices.png")
    save_slice_plot(model=model, center=center, scale=scale, out_path=plot_path)

    meta = {
        "checkpoint": ckpt_path,
        "history": hist_path,
        "plot": plot_path,
        "source": source,
        "n_surface": int(points.shape[0]),
        "center": center.tolist(),
        "scale": scale,
        "final_total": history["total"][-1],
        "final_surface": history["surface"][-1],
        "final_eikonal": history["eikonal"][-1],
        "final_normal": history["normal"][-1],
        "final_sign": history["sign"][-1],
    }
    meta_path = os.path.join(args.output_dir, "rabbit_sdf_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved artifacts")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  history:    {hist_path}")
    print(f"  plot:       {plot_path}")
    print(f"  meta:       {meta_path}")

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meshfree rabbit SDF from point cloud or procedural fallback")
    parser.add_argument("--points-file", default=None, help="Optional .npz/.npy/.csv/.txt point cloud path")
    parser.add_argument("--output-dir", default="runs/rabbit_sdf", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)

    parser.add_argument("--n-surface", type=int, default=20000, help="Used for procedural fallback")
    parser.add_argument("--batch-surface", type=int, default=2048)
    parser.add_argument("--batch-eikonal", type=int, default=2048)
    parser.add_argument("--batch-far", type=int, default=1024)
    parser.add_argument("--anchor-offset", type=float, default=0.025)

    parser.add_argument("--w-surface", type=float, default=10.0)
    parser.add_argument("--w-eikonal", type=float, default=1.0)
    parser.add_argument("--w-normal", type=float, default=1.0)
    parser.add_argument("--w-sign", type=float, default=2.0)

    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_sdf(args)


if __name__ == "__main__":
    main()
