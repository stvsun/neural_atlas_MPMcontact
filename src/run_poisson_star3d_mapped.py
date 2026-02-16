#!/usr/bin/env python3
"""
Solve mapped 3D Poisson on unit ball using a pretrained sphere->star mapping.

Hard requirement:
- mapping gate metrics must pass before training starts.
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, Tuple

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


class ScalarPINN(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_interior_unit_ball(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    r = torch.rand((n, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
    return r * d


def sample_boundary_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    return d


def load_mapping(mapping_checkpoint: str, device: torch.device, dtype: torch.dtype) -> MappingNet:
    ckpt = torch.load(mapping_checkpoint, map_location=device)
    kwargs = ckpt["model_kwargs"]
    model = MappingNet(width=kwargs["width"], depth=kwargs["depth"], disp_cap=kwargs["disp_cap"]).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def ensure_mapping_passed(mapping_metrics_path: str) -> Dict[str, float]:
    with open(mapping_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    if not metrics.get("passed", False):
        raise RuntimeError(
            "Mapping gate check failed. Refusing to run Poisson solver. "
            f"See metrics: {mapping_metrics_path}"
        )
    return metrics


def map_and_jacobian(mapping: MappingNet, y_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = y_in.clone().detach().requires_grad_(True)
    x = mapping(y)
    grads = []
    for i in range(3):
        gi = torch.autograd.grad(
            x[:, i],
            y,
            grad_outputs=torch.ones_like(x[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(gi)
    jac = torch.stack(grads, dim=1)
    return x, y, jac


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return (
        torch.sin(pi * x[:, 0:1])
        * torch.sin(pi * x[:, 1:2])
        * torch.sin(pi * x[:, 2:3])
    )


def source_term(x: torch.Tensor) -> torch.Tensor:
    return (3.0 * (math.pi**2)) * exact_solution(x)


def mapped_poisson_residual(u_model: ScalarPINN, mapping: MappingNet, y: torch.Tensor) -> torch.Tensor:
    x, y_var, jac = map_and_jacobian(mapping, y)

    det_j = torch.det(jac)
    inv_j = torch.linalg.inv(jac)
    a = det_j.abs().unsqueeze(-1).unsqueeze(-1) * (inv_j @ inv_j.transpose(1, 2))

    u = u_model(y_var)
    grad_u = torch.autograd.grad(
        u,
        y_var,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    flux = torch.bmm(a, grad_u.unsqueeze(-1)).squeeze(-1)
    div_flux = 0.0
    for j in range(3):
        dflux_j = torch.autograd.grad(
            flux[:, j],
            y_var,
            grad_outputs=torch.ones_like(flux[:, j]),
            create_graph=True,
            retain_graph=True,
        )[0][:, j : j + 1]
        div_flux = div_flux + dflux_j

    rhs = det_j.abs().unsqueeze(-1) * source_term(x)
    residual = -div_flux - rhs
    return residual


def evaluate(
    u_model: ScalarPINN,
    mapping: MappingNet,
    n_eval: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    y = sample_interior_unit_ball(n_eval, device=device, dtype=dtype)
    y = y.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        x = mapping(y)
        u_pred = u_model(y)
        u_ex = exact_solution(x)
        abs_err = torch.abs(u_pred - u_ex)
        l2 = torch.sqrt(torch.mean((u_pred - u_ex) ** 2)).item()
        rel_l2 = torch.sqrt(
            torch.mean((u_pred - u_ex) ** 2)
            / torch.clamp(torch.mean(u_ex**2), min=1e-12)
        ).item()
        max_err = torch.max(abs_err).item()
    return {
        "l2_error": float(l2),
        "relative_l2_error": float(rel_l2),
        "max_error": float(max_err),
    }


def make_plots(
    u_model: ScalarPINN,
    mapping: MappingNet,
    history: Dict[str, list],
    out_path: str,
    grid_n: int = 180,
) -> None:
    device = next(u_model.parameters()).device
    dtype = next(u_model.parameters()).dtype

    line = np.linspace(-1.0, 1.0, grid_n)
    Y1, Y2 = np.meshgrid(line, line)
    mask = (Y1**2 + Y2**2) <= 1.0

    pred = np.full_like(Y1, np.nan, dtype=float)
    exact = np.full_like(Y1, np.nan, dtype=float)

    if np.any(mask):
        y_pts = np.column_stack([Y1[mask], Y2[mask], np.zeros(np.sum(mask), dtype=float)])
        y_t = torch.tensor(y_pts, device=device, dtype=dtype)
        with torch.no_grad():
            x = mapping(y_t)
            u_p = u_model(y_t)
            u_e = exact_solution(x)
        pred[mask] = u_p.cpu().numpy().reshape(-1)
        exact[mask] = u_e.cpu().numpy().reshape(-1)

    err = np.abs(pred - exact)

    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im0 = axes[0, 0].contourf(Y1, Y2, pred, levels=25, cmap="viridis")
    axes[0, 0].set_title("Predicted u(y1,y2,z=0)")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].contourf(Y1, Y2, exact, levels=25, cmap="viridis")
    axes[0, 1].set_title("Exact u(psi(y1,y2,0))")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].contourf(Y1, Y2, err, levels=25, cmap="magma")
    axes[1, 0].set_title("Absolute error")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1, 0])

    axes[1, 1].semilogy(epochs, np.maximum(history["total"], 1e-16), label="total")
    axes[1, 1].semilogy(epochs, np.maximum(history["pde"], 1e-16), label="pde")
    axes[1, 1].semilogy(epochs, np.maximum(history["bc"], 1e-16), label="bc")
    axes[1, 1].set_title("Poisson mapped-PINN training losses")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cpu")
    dtype = torch.float64

    map_metrics = ensure_mapping_passed(args.mapping_metrics)
    mapping = load_mapping(args.mapping_checkpoint, device=device, dtype=dtype)
    u_model = ScalarPINN(width=args.width, depth=args.depth).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(u_model.parameters(), lr=args.lr)

    history = {"total": [], "pde": [], "bc": []}
    start = time.time()

    stop_reason = "max_epochs"
    best_total = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        y_int = sample_interior_unit_ball(args.n_int, device=device, dtype=dtype)
        y_bc = sample_boundary_unit_sphere(args.n_bc, device=device, dtype=dtype)

        res = mapped_poisson_residual(u_model=u_model, mapping=mapping, y=y_int)
        loss_pde = torch.mean(res**2)

        with torch.no_grad():
            x_bc = mapping(y_bc)
            u_bc_target = exact_solution(x_bc)
        u_bc_pred = u_model(y_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)

        loss_total = loss_pde + args.bc_weight * loss_bc
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(u_model.parameters(), max_norm=5.0)
        optimizer.step()

        history["total"].append(float(loss_total.item()))
        history["pde"].append(float(loss_pde.item()))
        history["bc"].append(float(loss_bc.item()))

        if loss_total.item() < best_total - args.min_delta:
            best_total = float(loss_total.item())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch}/{args.epochs} | total={loss_total.item():.4e} "
                f"pde={loss_pde.item():.3e} bc={loss_bc.item():.3e} "
                f"time={elapsed:.1f}s"
            )

        if args.target_loss is not None and loss_total.item() <= args.target_loss:
            stop_reason = f"target_loss({args.target_loss:.2e}) reached"
            break

        if bad_epochs >= args.patience:
            stop_reason = f"plateau(patience={args.patience}, min_delta={args.min_delta})"
            break

    train_time = time.time() - start

    metrics = evaluate(u_model=u_model, mapping=mapping, n_eval=args.n_eval, device=device, dtype=dtype)
    metrics["final_total_loss"] = history["total"][-1]
    metrics["epochs_ran"] = len(history["total"])
    metrics["train_time_sec"] = train_time
    metrics["stop_reason"] = stop_reason
    metrics["mapping_min_det"] = map_metrics.get("min_det")
    metrics["mapping_boundary_sdf_rmse"] = map_metrics.get("boundary_sdf_rmse")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_path = os.path.join(args.output_dir, "poisson_star3d_mapped_results.png")
    make_plots(u_model=u_model, mapping=mapping, history=history, out_path=plot_path)

    metrics_path = os.path.join(args.output_dir, "poisson_star3d_mapped_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.output_dir, "poisson_star3d_pinn.pt")
    torch.save(
        {
            "model_state": u_model.state_dict(),
            "model_kwargs": {"width": args.width, "depth": args.depth},
            "metrics": metrics,
            "history": history,
            "mapping_checkpoint": args.mapping_checkpoint,
            "mapping_metrics": args.mapping_metrics,
        },
        model_path,
    )

    print("\nPoisson mapped-PINN artifacts")
    print(f"  plot:    {plot_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  model:   {model_path}")
    print("\nFinal metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mapped Poisson PINN on star domain")
    parser.add_argument("--mapping-checkpoint", required=True)
    parser.add_argument("--mapping-metrics", required=True)
    parser.add_argument("--output-dir", default="runs/poisson_star3d_mapped")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-int", type=int, default=1024)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--bc-weight", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)

    parser.add_argument("--target-loss", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1200)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--n-eval", type=int, default=10000)

    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
