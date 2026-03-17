#!/usr/bin/env python3
"""
Mapped inverse neo-Hookean PINN on rabbit-like domain.

Pipeline assumptions:
- A mapping checkpoint exists and passed gate validation.
- Mapping is frozen during inverse solve.
- Unknown parameter: global shear modulus mu.
- Bulk modulus K is fixed.
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

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


class VectorPINN(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


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
            "Mapping gate check failed. Refusing to run inverse solver. "
            f"See metrics: {mapping_metrics_path}"
        )
    return metrics


def sample_interior_unit_ball(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    r = torch.rand((n, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
    return r * d


def sample_boundary_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    return d


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps)


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


def neo_hookean_p(F: torch.Tensor, mu: torch.Tensor, K: float) -> torch.Tensor:
    det_f = torch.det(F)
    det_safe = torch.clamp(det_f, min=1e-8)
    finv_t = torch.linalg.inv(F).transpose(1, 2)
    log_j = torch.log(det_safe).unsqueeze(-1).unsqueeze(-1)
    return mu * (F - finv_t) + K * log_j * finv_t


def manufactured_displacement(x: torch.Tensor, load_scale: float) -> torch.Tensor:
    pi = math.pi
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    u1 = 0.055 * torch.sin(pi * x1) * torch.cos(0.5 * pi * x2)
    u2 = -0.047 * torch.sin(pi * x2) * torch.cos(0.5 * pi * x3)
    u3 = 0.039 * torch.sin(pi * x3) * torch.cos(0.5 * pi * x1)
    return load_scale * torch.cat([u1, u2, u3], dim=1)


def manufactured_body_force(x: torch.Tensor, load_scale: float, mu_true: float, K: float) -> torch.Tensor:
    with torch.enable_grad():
        x_var = x.clone().detach().requires_grad_(True)
        u_true = manufactured_displacement(x_var, load_scale)
        grad_u = gradient_tensor(u_true, x_var, create_graph=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        F = eye + grad_u
        mu_t = torch.tensor(mu_true, device=x.device, dtype=x.dtype)
        P = neo_hookean_p(F, mu_t, K)

        div_P = []
        for i in range(3):
            comp = 0.0
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
    load_scale: float,
    mu_true: float,
    K: float,
) -> torch.Tensor:
    with torch.enable_grad():
        x_var = x.clone().detach().requires_grad_(True)
        u_true = manufactured_displacement(x_var, load_scale)
        grad_u = gradient_tensor(u_true, x_var, create_graph=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        F = eye + grad_u
        mu_t = torch.tensor(mu_true, device=x.device, dtype=x.dtype)
        P = neo_hookean_p(F, mu_t, K)
        t = torch.bmm(P, n_phys.unsqueeze(-1)).squeeze(-1)
    return t.detach()


def parse_load_scales(scales: str) -> List[float]:
    values = [float(x.strip()) for x in scales.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one load scale is required")
    return values


def build_observations(
    mapping: MappingNet,
    load_scales: List[float],
    n_obs_bc: int,
    n_obs_int: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[float, Dict[str, torch.Tensor]]:
    obs: Dict[float, Dict[str, torch.Tensor]] = {}
    for scale in load_scales:
        y_bc = sample_boundary_unit_sphere(n_obs_bc, device=device, dtype=dtype)
        y_int = sample_interior_unit_ball(n_obs_int, device=device, dtype=dtype)
        with torch.no_grad():
            x_bc = mapping(y_bc)
            x_int = mapping(y_int)
            u_bc = manufactured_displacement(x_bc, scale)
            u_int = manufactured_displacement(x_int, scale)
        obs[scale] = {
            "y_bc": y_bc,
            "y_int": y_int,
            "u_bc": u_bc,
            "u_int": u_int,
        }
    return obs


def mapped_equilibrium_loss(
    u_model: VectorPINN,
    mapping: MappingNet,
    y_int: torch.Tensor,
    mu_est: torch.Tensor,
    K: float,
    load_scale: float,
    mu_true: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y_var, jac = map_and_jacobian(mapping, y_int)
    inv_j = torch.linalg.inv(jac)
    det_j = torch.det(jac)
    det_abs = det_j.abs()

    u_pred = u_model(y_var)
    grad_u_y = gradient_tensor(u_pred, y_var, create_graph=True)
    grad_u_x = torch.bmm(grad_u_y, inv_j)

    eye = torch.eye(3, device=y_int.device, dtype=y_int.dtype).unsqueeze(0)
    F_pred = eye + grad_u_x
    P_pred = neo_hookean_p(F_pred, mu_est, K)

    T = det_abs.unsqueeze(-1).unsqueeze(-1) * torch.bmm(P_pred, inv_j.transpose(1, 2))

    div_T = []
    for i in range(3):
        comp = 0.0
        for j in range(3):
            dT_ij = torch.autograd.grad(
                T[:, i, j],
                y_var,
                grad_outputs=torch.ones_like(T[:, i, j]),
                create_graph=True,
                retain_graph=True,
            )[0][:, j : j + 1]
            comp = comp + dT_ij
        div_T.append(comp)
    div_T = torch.cat(div_T, dim=1)

    b_true = manufactured_body_force(x.detach(), load_scale=load_scale, mu_true=mu_true, K=K)
    residual = div_T + det_abs.unsqueeze(-1) * b_true
    loss_eq = torch.mean(residual**2)

    det_f = torch.det(F_pred)
    loss_det_barrier = torch.mean(torch.nn.functional.softplus(1e-4 - det_f) ** 2)
    return loss_eq, loss_det_barrier


def mapped_traction_loss(
    u_model: VectorPINN,
    mapping: MappingNet,
    y_bc: torch.Tensor,
    mu_est: torch.Tensor,
    K: float,
    load_scale: float,
    mu_true: float,
) -> torch.Tensor:
    x, y_var, jac = map_and_jacobian(mapping, y_bc)
    inv_j = torch.linalg.inv(jac)

    n_ref = normalize_rows(y_bc)
    n_phys = normalize_rows(torch.bmm(inv_j.transpose(1, 2), n_ref.unsqueeze(-1)).squeeze(-1))

    u_pred = u_model(y_var)
    grad_u_y = gradient_tensor(u_pred, y_var, create_graph=True)
    grad_u_x = torch.bmm(grad_u_y, inv_j)

    eye = torch.eye(3, device=y_bc.device, dtype=y_bc.dtype).unsqueeze(0)
    F_pred = eye + grad_u_x
    P_pred = neo_hookean_p(F_pred, mu_est, K)
    t_pred = torch.bmm(P_pred, n_phys.unsqueeze(-1)).squeeze(-1)

    t_true = manufactured_traction(
        x=x.detach(),
        n_phys=n_phys.detach(),
        load_scale=load_scale,
        mu_true=mu_true,
        K=K,
    )
    return torch.mean((t_pred - t_true) ** 2)


def evaluate(
    u_model: VectorPINN,
    mapping: MappingNet,
    load_scales: List[float],
    mu_est: float,
    mu_true: float,
    n_eval: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    rel_l2_list = []
    max_err_list = []

    for scale in load_scales:
        y = sample_interior_unit_ball(n_eval, device=device, dtype=dtype)
        with torch.no_grad():
            x = mapping(y)
            u_pred = u_model(y)
            u_true = manufactured_displacement(x, scale)
            err = u_pred - u_true
            rel_l2 = torch.sqrt(
                torch.mean(err**2)
                / torch.clamp(torch.mean(u_true**2), min=1e-12)
            ).item()
            max_err = torch.max(torch.abs(err)).item()
        rel_l2_list.append(rel_l2)
        max_err_list.append(max_err)

    return {
        "mu_true": float(mu_true),
        "mu_est": float(mu_est),
        "mu_abs_error": float(abs(mu_est - mu_true)),
        "mu_rel_error_percent": float(100.0 * abs(mu_est - mu_true) / max(mu_true, 1e-12)),
        "disp_rel_l2_mean": float(np.mean(rel_l2_list)),
        "disp_rel_l2_std": float(np.std(rel_l2_list)),
        "disp_max_error_mean": float(np.mean(max_err_list)),
    }


def make_plot(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(epochs, np.maximum(history["total"], 1e-16), label="total")
    axes[0].semilogy(epochs, np.maximum(history["eq"], 1e-16), label="eq")
    axes[0].semilogy(epochs, np.maximum(history["bc"], 1e-16), label="bc/traction")
    axes[0].semilogy(epochs, np.maximum(history["data"], 1e-16), label="data")
    axes[0].semilogy(epochs, np.maximum(history["reg"], 1e-16), label="reg")
    axes[0].set_title("Inverse neo-Hookean losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["mu"], color="tab:red", linewidth=2)
    axes[1].set_title("Estimated shear modulus mu")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mu")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cpu")
    dtype = torch.float64

    _ = ensure_mapping_passed(args.mapping_metrics)
    mapping = load_mapping(args.mapping_checkpoint, device=device, dtype=dtype)

    load_scales = parse_load_scales(args.load_scales)
    obs = build_observations(
        mapping=mapping,
        load_scales=load_scales,
        n_obs_bc=args.n_obs_bc,
        n_obs_int=args.n_obs_int,
        device=device,
        dtype=dtype,
    )

    u_model = VectorPINN(width=args.width, depth=args.depth).to(device=device, dtype=dtype)

    mu_init_adj = max(args.mu_init - args.mu_min, 1e-6)
    mu_raw_init = math.log(math.expm1(mu_init_adj))
    mu_raw = torch.nn.Parameter(torch.tensor(mu_raw_init, device=device, dtype=dtype))

    optimizer = torch.optim.Adam(list(u_model.parameters()) + [mu_raw], lr=args.lr)

    history: Dict[str, List[float]] = {
        "total": [],
        "eq": [],
        "bc": [],
        "data": [],
        "reg": [],
        "mu": [],
    }

    start = time.time()
    best_total = float("inf")
    bad_epochs = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        mu_est = torch.nn.functional.softplus(mu_raw) + args.mu_min

        loss_eq_acc = 0.0
        loss_bc_acc = 0.0
        loss_data_acc = 0.0
        loss_reg_acc = 0.0

        for scale in load_scales:
            y_int = sample_interior_unit_ball(args.n_int, device=device, dtype=dtype)
            y_bc = sample_boundary_unit_sphere(args.n_bc, device=device, dtype=dtype)

            loss_eq, loss_det_barrier = mapped_equilibrium_loss(
                u_model=u_model,
                mapping=mapping,
                y_int=y_int,
                mu_est=mu_est,
                K=args.bulk_modulus,
                load_scale=scale,
                mu_true=args.mu_true,
            )
            loss_bc = mapped_traction_loss(
                u_model=u_model,
                mapping=mapping,
                y_bc=y_bc,
                mu_est=mu_est,
                K=args.bulk_modulus,
                load_scale=scale,
                mu_true=args.mu_true,
            )

            obs_pack = obs[scale]
            u_bc_pred = u_model(obs_pack["y_bc"])
            u_int_pred = u_model(obs_pack["y_int"])
            loss_data = 0.5 * (
                torch.mean((u_bc_pred - obs_pack["u_bc"]) ** 2)
                + torch.mean((u_int_pred - obs_pack["u_int"]) ** 2)
            )

            loss_reg = loss_det_barrier + args.mu_reg_weight * ((mu_est - args.mu_prior) / args.mu_prior) ** 2

            loss_eq_acc = loss_eq_acc + loss_eq
            loss_bc_acc = loss_bc_acc + loss_bc
            loss_data_acc = loss_data_acc + loss_data
            loss_reg_acc = loss_reg_acc + loss_reg

        n_load = float(len(load_scales))
        loss_eq_mean = loss_eq_acc / n_load
        loss_bc_mean = loss_bc_acc / n_load
        loss_data_mean = loss_data_acc / n_load
        loss_reg_mean = loss_reg_acc / n_load

        loss_total = (
            args.beta_eq * loss_eq_mean
            + args.beta_bc * loss_bc_mean
            + args.beta_data * loss_data_mean
            + args.beta_reg * loss_reg_mean
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(list(u_model.parameters()) + [mu_raw], max_norm=5.0)
        optimizer.step()

        total_val = float(loss_total.item())
        history["total"].append(total_val)
        history["eq"].append(float(loss_eq_mean.item()))
        history["bc"].append(float(loss_bc_mean.item()))
        history["data"].append(float(loss_data_mean.item()))
        history["reg"].append(float(loss_reg_mean.item()))
        history["mu"].append(float((torch.nn.functional.softplus(mu_raw) + args.mu_min).item()))

        if total_val < best_total - args.min_delta:
            best_total = total_val
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch}/{args.epochs} | total={total_val:.4e} "
                f"eq={history['eq'][-1]:.3e} bc={history['bc'][-1]:.3e} "
                f"data={history['data'][-1]:.3e} reg={history['reg'][-1]:.3e} "
                f"mu={history['mu'][-1]:.6f} time={elapsed:.1f}s"
            )

        if args.target_loss is not None and total_val <= args.target_loss:
            stop_reason = f"target_loss({args.target_loss:.2e}) reached"
            break

        if bad_epochs >= args.patience:
            stop_reason = f"plateau(patience={args.patience}, min_delta={args.min_delta})"
            break

    train_time = time.time() - start
    mu_est_final = float((torch.nn.functional.softplus(mu_raw) + args.mu_min).item())

    metrics = evaluate(
        u_model=u_model,
        mapping=mapping,
        load_scales=load_scales,
        mu_est=mu_est_final,
        mu_true=args.mu_true,
        n_eval=args.n_eval,
        device=device,
        dtype=dtype,
    )
    metrics["final_total_loss"] = history["total"][-1]
    metrics["epochs_ran"] = len(history["total"])
    metrics["train_time_sec"] = train_time
    metrics["stop_reason"] = stop_reason

    os.makedirs(args.output_dir, exist_ok=True)

    plot_path = os.path.join(args.output_dir, "rabbit_inverse_neohookean_mapped_results.png")
    make_plot(history=history, out_path=plot_path)

    metrics_path = os.path.join(args.output_dir, "rabbit_inverse_neohookean_mapped_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.output_dir, "rabbit_inverse_neohookean_pinn.pt")
    torch.save(
        {
            "u_model_state": u_model.state_dict(),
            "u_model_kwargs": {"width": args.width, "depth": args.depth},
            "mu_est": mu_est_final,
            "history": history,
            "metrics": metrics,
            "load_scales": load_scales,
            "mapping_checkpoint": args.mapping_checkpoint,
            "mapping_metrics": args.mapping_metrics,
        },
        model_path,
    )

    print("\nInverse mapped-PINN artifacts")
    print(f"  plot:    {plot_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  model:   {model_path}")
    print("\nFinal metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mapped inverse neo-Hookean PINN (rabbit)")
    parser.add_argument("--config", default=None, help="Optional simple YAML with top-level key:value entries")
    parser.add_argument("--mapping-checkpoint", required=False, default=None)
    parser.add_argument("--mapping-metrics", required=False, default=None)
    parser.add_argument("--output-dir", default="runs/rabbit_inverse_mapped")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)

    parser.add_argument("--n-int", type=int, default=384)
    parser.add_argument("--n-bc", type=int, default=256)
    parser.add_argument("--n-obs-bc", type=int, default=128)
    parser.add_argument("--n-obs-int", type=int, default=96)
    parser.add_argument("--n-eval", type=int, default=2500)

    parser.add_argument("--mu-true", type=float, default=1.8)
    parser.add_argument("--mu-init", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=1e-3)
    parser.add_argument("--mu-prior", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=20.0)
    parser.add_argument("--mu-reg-weight", type=float, default=1e-3)

    parser.add_argument("--load-scales", default="1.0,1.35")

    parser.add_argument("--beta-eq", type=float, default=1.0)
    parser.add_argument("--beta-bc", type=float, default=2.0)
    parser.add_argument("--beta-data", type=float, default=10.0)
    parser.add_argument("--beta-reg", type=float, default=0.5)

    parser.add_argument("--target-loss", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=1200)
    parser.add_argument("--min-delta", type=float, default=1e-7)

    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.config is not None:
        cfg = parse_simple_yaml(args.config)
        for key, val in cfg.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            current = getattr(args, attr)
            if isinstance(current, bool):
                parsed = val.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                parsed = int(val)
            elif isinstance(current, float):
                parsed = float(val)
            else:
                parsed = val
            setattr(args, attr, parsed)

    if args.mapping_checkpoint is None or args.mapping_metrics is None:
        raise ValueError("--mapping-checkpoint and --mapping-metrics are required (directly or via --config)")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
