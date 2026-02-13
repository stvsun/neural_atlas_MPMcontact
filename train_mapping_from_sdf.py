#!/usr/bin/env python3
"""
Train a 3D sphere-to-domain mapping from meshfree geometry signals.

Supported domains:
- rabbit: learned neural SDF from train_sdf_rabbit.py
- star: analytic star-shaped implicit domain for Poisson benchmark

The script enforces map-quality gates and writes a metrics report used by
mapped-PINN solvers.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
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


class SDFNet(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MappingNet(torch.nn.Module):
    """
    Residual mapping with global positive scaling + translation.

    This is not strictly invertible by construction, so training uses Jacobian
    barrier and distortion objectives to maintain practical bijectivity.
    """

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


class DomainOracle:
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sdf_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_req = x.clone().detach().requires_grad_(True)
        phi = self.sdf(x_req)
        grad_phi = torch.autograd.grad(
            phi,
            x_req,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        return phi, grad_phi


class StarDomainOracle(DomainOracle):
    def __init__(self, amp: float = 0.20, k_theta: int = 7, k_phi: int = 6, base_r: float = 1.0):
        self.amp = amp
        self.k_theta = k_theta
        self.k_phi = k_phi
        self.base_r = base_r

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        r = torch.linalg.norm(x, dim=1)
        z = x[:, 2]
        theta = torch.acos(torch.clamp(z / torch.clamp(r, min=eps), min=-1.0, max=1.0))
        phi = torch.atan2(x[:, 1], x[:, 0])
        radius = self.base_r + self.amp * torch.sin(self.k_theta * theta) * torch.sin(self.k_phi * phi)
        return r - radius


class LearnedSDFDomain(DomainOracle):
    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
        ckpt = torch.load(checkpoint_path, map_location=device)
        kwargs = ckpt["model_kwargs"]
        self.model = SDFNet(width=kwargs["width"], depth=kwargs["depth"]).to(device=device, dtype=dtype)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.center = torch.tensor(ckpt["center"], device=device, dtype=dtype)
        self.scale = torch.tensor(float(ckpt["scale"]), device=device, dtype=dtype)

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.center.unsqueeze(0)) / self.scale
        phi_norm = self.model(x_norm)
        return phi_norm * self.scale


def sample_interior_unit_ball(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    r = torch.rand((n, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
    return r * d


def sample_boundary_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    d = torch.randn((n, 3), device=device, dtype=dtype)
    d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
    return d


def clamp_to_unit_ball(y: torch.Tensor, max_r: float = 0.999) -> torch.Tensor:
    r = torch.linalg.norm(y, dim=1, keepdim=True)
    scale = torch.where(r > max_r, max_r / torch.clamp(r, min=1e-12), torch.ones_like(r))
    return y * scale


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps)


def map_and_jacobian(model: MappingNet, y_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = y_in.clone().detach().requires_grad_(True)
    x = model(y)
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


def compute_A_and_metrics(jac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    det_j = torch.det(jac)
    inv_j = torch.linalg.inv(jac)
    a = det_j.abs().unsqueeze(-1).unsqueeze(-1) * (inv_j @ inv_j.transpose(1, 2))
    a = 0.5 * (a + a.transpose(1, 2))

    eps = 1e-12
    eye = torch.eye(3, device=jac.device, dtype=jac.dtype).unsqueeze(0)
    eig = torch.linalg.eigvalsh(a + eps * eye)
    lam_min = torch.clamp(eig[:, 0], min=eps)
    lam_max = torch.clamp(eig[:, -1], min=eps)
    kappa = lam_max / lam_min
    return a, det_j, kappa


@dataclass
class LossBundle:
    bij: torch.Tensor
    ellip: torch.Tensor
    iso: torch.Tensor
    smooth: torch.Tensor
    rhs: torch.Tensor
    dOmega: torch.Tensor


def mapping_losses(
    model: MappingNet,
    domain: DomainOracle,
    y_int: torch.Tensor,
    y_bc: torch.Tensor,
    delta: float,
    smooth_eps: float,
    w_boundary_normal: float,
    w_boundary_dist: float,
    w_inside: float,
) -> Tuple[LossBundle, Dict[str, float], Dict[str, torch.Tensor]]:
    x_int, _, jac_int = map_and_jacobian(model, y_int)
    a_int, det_int, kappa_int = compute_A_and_metrics(jac_int)

    loss_bij = torch.mean(torch.nn.functional.softplus(delta - det_int) ** 2)
    loss_ellip = torch.mean(torch.log(torch.clamp(kappa_int, min=1.0)) ** 2)

    tr_a = torch.diagonal(a_int, dim1=1, dim2=2).sum(dim=1) / 3.0
    iso_target = torch.eye(3, device=y_int.device, dtype=y_int.dtype).unsqueeze(0) * tr_a[:, None, None]
    loss_iso = torch.mean((a_int - iso_target) ** 2)

    noise = torch.randn_like(y_int)
    y_pert = clamp_to_unit_ball(y_int + smooth_eps * noise)
    x_pert, _, jac_pert = map_and_jacobian(model, y_pert)
    a_pert, det_pert, _ = compute_A_and_metrics(jac_pert)

    log_det = torch.log(torch.clamp(det_int.abs(), min=1e-12))
    log_det_pert = torch.log(torch.clamp(det_pert.abs(), min=1e-12))
    loss_smooth = (
        torch.mean((a_int - a_pert) ** 2) / (smooth_eps**2)
        + torch.mean((log_det - log_det_pert) ** 2) / (smooth_eps**2)
    )

    loss_rhs = torch.mean((log_det - torch.mean(log_det)) ** 2)

    x_bc, _, jac_bc = map_and_jacobian(model, y_bc)
    phi_bc, grad_phi_bc = domain.sdf_and_grad(x_bc)
    loss_bc_phi = torch.mean(phi_bc**2)

    n_ref = normalize_rows(y_bc)
    inv_jt = torch.linalg.inv(jac_bc).transpose(1, 2)
    n_map = normalize_rows(torch.bmm(inv_jt, n_ref.unsqueeze(-1)).squeeze(-1))
    n_dom = normalize_rows(grad_phi_bc)
    loss_bc_normal = torch.mean(1.0 - torch.sum(n_map * n_dom, dim=1))

    radial_stretch = torch.linalg.norm(torch.bmm(jac_bc, n_ref.unsqueeze(-1)).squeeze(-1), dim=1)
    log_stretch = torch.log(torch.clamp(radial_stretch, min=1e-12))
    loss_bc_dist = torch.mean((log_stretch - torch.mean(log_stretch)) ** 2)

    phi_int = domain.sdf(x_int)
    loss_inside = torch.mean(torch.nn.functional.softplus(phi_int / 0.02))

    loss_dOmega = loss_bc_phi + w_boundary_normal * loss_bc_normal + w_boundary_dist * loss_bc_dist + w_inside * loss_inside

    losses = LossBundle(
        bij=loss_bij,
        ellip=loss_ellip,
        iso=loss_iso,
        smooth=loss_smooth,
        rhs=loss_rhs,
        dOmega=loss_dOmega,
    )

    diagnostics = {
        "min_det": float(det_int.min().item()),
        "neg_det_ratio": float((det_int <= 0.0).double().mean().item()),
        "kappa_mean": float(torch.mean(kappa_int).item()),
        "kappa_p95": float(torch.quantile(kappa_int.detach(), 0.95).item()),
        "boundary_sdf_rmse": float(torch.sqrt(torch.mean(phi_bc**2)).item()),
        "boundary_normal_mismatch": float(loss_bc_normal.item()),
        "inside_violation_ratio": float((phi_int > 0.0).double().mean().item()),
    }

    tensors = {
        "x_int": x_int,
        "det_int": det_int,
        "kappa_int": kappa_int,
        "phi_bc": phi_bc,
    }
    return losses, diagnostics, tensors


def evaluate_gates(
    model: MappingNet,
    domain: DomainOracle,
    n_val_int: int,
    n_val_bc: int,
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    y_int = sample_interior_unit_ball(n_val_int, device=device, dtype=dtype)
    y_bc = sample_boundary_unit_sphere(n_val_bc, device=device, dtype=dtype)

    with torch.enable_grad():
        losses, diagnostics, _ = mapping_losses(
            model=model,
            domain=domain,
            y_int=y_int,
            y_bc=y_bc,
            delta=thresholds["det_delta"],
            smooth_eps=0.01,
            w_boundary_normal=0.2,
            w_boundary_dist=0.1,
            w_inside=0.2,
        )

    gates = {
        "gate_min_det": diagnostics["min_det"] > thresholds["min_det"],
        "gate_neg_det_ratio": diagnostics["neg_det_ratio"] <= thresholds["max_neg_det_ratio"],
        "gate_boundary_sdf": diagnostics["boundary_sdf_rmse"] <= thresholds["max_boundary_sdf_rmse"],
        "gate_boundary_normal": diagnostics["boundary_normal_mismatch"] <= thresholds["max_boundary_normal_mismatch"],
        "gate_kappa_p95": diagnostics["kappa_p95"] <= thresholds["max_kappa_p95"],
        "gate_inside_ratio": diagnostics["inside_violation_ratio"] <= thresholds["max_inside_violation_ratio"],
    }

    passed = all(gates.values())
    report = {
        "passed": passed,
        **diagnostics,
        **gates,
        "det_delta": thresholds["det_delta"],
    }

    _ = losses
    return report


def save_training_plot(history: Dict[str, list], out_path: str) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].semilogy(epochs, np.maximum(history["total"], 1e-16), label="total")
    axes[0].semilogy(epochs, np.maximum(history["bij"], 1e-16), label="bij")
    axes[0].semilogy(epochs, np.maximum(history["dOmega"], 1e-16), label="dOmega")
    axes[0].set_title("Primary Mapping Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(epochs, np.maximum(history["ellip"], 1e-16), label="ellip")
    axes[1].semilogy(epochs, np.maximum(history["iso"], 1e-16), label="iso")
    axes[1].semilogy(epochs, np.maximum(history["smooth"], 1e-16), label="smooth")
    axes[1].semilogy(epochs, np.maximum(history["rhs"], 1e-16), label="rhs")
    axes[1].set_title("Secondary Mapping Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_mapping(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cpu")
    dtype = torch.float64

    if args.domain == "rabbit":
        if args.sdf_checkpoint is None:
            raise ValueError("--sdf-checkpoint is required for --domain rabbit")
        domain = LearnedSDFDomain(checkpoint_path=args.sdf_checkpoint, device=device, dtype=dtype)
    elif args.domain == "star":
        domain = StarDomainOracle(
            amp=args.star_amp,
            k_theta=args.star_k_theta,
            k_phi=args.star_k_phi,
            base_r=args.star_base_radius,
        )
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    model = MappingNet(width=args.width, depth=args.depth, disp_cap=args.disp_cap).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "total": [],
        "bij": [],
        "ellip": [],
        "iso": [],
        "smooth": [],
        "rhs": [],
        "dOmega": [],
        "min_det": [],
        "neg_det_ratio": [],
        "kappa_p95": [],
        "boundary_sdf_rmse": [],
        "boundary_normal_mismatch": [],
        "inside_violation_ratio": [],
    }

    alpha = {
        "bij": args.alpha_bij,
        "ellip": args.alpha_ellip,
        "iso": args.alpha_iso,
        "smooth": args.alpha_smooth,
        "rhs": args.alpha_rhs,
        "dOmega": args.alpha_dOmega,
    }

    warmup_epochs = int(max(0, args.warmup_epochs))
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        y_int = sample_interior_unit_ball(args.n_int, device=device, dtype=dtype)
        y_bc = sample_boundary_unit_sphere(args.n_bc, device=device, dtype=dtype)

        losses, diagnostics, _ = mapping_losses(
            model=model,
            domain=domain,
            y_int=y_int,
            y_bc=y_bc,
            delta=args.det_delta,
            smooth_eps=args.smooth_eps,
            w_boundary_normal=args.w_boundary_normal,
            w_boundary_dist=args.w_boundary_dist,
            w_inside=args.w_inside,
        )

        if epoch <= warmup_epochs:
            alpha_eff = {
                "bij": alpha["bij"] * 1.0,
                "ellip": alpha["ellip"] * 0.0,
                "iso": alpha["iso"] * 0.0,
                "smooth": alpha["smooth"] * 0.2,
                "rhs": alpha["rhs"] * 0.2,
                "dOmega": alpha["dOmega"] * 1.2,
            }
        else:
            alpha_eff = alpha

        loss_total = (
            alpha_eff["bij"] * losses.bij
            + alpha_eff["ellip"] * losses.ellip
            + alpha_eff["iso"] * losses.iso
            + alpha_eff["smooth"] * losses.smooth
            + alpha_eff["rhs"] * losses.rhs
            + alpha_eff["dOmega"] * losses.dOmega
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        history["total"].append(float(loss_total.item()))
        history["bij"].append(float(losses.bij.item()))
        history["ellip"].append(float(losses.ellip.item()))
        history["iso"].append(float(losses.iso.item()))
        history["smooth"].append(float(losses.smooth.item()))
        history["rhs"].append(float(losses.rhs.item()))
        history["dOmega"].append(float(losses.dOmega.item()))

        history["min_det"].append(diagnostics["min_det"])
        history["neg_det_ratio"].append(diagnostics["neg_det_ratio"])
        history["kappa_p95"].append(diagnostics["kappa_p95"])
        history["boundary_sdf_rmse"].append(diagnostics["boundary_sdf_rmse"])
        history["boundary_normal_mismatch"].append(diagnostics["boundary_normal_mismatch"])
        history["inside_violation_ratio"].append(diagnostics["inside_violation_ratio"])

        if epoch % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch}/{args.epochs} | total={loss_total.item():.4e} "
                f"bij={losses.bij.item():.3e} dOmega={losses.dOmega.item():.3e} "
                f"min_det={diagnostics['min_det']:.3e} neg_ratio={diagnostics['neg_det_ratio']:.2e} "
                f"b_sdf={diagnostics['boundary_sdf_rmse']:.3e} k95={diagnostics['kappa_p95']:.2f} "
                f"time={elapsed:.1f}s"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, f"mapping_{args.domain}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {"width": args.width, "depth": args.depth, "disp_cap": args.disp_cap},
            "domain": args.domain,
            "domain_meta": {
                "sdf_checkpoint": args.sdf_checkpoint,
                "star_amp": args.star_amp,
                "star_k_theta": args.star_k_theta,
                "star_k_phi": args.star_k_phi,
                "star_base_radius": args.star_base_radius,
            },
            "history": history,
        },
        ckpt_path,
    )

    thresholds = {
        "det_delta": args.det_delta,
        "min_det": args.gate_min_det,
        "max_neg_det_ratio": args.gate_max_neg_det_ratio,
        "max_boundary_sdf_rmse": args.gate_max_boundary_sdf_rmse,
        "max_boundary_normal_mismatch": args.gate_max_boundary_normal_mismatch,
        "max_kappa_p95": args.gate_max_kappa_p95,
        "max_inside_violation_ratio": args.gate_max_inside_violation_ratio,
    }

    gate_report = evaluate_gates(
        model=model,
        domain=domain,
        n_val_int=args.gate_val_int,
        n_val_bc=args.gate_val_bc,
        thresholds=thresholds,
    )

    gate_report["thresholds"] = thresholds
    gate_report["checkpoint"] = ckpt_path
    gate_report["domain"] = args.domain
    gate_report["epochs"] = args.epochs
    gate_report["train_time_sec"] = float(time.time() - start)

    metrics_path = os.path.join(args.output_dir, f"mapping_{args.domain}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)

    history_path = os.path.join(args.output_dir, f"mapping_{args.domain}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join(args.output_dir, f"mapping_{args.domain}_training_curves.png")
    save_training_plot(history=history, out_path=plot_path)

    print("\nMapping artifacts")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  metrics:    {metrics_path}")
    print(f"  history:    {history_path}")
    print(f"  plot:       {plot_path}")
    print("\nGate report")
    for k, v in gate_report.items():
        if k == "thresholds":
            continue
        print(f"  {k}: {v}")

    return gate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sphere-to-domain mapping from learned SDF or analytic implicit domain")
    parser.add_argument("--domain", choices=["rabbit", "star"], default="rabbit")
    parser.add_argument("--sdf-checkpoint", default=None, help="Required for rabbit domain")
    parser.add_argument("--output-dir", default="runs/mapping_3d")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--disp-cap", type=float, default=0.45)
    parser.add_argument("--n-int", type=int, default=2048)
    parser.add_argument("--n-bc", type=int, default=1024)
    parser.add_argument("--warmup-epochs", type=int, default=1200)

    parser.add_argument("--alpha-bij", type=float, default=25.0)
    parser.add_argument("--alpha-ellip", type=float, default=1.0)
    parser.add_argument("--alpha-iso", type=float, default=1.0)
    parser.add_argument("--alpha-smooth", type=float, default=0.2)
    parser.add_argument("--alpha-rhs", type=float, default=1.0)
    parser.add_argument("--alpha-dOmega", type=float, default=15.0)

    parser.add_argument("--det-delta", type=float, default=1e-3)
    parser.add_argument("--smooth-eps", type=float, default=0.015)
    parser.add_argument("--w-boundary-normal", type=float, default=0.4)
    parser.add_argument("--w-boundary-dist", type=float, default=0.2)
    parser.add_argument("--w-inside", type=float, default=0.3)

    parser.add_argument("--star-amp", type=float, default=0.17)
    parser.add_argument("--star-k-theta", type=int, default=7)
    parser.add_argument("--star-k-phi", type=int, default=6)
    parser.add_argument("--star-base-radius", type=float, default=1.0)

    parser.add_argument("--gate-min-det", type=float, default=1e-4)
    parser.add_argument("--gate-max-neg-det-ratio", type=float, default=0.0)
    parser.add_argument("--gate-max-boundary-sdf-rmse", type=float, default=2.5e-2)
    parser.add_argument("--gate-max-boundary-normal-mismatch", type=float, default=0.18)
    parser.add_argument("--gate-max-kappa-p95", type=float, default=35.0)
    parser.add_argument("--gate-max-inside-violation-ratio", type=float, default=2e-2)
    parser.add_argument("--gate-val-int", type=int, default=8000)
    parser.add_argument("--gate-val-bc", type=int, default=4000)

    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_mapping(args)


if __name__ == "__main__":
    main()
