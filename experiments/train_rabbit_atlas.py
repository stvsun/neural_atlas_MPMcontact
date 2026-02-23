#!/usr/bin/env python3
"""
Train a meshfree multi-chart atlas for rabbit reconstruction.

This script trains:
- local chart decoders Phi_i(xi) -> x
- chart validity masks m_i(xi)

and evaluates atlas gates before PDE solving.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

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


class ChartDecoder(torch.nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=3, width=width, depth=depth)
        self.raw_scale = torch.nn.Parameter(torch.tensor(-1.8, dtype=torch.float64))

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


# ---------------------------------------------------------------------------
# M6: SDF network for volumetric atlas training
# ---------------------------------------------------------------------------

class _SDFNetAtlasTrain(torch.nn.Module):
    """Thin SDF network wrapper used only during volumetric atlas training."""

    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_sdf_for_atlas(
    path: str,
    device: torch.device,
) -> Tuple["_SDFNetAtlasTrain", torch.Tensor, float]:
    """Load an SDF network for use in volumetric atlas training.

    Handles two checkpoint formats:
      Format A (train_sdf_rabbit.py): ``model_state``, ``model_kwargs``, ``center``, ``scale``
      Format B (simplified):          ``model``, ``width``, ``depth``, ``center``, ``scale``

    Returns (net, center_tensor, scale_float) — net is frozen on *device*.
    """
    ckpt = torch.load(path, map_location=device)
    if "model_state" in ckpt:
        kw = ckpt.get("model_kwargs", {})
        net = _SDFNetAtlasTrain(width=int(kw.get("width", 128)), depth=int(kw.get("depth", 6)))
        net.load_state_dict(ckpt["model_state"])
    else:
        net = _SDFNetAtlasTrain(width=int(ckpt.get("width", 128)), depth=int(ckpt.get("depth", 6)))
        net.load_state_dict(ckpt["model"])
    net.to(device)
    net.eval()
    net.requires_grad_(False)
    center = torch.tensor(ckpt["center"], dtype=torch.float64, device=device)
    scale = float(ckpt["scale"])
    return net, center, scale


def jacobian_det(decoder: ChartDecoder, xi: torch.Tensor, seed, t1, t2, n, chart_scale) -> torch.Tensor:
    xi_req = xi.clone().detach().requires_grad_(True)
    x_pred = decoder(xi_req, seed=seed, t1=t1, t2=t2, n=n, chart_scale=chart_scale)
    grads = []
    for j in range(3):
        gj = torch.autograd.grad(
            x_pred[:, j],
            xi_req,
            grad_outputs=torch.ones_like(x_pred[:, j]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(gj)
    J = torch.stack(grads, dim=1)
    return torch.det(J)


def save_curves(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for key in ["total", "recon", "mask", "overlap", "jac", "coverage"]:
        ax.semilogy(epochs, np.maximum(history[key], 1e-16), label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Atlas training losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def train_atlas(args: argparse.Namespace) -> Dict[str, object]:
    data = np.load(args.atlas_data)
    points = torch.tensor(data["points"], dtype=torch.float64)
    normals = torch.tensor(data["normals"], dtype=torch.float64)
    seeds = torch.tensor(data["seed_points"], dtype=torch.float64)
    t1 = torch.tensor(data["frame_t1"], dtype=torch.float64)
    t2 = torch.tensor(data["frame_t2"], dtype=torch.float64)
    nvec = torch.tensor(data["frame_n"], dtype=torch.float64)
    membership = torch.tensor(data["membership"].astype(np.int64), dtype=torch.int64)
    primary = torch.tensor(data["primary_chart"].astype(np.int64), dtype=torch.int64)
    support_r = torch.tensor(data["support_radii"], dtype=torch.float64)

    n_points, n_charts = membership.shape
    device = torch.device("cpu")
    points = points.to(device)
    normals = normals.to(device)
    seeds = seeds.to(device)
    t1 = t1.to(device)
    t2 = t2.to(device)
    nvec = nvec.to(device)
    membership = membership.to(device)
    primary = primary.to(device)
    support_r = support_r.to(device)

    # M6: optionally load SDF for volumetric training (L_domain replaces L_recon)
    sdf_net_vol: Optional[_SDFNetAtlasTrain] = None
    sdf_center_vol: Optional[torch.Tensor] = None
    sdf_scale_vol: float = 1.0
    if getattr(args, "volumetric", False):
        sdf_ckpt_path = getattr(args, "sdf_checkpoint", None)
        if not sdf_ckpt_path:
            raise RuntimeError("--volumetric requires --sdf-checkpoint <path>")
        print(f"[M6] Loading SDF for volumetric atlas training: {sdf_ckpt_path}")
        sdf_net_vol, sdf_center_vol, sdf_scale_vol = load_sdf_for_atlas(sdf_ckpt_path, device)
        print(f"[M6] Volumetric training enabled — L_recon replaced by L_domain.")

    chart_pos_idx: List[torch.Tensor] = []
    chart_neg_idx: List[torch.Tensor] = []
    for i in range(n_charts):
        pos = torch.where(membership[:, i] > 0)[0]
        neg = torch.where(membership[:, i] == 0)[0]
        chart_pos_idx.append(pos)
        chart_neg_idx.append(neg)

    overlap_idx = torch.where(torch.sum(membership, dim=1) > 1)[0]

    decoders = [ChartDecoder(width=args.width, depth=args.depth).to(device) for _ in range(n_charts)]
    masks = [MaskNet(width=args.mask_width, depth=args.mask_depth).to(device) for _ in range(n_charts)]

    params = []
    for m in decoders + masks:
        params.extend(list(m.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    history = {
        "total": [],
        "recon": [],
        "mask": [],
        "overlap": [],
        "jac": [],
        "coverage": [],
    }

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        loss_recon = torch.tensor(0.0, device=device)
        loss_mask = torch.tensor(0.0, device=device)
        loss_jac = torch.tensor(0.0, device=device)

        for i in range(n_charts):
            pos = chart_pos_idx[i]
            neg = chart_neg_idx[i]
            if pos.numel() == 0:
                continue

            bpos = min(args.batch_pos, int(pos.numel()))
            p_idx = pos[torch.randint(0, pos.numel(), (bpos,), device=device)]
            x_pos = points[p_idx]
            xi_pos = local_coords(x_pos, seeds[i], t1[i], t2[i], nvec[i])

            if getattr(args, "volumetric", False) and sdf_net_vol is not None:
                # M6 L_domain: sample random xi in [-r,r]^3, decode to physical x,
                # penalise points that exit the domain (SDF >= threshold).
                r_i = float(support_r[i].item())
                n_vol = int(getattr(args, "n_vol_sample", 2048))
                xi_rand = (
                    2.0 * torch.rand(n_vol, 3, device=device, dtype=torch.float64) - 1.0
                ) * r_i
                x_dec = decoders[i](
                    xi_rand,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                )
                # sdf_net_vol is frozen; gradient flows only through decoders[i]
                x_norm_dec = (x_dec - sdf_center_vol.unsqueeze(0)) / sdf_scale_vol
                sdf_vals = sdf_net_vol(x_norm_dec)
                viol = torch.nn.functional.relu(
                    sdf_vals - float(getattr(args, "sdf_threshold", 0.0))
                )
                loss_recon = loss_recon + torch.mean(viol ** 2)
            else:
                x_hat = decoders[i](
                    xi_pos,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                )
                loss_recon = loss_recon + torch.mean((x_hat - x_pos) ** 2)

            logits_pos = masks[i](xi_pos, chart_scale=support_r[i])
            loss_mask = loss_mask + torch.mean(torch.nn.functional.softplus(-logits_pos))

            if neg.numel() > 0:
                bneg = min(args.batch_neg, int(neg.numel()))
                n_idx = neg[torch.randint(0, neg.numel(), (bneg,), device=device)]
                x_neg = points[n_idx]
                xi_neg = local_coords(x_neg, seeds[i], t1[i], t2[i], nvec[i])
                logits_neg = masks[i](xi_neg, chart_scale=support_r[i])
                loss_mask = loss_mask + torch.mean(torch.nn.functional.softplus(logits_neg))

            bjac = min(args.batch_jac, xi_pos.shape[0])
            xi_j = xi_pos[:bjac]
            detJ = jacobian_det(
                decoders[i],
                xi_j,
                seed=seeds[i],
                t1=t1[i],
                t2=t2[i],
                n=nvec[i],
                chart_scale=support_r[i],
            )
            loss_jac = loss_jac + torch.mean(torch.nn.functional.softplus(args.det_delta - detJ) ** 2)

        if overlap_idx.numel() > 0:
            bo = min(args.batch_overlap, int(overlap_idx.numel()))
            o_idx = overlap_idx[torch.randint(0, overlap_idx.numel(), (bo,), device=device)]
            x_o = points[o_idx]

            preds = []
            active = []
            for i in range(n_charts):
                m = membership[o_idx, i] > 0
                if torch.any(m):
                    xi = local_coords(x_o[m], seeds[i], t1[i], t2[i], nvec[i])
                    xh = decoders[i](xi, seed=seeds[i], t1=t1[i], t2=t2[i], n=nvec[i], chart_scale=support_r[i])
                    preds.append((m, xh))
                    active.append(i)

            loss_overlap = torch.tensor(0.0, device=device)
            n_terms = 0
            for a in range(len(preds)):
                for b in range(a + 1, len(preds)):
                    ma, xa = preds[a]
                    mb, xb = preds[b]
                    both = ma & mb
                    if torch.any(both):
                        ixa = torch.where(ma)[0]
                        ixb = torch.where(mb)[0]
                        map_a = {int(k): idx for idx, k in enumerate(ixa.tolist())}
                        map_b = {int(k): idx for idx, k in enumerate(ixb.tolist())}
                        common = torch.where(both)[0].tolist()
                        if common:
                            pa = torch.stack([xa[map_a[c]] for c in common], dim=0)
                            pb = torch.stack([xb[map_b[c]] for c in common], dim=0)
                            loss_overlap = loss_overlap + torch.mean((pa - pb) ** 2)
                            n_terms += 1
            if n_terms > 0:
                loss_overlap = loss_overlap / n_terms
        else:
            loss_overlap = torch.tensor(0.0, device=device)

        bg = min(args.batch_global, n_points)
        g_idx = torch.randint(0, n_points, (bg,), device=device)
        xg = points[g_idx]
        pri = primary[g_idx]

        logits = []
        for i in range(n_charts):
            xi = local_coords(xg, seeds[i], t1[i], t2[i], nvec[i])
            logits.append(masks[i](xi, chart_scale=support_r[i]))
        logits_t = torch.stack(logits, dim=1)
        probs = torch.sigmoid(logits_t)

        coverage_term = torch.nn.functional.softplus(args.coverage_thresh - torch.max(probs, dim=1).values)
        loss_cov = torch.mean(coverage_term)

        pou = torch.softmax(logits_t, dim=1)
        nll = -torch.log(torch.clamp(pou[torch.arange(bg), pri], min=1e-12))
        loss_cov = loss_cov + torch.mean(nll)

        # M6: use w_domain instead of w_recon in volumetric mode
        w_recon_eff = (
            float(getattr(args, "w_domain", args.w_recon))
            if getattr(args, "volumetric", False)
            else args.w_recon
        )
        loss_total = (
            w_recon_eff * loss_recon
            + args.w_mask * loss_mask
            + args.w_overlap * loss_overlap
            + args.w_jac * loss_jac
            + args.w_coverage * loss_cov
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        history["total"].append(float(loss_total.item()))
        history["recon"].append(float(loss_recon.item()))
        history["mask"].append(float(loss_mask.item()))
        history["overlap"].append(float(loss_overlap.item()))
        history["jac"].append(float(loss_jac.item()))
        history["coverage"].append(float(loss_cov.item()))

        if epoch % max(1, args.log_every) == 0:
            elapsed = time.time() - start
            recon_label = "domain" if getattr(args, "volumetric", False) else "recon"
            print(
                f"Epoch {epoch}/{args.epochs} | total={loss_total.item():.4e} "
                f"{recon_label}={loss_recon.item():.3e} mask={loss_mask.item():.3e} "
                f"overlap={loss_overlap.item():.3e} jac={loss_jac.item():.3e} cov={loss_cov.item():.3e} "
                f"time={elapsed:.1f}s"
            )

    with torch.no_grad():
        logits_all = []
        for i in range(n_charts):
            xi = local_coords(points, seeds[i], t1[i], t2[i], nvec[i])
            logits_all.append(masks[i](xi, chart_scale=support_r[i]))
        logits_all = torch.stack(logits_all, dim=1)
        probs_all = torch.sigmoid(logits_all)
        weights_all = torch.softmax(logits_all, dim=1)

        coverage_ratio = float(torch.mean((torch.max(probs_all, dim=1).values > args.coverage_thresh).double()).item())

        xh_all = []
        for i in range(n_charts):
            xi = local_coords(points, seeds[i], t1[i], t2[i], nvec[i])
            xh = decoders[i](xi, seed=seeds[i], t1=t1[i], t2=t2[i], n=nvec[i], chart_scale=support_r[i])
            xh_all.append(xh)
        xh_all = torch.stack(xh_all, dim=1)
        x_blend = torch.sum(weights_all.unsqueeze(-1) * xh_all, dim=1)

        rmse = float(torch.sqrt(torch.mean((x_blend - points) ** 2)).item())

        if overlap_idx.numel() > 0:
            xh_overlap = xh_all[overlap_idx]
            mem_overlap = membership[overlap_idx] > 0
            consistency = []
            for r in range(mem_overlap.shape[0]):
                ids = torch.where(mem_overlap[r])[0]
                if ids.numel() > 1:
                    pts = xh_overlap[r, ids]
                    c = torch.mean(torch.linalg.norm(pts - torch.mean(pts, dim=0, keepdim=True), dim=1))
                    consistency.append(c)
            if consistency:
                overlap_consistency = float(torch.mean(torch.stack(consistency)).item())
            else:
                overlap_consistency = 0.0
        else:
            overlap_consistency = 0.0

        with torch.enable_grad():
            fold_viol = []
            total_j = 0
            for i in range(n_charts):
                pos = chart_pos_idx[i]
                if pos.numel() == 0:
                    continue
                b = min(args.fold_eval_points, int(pos.numel()))
                idx = pos[torch.randint(0, pos.numel(), (b,), device=device)]
                xi = local_coords(points[idx], seeds[i], t1[i], t2[i], nvec[i])
                detJ = jacobian_det(
                    decoders[i],
                    xi,
                    seed=seeds[i],
                    t1=t1[i],
                    t2=t2[i],
                    n=nvec[i],
                    chart_scale=support_r[i],
                )
                fold_viol.append(torch.sum(detJ <= 0).item())
                total_j += detJ.numel()
            fold_ratio = float(sum(fold_viol) / max(1, total_j))

    gate = {
        "coverage_ratio": coverage_ratio,
        "overlap_consistency": overlap_consistency,
        "foldover_ratio": fold_ratio,
        "boundary_rmse": rmse,
        "gate_coverage": coverage_ratio >= args.gate_coverage,
        "gate_overlap": overlap_consistency <= args.gate_overlap_consistency,
        "gate_foldover": fold_ratio <= args.gate_foldover_ratio,
        "gate_rmse": rmse <= args.gate_boundary_rmse,
    }
    # M6: in volumetric mode, gate_rmse is not meaningful (no surface ground truth)
    gate_rmse_check = True if getattr(args, "volumetric", False) else gate["gate_rmse"]
    gate["passed"] = bool(
        gate["gate_coverage"]
        and gate["gate_overlap"]
        and gate["gate_foldover"]
        and gate_rmse_check
    )

    ckpt = {
        "decoder_states": [m.state_dict() for m in decoders],
        "mask_states": [m.state_dict() for m in masks],
        "decoder_kwargs": {"width": args.width, "depth": args.depth},
        "mask_kwargs": {"width": args.mask_width, "depth": args.mask_depth},
        "atlas_data_path": args.atlas_data,
        "seed": args.seed,
        "gate": gate,
        "history": history,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "rabbit_atlas_trained.pt")
    torch.save(ckpt, ckpt_path)

    hist_path = os.path.join(args.output_dir, "rabbit_atlas_train_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    gate_path = os.path.join(args.output_dir, "rabbit_atlas_gate_report.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2)

    plot_path = os.path.join(args.output_dir, "rabbit_atlas_training_curves.png")
    save_curves(history, plot_path)

    print("Atlas training complete")
    print(f"  checkpoint:  {ckpt_path}")
    print(f"  gate_report: {gate_path}")
    print(f"  passed:      {gate['passed']}")
    print(f"  coverage:    {gate['coverage_ratio']:.4f}")
    print(f"  overlap:     {gate['overlap_consistency']:.4e}")
    print(f"  fold_ratio:  {gate['foldover_ratio']:.4e}")
    print(f"  rmse:        {gate['boundary_rmse']:.4e}")

    return {
        "checkpoint": ckpt_path,
        "gate_report": gate_path,
        "history": hist_path,
        "plot": plot_path,
        "passed": gate["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rabbit atlas chart decoders and masks")
    parser.add_argument("--atlas-data", required=True, help="Path to rabbit_atlas_data.npz")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mask-width", type=int, default=48)
    parser.add_argument("--mask-depth", type=int, default=3)

    parser.add_argument("--batch-pos", type=int, default=512)
    parser.add_argument("--batch-neg", type=int, default=512)
    parser.add_argument("--batch-jac", type=int, default=128)
    parser.add_argument("--batch-overlap", type=int, default=512)
    parser.add_argument("--batch-global", type=int, default=1024)
    parser.add_argument("--fold-eval-points", type=int, default=256)

    parser.add_argument("--det-delta", type=float, default=1e-3)
    parser.add_argument("--coverage-thresh", type=float, default=0.5)

    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--w-mask", type=float, default=0.5)
    parser.add_argument("--w-overlap", type=float, default=0.8)
    parser.add_argument("--w-jac", type=float, default=2.0)
    parser.add_argument("--w-coverage", type=float, default=0.7)

    parser.add_argument("--gate-coverage", type=float, default=0.99)
    parser.add_argument("--gate-overlap-consistency", type=float, default=2.5e-2)
    parser.add_argument("--gate-foldover-ratio", type=float, default=0.0)
    parser.add_argument("--gate-boundary-rmse", type=float, default=3.0e-2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)

    # M6: volumetric atlas training flags
    parser.add_argument(
        "--volumetric",
        action="store_true",
        default=False,
        help="M6: enable volumetric atlas training — replaces L_recon with L_domain "
             "(penalise decoder outputs that exit the SDF domain).",
    )
    parser.add_argument(
        "--sdf-checkpoint",
        default=None,
        help="M6: path to a trained SDF network checkpoint (.pt). Required with --volumetric.",
    )
    parser.add_argument(
        "--n-vol-sample",
        type=int,
        default=2048,
        help="M6: number of random xi samples per chart per epoch for L_domain.",
    )
    parser.add_argument(
        "--w-domain",
        type=float,
        default=1.0,
        help="M6: weight for L_domain (overrides --w-recon when --volumetric is set).",
    )
    parser.add_argument(
        "--sdf-threshold",
        type=float,
        default=0.0,
        help="M6: SDF acceptance threshold — penalise sdf_val >= this value (0 = exactly inside).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_atlas(args)


if __name__ == "__main__":
    main()
