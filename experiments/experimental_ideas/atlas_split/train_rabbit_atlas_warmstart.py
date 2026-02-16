#!/usr/bin/env python3
"""
Train a meshfree multi-chart atlas for rabbit reconstruction with warmstart.

This script extends train_rabbit_atlas.py by supporting:
- optional decoder/mask initialization from an existing atlas checkpoint
- optional old->new chart remapping (for adaptive split child initialization)
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


def warmstart_models(
    decoders: List[ChartDecoder],
    masks: List[MaskNet],
    init_checkpoint: Optional[str],
    split_map_json: Optional[str],
) -> Dict[str, object]:
    info: Dict[str, object] = {
        "enabled": False,
        "loaded": False,
        "source_checkpoint": init_checkpoint,
        "split_map": split_map_json,
        "copied_direct": 0,
        "copied_from_parent": 0,
        "skipped": 0,
    }
    if init_checkpoint is None:
        return info
    if not os.path.isfile(init_checkpoint):
        print(f"Warmstart checkpoint not found: {init_checkpoint}")
        return info

    ckpt = torch.load(init_checkpoint, map_location=torch.device("cpu"))
    dec_states = ckpt.get("decoder_states")
    mask_states = ckpt.get("mask_states")
    if not isinstance(dec_states, list) or not isinstance(mask_states, list):
        print("Warmstart checkpoint missing decoder_states/mask_states; skipping warmstart.")
        return info

    info["enabled"] = True
    n_old = min(len(dec_states), len(mask_states))
    n_new = len(decoders)

    new_parent: List[int] = list(range(n_new))
    if split_map_json is not None and os.path.isfile(split_map_json):
        with open(split_map_json, "r", encoding="utf-8") as f:
            sm = json.load(f)
        if isinstance(sm.get("new_parent"), list):
            npv = [int(x) for x in sm["new_parent"]]
            if len(npv) == n_new:
                new_parent = npv

    for i in range(n_new):
        src = int(new_parent[i])
        if 0 <= src < n_old:
            try:
                decoders[i].load_state_dict(dec_states[src], strict=False)
                masks[i].load_state_dict(mask_states[src], strict=False)
                if src == i:
                    info["copied_direct"] = int(info["copied_direct"]) + 1
                else:
                    info["copied_from_parent"] = int(info["copied_from_parent"]) + 1
            except Exception:
                info["skipped"] = int(info["skipped"]) + 1
        else:
            info["skipped"] = int(info["skipped"]) + 1

    info["loaded"] = True
    return info


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
    warm_info = warmstart_models(
        decoders=decoders,
        masks=masks,
        init_checkpoint=args.init_atlas_checkpoint,
        split_map_json=args.split_map_json,
    )

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

        loss_total = (
            args.w_recon * loss_recon
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
            print(
                f"Epoch {epoch}/{args.epochs} | total={loss_total.item():.4e} "
                f"recon={loss_recon.item():.3e} mask={loss_mask.item():.3e} "
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
    gate["passed"] = bool(
        gate["gate_coverage"]
        and gate["gate_overlap"]
        and gate["gate_foldover"]
        and gate["gate_rmse"]
    )

    ckpt = {
        "decoder_states": [m.state_dict() for m in decoders],
        "mask_states": [m.state_dict() for m in masks],
        "decoder_kwargs": {"width": args.width, "depth": args.depth},
        "mask_kwargs": {"width": args.mask_width, "depth": args.mask_depth},
        "atlas_data_path": args.atlas_data,
        "warmstart": warm_info,
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
    if bool(warm_info.get("enabled", False)):
        print("Warmstart summary")
        print(f"  direct copy:   {warm_info['copied_direct']}")
        print(f"  parent copy:   {warm_info['copied_from_parent']}")
        print(f"  skipped:       {warm_info['skipped']}")

    return {
        "checkpoint": ckpt_path,
        "gate_report": gate_path,
        "history": hist_path,
        "plot": plot_path,
        "warmstart": warm_info,
        "passed": gate["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rabbit atlas chart decoders and masks")
    parser.add_argument("--atlas-data", required=True, help="Path to rabbit_atlas_data.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-atlas-checkpoint", default=None, help="Optional source atlas checkpoint for warmstart")
    parser.add_argument("--split-map-json", default=None, help="Optional split map with new_parent remap")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_atlas(args)


if __name__ == "__main__":
    main()
