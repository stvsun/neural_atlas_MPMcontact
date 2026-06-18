#!/usr/bin/env python3
"""
Atlas quality verification: distortion metrics, Jacobian determinant histograms,
coverage statistics, and transition-map consistency errors.

Loads a trained atlas checkpoint and produces a combined diagnostic report
with multi-panel figure for manuscript reviewer evidence.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Model classes (copied from core/train_rabbit_atlas.py — self-contained)
# ---------------------------------------------------------------------------

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
# Jacobian computation (full 3x3 matrix, not just determinant)
# ---------------------------------------------------------------------------

def compute_full_jacobian(
    decoder: ChartDecoder,
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute full 3x3 Jacobian J_i = D_xi phi_i at each sample point.

    Returns: (N, 3, 3) tensor where J[k, i, j] = d(x_i)/d(xi_j).
    """
    xi_req = xi.clone().detach().requires_grad_(True)
    x_pred = decoder(xi_req, seed=seed, t1=t1, t2=t2, n=n, chart_scale=chart_scale)
    grads = []
    for j in range(3):
        gj = torch.autograd.grad(
            x_pred[:, j],
            xi_req,
            grad_outputs=torch.ones_like(x_pred[:, j]),
            create_graph=False,
            retain_graph=True,
        )[0]
        grads.append(gj)
    J = torch.stack(grads, dim=1)  # (N, 3, 3)
    return J


def sample_ball(n_samples: int, radius: float) -> torch.Tensor:
    """Sample uniformly inside a 3D ball of given radius."""
    # Rejection sampling
    pts = []
    while len(pts) < n_samples:
        batch = torch.randn(n_samples * 2, 3)
        batch = batch / batch.norm(dim=1, keepdim=True)  # unit sphere
        r = torch.rand(n_samples * 2, 1) ** (1.0 / 3.0)  # uniform in ball
        batch = batch * r * radius
        pts.append(batch)
    return torch.cat(pts, dim=0)[:n_samples]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_atlas(
    atlas_checkpoint: str,
    atlas_data_path: str,
) -> Tuple[List[ChartDecoder], List[MaskNet], dict, dict]:
    """Load atlas checkpoint and geometry data."""
    ckpt = torch.load(atlas_checkpoint, map_location="cpu", weights_only=False)
    data = dict(np.load(atlas_data_path, allow_pickle=True))

    dec_kwargs = ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kwargs = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    n_charts = len(ckpt["decoder_states"])

    decoders = []
    for i in range(n_charts):
        dec = ChartDecoder(**dec_kwargs)
        dec.load_state_dict(ckpt["decoder_states"][i])
        dec.eval()
        decoders.append(dec)

    masks = []
    for i in range(n_charts):
        m = MaskNet(**mask_kwargs)
        m.load_state_dict(ckpt["mask_states"][i])
        m.eval()
        masks.append(m)

    atlas_geom = {
        "points": torch.tensor(data["points"], dtype=torch.float64),
        "normals": torch.tensor(data["normals"], dtype=torch.float64),
        "seeds": torch.tensor(data["seed_points"], dtype=torch.float64),
        "t1": torch.tensor(data["frame_t1"], dtype=torch.float64),
        "t2": torch.tensor(data["frame_t2"], dtype=torch.float64),
        "n": torch.tensor(data["frame_n"], dtype=torch.float64),
        "membership": torch.tensor(data["membership"].astype(np.int64), dtype=torch.int64),
        "support_r": torch.tensor(data["support_radii"], dtype=torch.float64),
    }

    return decoders, masks, atlas_geom, ckpt


# ---------------------------------------------------------------------------
# 1. Jacobian determinant histograms + distortion metrics
# ---------------------------------------------------------------------------

def _jacobian_stats_for_points(
    decoder: ChartDecoder,
    xi: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
    chart_idx: int,
) -> Dict:
    """Compute Jacobian statistics for a set of reference-coordinate samples."""
    with torch.enable_grad():
        J = compute_full_jacobian(decoder, xi, seed=seed, t1=t1, t2=t2,
                                  n=n, chart_scale=chart_scale)

    with torch.no_grad():
        det_J = torch.det(J)
        svd_vals = torch.linalg.svdvals(J)  # (N, 3), descending
        kappa = svd_vals[:, 0] / svd_vals[:, 2].clamp(min=1e-12)

        return {
            "chart": chart_idx,
            "n_samples": int(xi.shape[0]),
            "det_J": det_J.numpy(),
            "det_min": float(det_J.min()),
            "det_max": float(det_J.max()),
            "det_mean": float(det_J.mean()),
            "det_std": float(det_J.std()),
            "foldover_frac": float((det_J <= 0).float().mean()),
            "foldover_count": int((det_J <= 0).sum()),
            "kappa_mean": float(kappa.mean()),
            "kappa_median": float(kappa.median()),
            "kappa_p95": float(kappa.quantile(0.95)),
            "kappa_max": float(kappa.max()),
            "isotropy_mean": float(kappa.mean()),
            "isotropy_max": float(kappa.max()),
            "area_distortion": float((det_J - 1.0).abs().mean()),
            "svd_vals": svd_vals.numpy(),
        }


def compute_jacobian_metrics(
    decoders: List[ChartDecoder],
    atlas_geom: dict,
    n_samples: int = 5000,
) -> Tuple[List[Dict], List[Dict]]:
    """Per-chart Jacobian statistics from both dense ball and membership-filtered sampling.

    Returns:
        (ball_results, membership_results) — ball samples the full reference domain;
        membership restricts to points the SDF/mask identifies as inside the physical body.
    """
    n_charts = len(decoders)
    ball_results = []
    mem_results = []
    membership = atlas_geom["membership"]
    points = atlas_geom["points"]

    for i in range(n_charts):
        r_i = float(atlas_geom["support_r"][i])

        # --- Dense ball sampling (full reference domain) ---
        xi_ball = sample_ball(n_samples, r_i)
        stats_ball = _jacobian_stats_for_points(
            decoders[i], xi_ball,
            seed=atlas_geom["seeds"][i], t1=atlas_geom["t1"][i],
            t2=atlas_geom["t2"][i], n=atlas_geom["n"][i],
            chart_scale=atlas_geom["support_r"][i], chart_idx=i,
        )
        ball_results.append(stats_ball)

        # --- Membership-filtered sampling (physical domain only) ---
        pos_idx = torch.where(membership[:, i] > 0)[0]
        if pos_idx.numel() > 0:
            n_mem = min(n_samples, pos_idx.numel())
            sel = pos_idx[torch.randperm(pos_idx.numel())[:n_mem]]
            x_mem = points[sel]
            xi_mem = local_coords(
                x_mem, atlas_geom["seeds"][i], atlas_geom["t1"][i],
                atlas_geom["t2"][i], atlas_geom["n"][i],
            )
            stats_mem = _jacobian_stats_for_points(
                decoders[i], xi_mem,
                seed=atlas_geom["seeds"][i], t1=atlas_geom["t1"][i],
                t2=atlas_geom["t2"][i], n=atlas_geom["n"][i],
                chart_scale=atlas_geom["support_r"][i], chart_idx=i,
            )
        else:
            stats_mem = {"chart": i, "n_samples": 0, "foldover_frac": 0.0,
                         "foldover_count": 0, "det_J": np.array([]),
                         "det_min": float("nan"), "det_max": float("nan"),
                         "det_mean": float("nan"), "det_std": float("nan"),
                         "kappa_mean": float("nan"), "kappa_median": float("nan"),
                         "kappa_p95": float("nan"), "kappa_max": float("nan"),
                         "isotropy_mean": float("nan"), "isotropy_max": float("nan"),
                         "area_distortion": float("nan"), "svd_vals": np.array([])}
        mem_results.append(stats_mem)

    return ball_results, mem_results


# ---------------------------------------------------------------------------
# 2. Coverage statistics
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    masks: List[MaskNet],
    atlas_geom: dict,
) -> Dict:
    """Coverage and overlap degree statistics."""
    membership = atlas_geom["membership"].numpy()
    n_points, n_charts = membership.shape
    charts_per_point = membership.sum(axis=1)

    # Coverage from membership
    coverage_frac = float((charts_per_point >= 1).mean())

    # Overlap degree distribution
    max_degree = int(charts_per_point.max())
    degree_dist = {}
    for k in range(max_degree + 1):
        degree_dist[k] = float((charts_per_point == k).mean())

    # Per-chart membership count
    chart_counts = [int(membership[:, i].sum()) for i in range(n_charts)]

    # PoU entropy from mask networks
    points = atlas_geom["points"]
    with torch.no_grad():
        logits = []
        for i in range(n_charts):
            xi = local_coords(
                points,
                atlas_geom["seeds"][i],
                atlas_geom["t1"][i],
                atlas_geom["t2"][i],
                atlas_geom["n"][i],
            )
            logits.append(masks[i](xi, chart_scale=atlas_geom["support_r"][i]))
        logits_t = torch.stack(logits, dim=1)  # (N, M)
        omega = torch.softmax(logits_t, dim=1)  # PoU weights

        # Shannon entropy: H = -sum(omega_i * log(omega_i))
        log_omega = torch.log(omega.clamp(min=1e-12))
        entropy = -(omega * log_omega).sum(dim=1)  # (N,)

    return {
        "coverage_frac": coverage_frac,
        "degree_dist": degree_dist,
        "chart_counts": chart_counts,
        "n_points": n_points,
        "n_charts": n_charts,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "entropy_max": float(entropy.max()),
        "max_omega": omega.max(dim=1).values.numpy(),
        "entropy": entropy.numpy(),
    }


# ---------------------------------------------------------------------------
# 3. Transition-map consistency errors
# ---------------------------------------------------------------------------

def compute_transition_errors(
    decoders: List[ChartDecoder],
    atlas_geom: dict,
) -> Dict:
    """Pairwise transition-map consistency: ||phi_i(zeta_i(x)) - phi_j(zeta_j(x))||."""
    membership = atlas_geom["membership"]
    points = atlas_geom["points"]
    n_charts = len(decoders)

    error_matrix = np.full((n_charts, n_charts), np.nan)
    max_error_matrix = np.full((n_charts, n_charts), np.nan)
    pair_details = []

    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            overlap_mask = (membership[:, i] > 0) & (membership[:, j] > 0)
            n_overlap = int(overlap_mask.sum())
            if n_overlap == 0:
                continue

            x_ov = points[overlap_mask]
            # Subsample if too many
            if n_overlap > 2000:
                idx = torch.randperm(n_overlap)[:2000]
                x_ov = x_ov[idx]

            with torch.no_grad():
                xi_i = local_coords(x_ov, atlas_geom["seeds"][i],
                                    atlas_geom["t1"][i], atlas_geom["t2"][i],
                                    atlas_geom["n"][i])
                x_hat_i = decoders[i](
                    xi_i, seed=atlas_geom["seeds"][i],
                    t1=atlas_geom["t1"][i], t2=atlas_geom["t2"][i],
                    n=atlas_geom["n"][i], chart_scale=atlas_geom["support_r"][i],
                )

                xi_j = local_coords(x_ov, atlas_geom["seeds"][j],
                                    atlas_geom["t1"][j], atlas_geom["t2"][j],
                                    atlas_geom["n"][j])
                x_hat_j = decoders[j](
                    xi_j, seed=atlas_geom["seeds"][j],
                    t1=atlas_geom["t1"][j], t2=atlas_geom["t2"][j],
                    n=atlas_geom["n"][j], chart_scale=atlas_geom["support_r"][j],
                )

                err = torch.norm(x_hat_i - x_hat_j, dim=1)
                mean_err = float(err.mean())
                max_err = float(err.max())

            error_matrix[i, j] = mean_err
            error_matrix[j, i] = mean_err
            max_error_matrix[i, j] = max_err
            max_error_matrix[j, i] = max_err

            pair_details.append({
                "i": i, "j": j,
                "n_overlap": n_overlap,
                "mean_error": mean_err,
                "max_error": max_err,
                "std_error": float(err.std()),
            })

    # Overall statistics
    all_means = [p["mean_error"] for p in pair_details]
    all_maxes = [p["max_error"] for p in pair_details]

    return {
        "error_matrix": error_matrix,
        "max_error_matrix": max_error_matrix,
        "pair_details": pair_details,
        "overall_mean": float(np.mean(all_means)) if all_means else 0.0,
        "overall_max": float(np.max(all_maxes)) if all_maxes else 0.0,
        "n_overlapping_pairs": len(pair_details),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_report_figure(
    jac_results: List[Dict],
    coverage: Dict,
    transition: Dict,
    output_dir: str,
) -> str:
    """Generate combined 4-panel diagnostic figure."""
    n_charts = len(jac_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Atlas Quality Verification Report", fontsize=14, fontweight="bold")

    # Panel (a): Jacobian determinant histograms
    ax = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, n_charts))
    for r in jac_results:
        ax.hist(r["det_J"], bins=60, alpha=0.5, label=f"Chart {r['chart']}",
                color=colors[r["chart"]], density=True)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="det J = 0")
    ax.set_xlabel("det(J)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Jacobian determinant distribution")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (b): Condition number box plot
    ax = axes[0, 1]
    kappa_data = [r["svd_vals"][:, 0] / np.clip(r["svd_vals"][:, 2], 1e-12, None)
                  for r in jac_results]
    bp = ax.boxplot(kappa_data, labels=[str(i) for i in range(n_charts)],
                    showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Chart index")
    ax.set_ylabel(r"$\kappa(J)$")
    ax.set_title(r"(b) Condition number $\kappa(J) = \sigma_1/\sigma_3$")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (c): Overlap degree distribution
    ax = axes[1, 0]
    degree_dist = coverage["degree_dist"]
    degrees = sorted(degree_dist.keys())
    fracs = [degree_dist[k] * 100 for k in degrees]
    bars = ax.bar(degrees, fracs, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Number of charts covering a point")
    ax.set_ylabel("Fraction of points (%)")
    ax.set_title(f"(c) Overlap degree distribution (coverage={coverage['coverage_frac']:.1%})")
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, frac in zip(bars, fracs):
        if frac > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{frac:.1f}%", ha="center", va="bottom", fontsize=8)

    # Panel (d): Transition-map consistency error heatmap
    ax = axes[1, 1]
    err_mat = transition["error_matrix"]
    mask = np.isnan(err_mat)
    err_display = np.where(mask, 0, err_mat)
    im = ax.imshow(err_display, cmap="YlOrRd", interpolation="nearest")
    # Mark NaN cells (no overlap)
    for ii in range(n_charts):
        for jj in range(n_charts):
            if ii == jj:
                ax.text(jj, ii, "-", ha="center", va="center", fontsize=7, color="gray")
            elif mask[ii, jj]:
                ax.text(jj, ii, "n/a", ha="center", va="center", fontsize=6, color="gray")
            else:
                val = err_mat[ii, jj]
                ax.text(jj, ii, f"{val:.1e}", ha="center", va="center", fontsize=5.5,
                        color="white" if val > 0.015 else "black")
    ax.set_xlabel("Chart j")
    ax.set_ylabel("Chart i")
    ax.set_title(f"(d) Transition consistency ||phi_i - phi_j|| (mean={transition['overall_mean']:.2e})")
    ax.set_xticks(range(n_charts))
    ax.set_yticks(range(n_charts))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean error")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "atlas_quality_verification.pdf")
    png_path = os.path.join(output_dir, "atlas_quality_verification.png")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(
    ball_results: List[Dict],
    mem_results: List[Dict],
    coverage: Dict,
    transition: Dict,
    ckpt_path: str,
) -> str:
    """Print and return formatted text report with both ball and membership-filtered metrics."""
    lines = []
    lines.append("=" * 70)
    lines.append("Atlas Quality Verification Report")
    lines.append("=" * 70)
    lines.append(f"Checkpoint: {ckpt_path}")
    lines.append(f"Charts: {len(ball_results)}")
    lines.append("")

    # --- Membership-filtered (physical domain) Jacobian ---
    lines.append("--- Jacobian Determinant (membership-filtered, physical domain) ---")
    lines.append(f"  {'Chart':>5}  {'N':>6}  {'min(det J)':>11}  {'max(det J)':>11}  "
                 f"{'mean(det J)':>12}  {'kappa_p95':>10}  {'foldover%':>9}")
    for r in mem_results:
        if r["n_samples"] == 0:
            lines.append(f"  {r['chart']:5d}  {0:6d}  {'n/a':>11}  {'n/a':>11}  "
                         f"{'n/a':>12}  {'n/a':>10}  {'n/a':>9}")
        else:
            lines.append(f"  {r['chart']:5d}  {r['n_samples']:6d}  {r['det_min']:11.4f}  "
                         f"{r['det_max']:11.4f}  {r['det_mean']:12.4f}  "
                         f"{r['kappa_p95']:10.3f}  {r['foldover_frac']*100:8.2f}%")
    valid_mem = [r for r in mem_results if r["n_samples"] > 0]
    if valid_mem:
        total_fold_mem = np.mean([r["foldover_frac"] for r in valid_mem])
        lines.append(f"  {'AVG':>5}  {'':>6}  {np.mean([r['det_min'] for r in valid_mem]):11.4f}  "
                     f"{np.mean([r['det_max'] for r in valid_mem]):11.4f}  "
                     f"{np.mean([r['det_mean'] for r in valid_mem]):12.4f}  "
                     f"{np.mean([r['kappa_p95'] for r in valid_mem]):10.3f}  "
                     f"{total_fold_mem*100:8.2f}%")
    lines.append("")

    # --- Full ball sampling Jacobian ---
    lines.append("--- Jacobian Determinant (full reference ball, incl. exterior) ---")
    lines.append(f"  {'Chart':>5}  {'min(det J)':>11}  {'max(det J)':>11}  "
                 f"{'mean(det J)':>12}  {'kappa_p95':>10}  {'foldover%':>9}")
    for r in ball_results:
        lines.append(f"  {r['chart']:5d}  {r['det_min']:11.4f}  {r['det_max']:11.4f}  "
                     f"{r['det_mean']:12.4f}  {r['kappa_p95']:10.3f}  "
                     f"{r['foldover_frac']*100:8.2f}%")
    total_fold_ball = np.mean([r["foldover_frac"] for r in ball_results])
    lines.append(f"  {'AVG':>5}  {np.mean([r['det_min'] for r in ball_results]):11.4f}  "
                 f"{np.mean([r['det_max'] for r in ball_results]):11.4f}  "
                 f"{np.mean([r['det_mean'] for r in ball_results]):12.4f}  "
                 f"{np.mean([r['kappa_p95'] for r in ball_results]):10.3f}  "
                 f"{total_fold_ball*100:8.2f}%")
    lines.append("")

    # Condition number detail
    lines.append("--- Condition Number kappa(J) (membership-filtered) ---")
    lines.append(f"  {'Chart':>5}  {'mean':>8}  {'median':>8}  {'p95':>8}  {'max':>8}")
    for r in mem_results:
        if r["n_samples"] > 0:
            lines.append(f"  {r['chart']:5d}  {r['kappa_mean']:8.3f}  {r['kappa_median']:8.3f}  "
                         f"{r['kappa_p95']:8.3f}  {r['kappa_max']:8.3f}")
    lines.append("")

    # Coverage
    lines.append("--- Coverage Statistics ---")
    lines.append(f"  Total coverage: {coverage['coverage_frac']:.2%}")
    lines.append(f"  Points: {coverage['n_points']}, Charts: {coverage['n_charts']}")
    lines.append("  Overlap degree distribution:")
    for k in sorted(coverage["degree_dist"].keys()):
        lines.append(f"    {k}-chart: {coverage['degree_dist'][k]:.1%}")
    lines.append(f"  Per-chart membership counts: {coverage['chart_counts']}")
    lines.append(f"  PoU entropy: mean={coverage['entropy_mean']:.4f}, "
                 f"std={coverage['entropy_std']:.4f}, max={coverage['entropy_max']:.4f}")
    lines.append("")

    # Transition map consistency
    lines.append("--- Transition Map Consistency ---")
    lines.append(f"  Overlapping pairs: {transition['n_overlapping_pairs']}")
    lines.append(f"  Overall mean error: {transition['overall_mean']:.4e}")
    lines.append(f"  Overall max error:  {transition['overall_max']:.4e}")
    lines.append(f"  {'Pair':>8}  {'n_overlap':>9}  {'mean':>10}  {'max':>10}  {'std':>10}")
    for p in transition["pair_details"]:
        pi, pj = p["i"], p["j"]
        pair_str = f"({pi},{pj})"
        pad = " " * max(0, 8 - len(pair_str))
        lines.append(f"  {pair_str}{pad}  {p['n_overlap']:9d}  {p['mean_error']:10.4e}  "
                     f"{p['max_error']:10.4e}  {p['std_error']:10.4e}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify atlas quality metrics")
    parser.add_argument("--atlas-checkpoint", required=True,
                        help="Path to rabbit_atlas_trained.pt")
    parser.add_argument("--atlas-data", required=True,
                        help="Path to rabbit_atlas_data.npz")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for figures and reports")
    parser.add_argument("--n-jac-samples", type=int, default=5000,
                        help="Number of samples per chart for Jacobian evaluation")
    args = parser.parse_args()

    print(f"Loading atlas from {args.atlas_checkpoint} ...")
    decoders, masks, atlas_geom, ckpt = load_atlas(args.atlas_checkpoint, args.atlas_data)
    n_charts = len(decoders)
    print(f"  Loaded {n_charts} charts")

    # Existing gate report if available
    gate = ckpt.get("gate", {})
    if gate:
        print(f"  Gate report: passed={gate.get('passed')}, "
              f"coverage={gate.get('coverage_ratio')}, "
              f"overlap={gate.get('overlap_consistency'):.4e}, "
              f"foldover={gate.get('foldover_ratio')}, "
              f"rmse={gate.get('boundary_rmse'):.4e}")

    print("\n[1/4] Computing Jacobian metrics (ball + membership-filtered) ...")
    ball_results, mem_results = compute_jacobian_metrics(
        decoders, atlas_geom, n_samples=args.n_jac_samples)

    print("[2/4] Computing coverage statistics ...")
    coverage = compute_coverage_stats(masks, atlas_geom)

    print("[3/4] Computing transition-map consistency ...")
    transition = compute_transition_errors(decoders, atlas_geom)

    print("[4/4] Generating figures and report ...\n")
    # Use membership-filtered results for the main figure panels
    fig_path = generate_report_figure(mem_results, coverage, transition, args.output_dir)

    report_text = print_report(
        ball_results, mem_results, coverage, transition, args.atlas_checkpoint)

    # Save text report
    report_path = os.path.join(args.output_dir, "atlas_quality_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save JSON summary (both sampling modes)
    def _filter_dict(d):
        return {k: v for k, v in d.items() if k not in ("det_J", "svd_vals")}

    json_summary = {
        "checkpoint": args.atlas_checkpoint,
        "n_charts": n_charts,
        "jacobian_membership_filtered": [_filter_dict(r) for r in mem_results],
        "jacobian_full_ball": [_filter_dict(r) for r in ball_results],
        "coverage": {k: v for k, v in coverage.items()
                     if k not in ("max_omega", "entropy")},
        "transition": {k: v for k, v in transition.items()
                       if k not in ("error_matrix", "max_error_matrix")},
    }
    json_path = os.path.join(args.output_dir, "atlas_quality_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    print(f"\nOutputs saved:")
    print(f"  Figure:  {fig_path}")
    print(f"  Report:  {report_path}")
    print(f"  JSON:    {json_path}")


if __name__ == "__main__":
    main()
