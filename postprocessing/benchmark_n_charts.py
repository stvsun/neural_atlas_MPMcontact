#!/usr/bin/env python3
"""
benchmark_n_charts.py — Compare coordinate chart quality and PINN accuracy
across different n_charts configurations (8, 12, 16, 32).

Metrics collected per configuration:

  From atlas NPZ:
    - min_chart_pts      : smallest chart (important for Stanford Bunny ears)
    - support_radii_mean : mean chart support radius (larger = wider charts)

  From gate report JSON:
    - overlap_consistency : mean chart-pair overlap mismatch (lower = better)
    - coverage_ratio      : fraction of interior volume covered (target = 1.0)

  NEW — Surface coverage entropy (requires PLY file):
    For each PLY triangle centroid, find the nearest atlas seed (KNN k=1).
    per_chart_fraction[i] = count(chart_assignment == i) / n_triangles
    entropy = -sum(p * log(p))  [higher = more uniform surface coverage]

  From PINN history JSON:
    - if_flux_iter1  : interface flux at Schwarz iteration 1 (target < 0.05)
    - rel_l2_iter1   : relative L2 at Schwarz iteration 1
    - best_rel_l2    : minimum relative L2 across all iterations

Output:
    <output_dir>/benchmark_comparison.pdf    — 4-panel figure (n_charts × metrics)
    <output_dir>/surface_coverage_*.pdf      — per-chart coverage bar charts
    <output_dir>/benchmark_table.txt         — human-readable summary table
    <output_dir>/benchmark_data.json         — raw metrics for downstream use

Usage:
    python postprocessing/benchmark_n_charts.py \\
        --ply-file  runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply \\
        --configs 8 12 16 32 \\
        --output-dir runs/benchmark_summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from postprocessing.utils import set_pub_style, DOUBLE_COL_W, GOLDEN, PUB_COLORS
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_atlas_metrics(n: int) -> dict:
    """Load metrics from atlas NPZ for a given n_charts config."""
    npz_path = f"runs/benchmark_atlas_{n}chart/rabbit_atlas_data.npz"
    if not os.path.exists(npz_path):
        return {}
    d = np.load(npz_path)
    membership = d["membership"]          # (N, n_charts)
    support_radii = d["support_radii"]    # (n_charts,)
    seed_points = d["seed_points"]        # (n_charts, 3)

    # Points per chart (primary assignment)
    primary = np.argmax(membership, axis=1)
    chart_sizes = np.bincount(primary, minlength=membership.shape[1])

    return {
        "n_interior_pts": membership.shape[0],
        "n_charts": membership.shape[1],
        "min_chart_pts": int(chart_sizes.min()),
        "max_chart_pts": int(chart_sizes.max()),
        "chart_sizes": chart_sizes.tolist(),
        "support_radii": support_radii.tolist(),
        "support_radii_mean": float(support_radii.mean()),
        "support_radii_min": float(support_radii.min()),
        "seed_points": seed_points.tolist(),
    }


def load_gate_metrics(n: int) -> dict:
    """Load metrics from decoder gate report JSON for a given n_charts config."""
    gate_path = f"runs/benchmark_dec_{n}chart/rabbit_atlas_gate_report.json"
    if not os.path.exists(gate_path):
        return {}
    with open(gate_path) as f:
        g = json.load(f)
    return {
        "overlap_consistency": float(g.get("overlap_consistency", float("nan"))),
        "coverage_ratio": float(g.get("coverage_ratio", float("nan"))),
        "foldover_ratio": float(g.get("foldover_ratio", 0.0)),
        "gate_passed": bool(g.get("passed", False)),
    }


def load_pinn_metrics(n: int) -> dict:
    """Load metrics from PINN history JSON for a given n_charts config."""
    pinn_dir = f"runs/benchmark_pinn_{n}chart"
    hist_pattern = os.path.join(pinn_dir, "*_history.json")
    hist_files = sorted(glob.glob(hist_pattern))
    if not hist_files:
        return {}
    hist_path = hist_files[-1]  # take the most recent
    with open(hist_path) as f:
        h = json.load(f)

    if_flux = h.get("interface_flux", [])
    rel_l2 = h.get("rel_l2_eval", [])
    global_res = h.get("global_residual", [])

    return {
        "n_schwarz_iters": len(if_flux),
        "if_flux_iter1": float(if_flux[0]) if if_flux else float("nan"),
        "rel_l2_iter1": float(rel_l2[0]) if rel_l2 else float("nan"),
        "best_rel_l2": float(min(r for r in rel_l2 if r == r)) if rel_l2 else float("nan"),
        "final_rel_l2": float(rel_l2[-1]) if rel_l2 else float("nan"),
        "pde_iter1": float(global_res[0]) if global_res else float("nan"),
        "if_flux_series": if_flux[:30],
        "rel_l2_series": rel_l2[:30],
    }


def compute_surface_coverage_entropy(
    ply_path: str, seed_points: list
) -> dict:
    """Measure how uniformly atlas seeds cover the PLY surface.

    Algorithm:
      1. Load PLY mesh triangles
      2. Compute centroid of each triangle
      3. Assign each triangle to its nearest seed (KNN k=1)
      4. Compute per-chart fraction of triangles
      5. Compute entropy = -sum(p * log(p))  [max = log(n_charts)]

    Returns dict with entropy, per_chart_fractions, and normalised_entropy.
    """
    try:
        import trimesh
    except ImportError:
        return {"error": "trimesh not available"}

    if not os.path.exists(ply_path):
        return {"error": f"PLY file not found: {ply_path}"}

    mesh = trimesh.load(ply_path, force="mesh")
    faces = np.asarray(mesh.faces)         # (F, 3)
    verts = np.asarray(mesh.vertices)      # (V, 3)
    centroids = verts[faces].mean(axis=1)  # (F, 3)

    seeds = np.array(seed_points)          # (n_charts, 3)
    n_charts = len(seeds)
    n_triangles = len(centroids)

    # KNN k=1: assign each triangle centroid to nearest seed
    # Use brute-force (n_triangles × n_charts distances)
    diff = centroids[:, None, :] - seeds[None, :, :]   # (F, K, 3)
    dists = np.linalg.norm(diff, axis=-1)               # (F, K)
    assignments = np.argmin(dists, axis=1)              # (F,)

    counts = np.bincount(assignments, minlength=n_charts)
    fractions = counts / n_triangles

    # Entropy (using small epsilon to avoid log(0))
    eps = 1e-12
    entropy = -float(np.sum(fractions * np.log(fractions + eps)))
    max_entropy = float(np.log(n_charts))
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "surface_coverage_entropy": entropy,
        "surface_coverage_entropy_normalised": normalised_entropy,
        "per_chart_triangle_fractions": fractions.tolist(),
        "per_chart_triangle_counts": counts.tolist(),
        "n_triangles": n_triangles,
        "max_entropy": max_entropy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Collect all metrics
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_metrics(configs: list, ply_path: str) -> dict:
    """Collect all metrics for all n_charts configurations."""
    all_metrics = {}
    for n in configs:
        print(f"  Loading n_charts={n}…")
        atlas = load_atlas_metrics(n)
        gate = load_gate_metrics(n)
        pinn = load_pinn_metrics(n)

        surface = {}
        if atlas.get("seed_points") and ply_path:
            surface = compute_surface_coverage_entropy(ply_path, atlas["seed_points"])
            if "error" in surface:
                print(f"    WARNING surface coverage: {surface['error']}")

        all_metrics[n] = {
            "n_charts": n,
            **atlas,
            **gate,
            **pinn,
            **surface,
        }

        # Print one-line summary
        se = all_metrics[n].get("surface_coverage_entropy_normalised", float("nan"))
        oc = all_metrics[n].get("overlap_consistency", float("nan"))
        flux = all_metrics[n].get("if_flux_iter1", float("nan"))
        rl2 = all_metrics[n].get("best_rel_l2", float("nan"))
        print(
            f"    n={n:2d}  entropy_norm={se:.3f}  overlap={oc:.5f}  "
            f"if_flux@1={flux:.4f}  best_rel_l2={rl2:.4f}"
        )

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(all_metrics: dict, configs: list, output_dir: str) -> None:
    """4-panel figure: n_charts × {support_radii, overlap_consistency, if_flux@1, best_rel_l2}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _HAS_UTILS:
        set_pub_style()

    valid = [n for n in configs if all_metrics[n]]
    ns = np.array(valid)

    def get(key, default=float("nan")):
        return np.array([all_metrics[n].get(key, default) for n in valid])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Coordinate Chart Benchmark: Quality vs Number of Charts", fontsize=12)

    panels = [
        (axes[0, 0], "support_radii_mean", "Support radius (mean)", "blue", False),
        (axes[0, 1], "overlap_consistency", "Overlap consistency (↓ better)", "green", False),
        (axes[1, 0], "if_flux_iter1",       "Interface flux @ iter 1 (↓ better)", "red", True),
        (axes[1, 1], "best_rel_l2",          "Best rel-L² error (↓ better)", "purple", True),
    ]

    for ax, key, ylabel, color, log_scale in panels:
        vals = get(key)
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(ns[mask], vals[mask], "o-", color=color, linewidth=2, markersize=8)
            for n, v in zip(ns[mask], vals[mask]):
                ax.annotate(
                    f"{v:.3f}", (n, v),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8
                )
        ax.set_xlabel("Number of charts")
        ax.set_ylabel(ylabel)
        ax.set_xticks(configs)
        ax.grid(True, alpha=0.3)
        if log_scale and mask.any() and vals[mask].min() > 0:
            ax.set_yscale("log")

        # Reference lines for key metrics
        if key == "if_flux_iter1":
            ax.axhline(0.05, linestyle="--", color="gray", alpha=0.6, label="target 0.05")
            ax.axhline(0.015, linestyle=":", color="black", alpha=0.6, label="procedural rabbit")
            ax.legend(fontsize=7)
        elif key == "overlap_consistency":
            ax.axhline(0.007, linestyle="--", color="gray", alpha=0.6, label="target 0.007")
            ax.axhline(0.006, linestyle=":", color="black", alpha=0.6, label="procedural rabbit")
            ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_comparison.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_surface_coverage(all_metrics: dict, configs: list, output_dir: str) -> None:
    """Per-chart surface coverage bar charts for each configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [n for n in configs
             if all_metrics[n].get("per_chart_triangle_fractions")]
    if not valid:
        print("  Skipping surface coverage plots (no data)")
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(4 * len(valid), 4))
    if len(valid) == 1:
        axes = [axes]

    fig.suptitle("Surface Coverage: fraction of PLY triangles per chart", fontsize=11)

    for ax, n in zip(axes, valid):
        fracs = np.array(all_metrics[n]["per_chart_triangle_fractions"])
        uniform = 1.0 / len(fracs)
        bars = ax.bar(range(len(fracs)), fracs, color="steelblue", alpha=0.8)
        ax.axhline(uniform, linestyle="--", color="red", alpha=0.6, label=f"uniform ({uniform:.3f})")
        ax.set_title(f"n_charts={n}")
        ax.set_xlabel("Chart index")
        ax.set_ylabel("Triangle fraction")
        entr = all_metrics[n].get("surface_coverage_entropy_normalised", float("nan"))
        ax.text(
            0.97, 0.95, f"entropy_norm={entr:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8
        )
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "surface_coverage_bar.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pinn_convergence(all_metrics: dict, configs: list, output_dir: str) -> None:
    """Overlay PINN convergence curves (if_flux and rel_l2 vs Schwarz iteration)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PINN Convergence vs Number of Charts", fontsize=11)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for i, n in enumerate(configs):
        m = all_metrics[n]
        flux_series = m.get("if_flux_series", [])
        rel_l2_series = m.get("rel_l2_series", [])
        if flux_series:
            ax1.semilogy(range(1, len(flux_series) + 1), flux_series,
                         "o-", color=colors[i], label=f"n={n}", linewidth=1.5, markersize=4)
        if rel_l2_series:
            ax2.semilogy(range(1, len(rel_l2_series) + 1), rel_l2_series,
                         "o-", color=colors[i], label=f"n={n}", linewidth=1.5, markersize=4)

    ax1.axhline(0.05, linestyle="--", color="gray", alpha=0.6, label="target 0.05")
    ax1.set_xlabel("Schwarz iteration")
    ax1.set_ylabel("Interface flux")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Schwarz iteration")
    ax2.set_ylabel("Relative L² error")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "pinn_convergence.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_table(all_metrics: dict, configs: list, output_dir: str) -> None:
    """Print a human-readable summary table and save to file."""
    header = (
        f"{'n':>4}  {'min_pts':>8}  {'r_mean':>7}  {'overlap':>9}  "
        f"{'coverage':>9}  {'ent_norm':>9}  {'flux@1':>8}  {'rel_l2@1':>9}  {'best_rl2':>9}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for n in configs:
        m = all_metrics.get(n, {})
        line = (
            f"{n:>4}  "
            f"{m.get('min_chart_pts', float('nan')):>8.0f}  "
            f"{m.get('support_radii_mean', float('nan')):>7.4f}  "
            f"{m.get('overlap_consistency', float('nan')):>9.5f}  "
            f"{m.get('coverage_ratio', float('nan')):>9.4f}  "
            f"{m.get('surface_coverage_entropy_normalised', float('nan')):>9.4f}  "
            f"{m.get('if_flux_iter1', float('nan')):>8.4f}  "
            f"{m.get('rel_l2_iter1', float('nan')):>9.4f}  "
            f"{m.get('best_rel_l2', float('nan')):>9.4f}"
        )
        lines.append(line)
    lines.append(sep)

    # Reference rows
    lines.append("  Reference: procedural rabbit (attempt20c_compact)")
    lines.append(
        f"{'ref':>4}  {'2526':>8}  {'0.4100':>7}  {'0.00640':>9}  "
        f"{'1.0000':>9}  {'  N/A':>9}  {'0.0149':>8}  {'0.0410':>9}  {'0.0220':>9}"
    )
    lines.append(sep)

    table_str = "\n".join(lines)
    print("\nBenchmark Summary:")
    print(table_str)

    table_path = os.path.join(output_dir, "benchmark_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str + "\n")
    print(f"\n  Saved: {table_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("n-CHARTS BENCHMARK REPORT")
    print("=" * 60)
    print(f"Configurations: {args.configs}")
    print(f"PLY file: {args.ply_file}")
    print()

    print("Loading metrics…")
    all_metrics = collect_all_metrics(args.configs, args.ply_file)

    # Save raw data
    data_path = os.path.join(args.output_dir, "benchmark_data.json")
    serialisable = {}
    for n, m in all_metrics.items():
        s = {}
        for k, v in m.items():
            if isinstance(v, (int, float, bool, str, type(None))):
                s[k] = v
            elif isinstance(v, (list, dict)):
                s[k] = v
            elif hasattr(v, "tolist"):
                s[k] = v.tolist()
            else:
                s[k] = str(v)
        serialisable[str(n)] = s
    with open(data_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Raw data saved: {data_path}")

    print("\nGenerating figures…")
    plot_comparison(all_metrics, args.configs, args.output_dir)
    plot_surface_coverage(all_metrics, args.configs, args.output_dir)
    plot_pinn_convergence(all_metrics, args.configs, args.output_dir)

    print_and_save_table(all_metrics, args.configs, args.output_dir)

    print(f"\nAll outputs in: {args.output_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark coordinate chart quality and PINN accuracy vs n_charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ply-file",
        default="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply",
        help="Path to Stanford Bunny PLY file (for surface coverage metric)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        type=int,
        default=[8, 12, 16, 32],
        help="n_charts configurations to compare",
    )
    p.add_argument(
        "--output-dir",
        default="runs/benchmark_summary",
        help="Output directory for figures and table",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
