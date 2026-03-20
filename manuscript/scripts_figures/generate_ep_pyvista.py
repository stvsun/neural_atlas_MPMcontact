#!/usr/bin/env python3
"""Generate torus elastoplastic PyVista figures from real atlas checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pyvista as pv
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

from experiments.torus_elastoplastic.reconstruct_torus_fields import (
    find_checkpoints,
    find_latest_checkpoint,
    interpolate_plastic_surface_fields,
    reconstruct_checkpoint,
)
from experiments.torus_elastoplastic.return_mapping import sym_logm


OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "figures_cmame_core" / "example5_torus_elastoplastic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_POS = [(3.0, -2.5, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
E_VAL = 200.0
NU_VAL = 0.3
MU_VAL = E_VAL / (2.0 * (1.0 + NU_VAL))
K_VAL = E_VAL / (3.0 * (1.0 - 2.0 * NU_VAL))


def render_torus(surf, scalar_name, title, cmap="turbo", fmt="%.4f"):
    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    pl.add_mesh(
        surf,
        scalars=scalar_name,
        cmap=cmap,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": title,
            "title_font_size": 16,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": fmt,
            "position_x": 0.78,
            "position_y": 0.15,
            "width": 0.14,
            "height": 0.65,
            "color": "black",
        },
    )
    pl.camera_position = CAMERA_POS
    pl.screenshot("/tmp/_torus_ep.png", transparent_background=False)
    img = plt.imread("/tmp/_torus_ep.png")
    pl.close()
    return img


def extract_hysteresis_from_checkpoints(checkpoints):
    final_result = reconstruct_checkpoint(checkpoints[-1])
    chart0 = final_result.charts[0]
    centroids_ref = chart0["centroids_ref"]
    order = np.argsort(centroids_ref[:, 0])
    n_elem = len(order)
    sample = {
        "A": int(order[-max(1, n_elem // 8)]),
        "B": int(order[n_elem // 2]),
        "C": int(order[max(0, n_elem // 8)]),
    }

    curves = {key: {"eps": [], "tau": []} for key in sample}
    eye = torch.eye(3, dtype=torch.float64)
    for checkpoint in checkpoints:
        result = reconstruct_checkpoint(checkpoint)
        chart = result.charts[0]
        for label, elem_idx in sample.items():
            F = torch.from_numpy(chart["F_phys"][elem_idx]).to(dtype=torch.float64)
            C = F.T @ F
            eps = 0.5 * sym_logm(C.unsqueeze(0)).squeeze(0)

            Be = torch.from_numpy(chart["Be"][elem_idx]).to(dtype=torch.float64)
            eps_e = 0.5 * sym_logm(Be.unsqueeze(0)).squeeze(0)
            tr_e = torch.trace(eps_e)
            tau = 2.0 * MU_VAL * (eps_e - tr_e / 3.0 * eye) + K_VAL * tr_e * eye

            curves[label]["eps"].append(eps[0, 0].item())
            curves[label]["tau"].append(tau[0, 0].item())

    return {
        label: (np.asarray(vals["eps"]), np.asarray(vals["tau"]))
        for label, vals in curves.items()
    }


def render_hysteresis(checkpoints):
    if len(checkpoints) < 2:
        print("Skipping hysteresis: fewer than two atlas checkpoints are available.")
        return

    curves = extract_hysteresis_from_checkpoints(checkpoints)
    subtitles = [
        "Point A (near loaded face)",
        "Point B (mid-domain)",
        "Point C (near fixed face)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    for ax, label, subtitle in zip(axes, ["A", "B", "C"], subtitles):
        eps, tau = curves[label]
        ax.plot(eps, tau, "b-o", lw=1.4, ms=3.5, label="Atlas checkpoints")
        ax.set_xlabel(r"$\varepsilon_{11}$ (log strain)", fontsize=12)
        if label == "A":
            ax.set_ylabel(r"$\tau_{11}$ (Kirchhoff stress)", fontsize=12)
        ax.set_title(subtitle, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle("Stress-strain extracted from saved torus atlas checkpoints", fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"ep_hysteresis_multipoint.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/torus_forward_bvp_debug_small_max25",
        help="Used when --checkpoint is omitted and for hysteresis checkpoint discovery.",
    )
    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint(Path(_REPO_ROOT) / args.run_dir)
    )
    run_dir = checkpoint.parent if args.checkpoint else Path(_REPO_ROOT) / args.run_dir

    result = reconstruct_checkpoint(checkpoint)
    surf = interpolate_plastic_surface_fields(result)

    panels = [
        ("det_Fp", r"det $\mathbf{F}^p$", "coolwarm", "%.4f"),
        ("iso_eig1", r"$\bar{\lambda}^p_1$ (max plastic stretch)", "turbo", "%.4f"),
        ("iso_eig2", r"$\bar{\lambda}^p_2$ (mid plastic stretch)", "turbo", "%.4f"),
        ("iso_eig3", r"$\bar{\lambda}^p_3$ (min plastic stretch)", "turbo", "%.4f"),
    ]
    images = [render_torus(surf, name, title, cmap, fmt) for name, title, cmap, fmt in panels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    for ax, img, label in zip(axes.ravel(), images, ["(a)", "(b)", "(c)", "(d)"]):
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
        )
    fig.tight_layout(pad=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"ep_plastic_strain.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    render_hysteresis(find_checkpoints(run_dir))

    report = {
        "checkpoint": str(result.checkpoint_path),
        "classification": result.classification,
        "metrics": result.metrics,
    }
    (OUT_DIR / "ep_reconstruction_report.json").write_text(json.dumps(report, indent=2))
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Classification: {result.classification}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6e}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
