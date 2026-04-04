#!/usr/bin/env python3
"""Generate publication-quality benchmark result figures.

Produces a 3x2 panel figure covering:
  (a) Persistence diagrams for ball, torus, shell
  (b) LEFM K_I vs a/W validation
  (c) Williams near-tip stress field
  (d) Constitutive model stress-strain curves
  (e) Cracked plate SDF cross-section
  (f) GUDHI timing vs grid resolution

Usage:
    python postprocessing/visualize_benchmarks.py
"""

import math
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W, GOLDEN

# ── styling ──────────────────────────────────────────────────────────────
set_pub_style(fontsize=8, usetex=False)
DIM_COLORS = {0: PUB_COLORS[0], 1: PUB_COLORS[1], 2: PUB_COLORS[2]}
DIM_LABELS = {0: "$H_0$", 1: "$H_1$", 2: "$H_2$"}


# =====================================================================
# Panel (a): Persistence Diagrams
# =====================================================================
def panel_persistence(ax):
    from atlas.topo.filtration import (
        sdf_ball, sdf_solid_torus, sdf_thick_spherical_shell, clip_to_interior,
    )
    from atlas.topo.persistence import compute_persistence_diagrams, filter_by_lifetime, betti_numbers_at
    from atlas.topo.ls_category import compute_m_min

    N = 32
    geometries = [
        ("Ball", sdf_ball, dict(radius=0.8), (-1.5, 1.5)),
        ("Solid torus", sdf_solid_torus, dict(R=1.0, r=0.35), (-1.8, 1.8)),
        ("Shell", sdf_thick_spherical_shell, dict(R_inner=0.5, R_outer=1.0), (-1.5, 1.5)),
    ]

    markers = ["o", "s", "D"]
    offsets_y = [0.0, 0.0, 0.0]

    all_births, all_deaths, all_dims, all_geom = [], [], [], []

    for gi, (name, sdf_fn, kwargs, bbox) in enumerate(geometries):
        lin = np.linspace(bbox[0], bbox[1], N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = sdf_fn(coords, **kwargs).reshape(N, N, N).astype("float32")
        grid = clip_to_interior(vals)

        diagrams = compute_persistence_diagrams(grid, max_dimension=2)
        filt_range = (float(grid.min()), 0.0)
        filtered = filter_by_lifetime(diagrams, threshold=0.08, filtration_range=filt_range)
        betti = betti_numbers_at(filtered, t=-1e-6)
        m_min = compute_m_min(betti)

        for dim in [0, 1, 2]:
            for b, d in filtered.get(dim, []):
                all_births.append(b)
                all_deaths.append(d)
                all_dims.append(dim)
                all_geom.append(gi)

        # Annotate
        beta_str = ", ".join(f"$\\beta_{k}$={v}" for k, v in sorted(betti.items()))
        ax.text(
            0.02, 0.95 - gi * 0.12, f"{name}: {beta_str}, $M_{{\\min}}$={m_min}",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    # Plot persistence diagram
    b_arr = np.array(all_births)
    d_arr = np.array(all_deaths)
    dim_arr = np.array(all_dims)
    geom_arr = np.array(all_geom)

    for dim in [0, 1, 2]:
        mask = dim_arr == dim
        if mask.any():
            ax.scatter(
                b_arr[mask], d_arr[mask],
                c=DIM_COLORS[dim], s=25, alpha=0.8,
                marker="o", edgecolors="k", linewidths=0.3,
                label=DIM_LABELS[dim], zorder=3,
            )

    # Diagonal
    lims = [min(b_arr.min(), d_arr.min()) - 0.05, max(d_arr.max(), 0.05)]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("(a) Persistence diagrams", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)


# =====================================================================
# Panel (b): K_I vs a/W
# =====================================================================
def panel_K_I(ax):
    from benchmarks.fracture.lefm_reference import (
        mode_i_reference_table, geometry_factor_edge_crack,
    )

    a_dense = np.linspace(0.05, 0.6, 100)
    W, sigma = 1.0, 1.0
    K_dense = np.array([
        sigma * math.sqrt(math.pi * a) * geometry_factor_edge_crack(a)
        for a in a_dense
    ])

    table = mode_i_reference_table(W=W, sigma_inf=sigma)

    ax.plot(a_dense, K_dense, color=PUB_COLORS[0], lw=1.5, label="$K_I$ (LEFM)")
    ax.plot(
        table["a_over_W"], table["K_I"], "o",
        color=PUB_COLORS[1], ms=5, mec="k", mew=0.4,
        label="Reference points",
    )

    # Secondary axis for F(a/W)
    ax2 = ax.twinx()
    F_dense = np.array([geometry_factor_edge_crack(a) for a in a_dense])
    ax2.plot(a_dense, F_dense, color=PUB_COLORS[2], lw=1.0, ls="--", label="$F(a/W)$")
    ax2.set_ylabel("$F(a/W)$", color=PUB_COLORS[2], fontsize=7)
    ax2.tick_params(axis="y", labelcolor=PUB_COLORS[2], labelsize=6)

    ax.set_xlabel("$a/W$")
    ax.set_ylabel("$K_I$")
    ax.set_title("(b) Stress intensity factor", fontsize=9)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")


# =====================================================================
# Panel (c): Williams stress field
# =====================================================================
def panel_williams_stress(ax):
    from benchmarks.fracture.lefm_reference import williams_stress

    K_I = 1.0
    nr, ntheta = 80, 120
    r_vals = np.linspace(0.02, 1.0, nr)
    theta_vals = np.linspace(-math.pi, math.pi, ntheta)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    sigma_xx, sigma_yy, sigma_xy = williams_stress(R.ravel(), Theta.ravel(), K_I)
    S_yy = sigma_yy.reshape(ntheta, nr)

    # Convert to Cartesian for plotting
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    vmax = 3.0
    cf = ax.contourf(
        X, Y, np.clip(S_yy, -vmax, vmax),
        levels=np.linspace(-vmax, vmax, 25),
        cmap="RdBu_r",
    )
    # Crack line
    ax.plot([-1.0, 0], [0, 0], "k-", lw=2.0)
    ax.plot(0, 0, "ko", ms=4, zorder=5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax, label="$\\sigma_{yy} / (K_I/\\sqrt{2\\pi})$")
    cax.tick_params(labelsize=5)
    cax.yaxis.label.set_size(6)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(c) Williams stress $\\sigma_{yy}$", fontsize=9)
    ax.set_aspect("equal")


# =====================================================================
# Panel (d): Constitutive stress-strain
# =====================================================================
def panel_constitutive(ax):
    from solvers.mpm.constitutive import NeoHookeanModel, ElastoplasticModel

    E, nu = 1e5, 0.3
    neo = NeoHookeanModel(E=E, nu=nu)
    ep_low = ElastoplasticModel(E=E, nu=nu, yield_stress=2e4, hardening=5e3)
    ep_high = ElastoplasticModel(E=E, nu=nu, yield_stress=2e4, hardening=2e4)

    stretch = np.linspace(1.0, 1.5, 100)
    models = [
        ("Neo-Hookean", neo, PUB_COLORS[0], "-"),
        ("J2 ($H$=5e3)", ep_low, PUB_COLORS[1], "--"),
        ("J2 ($H$=2e4)", ep_high, PUB_COLORS[2], "-."),
    ]

    for label, model, color, ls in models:
        stresses = []
        state = None
        for lam in stretch:
            F = torch.diag(torch.tensor([lam, 1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
            sigma, state = model.compute_stress(F, state if state else {})
            stresses.append(sigma[0, 0, 0].item())
            state = None  # Reset state each step for monotonic loading curve
        ax.plot(stretch - 1.0, np.array(stresses) / 1e3, color=color, ls=ls, lw=1.2, label=label)

    ax.set_xlabel("Engineering strain $\\varepsilon_{11}$")
    ax.set_ylabel("Cauchy stress $\\sigma_{11}$ (kPa)")
    ax.set_title("(d) Constitutive models", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    ax.axhline(0, color="k", lw=0.3)
    ax.grid(True, alpha=0.2)


# =====================================================================
# Panel (e): Crack SDF cross-section
# =====================================================================
def panel_crack_sdf(ax):
    from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate

    W, H, T = 1.0, 4.0, 0.5
    a = 0.5

    nx, ny = 200, 200
    x1 = np.linspace(-1.3, 1.3, nx)
    x2 = np.linspace(-2.5, 2.5, ny)
    X1, X2 = np.meshgrid(x1, x2)
    coords = np.stack([X1.ravel(), X2.ravel(), np.zeros(nx * ny)], axis=1)

    sdf_vals = sdf_cracked_plate(coords, a=a, W=W, H=H, T=T, delta=0.03)
    SDF = sdf_vals.reshape(ny, nx)

    cf = ax.contourf(X1, X2, SDF, levels=np.linspace(-0.5, 0.5, 30), cmap="coolwarm")
    ax.contour(X1, X2, SDF, levels=[0], colors="k", linewidths=1.0)

    # Annotate crack
    ax.plot([-W, -W + a], [0, 0], "r-", lw=2.5, solid_capstyle="round")
    ax.annotate(
        f"crack ($a$={a})", xy=(-W + a, 0), xytext=(-0.2, 0.5),
        fontsize=6, arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        color="red",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax, label="SDF")
    cax.tick_params(labelsize=5)
    cax.yaxis.label.set_size(6)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(e) Cracked plate SDF ($x_3$=0)", fontsize=9)
    ax.set_aspect("equal")


# =====================================================================
# Panel (f): GUDHI timing
# =====================================================================
def panel_gudhi_timing(ax):
    from atlas.topo.filtration import sdf_solid_torus, clip_to_interior

    try:
        from atlas.topo.persistence import compute_persistence_diagrams
    except ImportError:
        ax.text(0.5, 0.5, "GUDHI not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_title("(f) GUDHI timing", fontsize=9)
        return

    resolutions = [16, 24, 32]
    timings = []
    for N in resolutions:
        lin = np.linspace(-1.8, 1.8, N)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = sdf_solid_torus(coords, R=1.0, r=0.35).reshape(N, N, N).astype("float32")
        grid = clip_to_interior(vals)

        # Warm-up
        compute_persistence_diagrams(grid, max_dimension=2)
        # Timed run
        t0 = time.perf_counter()
        compute_persistence_diagrams(grid, max_dimension=2)
        timings.append((time.perf_counter() - t0) * 1000)

    bars = ax.bar(
        range(len(resolutions)), timings,
        color=[PUB_COLORS[0], PUB_COLORS[2], PUB_COLORS[1]],
        edgecolor="k", linewidth=0.4, width=0.6,
    )
    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels([f"${n}^3$" for n in resolutions])
    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Time (ms)")
    ax.set_title("(f) GUDHI persistence timing", fontsize=9)

    for bar, t in zip(bars, timings):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{t:.0f} ms", ha="center", va="bottom", fontsize=6,
        )

    # Overhead annotation
    solver_time_250steps = 25000  # ms (estimated for 250 FEM Newton steps)
    n_calls = 5  # monitoring every 50 steps
    gudhi_total = timings[0] * n_calls  # use 16^3
    overhead = gudhi_total / (solver_time_250steps + gudhi_total) * 100
    ax.text(
        0.95, 0.95,
        f"Production ($16^3$, every 50 steps):\n{n_calls} calls = {gudhi_total:.0f} ms\nOverhead: {overhead:.1f}%",
        transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray"),
    )


# =====================================================================
# Main
# =====================================================================
def main():
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 1.35))

    panel_persistence(axes[0, 0])
    panel_K_I(axes[0, 1])
    panel_williams_stress(axes[1, 0])
    panel_constitutive(axes[1, 1])
    panel_crack_sdf(axes[2, 0])
    panel_gudhi_timing(axes[2, 1])

    plt.tight_layout(h_pad=1.5, w_pad=1.2)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_results.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save PDF for publication
    out_pdf = os.path.join(out_dir, "benchmark_results.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
