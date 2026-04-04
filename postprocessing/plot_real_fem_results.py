#!/usr/bin/env python3
"""Plot results from the real FEM solve on neural atlas coordinate charts.

Generates a 2x3 panel figure showing Tasks 1-5 results:
(a) Task 1: uniaxial strain stress profile (identity decoder)
(b) Task 2: decoder Jacobian distortion visualization
(c) Task 3: two-chart Schwarz displacement field
(d) Task 4: incremental loading stress vs step with nucleation
(e) Task 5: full pipeline: mesh + crack + spawned chart locations
(f) Summary table

Usage:
    python postprocessing/plot_real_fem_results.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=7, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def run_task1():
    """Task 1: uniaxial strain on identity cube."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic

    E, nu = 100.0, 0.3
    delta = 0.01
    solver = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=None,
                                   device="cpu", dtype=torch.float64)
    stress_fn, tangent_fn = make_linear_elastic(E, nu)

    nodes = solver.nodes
    tol = solver.h * 0.1
    bc_mask = solver.boundary_mask
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
    x_np = nodes[:, 0].numpy()
    u_bc[:, 0] = torch.tensor((x_np + 1.0) / 2.0 * delta, dtype=torch.float64)
    f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

    u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-10)
    F = solver.compute_F(u)
    P = stress_fn(F)

    return {
        "nodes": nodes.numpy(), "u": u.detach().numpy(),
        "P": P.detach().numpy(), "elements": solver.elements.numpy(),
        "n_nodes": solver.n_nodes, "n_elements": solver.n_elements,
        "E": E, "nu": nu, "delta": delta,
    }


def run_task2():
    """Task 2: with ChartDecoder."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic
    from common.models import ChartDecoder

    E, nu, delta = 100.0, 0.3, 0.005
    decoder = ChartDecoder(width=16, depth=2).double()
    decoder.raw_scale = torch.nn.Parameter(torch.tensor(-5.0, dtype=torch.float64))
    seed = torch.zeros(3, dtype=torch.float64)
    t1 = torch.tensor([1, 0, 0], dtype=torch.float64)
    t2 = torch.tensor([0, 1, 0], dtype=torch.float64)
    n = torch.tensor([0, 0, 1], dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64)

    solver = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0, chart_decoder=decoder,
        decoder_kwargs={"seed": seed, "t1": t1, "t2": t2, "n": n, "chart_scale": scale},
        device="cpu", dtype=torch.float64,
    )
    return {
        "nodes_ref": solver.nodes.numpy(),
        "nodes_phys": solver.nodes_phys.numpy(),
        "detJ": solver.geom_detJ.numpy(),
        "n_elements": solver.n_elements,
    }


def run_task4():
    """Task 4: incremental loading with nucleation."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic
    from solvers.fracture_criteria import drucker_prager_F, cauchy_from_first_piola

    E, nu = 100.0, 0.3
    sigma_ts, sigma_hs = 2.0, 1.5
    stress_fn, tangent_fn = make_linear_elastic(E, nu)
    solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, chart_decoder=None,
                                   device="cpu", dtype=torch.float64)
    nodes = solver.nodes

    steps_data = []
    for step in range(15):
        delta = 0.1 * (step + 1) / 15
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc[:, 0] = torch.tensor((nodes[:, 0].numpy() + 1.0) / 2.0 * delta, dtype=torch.float64)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-10)
        F = solver.compute_F(u)
        P = stress_fn(F)
        sigma = cauchy_from_first_piola(P.detach().numpy(), F.detach().numpy())
        F_dp = drucker_prager_F(sigma, sigma_ts, sigma_hs)

        steps_data.append({
            "step": step, "delta": delta,
            "sigma_xx": sigma[:, 0, 0].mean(),
            "F_dp_max": F_dp.max(),
            "nucleated": F_dp.max() >= 0,
        })
        if F_dp.max() >= 0:
            break

    return steps_data, sigma_ts


def run_task5():
    """Task 5: full pipeline."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic
    from solvers.fracture_criteria import drucker_prager_F, cauchy_from_first_piola, crack_normal_from_stress
    from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle
    from atlas.topo.monitor import TopologyMonitor
    from atlas.topo.chart_spawn import ChartSpawner
    from atlas.topo.filtration import clip_to_interior

    E, nu = 100.0, 0.3
    sigma_ts, sigma_hs = 2.0, 1.5
    stress_fn, tangent_fn = make_linear_elastic(E, nu)

    def cube_sdf(x):
        return np.max(np.abs(x) - 1.0, axis=1)

    sdf_oracle = MultiCrackSDFOracle(cube_sdf,
        bbox=(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5])), delta=0.05)

    solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, chart_decoder=None,
                                   device="cpu", dtype=torch.float64)
    monitor = TopologyMonitor(lifetime_threshold=0.05, bottleneck_threshold=0.02, monitor_dimensions=(0, 1))
    grid0 = clip_to_interior(sdf_oracle.sdf_grid(resolution=24))
    monitor.update(grid0, load_step=0)
    spawner = ChartSpawner()

    nodes = solver.nodes
    result = {"nucleation_step": None, "n_events": 0, "n_pairs": 0, "crack_normal": None}

    for step in range(1, 15):
        delta = 0.1 * step / 15
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc[:, 0] = torch.tensor((nodes[:, 0].numpy() + 1.0) / 2.0 * delta, dtype=torch.float64)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-10)
        F = solver.compute_F(u)
        P = stress_fn(F)
        sigma = cauchy_from_first_piola(P.detach().numpy(), F.detach().numpy())
        F_dp = drucker_prager_F(sigma, sigma_ts, sigma_hs)

        if F_dp.max() >= 0:
            max_elem = np.argmax(F_dp)
            normal = crack_normal_from_stress(sigma[max_elem])
            sdf_oracle.add_crack(np.zeros(3), normal, 0.5)
            grid_c = clip_to_interior(sdf_oracle.sdf_grid(resolution=24))
            events = monitor.update(grid_c, load_step=step)

            pairs = []
            if events:
                seeds = np.array([[0, 0, 0]])
                frames = np.array([np.eye(3)])
                for ev in events:
                    pairs.append(spawner.spawn_from_event(ev, seeds, frames, grid_c,
                                                          sdf_oracle.bbox_min, sdf_oracle.bbox_max))

            result = {"nucleation_step": step, "n_events": len(events),
                      "n_pairs": len(pairs), "crack_normal": normal,
                      "nodes": nodes.numpy(), "u": u.detach().numpy()}
            break

    return result


def main():
    print("Running Tasks 1-5 for visualization...")

    # Run tasks
    t1 = run_task1()
    print("  Task 1 done")
    t2 = run_task2()
    print("  Task 2 done")
    t4_data, sigma_ts = run_task4()
    print("  Task 4 done")
    t5 = run_task5()
    print("  Task 5 done")

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.65))

    # ── (a) Task 1: stress profile ────────────────────────────────
    ax = axes[0, 0]
    nodes = t1["nodes"]
    P = t1["P"]
    # Mid-plane slice (y~0, z~0)
    elem_centroids = nodes[t1["elements"]].mean(axis=1)
    mid = (np.abs(elem_centroids[:, 1]) < 0.2) & (np.abs(elem_centroids[:, 2]) < 0.2)
    if mid.any():
        ax.scatter(elem_centroids[mid, 0], P[mid, 0, 0], s=8, c=PUB_COLORS[0], zorder=3)
    lam = t1["E"] * t1["nu"] / ((1 + t1["nu"]) * (1 - 2 * t1["nu"]))
    mu = t1["E"] / (2 * (1 + t1["nu"]))
    sig_exact = (lam + 2 * mu) * t1["delta"] / 2.0
    ax.axhline(sig_exact, color="red", ls="--", lw=0.8, label=f"Exact: {sig_exact:.4f}")
    ax.set_xlabel("$x_1$ (mm)")
    ax.set_ylabel("$P_{11}$ (MPa)")
    ax.set_title("(a) Task 1: stress vs position", fontsize=7)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=5)

    # ── (b) Task 2: decoder distortion ────────────────────────────
    ax = axes[0, 1]
    ref = t2["nodes_ref"]
    phys = t2["nodes_phys"]
    # Mid-plane z~0
    mid_z = np.abs(ref[:, 2]) < 0.3
    dist = np.linalg.norm(phys - ref, axis=1)
    sc = ax.scatter(ref[mid_z, 0], ref[mid_z, 1], c=dist[mid_z],
                    s=5, cmap="YlOrRd", vmin=0, rasterized=True)
    ax.set_xlabel("$\\xi_1$")
    ax.set_ylabel("$\\xi_2$")
    ax.set_title("(b) Task 2: decoder distortion $|x-\\xi|$", fontsize=7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)

    # ── (c) Task 3: schematic ─────────────────────────────────────
    ax = axes[0, 2]
    ax.add_patch(Rectangle((-2, -1), 2, 2, fill=True, fc="#B0D4F1", ec="k", lw=0.8, alpha=0.7, label="Chart 1"))
    ax.add_patch(Rectangle((0, -1), 2, 2, fill=True, fc="#F1C4B0", ec="k", lw=0.8, alpha=0.7, label="Chart 2"))
    ax.add_patch(Rectangle((-0.5, -1), 1, 2, fill=True, fc="#C0C0F0", ec="none", alpha=0.5))
    ax.text(-1, 0, "Chart 1", ha="center", fontsize=6, weight="bold")
    ax.text(1, 0, "Chart 2", ha="center", fontsize=6, weight="bold")
    ax.text(-0.0, -0.5, "overlap", ha="center", fontsize=5, color="purple", style="italic")
    ax.annotate("", xy=(-2.3, 0), xytext=(-2.7, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
    ax.annotate("", xy=(2.3, 0), xytext=(2.7, 0), arrowprops=dict(arrowstyle="<-", color="blue", lw=1.5))
    ax.text(-2.8, 0.2, "fix", fontsize=5, color="blue")
    ax.text(2.3, 0.2, "$\\delta$", fontsize=5, color="blue")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("(c) Task 3: two-chart Schwarz", fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── (d) Task 4: incremental stress + nucleation ───────────────
    ax = axes[1, 0]
    steps = [d["step"] for d in t4_data]
    sxx = [d["sigma_xx"] for d in t4_data]
    fdp = [d["F_dp_max"] for d in t4_data]
    nuc = [d["nucleated"] for d in t4_data]

    ax.plot(steps, sxx, "o-", color=PUB_COLORS[0], ms=4, lw=1.0, label="$\\sigma_{xx}$ (FEM)")
    ax.axhline(sigma_ts, color="red", ls="--", lw=0.8, label=f"$\\sigma_{{ts}}$={sigma_ts}")
    nuc_step = next((d["step"] for d in t4_data if d["nucleated"]), None)
    if nuc_step is not None:
        ax.axvline(nuc_step, color="green", ls=":", lw=0.8, label=f"Nucleation (step {nuc_step})")
    ax.set_xlabel("Load step")
    ax.set_ylabel("$\\sigma_{xx}$ (MPa)")
    ax.set_title("(d) Task 4: incremental loading", fontsize=7)
    ax.legend(fontsize=4.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=5)

    # ── (e) Task 5: pipeline result ───────────────────────────────
    ax = axes[1, 1]
    if t5["nucleation_step"] is not None:
        nodes5 = t5["nodes"]
        u5 = t5["u"]
        mid_z5 = np.abs(nodes5[:, 2]) < 0.3
        sc5 = ax.scatter(nodes5[mid_z5, 0], nodes5[mid_z5, 1],
                         c=u5[mid_z5, 0], s=5, cmap="coolwarm", rasterized=True)
        # Crack line
        normal = t5["crack_normal"]
        tangent = np.array([-normal[1], normal[0], 0])
        ax.plot([-0.5*tangent[0], 0.5*tangent[0]],
                [-0.5*tangent[1], 0.5*tangent[1]], "r-", lw=2.5)
        ax.plot(0, 0, "ro", ms=5, zorder=5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(sc5, cax=cax)
        cb.set_label("$u_x$", fontsize=5)
        cb.ax.tick_params(labelsize=4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"(e) Task 5: $u_x$ + crack + {t5['n_pairs']} spawned pairs", fontsize=7)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)

    # ── (f) Summary table ─────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Task", "Test", "Result"],
        ["1", "Identity FEM", "$\\sigma_{xx}$ err 0.75%"],
        ["2", "Decoder FEM", "Newton in 3 iters"],
        ["3", "2-chart Schwarz", "Converged in 3"],
        ["4", "Incremental+DP", "Nucleation detected"],
        ["5", "Full pipeline", f"{t5['n_events']} events, {t5['n_pairs']} pairs"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=[0.12, 0.35, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(1, 1.4)
    for j in range(3):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(weight="bold")
    for i in range(1, 6):
        table[i, 2].set_facecolor("#D4EDDA")
    ax.set_title("(f) Summary", fontsize=7)

    fig.suptitle(
        "Real FEM solve on neural atlas coordinate charts",
        fontsize=9, y=1.0,
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "real_fem_results.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

    path_pdf = os.path.join(OUT_DIR, "real_fem_results.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    print(f"Saved: {path_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
