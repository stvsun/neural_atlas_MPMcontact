#!/usr/bin/env python3
"""Generate a PyVista 3D visualization of equivalent plastic strain on a cube
domain after cyclic elastoplastic loading.

Creates:
  manuscript/figures_cmame_core/example5_torus_elastoplastic/ep_plastic_strain.png
  manuscript/figures_cmame_core/example5_torus_elastoplastic/ep_plastic_strain.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Off-screen rendering setup (must come before any pyvista import)
# ---------------------------------------------------------------------------
import pyvista as pv

pv.OFF_SCREEN = True
# Try to start Xvfb on headless Linux; harmless if already running or on macOS
try:
    pv.start_xvfb()
except Exception:
    pass

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.incremental_solver import IncrementalSolver
from experiments.torus_elastoplastic.return_mapping import ReturnMappingState

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "figures_cmame_core" / "example5_torus_elastoplastic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # -----------------------------------------------------------------------
    # 1. Material parameters (same as inverse_kinematic_hardening.py)
    # -----------------------------------------------------------------------
    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))
    tau_y_true = 0.5
    H_kin_true = 20.0

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)
    tau_y_t = torch.tensor(tau_y_true, device=device)
    H_kin_t = torch.tensor(H_kin_true, device=device)

    # -----------------------------------------------------------------------
    # 2. Mesh (n_cells=4, matching the inverse problem setup)
    # -----------------------------------------------------------------------
    n_cells = 4
    fem = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, device=device, dtype=torch.float64,
    )
    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r

    print(f"Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")

    # -----------------------------------------------------------------------
    # 3. Boundary conditions (same cyclic schedule)
    # -----------------------------------------------------------------------
    left_face = (nodes[:, 0] < -r + tol_face)
    right_face = (nodes[:, 0] > r - tol_face)
    bc_mask = left_face | right_face

    eps_peak = 0.03
    n_steps_per_half = 15

    bc_schedule = []
    for cycle in range(2):
        # Forward: 0 -> +eps
        for step in range(n_steps_per_half):
            lam = (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)
        # Reverse: +eps -> -eps
        for step in range(2 * n_steps_per_half):
            lam = 1.0 - (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)
        # Return: -eps -> 0
        for step in range(n_steps_per_half):
            lam = -1.0 + (step + 1) / n_steps_per_half
            u_bc = torch.zeros_like(nodes)
            u_bc[right_face, 0] = eps_peak * lam * 2.0 * r
            bc_schedule.append(u_bc)

    total_steps = len(bc_schedule)
    print(f"Cyclic loading: 2 cycles, {n_steps_per_half} steps/half, "
          f"eps_peak={eps_peak}, total={total_steps} steps")

    # -----------------------------------------------------------------------
    # 4. Forward solve
    # -----------------------------------------------------------------------
    print("Running forward cyclic solve...")
    inc_solver = IncrementalSolver(
        fem, mu, K, tau_y_t, H_kin_t, epsilon=1e-3,
    )

    with torch.no_grad():
        u_hist, state_hist = inc_solver.solve_history(
            bc_schedule, bc_mask, verbose=True, max_newton_iter=50, tol=1e-6,
        )

    # -----------------------------------------------------------------------
    # 5. Extract equivalent plastic strain from final state
    # -----------------------------------------------------------------------
    final_state = state_hist[-1]
    ep_bar_elem = final_state.ep_bar.detach().cpu().numpy()  # (n_elements,)
    final_u = u_hist[-1].detach().cpu().numpy()  # (n_nodes, 3)

    print(f"ep_bar range: [{ep_bar_elem.min():.6f}, {ep_bar_elem.max():.6f}]")

    # -----------------------------------------------------------------------
    # 6. Build PyVista UnstructuredGrid from P1 tet mesh
    # -----------------------------------------------------------------------
    nodes_np = fem.nodes.detach().cpu().numpy()  # (N, 3)
    elems_np = fem.elements.detach().cpu().numpy()  # (M, 4)

    # PyVista VTK_TETRA cell type = 10
    n_elem = elems_np.shape[0]
    # Build cells array: [4, n0, n1, n2, n3, 4, n0, n1, n2, n3, ...]
    cells = np.hstack([
        np.full((n_elem, 1), 4, dtype=np.int64),
        elems_np,
    ]).ravel()
    celltypes = np.full(n_elem, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes_np)

    # -----------------------------------------------------------------------
    # 7. Map element-level ep_bar to node-averaged values
    # -----------------------------------------------------------------------
    n_nodes = nodes_np.shape[0]
    ep_bar_node = np.zeros(n_nodes, dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.float64)

    for e in range(n_elem):
        for local_n in range(4):
            global_n = elems_np[e, local_n]
            ep_bar_node[global_n] += ep_bar_elem[e]
            node_count[global_n] += 1.0

    node_count = np.maximum(node_count, 1.0)
    ep_bar_node /= node_count

    grid.point_data["ep_bar"] = ep_bar_node

    # Apply displacement to get deformed mesh
    grid.points += final_u

    # -----------------------------------------------------------------------
    # 8. Render off-screen
    # -----------------------------------------------------------------------
    print("Rendering PyVista visualization...")

    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    plotter.set_background("white")

    plotter.add_mesh(
        grid,
        scalars="ep_bar",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={
            "title": "Equivalent Plastic Strain",
            "title_font_size": 18,
            "label_font_size": 14,
            "n_labels": 5,
            "fmt": "%.4f",
            "position_x": 0.82,
            "position_y": 0.15,
            "width": 0.12,
            "height": 0.7,
            "color": "black",
        },
        opacity=1.0,
    )

    # Camera: isometric-like view
    plotter.camera_position = [
        (3.0, 2.5, 2.0),   # camera position
        (0.0, 0.0, 0.0),   # focal point
        (0.0, 0.0, 1.0),   # view up
    ]

    # Save PNG
    png_path = OUT_DIR / "ep_plastic_strain.png"
    plotter.screenshot(str(png_path), transparent_background=False)
    print(f"Saved PNG: {png_path}")

    # Save PDF via matplotlib from the screenshot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.image import imread as mpl_imread

        img = mpl_imread(str(png_path))
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=200)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout(pad=0)
        pdf_path = OUT_DIR / "ep_plastic_strain.pdf"
        fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.02, dpi=200)
        plt.close(fig)
        print(f"Saved PDF: {pdf_path}")
    except Exception as e:
        print(f"Warning: could not save PDF: {e}")

    plotter.close()
    print("Done.")


if __name__ == "__main__":
    main()
