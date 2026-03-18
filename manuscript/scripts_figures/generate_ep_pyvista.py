#!/usr/bin/env python3
"""Generate PyVista 3D visualization of equivalent plastic strain on a torus
sector, plus multi-point hysteresis figures with true-vs-recovered overlays.

Creates:
  ep_plastic_strain.png / .pdf   -- 3D torus with annotated sample points
  ep_hysteresis_multipoint.png / .pdf -- 1x3 hysteresis loops at A, B, C
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Off-screen rendering (must come before any pyvista import)
# ---------------------------------------------------------------------------
import pyvista as pv
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.incremental_solver import IncrementalSolver
from experiments.torus_elastoplastic.return_mapping import (
    ReturnMappingState,
    smooth_return_map,
    sym_logm,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "figures_cmame_core"
    / "example5_torus_elastoplastic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Torus chart decoder (matches run_torus_elastoplastic_inverse.py)
# ---------------------------------------------------------------------------
R_MAJOR = 1.0
R_MINOR = 0.3
PHI_CENTER = 0.0
PHI_HALFWIDTH = math.pi / 4


def cube_to_torus(xi: torch.Tensor) -> torch.Tensor:
    """Map reference cube [-1,1]^3 nodes to torus physical coordinates.

    xi_0 -> phi   = phi_center + phi_halfwidth * xi_0
    xi_1 -> theta = pi * xi_1
    xi_2 -> rho   = r_minor * xi_2   (xi_2 in [0,1] after rescaling from [-1,1])
    """
    phi = PHI_CENTER + PHI_HALFWIDTH * xi[:, 0]
    theta = math.pi * xi[:, 1]
    # Map xi_2 from [-1,1] to [0, r_minor]
    rho = R_MINOR * 0.5 * (1.0 + xi[:, 2])

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rr = R_MAJOR + rho * cos_theta
    x = rr * cos_phi
    y = rr * sin_phi
    z = rho * sin_theta
    return torch.stack([x, y, z], dim=1)


# ---------------------------------------------------------------------------
# Forward solve helper
# ---------------------------------------------------------------------------
def run_forward_solve(
    fem, mu, K, tau_y_val, H_kin_val, bc_schedule, bc_mask, device="cpu"
):
    """Run the cyclic forward solve and return (u_hist, state_hist)."""
    tau_y_t = torch.tensor(tau_y_val, device=device)
    H_kin_t = torch.tensor(H_kin_val, device=device)
    inc_solver = IncrementalSolver(
        fem, mu, K, tau_y_t, H_kin_t, epsilon=1e-3,
    )
    with torch.no_grad():
        u_hist, state_hist = inc_solver.solve_history(
            bc_schedule, bc_mask, verbose=True, max_newton_iter=50, tol=1e-6,
        )
    return u_hist, state_hist


# ---------------------------------------------------------------------------
# Hysteresis extraction
# ---------------------------------------------------------------------------
def extract_hysteresis(fem, u_hist, state_hist, elem_idx):
    """Extract Kirchhoff stress tau_11 and logarithmic strain eps_11 at elem_idx.

    For each load step:
      - Compute F = I + grad(u) at the element
      - Compute log strain: eps = 0.5 * log(F^T F) -> eps_11
      - Get tau from Be via the return mapping state
    """
    eps_list = []
    tau_list = []
    mu_val = 200.0 / (2.0 * (1.0 + 0.3))  # for stress reconstruction
    K_val = 200.0 / (3.0 * (1.0 - 2.0 * 0.3))

    for step_idx in range(len(u_hist)):
        u = u_hist[step_idx].detach()
        state = state_hist[step_idx]

        # Deformation gradient at element
        F = fem.compute_F(u)  # (M, 3, 3)
        F_e = F[elem_idx]  # (3, 3)

        # Total logarithmic strain
        C = F_e.T @ F_e
        eps_log = 0.5 * sym_logm(C.unsqueeze(0)).squeeze(0)
        eps_11 = eps_log[0, 0].item()

        # Kirchhoff stress from elastic Be
        Be_e = state.Be[elem_idx]  # (3, 3)
        eps_e = 0.5 * sym_logm(Be_e.unsqueeze(0)).squeeze(0)
        tr_eps_e = torch.trace(eps_e).item()
        dev_eps_e = eps_e - (tr_eps_e / 3.0) * torch.eye(3, dtype=eps_e.dtype)
        tau_tensor = 2.0 * mu_val * dev_eps_e + K_val * tr_eps_e * torch.eye(
            3, dtype=eps_e.dtype
        )
        tau_11 = tau_tensor[0, 0].item()

        eps_list.append(eps_11)
        tau_list.append(tau_11)

    return np.array(eps_list), np.array(tau_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # ------------------------------------------------------------------
    # 1. Material parameters
    # ------------------------------------------------------------------
    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))
    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)

    # True and recovered parameters
    tau_y_true, H_kin_true = 0.5, 20.0
    tau_y_rec, H_kin_rec = 0.4988, 19.58

    # ------------------------------------------------------------------
    # 2. Build mesh
    # ------------------------------------------------------------------
    n_cells = 4
    fem = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, device=device, dtype=torch.float64,
    )
    nodes = fem.nodes
    tol_face = fem.h * 0.1
    r = fem.r

    print(f"Mesh: {fem.n_nodes} nodes, {fem.n_elements} elements")

    # ------------------------------------------------------------------
    # 3. Boundary conditions (cyclic uniaxial)
    # ------------------------------------------------------------------
    left_face = nodes[:, 0] < -r + tol_face
    right_face = nodes[:, 0] > r - tol_face
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
    print(
        f"Cyclic loading: 2 cycles, {n_steps_per_half} steps/half, "
        f"eps_peak={eps_peak}, total={total_steps} steps"
    )

    # ------------------------------------------------------------------
    # 4. Forward solve with TRUE parameters
    # ------------------------------------------------------------------
    print("\n=== Forward solve with TRUE parameters ===")
    u_hist_true, state_hist_true = run_forward_solve(
        fem, mu, K, tau_y_true, H_kin_true, bc_schedule, bc_mask, device
    )

    # ------------------------------------------------------------------
    # 5. Forward solve with RECOVERED parameters
    # ------------------------------------------------------------------
    print("\n=== Forward solve with RECOVERED parameters ===")
    u_hist_rec, state_hist_rec = run_forward_solve(
        fem, mu, K, tau_y_rec, H_kin_rec, bc_schedule, bc_mask, device
    )

    # ------------------------------------------------------------------
    # 6. Select 3 sample points (elements)
    # ------------------------------------------------------------------
    # Compute element centroids in reference coords
    elems_np = fem.elements.detach().cpu().numpy()
    nodes_np = fem.nodes.detach().cpu().numpy()
    n_elem = elems_np.shape[0]

    centroids_ref = np.zeros((n_elem, 3))
    for e in range(n_elem):
        centroids_ref[e] = nodes_np[elems_np[e]].mean(axis=0)

    # Sort elements by xi_0 (position along the loading axis)
    xi0_sorted = np.argsort(centroids_ref[:, 0])

    # Point A: near the right (loaded) face, high plastic strain
    elem_A = xi0_sorted[-n_elem // 8]
    # Point B: in the middle of the domain
    elem_B = xi0_sorted[n_elem // 2]
    # Point C: near the left (fixed) face / interior, low strain
    elem_C = xi0_sorted[n_elem // 8]

    sample_elems = {"A": elem_A, "B": elem_B, "C": elem_C}

    # Get final ep_bar at each point
    final_state = state_hist_true[-1]
    ep_bar_elem = final_state.ep_bar.detach().cpu().numpy()
    for label, eidx in sample_elems.items():
        print(
            f"  Point {label}: element {eidx}, "
            f"centroid_ref=({centroids_ref[eidx, 0]:.3f}, "
            f"{centroids_ref[eidx, 1]:.3f}, {centroids_ref[eidx, 2]:.3f}), "
            f"ep_bar={ep_bar_elem[eidx]:.6f}"
        )

    # ------------------------------------------------------------------
    # 7. Map mesh to torus and build PyVista grid
    # ------------------------------------------------------------------
    print("\nMapping cube mesh to torus geometry...")

    with torch.no_grad():
        nodes_torus = cube_to_torus(fem.nodes).cpu().numpy()

    # Apply scaled displacement (for visualization; scale factor for visibility)
    final_u = u_hist_true[-1].detach().cpu().numpy()
    # The displacement is in reference space; for visualization on the torus
    # we keep the torus geometry without deforming (the plastic strain is the
    # primary field to show).

    # Node-averaged ep_bar
    n_nodes = nodes_torus.shape[0]
    ep_bar_node = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)
    for e in range(n_elem):
        for ln in range(4):
            gn = elems_np[e, ln]
            ep_bar_node[gn] += ep_bar_elem[e]
            node_count[gn] += 1.0
    node_count = np.maximum(node_count, 1.0)
    ep_bar_node /= node_count

    # Build PyVista UnstructuredGrid
    cells = np.hstack(
        [np.full((n_elem, 1), 4, dtype=np.int64), elems_np]
    ).ravel()
    celltypes = np.full(n_elem, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes_torus)
    grid.point_data["ep_bar"] = ep_bar_node

    # Sample point locations on the torus (element centroids mapped to torus)
    centroids_torus = {}
    for label, eidx in sample_elems.items():
        c_ref = torch.tensor(
            centroids_ref[eidx : eidx + 1], dtype=torch.float64
        )
        c_torus = cube_to_torus(c_ref).cpu().numpy()[0]
        centroids_torus[label] = c_torus

    # ------------------------------------------------------------------
    # 8. Render PyVista 3D plastic strain plot with annotations
    # ------------------------------------------------------------------
    print("Rendering PyVista visualization...")

    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    plotter.set_background("white")

    plotter.add_mesh(
        grid,
        scalars="ep_bar",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={
            "title": "Equiv. Plastic Strain",
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

    # Add point labels with annotations
    colors = {"A": "red", "B": "darkorange", "C": "blue"}
    for label, eidx in sample_elems.items():
        pt = centroids_torus[label]
        ep_val = ep_bar_elem[eidx]

        # Add a sphere marker at the point
        sphere = pv.Sphere(radius=0.025, center=pt)
        plotter.add_mesh(sphere, color=colors[label], opacity=1.0)

        # Add text label
        plotter.add_point_labels(
            pv.PolyData(pt.reshape(1, 3)),
            [f"  {label} ($\\bar{{\\epsilon}}_p$={ep_val:.4f})"],
            font_size=16,
            text_color=colors[label],
            point_color=colors[label],
            point_size=0,
            render_points_as_spheres=False,
            always_visible=True,
            bold=True,
            shape=None,
        )

    # Camera: view the torus sector from an angle
    plotter.camera_position = [
        (2.0, -1.5, 1.5),  # camera position
        (0.9, 0.0, 0.0),  # focal point (center of torus sector)
        (0.0, 0.0, 1.0),  # view up
    ]

    # Save PNG
    png_path = OUT_DIR / "ep_plastic_strain.png"
    plotter.screenshot(str(png_path), transparent_background=False)
    print(f"Saved PNG: {png_path}")

    # Save PDF via matplotlib
    try:
        img = plt.imread(str(png_path))
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

    # ------------------------------------------------------------------
    # 9. Extract hysteresis data at 3 sample points
    # ------------------------------------------------------------------
    print("\nExtracting hysteresis data...")

    hysteresis_true = {}
    hysteresis_rec = {}
    for label, eidx in sample_elems.items():
        print(f"  Extracting at point {label} (element {eidx})...")
        eps_t, tau_t = extract_hysteresis(
            fem, u_hist_true, state_hist_true, eidx
        )
        eps_r, tau_r = extract_hysteresis(
            fem, u_hist_rec, state_hist_rec, eidx
        )
        hysteresis_true[label] = (eps_t, tau_t)
        hysteresis_rec[label] = (eps_r, tau_r)

    # ------------------------------------------------------------------
    # 10. Plot multi-point hysteresis figure (1x3)
    # ------------------------------------------------------------------
    print("\nGenerating multi-point hysteresis figure...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    labels_list = ["A", "B", "C"]
    subtitles = [
        "Point A (near loaded face)",
        "Point B (mid-domain)",
        "Point C (near fixed face)",
    ]

    for i, (label, subtitle) in enumerate(zip(labels_list, subtitles)):
        ax = axes[i]
        eps_t, tau_t = hysteresis_true[label]
        eps_r, tau_r = hysteresis_rec[label]

        # True
        ax.plot(
            eps_t, tau_t, "b-", linewidth=1.5, label="True", zorder=3
        )
        # Recovered
        ax.plot(
            eps_r,
            tau_r,
            "r--",
            linewidth=1.5,
            label="Recovered",
            zorder=2,
        )

        ax.set_xlabel(r"$\varepsilon_{11}$ (log strain)", fontsize=12)
        if i == 0:
            ax.set_ylabel(r"$\tau_{11}$ (Kirchhoff stress)", fontsize=12)
        ax.set_title(subtitle, fontsize=13)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        r"Stress--strain hysteresis: true ($\tau_y=0.5,\, H_\mathrm{kin}=20$) "
        r"vs recovered ($\tau_y=0.4988,\, H_\mathrm{kin}=19.58$)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    hyst_png = OUT_DIR / "ep_hysteresis_multipoint.png"
    hyst_pdf = OUT_DIR / "ep_hysteresis_multipoint.pdf"
    fig.savefig(str(hyst_png), bbox_inches="tight", dpi=200)
    fig.savefig(str(hyst_pdf), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {hyst_png}")
    print(f"Saved: {hyst_pdf}")

    print("\nDone.")


if __name__ == "__main__":
    main()
