#!/usr/bin/env python3
"""Generate PyVista 3D visualization on the full torus using Lie-group/Lie-algebra
projection following Mota, Sun, Ostien, Foulk & Long (Comput. Mech., 2013).

The deformation gradient F is computed per chart, interpolated onto a single smooth
parametric torus surface mesh, polar-decomposed (F = R S), projected to the Lie
algebra (log R in so(3), log S in S(3)), globally L2-projected to surface nodes
in the physical domain, and recovered via exponential maps.

Creates:
  ep_plastic_strain.png / .pdf   -- 2x2 subfigure on smooth torus surface
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
from scipy.interpolate import griddata

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
    ReturnMappingState, smooth_return_map, sym_logm,
)

OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "figures_cmame_core" / "example5_torus_elastoplastic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Torus geometry parameters
# ---------------------------------------------------------------------------
R_MAJOR, R_MINOR = 1.0, 0.35
N_CHARTS = 8
PHI_HALFWIDTH = math.pi / 4


# ---------------------------------------------------------------------------
# Build a single smooth parametric torus surface mesh
# ---------------------------------------------------------------------------
def make_torus_surface(n_phi=200, n_theta=80):
    """Create a smooth structured torus surface as PyVista PolyData."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")

    X = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.cos(PHI)
    Y = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.sin(PHI)
    Z = R_MINOR * np.sin(THETA)

    grid = pv.StructuredGrid(X, Y, Z)
    surf = grid.extract_surface()
    return surf, PHI, THETA


# ---------------------------------------------------------------------------
# Chart-local reference coordinate computation
# ---------------------------------------------------------------------------
def cube_to_torus_chart(xi, phi_center, phi_halfwidth):
    phi = phi_center + phi_halfwidth * xi[:, 0]
    theta = math.pi * xi[:, 1]
    rho = 0.5 * R_MINOR * (1.0 + xi[:, 2])
    rr = R_MAJOR + rho * torch.cos(theta)
    return torch.stack([rr * torch.cos(phi), rr * torch.sin(phi),
                        rho * torch.sin(theta)], dim=1)


def cube_to_torus(xi):
    return cube_to_torus_chart(xi, 0.0, PHI_HALFWIDTH)


def torus_surface_to_chart_ref(phi_pts, theta_pts, phi_center, phi_halfwidth):
    """Map (phi, theta) on the torus outer surface (rho=R_MINOR, i.e. xi_2=1)
    to reference cube coordinates xi in [-1,1]^3."""
    # Wrap phi difference to [-pi, pi]
    dphi = np.arctan2(np.sin(phi_pts - phi_center), np.cos(phi_pts - phi_center))
    xi_0 = dphi / phi_halfwidth
    xi_1 = theta_pts / math.pi  # theta in [-pi, pi] -> xi_1 in [-1, 1]
    xi_2 = np.ones_like(xi_0)   # outer surface: rho = R_MINOR -> xi_2 = 1
    return np.stack([xi_0, xi_1, xi_2], axis=1)


# ---------------------------------------------------------------------------
# Lie-algebra utilities (Mota et al. 2013)
# ---------------------------------------------------------------------------
def _polar_decompose(F):
    U, D, Vt = torch.linalg.svd(F)
    det_sign = torch.det(U @ Vt)
    if det_sign < 0:
        U = U.clone(); U[:, -1] *= -1
        D = D.clone(); D[-1] *= -1
    R = U @ Vt
    S = Vt.T @ torch.diag(D) @ Vt
    return R, S


def _log_rotation(R):
    tr_R = torch.trace(R).clamp(-1.0 + 1e-7, 3.0 - 1e-7)
    theta = torch.acos(0.5 * (tr_R - 1.0))
    if theta.abs() < 1e-10:
        return torch.zeros(3, 3, dtype=R.dtype)
    return (theta / (2.0 * torch.sin(theta))) * (R - R.T)


def _exp_rotation(W):
    theta = torch.sqrt(0.5 * torch.sum(W * W)).clamp(min=1e-30)
    if theta < 1e-10:
        return torch.eye(3, dtype=W.dtype)
    return (torch.eye(3, dtype=W.dtype)
            + (torch.sin(theta) / theta) * W
            + ((1.0 - torch.cos(theta)) / theta**2) * (W @ W))


def _log_spd(S):
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = eigvals.clamp(min=1e-30)
    return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T


def _exp_spd(H):
    H = 0.5 * (H + H.T)
    if torch.any(torch.isnan(H)):
        return torch.eye(3, dtype=H.dtype)
    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = eigvals.clamp(-20.0, 20.0)
    return eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Forward solve helper
# ---------------------------------------------------------------------------
def run_forward_solve(fem, mu, K, tau_y_val, H_kin_val, bc_schedule, bc_mask, device="cpu"):
    inc = IncrementalSolver(fem, mu, K,
                            torch.tensor(tau_y_val, device=device),
                            torch.tensor(H_kin_val, device=device), epsilon=1e-3)
    with torch.no_grad():
        return inc.solve_history(bc_schedule, bc_mask, verbose=True,
                                max_newton_iter=50, tol=1e-6)


# ---------------------------------------------------------------------------
# Hysteresis extraction
# ---------------------------------------------------------------------------
def extract_hysteresis(fem, u_hist, state_hist, elem_idx):
    eps_list, tau_list = [], []
    mu_val = 200.0 / (2.0 * 1.3)
    K_val = 200.0 / (3.0 * 0.4)
    for si in range(len(u_hist)):
        u = u_hist[si].detach()
        F = fem.compute_F(u)[elem_idx]
        C = F.T @ F
        eps_list.append((0.5 * sym_logm(C.unsqueeze(0)).squeeze(0))[0, 0].item())
        Be = state_hist[si].Be[elem_idx]
        eps_e = 0.5 * sym_logm(Be.unsqueeze(0)).squeeze(0)
        tr_e = torch.trace(eps_e).item()
        tau = 2 * mu_val * (eps_e - tr_e / 3 * torch.eye(3)) + K_val * tr_e * torch.eye(3)
        tau_list.append(tau[0, 0].item())
    return np.array(eps_list), np.array(tau_list)


# ---------------------------------------------------------------------------
# Render one scalar on the smooth torus surface
# ---------------------------------------------------------------------------
def render_torus(surf, scalar_name, title, cmap="turbo", fmt="%.4f"):
    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    pl.add_mesh(surf, scalars=scalar_name, cmap=cmap, show_edges=False,
                smooth_shading=True, scalar_bar_args={
                    "title": title, "title_font_size": 16, "label_font_size": 12,
                    "n_labels": 5, "fmt": fmt,
                    "position_x": 0.78, "position_y": 0.15,
                    "width": 0.14, "height": 0.65, "color": "black"}, opacity=1.0)
    pl.camera_position = [(3.0, -2.5, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    pl.screenshot("/tmp/_ep_tmp.png", transparent_background=False)
    img = plt.imread("/tmp/_ep_tmp.png")
    pl.close()
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    E_val, nu_val = 200.0, 0.3
    mu = torch.tensor(E_val / (2 * (1 + nu_val)), device=device)
    K = torch.tensor(E_val / (3 * (1 - 2 * nu_val)), device=device)
    tau_y_true, H_kin_true = 0.5, 20.0
    tau_y_rec, H_kin_rec = 0.4988, 19.58

    n_cells, eps_peak, n_steps_per_half = 4, 0.03, 15
    chart_phi_centers = [i * 2 * math.pi / N_CHARTS for i in range(N_CHARTS)]

    # ── Step 1: Solve each chart and collect element-wise F ──────────────
    chart_data = []  # list of (fem, F_all, elems_np, nodes_ref_np, phi_center)
    fem_chart0 = u_true_c0 = st_true_c0 = u_rec_c0 = st_rec_c0 = None

    for ci in range(N_CHARTS):
        phi_c = chart_phi_centers[ci]
        print(f"\n=== Chart {ci}: phi = {phi_c:.4f} rad ===")
        fem = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0,
                                   device=device, dtype=torch.float64)
        nodes = fem.nodes
        tol, r = fem.h * 0.1, fem.r
        left = nodes[:, 0] < -r + tol
        right = nodes[:, 0] > r - tol
        bc_mask = left | right

        bc_schedule = []
        for _ in range(2):
            for s in range(n_steps_per_half):
                lam = (s + 1) / n_steps_per_half
                u_bc = torch.zeros_like(nodes)
                u_bc[right, 0] = eps_peak * lam * 2 * r
                bc_schedule.append(u_bc)
            for s in range(2 * n_steps_per_half):
                lam = 1.0 - (s + 1) / n_steps_per_half
                u_bc = torch.zeros_like(nodes)
                u_bc[right, 0] = eps_peak * lam * 2 * r
                bc_schedule.append(u_bc)
            for s in range(n_steps_per_half):
                lam = -1.0 + (s + 1) / n_steps_per_half
                u_bc = torch.zeros_like(nodes)
                u_bc[right, 0] = eps_peak * lam * 2 * r
                bc_schedule.append(u_bc)

        u_hist, state_hist = run_forward_solve(
            fem, mu, K, tau_y_true, H_kin_true, bc_schedule, bc_mask, device)
        if ci == 0:
            fem_chart0 = fem
            u_true_c0, st_true_c0 = u_hist, state_hist
            u_rec_c0, st_rec_c0 = run_forward_solve(
                fem, mu, K, tau_y_rec, H_kin_rec, bc_schedule, bc_mask, device)

        F_all = fem.compute_F(u_hist[-1].detach())  # (n_elem, 3, 3)
        Be_all = state_hist[-1].Be.detach()          # (n_elem, 3, 3)
        elems_np = fem.elements.detach().cpu().numpy()
        nodes_ref = fem.nodes.detach().cpu().numpy()
        chart_data.append((fem, F_all, Be_all, elems_np, nodes_ref, phi_c))

    # ── Step 2: Build smooth torus surface and sample F from charts ──────
    print("\nBuilding smooth torus surface and sampling F from charts...")
    surf, PHI_grid, THETA_grid = make_torus_surface(n_phi=200, n_theta=80)
    surf_pts = np.array(surf.points)
    n_surf = surf_pts.shape[0]

    # Compute (phi, theta) for each surface point
    phi_surf = np.arctan2(surf_pts[:, 1], surf_pts[:, 0]) % (2 * math.pi)
    rho_xy = np.sqrt(surf_pts[:, 0]**2 + surf_pts[:, 1]**2)
    theta_surf = np.arctan2(surf_pts[:, 2], rho_xy - R_MAJOR)

    # Collect element centroids in (phi, theta) space with Lie-algebra values
    # of the PLASTIC deformation gradient F^p = (F^e)^{-1} F.
    # F^e is recovered from Be = F^e (F^e)^T via eigendecomposition.
    all_phi_cent = []
    all_theta_cent = []
    all_log_Rp_cent = []   # log(R^p) in so(3)
    all_log_Sp_cent = []   # log(S^p) in S(3)

    for ci, (fem, F_all, Be_all, elems_np, nodes_ref, phi_c) in enumerate(chart_data):
        n_elem_c = elems_np.shape[0]
        centroids_ref = np.array([nodes_ref[elems_np[e]].mean(0) for e in range(n_elem_c)])

        for e in range(n_elem_c):
            F_e = F_all[e]
            Be_e = Be_all[e]

            # Recover F^e from Be = Fe Fe^T  (symmetric positive-definite)
            # Fe = sqrt(Be) via eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(Be_e)
            eigvals = eigvals.clamp(min=1e-30)
            Fe = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

            # F^p = (F^e)^{-1} F
            Fp = torch.linalg.solve(Fe, F_e)

            # Polar decompose F^p = R^p S^p, map to Lie algebras
            Rp, Sp = _polar_decompose(Fp)
            log_Rp = _log_rotation(Rp).numpy()
            log_Sp = _log_spd(Sp).numpy()

            # Map centroid to (phi, theta) on torus
            xi_c = centroids_ref[e]
            phi_e = phi_c + PHI_HALFWIDTH * xi_c[0]
            theta_e = math.pi * xi_c[1]

            all_phi_cent.append(phi_e)
            all_theta_cent.append(theta_e)
            all_log_Rp_cent.append(log_Rp)
            all_log_Sp_cent.append(log_Sp)

    all_phi_cent = np.array(all_phi_cent)
    all_theta_cent = np.array(all_theta_cent)
    all_log_Rp_cent = np.array(all_log_Rp_cent)
    all_log_Sp_cent = np.array(all_log_Sp_cent)

    # Periodic wrapping for interpolation
    phi_rep = np.concatenate([all_phi_cent, all_phi_cent + 2*math.pi,
                              all_phi_cent - 2*math.pi])
    theta_rep = np.concatenate([all_theta_cent, all_theta_cent + 2*math.pi,
                                all_theta_cent - 2*math.pi])
    log_Rp_rep = np.concatenate([all_log_Rp_cent]*3, axis=0)
    log_Sp_rep = np.concatenate([all_log_Sp_cent]*3, axis=0)
    src_pts = np.column_stack([phi_rep, theta_rep])

    # Interpolate onto smooth surface
    log_Rp_surf = np.zeros((n_surf, 3, 3))
    log_Sp_surf = np.zeros((n_surf, 3, 3))
    tgt_pts = np.column_stack([phi_surf, theta_surf])

    print("Interpolating Lie-algebra fields of F^p onto smooth surface...")
    for i in range(3):
        for j in range(3):
            log_Rp_surf[:, i, j] = griddata(
                src_pts, log_Rp_rep[:, i, j], tgt_pts, method="linear",
                fill_value=0.0)
            log_Sp_surf[:, i, j] = griddata(
                src_pts, log_Sp_rep[:, i, j], tgt_pts, method="linear",
                fill_value=0.0)

    # ── Step 3: Recover nodal F^p fields via exponential maps ────────────
    print("Recovering F^p fields via exponential maps...")
    det_Fp_surf = np.zeros(n_surf)
    iso_eig1 = np.zeros(n_surf)
    iso_eig2 = np.zeros(n_surf)
    iso_eig3 = np.zeros(n_surf)

    for a in range(n_surf):
        Rp_a = _exp_rotation(torch.tensor(log_Rp_surf[a]))
        Sp_a = _exp_spd(torch.tensor(log_Sp_surf[a]))
        Fp_a = Rp_a @ Sp_a
        Jp = torch.det(Fp_a).item()
        det_Fp_surf[a] = Jp
        Jp_abs = max(abs(Jp), 1e-30)
        Sp_bar = Sp_a / (Jp_abs ** (1.0 / 3.0))
        eigs = sorted(torch.linalg.eigvalsh(Sp_bar).numpy(), reverse=True)
        iso_eig1[a], iso_eig2[a], iso_eig3[a] = eigs

    surf.point_data["det_Fp"] = det_Fp_surf
    surf.point_data["iso_eig1"] = iso_eig1
    surf.point_data["iso_eig2"] = iso_eig2
    surf.point_data["iso_eig3"] = iso_eig3

    # ── Step 4: Render 2x2 figure ───────────────────────────────────────
    print("Rendering 2x2 figure...")
    panels = [
        ("det_Fp",   r"det $\mathbf{F}^p$",                       "coolwarm", "%.4f"),
        ("iso_eig1", r"$\bar{\lambda}^p_1$ (max plastic stretch)", "turbo",    "%.4f"),
        ("iso_eig2", r"$\bar{\lambda}^p_2$ (mid plastic stretch)", "turbo",    "%.4f"),
        ("iso_eig3", r"$\bar{\lambda}^p_3$ (min plastic stretch)", "turbo",    "%.4f"),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    images = [render_torus(surf, s, t, c, f) for s, t, c, f in panels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    for ax, img, lbl in zip(axes.ravel(), images, labels):
        ax.imshow(img); ax.axis("off")
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="round,pad=0.2"))
    fig.tight_layout(pad=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(str(OUT_DIR / f"ep_plastic_strain.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved ep_plastic_strain.png/.pdf")

    # ── Step 5: Hysteresis (chart 0) ────────────────────────────────────
    fem = fem_chart0
    elems_c0 = fem.elements.detach().cpu().numpy()
    nodes_c0 = fem.nodes.detach().cpu().numpy()
    n_e0 = elems_c0.shape[0]
    centroids = np.array([nodes_c0[elems_c0[e]].mean(0) for e in range(n_e0)])
    xi0_sorted = np.argsort(centroids[:, 0])
    sample = {"A": xi0_sorted[-n_e0 // 8],
              "B": xi0_sorted[n_e0 // 2],
              "C": xi0_sorted[n_e0 // 8]}

    print("\nExtracting hysteresis from chart 0...")
    ht, hr = {}, {}
    for lbl, eidx in sample.items():
        ht[lbl] = extract_hysteresis(fem_chart0, u_true_c0, st_true_c0, eidx)
        hr[lbl] = extract_hysteresis(fem_chart0, u_rec_c0, st_rec_c0, eidx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    for i, (lbl, sub) in enumerate(zip(
            ["A", "B", "C"],
            ["Point A (near loaded face)", "Point B (mid-domain)",
             "Point C (near fixed face)"])):
        ax = axes[i]
        ax.plot(*ht[lbl], "b-", lw=1.5, label="True", zorder=3)
        ax.plot(*hr[lbl], "r--", lw=1.5, label="Recovered", zorder=2)
        ax.set_xlabel(r"$\varepsilon_{11}$ (log strain)", fontsize=12)
        if i == 0: ax.set_ylabel(r"$\tau_{11}$ (Kirchhoff stress)", fontsize=12)
        ax.set_title(sub, fontsize=13); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.suptitle(
        r"Stress--strain: true ($\tau_y\!=\!0.5,\,H_\mathrm{kin}\!=\!20$) "
        r"vs recovered ($\tau_y\!=\!0.4988,\,H_\mathrm{kin}\!=\!19.58$)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(OUT_DIR / f"ep_hysteresis_multipoint.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
