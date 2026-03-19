#!/usr/bin/env python3
"""Generate PyVista 3D visualization of F^p fields on the torus from the
forward BVP (n_cells=8, finest mesh).

Plots: (a) det(F^p), (b) largest eigenvalue of S^p, (c) lambda_1^p - lambda_3^p.
Uses Lie-group interpolation onto a smooth parametric torus surface.
"""

from __future__ import annotations
import math, sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

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
from experiments.torus_elastoplastic.run_forward_bvp_schwarz import (
    TorusSDF, TorusChartDecoder, SchwarzVectorSolver,
    make_ep_stress_fn, make_ep_tangent_fn, build_bc,
    R_MAJOR, R_MINOR, N_CHARTS,
)

OUT_DIR = Path(_REPO) / "manuscript" / "figures_cmame_core" / "example_forward_bvp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHI_HALFWIDTH = math.pi / 4

# ── Lie-algebra utilities ─────────────────────────────────────────────
def _polar_decompose(F):
    U, D, Vt = torch.linalg.svd(F)
    if torch.det(U @ Vt) < 0:
        U = U.clone(); U[:, -1] *= -1; D = D.clone(); D[-1] *= -1
    return U @ Vt, Vt.T @ torch.diag(D) @ Vt

def _log_rotation(R):
    tr = torch.trace(R).clamp(-1 + 1e-7, 3 - 1e-7)
    th = torch.acos(0.5 * (tr - 1))
    return torch.zeros(3, 3, dtype=R.dtype) if th.abs() < 1e-10 else (th / (2 * torch.sin(th))) * (R - R.T)

def _log_spd(S):
    ev, V = torch.linalg.eigh(S)
    return V @ torch.diag(torch.log(ev.clamp(min=1e-30))) @ V.T

def _exp_rotation(W):
    th = torch.sqrt(0.5 * torch.sum(W * W)).clamp(min=1e-30)
    if th < 1e-10: return torch.eye(3, dtype=W.dtype)
    I = torch.eye(3, dtype=W.dtype)
    return I + (torch.sin(th) / th) * W + ((1 - torch.cos(th)) / th**2) * (W @ W)

def _exp_spd(H):
    H = 0.5 * (H + H.T)
    if torch.any(torch.isnan(H)): return torch.eye(3, dtype=H.dtype)
    ev, V = torch.linalg.eigh(H)
    return V @ torch.diag(torch.exp(ev.clamp(-20, 20))) @ V.T

# ── Smooth torus surface ──────────────────────────────────────────────
def make_torus_surface(n_phi=200, n_theta=80):
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    X = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.cos(PHI)
    Y = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.sin(PHI)
    Z = R_MINOR * np.sin(THETA)
    grid = pv.StructuredGrid(X, Y, Z)
    return grid.extract_surface(), PHI, THETA

def cube_to_torus_chart(xi, phi_c, phw):
    phi = phi_c + phw * xi[:, 0]
    theta = math.pi * xi[:, 1]
    rho = 0.5 * R_MINOR * (1 + xi[:, 2])
    rr = R_MAJOR + rho * torch.cos(theta)
    return torch.stack([rr * torch.cos(phi), rr * torch.sin(phi),
                        rho * torch.sin(theta)], dim=1)

# ── Render helper ─────────────────────────────────────────────────────
def render_panel(surf, scalar, title, cmap="turbo", fmt="%.4f"):
    pl = pv.Plotter(off_screen=True, window_size=[900, 700])
    pl.set_background("white")
    pl.add_mesh(surf, scalars=scalar, cmap=cmap, show_edges=False,
                smooth_shading=True, scalar_bar_args={
                    "title": title, "title_font_size": 16, "label_font_size": 12,
                    "n_labels": 5, "fmt": fmt,
                    "position_x": 0.78, "position_y": 0.15,
                    "width": 0.14, "height": 0.65, "color": "black"}, opacity=1.0)
    # Same camera as torus inverse figures (Fig 9/11 in manuscript)
    pl.camera_position = [(3.0, -2.5, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    pl.screenshot("/tmp/_ep_bvp_tmp.png", transparent_background=False)
    img = plt.imread("/tmp/_ep_bvp_tmp.png")
    pl.close()
    return img

# ── Main ──────────────────────────────────────────────────────────────
def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    n_cells = 8  # finest mesh for publication

    E, nu = 200.0, 0.3
    mu = torch.tensor(E / (2 * (1 + nu)), device=device)
    K_mat = torch.tensor(E / (3 * (1 - 2 * nu)), device=device)
    tau_y = torch.tensor(0.5, device=device)
    H_kin = torch.tensor(20.0, device=device)
    eps_rm = 0.01

    # Build 8-chart atlas
    sdf = TorusSDF()
    phi_centers = [i * 2 * math.pi / N_CHARTS for i in range(N_CHARTS)]
    decoders = [TorusChartDecoder(phi_center=pc) for pc in phi_centers]
    solvers = [ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0,
               chart_decoder=dec, sdf_oracle=sdf, sdf_threshold=-0.005,
               device=device, dtype=torch.float64) for dec in decoders]

    schwarz = SchwarzVectorSolver(solvers=solvers, chart_decoders=decoders)
    states = [ReturnMappingState.zeros((s.n_elements,), device=device,
              dtype=torch.float64) for s in solvers]
    F_olds = [torch.eye(3, device=device, dtype=torch.float64
              ).unsqueeze(0).expand(s.n_elements, 3, 3).clone() for s in solvers]

    # Loading schedule: 1 cycle, 30 steps/half
    delta_max = 0.02 * R_MINOR
    sph = 30
    deltas = []
    for s in range(sph):
        deltas.append(delta_max * (s + 1) / sph)
    for s in range(2 * sph):
        deltas.append(delta_max * (1 - (s + 1) / sph))
    for s in range(sph):
        deltas.append(delta_max * (-1 + (s + 1) / sph))

    print(f"Running {len(deltas)} load steps on n_cells={n_cells} (8 charts)...")
    for step_idx, delta in enumerate(deltas):
        bc_masks, u_bcs = build_bc(solvers, decoders, delta)
        sfns = [make_ep_stress_fn(states[ci], F_olds[ci], mu, K_mat, tau_y, H_kin, eps_rm)
                for ci in range(N_CHARTS)]
        tfns = [make_ep_tangent_fn(states[ci], F_olds[ci], mu, K_mat, tau_y, H_kin, eps_rm)
                for ci in range(N_CHARTS)]

        for sweep in range(8):
            for group in schwarz.color_groups:
                for ci in group:
                    if solvers[ci].n_elements == 0: continue
                    im, iv = schwarz._interpolate_interface_bc(ci)
                    bm = bc_masks[ci].clone(); bv = u_bcs[ci].clone()
                    io = im & ~bm; bm |= io; bv[io] = iv[io]
                    fe = torch.zeros(solvers[ci].n_nodes, 3, device=device, dtype=torch.float64)
                    try:
                        schwarz.u_charts[ci] = solvers[ci].solve_nonlinear(
                            stress_fn=sfns[ci], tangent_fn=tfns[ci], f_ext=fe,
                            u_bc=bv, bc_mask=bm, u_init=schwarz.u_charts[ci],
                            max_iter=25, tol=1e-7)
                    except Exception as e:
                        print(f"  [WARN] chart {ci}: {e}")

        with torch.no_grad():
            for ci in range(N_CHARTS):
                if solvers[ci].n_elements == 0: continue
                Fc = solvers[ci].compute_F(schwarz.u_charts[ci])
                Fdi = torch.einsum("eij,ejk->eik", Fc, torch.linalg.inv(F_olds[ci]))
                _, ns = smooth_return_map(Fdi, states[ci], mu, K_mat, tau_y, H_kin, eps_rm)
                states[ci] = ns; F_olds[ci] = Fc.detach().clone()

        me = max(states[ci].ep_bar.max().item() for ci in range(N_CHARTS) if solvers[ci].n_elements > 0)
        if (step_idx + 1) % 20 == 0 or step_idx == len(deltas) - 1:
            print(f"  Step {step_idx+1}/{len(deltas)}: ep_bar={me:.4e}")

        # Save per-step VTU snapshots every 10 steps (in physical coordinates)
        if (step_idx + 1) % 10 == 0 or step_idx == len(deltas) - 1:
            step_dir = OUT_DIR / "vtu" / f"step_{step_idx+1:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            blocks = pv.MultiBlock()
            for ci in range(N_CHARTS):
                if solvers[ci].n_elements == 0: continue
                enp_s = solvers[ci].elements.detach().cpu().numpy()
                u_s = schwarz.u_charts[ci].detach().cpu().numpy()
                ep_s = states[ci].ep_bar.detach().cpu().numpy()
                with torch.no_grad():
                    np_s = cube_to_torus_chart(solvers[ci].nodes, phi_centers[ci], PHI_HALFWIDTH).cpu().numpy()
                ne_s = enp_s.shape[0]
                cells_s = np.hstack([np.full((ne_s, 1), 4, dtype=np.int64), enp_s]).ravel()
                ct_s = np.full(ne_s, pv.CellType.TETRA, dtype=np.uint8)
                g = pv.UnstructuredGrid(cells_s, ct_s, np_s)
                g.point_data["displacement"] = u_s
                g.point_data["displacement_mag"] = np.linalg.norm(u_s, axis=1)
                g.cell_data["ep_bar"] = ep_s
                g.cell_data["chart_id"] = np.full(ne_s, ci)
                g.save(str(step_dir / f"chart_{ci:02d}.vtu"))
                blocks.append(g, f"chart_{ci}")
            # Save combined multi-block (preserves all charts without merge artifacts)
            blocks.save(str(step_dir / "all_charts.vtm"))
            # Also save a flat concatenation for convenience
            if blocks.n_blocks > 0:
                combined = blocks[0]
                for bi in range(1, blocks.n_blocks):
                    combined = combined.merge(blocks[bi])
                combined.save(str(step_dir / "all_charts_concatenated.vtu"))
            print(f"    Saved VTU snapshot step {step_idx+1}: {step_dir}")

    # ── Export per-chart VTU files for inspection ───────────────────
    vtu_dir = OUT_DIR / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting per-chart VTU files to {vtu_dir}...")
    for ci in range(N_CHARTS):
        if solvers[ci].n_elements == 0:
            continue
        enp = solvers[ci].elements.detach().cpu().numpy()
        nnp = solvers[ci].nodes.detach().cpu().numpy()
        u_ci = schwarz.u_charts[ci].detach().cpu().numpy()
        Fa = solvers[ci].compute_F(schwarz.u_charts[ci])
        Be_ci = states[ci].Be.detach()
        ep_bar_ci = states[ci].ep_bar.detach().cpu().numpy()

        # Map nodes to torus
        with torch.no_grad():
            nodes_phys = cube_to_torus_chart(solvers[ci].nodes, phi_centers[ci], PHI_HALFWIDTH).cpu().numpy()

        # Compute per-element F^p quantities
        ne = enp.shape[0]
        det_Fp_e = np.zeros(ne)
        eig1_Sp_e = np.zeros(ne)
        eig_diff_e = np.zeros(ne)
        for e in range(ne):
            ev, V = torch.linalg.eigh(Be_ci[e])
            Fe_e = V @ torch.diag(torch.sqrt(ev.clamp(min=1e-30))) @ V.T
            Fp_e = torch.linalg.solve(Fe_e, Fa[e])
            det_Fp_e[e] = torch.det(Fp_e).item()
            _, Sp_e = _polar_decompose(Fp_e)
            eigs = sorted(torch.linalg.eigvalsh(Sp_e).numpy(), reverse=True)
            eig1_Sp_e[e] = eigs[0]
            eig_diff_e[e] = eigs[0] - eigs[2]

        # Build PyVista grid in physical (torus) coordinates
        cells = np.hstack([np.full((ne, 1), 4, dtype=np.int64), enp]).ravel()
        ctypes = np.full(ne, pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, ctypes, nodes_phys)
        grid.point_data["displacement"] = u_ci
        grid.point_data["displacement_mag"] = np.linalg.norm(u_ci, axis=1)
        grid.cell_data["ep_bar"] = ep_bar_ci
        grid.cell_data["det_Fp"] = det_Fp_e
        grid.cell_data["eig1_Sp"] = eig1_Sp_e
        grid.cell_data["eig_diff_Sp"] = eig_diff_e
        grid.cell_data["chart_id"] = np.full(ne, ci)

        vtu_path = vtu_dir / f"chart_{ci:02d}.vtu"
        grid.save(str(vtu_path))
        print(f"  Chart {ci}: {vtu_path.name} ({ne} elements, "
              f"det_Fp range [{det_Fp_e.min():.4f}, {det_Fp_e.max():.4f}], "
              f"ep_bar max {ep_bar_ci.max():.4e})")

    # Save combined multi-block (correct: preserves all charts separately)
    final_blocks = pv.MultiBlock()
    for ci in range(N_CHARTS):
        vtu_path = vtu_dir / f"chart_{ci:02d}.vtu"
        if vtu_path.exists():
            final_blocks.append(pv.read(str(vtu_path)), f"chart_{ci}")
    if final_blocks.n_blocks > 0:
        final_blocks.save(str(vtu_dir / "all_charts.vtm"))
        # Also save flat concatenation
        combined = final_blocks[0]
        for bi in range(1, final_blocks.n_blocks):
            combined = combined.merge(final_blocks[bi])
        combined.save(str(vtu_dir / "all_charts_concatenated.vtu"))
        print(f"  Combined: all_charts_concatenated.vtu ({combined.n_cells} cells)")

    # ── Collect F^p via Lie algebra ───────────────────────────────────
    print("\nCollecting F^p fields...")
    surf, _, _ = make_torus_surface(n_phi=200, n_theta=80)
    sp = np.array(surf.points)
    n_s = sp.shape[0]
    phi_s = np.arctan2(sp[:, 1], sp[:, 0]) % (2 * math.pi)
    rho_xy = np.sqrt(sp[:, 0]**2 + sp[:, 1]**2)
    theta_s = np.arctan2(sp[:, 2], rho_xy - R_MAJOR)

    # Collect element centroid positions in PHYSICAL Cartesian (x,y,z) on the
    # torus surface, plus their Lie-algebra F^p data.
    all_xyz_src = []   # (N_total, 3) physical coords of element centroids
    all_logRp, all_logSp = [], []

    for ci in range(N_CHARTS):
        if solvers[ci].n_elements == 0: continue
        enp = solvers[ci].elements.detach().cpu().numpy()
        nnp = solvers[ci].nodes.detach().cpu().numpy()
        Fa = solvers[ci].compute_F(schwarz.u_charts[ci])
        Be = states[ci].Be.detach()
        cents_ref = np.array([nnp[enp[e]].mean(0) for e in range(enp.shape[0])])

        # Map centroids to physical torus coordinates
        with torch.no_grad():
            cents_phys = cube_to_torus_chart(
                torch.tensor(cents_ref, dtype=torch.float64),
                phi_centers[ci], PHI_HALFWIDTH
            ).cpu().numpy()

        for e in range(enp.shape[0]):
            ev, V = torch.linalg.eigh(Be[e])
            Fe = V @ torch.diag(torch.sqrt(ev.clamp(min=1e-30))) @ V.T
            Fp = torch.linalg.solve(Fe, Fa[e])
            Rp, Sp = _polar_decompose(Fp)
            all_xyz_src.append(cents_phys[e])
            all_logRp.append(_log_rotation(Rp).numpy())
            all_logSp.append(_log_spd(Sp).numpy())

    all_xyz_src = np.array(all_xyz_src)  # (N_total, 3)
    all_logRp = np.array(all_logRp)
    all_logSp = np.array(all_logSp)

    # Interpolate in 3D Cartesian space (no periodicity wrapping needed —
    # the torus surface is closed and the physical coordinates are unique)
    src_3d = all_xyz_src
    tgt_3d = sp  # surface mesh points in physical (x,y,z)

    # Use linear interpolation with nearest-neighbor fallback for points
    # outside the convex hull (surface points may lie outside the hull of
    # volumetric element centroids).
    print("Interpolating Lie-algebra fields in Cartesian (x,y,z)...")
    logRs = np.zeros((n_s, 3, 3)); logSs = np.zeros((n_s, 3, 3))
    for i in range(3):
        for j in range(3):
            vals_lin_R = griddata(src_3d, all_logRp[:, i, j],
                                 tgt_3d, method="linear", fill_value=np.nan)
            vals_nn_R = griddata(src_3d, all_logRp[:, i, j],
                                tgt_3d, method="nearest")
            mask_R = np.isnan(vals_lin_R)
            logRs[:, i, j] = np.where(mask_R, vals_nn_R, vals_lin_R)

            vals_lin_S = griddata(src_3d, all_logSp[:, i, j],
                                 tgt_3d, method="linear", fill_value=np.nan)
            vals_nn_S = griddata(src_3d, all_logSp[:, i, j],
                                tgt_3d, method="nearest")
            mask_S = np.isnan(vals_lin_S)
            logSs[:, i, j] = np.where(mask_S, vals_nn_S, vals_lin_S)
    n_fallback = int(mask_R.sum())
    print(f"  {n_fallback}/{n_s} surface points used nearest-neighbor fallback "
          f"({100*n_fallback/n_s:.1f}%)")

    # Recover fields
    print("Recovering nodal fields via exp maps...")
    det_Fp = np.zeros(n_s); eig1 = np.zeros(n_s); eig_diff = np.zeros(n_s)
    for a in range(n_s):
        Ra = _exp_rotation(torch.tensor(logRs[a]))
        Sa = _exp_spd(torch.tensor(logSs[a]))
        Fpa = Ra @ Sa
        det_Fp[a] = torch.det(Fpa).item()
        eigs = sorted(torch.linalg.eigvalsh(Sa).numpy(), reverse=True)
        eig1[a] = eigs[0]
        eig_diff[a] = eigs[0] - eigs[2]

    surf.point_data["det_Fp"] = det_Fp
    surf.point_data["eig1_Sp"] = eig1
    surf.point_data["eig_diff"] = eig_diff

    # ── Render 1x3 figure ─────────────────────────────────────────────
    print("Rendering 1x3 figure...")
    panels = [
        ("det_Fp",   r"det $\mathbf{F}^p$",                                "coolwarm", "%.4f"),
        ("eig1_Sp",  r"$\lambda^p_1$ (max stretch of $\mathbf{S}^p$)",     "turbo",    "%.4f"),
        ("eig_diff", r"$\lambda^p_1 - \lambda^p_3$",                       "turbo",    "%.4f"),
    ]
    labels = ["(a)", "(b)", "(c)"]
    images = [render_panel(surf, s, t, c, f) for s, t, c, f in panels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(img); ax.axis("off")
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="round,pad=0.2"))
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(str(OUT_DIR / f"forward_bvp_Fp_fields.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved forward_bvp_Fp_fields.{{png,pdf}} to {OUT_DIR}")


if __name__ == "__main__":
    main()
