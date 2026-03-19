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
    n_cells = 8  # finest mesh

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

        if (step_idx + 1) % 20 == 0 or step_idx == len(deltas) - 1:
            me = max(states[ci].ep_bar.max().item() for ci in range(N_CHARTS) if solvers[ci].n_elements > 0)
            print(f"  Step {step_idx+1}/{len(deltas)}: ep_bar={me:.4e}")

    # ── Collect F^p via Lie algebra ───────────────────────────────────
    print("\nCollecting F^p fields...")
    surf, _, _ = make_torus_surface(n_phi=200, n_theta=80)
    sp = np.array(surf.points)
    n_s = sp.shape[0]
    phi_s = np.arctan2(sp[:, 1], sp[:, 0]) % (2 * math.pi)
    rho_xy = np.sqrt(sp[:, 0]**2 + sp[:, 1]**2)
    theta_s = np.arctan2(sp[:, 2], rho_xy - R_MAJOR)

    all_phi, all_theta, all_logRp, all_logSp = [], [], [], []
    for ci in range(N_CHARTS):
        if solvers[ci].n_elements == 0: continue
        enp = solvers[ci].elements.detach().cpu().numpy()
        nnp = solvers[ci].nodes.detach().cpu().numpy()
        Fa = solvers[ci].compute_F(schwarz.u_charts[ci])
        Be = states[ci].Be.detach()
        cents = np.array([nnp[enp[e]].mean(0) for e in range(enp.shape[0])])
        for e in range(enp.shape[0]):
            ev, V = torch.linalg.eigh(Be[e])
            Fe = V @ torch.diag(torch.sqrt(ev.clamp(min=1e-30))) @ V.T
            Fp = torch.linalg.solve(Fe, Fa[e])
            Rp, Sp = _polar_decompose(Fp)
            xi = cents[e]
            all_phi.append(phi_centers[ci] + PHI_HALFWIDTH * xi[0])
            all_theta.append(math.pi * xi[1])
            all_logRp.append(_log_rotation(Rp).numpy())
            all_logSp.append(_log_spd(Sp).numpy())

    all_phi = np.array(all_phi); all_theta = np.array(all_theta)
    all_logRp = np.array(all_logRp); all_logSp = np.array(all_logSp)

    # Periodic wrapping + interpolation
    pr = np.concatenate([all_phi, all_phi + 2*math.pi, all_phi - 2*math.pi])
    tr = np.concatenate([all_theta]*3)
    lRr = np.concatenate([all_logRp]*3); lSr = np.concatenate([all_logSp]*3)
    src = np.column_stack([pr, tr]); tgt = np.column_stack([phi_s, theta_s])

    print("Interpolating Lie-algebra fields...")
    logRs = np.zeros((n_s, 3, 3)); logSs = np.zeros((n_s, 3, 3))
    for i in range(3):
        for j in range(3):
            logRs[:, i, j] = griddata(src, lRr[:, i, j], tgt, method="linear", fill_value=0)
            logSs[:, i, j] = griddata(src, lSr[:, i, j], tgt, method="linear", fill_value=0)

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
