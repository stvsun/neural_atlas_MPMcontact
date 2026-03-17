#!/usr/bin/env python3
"""
Generate publication-quality 3D torus inverse figures for CMAME manuscript.

Uses PyVista for off-screen rendering with proper lighting and surface meshes.

Creates:
  - Figure 10: 3-panel (deformed torus + |u|, traction error, chart decomposition)
  - Figure 12: 2×2 Schwarz (traction error + per-chart mu for both modes)
  - Convergence plot (updated to canonical run)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# ── paths ──
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_VTU = os.path.join(
    REPO, "runs", "torus_inverse_mps_dense_v4",
    "torus_inverse_neohookean_atlas_boundary_full_error.vtu",
)
SCHWARZ_TRAC_VTU = os.path.join(
    REPO, "runs", "torus_schwarz_dual_accurate_traction_20260215_140800",
    "torus_inverse_neohookean_schwarz_traction_accurate_best_boundary_fields.vtu",
)
SCHWARZ_DISP_VTU = os.path.join(
    REPO, "runs", "torus_schwarz_dual_accurate_displacement_20260215_140200",
    "torus_inverse_neohookean_schwarz_displacement_accurate_best_boundary_fields.vtu",
)
OUTDIR = os.path.join(REPO, "manuscript", "figures_cmame_core", "example3_torus_inverse_original")
OUTDIR_SCHWARZ = os.path.join(REPO, "manuscript", "figures_cmame_core", "example4_torus_inverse_schwarz_dual")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(OUTDIR_SCHWARZ, exist_ok=True)

R_MAJOR = 1.0
R_MINOR = 0.35

# ═══════════════════════════════════════════════════════════════════════════
#  Publication color palettes (muted, print-friendly)
# ═══════════════════════════════════════════════════════════════════════════

# 8 muted, distinguishable chart colors (Tableau-style muted palette)
CHART_COLORS_8 = [
    "#4e79a7",  # steel blue
    "#f28e2b",  # muted orange
    "#e15759",  # muted red
    "#76b7b2",  # teal
    "#59a14f",  # muted green
    "#edc948",  # muted gold
    "#b07aa1",  # muted purple
    "#9c755f",  # muted brown
]


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_torus_polydata(n_phi: int = 300, n_theta: int = 150):
    """Create a high-quality structured torus surface as a PyVista PolyData."""
    import pyvista as pv
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")

    X = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.cos(PHI)
    Y = (R_MAJOR + R_MINOR * np.cos(THETA)) * np.sin(PHI)
    Z = R_MINOR * np.sin(THETA)

    grid = pv.StructuredGrid(X, Y, Z)
    surf = grid.extract_surface()
    return surf, PHI, THETA, X, Y, Z


def compute_phi_theta_from_xyz(coords: np.ndarray):
    """Compute (phi, theta) from 3D torus surface coordinates."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    phi = np.arctan2(y, x) % (2 * np.pi)
    rho = np.sqrt(x**2 + y**2)
    dx = rho - R_MAJOR
    theta = np.arctan2(z, dx) % (2 * np.pi)
    return phi, theta


def interpolate_to_grid(phi_pts, theta_pts, field_pts, PHI_grid, THETA_grid):
    """Interpolate scattered (phi, theta) field data onto structured grid."""
    from scipy.interpolate import griddata

    phi_pts = phi_pts % (2 * np.pi)
    theta_pts = theta_pts % (2 * np.pi)

    phi_all = np.concatenate([phi_pts, phi_pts + 2 * np.pi, phi_pts - 2 * np.pi])
    theta_all = np.concatenate([theta_pts, theta_pts + 2 * np.pi, theta_pts - 2 * np.pi])
    field_all = np.concatenate([field_pts, field_pts, field_pts])

    pts = np.column_stack([phi_all, theta_all])
    grid_vals = griddata(pts, field_all, (PHI_grid, THETA_grid), method="nearest")
    return grid_vals


def apply_torsion(PHI, THETA):
    """Apply prescribed torsion displacement, return deformed (X,Y,Z) and |u|."""
    tau = 0.085
    phi_center = np.pi / 2
    phi_halfwidth = 1.1
    z_scale = 0.75

    dphi = np.arctan2(np.sin(PHI - phi_center), np.cos(PHI - phi_center))
    window = np.exp(-0.5 * (dphi / phi_halfwidth) ** 6) * (np.abs(dphi) < 2.0 * phi_halfwidth)
    alpha = tau * window * z_scale

    dx0 = R_MINOR * np.cos(THETA)
    dz0 = R_MINOR * np.sin(THETA)
    dx_rot = np.cos(alpha) * dx0 - np.sin(alpha) * dz0
    dz_rot = np.sin(alpha) * dx0 + np.cos(alpha) * dz0

    X0 = (R_MAJOR + dx0) * np.cos(PHI)
    Y0 = (R_MAJOR + dx0) * np.sin(PHI)
    Z0 = dz0

    Xd = (R_MAJOR + dx_rot) * np.cos(PHI)
    Yd = (R_MAJOR + dx_rot) * np.sin(PHI)
    Zd = dz_rot

    u_mag = np.sqrt((Xd - X0)**2 + (Yd - Y0)**2 + (Zd - Z0)**2)
    return Xd, Yd, Zd, u_mag


def read_vtu_fields(vtu_path: str) -> dict:
    """Parse ASCII VTU and return point coordinates + fields."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(vtu_path)
    root = tree.getroot()
    data = {}
    for pts_el in root.iter("Points"):
        for da in pts_el:
            vals = np.fromstring(da.text, sep=" ", dtype=np.float64)
            data["coords"] = vals.reshape(-1, 3)
    for pd in root.iter("PointData"):
        for arr in pd:
            name = arr.attrib["Name"]
            nc = int(arr.attrib.get("NumberOfComponents", "1"))
            vals = np.fromstring(arr.text, sep=" ", dtype=np.float64)
            data[name] = vals.reshape(-1, nc) if nc > 1 else vals
    return data


def map_surf_idx(X, Y, Z, surf):
    """Build KD-tree index from structured grid to extracted surface points."""
    from scipy.spatial import cKDTree
    flat = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
    tree = cKDTree(flat)
    _, idx = tree.query(np.array(surf.points))
    return idx


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering with PyVista
# ═══════════════════════════════════════════════════════════════════════════

def render_torus_with_field(
    mesh,
    scalars: np.ndarray,
    scalar_name: str,
    cmap: str = "viridis",
    clim: tuple | None = None,
    title: str = "",
    camera_position: str | list = "iso",
    show_scalar_bar: bool = True,
    scalar_bar_title: str = "",
    log_scale: bool = False,
    n_colors: int = 256,
    window_size: tuple = (1600, 1100),
    categories: bool = False,
    annotations: dict | None = None,
    scalar_bar_fmt: str | None = None,
) -> np.ndarray:
    """Render a torus mesh colored by a scalar field, return image as numpy array."""
    import pyvista as pv
    pv.global_theme.anti_aliasing = "ssaa"

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")

    if scalar_bar_fmt is None:
        scalar_bar_fmt = "%.2e" if not categories else None

    sbar_args = dict(
        title=scalar_bar_title or scalar_name,
        title_font_size=28,
        label_font_size=22,
        shadow=False,
        n_labels=5,
        italic=False,
        fmt=scalar_bar_fmt,
        font_family="arial",
        color="black",
        position_x=0.72,
        position_y=0.10,
        width=0.16,
        height=0.80,
    )

    mesh[scalar_name] = scalars

    # Remove default lights so we have full control
    pl.remove_all_lights()

    pl.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=sbar_args,
        smooth_shading=True,
        specular=0.15,
        specular_power=20,
        ambient=0.35,
        diffuse=0.6,
        log_scale=log_scale,
        n_colors=n_colors,
        categories=categories,
        annotations=annotations or {},
    )

    # Camera: elevated front-right view showing full torus
    if camera_position == "iso":
        pl.camera_position = [
            (3.0, -2.5, 2.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    elif camera_position == "front":
        pl.camera_position = [
            (0.0, -4.0, 1.8),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    else:
        pl.camera_position = camera_position

    # Three-point studio lighting: key, fill, rim
    # Key light: main illumination from upper-front-right
    key = pv.Light(
        position=(4, -3, 5), focal_point=(0, 0, 0),
        intensity=0.55, color="white",
    )
    pl.add_light(key)
    # Fill light: softer, from the left to reduce harsh shadows
    fill = pv.Light(
        position=(-4, 2, 2), focal_point=(0, 0, 0),
        intensity=0.30, color=(0.95, 0.95, 1.0),
    )
    pl.add_light(fill)
    # Rim/back light: subtle highlight on the far edge for depth
    rim = pv.Light(
        position=(0, 4, 1), focal_point=(0, 0, 0),
        intensity=0.20, color=(1.0, 0.98, 0.95),
    )
    pl.add_light(rim)

    img = pl.screenshot(transparent_background=False, return_img=True)
    pl.close()
    return img


def build_chart_cmap():
    """Build a ListedColormap from the 8 muted chart colors."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(CHART_COLORS_8, name="chart8")


def make_figure10(global_data: dict):
    """3-panel figure: (a) deformed + |u|, (b) traction error, (c) chart decomposition."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n_phi, n_theta = 300, 150
    surf_ref, PHI, THETA, X, Y, Z = make_torus_polydata(n_phi, n_theta)

    phi_pts = global_data.get("phi")
    theta_pts = global_data.get("theta")
    if phi_pts is None:
        phi_pts, theta_pts = compute_phi_theta_from_xyz(global_data["coords"])

    # ── Panel (a): Deformed configuration colored by |u| ──
    Xd, Yd, Zd, u_mag_grid = apply_torsion(PHI, THETA)
    grid_def = pv.StructuredGrid(Xd, Yd, Zd)
    surf_def = grid_def.extract_surface()
    idx_def = map_surf_idx(Xd, Yd, Zd, surf_def)
    u_on_surf = u_mag_grid.ravel(order="F")[idx_def]

    img_a = render_torus_with_field(
        surf_def, u_on_surf, "u_mag",
        cmap="Blues",
        clim=(0, np.max(u_on_surf)),
        scalar_bar_title="|u|",
        scalar_bar_fmt="%.4f",
    )

    # ── Panel (b): Traction error on reference torus ──
    trac_err_mag = global_data["traction_error_mag"]
    trac_err_grid = interpolate_to_grid(phi_pts, theta_pts, trac_err_mag, PHI, THETA)
    idx_ref = map_surf_idx(X, Y, Z, surf_ref)
    trac_on_surf = trac_err_grid.ravel(order="F")[idx_ref]

    img_b = render_torus_with_field(
        surf_ref.copy(), trac_on_surf, "trac_error",
        cmap="OrRd",
        clim=(0, np.percentile(trac_on_surf, 99)),
        scalar_bar_title="|t - t_obs|",
    )

    # ── Panel (c): Chart decomposition ──
    chart_id = global_data["chart_id"]
    chart_grid = interpolate_to_grid(phi_pts, theta_pts, chart_id, PHI, THETA)
    chart_on_surf = np.round(chart_grid.ravel(order="F")[idx_ref]).astype(int)

    # Use the muted 8-color palette via pyvista
    chart_cmap = build_chart_cmap()

    img_c = render_torus_with_field(
        surf_ref.copy(), chart_on_surf.astype(float), "chart_id",
        cmap=chart_cmap,
        clim=(-0.5, 7.5),
        scalar_bar_title="Chart ID",
        n_colors=8,
        categories=False,
        scalar_bar_fmt="%.0f",
    )

    # ── Compose into one figure ──
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    labels = [
        r"(a) Deformed configuration, $|\mathbf{u}|$",
        r"(b) Traction error, $|\mathbf{t} - \mathbf{t}^{\mathrm{obs}}|$",
        "(c) Chart decomposition (8 charts)",
    ]
    for ax, img, lab in zip(axes, [img_a, img_b, img_c], labels):
        ax.imshow(img)
        ax.set_title(lab, fontsize=14, pad=8)
        ax.set_axis_off()

    plt.tight_layout(w_pad=0.5)
    out_png = os.path.join(OUTDIR, "example3_torus_inverse_fields_pub.png")
    out_pdf = os.path.join(OUTDIR, "example3_torus_inverse_fields_pub.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure 10 → {out_png}")


def make_figure12(schwarz_trac_data: dict, schwarz_disp_data: dict):
    """2×2 Schwarz figure: error + per-chart mu for traction and displacement modes."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_phi, n_theta = 300, 150
    surf_ref, PHI, THETA, X, Y, Z = make_torus_polydata(n_phi, n_theta)
    idx_ref = map_surf_idx(X, Y, Z, surf_ref)

    images = []
    labels = []

    for data, mode_label, mode in [
        (schwarz_trac_data, "Schwarz traction mode", "traction"),
        (schwarz_disp_data, "Schwarz displacement mode", "displacement"),
    ]:
        phi_pts, theta_pts = compute_phi_theta_from_xyz(data["coords"])

        # Error field
        if mode == "traction":
            err_field = data["traction_error_mag"]
            err_label = "|t - t_obs|"
        else:
            err_field = data["u_error_mag"]
            err_label = "|u - u_obs|"

        err_grid = interpolate_to_grid(phi_pts, theta_pts, err_field, PHI, THETA)
        err_on_surf = err_grid.ravel(order="F")[idx_ref]

        img_err = render_torus_with_field(
            surf_ref.copy(), err_on_surf, "error",
            cmap="OrRd",
            clim=(0, np.percentile(err_on_surf, 99)),
            scalar_bar_title=err_label,
        )
        images.append(img_err)
        panel = "a" if mode == "traction" else "c"
        labels.append(f"({panel}) {mode_label}: observation error")

        # Per-chart mu
        mu_field = data["mu_est"]
        mu_grid = interpolate_to_grid(phi_pts, theta_pts, mu_field, PHI, THETA)
        mu_on_surf = mu_grid.ravel(order="F")[idx_ref]

        # Show the actual per-chart variation; pad range slightly for visibility
        mu_range = np.max(mu_on_surf) - np.min(mu_on_surf)
        if mu_range < 1e-6:
            # Machine-precision consensus: use a narrow band around the mean
            mu_center = np.mean(mu_on_surf)
            mu_min_v = mu_center - 0.005
            mu_max_v = mu_center + 0.005
        else:
            mu_pad = max(mu_range * 0.15, 0.001)
            mu_min_v = np.min(mu_on_surf) - mu_pad
            mu_max_v = np.max(mu_on_surf) + mu_pad

        img_mu = render_torus_with_field(
            surf_ref.copy(), mu_on_surf, "mu_est",
            cmap="viridis",
            clim=(mu_min_v, mu_max_v),
            scalar_bar_title="mu_i",
            scalar_bar_fmt="%.4f",
        )
        images.append(img_mu)
        panel = "b" if mode == "traction" else "d"
        labels.append(f"({panel}) {mode_label}: " + r"per-chart $\hat{\mu}_i$")

    # Compose 2×2
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, img, lab in zip(axes.ravel(), images, labels):
        ax.imshow(img)
        ax.set_title(lab, fontsize=14, pad=8)
        ax.set_axis_off()

    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    out_png = os.path.join(OUTDIR_SCHWARZ, "torus_schwarz_3d_fields_pub.png")
    out_pdf = os.path.join(OUTDIR_SCHWARZ, "torus_schwarz_3d_fields_pub.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure 12 → {out_png}")


def make_convergence_figure():
    """Convergence plot from the canonical dense v4 run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_path = os.path.join(REPO, "runs", "torus_inverse_mps_dense_v4",
                             "torus_inverse_neohookean_atlas_history.json")
    with open(hist_path) as f:
        history = json.load(f)

    mu_true, K_true = 1.8, 25.0
    mu_key = "mu" if "mu" in history else "mu_guess"
    K_key = "K" if "K" in history else "K_guess"
    trac_key = "loss_trac" if "loss_trac" in history else "traction"
    epochs = np.arange(1, len(history[mu_key]) + 1)
    mu_err = np.abs(np.array(history[mu_key]) - mu_true) / mu_true * 100
    K_err = np.abs(np.array(history[K_key]) - K_true) / K_true * 100
    trac_loss = np.array(history[trac_key])

    # Publication style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)

    # Muted colors
    mu_color = "#4e79a7"
    K_color = "#e15759"

    axes[0].semilogy(epochs, mu_err, color=mu_color, linewidth=1.8,
                     label=r"$\mu$ relative error (%)")
    axes[0].semilogy(epochs, K_err, color=K_color, linewidth=1.8,
                     label=r"$K$ relative error (%)")
    axes[0].set_ylabel("Relative error (%)")
    axes[0].set_title("(a) Parameter recovery")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25, linewidth=0.5)
    axes[0].set_ylim(bottom=1e-4)

    axes[1].semilogy(epochs, np.maximum(trac_loss, 1e-16), color="#333333",
                     linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\mathcal{L}_{\mathrm{trac}}$")
    axes[1].set_title("(b) Traction mismatch")
    axes[1].grid(True, alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    out_png = os.path.join(OUTDIR, "example3_torus_inverse_convergence_pub.png")
    out_pdf = os.path.join(OUTDIR, "example3_torus_inverse_convergence_pub.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Convergence → {out_png}")


def main():
    print("Loading VTU data...")
    global_data = read_vtu_fields(GLOBAL_VTU)
    print(f"  Global: {len(global_data['coords'])} points")
    schwarz_trac_data = read_vtu_fields(SCHWARZ_TRAC_VTU)
    print(f"  Schwarz traction: {len(schwarz_trac_data['coords'])} points")
    schwarz_disp_data = read_vtu_fields(SCHWARZ_DISP_VTU)
    print(f"  Schwarz displacement: {len(schwarz_disp_data['coords'])} points")

    print("\nGenerating Figure 10 (3-panel torus fields)...")
    make_figure10(global_data)

    print("\nGenerating Figure 12 (Schwarz 3D fields)...")
    make_figure12(schwarz_trac_data, schwarz_disp_data)

    print("\nGenerating convergence figure...")
    make_convergence_figure()

    print("\nDone!")


if __name__ == "__main__":
    main()
