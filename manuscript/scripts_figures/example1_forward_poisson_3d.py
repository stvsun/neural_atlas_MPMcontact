"""Generate publication-quality 3D ellipsoid Poisson benchmark figure using PyVista.

Produces a 2-panel figure:
  (a) Predicted field u_h on a half-clipped ellipsoid (showing interior field)
  (b) Absolute error |u_h - u*| on the same clipped view

Uses the same rendering style as the torus figures (three-point studio lighting,
scalar bars, smooth shading).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt

REPO_ROOT = Path("/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry")
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pinn_3d_ellipsoid_mapped_sphere as ellipsoid_mod  # noqa: E402

ELLIPSOID_CACHE = REPO_ROOT / "manuscript" / "scripts_figures" / "cache" / "example1_ellipsoid_checkpoint.pt"
OUT_DIR = REPO_ROOT / "manuscript" / "figures_cmame_core" / "example1_forward_poisson"

# Ellipsoid semi-axes
A_AXIS, B_AXIS, C_AXIS = 2.0, 1.4, 0.9


def load_model():
    """Load the cached ellipsoid PINN model."""
    device = torch.device("cpu")
    dtype = torch.float64
    ckpt = torch.load(ELLIPSOID_CACHE, map_location=device)
    model = ellipsoid_mod.PINN3D(
        width=ckpt["model_kwargs"]["width"],
        depth=ckpt["model_kwargs"]["depth"],
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["metrics"]


def build_ellipsoid_surface(model, n_theta=300, n_phi=150):
    """Build an ellipsoid surface mesh with field data using PyVista's parametric surface."""
    # Use PyVista's built-in parametric ellipsoid for a clean triangulated surface
    ellipsoid = pv.ParametricEllipsoid(A_AXIS, B_AXIS, C_AXIS)
    # Subdivide for smoothness
    ellipsoid = ellipsoid.subdivide(2)

    # Get surface points and compute reference coordinates
    pts = ellipsoid.points.copy()
    ref_xi = pts[:, 0] / A_AXIS
    ref_eta = pts[:, 1] / B_AXIS
    ref_zeta = pts[:, 2] / C_AXIS

    # Pull slightly inside to avoid exact boundary (where u=0)
    r = np.sqrt(ref_xi**2 + ref_eta**2 + ref_zeta**2)
    scale = np.where(r > 0, 0.998 / np.maximum(r, 1e-10), 1.0)
    scale = np.minimum(scale, 1.0)  # don't scale up interior points
    ref_xi_scaled = ref_xi * scale
    ref_eta_scaled = ref_eta * scale
    ref_zeta_scaled = ref_zeta * scale

    ref_pts = np.column_stack([ref_xi_scaled, ref_eta_scaled, ref_zeta_scaled])
    ref_t = torch.tensor(ref_pts, dtype=torch.float64)
    with torch.no_grad():
        pred = model(ref_t).cpu().numpy().flatten()
    exact = 1.0 - (ref_pts[:, 0] ** 2 + ref_pts[:, 1] ** 2 + ref_pts[:, 2] ** 2)
    error = np.abs(pred - exact)

    ellipsoid["u_h"] = pred
    ellipsoid["error"] = error
    return ellipsoid


def build_interior_cross_section(model, n_grid=250):
    """Build a dense 2D cross-section through the interior at y=0."""
    lin_x = np.linspace(-0.998, 0.998, n_grid)
    lin_z = np.linspace(-0.998, 0.998, n_grid)
    XI_2d, ZETA_2d = np.meshgrid(lin_x, lin_z)
    ETA_2d = np.zeros_like(XI_2d)

    # Mask: inside unit ball (xi^2 + zeta^2 <= 1, eta=0)
    r2 = XI_2d ** 2 + ZETA_2d ** 2
    mask = r2 <= 1.0

    # Only evaluate inside
    valid_xi = XI_2d[mask]
    valid_eta = ETA_2d[mask]
    valid_zeta = ZETA_2d[mask]
    ref_pts = np.column_stack([valid_xi, valid_eta, valid_zeta])
    ref_t = torch.tensor(ref_pts, dtype=torch.float64)
    with torch.no_grad():
        pred = model(ref_t).cpu().numpy().flatten()
    exact = 1.0 - (ref_pts[:, 0] ** 2 + ref_pts[:, 1] ** 2 + ref_pts[:, 2] ** 2)
    error = np.abs(pred - exact)

    # Physical coords for the cross-section
    phys_x = valid_xi * A_AXIS
    phys_y = valid_eta * B_AXIS
    phys_z = valid_zeta * C_AXIS

    points = np.column_stack([phys_x, phys_y, phys_z])
    cloud = pv.PolyData(points)
    cloud["u_h"] = pred
    cloud["error"] = error

    # Delaunay to create a surface from the cross-section
    surf = cloud.delaunay_2d()
    return surf


def render_3d_panel(mesh, scalar_name, cmap, clim, sbar_title,
                    camera_pos, scalar_bar_fmt="%.2f",
                    window_size=(1800, 1400), clip_y=True,
                    cross_section=None):
    """Render one 3D panel with studio lighting."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")
    pl.remove_all_lights()

    # Three-point studio lighting
    key = pv.Light(position=(5, -4, 6), focal_point=(0, 0, 0),
                   intensity=0.55, color="white")
    fill = pv.Light(position=(-5, 3, 3), focal_point=(0, 0, 0),
                    intensity=0.30, color=(0.95, 0.95, 1.0))
    rim = pv.Light(position=(0, 5, 2), focal_point=(0, 0, 0),
                   intensity=0.20, color=(1.0, 0.98, 0.95))
    pl.add_light(key)
    pl.add_light(fill)
    pl.add_light(rim)

    sbar_args = dict(
        title=sbar_title,
        title_font_size=32,
        label_font_size=26,
        shadow=False,
        n_labels=5,
        italic=False,
        fmt=scalar_bar_fmt,
        font_family="arial",
        color="black",
        position_x=0.72,
        position_y=0.08,
        width=0.18,
        height=0.84,
    )

    # Clip the surface to show half — offset slightly to expose cross-section
    display_mesh = mesh
    if clip_y:
        display_mesh = mesh.clip(normal=(0, 1, 0), origin=(0, 0.03, 0))

    pl.add_mesh(
        display_mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args=sbar_args,
        smooth_shading=True,
        specular=0.15,
        specular_power=20,
        ambient=0.35,
        diffuse=0.6,
        nan_opacity=0.0,
        interpolate_before_map=True,
    )

    # Add the interior cross-section if provided
    if cross_section is not None:
        pl.add_mesh(
            cross_section,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            smooth_shading=False,
            ambient=0.40,
            diffuse=0.55,
            specular=0.05,
        )

    pl.camera_position = camera_pos
    img = pl.screenshot(transparent_background=False, return_img=True)
    pl.close()
    return img


def main():
    print("Loading model...")
    model, metrics = load_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building ellipsoid surface mesh...")
    surface = build_ellipsoid_surface(model)

    print("Building interior cross-section...")
    cross_section = build_interior_cross_section(model, n_grid=300)

    # Camera: isometric elevated view from y>0 side to see the cut face
    cam_pos = [
        (4.5, 4.0, 3.0),   # eye (positive y to see the y=0 cut face)
        (0.0, 0.0, 0.0),   # focal
        (0.0, 0.0, 1.0),   # up
    ]

    # Get data ranges — must include cross-section (interior has larger values)
    cs_pred = cross_section["u_h"]
    cs_err = cross_section["error"]
    pred_max = max(float(np.max(np.abs(surface["u_h"]))),
                   float(np.max(np.abs(cs_pred))))
    err_max = max(float(np.max(surface["error"])),
                  float(np.max(cs_err)))

    # --- Panel (a): Predicted field on half-ellipsoid with interior cross-section ---
    print("Rendering panel (a): predicted field...")
    img_a = render_3d_panel(
        surface, "u_h", "coolwarm", (-pred_max, pred_max),
        sbar_title="Predicted u_h",
        camera_pos=cam_pos, clip_y=True,
        cross_section=cross_section,
        scalar_bar_fmt="%.2f",
    )

    # --- Panel (b): Absolute error on half-ellipsoid with interior cross-section ---
    print("Rendering panel (b): absolute error...")
    img_b = render_3d_panel(
        surface, "error", "OrRd", (0, err_max),
        sbar_title="Absolute error",
        camera_pos=cam_pos, clip_y=True,
        cross_section=cross_section,
        scalar_bar_fmt="%.2e",
    )

    # --- Composite 2-panel figure ---
    print("Compositing panels...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, img, label in zip(axes, [img_a, img_b],
                               ["(a) Predicted field", "(b) Absolute error"]):
        ax.imshow(img)
        ax.set_title(label, fontsize=16, fontweight="bold", pad=10,
                     fontfamily="serif")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.02)

    out_png = OUT_DIR / "example1_ellipsoid_3d_pub.png"
    out_pdf = OUT_DIR / "example1_ellipsoid_3d_pub.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved to {out_png}")
    print(f"Metrics: rel-L2 = {metrics['relative_l2_error']:.4e}, "
          f"max error = {metrics['max_error']:.4e}")


if __name__ == "__main__":
    main()
