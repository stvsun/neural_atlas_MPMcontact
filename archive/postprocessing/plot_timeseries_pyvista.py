#!/usr/bin/env python3
"""Plot epsilon_11 at multiple time steps using PyVista.

Creates a multi-panel figure showing the strain field evolution:
pre-fracture (uniform) → fracture nucleation → post-fracture (crack-tip concentration).

Usage:
    python postprocessing/plot_timeseries_pyvista.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
SERIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "runs", "biaxial_timeseries")


def main():
    # Select key time steps to display
    steps = [0, 10, 15, 19, 20, 21, 23, 25]
    labels = [
        "Step 0\n$\\sigma$=0 MPa",
        "Step 10\n$\\sigma$=13.5 MPa",
        "Step 15\n$\\sigma$=20.2 MPa",
        "Step 19\n$\\sigma$=25.6 MPa",
        "Step 20 (FRAC)\n$\\sigma$=27.0 MPa",
        "Step 21 (post)\n$\\sigma$=0 MPa",
        "Step 23 (post)\ndecaying",
        "Step 25 (post)\ndecaying",
    ]

    n_panels = len(steps)
    cols = 4
    rows = 2

    p = pv.Plotter(
        shape=(rows, cols),
        window_size=[1600, 900],
        off_screen=True,
    )

    # Find global range for consistent color mapping
    eps_max = 0.0
    for step in steps:
        mesh = pv.read(os.path.join(SERIES_DIR, f"biaxial_{step:04d}.vtu"))
        eps = mesh.point_data["epsilon_11"]
        eps_max = max(eps_max, float(np.max(np.abs(eps))))

    # Clip for visualization (crack-tip singularity can dominate)
    eps_clim = min(eps_max, 5e-4)

    for idx, (step, label) in enumerate(zip(steps, labels)):
        row = idx // cols
        col = idx % cols
        p.subplot(row, col)

        mesh = pv.read(os.path.join(SERIES_DIR, f"biaxial_{step:04d}.vtu"))
        eps_11 = mesh.point_data["epsilon_11"].copy()
        phase = mesh.point_data["phase_field"]

        # Clip strain for visualization
        eps_clipped = np.clip(eps_11, 0, eps_clim)
        mesh.point_data["eps_11_clipped"] = eps_clipped

        # Point size based on whether it's intact or cracked
        pt_size = 3.0

        # Use solid color since this PyVista can't do custom cmaps
        if step <= 20:
            # Pre-fracture: uniform color proportional to strain
            frac = float(eps_11[0]) / eps_clim if eps_clim > 0 else 0
            r = min(int(255 * frac), 255)
            g = min(int(180 * (1 - frac)), 180)
            b = min(int(50 * (1 - frac)), 50)
            color = f"#{r:02x}{g:02x}{b:02x}"
            p.add_mesh(mesh, color=color, point_size=pt_size,
                       render_points_as_spheres=True)
        else:
            # Post-fracture: show crack via two regions
            # Intact region (phase > 0.5): strain-colored
            intact_mask = phase > 0.5
            cracked_mask = phase <= 0.5

            if np.any(intact_mask):
                intact_pts = mesh.points[intact_mask]
                intact_cloud = pv.PolyData(intact_pts)
                intact_eps = eps_clipped[intact_mask]
                # Map to color: low strain = blue, high = red
                normed = intact_eps / eps_clim if eps_clim > 0 else intact_eps
                colors = np.zeros((len(intact_pts), 4), dtype=np.uint8)
                colors[:, 0] = (normed * 230).clip(0, 255).astype(np.uint8)
                colors[:, 1] = ((1 - normed) * 100).clip(0, 255).astype(np.uint8)
                colors[:, 2] = ((1 - normed) * 180).clip(0, 255).astype(np.uint8)
                colors[:, 3] = 255
                intact_cloud["rgba"] = colors
                try:
                    p.add_mesh(intact_cloud, scalars="rgba", rgba=True,
                               point_size=pt_size, render_points_as_spheres=True)
                except Exception:
                    # Fallback if rgba doesn't work
                    p.add_mesh(intact_cloud, color="#CC8844",
                               point_size=pt_size, render_points_as_spheres=True)

            if np.any(cracked_mask):
                crack_pts = mesh.points[cracked_mask]
                crack_cloud = pv.PolyData(crack_pts)
                p.add_mesh(crack_cloud, color="black",
                           point_size=pt_size * 0.8, render_points_as_spheres=True)

        p.add_text(label, position="upper_left", font_size=8, color="black")
        p.view_xy()
        p.camera.zoom(1.4)

    out = os.path.join(OUT_DIR, "biaxial_timeseries_eps11.png")
    p.screenshot(out, transparent_background=False, scale=2)
    p.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
