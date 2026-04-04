#!/usr/bin/env python3
"""Generate 3D PyVista visualizations of benchmark geometries and results.

Produces PNG images suitable for README.md:
  1. Topology benchmark geometries (ball, torus, shell) with Betti annotations
  2. Cracked plate SDF isosurface with crack highlighted
  3. Williams stress field on a disk around the crack tip

Usage:
    python postprocessing/visualize_pyvista.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"
pv.global_theme.font.size = 14

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def _sdf_isosurface(sdf_fn, kwargs, bbox, resolution=80, level=0.0):
    """Create a PyVista mesh from an SDF zero-isosurface."""
    lin = [np.linspace(bbox[i], bbox[i + 3], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(*lin, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vals = sdf_fn(coords, **kwargs)

    grid = pv.RectilinearGrid(lin[0], lin[1], lin[2])
    grid["sdf"] = vals.ravel(order="F")
    return grid.contour(isosurfaces=[level], scalars="sdf")


# =====================================================================
# Figure 1: Topology benchmark geometries
# =====================================================================
def figure_topology_geometries():
    from atlas.topo.filtration import sdf_ball, sdf_solid_torus, sdf_thick_spherical_shell

    geometries = [
        {
            "name": "Ball",
            "sdf": sdf_ball, "kwargs": {"radius": 0.8},
            "bbox": [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2],
            "betti": r"β₀=1, β₁=0, β₂=0",
            "m_min": 1, "color": "#4C9BE8",
        },
        {
            "name": "Solid Torus",
            "sdf": sdf_solid_torus, "kwargs": {"R": 1.0, "r": 0.35},
            "bbox": [-1.8, -1.8, -0.8, 1.8, 1.8, 0.8],
            "betti": r"β₀=1, β₁=1, β₂=0",
            "m_min": 2, "color": "#E8854C",
        },
        {
            "name": "Spherical Shell",
            "sdf": sdf_thick_spherical_shell, "kwargs": {"R_inner": 0.5, "R_outer": 1.0},
            "bbox": [-1.3, -1.3, -1.3, 1.3, 1.3, 1.3],
            "betti": r"β₀=1, β₁=0, β₂=1",
            "m_min": 2, "color": "#5CB85C",
        },
    ]

    p = pv.Plotter(shape=(1, 3), window_size=[1800, 600], off_screen=True)

    for i, g in enumerate(geometries):
        p.subplot(0, i)
        mesh = _sdf_isosurface(g["sdf"], g["kwargs"], g["bbox"], resolution=100)

        if g["name"] == "Spherical Shell":
            # Clip to show interior
            mesh = mesh.clip(normal="y", origin=[0, 0, 0])

        p.add_mesh(
            mesh, color=g["color"], opacity=0.85,
            smooth_shading=True, specular=0.3,
        )
        p.add_text(
            f"{g['name']}\n{g['betti']}\nM_min = {g['m_min']}",
            position="upper_left", font_size=10, color="black",
        )
        p.camera_position = "iso"
        p.camera.zoom(1.3)

    out = os.path.join(OUT_DIR, "topology_geometries.png")
    p.screenshot(out, transparent_background=False, scale=2)
    p.close()
    print(f"Saved: {out}")


# =====================================================================
# Figure 2: Cracked plate with SDF coloring
# =====================================================================
def figure_cracked_plate():
    from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate

    W, H, T = 1.0, 4.0, 0.5
    a = 0.5

    bbox = [-1.3, -2.3, -0.4, 1.3, 2.3, 0.4]
    resolution = 120

    lin = [np.linspace(bbox[i], bbox[i + 3], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(*lin, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vals = sdf_cracked_plate(coords, a=a, W=W, H=H, T=T, delta=0.03)

    grid = pv.RectilinearGrid(lin[0], lin[1], lin[2])
    grid["sdf"] = vals.ravel(order="F")

    # Extract isosurface
    plate = grid.contour(isosurfaces=[0.0], scalars="sdf")

    p = pv.Plotter(window_size=[1000, 800], off_screen=True)

    # Show plate surface colored by distance to crack tip
    crack_tip = np.array([-W + a, 0.0, 0.0])
    if plate.n_points > 0:
        p.add_mesh(
            plate, color="#A0C4E8", opacity=0.85,
            smooth_shading=True, specular=0.3,
        )

    # Add crack line
    crack_line = pv.Line([-W, 0, 0], [-W + a, 0, 0])
    p.add_mesh(crack_line, color="red", line_width=5)

    # Add crack tip marker
    tip_sphere = pv.Sphere(radius=0.03, center=crack_tip)
    p.add_mesh(tip_sphere, color="red")

    p.add_text(
        f"Mode-I Edge Crack\na/W = {a/W:.1f}",
        position="upper_left", font_size=12, color="black",
    )

    p.camera_position = [(3.5, 2.5, 2.0), (0, 0, 0), (0, 1, 0)]
    p.camera.zoom(1.1)

    out = os.path.join(OUT_DIR, "cracked_plate_3d.png")
    p.screenshot(out, transparent_background=False, scale=2)
    p.close()
    print(f"Saved: {out}")


# =====================================================================
# Figure 3: Williams stress field on disk
# =====================================================================
def figure_williams_stress():
    from benchmarks.fracture.lefm_reference import williams_stress

    K_I = 1.0

    # Create a disk mesh around the crack tip
    nr, ntheta = 60, 120
    r_vals = np.linspace(0.02, 1.0, nr)
    theta_vals = np.linspace(-math.pi, math.pi, ntheta, endpoint=False)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.zeros_like(X)

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    sigma_xx, sigma_yy, sigma_xy = williams_stress(R.ravel(), Theta.ravel(), K_I)

    # Build structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    grid["σ_yy"] = np.clip(sigma_yy, -3, 3)

    # Warp by sigma_yy for 3D effect
    warped = grid.copy()
    warped.points[:, 2] = np.clip(sigma_yy, -3, 3) * 0.15

    p = pv.Plotter(window_size=[1000, 800], off_screen=True)

    # Threshold into tension/compression regions to avoid cmap issues
    warped["sigma_yy"] = np.clip(sigma_yy, -3, 3)
    tension = warped.threshold(value=0.0, scalars="sigma_yy")
    compression = warped.threshold(value=0.0, scalars="sigma_yy", invert=True)

    if tension.n_points > 0:
        p.add_mesh(tension, color="#D44040", opacity=0.9, smooth_shading=True)
    if compression.n_points > 0:
        p.add_mesh(compression, color="#4060D4", opacity=0.9, smooth_shading=True)

    # Crack line
    crack = pv.Line([-1.0, 0, 0], [0, 0, 0])
    p.add_mesh(crack, color="black", line_width=4)

    p.add_text(
        "Williams σ_yy near crack tip\n(warped by stress magnitude)",
        position="upper_left", font_size=11, color="black",
    )

    p.camera_position = [(1.5, -1.5, 2.0), (0, 0, 0), (0, 0, 1)]

    out = os.path.join(OUT_DIR, "williams_stress_3d.png")
    p.screenshot(out, transparent_background=False, scale=2)
    p.close()
    print(f"Saved: {out}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("Generating PyVista 3D visualizations...")
    figure_topology_geometries()
    figure_cracked_plate()
    figure_williams_stress()
    print("Done.")


if __name__ == "__main__":
    main()
