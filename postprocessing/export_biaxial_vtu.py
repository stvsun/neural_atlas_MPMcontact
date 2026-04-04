#!/usr/bin/env python3
"""Export biaxial tension benchmark results as VTU files for ParaView.

Generates:
1. Intact plate with uniform biaxial strain field
2. Cracked plate with near-tip strain concentration
3. Phase field (intact vs fractured)
4. SDF field on the plate volume
5. Cracked plate isosurface mesh

Usage:
    python postprocessing/export_biaxial_vtu.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocessing.utils import write_vtu_points, write_vtu_rectilinear_grid
from benchmarks.fracture.biaxial_tension import sdf_circular_plate, sdf_cracked_circular_plate
from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
from benchmarks.fracture.lefm_reference import williams_displacement, williams_stress

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "biaxial_vtu")
os.makedirs(OUT_DIR, exist_ok=True)

# Material and geometry
E = 70e3      # MPa (soda-lime glass)
nu = 0.22
sigma_bs = 27.0  # MPa
R = 5.0       # mm
W = R
H = 10.0
T = 0.5
a = R * 0.8   # crack half-length for cracked state
eps_uniform = sigma_bs * (1 - nu) / E
K_I = sigma_bs * math.sqrt(math.pi * a) * 1.12
crack_tip = np.array([a, 0.0, 0.0])  # crack runs along x-axis through center


def export_sdf_volume():
    """Export SDF fields as point clouds (VTU)."""
    nx, ny, nz = 60, 60, 10
    x = np.linspace(-R * 1.2, R * 1.2, nx)
    y = np.linspace(-R * 1.2, R * 1.2, ny)
    z = np.linspace(-T * 0.8, T * 0.8, nz)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    sdf_intact = sdf_circular_plate(coords, R=R, L=T)
    sdf_cracked = sdf_cracked_circular_plate(coords, R=R, L=T, crack_angle=0.0, delta=0.03)

    # Filter to inside or near-surface
    near = sdf_intact < 0.3
    write_vtu_points(
        os.path.join(OUT_DIR, "sdf_intact_volume.vtu"),
        coords[near],
        {"SDF": sdf_intact[near]},
    )

    write_vtu_points(
        os.path.join(OUT_DIR, "sdf_cracked_volume.vtu"),
        coords[near],
        {
            "SDF_intact": sdf_intact[near],
            "SDF_cracked": sdf_cracked[near],
        },
    )
    print(f"  SDF volumes: {near.sum()} points (of {len(coords)})")


def export_strain_fields():
    """Export strain and stress fields as point clouds on the mid-plane."""
    nr, ntheta = 60, 120
    r_vals = np.linspace(0.05, R * 0.9, nr)
    theta_vals = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
    RR, TT = np.meshgrid(r_vals, theta_vals)
    r_flat = RR.ravel()
    theta_flat = TT.ravel()

    x1 = r_flat * np.cos(theta_flat)
    x2 = r_flat * np.sin(theta_flat)
    x3 = np.zeros_like(x1)
    points = np.column_stack([x1, x2, x3])

    # Filter to inside the plate
    inside = np.sqrt(x1**2 + x2**2) <= R

    # --- Intact: uniform strain ---
    eps_11_intact = np.full(len(x1), eps_uniform)
    eps_22_intact = np.full(len(x1), eps_uniform)
    eps_11_intact[~inside] = 0.0
    eps_22_intact[~inside] = 0.0

    write_vtu_points(
        os.path.join(OUT_DIR, "strain_intact.vtu"),
        points[inside],
        {
            "epsilon_11": eps_11_intact[inside],
            "epsilon_22": eps_22_intact[inside],
            "stress_biaxial": np.full(inside.sum(), sigma_bs * eps_uniform / (sigma_bs * (1 - nu) / E)),
        },
    )

    # --- Cracked: near-tip fields ---
    dx = x1 - crack_tip[0]
    dy = x2 - crack_tip[1]
    r_tip = np.sqrt(dx**2 + dy**2)
    theta_tip = np.arctan2(dy, dx)

    # Second crack tip (symmetric)
    dx2 = x1 - (-crack_tip[0])
    dy2 = x2 - (-crack_tip[1])
    r_tip2 = np.sqrt(dx2**2 + dy2**2)
    theta_tip2 = np.arctan2(dy2, dx2)

    # Williams stress at both tips
    sxx1, syy1, sxy1 = williams_stress(r_tip, theta_tip, K_I)
    sxx2, syy2, sxy2 = williams_stress(r_tip2, theta_tip2, K_I)

    sigma_xx = sxx1 + sxx2
    sigma_yy = syy1 + syy2
    sigma_xy = sxy1 + sxy2

    # Von Mises
    von_mises = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)

    # Williams displacement at both tips
    ux1, uy1 = williams_displacement(r_tip, theta_tip, K_I, E, nu, plane_strain=True)
    ux2, uy2 = williams_displacement(r_tip2, theta_tip2, K_I, E, nu, plane_strain=True)
    u_x = ux1 + ux2
    u_y = uy1 + uy2

    # Clip extreme values for visualization
    clip_val = 200.0
    sigma_xx = np.clip(sigma_xx, -clip_val, clip_val)
    sigma_yy = np.clip(sigma_yy, -clip_val, clip_val)
    sigma_xy = np.clip(sigma_xy, -clip_val, clip_val)
    von_mises = np.clip(von_mises, 0, clip_val)

    # Phase field: v=0 on crack, v=1 elsewhere
    on_crack = (np.abs(x2) < 0.03) & (np.abs(x1) < a)
    phase_field = np.where(on_crack, 0.0, 1.0)

    write_vtu_points(
        os.path.join(OUT_DIR, "stress_cracked.vtu"),
        points[inside],
        {
            "sigma_xx": sigma_xx[inside],
            "sigma_yy": sigma_yy[inside],
            "sigma_xy": sigma_xy[inside],
            "von_Mises": von_mises[inside],
            "displacement": np.column_stack([u_x, u_y, np.zeros_like(u_x)])[inside],
            "phase_field": phase_field[inside],
        },
    )
    print(f"  Stress fields: {inside.sum()} points")


def export_crack_surface():
    """Export the crack surface as a point cloud with displacement vectors."""
    # Points along the crack
    n_along = 100
    n_across = 20
    x1_crack = np.linspace(-a * 0.95, a * 0.95, n_along)
    x2_offsets = np.linspace(-0.5, 0.5, n_across)

    X1, X2 = np.meshgrid(x1_crack, x2_offsets)
    points = np.column_stack([X1.ravel(), X2.ravel(), np.zeros(X1.size)])

    # Distance from each crack tip
    r1 = np.sqrt((X1.ravel() - a)**2 + X2.ravel()**2)
    theta1 = np.arctan2(X2.ravel(), X1.ravel() - a)
    r2 = np.sqrt((X1.ravel() + a)**2 + X2.ravel()**2)
    theta2 = np.arctan2(X2.ravel(), X1.ravel() + a)

    ux1, uy1 = williams_displacement(r1, theta1, K_I, E, nu, True)
    ux2, uy2 = williams_displacement(r2, theta2, K_I, E, nu, True)

    sxx1, syy1, sxy1 = williams_stress(r1, theta1, K_I)
    sxx2, syy2, sxy2 = williams_stress(r2, theta2, K_I)

    displacement = np.column_stack([ux1 + ux2, uy1 + uy2, np.zeros(X1.size)])
    von_mises = np.sqrt(
        (sxx1 + sxx2)**2 - (sxx1 + sxx2) * (syy1 + syy2)
        + (syy1 + syy2)**2 + 3 * (sxy1 + sxy2)**2
    )

    write_vtu_points(
        os.path.join(OUT_DIR, "crack_zone.vtu"),
        points,
        {
            "displacement": displacement,
            "von_Mises": np.clip(von_mises, 0, 200),
            "distance_to_crack": np.abs(X2.ravel()),
        },
    )
    print(f"  Crack zone: {X1.size} points")


def export_plate_surface():
    """Export the plate boundary as a 2D point cloud with SDF values."""
    n = 150
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)

    # Lateral surface points
    x1 = R * np.cos(theta)
    x2 = R * np.sin(theta)
    x3 = np.zeros(n)
    points = np.column_stack([x1, x2, x3])

    # SDF on these points (should be ~0 for intact, variable for cracked)
    coords_3d = np.column_stack([x1, x2, x3])
    sdf_intact = sdf_circular_plate(coords_3d, R=R, L=T)
    sdf_cracked = sdf_cracked_circular_plate(coords_3d, R=R, L=T, crack_angle=0.0, delta=0.03)

    write_vtu_points(
        os.path.join(OUT_DIR, "plate_boundary.vtu"),
        points,
        {
            "SDF_intact": sdf_intact,
            "SDF_cracked": sdf_cracked,
            "angle": theta,
        },
    )
    print(f"  Plate boundary: {n} points")


def main():
    print(f"Exporting VTU files to {OUT_DIR}/")
    print(f"  Geometry: R={R}, H={H}, T={T}, a={a}")
    print(f"  Material: E={E}, nu={nu}, sigma_bs={sigma_bs}")
    print(f"  K_I = {K_I:.2f} MPa*sqrt(mm)")
    print()

    export_sdf_volume()
    export_strain_fields()
    export_crack_surface()
    export_plate_surface()

    print(f"\nAll VTU files saved to {OUT_DIR}/")
    print("Open in ParaView:")
    print(f"  paraview {OUT_DIR}/sdf_intact_volume.vtr")
    print(f"  paraview {OUT_DIR}/stress_cracked.vtu")
    print(f"  paraview {OUT_DIR}/crack_zone.vtu")


if __name__ == "__main__":
    main()
