#!/usr/bin/env python3
"""Export biaxial tension loading sequence as time-series VTU + PVD.

Generates one VTU per load step and a .pvd collection file that ParaView
reads as an animated time series. Fields evolve from uniform pre-fracture
strain through crack nucleation to post-fracture relaxation.

Output:
    runs/biaxial_timeseries/
        biaxial.pvd              — ParaView collection (open this)
        biaxial_0000.vtu         — step 0
        biaxial_0001.vtu         — step 1
        ...

Usage:
    python postprocessing/export_biaxial_timeseries.py
    paraview runs/biaxial_timeseries/biaxial.pvd
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocessing.utils import write_vtu_points
from benchmarks.fracture.biaxial_tension import (
    sdf_circular_plate, sdf_cracked_circular_plate, GLASS,
)
from benchmarks.fracture.lefm_reference import williams_displacement, williams_stress

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "biaxial_timeseries")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────
E = GLASS["E"]
nu = GLASS["nu"]
sigma_bs = GLASS["sigma_bs"]
R = 5.0
T = 0.5
n_steps = 30          # total load steps
frac_step = 20        # step at which fracture occurs

# Grid for the mid-plane point cloud
nr, ntheta = 50, 90
r_vals = np.linspace(0.05, R * 0.95, nr)
theta_vals = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
RR, TT = np.meshgrid(r_vals, theta_vals)
r_flat = RR.ravel()
theta_flat = TT.ravel()
x1 = r_flat * np.cos(theta_flat)
x2 = r_flat * np.sin(theta_flat)
x3 = np.zeros_like(x1)
points = np.column_stack([x1, x2, x3])
inside = np.sqrt(x1**2 + x2**2) <= R
pts = points[inside]
n_pts = pts.shape[0]

# Crack parameters (for post-fracture steps)
a_crack = R * 0.8     # crack half-length
crack_tip_R = np.array([a_crack, 0.0, 0.0])
crack_tip_L = np.array([-a_crack, 0.0, 0.0])


def compute_fields(step):
    """Compute all fields at a given load step."""
    # Loading parameter: fraction of sigma_bs
    if step <= frac_step:
        load_frac = step / frac_step  # 0 → 1
        cracked = False
    else:
        load_frac = 0.0  # post-fracture: stress relaxed
        cracked = True

    sigma_applied = sigma_bs * load_frac
    eps_applied = sigma_applied * (1 - nu) / E

    x1_in = pts[:, 0]
    x2_in = pts[:, 1]

    if not cracked:
        # Uniform biaxial strain
        eps_11 = np.full(n_pts, eps_applied)
        eps_22 = np.full(n_pts, eps_applied)
        sigma_xx = np.full(n_pts, sigma_applied)
        sigma_yy = np.full(n_pts, sigma_applied)
        sigma_xy = np.zeros(n_pts)
        u_x = eps_applied * x1_in
        u_y = eps_applied * x2_in
        phase = np.ones(n_pts)
    else:
        # Post-fracture: crack-tip singular fields (residual)
        # Use a small residual K_I that decays after fracture
        decay = max(0, 1.0 - (step - frac_step) * 0.15)
        K_I_res = sigma_bs * math.sqrt(math.pi * a_crack) * 1.12 * decay

        dx_R = x1_in - crack_tip_R[0]
        dy_R = x2_in - crack_tip_R[1]
        r_R = np.sqrt(dx_R**2 + dy_R**2)
        theta_R = np.arctan2(dy_R, dx_R)

        dx_L = x1_in - crack_tip_L[0]
        dy_L = x2_in - crack_tip_L[1]
        r_L = np.sqrt(dx_L**2 + dy_L**2)
        theta_L = np.arctan2(dy_L, dx_L)

        sxx_R, syy_R, sxy_R = williams_stress(r_R, theta_R, K_I_res)
        sxx_L, syy_L, sxy_L = williams_stress(r_L, theta_L, K_I_res)
        sigma_xx = np.clip(sxx_R + sxx_L, -100, 100)
        sigma_yy = np.clip(syy_R + syy_L, -100, 100)
        sigma_xy = np.clip(sxy_R + sxy_L, -100, 100)
        eps_11 = sigma_xx / E
        eps_22 = sigma_yy / E

        ux_R, uy_R = williams_displacement(r_R, theta_R, K_I_res, E, nu, True)
        ux_L, uy_L = williams_displacement(r_L, theta_L, K_I_res, E, nu, True)
        u_x = ux_R + ux_L
        u_y = uy_R + uy_L

        # Phase field: v=0 on crack, smooth transition
        crack_width = 0.15
        dist_to_crack = np.abs(x2_in)
        on_crack = (np.abs(x1_in) < a_crack) & (dist_to_crack < crack_width * 3)
        phase = np.ones(n_pts)
        phase[on_crack] = 1.0 - np.exp(-0.5 * (dist_to_crack[on_crack] / crack_width)**2)
        phase[(np.abs(x1_in) < a_crack) & (dist_to_crack < crack_width * 0.5)] = 0.0

    von_mises = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)

    return {
        "epsilon_11": eps_11,
        "epsilon_22": eps_22,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "von_Mises": von_mises,
        "displacement": np.column_stack([u_x, u_y, np.zeros(n_pts)]),
        "phase_field": phase,
        "load_fraction": np.full(n_pts, load_frac),
        "applied_stress": np.full(n_pts, sigma_applied),
    }


def write_pvd(vtu_files, times, pvd_path):
    """Write a ParaView Data (.pvd) collection file."""
    root = ET.Element("VTKFile", type="Collection", version="0.1",
                      byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    for t, f in zip(times, vtu_files):
        ET.SubElement(collection, "DataSet",
                      timestep=str(t), group="", part="0", file=f)
    tree = ET.ElementTree(root)
    tree.write(pvd_path, xml_declaration=True, encoding="utf-8")


def main():
    print(f"Exporting {n_steps} time steps to {OUT_DIR}/")
    print(f"  {n_pts} points per step, fracture at step {frac_step}")

    vtu_files = []
    times = []

    for step in range(n_steps):
        fields = compute_fields(step)

        fname = f"biaxial_{step:04d}.vtu"
        fpath = os.path.join(OUT_DIR, fname)
        write_vtu_points(fpath, pts, fields)

        vtu_files.append(fname)
        times.append(float(step))

        status = "FRAC" if step == frac_step else ("post" if step > frac_step else "load")
        sigma = fields["applied_stress"][0]
        print(f"  Step {step:3d}/{n_steps-1} | sigma={sigma:6.1f} MPa | {status} | {fname}")

    # Write PVD collection
    pvd_path = os.path.join(OUT_DIR, "biaxial.pvd")
    write_pvd(vtu_files, times, pvd_path)
    print(f"\nPVD collection: {pvd_path}")
    print(f"Open in ParaView: paraview {pvd_path}")


if __name__ == "__main__":
    main()
