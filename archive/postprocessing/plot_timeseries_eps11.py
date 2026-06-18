#!/usr/bin/env python3
"""Plot epsilon_11 at multiple time steps from the VTU time series.

Reads the VTU files and creates a 2x4 panel figure showing the
strain field evolution on the plate mid-plane.

Usage:
    python postprocessing/plot_timeseries_eps11.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyvista as pv

pv.OFF_SCREEN = True

from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=8, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
SERIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "runs", "biaxial_timeseries")
R = 5.0


def main():
    steps = [0, 10, 15, 19, 20, 21, 23, 25]
    titles = [
        "Step 0\n$S$=0 MPa",
        "Step 10\n$S$=13.5 MPa",
        "Step 15\n$S$=20.2 MPa",
        "Step 19\n$S$=25.6 MPa",
        "Step 20 (fracture)\n$S$=27.0 MPa",
        "Step 21 (post)\ncrack nucleated",
        "Step 23 (post)\nrelaxing",
        "Step 25 (post)\nrelaxing",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.55))

    # Global color scale
    eps_clim = 5e-4

    for idx, (step, title) in enumerate(zip(steps, titles)):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        mesh = pv.read(os.path.join(SERIES_DIR, f"biaxial_{step:04d}.vtu"))
        pts = mesh.points
        eps_11 = mesh.point_data["epsilon_11"]
        phase = mesh.point_data["phase_field"]

        x1 = pts[:, 0]
        x2 = pts[:, 1]

        # Scatter plot colored by epsilon_11
        sc = ax.scatter(
            x1, x2, c=np.clip(eps_11, 0, eps_clim),
            s=1.5, cmap="YlOrRd", vmin=0, vmax=eps_clim,
            rasterized=True,
        )

        # Overlay crack (phase < 0.5) in black
        crack_mask = phase < 0.5
        if crack_mask.any():
            ax.scatter(x1[crack_mask], x2[crack_mask], c="black", s=2.0, zorder=5)

        # Plate boundary
        ax.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.6))

        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7, pad=2)

        if row == 1:
            ax.set_xlabel("$x_1$ (mm)", fontsize=6)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("$x_2$ (mm)", fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, eps_clim))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("$\\varepsilon_{11}$", fontsize=9)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Biaxial tension: $\\varepsilon_{11}$ evolution through fracture nucleation\n"
        "(soda-lime glass, $\\sigma_{bs}$ = 27 MPa)",
        fontsize=9, y=1.0,
    )

    out = os.path.join(OUT_DIR, "biaxial_timeseries_eps11.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    out_pdf = os.path.join(OUT_DIR, "biaxial_timeseries_eps11.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
