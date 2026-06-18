#!/usr/bin/env python3
"""
Figure: Minimum chart count via Lusternik-Schnirelmann category (Section 2.3).

Shows 4 domains with different topologies, their Betti numbers, and the
minimum number of coordinate charts needed to cover each.

Layout: 1 x 4 panels (3D views)
  (a) Solid ball:   beta=(1,0,0), M_min=1
  (b) Solid torus:  beta=(1,1,0), M_min=2
  (c) Thick shell:  beta=(1,0,1), M_min=2
  (d) Cracked plate: beta=(2,0,0), M_min=1/piece
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    set_pub_style, PUB_COLORS, DOUBLE_COL_W,
)

set_pub_style(fontsize=9, usetex=False)
plt.rcParams["axes.grid"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────
C1 = PUB_COLORS[0]   # blue (chart 1)
C2 = PUB_COLORS[1]   # vermillion (chart 2)
C3 = PUB_COLORS[2]   # green
GRAY = "0.75"


def _sphere_surface(cx=0, cy=0, cz=0, r=1.0, n=40):
    """Return X, Y, Z meshes for a sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


def _torus_surface(R=1.0, r_tube=0.35, n=60, u_range=(0, 2*np.pi)):
    """Return X, Y, Z meshes for a torus."""
    u = np.linspace(u_range[0], u_range[1], n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = (R + r_tube * np.cos(V)) * np.cos(U)
    Y = (R + r_tube * np.cos(V)) * np.sin(U)
    Z = r_tube * np.sin(V)
    return X, Y, Z


def _half_sphere(cx=0, cy=0, cz=0, r=1.0, n=30, upper=True):
    """Return half-sphere surface."""
    u = np.linspace(0, 2 * np.pi, n)
    if upper:
        v = np.linspace(0, np.pi / 2, n // 2)
    else:
        v = np.linspace(np.pi / 2, np.pi, n // 2)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


def _box_faces(cx, cy, cz, hx, hy, hz):
    """Return list of 6 face vertex arrays for a box."""
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    # 6 faces, each a list of 4 corners
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bottom
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # top
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # front
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # back
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # left
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # right
    ]
    return verts


def _setup_3d_ax(ax):
    """Clean 3D axis appearance."""
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.dist = 7


# ── Figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.36))

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (a) Solid ball — beta=(1,0,0), M_min=1                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(1, 4, 1, projection="3d")
X, Y, Z = _sphere_surface(r=1.0, n=30)
ax.plot_surface(X, Y, Z, color=C1, alpha=0.35, edgecolor=C1,
                linewidth=0.15, antialiased=True, shade=True)
ax.view_init(elev=22, azim=-55)
_setup_3d_ax(ax)
ax.set_title("(a) Ball", fontsize=9, fontweight="bold", pad=-2)
ax.text2D(0.50, -0.02,
          r"$\beta = (1,0,0)$" + "\n" + r"$M_{\min} = 1$" + "\n1 chart",
          transform=ax.transAxes, ha="center", fontsize=7, linespacing=1.4)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (b) Solid torus — beta=(1,1,0), M_min=2                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(1, 4, 2, projection="3d")

# Two overlapping chart regions on the torus
# Chart 1: covers u in [0, 4pi/3]
X1t, Y1t, Z1t = _torus_surface(u_range=(0, 4*np.pi/3), n=50)
ax.plot_surface(X1t, Y1t, Z1t, color=C1, alpha=0.30, edgecolor=C1,
                linewidth=0.1, antialiased=True, shade=True)

# Chart 2: covers u in [2pi/3, 2pi]
X2t, Y2t, Z2t = _torus_surface(u_range=(2*np.pi/3, 2*np.pi), n=50)
ax.plot_surface(X2t, Y2t, Z2t, color=C2, alpha=0.30, edgecolor=C2,
                linewidth=0.1, antialiased=True, shade=True)

ax.view_init(elev=28, azim=-50)
_setup_3d_ax(ax)
ax.set_title("(b) Solid torus", fontsize=9, fontweight="bold", pad=-2)
ax.text2D(0.50, -0.02,
          r"$\beta = (1,1,0)$" + "\n" + r"$M_{\min} = 2$" + "\n2 charts",
          transform=ax.transAxes, ha="center", fontsize=7, linespacing=1.4)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (c) Thick spherical shell — beta=(1,0,1), M_min=2                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(1, 4, 3, projection="3d")

r_outer = 1.0
r_inner = 0.6

# Outer shell — upper hemisphere (chart 1)
Xo, Yo, Zo = _half_sphere(r=r_outer, n=30, upper=True)
ax.plot_surface(Xo, Yo, Zo, color=C1, alpha=0.25, edgecolor=C1,
                linewidth=0.1, antialiased=True, shade=True)

# Outer shell — lower hemisphere (chart 2)
Xo2, Yo2, Zo2 = _half_sphere(r=r_outer, n=30, upper=False)
ax.plot_surface(Xo2, Yo2, Zo2, color=C2, alpha=0.25, edgecolor=C2,
                linewidth=0.1, antialiased=True, shade=True)

# Inner shell cutaway — show the void via a partial inner sphere
# Only draw the front-facing part for visibility
u = np.linspace(0, np.pi, 20)
v = np.linspace(0, np.pi, 20)  # half circle for cutaway
Xi = r_inner * np.outer(np.cos(u), np.sin(v))
Yi = r_inner * np.outer(np.sin(u), np.sin(v))
Zi = r_inner * np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(Xi, Yi, Zi, color="0.85", alpha=0.4, edgecolor="0.6",
                linewidth=0.1, antialiased=True, shade=True)

# Equator ring to show the overlap boundary
theta_ring = np.linspace(0, 2*np.pi, 80)
ax.plot(r_outer * np.cos(theta_ring), r_outer * np.sin(theta_ring),
        np.zeros_like(theta_ring), color="0.3", lw=0.8, ls="--", alpha=0.7)

ax.view_init(elev=18, azim=-55)
_setup_3d_ax(ax)
ax.set_title("(c) Thick shell", fontsize=9, fontweight="bold", pad=-2)
ax.text2D(0.50, -0.02,
          r"$\beta = (1,0,1)$" + "\n" + r"$M_{\min} = 2$" + "\n2 charts",
          transform=ax.transAxes, ha="center", fontsize=7, linespacing=1.4)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (d) Cracked plate — beta=(2,0,0), M_min=1 per piece               ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(1, 4, 4, projection="3d")

# Two rectangular pieces separated by a crack gap
gap = 0.08
hx, hy, hz = 0.3, 1.0, 0.15

# Piece 1 (above crack)
verts1 = _box_faces(0, gap/2 + hy, 0, hx, hy, hz)
pc1 = Poly3DCollection(verts1, alpha=0.35, facecolor=C1,
                        edgecolor=C1, linewidth=0.4)
ax.add_collection3d(pc1)

# Piece 2 (below crack)
verts2 = _box_faces(0, -(gap/2 + hy), 0, hx, hy, hz)
pc2 = Poly3DCollection(verts2, alpha=0.35, facecolor=C2,
                        edgecolor=C2, linewidth=0.4)
ax.add_collection3d(pc2)

# Crack gap highlight — red line between pieces
crack_z = np.array([-hz, hz])
for zc in crack_z:
    ax.plot([-hx, hx], [0, 0], [zc, zc], color=PUB_COLORS[1],
            lw=1.5, zorder=10)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-1.0, 1.0)
ax.view_init(elev=25, azim=-50)
_setup_3d_ax(ax)
ax.set_title("(d) Cracked plate", fontsize=9, fontweight="bold", pad=-2)
ax.text2D(0.50, -0.02,
          r"$\beta = (2,0,0)$" + "\n" + r"$M_{\min} = 1$/piece"
          + "\n2 charts",
          transform=ax.transAxes, ha="center", fontsize=7, linespacing=1.4)

# ── Legend strip at bottom ───────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C1, alpha=0.4, edgecolor=C1, label="Chart 1"),
    Patch(facecolor=C2, alpha=0.4, edgecolor=C2, label="Chart 2"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           fontsize=7, frameon=False, borderpad=0.1,
           bbox_to_anchor=(0.5, -0.01))

# ── Save ─────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.18, top=0.88,
                    wspace=0.05)

for ext, kw in [(".pdf", {}), (".png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, "theory_fig_min_chart_count" + ext)
    fig.savefig(path, bbox_inches="tight", **kw)
    print("Saved:", path)

plt.close(fig)
