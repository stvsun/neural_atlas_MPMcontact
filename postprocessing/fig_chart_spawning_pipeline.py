#!/usr/bin/env python3
"""
Figure: Topology-driven chart spawning pipeline (Section 12).

Shows the 4-step pipeline: Detect -> Localize -> Spawn -> Mesh.

Layout: 2x2 panels
  (a) Persistence diagram showing new H1 pair
  (b) SDF contours + feature center + normal arrow
  (c) Chart seeds s+/s- with orthonormal frames
  (d) CrackTipDecoder concentrated mesh near crack tip
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import (
    set_pub_style, PUB_COLORS, DOUBLE_COL_W, add_colorbar,
)

set_pub_style(fontsize=9, usetex=False)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plate geometry ───────────────────────────────────────────────────────
W = 1.0     # plate half-width
H = 4.0     # plate full height
a = 0.5     # crack length
delta = 0.03  # crack slit half-opening
tip_x = -W + a  # crack tip x-coordinate = -0.5
tip_y = 0.0

# ── Analytical SDF for the cracked plate (2D slice at z=0) ──────────────
nx, ny = 250, 250
x1 = np.linspace(-1.4, 1.4, nx)
x2 = np.linspace(-2.3, 2.3, ny)
X1, X2 = np.meshgrid(x1, x2)
coords = np.stack([X1.ravel(), X2.ravel(), np.zeros(nx * ny)], axis=1)

# Inline SDF computation (avoid import issues)
dx_plate = np.abs(coords[:, 0]) - W
dy_plate = np.abs(coords[:, 1]) - H / 2.0
dz_plate = np.abs(coords[:, 2]) - 0.25  # T/2
out_plate = np.sqrt(np.maximum(dx_plate, 0)**2 + np.maximum(dy_plate, 0)**2
                    + np.maximum(dz_plate, 0)**2)
in_plate = np.minimum(np.maximum(np.maximum(dx_plate, dy_plate), dz_plate), 0)
sdf_plate_vals = out_plate + in_plate

x1_local = coords[:, 0] - (-W + a / 2.0)
x2_local = coords[:, 1]
dx_crack = np.abs(x1_local) - a / 2.0
dy_crack = np.abs(x2_local) - delta
out_crack = np.sqrt(np.maximum(dx_crack, 0)**2 + np.maximum(dy_crack, 0)**2)
in_crack = np.minimum(np.maximum(dx_crack, dy_crack), 0)
sdf_crack_vals = out_crack + in_crack

sdf_vals = np.maximum(sdf_plate_vals, -sdf_crack_vals)
SDF = sdf_vals.reshape(ny, nx)

# ── Figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.72))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.40)

step_titles = ["(a) Detect", "(b) Localize", "(c) Spawn", "(d) Mesh"]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (a) Persistence diagram — new H1 pair detected                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[0, 0])

# Diagonal line b = d
diag = np.linspace(-1.2, 0.2, 100)
ax.plot(diag, diag, "k--", lw=0.6, alpha=0.5, zorder=1)

# Shaded region below diagonal (invalid)
ax.fill_between(diag, diag, -1.5, color="0.92", zorder=0)

# Old H0 pairs (noise near diagonal)
old_b = np.array([-1.0, -0.90, -0.72, -0.55])
old_d = np.array([0.05, -0.85, -0.68, -0.50])
ax.scatter(old_b[:1], old_d[:1], c=PUB_COLORS[0], s=40, edgecolors="k",
           linewidths=0.5, zorder=4, marker="o", label=r"$H_0$ (existing)")
ax.scatter(old_b[1:], old_d[1:], c="0.7", s=20, edgecolors="0.4",
           linewidths=0.3, zorder=3, marker="o", label="Noise")

# New H1 pair (the crack!) — large star marker
new_b, new_d = -0.30, -0.04
ax.plot(new_b, new_d, "*", color=PUB_COLORS[1], ms=16, mec="k", mew=0.7,
        zorder=5, label=r"$H_1$ (new crack!)")

# Annotation arrow pointing to the new pair
ax.annotate("New crack\ndetected!",
            xy=(new_b, new_d), xytext=(-0.65, 0.1),
            fontsize=7, fontweight="bold", color=PUB_COLORS[1],
            ha="center",
            arrowprops=dict(arrowstyle="->", color=PUB_COLORS[1],
                            lw=1.2, connectionstyle="arc3,rad=0.2"))

# Lifetime annotation
ax.annotate("", xy=(new_b, new_d), xytext=(new_b, new_b),
            arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
ax.text(new_b - 0.08, (new_b + new_d) / 2, r"$\ell$", fontsize=7,
        color="0.3", ha="right", va="center")

ax.set_xlabel("Birth $b$")
ax.set_ylabel("Death $d$")
ax.set_xlim(-1.15, 0.15)
ax.set_ylim(-1.15, 0.2)
ax.set_aspect("equal")
ax.legend(fontsize=6, loc="lower right", framealpha=0.9,
          handletextpad=0.3, borderpad=0.3)
ax.set_title("(a) Detect", fontsize=10, fontweight="bold", pad=6)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (b) SDF contours + feature localization                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[0, 1])

# SDF contour fill (interior only)
levels_fill = np.linspace(-1.0, 0.05, 20)
cf = ax.contourf(X1, X2, SDF, levels=levels_fill, cmap="Blues_r",
                 extend="both")

# Domain boundary (zero contour)
ax.contour(X1, X2, SDF, levels=[0], colors="k", linewidths=1.0)

# Crack line
ax.plot([-W, tip_x], [0, 0], color="k", lw=2.5, solid_capstyle="butt",
        zorder=4)
ax.plot(tip_x, tip_y, "*", color=PUB_COLORS[1], ms=12, mec="k", mew=0.6,
        zorder=5)

# SDF birth-value contour (where H1 feature localizes)
# Use a value that exists in the SDF range
sdf_min, sdf_max = SDF.min(), SDF.max()
birth_value = sdf_min * 0.3  # ~30% of the way from min to 0
if sdf_min < birth_value < sdf_max:
    ax.contour(X1, X2, SDF, levels=[birth_value], colors=PUB_COLORS[1],
               linewidths=1.0, linestyles="--")
    ax.text(0.72, 0.62, r"$s = b$", fontsize=7, color=PUB_COLORS[1],
            transform=ax.transAxes)

# Feature center (red dot) and normal arrow
center = np.array([tip_x, tip_y])
normal = np.array([0.0, 1.0])  # crack opening direction
ax.plot(center[0], center[1], "o", color=PUB_COLORS[1], ms=8, mec="k",
        mew=0.6, zorder=6)
ax.annotate("", xy=(center[0], center[1] + 0.6),
            xytext=(center[0], center[1] + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=PUB_COLORS[1],
                            lw=2.0, mutation_scale=18))
ax.text(center[0] + 0.1, center[1] + 0.35, r"$\mathbf{n}$", fontsize=9,
        color=PUB_COLORS[1], fontweight="bold")

# Labels
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.5, 1.5)
ax.set_title("(b) Localize", fontsize=10, fontweight="bold", pad=6)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (c) Chart seeds + orthonormal frames                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[1, 0])

# Light SDF background
ax.contourf(X1, X2, SDF, levels=levels_fill, cmap="Blues_r",
            extend="both", alpha=0.4)
ax.contour(X1, X2, SDF, levels=[0], colors="0.5", linewidths=0.8)

# Crack line
ax.plot([-W, tip_x], [0, 0], color="k", lw=2.5, solid_capstyle="butt",
        zorder=4)

# Chart radius
r_chart = 0.5

# Seed positions
s_plus = center + 0.5 * r_chart * normal
s_minus = center - 0.5 * r_chart * normal

# Draw chart support circles
theta_circ = np.linspace(0, 2 * np.pi, 100)
for seed, color, label in [(s_plus, PUB_COLORS[0], r"$\mathbf{s}_+$"),
                            (s_minus, PUB_COLORS[2], r"$\mathbf{s}_-$")]:
    circ_x = seed[0] + r_chart * np.cos(theta_circ)
    circ_y = seed[1] + r_chart * np.sin(theta_circ)
    ax.fill(circ_x, circ_y, color=color, alpha=0.15, zorder=2)
    ax.plot(circ_x, circ_y, color=color, lw=1.0, ls="--", alpha=0.7,
            zorder=3)
    ax.plot(seed[0], seed[1], "o", color=color, ms=8, mec="k", mew=0.5,
            zorder=6)
    # Label
    offset_x = 0.12
    ax.text(seed[0] + offset_x, seed[1] + 0.08, label, fontsize=8,
            color=color, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Draw orthonormal frames at each seed
arrow_len = 0.3
# Frame for s+: [t1, t2, n] = [along crack, out-of-plane, opening]
# 2D: t1 = (1,0), n = (0,1)
for seed, col in [(s_plus, PUB_COLORS[0]), (s_minus, PUB_COLORS[2])]:
    # t1 direction (along crack)
    ax.annotate("", xy=(seed[0] + arrow_len, seed[1]),
                xytext=(seed[0], seed[1]),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5,
                                mutation_scale=12))
    # n direction (opening)
    n_sign = 1.0 if seed[1] > 0 else -1.0
    ax.annotate("", xy=(seed[0], seed[1] + n_sign * arrow_len),
                xytext=(seed[0], seed[1]),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.5,
                                mutation_scale=12))

# Frame legend
ax.text(0.03, 0.03, r"$\mathbf{t}_1$", fontsize=7, color="red",
        transform=ax.transAxes, fontweight="bold")
ax.text(0.13, 0.03, r"$\mathbf{n}$", fontsize=7, color="blue",
        transform=ax.transAxes, fontweight="bold")

# Crack tip
ax.plot(tip_x, tip_y, "*", color=PUB_COLORS[1], ms=12, mec="k", mew=0.6,
        zorder=7)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_xlim(-1.35, 0.5)
ax.set_ylim(-1.0, 1.0)
ax.set_title("(c) Spawn", fontsize=10, fontweight="bold", pad=6)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (d) CrackTipDecoder concentrated mesh                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[1, 1])

# Light SDF background
ax.contourf(X1, X2, SDF, levels=levels_fill, cmap="Blues_r",
            extend="both", alpha=0.3)
ax.contour(X1, X2, SDF, levels=[0], colors="0.5", linewidths=0.8)

# Crack line
ax.plot([-W, tip_x], [0, 0], color="k", lw=2.5, solid_capstyle="butt",
        zorder=3)

# CrackTipDecoder mesh on the + side
R_chart = r_chart
p_pow = 2.0
n_mesh = 10
in_plane_scale = R_chart * 0.6
xi_n = np.linspace(-1, 1, n_mesh + 1)
XI_t, XI_r = np.meshgrid(xi_n, xi_n)

# Map to physical space (centered at s_plus)
X_mesh = s_plus[0] + in_plane_scale * XI_t
t_mesh = np.clip((XI_r + 1.0) / 2.0, 0.01, 1.0)
Y_mesh = s_plus[1] + R_chart * t_mesh ** p_pow * 1.0  # expand upward

# Draw mesh lines
mesh_color = PUB_COLORS[0]
for i in range(n_mesh + 1):
    ax.plot(X_mesh[i, :], Y_mesh[i, :], color=mesh_color, lw=0.5,
            alpha=0.8, zorder=4)
    ax.plot(X_mesh[:, i], Y_mesh[:, i], color=mesh_color, lw=0.5,
            alpha=0.8, zorder=4)

# Mesh boundary
bnd_xi_t = np.concatenate([xi_n, np.ones(n_mesh + 1),
                           xi_n[::-1], -np.ones(n_mesh + 1)])
bnd_xi_r = np.concatenate([-np.ones(n_mesh + 1), xi_n,
                           np.ones(n_mesh + 1), xi_n[::-1]])
bnd_t_v = np.clip((bnd_xi_r + 1.0) / 2.0, 0.01, 1.0)
bnd_mx = s_plus[0] + in_plane_scale * bnd_xi_t
bnd_my = s_plus[1] + R_chart * bnd_t_v ** p_pow
ax.plot(bnd_mx, bnd_my, color=mesh_color, lw=1.2, zorder=5)

# Similarly for minus side (mirrored)
Y_mesh_m = s_minus[1] - R_chart * t_mesh ** p_pow
X_mesh_m = s_minus[0] + in_plane_scale * XI_t
mesh_color_m = PUB_COLORS[2]
for i in range(n_mesh + 1):
    ax.plot(X_mesh_m[i, :], Y_mesh_m[i, :], color=mesh_color_m, lw=0.5,
            alpha=0.8, zorder=4)
    ax.plot(X_mesh_m[:, i], Y_mesh_m[:, i], color=mesh_color_m, lw=0.5,
            alpha=0.8, zorder=4)

bnd_my_m = s_minus[1] - R_chart * bnd_t_v ** p_pow
ax.plot(s_minus[0] + in_plane_scale * bnd_xi_t, bnd_my_m,
        color=mesh_color_m, lw=1.2, zorder=5)

# Crack tip
ax.plot(tip_x, tip_y, "*", color=PUB_COLORS[1], ms=14, mec="k", mew=0.7,
        zorder=8)

# Annotation
ax.annotate("Concentrated\nnear tip", xy=(tip_x + 0.05, 0.08),
            xytext=(0.35, 0.55), fontsize=7, color=PUB_COLORS[1],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PUB_COLORS[1], lw=1.0),
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Schwarz overlap label
ax.text(0.55, 0.12, "Schwarz\noverlap", fontsize=6, color="0.4",
        transform=ax.transAxes, ha="center", style="italic")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_xlim(-1.35, 0.5)
ax.set_ylim(-1.0, 1.0)
ax.set_title("(d) Mesh", fontsize=10, fontweight="bold", pad=6)

# ── Pipeline arrows between panels ──────────────────────────────────────
# These arrows connect the steps visually
arrow_kw = dict(arrowstyle="-|>", mutation_scale=16, lw=1.8,
                color="0.35")

# (a) -> (b): horizontal arrow in top row
arrow1 = FancyArrowPatch((0.47, 0.78), (0.53, 0.78),
                         transform=fig.transFigure, **arrow_kw)
fig.patches.append(arrow1)

# (b) -> (c): vertical arrow on right side going down-left
arrow2 = FancyArrowPatch((0.38, 0.50), (0.38, 0.44),
                         transform=fig.transFigure, **arrow_kw)
fig.patches.append(arrow2)

# (c) -> (d): horizontal arrow in bottom row
arrow3 = FancyArrowPatch((0.47, 0.27), (0.53, 0.27),
                         transform=fig.transFigure, **arrow_kw)
fig.patches.append(arrow3)

# ── Save ─────────────────────────────────────────────────────────────────
for ext, kw in [(".pdf", {}), (".png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, "theory_fig_spawning_pipeline" + ext)
    fig.savefig(path, bbox_inches="tight", **kw)
    print("Saved:", path)

plt.close(fig)
