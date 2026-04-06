#!/usr/bin/env python3
"""
Figure: CrackTipDecoder singularity absorption mechanism (Section 9).

Shows how the radial power-law mapping r = R * t^p absorbs the 1/sqrt(r)
stress singularity, making the Williams displacement field smooth in
reference coordinates.

Layout: 2x2 panels
  (a) Reference domain [-1,1]^2 with uniform grid
  (b) Physical domain: mapped grid with mesh concentration near crack tip
  (c) Williams displacement u_y ~ sqrt(r) in physical space — singular
  (d) Same displacement in reference space — smooth, linear in xi
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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

# ── Parameters ───────────────────────────────────────────────────────────
R = 1.0          # support radius
p = 2.0          # power exponent
scale = 0.5      # in-plane scale (smaller than R for better aspect)
n_cells = 10     # grid cells per axis
t_min = 0.01     # clamp to avoid singular Jacobian

# ── Grid in reference space ──────────────────────────────────────────────
xi_nodes = np.linspace(-1, 1, n_cells + 1)
XI0, XI2 = np.meshgrid(xi_nodes, xi_nodes)

# ── Map to physical space (2D slice: xi_0 -> tangent, xi_2 -> radial) ───
X_phys = scale * XI0
t_arr = np.clip((XI2 + 1.0) / 2.0, t_min, 1.0)
Y_phys = R * t_arr ** p

# ── Fine grids for color plots ──────────────────────────────────────────
n_fine = 300
xi0_fine = np.linspace(-1, 1, n_fine)
xi2_fine = np.linspace(-1, 1, n_fine)
XI0f, XI2f = np.meshgrid(xi0_fine, xi2_fine)

X_phys_f = scale * XI0f
t_fine = np.clip((XI2f + 1.0) / 2.0, t_min, 1.0)
Y_phys_f = R * t_fine ** p

r_field = Y_phys_f
u_field = np.sqrt(np.maximum(r_field, 0.0))

# ── 1D profiles for insets ──────────────────────────────────────────────
xi2_line = np.linspace(-1, 1, 200)
t_line = np.clip((xi2_line + 1.0) / 2.0, t_min, 1.0)
r_line = R * t_line ** p
u_phys_line = np.sqrt(np.maximum(r_line, 0.0))  # sqrt(r) in physical
# In reference space, r_phys is the y-coordinate, u ~ sqrt(r_phys)

# ── Figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.75))

# Use gridspec for fine control
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

grid_color = "0.4"
grid_lw = 0.35

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (a) Reference domain — uniform grid                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[0, 0])
# Draw grid lines
for i in range(n_cells + 1):
    ax.plot(xi_nodes, np.full_like(xi_nodes, xi_nodes[i]),
            color=grid_color, lw=grid_lw, zorder=1)
    ax.plot(np.full_like(xi_nodes, xi_nodes[i]), xi_nodes,
            color=grid_color, lw=grid_lw, zorder=1)
# Domain boundary
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', lw=0.8, zorder=2)
# Highlight "tip" row at xi_2 = -1
ax.plot(xi_nodes, np.full_like(xi_nodes, -1.0),
        color=PUB_COLORS[0], lw=1.2, zorder=3)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_xlabel(r"$\xi_0$ (tangent)")
ax.set_ylabel(r"$\xi_2$ (radial)")
ax.set_aspect("equal")
ax.text(-0.22, 1.06, "(a)", transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="bottom")
ax.set_title("Reference domain", fontsize=9, pad=6)
# Uniform spacing label
ax.annotate("Uniform\nspacing",
            xy=(0.55, -0.5), fontsize=7, color="0.3", ha="center",
            style="italic")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (b) Physical domain — concentrated mesh near crack tip             ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[0, 1])
# Draw mapped grid lines
for i in range(n_cells + 1):
    ax.plot(X_phys[i, :], Y_phys[i, :], color=grid_color, lw=grid_lw, zorder=1)
    ax.plot(X_phys[:, i], Y_phys[:, i], color=grid_color, lw=grid_lw, zorder=1)

# Highlight innermost ring (maps from xi_2 = -1 row)
ax.plot(X_phys[0, :], Y_phys[0, :],
        color=PUB_COLORS[0], lw=1.2, zorder=3)

# Crack line
ax.plot([-scale * 1.3, 0], [0, 0], color="k", lw=2.5,
        solid_capstyle="butt", zorder=3)
# Crack tip marker
ax.plot(0, 0, "*", color=PUB_COLORS[1], ms=14, mec="k", mew=0.7, zorder=5)

# Domain boundary
bnd_xi0 = np.concatenate([xi_nodes, np.ones(n_cells + 1),
                          xi_nodes[::-1], -np.ones(n_cells + 1)])
bnd_xi2 = np.concatenate([-np.ones(n_cells + 1), xi_nodes,
                          np.ones(n_cells + 1), xi_nodes[::-1]])
bnd_t = np.clip((bnd_xi2 + 1.0) / 2.0, t_min, 1.0)
bnd_x = scale * bnd_xi0
bnd_y = R * bnd_t ** p
ax.plot(bnd_x, bnd_y, "k-", lw=0.8, zorder=2)

ax.set_xlim(-0.65, 0.65)
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel(r"$x$ (tangent)")
ax.set_ylabel(r"$y = r$ (radial)")
ax.text(-0.22, 1.06, "(b)", transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="bottom")
ax.set_title(r"Physical domain:  $r = R \cdot t^2$", fontsize=9, pad=6)

# Annotation: mesh concentration
ax.annotate("100:1 grading",
            xy=(0.02, 0.005), xytext=(0.25, 0.25),
            fontsize=7, color=PUB_COLORS[1], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PUB_COLORS[1], lw=1.0),
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# ── Mapping arrow between (a) and (b) ──
fig.text(0.495, 0.85, r"$\varphi$", ha="center", fontsize=12,
         fontweight="bold", transform=fig.transFigure,
         path_effects=[pe.withStroke(linewidth=3, foreground="white")])
from matplotlib.patches import FancyArrowPatch
arrow = FancyArrowPatch((0.47, 0.82), (0.53, 0.82),
                        transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=14,
                        lw=1.3, color="k")
fig.patches.append(arrow)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (c) Williams field in physical space — singular                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[1, 0])
im = ax.pcolormesh(X_phys_f, Y_phys_f, u_field,
                   cmap="viridis", shading="gouraud", rasterized=True)
# Crack line
ax.plot([-scale * 1.3, 0], [0, 0], color="w", lw=2.5,
        solid_capstyle="butt", zorder=3)
ax.plot(0, 0, "*", color=PUB_COLORS[1], ms=11, mec="w", mew=0.5, zorder=5)

ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y = r$")
ax.text(-0.22, 1.06, "(c)", transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="bottom")
ax.set_title(r"$u_y \propto \sqrt{r}$ — singular", fontsize=9, pad=6)
add_colorbar(fig, ax, im, label=r"$u_y$", shrink=0.85, pad=0.03)

# Inset: 1D profile showing sqrt(r)
ax_in = ax.inset_axes([0.55, 0.1, 0.40, 0.42])
ax_in.plot(r_line, u_phys_line, color=PUB_COLORS[0], lw=1.2)
ax_in.set_xlabel(r"$r$", fontsize=6, labelpad=1)
ax_in.set_ylabel(r"$u_y$", fontsize=6, labelpad=1)
ax_in.tick_params(labelsize=5, length=2)
ax_in.set_xlim(0, R)
ax_in.set_ylim(0, 1.05)
ax_in.annotate(r"$\sqrt{r}$", xy=(0.15, np.sqrt(0.15)),
               fontsize=7, color=PUB_COLORS[1], fontweight="bold")
ax_in.patch.set_alpha(0.85)
for spine in ax_in.spines.values():
    spine.set_linewidth(0.5)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  (d) Same field in reference space — smooth, linear                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
ax = fig.add_subplot(gs[1, 1])
im2 = ax.pcolormesh(XI0f, XI2f, u_field,
                    cmap="viridis", shading="gouraud", rasterized=True)
# Tip location in reference space
ax.plot(0, -1, "*", color=PUB_COLORS[1], ms=11, mec="w", mew=0.5, zorder=5)

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel(r"$\xi_0$")
ax.set_ylabel(r"$\xi_2$")
ax.set_aspect("equal")
ax.text(-0.22, 1.06, "(d)", transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="bottom")
ax.set_title(r"Same field in $\xi$-space — smooth", fontsize=9, pad=6)
add_colorbar(fig, ax, im2, label=r"$u_y$", shrink=0.85, pad=0.03)

# Inset: 1D profile showing linear in xi
ax_in2 = ax.inset_axes([0.55, 0.1, 0.40, 0.42])
u_ref_line = np.sqrt(R) * np.clip((xi2_line + 1) / 2, t_min, 1.0)
ax_in2.plot(xi2_line, u_ref_line, color=PUB_COLORS[0], lw=1.2)
ax_in2.set_xlabel(r"$\xi_2$", fontsize=6, labelpad=1)
ax_in2.set_ylabel(r"$u_y$", fontsize=6, labelpad=1)
ax_in2.tick_params(labelsize=5, length=2)
ax_in2.set_xlim(-1, 1)
ax_in2.set_ylim(0, 1.05)
ax_in2.annotate("Linear!", xy=(0.0, 0.5),
               fontsize=7, color=PUB_COLORS[2], fontweight="bold")
ax_in2.patch.set_alpha(0.85)
for spine in ax_in2.spines.values():
    spine.set_linewidth(0.5)

# ── Save ─────────────────────────────────────────────────────────────────
for ext, kw in [(".pdf", {}), (".png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, "theory_fig_crack_tip_decoder" + ext)
    fig.savefig(path, bbox_inches="tight", **kw)
    print("Saved:", path)

plt.close(fig)
