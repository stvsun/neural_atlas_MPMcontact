from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon, Circle
import numpy as np

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure


REPO_ROOT = Path('/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry')
MANUSCRIPT_DIR = REPO_ROOT / 'manuscript'
OUT_DIR = MANUSCRIPT_DIR / 'figures_cmame_core' / 'method'

BLUE = '#4C78A8'
GREEN = '#59A14F'
ORANGE = '#F28E2B'
RED = '#C44E52'
INK = '#222222'
MUTED = '#6B7280'
LIGHT = '#F5F7FA'


def add_round_box(ax, xy, w, h, text, fc='white', ec=INK, lw=1.0, fs=8.0, radius=0.04):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle=f'round,pad=0.015,rounding_size={radius}',
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=fs, color=INK)
    return patch


def add_arrow(ax, start, end, text=None, color=INK, lw=1.3, ms=12, text_offset=(0, 0.03), ls='-'):
    arr = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=ms, linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(arr)
    if text:
        mx = 0.5 * (start[0] + end[0]) + text_offset[0]
        my = 0.5 * (start[1] + end[1]) + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=7.7, color=color)
    return arr


def style_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('white')


def atlas_figure():
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.0))
    style_ax(ax)

    ax.text(0.07, 0.93, 'Reference charts', fontsize=8.8, fontweight='bold', color=INK)
    ax.text(0.57, 0.93, 'Physical atlas (2D slice through 3D volume)', fontsize=8.8, fontweight='bold', color=INK)

    ref_boxes = [
        ((0.08, 0.63), BLUE, r'$\widehat{\Omega}_1$'),
        ((0.08, 0.40), GREEN, r'$\widehat{\Omega}_2$'),
        ((0.08, 0.17), ORANGE, r'$\widehat{\Omega}_3$'),
    ]
    centers = []
    for (x, y), c, label in ref_boxes:
        add_round_box(ax, (x, y), 0.16, 0.14, label, fc=c + '20' if False else 'white', ec=c, lw=1.4, fs=9.0)
        inner = FancyBboxPatch((x + 0.012, y + 0.012), 0.136, 0.116, boxstyle='round,pad=0.01,rounding_size=0.03',
                               facecolor=c, edgecolor=c, linewidth=0.0, alpha=0.14)
        ax.add_patch(inner)
        centers.append((x + 0.16, y + 0.07))

    body = np.array([
        [0.56, 0.79], [0.69, 0.90], [0.87, 0.84], [0.93, 0.68], [0.90, 0.50],
        [0.84, 0.24], [0.69, 0.12], [0.56, 0.17], [0.49, 0.33], [0.50, 0.60]
    ])
    ax.add_patch(Polygon(body, closed=True, facecolor='none', edgecolor=INK, linewidth=1.6, joinstyle='round'))
    ax.text(0.78, 0.90, r'$\Omega$', fontsize=9.5, fontweight='bold')

    atlas_patches = [
        (0.66, 0.67, 0.18, 0.14, BLUE, r'$\Omega_1$'),
        (0.74, 0.50, 0.24, 0.17, GREEN, r'$\Omega_2$'),
        (0.74, 0.30, 0.21, 0.15, ORANGE, r'$\Omega_3$'),
    ]
    patch_centers = []
    for cx, cy, w, h, c, label in atlas_patches:
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=c, edgecolor=c, linewidth=1.4, alpha=0.18))
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor='none', edgecolor=c, linewidth=1.2))
        ax.text(cx - 0.045, cy + h / 2 + 0.03, label, fontsize=8.4, color=c, fontweight='bold')
        patch_centers.append((cx - w / 2, cy))

    add_arrow(ax, centers[0], patch_centers[0], text=r'$\varphi_1$', color=BLUE, text_offset=(0.0, 0.035))
    add_arrow(ax, centers[1], patch_centers[1], text=r'$\varphi_2$', color=GREEN, text_offset=(0.0, 0.035))
    add_arrow(ax, centers[2], patch_centers[2], text=r'$\varphi_3$', color=ORANGE, text_offset=(0.0, -0.035))

    ax.text(0.725, 0.605, r'$\Omega_{12}$', fontsize=7.6, color=INK,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.8))
    ax.text(0.765, 0.395, r'$\Omega_{23}$', fontsize=7.6, color=INK,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.8))

    add_round_box(
        ax, (0.55, 0.02), 0.36, 0.09,
        r'$u_h(x)=\sum_i \omega_i(x)\,\widehat{u}_i(\pi_i(x))$',
        fc=LIGHT, ec='#D1D5DB', lw=0.9, fs=8.1, radius=0.03
    )
    ax.text(0.73, 0.12, 'Global blend from local chart fields', fontsize=7.2, color=MUTED, ha='center')

    save_figure(fig, OUT_DIR / 'atlas_slice_pub.png', OUT_DIR / 'atlas_slice_pub.pdf')
    plt.close(fig)


def mapped_operator_figure():
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 2.95))
    style_ax(ax)

    ax.text(0.08, 0.91, 'Reference patch', fontsize=8.8, fontweight='bold')
    add_round_box(ax, (0.07, 0.34), 0.23, 0.38, r'$\widehat{\Omega}_i$', fc='white', ec=BLUE, lw=1.4, fs=10.0)
    ax.add_patch(FancyBboxPatch((0.085, 0.355), 0.20, 0.35, boxstyle='round,pad=0.01,rounding_size=0.03',
                                facecolor=BLUE, edgecolor='none', alpha=0.10))
    ax.add_patch(Circle((0.16, 0.48), 0.008, color=INK))
    ax.text(0.145, 0.44, r'$\zeta$', fontsize=8.0)
    add_arrow(ax, (0.22, 0.66), (0.22, 0.79), color=INK, lw=1.1, ms=10)
    ax.text(0.235, 0.735, r'$\widehat{n}_i$', fontsize=7.9)

    add_arrow(ax, (0.32, 0.53), (0.48, 0.53), text=r'$\varphi_i,\ \mathbf{J}_i$', color=INK, lw=1.4, text_offset=(0, 0.04))

    ax.text(0.57, 0.91, 'Physical patch', fontsize=8.8, fontweight='bold')
    poly = np.array([[0.57, 0.34], [0.65, 0.77], [0.86, 0.80], [0.93, 0.59], [0.89, 0.33], [0.67, 0.28]])
    ax.add_patch(Polygon(poly, closed=True, facecolor='none', edgecolor=INK, linewidth=1.6, joinstyle='round'))
    ax.add_patch(Polygon(poly, closed=True, facecolor=GREEN, edgecolor='none', alpha=0.10))
    ax.text(0.74, 0.84, r'$\Omega_i$', fontsize=9.5, fontweight='bold')
    ax.add_patch(Circle((0.72, 0.50), 0.008, color=INK))
    ax.text(0.685, 0.46, r'$x=\varphi_i(\zeta)$', fontsize=8.0)
    add_arrow(ax, (0.82, 0.71), (0.87, 0.85), color=INK, lw=1.1, ms=10)
    ax.text(0.875, 0.80, r'$n_i$', fontsize=7.9)

    # formula cards
    labels = [
        ('Gradient pullback', r'$\nabla_x u=\nabla_\zeta \widehat{u}_i\,\mathbf{J}_i^{-1}$'),
        ('Piola flux', r'$\widehat{\mathcal{P}}_i=j_i\,\widehat{\mathcal{G}}_i\,\mathbf{J}_i^{-T}$'),
        ('Normal map', r'$n_i=\mathbf{J}_i^{-T}\widehat{n}_i/\|\mathbf{J}_i^{-T}\widehat{n}_i\|$'),
    ]
    xs = [0.09, 0.38, 0.67]
    for x, (title, formula) in zip(xs, labels):
        add_round_box(ax, (x, 0.05), 0.23, 0.15, formula, fc=LIGHT, ec='#D1D5DB', lw=0.9, fs=8.0, radius=0.025)
        ax.text(x + 0.115, 0.215, title, fontsize=7.2, color=MUTED, ha='center')

    save_figure(fig, OUT_DIR / 'mapped_operator_pub.png', OUT_DIR / 'mapped_operator_pub.pdf')
    plt.close(fig)


def schwarz_figure():
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.35),
        gridspec_kw={'width_ratios': [1.02, 1.18]}
    )
    ax0, ax1 = axes
    style_ax(ax0)
    style_ax(ax1)

    # Panel labels
    ax0.text(0.02, 0.96, '(a)', fontsize=8.3, fontweight='bold', color=INK, va='top')
    ax1.text(0.02, 0.96, '(b)', fontsize=8.3, fontweight='bold', color=INK, va='top')

    # ------------------------------------------------------------------
    # (a) Overlapping chart geometry and transmission picture
    # ------------------------------------------------------------------
    ax0.text(0.50, 0.92, 'Overlapping chart images', ha='center', fontsize=8.7, fontweight='bold', color=INK)

    body = np.array([
        [0.10, 0.24], [0.18, 0.78], [0.50, 0.90], [0.86, 0.74], [0.91, 0.40], [0.73, 0.17], [0.36, 0.12]
    ])
    ax0.add_patch(Polygon(body, closed=True, facecolor='none', edgecolor=INK, linewidth=1.5, joinstyle='round'))
    ax0.text(0.79, 0.80, r'$\Omega$', fontsize=9.3, fontweight='bold', color=INK)

    e1 = Ellipse((0.40, 0.52), 0.42, 0.48, facecolor=BLUE, edgecolor=BLUE, alpha=0.18, linewidth=1.2)
    e2 = Ellipse((0.60, 0.52), 0.42, 0.48, facecolor=GREEN, edgecolor=GREEN, alpha=0.18, linewidth=1.2)
    ax0.add_patch(e1)
    ax0.add_patch(e2)
    ax0.add_patch(Ellipse((0.40, 0.52), 0.42, 0.48, facecolor='none', edgecolor=BLUE, linewidth=1.2))
    ax0.add_patch(Ellipse((0.60, 0.52), 0.42, 0.48, facecolor='none', edgecolor=GREEN, linewidth=1.2))

    ax0.text(0.26, 0.77, r'$\Omega_i$', fontsize=8.5, color=BLUE, fontweight='bold')
    ax0.text(0.68, 0.77, r'$\Omega_j$', fontsize=8.5, color=GREEN, fontweight='bold')
    ax0.text(0.50, 0.54, r'$\Omega_{ij}$', fontsize=8.0, color=INK,
             bbox=dict(facecolor='white', edgecolor='none', pad=0.4))

    ax0.plot([0.50, 0.50], [0.25, 0.79], linestyle='--', color=MUTED, linewidth=1.0)
    ax0.text(0.47, 0.72, r'$\Gamma_{ij}$', fontsize=7.6, color=BLUE, rotation=90, ha='center', va='center')
    ax0.text(0.53, 0.34, r'$\Gamma_{ji}$', fontsize=7.6, color=GREEN, rotation=90, ha='center', va='center')

    add_round_box(ax0, (0.11, 0.05), 0.30, 0.11, r'active chart $i$', fc='#EFF6FF', ec=BLUE, lw=1.0, fs=7.5, radius=0.02)
    add_round_box(ax0, (0.59, 0.05), 0.30, 0.11, r'frozen neighbor $j$', fc='#F0FDF4', ec=GREEN, lw=1.0, fs=7.5, radius=0.02)
    add_arrow(ax0, (0.66, 0.23), (0.57, 0.31), text='neighbor trace', color=GREEN, lw=1.1, ms=10, text_offset=(0.04, 0.05))
    add_arrow(ax0, (0.35, 0.30), (0.44, 0.24), text='interface loss', color=BLUE, lw=1.1, ms=10, text_offset=(-0.04, -0.05))

    # ------------------------------------------------------------------
    # (b) Multiplicative sweep as implemented in the rabbit solver
    # ------------------------------------------------------------------
    ax1.text(0.50, 0.92, 'One multiplicative sweep', ha='center', fontsize=8.7, fontweight='bold', color=INK)

    stage_x = [0.05, 0.28, 0.52, 0.76]
    stage_w = [0.17, 0.19, 0.18, 0.17]
    stage_titles = ['1. Snapshot', '2. Local updates', '3. Global check', '4. Checkpoint']
    fills = ['#EFF6FF', '#F0FDF4', '#FEF3F2', '#FFF7ED']
    edges = [BLUE, GREEN, RED, ORANGE]

    for x, w, title, fc, ec in zip(stage_x, stage_w, stage_titles, fills, edges):
        add_round_box(ax1, (x, 0.20), w, 0.60, '', fc=fc, ec=ec, lw=1.2, radius=0.03)
        ax1.text(x + w / 2, 0.76, title, ha='center', va='center', fontsize=7.7, fontweight='bold')

    for i in range(3):
        add_arrow(ax1, (stage_x[i] + stage_w[i], 0.50), (stage_x[i+1] - 0.015, 0.50), color=MUTED, lw=1.2, ms=10)

    ax1.text(0.135, 0.62, r'save $\Theta^{(k)}$', ha='center', fontsize=7.2)
    ax1.text(0.135, 0.47, 'rollback state', ha='center', fontsize=6.5, color=MUTED)
    ax1.text(0.135, 0.31, 'refresh pools', ha='center', fontsize=6.5, color=MUTED)

    # update charts miniature
    for cx, c, alpha in [(0.335, BLUE, 0.12), (0.395, GREEN, 0.16), (0.455, ORANGE, 0.12)]:
        ax1.add_patch(Ellipse((cx, 0.53), 0.08, 0.13, facecolor=c, edgecolor=c, alpha=alpha, linewidth=1.0))
        ax1.add_patch(Ellipse((cx, 0.53), 0.08, 0.13, facecolor='none', edgecolor=c, linewidth=1.0))
    ax1.add_patch(FancyBboxPatch((0.365, 0.44), 0.08, 0.18, boxstyle='round,pad=0.01,rounding_size=0.02',
                                 facecolor='none', edgecolor=RED, linewidth=1.6))
    ax1.text(0.375, 0.67, 'update one chart', ha='center', fontsize=6.3, color=MUTED)
    ax1.text(0.375, 0.62, 'or one color group', ha='center', fontsize=6.3, color=MUTED)
    ax1.text(0.375, 0.30, 'neighbors frozen', ha='center', fontsize=6.5, color=MUTED)

    # check sweep
    add_round_box(ax1, (0.555, 0.55), 0.11, 0.10, 'accept', fc='white', ec=GREEN, lw=1.0, fs=7.0, radius=0.02)
    add_round_box(ax1, (0.555, 0.34), 0.11, 0.10, 'rollback', fc='white', ec=RED, lw=1.0, fs=7.0, radius=0.02)
    add_arrow(ax1, (0.61, 0.50), (0.61, 0.55), color=GREEN, lw=1.0, ms=9)
    add_arrow(ax1, (0.61, 0.48), (0.61, 0.44), color=RED, lw=1.0, ms=9)
    add_round_box(ax1, (0.555, 0.21), 0.11, 0.08, 'PDE\ninterface\nrel-$L^2$', fc='white', ec='#D1D5DB', lw=0.8, fs=6.3, radius=0.02)

    add_round_box(ax1, (0.785, 0.58), 0.13, 0.07, 'blend field', fc='white', ec=INK, lw=0.9, fs=7.0, radius=0.02)
    add_round_box(ax1, (0.785, 0.45), 0.13, 0.07, 'score sweep', fc='white', ec=INK, lw=0.9, fs=7.0, radius=0.02)
    add_round_box(ax1, (0.785, 0.32), 0.13, 0.07, 'continue', fc='white', ec=INK, lw=0.9, fs=7.0, radius=0.02)
    ax1.text(0.50, 0.10, 'multiplicative on true overlaps; parallel only within non-overlapping color groups',
             ha='center', fontsize=6.5, color=MUTED)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.06, wspace=0.08)
    save_figure(fig, OUT_DIR / 'schwarz_schematic_pub.png', OUT_DIR / 'schwarz_schematic_pub.pdf')
    plt.close(fig)


def main():
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas_figure()
    mapped_operator_figure()
    schwarz_figure()


if __name__ == '__main__':
    main()
