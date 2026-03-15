from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pub_style import DOUBLE_COLUMN_WIDTH, apply_publication_style, save_figure

REPO_ROOT = Path('/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry')
MANUSCRIPT_DIR = REPO_ROOT / 'manuscript'
OUT_DIR = MANUSCRIPT_DIR / 'figures_cmame_core' / 'implementation'
PNG_PATH = OUT_DIR / 'human_ai_workflow_pub.png'
PDF_PATH = OUT_DIR / 'human_ai_workflow_pub.pdf'


def add_box(ax, xy, width, height, text, facecolor, edgecolor='#2a2a2a', fontsize=8.0, weight='normal'):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle='round,pad=0.02,rounding_size=0.04',
        linewidth=1.0, edgecolor=edgecolor, facecolor=facecolor
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)
    return patch


def add_arrow(ax, start, end, color='#4a4a4a', style='-|>', lw=1.2, connectionstyle='arc3,rad=0.0'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color, connectionstyle=connectionstyle)
    ax.add_patch(arrow)


def build_figure() -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(1.9, 5.55, 'Human-led scientific decisions', fontsize=9.0, fontweight='bold', ha='center')
    ax.text(8.0, 5.55, 'LLM implementation assistance', fontsize=9.0, fontweight='bold', ha='center')
    ax.text(5.0, 0.55, 'Validation and acceptance loop', fontsize=9.0, fontweight='bold', ha='center')

    add_box(ax, (0.6, 4.35), 2.6, 0.72, 'Problem definition\nand governing equations', '#dceeff', weight='bold')
    add_box(ax, (0.6, 3.20), 2.6, 0.72, 'Atlas design, losses,\nand quality criteria', '#dceeff')
    add_box(ax, (0.6, 2.05), 2.6, 0.72, 'Benchmark selection\nand acceptance thresholds', '#dceeff')

    add_box(ax, (6.6, 4.35), 2.8, 0.72, 'Code scaffolding,\nrefactoring, debugging', '#f5e7cf', weight='bold')
    add_box(ax, (6.6, 3.20), 2.8, 0.72, 'Figure scripts, tables,\nand artifact packaging', '#f5e7cf')
    add_box(ax, (6.6, 2.05), 2.8, 0.72, 'Candidate implementations\nand reproducibility glue', '#f5e7cf')

    add_box(ax, (3.75, 4.30), 2.5, 0.84, 'Specification handoff', '#eef2f5', fontsize=8.1)
    add_box(ax, (3.75, 3.15), 2.5, 0.84, 'Candidate code\nand plots returned', '#eef2f5', fontsize=8.1)

    add_box(ax, (2.15, 0.95), 2.25, 0.80, 'Unit tests and\ngeometry gates', '#e9f6df', weight='bold')
    add_box(ax, (4.95, 0.95), 2.25, 0.80, 'Canonical runs\nand metric checks', '#e9f6df', weight='bold')
    add_box(ax, (7.75, 0.95), 1.45, 0.80, 'Human review:\naccept or revise', '#e9f6df', weight='bold')

    add_arrow(ax, (3.2, 4.70), (3.75, 4.70))
    add_arrow(ax, (6.25, 4.70), (6.6, 4.70))
    add_arrow(ax, (6.6, 3.57), (6.25, 3.57))
    add_arrow(ax, (3.75, 3.57), (3.2, 3.57))

    add_arrow(ax, (1.9, 4.35), (1.9, 3.92))
    add_arrow(ax, (1.9, 3.20), (1.9, 2.77))
    add_arrow(ax, (8.0, 4.35), (8.0, 3.92))
    add_arrow(ax, (8.0, 3.20), (8.0, 2.77))

    add_arrow(ax, (3.0, 2.05), (3.0, 1.75), connectionstyle='arc3,rad=-0.15')
    add_arrow(ax, (7.9, 2.05), (6.8, 1.75), connectionstyle='arc3,rad=0.15')
    add_arrow(ax, (4.40, 1.35), (4.95, 1.35))
    add_arrow(ax, (7.20, 1.35), (7.75, 1.35))
    add_arrow(ax, (8.45, 0.95), (1.3, 0.95), connectionstyle='arc3,rad=-0.35')
    ax.text(5.2, 0.12, 'Rejected candidates return to the formulation or implementation stage; accepted candidates enter the benchmark registry.',
            fontsize=7.1, ha='center', va='center')

    ax.text(5.0, 5.95, 'Human-led research loop with low-level AI implementation assistance', ha='center', va='top', fontsize=9.2, fontweight='bold')

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.03)
    save_figure(fig, PNG_PATH, PDF_PATH)
    plt.close(fig)


if __name__ == '__main__':
    build_figure()
