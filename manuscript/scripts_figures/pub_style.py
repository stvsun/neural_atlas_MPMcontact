from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - registers matplotlib styles
import seaborn as sns


SINGLE_COLUMN_WIDTH = 3.35
DOUBLE_COLUMN_WIDTH = 6.9
EXPORT_DPI = 600


def apply_publication_style() -> None:
    plt.style.use(["science", "no-latex", "grid"])
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("white")

    mpl.rcParams.update(
        {
            "figure.dpi": EXPORT_DPI,
            "savefig.dpi": EXPORT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2a2a2a",
            "axes.labelcolor": "#1f1f1f",
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.55,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "image.cmap": "viridis",
        }
    )


def ensure_parent(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_figure(fig: mpl.figure.Figure, png_path: str | Path, pdf_path: str | Path | None = None) -> None:
    png_out = ensure_parent(png_path)
    fig.savefig(png_out, dpi=EXPORT_DPI, facecolor="white")
    if pdf_path is not None:
        pdf_out = ensure_parent(pdf_path)
        fig.savefig(pdf_out, facecolor="white")

