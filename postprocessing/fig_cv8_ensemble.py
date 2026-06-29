#!/usr/bin/env python3
r"""CV-8 deformable-Hertz MULTI-REALIZATION spread at the headline mesh (nx=192).

The manuscript reports the CV-8 deformable-Hertz half-width and peak-pressure errors from a SINGLE
jitter realization (mesh-seed 7: a_relerr 2.75 %, p0_relerr 5.82 %) and carries an honest
"single-realization plane-strain CST" caveat.  This figure EVIDENCES that caveat by running five
independent jitter realizations of the SAME production regime (R=2.0, delta=0.02, n_load=6, graded
mesh, jitter 0.03; mesh-seed = 7/11/17/23/31; all force balances ~1e-18..1e-19) and showing the
spread of the two error metrics across the five seeds.

The numbers below are the MEASURED ensemble (taken verbatim; not re-solved here -- this is the
reporting figure for an already-run sweep):

  seed   a_relerr(%)   p0_relerr(%)
    7       2.75          5.82       <- the reported HEADLINE (the BEST a_relerr of the five)
   11       3.52          5.96
   17       4.30          6.00
   23       3.71          6.02
   31       3.09          5.90

  half-width a_relerr : mean 3.47 %, sd 0.53 %, range [2.75, 4.30] %  -> realization-DEPENDENT
  peak-pressure p0    : mean 5.94 %, sd 0.07 %, range [5.82, 6.02] %  -> realization-INDEPENDENT

HONEST READING (the only one, and what the figure is built to show):
  - p0_relerr is a TIGHT realization-independent plateau (sd 0.07 %) -- a genuine systematic floor
    (CST + finite-penalty interior bias), NOT a lucky realization.
  - a_relerr SCATTERS in a ~2.75-4.30 % band; the reported 2.75 % is the BEST case, the ensemble
    mean is ~3.5 %, still inside the stated ~3-6 % discrete contact-edge floor (the Hertz edge slope
    is infinite, so the recovered edge moves by +-one surface node ~1/nx per realization).
  The figure marks the seed-7 headline explicitly so 2.75 % is NOT read as cherry-picked: it is the
  best of five, disclosed as such, with the ensemble mean band drawn alongside.

Reproducibility:  the five-seed arrays below are READ from runs/cv8_deformable_ot/ensemble.json when
that artifact exists (regenerate it with `python3 benchmarks/contact/cv_numerical/cv8_deformable_ot.py
--mode ensemble`, which threads --mesh-seed into block_mesh and writes the JSON).  If the JSON is
absent (a fresh checkout — runs/ is gitignored) the figure falls back to the committed MEASURED values
below, which match the driver output exactly (seed 7 -> 2.75/5.82, ..., seed 31 -> 3.09/5.90).

Run:  python3 postprocessing/fig_cv8_ensemble.py
Output: figures/fig_cv8_ensemble_pub.{png,pdf}  (+ printed ensemble statistics)
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
ENS_JSON = os.path.join(_ROOT, "runs", "cv8_deformable_ot", "ensemble.json")

# ---- MEASURED ensemble (production regime R=2.0, delta=0.02, n_load=6, graded mesh, jitter=0.03,
#      nx=192; mesh-seed = 7/11/17/23/31; all force balances ~1e-18..1e-19).  EXACT numbers, used as
#      the fallback if runs/cv8_deformable_ot/ensemble.json is absent. ----
SEEDS = np.array([7, 11, 17, 23, 31])
A_RELERR = np.array([2.75, 3.52, 4.30, 3.71, 3.09])   # half-width error, percent
P0_RELERR = np.array([5.82, 5.96, 6.00, 6.02, 5.90])  # peak-pressure error, percent
HEADLINE_SEED = 7                                     # the reported single-realization headline


def _load_ensemble():
    """Prefer the auditable driver artifact runs/cv8_deformable_ot/ensemble.json; else fall back to
    the committed MEASURED arrays.  Returns (seeds, a_relerr%, p0_relerr%, headline_seed, source)."""
    global SEEDS, A_RELERR, P0_RELERR, HEADLINE_SEED
    if os.path.exists(ENS_JSON):
        with open(ENS_JSON) as fh:
            d = json.load(fh)
        seeds = np.array([int(s) for s in d["seeds"]])
        a = np.array([float(x) for x in d["a_relerr_pct"]])
        p0 = np.array([float(x) for x in d["p0_relerr_pct"]])
        head = int(d.get("headline_seed", seeds[0]))
        SEEDS, A_RELERR, P0_RELERR, HEADLINE_SEED = seeds, a, p0, head
        return seeds, a, p0, head, ENS_JSON
    return SEEDS, A_RELERR, P0_RELERR, HEADLINE_SEED, "committed fallback (ensemble.json absent)"

# ---- palette (matches plot_two_body_ot.py / the CV figures) ----
BG = "#ffffff"
INK = "#1b2430"
MUTE = "#6b7686"
GRID = "#dfe4ea"
C_A = "#0072B2"        # blue      -- half-width (the scattering metric)
C_P0 = "#D55E00"       # vermilion -- peak pressure (the robust plateau)
C_HEAD = "#CC79A7"     # reddish purple accent -- the headline seed marker

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTE, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "cm",
})


def _stats(v):
    # population sd (ddof=0): the quoted ensemble sd is 0.53% (a) / 0.07% (p0) over the five seeds
    return float(np.mean(v)), float(np.std(v, ddof=0)), float(v.min()), float(v.max())


def _strip(ax, x0, vals, color, label, jitter_seed):
    """One vertical strip: mean+-sd band, mean line, and jittered points; the headline seed ringed."""
    mean, sd, lo, hi = _stats(vals)
    # mean +- sd band
    ax.fill_between([x0 - 0.30, x0 + 0.30], mean - sd, mean + sd,
                    color=color, alpha=0.13, lw=0, zorder=1)
    ax.plot([x0 - 0.30, x0 + 0.30], [mean, mean], color=color, lw=2.2, zorder=3,
            solid_capstyle="round")
    # full range whisker (thin) so the reader sees the extent, not just +-1sd
    ax.plot([x0, x0], [lo, hi], color=color, lw=1.0, alpha=0.55, zorder=2)
    # jittered scatter of the five realizations
    rng = np.random.default_rng(jitter_seed)
    xs = x0 + rng.uniform(-0.16, 0.16, size=vals.size)
    for xi, vi, sd_i in zip(xs, vals, SEEDS):
        is_head = (sd_i == HEADLINE_SEED)
        ax.scatter([xi], [vi], s=78 if is_head else 46,
                   facecolor=("white" if is_head else color),
                   edgecolor=(C_HEAD if is_head else color),
                   linewidth=(2.0 if is_head else 1.0), zorder=6 if is_head else 5)
    return mean, sd, lo, hi


def main():
    os.makedirs(FIG, exist_ok=True)

    seeds, a_arr, p0_arr, head, source = _load_ensemble()
    print(f"=== source: {source} ===")

    a_mean, a_sd, a_lo, a_hi = _stats(A_RELERR)
    p_mean, p_sd, p_lo, p_hi = _stats(P0_RELERR)
    a_head = float(A_RELERR[SEEDS == HEADLINE_SEED][0])
    p_head = float(P0_RELERR[SEEDS == HEADLINE_SEED][0])

    print("=== CV-8 deformable-Hertz multi-realization ensemble (nx=192, 5 mesh-seeds) ===")
    print(f"  seeds            : {list(SEEDS)}")
    print(f"  a_relerr  (%)    : {list(A_RELERR)}")
    print(f"  p0_relerr (%)    : {list(P0_RELERR)}")
    print(f"  half-width a     : mean {a_mean:.2f}  sd {a_sd:.2f}  range [{a_lo:.2f}, {a_hi:.2f}]  "
          f"-> REALIZATION-DEPENDENT")
    print(f"  peak pressure p0 : mean {p_mean:.2f}  sd {p_sd:.2f}  range [{p_lo:.2f}, {p_hi:.2f}]  "
          f"-> REALIZATION-INDEPENDENT (tight floor)")
    print(f"  HEADLINE seed {HEADLINE_SEED}: a {a_head:.2f}%  p0 {p_head:.2f}%  "
          f"(seed {HEADLINE_SEED} is the BEST of the five on BOTH metrics; p0 {p_head:.2f}% is the "
          f"low end of the [{p_lo:.2f}, {p_hi:.2f}]% band, mean {p_mean:.2f}%)")

    # ================================================================================================
    # figure: two strips on a shared error axis -- a (scatters) vs p0 (tight plateau)
    # ================================================================================================
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xa, xp = 1.0, 2.0
    _strip(ax, xa, A_RELERR, C_A, "half-width", jitter_seed=11)
    _strip(ax, xp, P0_RELERR, C_P0, "peak pressure", jitter_seed=29)

    # annotate the mean +- sd for each metric just outside its strip
    ax.text(xa - 0.42, a_mean, f"mean {a_mean:.2f}%\nsd {a_sd:.2f}%", ha="right", va="center",
            fontsize=9.2, color=C_A)
    ax.text(xp + 0.42, p_mean, f"mean {p_mean:.2f}%\nsd {p_sd:.2f}%", ha="left", va="center",
            fontsize=9.2, color=C_P0)

    # call out the headline seed
    ax.annotate(f"reported headline\nseed {HEADLINE_SEED}: {a_head:.2f}%\n(best of five)",
                xy=(xa + 0.16, a_head), xytext=(xa + 0.30, a_head - 1.05),
                fontsize=8.8, color=C_HEAD, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=C_HEAD, lw=1.1, alpha=0.9))

    # plateau callout for p0 (below the strip, away from the legend)
    ax.annotate("realization-independent\nsystematic floor (~5.9%)",
                xy=(xp, p_lo - 0.05), xytext=(xp, p_lo - 1.35),
                ha="center", va="top", fontsize=8.8, color=C_P0,
                arrowprops=dict(arrowstyle="-", color=C_P0, lw=1.0, alpha=0.7))
    # scatter callout for a
    ax.text(xa, a_hi + 0.55, "realization-dependent\ncontact-edge scatter",
            ha="center", va="bottom", fontsize=8.8, color=C_A)

    ax.set_xticks([xa, xp])
    ax.set_xticklabels([r"half-width  $a_{\mathrm{relerr}}$",
                        r"peak pressure  $p_{0,\mathrm{relerr}}$"], fontsize=10.5)
    ax.set_xlim(0.35, 2.75)
    ax.set_ylim(0.0, max(a_hi, p_hi) + 1.7)
    ax.set_ylabel("relative error  (%)")
    ax.set_title("CV-8 deformable Hertz: 5-realization spread at the headline mesh (nx=192)\n"
                 "half-width scatters 2.75-4.30%; peak pressure holds a tight ~5.9% plateau (sd 0.07%)",
                 fontsize=10.2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)

    # legend proxy for the headline marker
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=C_HEAD, markeredgewidth=2.0, markersize=9,
               label=f"reported headline (seed {HEADLINE_SEED})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_A,
               markeredgecolor=C_A, markersize=7, label="other realization"),
        Line2D([0], [0], color=MUTE, lw=2.2, label="ensemble mean"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.4, framealpha=0.95,
              ncol=1, handletextpad=0.6, borderaxespad=0.5)

    out_png = os.path.join(FIG, "fig_cv8_ensemble_pub.png")
    out_pdf = os.path.join(FIG, "fig_cv8_ensemble_pub.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=BG)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nwrote {out_png}")
    print(f"wrote {out_pdf}")
    return out_png, out_pdf


if __name__ == "__main__":
    main()
