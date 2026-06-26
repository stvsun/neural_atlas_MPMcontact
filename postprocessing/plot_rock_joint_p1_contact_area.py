"""Top-down REAL CONTACT AREA of the rough joint, on the same mesh-size (x) x shear-displacement (y) grid.

On a rough joint the two faces touch only on a SUBSET of the nominal area (the bearing asperities).  Here
the active contact set (gap < 0) and the contact pressure are recomputed from the saved FEM displacement
snapshots (rebuild the two-block GEOMETRY only — no stiffness assembly), and drawn top-down on the bottom
block's contact face: bright = in contact (coloured by pressure), dark = separated.  The real-contact-area
fraction A_c/A is annotated per cell.

Output: figures/cv7_refinement_p1_contact_area_pub.png

Run:  python3 postprocessing/plot_rock_joint_p1_contact_area.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from benchmarks.contact.cv_numerical.rock_joint_two_block import TwoBlockJointShear, _make_blocks  # noqa: E402

RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1_time")
FIG = os.path.join(_ROOT, "figures")
BG = "#ffffff"


def _levels():
    out = []
    for f in glob.glob(os.path.join(RUN, "field_n*.npz")):
        m = re.search(r"field_n(\d+)\.npz", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=0.08)
    ap.add_argument("--clim_pct", type=float, default=97.0)
    ap.add_argument("--drop_last_time", type=int, default=1)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    levels = _levels()
    if not levels:
        print("  (no P1 time field dumps yet; skip)"); return
    print("  retraining the (deterministic) decoders ...")
    _, info = _make_blocks(amp=args.amp, n_cells=6, iters=args.iters)        # same decoders as the study
    decL, dkL, decU, dkU = info["decL"], info["dkL"], info["decU"], info["dkU"]
    d_dir = np.array([1.0, 0.0])                                             # in-plane shear direction

    cells = {}; times = None
    for nc, f in levels:
        js = TwoBlockJointShear(decL, dkL, decU, dkU, n_cells=nc, surf_amp=args.amp, build_K=False)
        xy = js.pL[js.slave, :2]
        d = np.load(f); UL = d["U_L"]; UU = d["U_U"]; times = d["times"]
        press, frac = [], []
        for ti in range(UL.shape[0]):
            u = np.concatenate([UL[ti].reshape(-1), UU[ti].reshape(-1)])
            geom = js._contact_geom(u, d_dir)
            gN, active = geom[5], geom[6]
            press.append(js.eps_n * np.maximum(-gN, 0.0))                   # contact pressure (0 if separated)
            frac.append(float(active.mean()))                              # real-contact-area fraction
        cells[nc] = dict(xy=xy, press=np.array(press), frac=np.array(frac), n_elem=int(d["n_elem"]))
        print(f"    n_cells={nc}: contact set computed ({len(js.slave)} face nodes)")

    use_t = list(range(len(times) - max(0, args.drop_last_time)))
    pmax = float(np.percentile(np.concatenate([cells[nc]["press"][use_t].ravel() for nc, _ in levels]),
                               args.clim_pct))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fg = "#1a1a1a"; ncols = len(levels); nrows = len(use_t)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 0.9, 2.5 * nrows + 0.4))
    fig.patch.set_facecolor(BG); axs = np.atleast_2d(axs)
    cf = None
    for r, ti in enumerate(use_t):
        for c, (nc, _) in enumerate(levels):
            ax = axs[r, c]; dat = cells[nc]
            cf = ax.tricontourf(dat["xy"][:, 0], dat["xy"][:, 1], dat["press"][ti],
                                levels=np.linspace(0, pmax, 24), cmap="magma_r", vmin=0, vmax=pmax,
                                extend="max")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#ffffff")
            for s in ax.spines.values():
                s.set_color("0.7")
            ax.text(0.5, 0.035, f"$A_c/A={100*dat['frac'][ti]:.0f}\\%$", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
            if r == 0:
                ax.set_title(f"$n_{{\\mathrm{{cells}}}}={nc}$  ({dat['n_elem']} tets)", fontsize=9.5, color=fg)
            if c == 0:
                ax.text(-0.17, 0.5, f"$u_x={times[ti]:.3f}$ mm", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, color=fg)
    cb = fig.colorbar(cf, ax=list(axs.ravel()), fraction=0.013, pad=0.01)
    cb.set_label("contact pressure  (dark = bearing asperities)", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor("0.6")
    out = os.path.join(FIG, "cv7_refinement_p1_contact_area_pub.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  saved {out}  ({ncols} meshes [x] x {nrows} displacements [y])")


if __name__ == "__main__":
    main()
