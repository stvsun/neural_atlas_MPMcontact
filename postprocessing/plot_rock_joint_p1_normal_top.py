"""Top-down NORMAL-STRESS (sigma_zz) contour of the bottom block's rough top (contact) face, on the same
mesh-size (x) x shear-displacement (y) grid as the von Mises matrix (P1 two-block).

The time study only dumped von Mises, but the per-snapshot displacement U_L is saved, so sigma_zz is
RECOMPUTED here by rebuilding ONLY the lower-block chart-FEM (the decoder is deterministic; we need just
compute_F + Hooke's law, no stiffness assembly).  The contact-face normal stress (≈ contact pressure;
compression negative) is then drawn as a top-down filled contour for every (mesh, displacement) cell.

Output: figures/cv7_refinement_p1_normal_top_pub.png

Run:  python3 postprocessing/plot_rock_joint_p1_normal_top.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver                                # noqa: E402
from solvers.fem.linear_elastic import make_linear_elastic_small_strain                      # noqa: E402
from plot_rock_joint_p1_refinement_vm import _nodal_average                                  # noqa: E402

RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1_time")
FIG = os.path.join(_ROOT, "figures")
BG = "#ffffff"
torch.set_default_dtype(torch.float64)


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
    print("  retraining the (deterministic) lower-block decoder ...")
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=args.amp)     # noqa: E731
    decL, rmse, dkL = train_rough_decoder(tgt, rough_face="top", iters=args.iters)
    stress_fn, _ = make_linear_elastic_small_strain(2.0e3, 0.25)

    cells = {}
    times = None
    for nc, f in levels:
        sol = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=decL, decoder_kwargs=dkL,
                                   dtype=torch.float64)
        top = np.where(np.abs(sol.nodes.numpy()[:, 2] - 1.0) < 1e-9)[0]   # reference top face (z=+1)
        xy = sol.nodes_phys.numpy()[top, :2]
        d = np.load(f); UL = d["U_L"]; times = d["times"]
        szz = []
        for ti in range(UL.shape[0]):
            F = sol.compute_F(torch.from_numpy(UL[ti]))
            sig = stress_fn(F).detach().numpy()                           # (M,3,3) Cauchy
            nodal = _nodal_average(sol.elements.numpy(), sig[:, 2, 2], sol.n_nodes)
            szz.append(nodal[top])
        cells[nc] = dict(xy=xy, szz=np.array(szz), n_elem=int(d["n_elem"]))
        print(f"    n_cells={nc}: top-face sigma_zz computed ({len(top)} nodes)")

    use_t = list(range(len(times) - max(0, args.drop_last_time)))
    allszz = np.concatenate([cells[nc]["szz"][use_t].ravel() for nc, _ in levels])
    # sigma_zz is almost all compressive -> use the actual data range (not symmetric) for contrast,
    # but keep 0 in the map so any tension still reads on the diverging colormap
    lo = float(np.percentile(allszz, 100 - args.clim_pct)); hi = max(0.0, float(np.percentile(allszz, args.clim_pct)))
    levels_cf = np.linspace(lo, hi, 25)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fg = "#1a1a1a"; ncols = len(levels); nrows = len(use_t)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 0.9, 2.5 * nrows + 0.4))
    fig.patch.set_facecolor(BG); axs = np.atleast_2d(axs)
    cf = None
    for r, ti in enumerate(use_t):
        for c, (nc, _) in enumerate(levels):
            ax = axs[r, c]; dat = cells[nc]
            cf = ax.tricontourf(dat["xy"][:, 0], dat["xy"][:, 1], dat["szz"][ti], levels=levels_cf,
                                cmap="cividis", vmin=lo, vmax=hi, extend="both")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#ffffff")
            for s in ax.spines.values():
                s.set_color("0.7")
            if r == 0:
                ax.set_title(f"$n_{{\\mathrm{{cells}}}}={nc}$  ({dat['n_elem']} tets)", fontsize=9.5, color=fg)
            if c == 0:
                ax.text(-0.17, 0.5, f"$u_x={times[ti]:.3f}$ mm", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, color=fg)
    cb = fig.colorbar(cf, ax=list(axs.ravel()), fraction=0.013, pad=0.01)
    cb.set_label(r"normal stress $\sigma_{zz}$  ($-$ = compression)", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor("0.6")
    out = os.path.join(FIG, "cv7_refinement_p1_normal_top_pub.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  saved {out}  ({ncols} meshes [x] x {nrows} displacements [y])")


if __name__ == "__main__":
    main()
