"""4x4 (mesh x loading-time) matrix of 3-D von Mises contours for the TWO-BLOCK rough-joint shear (P1).

Rows = mesh refinement (n_cells), columns = shear increment (u_x = 25/50/75/100%).  Each cell is a 3-D
von Mises contour of BOTH deformable blocks (nodal-averaged, translucent domain + iso-contours, full
domain in view, rigid-platen boundary layer trimmed), so the stress EVOLUTION over loading and across
mesh refinement is read off a single grid.  Reads runs/cv7_refinement_p1_time/field_n{nc}.npz (4 time
snapshots/mesh).  Progressive: renders whatever meshes are present.

Output: figures/cv7_refinement_p1_time_pub.png

Run:  python3 postprocessing/plot_rock_joint_p1_time_matrix.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
sys.path.insert(0, _ROOT)
from plot_rock_joint_p1_refinement_vm import _add_block, _trim_top_layer  # noqa: E402

RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1_time")
FIG = os.path.join(_ROOT, "figures")
CMAP = "turbo"
BG = "#101418"


def _levels():
    out = []
    for f in glob.glob(os.path.join(RUN, "field_n*.npz")):
        m = re.search(r"field_n(\d+)\.npz", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def _render_cell(d, ti, vmax, def_scale, opacity, n_iso, iso_opacity, trim):
    import matplotlib.cm as cm
    import pyvista as pv
    cmap = cm.get_cmap(CMAP)
    pl = pv.Plotter(off_screen=True, window_size=(560, 600), lighting="three lights")
    pl.set_background(BG); pl.enable_anti_aliasing("fxaa")
    try:
        pl.enable_depth_peeling(max(12, n_iso + 4))
    except Exception:
        pass
    nc = int(d["n_cells"])
    ptsU = d["nodes_U"] + def_scale * d["U_U"][ti]
    elemU, vmU = _trim_top_layer(ptsU, d["elem_U"], d["VM_U"][ti], nc, trim)
    _add_block(pl, cmap, d["nodes_L"] + def_scale * d["U_L"][ti], d["elem_L"], d["VM_L"][ti], vmax,
               def_scale, opacity, n_iso, True, False, iso_opacity)
    _add_block(pl, cmap, ptsU, elemU, vmU, vmax, def_scale, opacity, n_iso, True, False, iso_opacity)
    pl.camera_position = [(4.2, -9.5, 4.0), (0.3, 0.0, 1.0), (0, 0, 1)]
    pl.reset_camera(); pl.camera.zoom(0.92)
    png = os.path.join(FIG, f"_p1t_n{nc}_t{ti}.png"); pl.screenshot(png); pl.close()
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def_scale", type=float, default=6.0)
    ap.add_argument("--clim_pct", type=float, default=90.0)
    ap.add_argument("--opacity", type=float, default=0.22)
    ap.add_argument("--iso", type=int, default=14)
    ap.add_argument("--iso_opacity", type=float, default=0.45)
    ap.add_argument("--trim_platen", type=float, default=1.3)
    ap.add_argument("--drop_last_time", type=int, default=1, help="drop the last N load increments")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    levels = _levels()                                          # (n_cells, file) sorted -> the X axis (columns)
    if not levels:
        print("  (no P1 time field dumps yet; skip)"); return
    times = np.load(levels[0][1])["times"]
    use_t = list(range(len(times) - max(0, args.drop_last_time)))   # the Y axis (rows): drop u_x=0.16
    ds = {nc: np.load(f) for nc, f in levels}
    # physical colour range over the USED (mesh, displacement) cells, excluding the platen boundary layer
    phys = []
    for nc, _ in levels:
        d = ds[nc]
        for ti in use_t:
            phys.append(d["VM_L"][ti])
            ptsU = d["nodes_U"] + args.def_scale * d["U_U"][ti]
            _, vmU = _trim_top_layer(ptsU, d["elem_U"], d["VM_U"][ti], int(d["n_cells"]), args.trim_platen)
            phys.append(vmU)
    vmax = float(np.percentile(np.concatenate(phys), args.clim_pct))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg, matplotlib.cm as cm
    from matplotlib.colors import Normalize
    fg = "#e8eaed"
    ncols = len(levels)                                         # X = mesh size
    nrows = len(use_t)                                          # Y = displacement progression
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 0.8, 2.45 * nrows + 0.3))
    fig.patch.set_facecolor(BG)
    axs = np.atleast_2d(axs)
    for r, ti in enumerate(use_t):                             # rows = increasing displacement (downward)
        for c, (nc, _) in enumerate(levels):                  # cols = increasing mesh size (rightward)
            d = ds[nc]
            png = _render_cell(d, ti, vmax, args.def_scale, args.opacity, args.iso, args.iso_opacity,
                               args.trim_platen)
            ax = axs[r, c]; ax.imshow(mpimg.imread(png)); ax.axis("off"); os.remove(png)
            if r == 0:
                ne = int(d["n_elem"])
                ax.set_title(f"$n_{{cells}}$={nc}  ({ne} tets)", fontsize=9.5, color=fg)
            if c == 0:
                ax.text(-0.06, 0.5, f"$u_x$={times[ti]:.3f} mm", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10, color=fg)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cm.get_cmap(CMAP)); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axs.ravel()), fraction=0.012, pad=0.01)
    cb.set_label("von Mises stress", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor(fg)
    fig.suptitle("Two-block rough-joint shear — von Mises stress: mesh size (x) × shear displacement (y "
                 "↓ progresses).  Both faces deformable; nodal-averaged, full domain, platen layer trimmed",
                 y=1.005, fontsize=10.5, color=fg)
    out = os.path.join(FIG, "cv7_refinement_p1_time_pub.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  saved {out}  ({ncols} meshes [x] x {nrows} displacements [y])")


if __name__ == "__main__":
    main()
