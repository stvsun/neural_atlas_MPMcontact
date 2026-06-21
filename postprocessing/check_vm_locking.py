"""Diagnose the checkerboard in the rough-joint von Mises field: CST per-element faceting / penalty-
contact concentration vs a genuine spurious (locking) mode.

Test: nodal-average the per-element von Mises.  If the checkerboard is CST/visualization faceting it
collapses under averaging (the smoothed field is clean); if it is a real spurious mode a structured
oscillation SURVIVES.  We also report a checkerboard index  RMS(vm - smoothed)/mean(vm)  per mesh — if
it DECREASES with refinement the oscillation is converging discretization error, not locking.

Run:  python3 postprocessing/check_vm_locking.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(_ROOT, "runs", "cv7_refinement")
FIG = os.path.join(_ROOT, "figures")


def _nodal_average(elements, vm, n_nodes):
    s = np.zeros(n_nodes); c = np.zeros(n_nodes)
    for a in range(4):
        np.add.at(s, elements[:, a], vm); np.add.at(c, elements[:, a], 1.0)
    return s / np.maximum(c, 1.0)


def _levels():
    out = []
    for f in glob.glob(os.path.join(RUN, "field_n*.npz")):
        m = re.search(r"field_n(\d+)\.npz", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main():
    levels = _levels()
    if not levels:
        print("  (no field dumps yet)"); return
    print("=== checkerboard index  RMS(vm - nodal-smoothed)/mean(vm)  per mesh ===")
    for nc, f in levels:
        d = np.load(f); elem = d["elements"]; vm = d["vm"]; N = d["nodes_ref"].shape[0]
        nodal = _nodal_average(elem, vm, N)
        elem_smooth = nodal[elem].mean(1)                       # smoothed per-element field
        idx = float(np.sqrt(np.mean((vm - elem_smooth) ** 2)) / max(vm.mean(), 1e-30))
        print(f"  n_cells={nc:2d}: checkerboard index = {idx:.3f}   (mean vM {vm.mean():.2f}, "
              f"peak {vm.max():.2f})")

    # side-by-side render of the FINEST available level: per-element (flat) vs nodal-averaged (smooth)
    nc, f = levels[-1]
    d = np.load(f); pts = (d["nodes_ref"] + 4.0 * d["u"]).astype(float)
    elem = d["elements"].astype(np.int64); vm = d["vm"]; N = pts.shape[0]
    nodal = _nodal_average(elem, vm, N)
    try:
        import matplotlib.cm as cm
        import pyvista as pv
        M = elem.shape[0]; cells = np.hstack([np.full((M, 1), 4, np.int64), elem]).ravel()
        vmax = float(np.percentile(vm, 97)); cmap = cm.get_cmap("turbo")
        cam = [(3.4, -3.8, 3.0), (0.0, 0.0, 0.55), (0, 0, 1)]
        outs = []
        for tag, mode in (("per-element (CST)", "cell"), ("nodal-averaged", "point")):
            g = pv.UnstructuredGrid(cells, np.full(M, 10, np.uint8), pts)
            if mode == "cell":
                g["vm"] = vm
            else:
                g.point_data["vm"] = nodal
            pl = pv.Plotter(off_screen=True, window_size=(740, 740), lighting="three lights")
            pl.set_background("#101418"); pl.enable_anti_aliasing("fxaa")
            pl.add_mesh(g, scalars="vm", cmap=cmap, clim=(0, vmax), show_edges=False,
                        smooth_shading=(mode == "point"), specular=0.0, ambient=0.5, diffuse=0.55,
                        interpolate_before_map=(mode == "point"), show_scalar_bar=False)
            pl.camera_position = cam; pl.camera.zoom(1.45)
            png = os.path.join(FIG, f"_lock_{mode}.png"); pl.screenshot(png); pl.close(); outs.append((tag, png))
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.colors import Normalize
        fg = "#e8eaed"
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3)); fig.patch.set_facecolor("#101418")
        for ax, (tag, png) in zip(axs, outs):
            ax.imshow(mpimg.imread(png)); ax.axis("off"); ax.set_title(tag, fontsize=10, color=fg)
        sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=list(axs), fraction=0.02, pad=0.01); cb.set_label("von Mises", color=fg)
        cb.ax.tick_params(colors=fg); cb.outline.set_edgecolor(fg)
        fig.suptitle(f"Checkerboard test (n_cells={nc}): per-element CST vs nodal-averaged von Mises",
                     y=1.02, fontsize=10, color=fg)
        out = os.path.join(FIG, "cv7_vm_locking_check_pub.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#101418"); plt.close(fig)
        for _, p in outs:
            os.remove(p)
        print("  saved", out)
    except Exception as e:
        print("  (render skipped:", repr(e), ")")


if __name__ == "__main__":
    main()
