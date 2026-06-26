"""3-D von Mises stress CONTOUR of the genuine TWO-BLOCK rough-joint shear (P1) at each refinement level.

Two-deformable-block analogue of `plot_rock_joint_refinement_vm.py`: reads `runs/cv7_refinement_p1/
field_n{nc}.npz` (BOTH blocks' deformed mesh + per-element von Mises) and renders both blocks as
nodal-averaged von Mises with a TRANSLUCENT domain + 3-D iso-contours, side-by-side over n_cells.
Progressive: re-runnable while finer meshes are still solving.

Output: figures/cv7_refinement_p1_vm_pub.png (+ per-level cv7_refine_p1_vm_n{nc}.png).

Run:  python3 postprocessing/plot_rock_joint_p1_refinement_vm.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(_ROOT, "runs", "cv7_refinement_p1")
FIG = os.path.join(_ROOT, "figures")
CMAP = "turbo"
BG = "#ffffff"


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


def _add_block(pl, cmap, pts, elem, vm, vmax, def_scale, opacity, n_iso, smooth, edges, iso_opacity=0.5):
    import pyvista as pv
    pts = (pts).astype(float); M = elem.shape[0]; N = pts.shape[0]
    cells = np.hstack([np.full((M, 1), 4, np.int64), elem.astype(np.int64)]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(M, 10, np.uint8), pts)
    if not smooth:
        g["von Mises"] = vm
        pl.add_mesh(g, scalars="von Mises", cmap=cmap, clim=(0, vmax), show_edges=edges,
                    edge_color="#2b2f36", line_width=0.25, smooth_shading=False, specular=0.0,
                    ambient=0.5, diffuse=0.55, show_scalar_bar=False)
        return
    g.point_data["von Mises"] = _nodal_average(elem, vm, N)
    pl.add_mesh(g, scalars="von Mises", cmap=cmap, clim=(0, vmax), opacity=opacity,
                smooth_shading=True, interpolate_before_map=True, specular=0.0, ambient=0.55,
                diffuse=0.5, show_edges=edges, edge_color="#3a3f47", line_width=0.2, show_scalar_bar=False)
    if n_iso > 0:
        try:                                                    # many nested semi-transparent iso-shells
            iso = g.contour(isosurfaces=np.linspace(0.10 * vmax, 0.95 * vmax, n_iso).tolist(),
                            scalars="von Mises")
            if iso.n_points > 0:
                pl.add_mesh(iso, scalars="von Mises", cmap=cmap, clim=(0, vmax), opacity=iso_opacity,
                            smooth_shading=True, specular=0.12, specular_power=15, show_scalar_bar=False)
        except Exception as e:
            print("  (iso skipped:", repr(e), ")")


def _trim_top_layer(pts, elem, vm, n_cells, frac):
    """Drop the upper block's TOP element layer — a spurious von Mises boundary layer at the rigid-platen
    Dirichlet BC (the prescribed-displacement top face concentrates the imposed shear into one element
    layer; NOT the joint physics).  Keeps the genuine joint/bulk stress."""
    if frac <= 0:
        return elem, vm
    zc = pts[elem].mean(1)[:, 2]; ztop = pts[:, 2].max()
    keep = zc <= ztop - frac * (2.0 / n_cells)
    return elem[keep], vm[keep]


def _render_one(nc, npz, vmax, def_scale, opacity, n_iso, smooth, edges, trim=1.3, iso_opacity=0.5):
    import matplotlib.cm as cm
    import pyvista as pv
    d = np.load(npz); cmap = cm.get_cmap(CMAP)
    pl = pv.Plotter(off_screen=True, window_size=(820, 860), lighting="three lights")
    pl.set_background(BG); pl.enable_anti_aliasing("fxaa")
    try:
        pl.enable_depth_peeling(max(12, n_iso + 4))             # enough peels for many nested iso-shells
    except Exception:
        pass
    ncc = int(d["n_cells"])
    ptsU = d["nodes_U"] + def_scale * d["u_U"]
    elemU, vmU = _trim_top_layer(ptsU, d["elem_U"], d["vm_U"], ncc, trim)   # remove platen-BC artifact
    _add_block(pl, cmap, d["nodes_L"] + def_scale * d["u_L"], d["elem_L"], d["vm_L"], vmax,
               def_scale, opacity, n_iso, smooth, edges, iso_opacity)
    _add_block(pl, cmap, ptsU, elemU, vmU, vmax, def_scale, opacity, n_iso, smooth, edges, iso_opacity)
    # frame the ENTIRE domain (both blocks): set the oblique view direction, then auto-fit the bounds
    pl.camera_position = [(4.2, -9.5, 4.0), (0.3, 0.0, 1.0), (0, 0, 1)]
    pl.reset_camera()
    pl.camera.zoom(0.92)                                                 # small margin around the full domain
    png = os.path.join(FIG, f"cv7_refine_p1_vm_n{nc}.png")
    pl.screenshot(png); pl.close()
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def_scale", type=float, default=6.0)
    ap.add_argument("--clim_pct", type=float, default=88.0)   # joint-stress band (not the top-platen reaction)
    ap.add_argument("--opacity", type=float, default=0.22)
    ap.add_argument("--iso", type=int, default=14)
    ap.add_argument("--iso_opacity", type=float, default=0.45)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--trim_platen", type=float, default=1.3,
                    help="trim N element-layers off the upper block top (rigid-platen BC artifact); 0=keep")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    smooth = not args.raw
    levels = _levels()
    if not levels:
        print("  (no P1 field dumps yet; skip)"); return
    # colour range from the PHYSICAL stress (excluding the trimmed platen-BC boundary layer)
    phys = []
    for _, f in levels:
        d = np.load(f); phys.append(d["vm_L"])
        ptsU = d["nodes_U"] + args.def_scale * d["u_U"]
        _, vmU = _trim_top_layer(ptsU, d["elem_U"], d["vm_U"], int(d["n_cells"]), args.trim_platen)
        phys.append(vmU)
    vmax = float(np.percentile(np.concatenate(phys), args.clim_pct))
    frames = [(nc, _render_one(nc, f, vmax, args.def_scale, args.opacity, args.iso, smooth, args.edges,
                               args.trim_platen, args.iso_opacity)) for nc, f in levels]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg, matplotlib.cm as cm
    from matplotlib.colors import Normalize
    n = len(frames); fg = "#1a1a1a"
    fig, axs = plt.subplots(1, n, figsize=(2.85 * n + 0.7, 3.3)); fig.patch.set_facecolor(BG)
    for ax, (nc, png) in zip(np.atleast_1d(axs), frames):
        ne = int(np.load(dict(levels)[nc])["n_elem"])
        ax.imshow(mpimg.imread(png)); ax.axis("off")
        ax.set_title(f"$n_{{\\mathrm{{cells}}}}={nc}$  ({ne} tets)", fontsize=9.5, color=fg)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cm.get_cmap(CMAP)); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(np.atleast_1d(axs)), fraction=0.018, pad=0.01)
    cb.set_label("von Mises stress", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=8, colors=fg); cb.outline.set_edgecolor("0.6")
    out = os.path.join(FIG, "cv7_refinement_p1_vm_pub.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  saved {out}  (levels n_cells={[nc for nc, _ in frames]})")


if __name__ == "__main__":
    main()
