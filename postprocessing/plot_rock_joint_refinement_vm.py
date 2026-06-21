"""3-D von Mises stress CONTOUR of the genuine rough-joint shear at each mesh-refinement level.

Reads the per-level field dumps `runs/cv7_refinement/field_n{nc}.npz` (deformed rough block + per-element
von Mises) written by `cv7_refinement_study.py`, and renders a 3-D von Mises contour for each mesh
resolution — so the stress field (and how its asperity-tip contours SHARPEN as the mesh refines) is
visible.  Progressive: re-runnable while finer meshes are still solving (renders whatever levels exist).

Output: figures/cv7_refinement_vm_pub.png (side-by-side over n_cells) + per-level cv7_refine_vm_n{nc}.png.

Run:  python3 postprocessing/plot_rock_joint_refinement_vm.py [--def_scale 4]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RUN = os.path.join(_ROOT, "runs", "cv7_refinement")
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


def _nodal_average(elements, vm, n_nodes):
    """Recover a continuous nodal stress field from the per-element (CST) von Mises — the standard
    post-processing for constant-strain tets (removes the per-element faceting; NOT a locking fix,
    just smooth stress recovery)."""
    s = np.zeros(n_nodes); c = np.zeros(n_nodes)
    for a in range(4):
        np.add.at(s, elements[:, a], vm); np.add.at(c, elements[:, a], 1.0)
    return s / np.maximum(c, 1.0)


def _rigid_surface_grid(z_p, u_x, surf_amp, L=1.0, n=80):
    """The RIGID mating rough surface of the one-block (P2) problem: z = z_p + h(x - u_x, y)."""
    import pyvista as pv
    from solvers.fem.rough_block_decoder import band_limited_rough_surface
    gx = np.linspace(-L, L, n); X, Y = np.meshgrid(gx, gx, indexing="ij")
    Z = z_p + band_limited_rough_surface(X - u_x, Y, amp=surf_amp)
    pts = np.c_[X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = [n, n, 1]
    return grid


def _add_vm_block(pl, g, elem, vm_cell, N, vmax, cmap, smooth, opacity, n_iso, edges):
    """Add the deformable block to the plotter as either raw-CST or nodal-averaged + iso-contours."""
    if not smooth:
        g["von Mises"] = vm_cell
        pl.add_mesh(g, scalars="von Mises", cmap=cmap, clim=(0, vmax), show_edges=edges,
                    edge_color="#2b2f36", line_width=0.25, smooth_shading=False, specular=0.0,
                    ambient=0.5, diffuse=0.55, show_scalar_bar=False)
        return
    g.point_data["von Mises"] = _nodal_average(elem, vm_cell, N)
    pl.add_mesh(g, scalars="von Mises", cmap=cmap, clim=(0, vmax), opacity=opacity,
                smooth_shading=True, interpolate_before_map=True, specular=0.0, ambient=0.55,
                diffuse=0.5, show_edges=edges, edge_color="#3a3f47", line_width=0.2, show_scalar_bar=False)
    if n_iso > 0:
        try:
            vals = np.linspace(0.12 * vmax, 0.95 * vmax, n_iso).tolist()
            iso = g.contour(isosurfaces=vals, scalars="von Mises")
            if iso.n_points > 0:
                # more isosurfaces -> more transparent each, so nested shells read as a layered gradient
                iso_op = float(np.clip(4.0 / max(n_iso, 1), 0.22, 0.92))
                pl.add_mesh(iso, scalars="von Mises", cmap=cmap, clim=(0, vmax), opacity=iso_op,
                            smooth_shading=True, specular=0.15, specular_power=15, show_scalar_bar=False)
        except Exception as e:
            print("  (iso-contour skipped:", repr(e), ")")


def _render_scene(out_png, pts, elem, vm_cell, vmax, smooth, opacity, n_iso, edges,
                  rigid=None, zoom=0.92, win=820):
    """Render one 3-D von Mises scene (block [+ optional rigid mating surface]) zoomed out."""
    import matplotlib.cm as cm
    import pyvista as pv
    M = elem.shape[0]; N = pts.shape[0]
    cells = np.hstack([np.full((M, 1), 4, np.int64), elem.astype(np.int64)]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(M, 10, np.uint8), pts.astype(float))
    cmap = cm.get_cmap(CMAP)
    pl = pv.Plotter(off_screen=True, window_size=(win, win), lighting="three lights")
    pl.set_background(BG); pl.enable_anti_aliasing("fxaa")
    try:
        pl.enable_depth_peeling(12)
    except Exception:
        pass
    _add_vm_block(pl, g, elem, vm_cell, N, vmax, cmap, smooth, opacity, n_iso, edges)
    if rigid is not None:                                       # translucent RIGID mating rough surface
        z_p, u_x, surf_amp = rigid
        pl.add_mesh(_rigid_surface_grid(z_p, u_x, surf_amp), color="#b9c0c8", opacity=0.45,
                    smooth_shading=True, specular=0.3, specular_power=20, show_scalar_bar=False)
    # zoomed-OUT oblique view showing the block AND the rigid surface above it
    pl.camera_position = [(4.6, -5.2, 3.6), (0.0, 0.0, 0.4), (0, 0, 1)]
    pl.camera.zoom(zoom)
    pl.screenshot(out_png); pl.close()
    return out_png


def _render_one(nc, npz, vmax, def_scale, edges, smooth=True, opacity=0.28, n_iso=5, show_rigid=True):
    d = np.load(npz)
    pts = (d["nodes_ref"] + def_scale * d["u"]).astype(float)
    rigid = (float(d["z_p"]), 0.16, float(d["surf_amp"])) if (show_rigid and "z_p" in d) else None
    return _render_scene(os.path.join(FIG, f"cv7_refine_vm_n{nc}.png"), pts, d["elements"], d["vm"],
                         vmax, smooth, opacity, n_iso, edges, rigid=rigid)


def _te_levels():
    out = []
    for f in glob.glob(os.path.join(RUN, "field_te_n*.npz")):
        m = re.search(r"field_te_n(\d+)\.npz", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def time_evolution_figure(def_scale=1.0, opacity=0.22, n_iso=12, edges=False,
                          out_name="cv7_time_evolution_vm_pub.png"):
    """4x4 grid (ROWS = mesh resolution, COLS = loading time) of the 3-D von Mises contour with the
    rigid mating surface — shows the stress evolution AND the mesh refinement in one figure."""
    levels = _te_levels()
    if not levels:
        print("  (no time-evolution field dumps yet; skip)"); return
    allvm = np.concatenate([np.load(f)["vm_t"].ravel() for _, f in levels])
    vmax = float(np.percentile(allvm, 97))
    nt = int(np.load(levels[0][1])["vm_t"].shape[0])
    ux = np.load(levels[0][1])["ux_t"]
    cells = [nc for nc, _ in levels]
    panels = {}
    for nc, f in levels:
        d = np.load(f)
        for t in range(nt):
            pts = (d["nodes_ref"] + def_scale * d["u_t"][t]).astype(float)
            rigid = (float(d["zp_t"][t]), float(d["ux_t"][t]), float(d["surf_amp"]))
            png = os.path.join(FIG, f"_te_{nc}_{t}.png")
            _render_scene(png, pts, d["elements"], d["vm_t"][t], vmax, True, opacity, n_iso, edges,
                          rigid=rigid, zoom=0.95, win=560)
            panels[(nc, t)] = png
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg, matplotlib.cm as cm
    from matplotlib.colors import Normalize
    fg = "#e8eaed"
    fig, axs = plt.subplots(len(cells), nt, figsize=(2.4 * nt + 0.8, 2.4 * len(cells) + 0.4))
    fig.patch.set_facecolor(BG); axs = np.atleast_2d(axs)
    for i, nc in enumerate(cells):
        ne = int(np.load(dict(levels)[nc])["n_elem"])
        for t in range(nt):
            ax = axs[i, t]; ax.imshow(mpimg.imread(panels[(nc, t)])); ax.axis("off")
            if i == 0:
                ax.set_title(f"$u_x$ = {ux[t]:.3f} mm", fontsize=10, color=fg)
            if t == 0:
                ax.text(-0.08, 0.5, f"$n_{{cells}}$={nc}\n({ne} tets)", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, color=fg, rotation=90)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cm.get_cmap(CMAP)); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axs.ravel()), fraction=0.012, pad=0.01)
    cb.set_label("von Mises stress", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor(fg)
    fig.suptitle("Genuine rough-joint shear — 3-D von Mises contour: loading time (cols) x mesh "
                 "refinement (rows), with the rigid mating surface", y=1.005, fontsize=11, color=fg)
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    for p in panels.values():
        os.remove(p)
    print(f"  saved {out}  (mesh {cells} x {nt} times)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def_scale", type=float, default=1.0, help="displacement amplification (1=true, for rigid-surface consistency)")
    ap.add_argument("--clim_pct", type=float, default=97.0)
    ap.add_argument("--edges", action="store_true", help="overlay mesh edges (default off for fine meshes)")
    ap.add_argument("--raw", action="store_true", help="raw per-element CST stress (default: nodal-averaged)")
    ap.add_argument("--opacity", type=float, default=0.22, help="translucent domain opacity")
    ap.add_argument("--iso", type=int, default=12, help="number of 3-D iso-contour surfaces (0=none)")
    ap.add_argument("--te", action="store_true", help="render the 4x4 (mesh x time) evolution figure")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if args.te:
        time_evolution_figure(def_scale=args.def_scale, opacity=args.opacity, n_iso=args.iso,
                              edges=args.edges)
        return
    smooth = not args.raw
    levels = _levels()
    if not levels:
        print("  (no field dumps yet; skip)"); return
    # shared colour range across levels so the contours are directly comparable
    allvm = np.concatenate([np.load(f)["vm"] for _, f in levels])
    vmax = float(np.percentile(allvm, args.clim_pct))
    frames = [(nc, _render_one(nc, f, vmax, args.def_scale, args.edges, smooth, args.opacity, args.iso))
              for nc, f in levels]
    # composite over n_cells (dark, shared colorbar)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg, matplotlib.cm as cm
    from matplotlib.colors import Normalize
    n = len(frames); fg = "#e8eaed"
    fig, axs = plt.subplots(1, n, figsize=(2.5 * n + 0.7, 2.9)); fig.patch.set_facecolor(BG)
    for ax, (nc, png) in zip(np.atleast_1d(axs), frames):
        ne = int(np.load(dict(levels)[nc])["n_elem"])
        ax.imshow(mpimg.imread(png)); ax.axis("off")
        ax.set_title(f"$n_{{cells}}$={nc}  ({ne} tets)", fontsize=9, color=fg)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cm.get_cmap(CMAP)); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(np.atleast_1d(axs)), fraction=0.018, pad=0.01)
    cb.set_label("von Mises stress", fontsize=8, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor(fg)
    stresstag = "per-element CST" if args.raw else f"nodal-averaged, translucent domain + {args.iso} iso-contours"
    fig.suptitle("3-D von Mises stress contour of the genuine rough-joint shear vs mesh refinement "
                 f"(decoder-FEM, P2; {stresstag}; deformation $\\times${args.def_scale:.0f})",
                 y=1.05, fontsize=9.0, color=fg)
    out = os.path.join(FIG, "cv7_refinement_vm_pub.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  saved {out}  (levels n_cells={[nc for nc, _ in frames]})")


if __name__ == "__main__":
    main()
