"""Von Mises stress of the GENUINE TWO-BLOCK rough-joint shear (Phase 1), as mid-plane cross-sections
over shear.

Runs the two-deformable-block decoder-FEM shear (`rock_joint_two_block.TwoBlockJointShear`) and, at
several shear increments, renders BOTH rough blocks clipped at the mid-y plane (x-z cross-section),
coloured by per-element von Mises stress — so you see the stress concentrate at the engaging asperities
of the mating rough faces as the joint shears.  Professional PyVista light-kit + oblique camera.

Output: figures/rock_joint_twoblock_vm_pub.png (+ .gif).

Run:  python3 postprocessing/plot_rock_joint_twoblock_vm.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from benchmarks.contact.cv_numerical.rock_joint_two_block import _make_blocks, MODES  # noqa: E402
from plot_rock_joint_3d_slices import von_mises                                       # noqa: E402

FIG = os.path.join(_ROOT, "figures")
AMP = 0.08
DEF_SCALE = 6.0          # displacement amplification for VISIBILITY (small strain) — labelled honestly


def capture(n_snap=5, n_cells=8, mode="in_plane", mu=0.4, compress=0.6, iters=3000):
    js, info = _make_blocks(amp=AMP, n_cells=n_cells, mu=mu, iters=iters, verbose=True)
    d = np.asarray(MODES[mode], float)
    uz = -compress * AMP                                          # CNV platen (fixed) -> stress builds
    u = np.zeros(js.ndof)
    snaps = []
    for ux in np.linspace(0.0, 0.16, n_snap):
        u, diag = js.solve_fixed((ux * d[0], ux * d[1], uz), d, u, max_iter=120)
        U = u.reshape(-1, 3)
        uL = U[:js.NL]; uU = U[js.NL:]
        vmL = von_mises(js.solL, uL.reshape(-1), js.stress_fn)
        vmU = von_mises(js.solU, uU.reshape(-1), js.stress_fn)
        snaps.append(dict(ux=float(ux),
                          ptsL=(js.pL + DEF_SCALE * uL).copy(), vmL=vmL.copy(),
                          ptsU=(js.pU + DEF_SCALE * uU).copy(), vmU=vmU.copy(),
                          resid=diag["resid_rel"], nC=diag["n_active"]))
        print(f"    u_x={ux:.3f}  vM_lo max={vmL.max():.1f}  vM_up max={vmU.max():.1f}  "
              f"nC={diag['n_active']}  resid={diag['resid_rel']:.1e}")
    return js, snaps


def _tet_grid(pts, elements, vm):
    import pyvista as pv
    M = elements.shape[0]
    cells = np.hstack([np.full((M, 1), 4, np.int64), elements.astype(np.int64)]).ravel()
    ctypes = np.full(M, 10, np.uint8)                            # VTK_TETRA
    g = pv.UnstructuredGrid(cells, ctypes, pts.astype(float))
    g["von Mises"] = vm
    return g


def render(js, snaps, out_png, out_gif, cmap_name="turbo", clim_pct=90.0, bg="#101418"):
    import matplotlib.cm as cm
    import pyvista as pv
    elemL = js.solL.elements.numpy(); elemU = js.solU.elements.numpy()
    # clip the colour range to the joint-stress band (not the top-platen reaction) so the asperity
    # pattern spans the full colormap; high-stress tips saturate at the top colour (stated honestly)
    vmax = float(np.percentile(np.concatenate([np.r_[s["vmL"], s["vmU"]] for s in snaps]), clim_pct))
    cmap = cm.get_cmap(cmap_name)
    cam = [(3.2, -8.5, 3.0), (0.0, 0.0, 1.0), (0, 0, 1)]         # oblique, facing the y=0 cut, joint at z~1
    frames = []
    for k, s in enumerate(snaps):
        pl = pv.Plotter(off_screen=True, window_size=(840, 860), lighting="three lights")
        pl.set_background(bg); pl.enable_anti_aliasing("fxaa")
        try:
            pl.enable_depth_peeling(8)
        except Exception:
            pass
        for pts, elem, vm in ((s["ptsL"], elemL, s["vmL"]), (s["ptsU"], elemU, s["vmU"])):
            g = _tet_grid(pts, elem, vm).clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
            # flat-ish shading (high ambient, no specular) so the COLOUR reads the stress directly,
            # not darkened by lighting; thin edges reveal the asperity geometry
            pl.add_mesh(g, scalars="von Mises", cmap=cmap, clim=(0, vmax), show_edges=True,
                        edge_color="#2b2f36", line_width=0.3, smooth_shading=False,
                        specular=0.0, ambient=0.55, diffuse=0.5, show_scalar_bar=False)
        pl.camera_position = cam; pl.camera.zoom(1.45)
        png = os.path.join(FIG, f"_tbvm_{k}.png"); pl.screenshot(png); pl.close(); frames.append(png)
    # composite with one shared colorbar (dark, to match the panels -> stress pattern pops)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.colors import Normalize
    n = len(frames); fg = "#e8eaed"
    fig, axs = plt.subplots(1, n, figsize=(2.5 * n + 0.6, 2.8)); fig.patch.set_facecolor(bg)
    for ax, png, s in zip(np.atleast_1d(axs), frames, snaps):
        ax.imshow(mpimg.imread(png)); ax.axis("off")
        ax.set_title(f"$u_x$={s['ux']:.3f} mm", fontsize=9, color=fg)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(np.atleast_1d(axs)), fraction=0.018, pad=0.01)
    cb.set_label("von Mises stress (saturated at joint band)", fontsize=8, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg); cb.outline.set_edgecolor(fg)
    fig.suptitle("GENUINE two-block rough joint — mid-plane cross-section (von Mises) over shear; BOTH "
                 f"mating faces deformable (decoder-FEM, deformation $\\times${DEF_SCALE:.0f})",
                 y=1.05, fontsize=9.5, color=fg)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=bg); plt.close(fig); print("  saved", out_png)
    try:
        import imageio
        imageio.mimsave(out_gif, [imageio.imread(f) for f in frames], duration=0.6)
        print("  saved", out_gif, "(%.1f MB)" % (os.path.getsize(out_gif) / 1e6))
    except Exception as e:
        print("  gif skipped:", repr(e))
    for f in frames:
        os.remove(f)


class _JS:                                                       # lightweight stand-in for --reuse render
    def __init__(self, elemL, elemU):
        class _S:
            def __init__(s, e): s.elements = torch.from_numpy(e)
        self.solL = _S(elemL); self.solU = _S(elemU)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse cached snapshots (skip the FEM solve)")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--clim_pct", type=float, default=90.0)
    ap.add_argument("--bg", default="#101418")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    cache = os.path.join(_ROOT, "runs", "rock_joint_decoder", "twoblock_vm_cache.npz")
    if args.reuse and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        snaps = list(d["snaps"]); js = _JS(d["elemL"], d["elemU"])
        print(f"  reused cached snapshots ({len(snaps)})")
    else:
        js, snaps = capture(n_snap=5, n_cells=8)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, snaps=np.array(snaps, dtype=object),
                 elemL=js.solL.elements.numpy(), elemU=js.solU.elements.numpy())
    render(js, snaps, os.path.join(FIG, "rock_joint_twoblock_vm_pub.png"),
           os.path.join(FIG, "rock_joint_twoblock_vm.gif"),
           cmap_name=args.cmap, clim_pct=args.clim_pct, bg=args.bg)


if __name__ == "__main__":
    main()
