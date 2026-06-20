"""Professional 2-D cross-section slices of the 3-D rough-joint contact problem, over shear time.

Runs the GENUINE decoder-FEM rough-joint shear (chart-FEM on a trained ChartDecoder + Coulomb friction,
converged consistent-tangent solver) and, at several shear increments, renders a PyVista view of the
deformable rough block **clipped at the mid-y plane** to expose the x-z cross-section, coloured by von
Mises stress, with the rigid mating rough counter-surface shown.  Professional light-kit + oblique
camera + depth peeling.

Output: figures/rock_joint_3d_slices_pub.png (multi-panel) and figures/rock_joint_3d_slices.gif.

Run:  /Users/wsun/opt/anaconda3/bin/python3 postprocessing/plot_rock_joint_3d_slices.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from solvers.fem.rough_block_decoder import band_limited_rough_surface, train_rough_decoder  # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import DecoderJointShear         # noqa: E402
FIG = os.path.join(_ROOT, "figures")
AMP = 0.08
DEF_SCALE = 6.0   # displacement amplification for VISIBILITY (small-strain) — labelled honestly


def von_mises(solver, u_block, stress_fn):
    """Per-element von Mises stress on the (chart-FEM) rough block."""
    F = solver.compute_F(torch.from_numpy(u_block.reshape(solver.n_nodes, 3)))
    sig = stress_fn(F).detach().numpy()                       # (M,3,3) Cauchy (small strain)
    s = 0.5 * (sig + np.transpose(sig, (0, 2, 1)))
    tr = (s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]) / 3.0
    dev = s - tr[:, None, None] * np.eye(3)[None]
    return np.sqrt(1.5 * (dev ** 2).sum((1, 2)))


def capture_snapshots(n_snap=5, n_cells=14, mu=0.4, sigma_n=2.0):
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=AMP)      # noqa: E731
    print("  training rough-block decoder ...")
    dec, rmse, dk = train_rough_decoder(tgt, iters=4000)
    print(f"  decoder reconstruction RMSE {rmse:.2e}")
    js = DecoderJointShear(dec, dk, n_cells=n_cells, mu=mu, surf_amp=AMP, eps_n=2.0e4)
    elements = js.solver.elements.numpy()
    xyz0 = js.xyz0                                            # rough physical reference coords
    z_p = 1.0 - 0.45 * AMP
    u = np.zeros(3 * js.N)
    snaps = []
    uxs = np.linspace(0.0, 0.16, n_snap)
    for ux in uxs:
        u, diag = js.solve_fixed(ux, z_p, u, max_iter=120)
        U = u.reshape(js.N, 3)
        vm = von_mises(js.solver, u, js.stress_fn)
        snaps.append(dict(ux=float(ux), pts=(xyz0 + DEF_SCALE * U).copy(), vm=vm.copy(),
                          z_p=z_p, resid=diag["resid_rel"], nC=diag["n_active"]))
        print(f"    u_x={ux:.3f}  von Mises max={vm.max():.2f}  nC={diag['n_active']}  resid={diag['resid_rel']:.1e}")
    return js, elements, snaps, dec, dk


def _tet_grid(pts, elements, vm):
    import pyvista as pv
    M = elements.shape[0]
    cells = np.hstack([np.full((M, 1), 4, np.int64), elements.astype(np.int64)]).ravel()
    ctypes = np.full(M, 10, np.uint8)                         # VTK_TETRA
    g = pv.UnstructuredGrid(cells, ctypes, pts.astype(float))
    g["von Mises"] = vm
    return g


def _upper_surface_grid(js, ux, z_p, n=60):
    import pyvista as pv
    L = js.L
    g = np.linspace(-L, L, n); X, Y = np.meshgrid(g, g, indexing="ij")
    Z = z_p + band_limited_rough_surface(X - ux, Y, amp=AMP)
    pts = np.c_[X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = [n, n, 1]
    return grid


def render(js, elements, snaps, out_png, out_gif):
    import matplotlib.cm as cm
    import pyvista as pv
    L = js.L
    vmax = float(np.percentile(np.concatenate([s["vm"] for s in snaps]), 98))
    cmap = cm.get_cmap("inferno")
    # near-frontal view of the x-z cut plane (the "2-D slice"), slightly off-axis for 3-D relief
    cam_pos = [(2.0, -6.4, 2.0), (0.0, 0.0, 0.7), (0, 0, 1)]
    frames = []
    for k, s in enumerate(snaps):
        pl = pv.Plotter(off_screen=True, window_size=(820, 820), lighting="light_kit")
        pl.set_background("white")
        pl.enable_anti_aliasing("fxaa")
        try:
            pl.enable_depth_peeling(8)
        except Exception:
            pass
        grid = _tet_grid(s["pts"], elements, s["vm"])
        clipped = grid.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)   # expose x-z cut face
        pl.add_mesh(clipped, scalars="von Mises", cmap=cmap, clim=(0, vmax), show_edges=False,
                    smooth_shading=True, specular=0.35, specular_power=18, ambient=0.28, diffuse=0.72,
                    show_scalar_bar=False)
        up = _upper_surface_grid(js, s["ux"], s["z_p"]).clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
        pl.add_mesh(up, color="#9aa0a6", opacity=0.40, smooth_shading=True, specular=0.2,
                    show_scalar_bar=False)
        pl.camera_position = cam_pos
        pl.camera.zoom(1.5)
        png = os.path.join(FIG, f"_slice_frame_{k}.png")
        pl.screenshot(png); pl.close()
        frames.append(png)
    # multi-panel composite with ONE shared colorbar
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.colors import Normalize
    n = len(frames)
    fig, axs = plt.subplots(1, n, figsize=(2.5 * n + 0.6, 2.7))
    for ax, png, s in zip(np.atleast_1d(axs), frames, snaps):
        ax.imshow(mpimg.imread(png)); ax.axis("off")
        ax.set_title(f"$u_x$={s['ux']:.3f} mm", fontsize=9)
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(np.atleast_1d(axs)), fraction=0.018, pad=0.01)
    cb.set_label("von Mises stress", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle("3-D rough-joint contact — mid-plane cross-section (von Mises) over shear; deformable "
                 f"atlas block on the rigid mating rough surface (deformation $\\times${DEF_SCALE:.0f})",
                 y=1.04, fontsize=9.5)
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  saved", out_png)
    # GIF
    try:
        import imageio
        imageio.mimsave(out_gif, [imageio.imread(f) for f in frames], duration=0.5)
        print("  saved", out_gif, "(%.1f MB)" % (os.path.getsize(out_gif) / 1e6))
    except Exception as e:
        print("  gif failed:", repr(e))
    for f in frames:
        os.remove(f)


class _JS:                                                   # minimal stand-in carrying L for render()
    def __init__(self, L, ux_zp):
        self.L = L; self._ux_zp = ux_zp


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--reuse", action="store_true",
                                                    help="reuse cached snapshots (skip the FEM solve)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    cache = os.path.join(_ROOT, "runs", "rock_joint_decoder", "slices_cache.npz")
    if args.reuse and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        elements = d["elements"]; snaps = list(d["snaps"]); L = float(d["L"])
        js = _JS(L, None)
        print(f"  reused cached snapshots ({len(snaps)})")
    else:
        js, elements, snaps, dec, dk = capture_snapshots(n_snap=5)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, elements=elements, snaps=np.array(snaps, dtype=object), L=js.L)
    render(js, elements, snaps,
           os.path.join(FIG, "rock_joint_3d_slices_pub.png"),
           os.path.join(FIG, "rock_joint_3d_slices.gif"))


if __name__ == "__main__":
    main()
