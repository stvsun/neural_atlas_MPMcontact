"""CV-6 figure 18 (3-D, PyVista): the measured neural-SDF refinement ceiling.

For each Koch fractal level n a fixed-capacity (12,801-parameter) plain coordinate neural
SDF is trained on the EXACT signed distance of the level-n Koch prism. We render, in lit
3-D, the neural SDF's reconstructed boundary wall -- each true boundary point displaced by
the neural signed distance along its outward normal -- coloured by the reconstruction error
|phi_nn|/L. The sharp true Koch outline is overlaid (black). As n grows the network can no
longer carve the self-similar spikes: the reconstruction rounds off and the error saturates
(the spectral-bias ceiling), while the true outline stays sharp at every level. The recursive
chart stores the O(1) generating rule and is exact at any depth.

Run:  <venv>/bin/python postprocessing/plot_koch_ceiling_3d.py
Out:  figures/koch_neural_ceiling_pub.png (+ .pdf)  -- replaces the old 2-D line plot.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "contact"))
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from solvers.contact import koch
from atlas.sdf.train_sdf import SDFNet
import koch_neural_ceiling as K

import pyvista as pv
pv.OFF_SCREEN = True

FIG = os.path.join(_ROOT, "figures")
DT = torch.float32
torch.manual_seed(0)
H = 0.5                       # prism half-height
CMAP = "inferno"


def train_level_fast(level, width=64, depth=4, epochs=3200, z_range=0.75, n_z=3,
                     n_near=1700, n_bulk=900, band=0.04, lr=2e-3, w_eik=0.1, seed=0):
    """Compact float32 retrain of the fixed-capacity SDFNet on the level-n Koch prism."""
    xy, d2 = K.build_dataset(level, n_near, n_bulk, band, seed, use_cache=True)
    zs = np.linspace(-z_range, z_range, n_z)
    X = torch.tensor(np.column_stack([np.tile(xy, (n_z, 1)), np.repeat(zs, len(xy))]), dtype=DT)
    g = torch.tensor(K.extrusion_sdf(np.tile(d2, n_z), np.repeat(zs, len(xy)), H), dtype=DT)
    model = SDFNet(width=width, depth=depth).to(DT)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed + 99); half = 1.3
    t0 = time.time()
    loss_data = torch.tensor(0.0)
    for ep in range(epochs):
        opt.zero_grad()
        loss_data = torch.mean(torch.abs(model(X).reshape(-1) - g))
        pe = np.column_stack([rng.uniform(-half, half, (1536, 2)),
                              rng.uniform(-z_range, z_range, (1536, 1))])
        pe_t = torch.tensor(pe, dtype=DT, requires_grad=True)
        ge = torch.autograd.grad(model(pe_t).sum(), pe_t, create_graph=True)[0]
        (loss_data + w_eik * torch.mean((torch.linalg.norm(ge, dim=1) - 1.0) ** 2)).backward()
        opt.step()
    npar = sum(p.numel() for p in model.parameters())
    print(f"  level {level}: {npar} params, {epochs} ep, {time.time()-t0:.1f}s, "
          f"|phi-d|={loss_data.item():.2e}")
    return model


def _extrude_ring(ring2d):
    """Build a closed vertical wall (PolyData of quads) from a 2-D ring, |z|<=H."""
    P = np.asarray(ring2d)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])
    M = len(P)
    top = np.column_stack([P[:, 0], P[:, 1], np.full(M, H)])
    bot = np.column_stack([P[:, 0], P[:, 1], np.full(M, -H)])
    pts = np.vstack([top, bot])
    faces = []
    for i in range(M - 1):
        faces += [4, i, i + 1, M + i + 1, M + i]
    return pv.PolyData(pts, faces=np.array(faces))


def make_geometry(model, level):
    """True wall (sharp), neural reconstruction wall (displaced), per-point error / L."""
    V = koch.snowflake_vertices(level)
    P = V[:-1]
    _, nrm = K.koch_sdf_and_normal(P, level)            # exact outward unit normals
    # sanitize degenerate normals (fall back to the radial direction)
    bad = ~np.isfinite(nrm).all(1)
    if bad.any():
        rad = P[bad] / np.clip(np.linalg.norm(P[bad], axis=1, keepdims=True), 1e-9, None)
        nrm[bad] = rad
    p3 = np.column_stack([P, np.zeros(len(P))]).astype(np.float32)
    with torch.no_grad():
        phi = model(torch.tensor(p3)).reshape(-1).numpy()
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    L = K.body_size(level)
    err = np.clip(np.abs(phi) / L, 0.0, 1.0)
    recon = P - phi[:, None] * nrm                      # move onto the neural zero-set
    recon = np.where(np.isfinite(recon), recon, P)
    true_wall = _extrude_ring(P)
    recon_wall = _extrude_ring(recon)
    recon_wall["err"] = np.concatenate([err, err[:1], err, err[:1]])  # top ring + bottom ring
    # sharp true outline at the top rim
    top = np.column_stack([V[:, 0], V[:, 1], np.full(len(V), H)])
    outline = pv.lines_from_points(top).tube(radius=0.011)
    # ceiling metrics on random ON-boundary points (paper convention): RMS|phi|/L and the
    # median normal-angle error (neural SDF gradient vs the exact outward normal).
    xb = K.sample_on_boundary(level, 1600, 4242 + level)
    _, na = K.koch_sdf_and_normal(xb, level)
    na = np.where(np.isfinite(na), na, 0.0)
    xb3 = torch.tensor(np.column_stack([xb, np.zeros(len(xb))]), dtype=DT, requires_grad=True)
    phib = model(xb3)
    grad = torch.autograd.grad(phib.sum(), xb3)[0].detach().numpy()[:, :2]
    nn2 = grad / np.clip(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9, None)
    cos = np.clip(np.sum(nn2 * na, axis=1), -1.0, 1.0)
    ang_med = float(np.median(np.degrees(np.arccos(cos))))
    bdev = float(np.sqrt(np.mean(phib.detach().numpy() ** 2)) / L)
    return true_wall, recon_wall, outline, bdev, ang_med, float(np.percentile(err, 99))


def _trim(img, bg=245):
    mask = (img[:, :, :3] < bg).any(2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    pad = 6
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def render_panel(recon_wall, outline, emax, size=(620, 700)):
    pl = pv.Plotter(off_screen=True, window_size=size, border=False, lighting="none")
    pl.set_background("white")
    pl.add_mesh(recon_wall, scalars="err", cmap=CMAP, clim=(0, emax), smooth_shading=True,
                specular=0.32, specular_power=20, ambient=0.30, diffuse=0.9,
                show_scalar_bar=False)
    pl.add_mesh(outline, color="black", smooth_shading=True, show_scalar_bar=False)
    pl.add_light(pv.Light(position=(4, -3, 5), focal_point=(0, 0, 0), intensity=0.85))
    pl.add_light(pv.Light(position=(-4, -4, 2), focal_point=(0, 0, 0), intensity=0.40))
    pl.add_light(pv.Light(position=(0, 5, 3), focal_point=(0, 0, 0), intensity=0.30))
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(3.2, -3.4, 2.6), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(1.45)
    img = pl.screenshot(None, return_img=True, scale=2)
    pl.close()
    return _trim(img)


def build_figure(levels=(1, 2, 3, 4, 5), out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style(fontsize=9.0)

    out = out or os.path.join(FIG, "koch_neural_ceiling_pub.png")
    geos = []
    for n in levels:
        model = train_level_fast(n)
        geos.append((n,) + make_geometry(model, n))
    emax = max(g[6] for g in geos)
    emax = float(np.ceil(emax * 20) / 20)               # round clim for a tidy colorbar

    imgs = [(n, render_panel(rw, ol, emax), bdev, ang)
            for (n, tw, rw, ol, bdev, ang, _e) in geos]

    fig, axes = plt.subplots(1, len(levels), figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.30))
    for ax, (n, img, bdev, ang) in zip(np.atleast_1d(axes), imgs):
        ax.imshow(img)
        ax.set_title(rf"$n={n}$", fontsize=10, pad=2)
        ax.text(0.5, -0.02, rf"normal err ${ang:.0f}^\circ$", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="0.2")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, emax))
    cbar = fig.colorbar(sm, ax=np.atleast_1d(axes), orientation="horizontal",
                        fraction=0.05, pad=0.10, aspect=45)
    cbar.set_label(r"neural-SDF reconstruction error $|\phi_{\mathrm{nn}}|/L$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", out)
    return out


if __name__ == "__main__":
    if os.environ.get("KOCH3D_TEST"):
        build_figure(levels=(2, 5), out=os.path.join(FIG, "_koch3d_test.png"))
    else:
        build_figure()
