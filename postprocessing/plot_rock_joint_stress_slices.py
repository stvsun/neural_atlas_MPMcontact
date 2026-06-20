"""Cross-section stress slices through the deformable cyclic rock-joint FEM.

For the two-block deformable joint under cyclic shear (rock_joint_cyclic_fem.py), this extracts the
in-block stress field at several TIME snapshots and plots, on x-z cross-section SLICES taken at the
FRONT, MID-section and BACK (three y-planes):

  * mean pressure         p   = -(s1 + s2 + s3)/3        (compression positive)
  * principal differences (s1 - s3), (s2 - s3), (s1 - s2)   with s1 >= s2 >= s3

The interface is driven by a spatially-varying (rough) dilation-angle field i0(x,y), so the three
slices carry genuinely different stress states (a uniform joint would make them identical).

Output: figures/rock_joint_stress_slice_{front,mid,back}.png  (rows = the four stress measures,
columns = time snapshots; the dashed line marks the joint plane).

Run:  python3 postprocessing/plot_rock_joint_stress_slices.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
from benchmarks.contact.cv_numerical.rock_joint_cyclic_fem import JointFEM, MODES  # noqa: E402
FIG = os.path.join(_ROOT, "figures")
torch.set_default_dtype(torch.float64)


_ROUGH_SEED = 0


def _rough_f(x, y, seed=_ROUGH_SEED):
    """Normalized rough field f(x,y) in [-1,1] (shared by the dilation angle AND the relief height)."""
    rng = np.random.RandomState(seed)
    f = np.zeros(np.shape(x))
    for kx, ky, ph in zip(rng.uniform(0.6, 2.5, 8), rng.uniform(0.6, 2.5, 8),
                          rng.uniform(0, 2 * np.pi, 8)):
        f = f + np.sin(2 * np.pi * (kx * x + ky * y) + ph)
    return f / np.abs(f).max()


def rough_dilation_field(xy: np.ndarray, base_deg=14.0, amp_deg=10.0, seed=_ROUGH_SEED) -> np.ndarray:
    """Spatially-varying asperity (dilation) angle on the interface nodes -> heterogeneous joint."""
    return np.radians(np.clip(base_deg + amp_deg * _rough_f(xy[:, 0], xy[:, 1], seed), 1.0, 38.0))


ASPERITY_AMP = 0.22                                    # asperity relief amplitude (rel. to half=1)


def rough_relief_grid(half, n=46, asperity_amp=ASPERITY_AMP):
    """Asperity relief surface z=h(x,y) (same rough field) at the joint plane, for the 3-D side view."""
    g = np.linspace(-half, half, n)
    Xg, Yg = np.meshgrid(g, g, indexing="ij")
    Zg = half + asperity_amp * _rough_f(Xg, Yg)
    return Xg, Yg, Zg


def drape_z(x, y, z, half, amp=ASPERITY_AMP, seed=_ROUGH_SEED):
    """Map flat-block z to the PHYSICAL (rough) domain: drape the joint-surface chart h(x,y) onto the
    interface and ramp it to zero at the fixed outer faces (z=-half lower bottom, z=3*half upper top).
    Continuous across the joint (both ramps -> 1 at z=half), so the two faces mate at the rough surface.
    NOTE: the stress was computed on the flat interface-element model; this warp is the geometry chart
    applied for display."""
    z = np.asarray(z, float)
    f = _rough_f(np.asarray(x, float), np.asarray(y, float), seed)
    w = np.where(z <= half, (z + half) / (2 * half), (3 * half - z) / (2 * half))
    return z + amp * f * np.clip(w, 0.0, 1.0)


def principal_fields(solver, u_block: np.ndarray, z_shift: float, stress_fn):
    """Per-element centroid (x,y,z world) + ordered principal stresses s1>=s2>=s3."""
    NL = solver.n_nodes
    F = solver.compute_F(torch.from_numpy(u_block.reshape(NL, 3)))
    sig = stress_fn(F).detach().numpy()                       # (M,3,3) = Cauchy (small strain)
    w = np.linalg.eigvalsh(0.5 * (sig + np.transpose(sig, (0, 2, 1))))  # ascending: s3,s2,s1
    c = solver.elem_centroids_phys.detach().numpy().copy(); c[:, 2] += z_shift
    return c, w[:, 2], w[:, 1], w[:, 0]                        # s1, s2, s3


def run_snapshots(mode="in_plane", n_cycles=1, amplitude=0.05, sigma_n0=4.0, n_per_quarter=8,
                  n_cells=8, E=2.0e3, nu=0.25, mu=0.4, n_snap=5):
    d = np.asarray(MODES[mode], float)
    fem = JointFEM(n_cells=n_cells, half=1.0, E=E, nu=nu, mu=mu, dilation_field=None)
    fem.i0 = rough_dilation_field(fem.xy); fem.tan_i0 = np.tan(fem.i0)   # heterogeneous roughness
    K = fem._Kblocks(); offU = 3 * fem.NL; uz0 = -sigma_n0 / fem.k_n
    segs = []
    for _ in range(n_cycles):
        segs += [np.linspace(0, amplitude, n_per_quarter, endpoint=False),
                 np.linspace(amplitude, -amplitude, 2 * n_per_quarter, endpoint=False),
                 np.linspace(-amplitude, 0, n_per_quarter, endpoint=False)]
    sched = np.concatenate(segs + [np.array([0.0])])
    snap_at = sorted(set(np.linspace(n_per_quarter, len(sched) - 1, n_snap).astype(int)))
    u = np.zeros(fem.ndof); snaps = []
    for j, umag in enumerate(sched):
        u, diag, _ = fem._solve_fixed_top((umag * d[0], umag * d[1], uz0), K, u, max_iter=120,
                                          commit=True)
        if j in snap_at:
            cL, s1L, s2L, s3L = principal_fields(fem.lo, u[:offU], 0.0, fem.stress_fn)
            cU, s1U, s2U, s3U = principal_fields(fem.up, u[offU:], fem.shift, fem.stress_fn)
            snaps.append(dict(u_par=float(umag),
                              c=np.vstack([cL, cU]), s1=np.concatenate([s1L, s1U]),
                              s2=np.concatenate([s2L, s2U]), s3=np.concatenate([s3L, s3U])))
            print(f"    snapshot j={j} u_par={umag:+.4f}  ({len(snaps)}/{len(snap_at)})")
    return fem, snaps


def plot_slice(snaps, y0, tol, half, title, fname):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style
    set_pub_style()
    quants = [("mean pressure $p$", lambda s: -(s["s1"] + s["s2"] + s["s3"]) / 3.0, "viridis"),
              ("$\\sigma_1-\\sigma_3$", lambda s: s["s1"] - s["s3"], "magma"),
              ("$\\sigma_2-\\sigma_3$", lambda s: s["s2"] - s["s3"], "magma"),
              ("$\\sigma_1-\\sigma_2$", lambda s: s["s1"] - s["s2"], "magma")]
    nT = len(snaps)
    fig, axs = plt.subplots(len(quants), nT, figsize=(2.0 * nT + 1.2, 2.0 * len(quants)),
                            squeeze=False)
    for r, (qname, qfn, cmap) in enumerate(quants):
        # consistent color scale per quantity across times
        vals_all = []
        for s in snaps:
            m = np.abs(s["c"][:, 1] - y0) < tol
            vals_all.append(qfn(s)[m])
        vcat = np.concatenate(vals_all) if vals_all else np.array([0.0, 1.0])
        vmin, vmax = np.percentile(vcat, 2), np.percentile(vcat, 98)
        if qname.startswith("mean"):
            vmin = min(vmin, 0.0)
        for cidx, s in enumerate(snaps):
            ax = axs[r][cidx]
            m = np.abs(s["c"][:, 1] - y0) < tol
            cx = s["c"][m, 0]
            cz = drape_z(cx, s["c"][m, 1], s["c"][m, 2], half)   # -> physical (rough) domain
            v = qfn(s)[m]
            try:
                tcf = ax.tricontourf(cx, cz, v, levels=14, cmap=cmap, vmin=vmin, vmax=vmax)
            except Exception:
                tcf = ax.scatter(cx, cz, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
            # the rough joint surface profile at this y-slice (chart h(x, y0))
            xl = np.linspace(-half, half, 200)
            ax.plot(xl, half + ASPERITY_AMP * _rough_f(xl, np.full_like(xl, y0)), color="w", lw=0.9)
            ax.set_xlim(-half, half); ax.set_ylim(-half, 3 * half)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$u$={s['u_par']:+.3f} mm", fontsize=8)
            if cidx == 0:
                ax.set_ylabel(qname, fontsize=8)
            if cidx == nT - 1:
                plt.colorbar(tcf, ax=axs[r], fraction=0.046, pad=0.02)
    fig.suptitle(title, y=1.005, fontsize=10)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, fname); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  saved", out)


def side_view_3d(snaps, half, fname, n_cells=8):
    """3-D SIDE VIEW: the rough joint surface relief + the three slice planes (front/mid/back) cut
    through the block volume, each colored by the four stress measures at peak-forward shear.
    Shows the geometry of the slices on the rough 2-D surface."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata
    Xs, Ys, Zs = rough_relief_grid(half)
    slices = [(-0.8 * half, "front"), (0.0, "mid"), (0.8 * half, "back")]
    quants = [("mean pressure $p$", lambda s: -(s["s1"] + s["s2"] + s["s3"]) / 3.0, "viridis"),
              ("$\\sigma_1-\\sigma_3$", lambda s: s["s1"] - s["s3"], "magma"),
              ("$\\sigma_2-\\sigma_3$", lambda s: s["s2"] - s["s3"], "magma"),
              ("$\\sigma_1-\\sigma_2$", lambda s: s["s1"] - s["s2"], "magma")]
    snap = max(snaps, key=lambda s: abs(s["u_par"]))           # peak-shear snapshot
    tol = max(0.9 * half / n_cells, half / 6)
    xg = np.linspace(-half, half, 30); zg = np.linspace(-half, 3 * half, 60)
    XX, ZZ = np.meshgrid(xg, zg, indexing="ij")
    fig = plt.figure(figsize=(13, 3.4))
    for q, (qname, qfn, cmap) in enumerate(quants):
        ax = fig.add_subplot(1, 4, q + 1, projection="3d")
        v_all = qfn(snap)
        vmin, vmax = np.percentile(v_all, 3), np.percentile(v_all, 97)
        if qname.startswith("mean"):
            vmin = min(vmin, 0.0)
        norm = Normalize(vmin, vmax); cmo = cm.get_cmap(cmap)
        # rough joint surface (geometry)
        ax.plot_surface(Xs, Ys, Zs, color="0.6", alpha=0.32, linewidth=0, antialiased=True,
                        rstride=2, cstride=2, zorder=1)
        # three slice planes colored by the stress quantity, WARPED to the physical (rough) domain
        for y0, nm in slices:
            m = np.abs(snap["c"][:, 1] - y0) < tol
            Vg = griddata((snap["c"][m, 0], snap["c"][m, 2]), v_all[m], (XX, ZZ),
                          method="linear")
            colors = cmo(norm(Vg)); colors[..., 3] = np.where(np.isfinite(Vg), 0.93, 0.0)
            YY = np.full_like(XX, y0)
            ZW = drape_z(XX, YY, ZZ, half)                       # chart h(x,y) -> physical geometry
            ax.plot_surface(XX, YY, ZW, facecolors=colors, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, shade=False, zorder=3)
            ax.text(half, y0, 3 * half, nm, fontsize=6, color="k")
        ax.set_title(qname + f"\n($u$={snap['u_par']:+.3f} mm)", fontsize=8)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
        ax.set_box_aspect((1, 1, 2)); ax.view_init(elev=12, azim=-72)
        ax.tick_params(labelsize=5)
        m_ = cm.ScalarMappable(norm=norm, cmap=cmo)
        fig.colorbar(m_, ax=ax, fraction=0.03, pad=0.08, shrink=0.6)
    fig.suptitle("Side view — slices mapped to the PHYSICAL (rough) domain via the joint chart "
                 "$h(x,y)$; front / mid / back. Stress from the flat interface-element model.",
                 y=1.02, fontsize=9)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, fname); fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
    print("  saved", out)


def main():
    print("=== running cyclic FEM + capturing stress snapshots ===")
    fem, snaps = run_snapshots(mode="in_plane", n_cycles=1, n_cells=8, n_snap=5)
    half = fem.half; tol = 0.9 * half / 8
    for y0, name in ((-0.8 * half, "front"), (0.0, "mid"), (0.8 * half, "back")):
        plot_slice(snaps, y0, tol=max(tol, half / 6), half=half,
                   title=f"Rock-joint cross-section stress — {name.upper()} slice (y={y0:+.2f}), "
                         f"physical (rough) domain via chart $h(x,y)$; cyclic in-plane shear",
                   fname=f"rock_joint_stress_slice_{name}.png")
    side_view_3d(snaps, half, "rock_joint_stress_slice_sideview.png")


if __name__ == "__main__":
    main()
