#!/usr/bin/env python3
"""CV-7: the GENUINE atlas rough-contact on a REAL Inada-granite surface patch (not synthetic).

§11.11's genuine decoder-FEM shear used a synthetic band-limited surface (for det J safety).  This
connects it to the actual rock joint: a real Inada-granite topography patch (Digital Rocks #273) is
extracted, resolved at the mesh scale (down-sampled + amplitude-normalised to keep det J > 0), used to
train the boundary-fitted decoder, and sheared — then compared to the ambient SDF on the SAME real
patch.  Self-contained (new runs/cv7_real_inada/ + figure); does not touch other CV-7 files.

Run:  python3 benchmarks/contact/cv_numerical/cv7_real_inada_decoder.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.rough_block_decoder import train_rough_decoder                # noqa: E402
from benchmarks.contact.cv_numerical.rock_joint_decoder_shear import DecoderJointShear  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(_ROOT, "downloads", "inada_granite")
RUN = os.path.join(_ROOT, "runs", "cv7_real_inada")
TARGET_RMS = 0.06          # amplitude the n_cells mesh resolves with det J > 0 (honest scaling)
L = 1.0


def real_inada_surface(patch_n=64, every=10, row0=600, col0=600, target_rms=TARGET_RMS):
    """A REAL Inada-granite topography patch mapped to [-L,L]^2, resolved at the mesh scale."""
    import pandas as pd
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    Z = pd.read_csv(os.path.join(RAW, "rough_footwall.csv"), header=None).to_numpy(float)
    p = Z[row0:row0 + patch_n * every:every, col0:col0 + patch_n * every:every]
    p = gaussian_filter(p, 0.6)                                    # mild denoise of sub-sample noise
    p = p - p.mean()
    p = p / (p.std() + 1e-12) * target_rms                        # normalise amplitude (resolvable)
    g = np.linspace(-L, L, p.shape[0]); g2 = np.linspace(-L, L, p.shape[1])
    itp = RegularGridInterpolator((g, g2), p, bounds_error=False, fill_value=None)
    fn = lambda x, y: itp(np.stack([np.clip(x, -L, L), np.clip(y, -L, L)], -1))   # noqa: E731
    return fn, float(np.std(p))


def shear_curve(dec, dk, surf_fn, mu, n_cells=12, eps_n=2.0e4, shear_total=0.15, n_inc=10):
    js = DecoderJointShear(dec, dk, n_cells=n_cells, mu=mu, surf_amp=TARGET_RMS, eps_n=eps_n)
    js._surf_fn = surf_fn
    u = np.zeros(3 * js.N); z_p = 1.0 - 0.5 * TARGET_RMS
    mu_app, resid = [], []
    for ux in np.linspace(0, shear_total, n_inc):
        u, diag = js.solve_fixed(ux, z_p, u, max_iter=140)
        Fn = js.normal_force(u, ux, z_p); U = u.reshape(js.N, 3)
        fc, _ = js.contact(U, ux, z_p); Fx = float(fc[js.top, 0].sum())
        mu_app.append(Fx / max(Fn, 1e-9)); resid.append(diag["resid_rel"])
    return np.array(mu_app), float(np.max(resid))


def main():
    if not os.path.isdir(RAW):
        print("  raw Inada CSVs absent — skip"); return
    os.makedirs(RUN, exist_ok=True)
    surf_fn, rms = real_inada_surface()
    print(f"=== REAL Inada patch resolved at mesh scale: RMS={rms:.3f} (normalised) ===")

    print("[atlas] train boundary-fitted decoder on the REAL surface")
    dec, rmse, dk = train_rough_decoder(lambda x, y: surf_fn(x, y), iters=4000)
    js = DecoderJointShear(dec, dk, n_cells=12, mu=0.3, surf_amp=TARGET_RMS, eps_n=2.0e4)
    valid = bool(js.solver.geom_valid.all()); detJmin = float(js.solver.geom_detJ.min())
    print(f"  reconstruction {rmse:.3e} ({rmse/rms*100:.1f}% of RMS); FEM valid={valid} detJ_min={detJmin:.3f}")

    print("[atlas] genuine shear on the REAL surface (frictionless + friction)")
    mu0, r0 = shear_curve(dec, dk, surf_fn, mu=0.0)
    mu3, r3 = shear_curve(dec, dk, surf_fn, mu=0.3)
    print(f"  frictionless peak mu_app={mu0.max():.3f} (resid {r0:.1e}); friction peak={mu3.max():.3f} (resid {r3:.1e})")

    # SDF (level set) on the SAME real surface
    print("[level set] ambient SDF on the REAL surface + extract level set, fit decoder, shear")
    from benchmarks.contact.cv_numerical.cv7_decoder_verify import train_ambient_sdf3d
    from benchmarks.contact.cv_numerical.cv7_atlas_vs_sdf_shear import sdf_levelset_fn
    sdf, sdf_rmse = train_ambient_sdf3d(lambda x, y: surf_fn(x, y), band=2.5 * TARGET_RMS)
    h_sdf = sdf_levelset_fn(sdf, band=2.5 * TARGET_RMS)
    dec_s, rmse_s, dk_s = train_rough_decoder(lambda x, y: h_sdf(x, y), iters=4000)
    smu0, _ = shear_curve(dec_s, dk_s, lambda x, y: h_sdf(x, y), mu=0.0)
    smu3, _ = shear_curve(dec_s, dk_s, lambda x, y: h_sdf(x, y), mu=0.3)
    print(f"  SDF recon {sdf_rmse:.3e} ({sdf_rmse/rms*100:.1f}%); frictionless peak={smu0.max():.3f}; friction peak={smu3.max():.3f}")

    ux = np.linspace(0, 0.15, 10)
    res = dict(surf_rms=rms, atlas_recon_pct=rmse / rms * 100, sdf_recon_pct=sdf_rmse / rms * 100,
               detJ_min=detJmin, all_valid=valid,
               atlas_frictionless_peak=float(mu0.max()), atlas_friction_peak=float(mu3.max()),
               sdf_frictionless_peak=float(smu0.max()), sdf_friction_peak=float(smu3.max()),
               max_resid=float(max(r0, r3)))
    json.dump(res, open(os.path.join(RUN, "results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.36))
    axs[0].plot(ux, mu0, "-o", ms=3, color="#0072B2", label="atlas (real surface)")
    axs[0].plot(ux, smu0, "-s", ms=3, color="#D55E00", label="level set (SDF)")
    axs[0].set_title("(a) frictionless dilatancy — REAL Inada patch", fontsize=8)
    axs[0].set_xlabel("shear $u_x$"); axs[0].set_ylabel("$\\mu_{app}$"); axs[0].legend(fontsize=6)
    axs[1].plot(ux, mu3, "-o", ms=3, color="#0072B2", label="atlas")
    axs[1].plot(ux, smu3, "-s", ms=3, color="#D55E00", label="level set")
    axs[1].set_title("(b) friction strength ($\\mu$=0.3) — REAL Inada patch", fontsize=8)
    axs[1].set_xlabel("shear $u_x$"); axs[1].set_ylabel("$\\mu_{app}$"); axs[1].legend(fontsize=6)
    fig.suptitle("CV-7 genuine atlas rough-contact on a REAL Inada-granite surface patch: atlas resolves "
                 "the asperities the level set smooths", y=1.03, fontsize=9)
    fig.tight_layout(); os.makedirs(os.path.join(_ROOT, "figures"), exist_ok=True)
    out = os.path.join(_ROOT, "figures", "cv7_real_inada_pub.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}\n  saved {os.path.join(RUN, 'results.json')}")


if __name__ == "__main__":
    main()
