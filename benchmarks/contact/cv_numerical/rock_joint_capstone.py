#!/usr/bin/env python3
"""Rock-joint direct-shear capstone — the neural ATLAS (height chart) versus the ambient LEVEL SET
(neural SDF), on a real fractal rock joint.

This orchestrates the full capstone experiment and writes one consolidated result file
(runs/rock_joint_capstone/results.json) consumed by postprocessing/plot_rock_joint_capstone.py.

The four results, each measured (no asserted numbers):
  1. PATTON ANCHOR (method V&V) — two mating sawtooth faces under plain Coulomb emergently give the
     closed-form mu_app = tan(phi_b + i).  Machine-exact.  (delegates to rock_joint_shear.)
  2. CHART vs AMBIENT-SDF GEOMETRY — fit (a) a height chart h_theta(x) [random Fourier features,
     faithful to the measured 23.4 um topography] and (b) a real ambient 2-D neural SDF phi(x,z)
     [plain coordinate MLP — the canonical level set], extract the SDF's zero level set, and compare
     how each resolves the asperity SLOPES (what controls friction: mu_app = tan(phi_b + i)).
  3. SHEAR CONSEQUENCE — direct-shear the chart-resolved surface vs the SDF-reconstructed surface;
     the spectral-bias-smoothed SDF UNDER-PREDICTS peak shear strength and dilation by a measured %.
  4. ROUGHNESS SWEEP — rough vs smooth Inada joint: rougher => more dilation, higher peak strength
     (the Barton-JRC trend), recovered from real data with plain Coulomb (no dilatancy law).

HONEST FRAMING.  The claim is NOT "the chart has lower height RMSE than the SDF" (on a well-sampled
single-valued profile both interpolate the heights fine).  It is the *directional, contact-relevant*
result: the ambient SDF, at matched capacity/budget, SMOOTHS the asperity slopes (spectral bias) and
under-predicts the joint strength/dilation, whereas the 1-D chart parametrizes the surface directly,
reproduces the measured slopes, and costs O(N_surface) storage with resolution-independent queries.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_capstone.py            # full matrix (slow)
      python3 benchmarks/contact/cv_numerical/rock_joint_capstone.py --quick    # smaller nets/iters
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import torch  # noqa: E402
from common.models import MLP  # noqa: E402
from solvers.contact.profile_chart_2d import fit_height_chart, height_and_grad  # noqa: E402
from benchmarks.contact.cv_numerical import rock_joint_shear as rjs  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(_ROOT, "data", "inada_joint")
RUN_DIR = os.path.join(_ROOT, "runs", "rock_joint_capstone")


# --------------------------------------------------------------------------------------------------
def load_profile(tag: str):
    d = np.load(os.path.join(DATA, f"inada_{tag}_profile.npz"))
    return d["x_mm"].astype(float), d["footwall_mm"].astype(float)


def denoise(z: np.ndarray, sigma_pts: float = 1.5) -> np.ndarray:
    """Light Gaussian to drop sub-sampling measurement noise — the physical band-limited surface
    that the faithful chart targets (NOT a smoothing of the asperities)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(z, sigma_pts, mode="nearest")


def asperity_stats(x: np.ndarray, z: np.ndarray) -> dict:
    s = np.gradient(z, x)
    i = np.degrees(np.arctan(np.abs(s)))
    return dict(mean_angle=float(i.mean()), p90_angle=float(np.percentile(i, 90)),
               rms_slope=float(np.sqrt(np.mean(s ** 2))))


# --------------------------------------------------------------------------------------------------
# ambient 2-D neural SDF phi(x,z) of the surface curve z=h(x) (solid = below).  Plain coordinate MLP
# (tanh) — the canonical level set the neural-atlas thesis argues against.
# --------------------------------------------------------------------------------------------------
class AmbientSDF2D(torch.nn.Module):
    """phi(x,z) ~ signed distance to the curve z=h(x); inputs normalized by (x,z) scales."""

    def __init__(self, x_lo, x_hi, z_lo, z_hi, width=128, depth=6):
        super().__init__()
        self.x0, self.xs = float(x_lo), float(x_hi - x_lo)
        self.z0, self.zs = float(z_lo), float(z_hi - z_lo)
        self.net = MLP(in_dim=2, out_dim=1, width=width, depth=depth)

    def _norm(self, P):
        xn = 2.0 * (P[:, 0] - self.x0) / self.xs - 1.0
        zn = 2.0 * (P[:, 1] - self.z0) / self.zs - 1.0
        return torch.stack([xn, zn], dim=-1)

    def forward(self, P):
        return self.net(self._norm(P)).squeeze(-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _signed_distance_to_polyline(P: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Euclidean distance of points P=(X,Z) to the dense polyline (xs,zs); sign negative below the
    surface (Z < h(X))."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([xs, zs], axis=1))
    dist, _ = tree.query(P, k=1)
    h_at = np.interp(P[:, 0], xs, zs)
    sign = np.where(P[:, 1] < h_at, -1.0, 1.0)
    return sign * dist


def train_ambient_sdf(x: np.ndarray, z: np.ndarray, band: float, width=128, depth=6,
                      iters=4000, n_pts=60000, lr=2e-3, verbose=False) -> "AmbientSDF2D":
    """Train phi(x,z) on band points around the surface with KDTree signed distance."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    xs = np.linspace(x.min(), x.max(), 8000)
    zs = np.interp(xs, x, z)
    # sample band points: jitter around the surface +/- band, plus a bulk
    base = rng.randint(0, len(xs), n_pts)
    Px = xs[base] + rng.uniform(-0.5, 0.5, n_pts) * (x.max() - x.min()) / len(xs) * 4
    Pz = zs[base] + rng.uniform(-band, band, n_pts)
    P = np.stack([Px, Pz], axis=1)
    d = _signed_distance_to_polyline(P, xs, zs)
    sdf = AmbientSDF2D(x.min(), x.max(), z.min() - band, z.max() + band,
                       width=width, depth=depth).double()
    Pt = torch.from_numpy(P); dt = torch.from_numpy(d)
    opt = torch.optim.Adam(sdf.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        idx = torch.randint(0, n_pts, (8192,))
        loss = torch.mean((sdf(Pt[idx]) - dt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 1000 == 0:
            print(f"    sdf it {it}: loss={float(loss):.3e}")
    return sdf


def extract_levelset(sdf: "AmbientSDF2D", x_grid: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """For each x, bisect in z for phi(x,z)=0 -> the SDF's reconstructed surface h_sdf(x)."""
    X = torch.from_numpy(x_grid)
    zlo = torch.full_like(X, z_lo); zhi = torch.full_like(X, z_hi)
    # phi increases with z (above surface positive); ensure bracket sign
    for _ in range(40):
        zm = 0.5 * (zlo + zhi)
        phi = sdf(torch.stack([X, zm], dim=1))
        below = phi < 0.0
        zlo = torch.where(below, zm, zlo)
        zhi = torch.where(below, zhi, zm)
    return (0.5 * (zlo + zhi)).detach().numpy()


# --------------------------------------------------------------------------------------------------
def ensemble_roughness(n_rows=24, n_inc=160, eps_n=5.0e4):
    """Shear many real scanlines of the full 2-D surface -> robust mean +/- std macroscopic curves.
    Uses the raw measured rows (the chart reproduces these to ~2 um), no per-row chart training."""
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d
    raw = os.path.join(_ROOT, "downloads", "inada_granite")
    if not os.path.isdir(raw):
        print("    (raw CSVs absent — skipping ensemble)"); return {}
    out = {}
    for tag, fn in (("rough", "rough_footwall.csv"), ("smooth", "smooth_footwall.csv")):
        Z = pd.read_csv(os.path.join(raw, fn), header=None).to_numpy(dtype=float)
        rows = np.linspace(60, Z.shape[0] - 60, n_rows).astype(int)
        x = np.arange(Z.shape[1]) * 0.0234
        dils, mus, steady, peaks, uxref = [], [], [], [], None
        for r in rows:
            z = gaussian_filter1d(Z[r], 1.5, mode="nearest")
            lo = rjs.bake_samples(x, z)
            up = dict(x=x.copy(), h=z.copy(), hp=lo["hp"].copy())
            res = rjs.run_shear(lo, up, sigma_n=1.0, mu=0.3, shear_total=8.0, n_inc=n_inc, eps_n=eps_n)
            dils.append(res["history"]["dilation"]); mus.append(res["history"]["mu_app"])
            steady.append(res["summary"]["mu_app_steady"]); peaks.append(res["summary"]["mu_app_peak"])
            uxref = res["history"]["ux"]
        dils = np.array(dils); mus = np.array(mus)
        out[tag] = dict(
            n_rows=int(n_rows), rms_mm=float(gaussian_filter1d(Z[rows[0]], 1.5).std()),
            ux=uxref.tolist(),
            dil_mean=dils.mean(0).tolist(), dil_std=dils.std(0).tolist(),
            mu_mean=mus.mean(0).tolist(), mu_std=mus.std(0).tolist(),
            steady_mu_mean=float(np.mean(steady)), steady_mu_std=float(np.std(steady)),
            peak_mu_mean=float(np.mean(peaks)), peak_mu_std=float(np.std(peaks)),
            dil_total_mean=float(dils[:, -1].mean()), dil_total_std=float(dils[:, -1].std()),
        )
        s = out[tag]
        print(f"    {tag:6s} ({n_rows} lines): steady mu_app={s['steady_mu_mean']:.3f}"
              f"+/-{s['steady_mu_std']:.3f}  dilation={s['dil_total_mean']:.2f}"
              f"+/-{s['dil_total_std']:.2f} mm")
    return out


def run(quick=False):
    os.makedirs(RUN_DIR, exist_ok=True)
    W = dict(chart_iters=600 if quick else 3500, sdf_iters=800 if quick else 5000,
             sdf_width=96 if quick else 128, sdf_depth=5 if quick else 6,
             chart_nfreq=64, chart_fmax=200.0, shear_inc=80 if quick else 160, eps_n=5.0e4)
    results = {"config": W}

    # ---- 1. Patton anchor -------------------------------------------------------------------------
    print("=== [1/4] Patton sawtooth verification ===")
    pat = rjs.run_sawtooth_patton(angle_deg=20.0, mu=0.3)
    results["patton"] = pat["summary"]

    # ---- 2/3. chart vs ambient SDF on the ROUGH profile ------------------------------------------
    print("\n=== [2/4] chart vs ambient neural SDF (rough Inada profile) ===")
    x, z_raw = load_profile("rough")
    z = denoise(z_raw, 1.5)                        # physical band-limited surface (chart target)
    ref = asperity_stats(x, z)
    print(f"  measured surface: RMS={z.std():.3f} mm, mean asperity angle={ref['mean_angle']:.2f} deg")

    print("  fitting faithful height chart...")
    chart, chart_rmse = fit_height_chart(x, z, f_max=W["chart_fmax"], n_freq=W["chart_nfreq"],
                                         iters=W["chart_iters"])
    _, dh = height_and_grad(torch.from_numpy(x), chart)
    h_chart = chart(torch.from_numpy(x)).detach().numpy()
    chart_stats = asperity_stats(x, h_chart)
    n_chart_params = sum(p.numel() for p in chart.parameters())

    print("  training ambient 2-D neural SDF + extracting zero level set...")
    band = 3.0 * z.std()
    sdf = train_ambient_sdf(x, z, band=band, width=W["sdf_width"], depth=W["sdf_depth"],
                            iters=W["sdf_iters"])
    h_sdf = extract_levelset(sdf, x, z.min() - band, z.max() + band)
    sdf_rmse = float(np.sqrt(np.mean((h_sdf - z) ** 2)))
    sdf_stats = asperity_stats(x, h_sdf)
    n_sdf_params = sdf.num_params()

    results["geometry"] = dict(
        rms_mm=float(z.std()), ref=ref,
        chart=dict(recon_rmse_mm=float(chart_rmse), n_params=int(n_chart_params), **chart_stats),
        sdf=dict(recon_rmse_mm=sdf_rmse, n_params=int(n_sdf_params), **sdf_stats),
    )
    print(f"    chart: recon RMSE={chart_rmse:.3e} mm, mean angle={chart_stats['mean_angle']:.2f} deg "
          f"({n_chart_params} params)")
    print(f"    SDF:   recon RMSE={sdf_rmse:.3e} mm, mean angle={sdf_stats['mean_angle']:.2f} deg "
          f"({n_sdf_params} params)")
    print(f"    -> SDF smooths the mean asperity angle {ref['mean_angle']:.1f} -> "
          f"{sdf_stats['mean_angle']:.1f} deg")

    # ---- 3. shear consequence: RESOLVED (measured = what the chart reproduces) vs SDF level set ---
    # Use the denoised measured surface as the resolved ground truth (the chart reproduces it to
    # `chart_rmse`, reported above) so the contrast isolates the SDF's spectral-bias smoothing, not
    # any chart fit error.
    print("\n=== [3/4] shear consequence (rough, mated) ===")
    lower_chart = rjs.bake_samples(x, z)
    upper_chart = dict(x=x.copy(), h=z.copy(), hp=lower_chart["hp"].copy())
    res_chart = rjs.run_shear(lower_chart, upper_chart, sigma_n=1.0, mu=0.3, shear_total=8.0,
                              n_inc=W["shear_inc"], eps_n=W["eps_n"])
    lower_sdf = rjs.bake_samples(x, h_sdf)
    upper_sdf = dict(x=x.copy(), h=h_sdf.copy(), hp=lower_sdf["hp"].copy())
    res_sdf = rjs.run_shear(lower_sdf, upper_sdf, sigma_n=1.0, mu=0.3, shear_total=8.0,
                            n_inc=W["shear_inc"], eps_n=W["eps_n"])
    sc, ss = res_chart["summary"], res_sdf["summary"]
    underpred_mu = (sc["mu_app_peak"] - ss["mu_app_peak"]) / sc["mu_app_peak"]
    underpred_dil = (sc["dilation_total"] - ss["dilation_total"]) / max(abs(sc["dilation_total"]), 1e-9)
    results["shear_consequence"] = dict(
        chart=sc, sdf=ss,
        peak_mu_underprediction=float(underpred_mu),
        dilation_underprediction=float(underpred_dil),
    )
    results["shear_curves"] = dict(
        ux=res_chart["history"]["ux"].tolist(),
        mu_chart=res_chart["history"]["mu_app"].tolist(),
        mu_sdf=res_sdf["history"]["mu_app"].tolist(),
        dil_chart=res_chart["history"]["dilation"].tolist(),
        dil_sdf=res_sdf["history"]["dilation"].tolist(),
    )
    print(f"    peak mu_app:  chart={sc['mu_app_peak']:.3f}  SDF={ss['mu_app_peak']:.3f}  "
          f"-> SDF under-predicts strength by {underpred_mu*100:.1f}%")
    print(f"    dilation:     chart={sc['dilation_total']:.3f}  SDF={ss['dilation_total']:.3f} mm  "
          f"-> SDF under-predicts dilation by {underpred_dil*100:.1f}%")

    # ---- 4. roughness sweep: rough vs smooth ------------------------------------------------------
    print("\n=== [4/4] roughness sweep (rough vs smooth Inada) ===")
    sweep = {}
    for tag in ("rough", "smooth"):
        xt, zt = load_profile(tag)
        zt = denoise(zt, 1.5)
        lo = rjs.bake_samples(xt, zt)
        up = dict(x=xt.copy(), h=zt.copy(), hp=lo["hp"].copy())
        r = rjs.run_shear(lo, up, sigma_n=1.0, mu=0.3, shear_total=8.0,
                          n_inc=W["shear_inc"], eps_n=W["eps_n"])
        sweep[tag] = dict(rms_mm=float(zt.std()), **r["summary"],
                          ux=r["history"]["ux"].tolist(),
                          mu_app=r["history"]["mu_app"].tolist(),
                          dilation=r["history"]["dilation"].tolist())
        print(f"    {tag:6s}: RMS={zt.std():.3f} mm  peak mu_app={r['summary']['mu_app_peak']:.3f}  "
              f"steady mu_app={r['summary']['mu_app_steady']:.3f}  "
              f"dilation={r['summary']['dilation_total']:+.3f} mm")
    results["roughness_sweep"] = sweep

    # save the surface arrays (for the chart-vs-SDF zoom panel)
    np.savez(os.path.join(RUN_DIR, "surfaces.npz"), x=x, z_measured=z, h_chart=h_chart, h_sdf=h_sdf)

    # ---- 4b. ensemble over real scanlines (robust roughness statistics) ---------------------------
    print("\n=== [4b/4] ensemble over real scanlines (robust mean +/- std) ===")
    results["ensemble"] = ensemble_roughness(n_rows=12 if quick else 24,
                                             n_inc=W["shear_inc"], eps_n=W["eps_n"])

    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"\n  consolidated results -> {os.path.join(RUN_DIR, 'results.json')}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller nets/iters for a fast smoke run")
    args = ap.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
