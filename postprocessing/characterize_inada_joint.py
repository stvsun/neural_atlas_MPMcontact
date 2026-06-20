"""Characterize the Inada-granite rock-joint surfaces (Digital Rocks Portal #273) for the
neural-atlas direct-shear capstone.

Loads the surface-topography height maps z(x,y) (downloads/inada_granite/*.csv), reports grid
size / physical extent / RMS roughness, fits the radially-averaged power spectral density (PSD)
slope -> Hurst exponent H and fractal dimension D (cross-check vs the published D~2.4-2.5), and
exports a COMPACT committable artifact (a representative 1-D shear profile + the aperture stats)
so the capstone is reproducible without the ~55 MB raw CSVs.

Data: Sawayama, Jiang & Tsuji (2020), "Digitalized natural rock fracture of Inada Granite",
DOI 10.17612/QXSA-TK92 (ODC-BY). Sampling 23.4 um; tensile fractures; footwall + hangingwall pairs.

Run:  python3 postprocessing/characterize_inada_joint.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "downloads", "inada_granite")
DATA = os.path.join(_ROOT, "data", "inada_joint")          # compact committable artifacts
FIG = os.path.join(_ROOT, "figures")
DX_MM = 0.0234                                              # 23.4 um horizontal sampling


def load_height_map(name: str) -> np.ndarray:
    """Load a surface CSV as a 2-D height grid z[i,j] in mm (pandas for speed)."""
    import pandas as pd
    path = os.path.join(RAW, name)
    z = pd.read_csv(path, header=None).to_numpy(dtype=float)
    return z


def radial_psd_slope(z: np.ndarray):
    """Radially-averaged 2-D PSD slope beta (P(k) ~ k^-beta) -> Hurst H=(beta-2)/2 and surface
    fractal dimension D = 3 - H (self-affine surface convention)."""
    zc = z - z.mean()
    win = np.hanning(z.shape[0])[:, None] * np.hanning(z.shape[1])[None, :]   # reduce leakage
    F = np.fft.fftshift(np.fft.fft2(zc * win))
    P = np.abs(F) ** 2
    ny, nx = z.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=DX_MM))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=DX_MM))
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX ** 2 + KY ** 2).ravel()
    Pr = P.ravel()
    kmax = min(kx.max(), ky.max())
    bins = np.logspace(np.log10(2.0 / (DX_MM * max(ny, nx))), np.log10(kmax * 0.7), 40)
    idx = np.digitize(k, bins)
    kc, Pc = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 8:
            kc.append(np.sqrt(bins[b - 1] * bins[b])); Pc.append(Pr[m].mean())
    kc, Pc = np.array(kc), np.array(Pc)
    # fit the inertial range (drop the lowest 3 and highest 3 bins)
    sl = slice(3, len(kc) - 3)
    beta, logA = np.polyfit(np.log10(kc[sl]), np.log10(Pc[sl]), 1)
    beta = -beta
    H = np.clip((beta - 2.0) / 2.0, 0.0, 1.0)
    D = 3.0 - H
    return kc, Pc, sl, float(beta), float(H), float(D)


def characterize(tag: str):
    foot = load_height_map(f"{tag}_footwall.csv")
    hang = load_height_map(f"{tag}_hangingwall.csv")
    ny, nx = foot.shape
    Lx, Ly = nx * DX_MM, ny * DX_MM
    rms = float(foot.std())
    kc, Pc, sl, beta, H, D = radial_psd_slope(foot)
    # aperture = mating gap (hangingwall above footwall); both are z of the two faces
    ap = hang - foot
    print(f"  [{tag}] grid {ny} x {nx}  -> {Ly:.1f} x {Lx:.1f} mm @ {DX_MM*1000:.1f} um")
    print(f"        footwall: RMS height = {rms:.3f} mm,  z-range {foot.min():.2f}..{foot.max():.2f} mm")
    print(f"        PSD slope beta = {beta:.2f}  ->  Hurst H = {H:.2f},  fractal D = {D:.2f}")
    print(f"        aperture (hang-foot): mean {ap.mean():.3f} mm, std {ap.std():.3f} mm")
    return dict(tag=tag, foot=foot, hang=hang, rms=rms, beta=beta, H=H, D=D,
                kc=kc, Pc=Pc, sl=sl, Lx=Lx, Ly=Ly, aperture=ap)


def export_profile(res: dict, row_frac: float = 0.5, n_keep: int = 2048):
    """Save a compact committable shear PROFILE (a single scan line across the surface): footwall
    + hangingwall heights along x, in mm, for the 2-D direct-shear capstone."""
    os.makedirs(DATA, exist_ok=True)
    i = int(res["foot"].shape[0] * row_frac)
    foot_p = res["foot"][i]
    hang_p = res["hang"][i]
    x = np.arange(len(foot_p)) * DX_MM
    # optionally decimate to n_keep points (keep full if smaller)
    step = max(1, len(x) // n_keep)
    np.savez(os.path.join(DATA, f"inada_{res['tag']}_profile.npz"),
             x_mm=x[::step], footwall_mm=foot_p[::step], hangingwall_mm=hang_p[::step],
             dx_mm=DX_MM * step, rms_mm=res["rms"], hurst=res["H"], fractal_D=res["D"],
             source="DigitalRocksPortal #273 (DOI 10.17612/QXSA-TK92), Sawayama+ 2020")
    return os.path.join(DATA, f"inada_{res['tag']}_profile.npz")


def main():
    if not os.path.isdir(RAW):
        print(f"  raw CSVs not found in {RAW} — run the downloader first."); return
    print("=== Inada granite rock-joint characterization (Digital Rocks #273) ===")
    rough = characterize("rough")
    smooth = characterize("smooth")

    # quick-look figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W      # noqa: E402
    set_pub_style()
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.78))
    # (A) rough surface height map
    im = axs[0, 0].imshow(rough["foot"], cmap="terrain", extent=[0, rough["Lx"], 0, rough["Ly"]])
    axs[0, 0].set_title(f"Rough joint surface (D={rough['D']:.2f}, RMS={rough['rms']:.2f} mm)", fontsize=8)
    axs[0, 0].set_xlabel("x (mm)"); axs[0, 0].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axs[0, 0], shrink=0.8, label="height (mm)")
    # (B) a shear profile (footwall + hangingwall) — the 2-D capstone line
    i = rough["foot"].shape[0] // 2
    x = np.arange(rough["foot"].shape[1]) * DX_MM
    axs[0, 1].plot(x, rough["foot"][i], color=PUB_COLORS[0], lw=0.5, label="footwall (lower)")
    axs[0, 1].plot(x, rough["hang"][i], color=PUB_COLORS[1], lw=0.5, label="hangingwall (upper)")
    axs[0, 1].set_title("Mating profile (the 2-D direct-shear line)", fontsize=8)
    axs[0, 1].set_xlabel("x (mm)"); axs[0, 1].set_ylabel("height (mm)")
    axs[0, 1].legend(fontsize=6); axs[0, 1].set_xlim(0, x.max())
    # (C) PSD with fitted self-affine slope
    for res, c, lab in ((rough, PUB_COLORS[1], "rough"), (smooth, PUB_COLORS[0], "smooth")):
        axs[1, 0].loglog(res["kc"], res["Pc"], ".", ms=3, color=c, label=f"{lab} (D={res['D']:.2f})")
        kf = res["kc"][res["sl"]]
        axs[1, 0].loglog(kf, res["Pc"][res["sl"]][0] * (kf / kf[0]) ** (-res["beta"]), "-", color=c, lw=1)
    axs[1, 0].set_title("Radial PSD — self-affine slope $\\to$ fractal $D$", fontsize=8)
    axs[1, 0].set_xlabel("wavenumber k (1/mm)"); axs[1, 0].set_ylabel("PSD")
    axs[1, 0].legend(fontsize=6)
    # (D) zoom of the rough profile — the multi-scale roughness a level-set must resolve everywhere
    axs[1, 1].plot(x, rough["foot"][i], color=PUB_COLORS[1], lw=0.6)
    z0 = rough["foot"][i]
    axs[1, 1].set_xlim(x.max() * 0.40, x.max() * 0.55)
    axs[1, 1].set_title("Multi-scale asperities (zoom): the level-set's nightmare", fontsize=8)
    axs[1, 1].set_xlabel("x (mm)"); axs[1, 1].set_ylabel("height (mm)")
    fig.suptitle("Inada-granite tensile rock joint (Digital Rocks #273) — real surface for the "
                 "neural-atlas direct-shear capstone", y=1.01, fontsize=9)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "inada_joint_characterization_pub.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("\n  Saved figure:", out)

    pr = export_profile(rough); ps = export_profile(smooth)
    print("  Saved compact profiles:", pr, "|", ps)
    print(f"\n  Published reference: D~2.4-2.5, RMS 1.3 (smooth) / 1.7 (rough) mm — "
          f"measured here: D_rough={rough['D']:.2f}, RMS_rough={rough['rms']:.2f} mm.")


if __name__ == "__main__":
    main()
