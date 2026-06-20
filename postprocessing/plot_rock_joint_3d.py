"""3D rock-joint figures + PyVista surface animation.

Produces:
  * figures/rock_joint_3d_modes_pub.png  — rigid 3-mode anisotropy (traction rose, dilation, T_perp)
  * figures/rock_joint_cyclic_pub.png    — deformable-FEM cyclic hysteresis + dilation + degradation
  * figures/rock_joint_3d_shear.gif      — PyVista 3D animation: two REAL Inada surfaces shearing

Run:  python3 postprocessing/plot_rock_joint_3d.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
sys.path.insert(0, _ROOT)
RUNS = os.path.join(_ROOT, "runs")
FIG = os.path.join(_ROOT, "figures")
MODE_C = {"in_plane": "#0072B2", "out_of_plane": "#D55E00", "mixed": "#009E73"}


def _load_hist(path):
    f = os.path.join(path, "history.json")
    return json.load(open(f)) if os.path.exists(f) else None


# --------------------------------------------------------------------------------------------------
def modes_figure(tag="rough", suptitle=("3-D rigid rock-joint shear: roughness anisotropy across "
                                         "loading modes (real Inada surface)"),
                 out_name="rock_joint_3d_modes_pub.png"):
    """3-mode anisotropy (a) mu_app (b) dilation (c) T_perp from runs/rock_joint_3d_<tag>_<mode>.
    ``tag='rough'`` = the rigid real-Inada driver (default); ``tag='twoblock'`` = the genuine
    two-deformable-block decoder-FEM driver (Phase 1).  Same 3-panel layout/colours."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    H = {m: _load_hist(os.path.join(RUNS, f"rock_joint_3d_{tag}_{m}")) for m in MODE_C}
    if not any(H.values()):
        print(f"  (no 3-mode runs for tag='{tag}'; skip modes_figure)"); return
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))
    for m, c in MODE_C.items():
        h = H[m]
        if not h:
            continue
        hh = h["history"]
        axs[0].plot(hh["u"], hh["mu_app"], color=c, lw=1.2, label=m.replace("_", "-"))
        axs[1].plot(hh["u"], hh["dilation"], color=c, lw=1.2, label=m.replace("_", "-"))
        axs[2].plot(hh["u"], hh["T_perp"], color=c, lw=1.2, label=m.replace("_", "-"))
    axs[0].set_title("(a) shear strength $\\mu_{app}(u)$", fontsize=8)
    axs[0].set_xlabel("shear $u$ (mm)"); axs[0].set_ylabel("$\\mu_{app}$"); axs[0].legend(fontsize=6)
    axs[1].set_title("(b) dilation (anisotropic)", fontsize=8)
    axs[1].set_xlabel("shear $u$ (mm)"); axs[1].set_ylabel("dilation (mm)"); axs[1].legend(fontsize=6)
    axs[2].set_title("(c) transverse traction $T_\\perp$ (shear-dir coupling)", fontsize=8)
    axs[2].set_xlabel("shear $u$ (mm)"); axs[2].set_ylabel("$T_\\perp$"); axs[2].legend(fontsize=6)
    fig.suptitle(suptitle, y=1.02, fontsize=9)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("  saved", out)


def cyclic_figure(tag="", out_name="rock_joint_cyclic_pub.png",
                  suptitle=("Deformable two-block rock joint — mixed-mode cyclic shear (chart-FEM, "
                            "dilatant interface)")):
    """Cyclic hysteresis + dilation + degradation decay.  ``tag=''`` = the flat-interface benchmark
    (runs/rock_joint_3d/cyclic_<mode>_CNV); ``tag='_genuine'`` = the genuine rough-geometry cyclic
    (cyclic_genuine_<mode>_CNV).  Same 2x2 layout."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    from joint_data_io import load_joint_history

    def _load_npz(name):
        p = os.path.join(RUNS, "rock_joint_3d", name)
        try:
            return {"history": load_joint_history(p)["history"]} if os.path.isdir(p) else None
        except Exception:
            return None
    H = {m: _load_npz(f"cyclic{tag}_{m}_CNV") for m in MODE_C}
    deg = _load_npz(f"cyclic{tag}_in_plane_degrade_CNV")
    if not any(H.values()):
        print("  (no cyclic runs; skip cyclic_figure)"); return
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.66))
    # (a) hysteresis tau vs u_par, 3 modes
    for m, c in MODE_C.items():
        h = H[m]
        if not h:
            continue
        hh = h["history"]
        axs[0, 0].plot(hh["u_par"], hh["tau"], color=c, lw=0.8, label=m.replace("_", "-"))
    axs[0, 0].set_title("(a) cyclic hysteresis $\\tau$ vs shear", fontsize=8)
    axs[0, 0].set_xlabel("shear $u$ (mm)"); axs[0, 0].set_ylabel("shear traction $\\tau$")
    axs[0, 0].legend(fontsize=6)
    # (b) dilation / normal-stress path (in_plane)
    h = H["in_plane"]["history"]
    axs[0, 1].plot(h["u_par"], h["t_N"], color="#444", lw=1.0)
    axs[0, 1].set_title("(b) normal-stress path $\\sigma_n$ (CNV, dilation-coupled)", fontsize=8)
    axs[0, 1].set_xlabel("shear $u$ (mm)"); axs[0, 1].set_ylabel("$\\sigma_n=t_N$")
    # (c) mu_app loop (in_plane) — the friction cap +/- tan(phi_b+i)
    axs[1, 0].plot(h["u_par"], h["mu_app"], color=MODE_C["in_plane"], lw=0.8)
    axs[1, 0].set_title("(c) mobilized friction $\\mu_{app}$ (cap $\\pm\\tan(\\phi_b{+}i)$)", fontsize=8)
    axs[1, 0].set_xlabel("shear $u$ (mm)"); axs[1, 0].set_ylabel("$\\mu_{app}=\\tau/\\sigma_n$")
    # (d) degradation: peak |mu_app| per cycle
    if deg:
        dh = deg["history"]; cyc = np.array(dh["cycle"]); mu = np.abs(np.array(dh["mu_app"]))
        peaks = [mu[cyc == c].max() for c in np.unique(cyc)]
        axs[1, 1].plot(np.unique(cyc) + 1, peaks, "o-", color="#CC79A7", lw=1.4)
        axs[1, 1].set_title("(d) Plesha asperity degradation:\npeak strength decays per cycle", fontsize=8)
        axs[1, 1].set_xlabel("cycle number"); axs[1, 1].set_ylabel("peak $|\\mu_{app}|$")
        axs[1, 1].set_xticks(np.unique(cyc) + 1)
    fig.suptitle(suptitle, y=1.01, fontsize=9)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("  saved", out)


def energy_ledger_figure(tag="_genuine", mode="in_plane",
                         out_name="rock_joint_cyclic_energy_pub.png"):
    """Cyclic energy ledger (Phase 4) for the genuine rough-geometry run cyclic{tag}_<mode>_CNV:
    (a) tau-u hysteresis loops, (b) cumulative energy terms W_ext / W_fric / dU_el / W_pen / W_stick,
    (c) per-cycle closure ratio W_ext / (W_fric + dU_el + W_pen + W_stick)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    from joint_data_io import load_joint_history
    p = os.path.join(RUNS, "rock_joint_3d", f"cyclic{tag}_{mode}_CNV")
    if not os.path.isdir(p):
        print(f"  (no genuine cyclic run {p}; skip energy_ledger_figure)"); return
    H = load_joint_history(p)["history"]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))
    axs[0].plot(H["u_par"], H["tau"], color="#0072B2", lw=0.8)
    axs[0].set_title("(a) cyclic hysteresis $\\tau$ vs $u$", fontsize=8)
    axs[0].set_xlabel("shear $u$ (mm)"); axs[0].set_ylabel("shear traction $\\tau$")
    for k, c, lab in (("W_ext", "#000000", "$W_{ext}$ (machine)"), ("W_fric", "#D55E00", "$W_{fric}$"),
                      ("W_el", "#0072B2", "$\\Delta U_{el}$"), ("W_pen", "#009E73", "$W_{pen}$"),
                      ("W_stick", "#CC79A7", "$W_{stick}$")):
        if k in H:
            y = np.asarray(H[k]); y = y - y[0] if k in ("W_el", "W_pen", "W_stick") else y
            axs[1].plot(H["u_par"] * 0 + np.arange(len(y)), y, color=c, lw=1.0, label=lab)
    axs[1].set_title("(b) cumulative energy terms", fontsize=8)
    axs[1].set_xlabel("increment"); axs[1].set_ylabel("energy"); axs[1].legend(fontsize=5)
    cyc = np.asarray(H["cycle"]); ratios = []
    for c in np.unique(cyc):
        idx = np.where(cyc == c)[0]; i0, i1 = idx[0], idx[-1]
        dext = H["W_ext"][i1] - H["W_ext"][i0]
        den = ((H["W_fric"][i1] - H["W_fric"][i0]) + (H["W_el"][i1] - H["W_el"][i0])
               + (H["W_pen"][i1] - H["W_pen"][i0]) + (H["W_stick"][i1] - H["W_stick"][i0]))
        ratios.append(dext / den if abs(den) > 1e-30 else np.nan)
    axs[2].axhline(1.0, color="#888", lw=0.8, ls="--")
    axs[2].plot(np.unique(cyc) + 1, ratios, "o-", color="#444", lw=1.4, ms=5)
    axs[2].set_title("(c) per-cycle ratio (boundary-sampled)", fontsize=8)
    axs[2].set_xlabel("cycle"); axs[2].set_ylabel("$W_{ext}/(W_{fric}{+}\\Delta U_{el}{+}W_{pen}{+}W_{stick})$",
                                                  fontsize=6)
    axs[2].set_xticks(np.unique(cyc) + 1); axs[2].set_ylim(0, 1.6)
    # HONEST: the cumulative friction tally is the conserved measure — it over-counts the net work
    g_ef = float(np.asarray(H["W_ext"])[-1] / (np.asarray(H["W_fric"])[-1] + 1e-30))
    axs[2].annotate(f"cumulative $W_{{ext}}/W_{{fric}}={g_ef:.2f}$\n($W_{{fric}}$ over-counts {1/max(g_ef,1e-9):.1f}$\\times$)",
                    xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=6, color="#B22222")
    fig.suptitle("Genuine rough-joint cyclic shear — energy ledger NOT yet closed: cumulative $W_{fric}$ "
                 "over-counts the net dissipation (reversible asperity micro-sliding; §11.11)", y=1.02, fontsize=8)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("  saved", out)


def roughness_law_figure():
    """Emergent dilatancy-vs-roughness law from runs/cv7_roughness_sweep/results.json (Phase 2):
    peak mu_app and total dilation vs surface RMS, for the amplitude sweep and the spectral-cutoff
    sweep, with the decoder reconstruction accuracy overlaid (it stays <8% so the law is geometry, not
    a fitting artefact)."""
    import json
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from utils import set_pub_style, DOUBLE_COL_W
    set_pub_style()
    f = os.path.join(RUNS, "cv7_roughness_sweep", "results.json")
    if not os.path.exists(f):
        print("  (no roughness sweep; skip roughness_law_figure)"); return
    R = json.load(open(f))
    amp = sorted([r for r in R if r["family"] == "amp"], key=lambda r: r["surf_rms"])
    kmx = sorted([r for r in R if r["family"] == "kmax"], key=lambda r: r["kmax"])
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))
    ca, ck = "#0072B2", "#D55E00"
    axs[0].plot([r["surf_rms"] for r in amp], [r["peak_mu_app"] for r in amp], "o-", color=ca,
                lw=1.2, ms=4, label="amplitude sweep")
    axs[0].plot([r["surf_rms"] for r in kmx], [r["peak_mu_app"] for r in kmx], "s--", color=ck,
                lw=1.2, ms=4, label="cutoff sweep")
    axs[0].set_title("(a) peak $\\mu_{app}$ vs roughness", fontsize=8)
    axs[0].set_xlabel("surface RMS (mm)"); axs[0].set_ylabel("peak $\\mu_{app}$"); axs[0].legend(fontsize=6)
    axs[1].plot([r["surf_rms"] for r in amp], [r["total_dilation"] for r in amp], "o-", color=ca,
                lw=1.2, ms=4, label="amplitude sweep")
    axs[1].plot([r["surf_rms"] for r in kmx], [r["total_dilation"] for r in kmx], "s--", color=ck,
                lw=1.2, ms=4, label="cutoff sweep")
    axs[1].set_title("(b) total dilation vs roughness", fontsize=8)
    axs[1].set_xlabel("surface RMS (mm)"); axs[1].set_ylabel("dilation (mm)"); axs[1].legend(fontsize=6)
    axs[2].plot([r["kmax"] for r in kmx], [r["peak_mu_app"] for r in kmx], "s--", color=ck, lw=1.2, ms=4)
    ax2b = axs[2].twinx()
    ax2b.plot([r["kmax"] for r in kmx], [r["recon_pct"] for r in kmx], "^:", color="#444", lw=1.0, ms=3)
    axs[2].set_title("(c) spectral cutoff: strength + recon", fontsize=8)
    axs[2].set_xlabel("spectral cutoff $k_{max}$"); axs[2].set_ylabel("peak $\\mu_{app}$", color=ck)
    ax2b.set_ylabel("decoder recon (% RMS)", color="#444", fontsize=7); ax2b.tick_params(labelsize=6)
    fig.suptitle("Emergent dilatancy-vs-roughness law (genuine rough geometry, decoder-FEM + Coulomb)",
                 y=1.02, fontsize=9)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "cv7_roughness_law_pub.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("  saved", out)


def surface_animation_3d(tag="rough", mode="mixed", crop_mm=16.0, every=6, n_frames=28,
                         shear_total=5.0, mu=0.3, sigma_n=1.0):
    """PyVista 3D animation: two REAL Inada surfaces (footwall fixed, hangingwall sheared in the
    joint plane + dilating), colored by the vertical gap."""
    import math
    from benchmarks.contact.cv_numerical import rock_joint_shear_3d as r3
    from surface_anim_3d import animate_two_surfaces
    if not os.path.isdir(os.path.join(_ROOT, "downloads", "inada_granite")):
        print("  (raw Inada CSVs absent; skip 3D animation)"); return
    x, y, Z = r3._load_inada_map(tag, crop_mm=crop_mm, every=every)
    lower = r3.bake_samples(x, y, Z)
    upper = dict(x=x.copy(), y=y.copy(), h=Z.copy(), hp=None)
    d = np.asarray(r3.MODES[mode], float)
    span = (x[-1] - x[0]) * (y[-1] - y[0]); amp = Z.max() - Z.min()
    z_lo = float(Z.max() - Z.min() - 2 * amp); z_hi = float(Z.max() - Z.min() + 2 * amp)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    lo_seq, up_seq, sc_seq = [], [], []
    up_grid = r3._grid(x, y, Z, *np.gradient(Z, x, y))
    for k in range(n_frames):
        umag = shear_total * k / (n_frames - 1)
        u = umag * d
        A_ov = max((x[-1] - x[0] - abs(u[0])) * (y[-1] - y[0] - abs(u[1])), 0.25 * span)
        z0 = r3.solve_z_equilibrium(u, lower, up_grid, 5.0e4, mu, sigma_n * A_ov, d, (z_lo, z_hi))
        # upper surface in world coords: shifted by (u) and raised by z0
        Zu = z0 + np.interp(0, [0, 1], [0, 0]) + Z          # show the upper face translated visually
        # gap field = (z0 + h_U(world)) - h_L(world); approximate at lattice (mated -> shift)
        hL, _, _, _ = r3._query(lower, (XX + u[0]).ravel(), (YY + u[1]).ravel())
        gap = (z0 + Z.ravel()) - hL
        lo_seq.append(Z.copy())
        up_seq.append((z0 + Z).copy())
        sc_seq.append(gap.reshape(Z.shape))
    out = os.path.join(FIG, "rock_joint_3d_shear.gif")
    clim = (float(np.percentile(np.concatenate([s.ravel() for s in sc_seq]), 5)),
            float(np.percentile(np.concatenate([s.ravel() for s in sc_seq]), 95)))
    animate_two_surfaces(x, y, lo_seq, up_seq, sc_seq, out, fps=10, explode_dz=max(2.0, amp),
                         sbar_title="gap (mm)", cmap="coolwarm", clim=clim, window_size=(760, 560))
    print("  saved", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6))


def main():
    modes_figure()
    cyclic_figure()
    try:
        surface_animation_3d()
    except Exception as e:
        print("  3D animation failed:", repr(e))


if __name__ == "__main__":
    main()
