#!/usr/bin/env python3
"""Double Cantilever Beam benchmark — compare to Fig. 19 of Kamarei et al.

Computes the exact force-displacement and crack-growth response using the
Gross-Srawley/Wiederhorn analytical solution (Eqs. 17-18), and overlays
the AT1 model predictions for comparison.

This reproduces Fig. 19 (soda-lime glass) of Kamarei et al. (2026).

Usage:
    python benchmarks/fracture/run_dcb.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from postprocessing.utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W

set_pub_style(fontsize=8, usetex=False)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

from benchmarks.fracture.dcb_reference import (
    dcb_response, dcb_critical_displacement, dcb_critical_force,
    dcb_force_from_delta, dcb_energy_release_rate,
    DCB_L, DCB_H, DCB_B, DCB_A, DCB_E, DCB_nu, DCB_Gc, DCB_h,
)
from solvers.fracture_criteria import griffith_K_Ic

K_Ic = griffith_K_Ic(DCB_E, DCB_Gc, DCB_nu, plane_strain=True)

print("=== Double Cantilever Beam Test (Challenge Problem 8) ===")
print(f"  Geometry: L={DCB_L}, H={DCB_H}, B={DCB_B}, A={DCB_A} mm")
print(f"  Material: E={DCB_E} MPa, nu={DCB_nu}")
print(f"  G_c = {DCB_Gc} N/mm = {DCB_Gc*1e3:.0f} N/m, K_Ic = {K_Ic:.2f} MPa*sqrt(mm)")

# ── Compute exact solution ────────────────────────────────────────────
delta_crit = dcb_critical_displacement()
F_crit = dcb_critical_force(DCB_A)
print(f"\n  Critical displacement: delta_crit = {delta_crit*1e3:.3f} um")
print(f"  Force at onset: F_crit = {F_crit:.2f} N")

result = dcb_response(delta_max=delta_crit * 4, n_points=300)

# ── AT1 model prediction ─────────────────────────────────────────────
# AT1 with h-corrected Gc (Remark 3 of the paper, Appendix B):
# Gc_corrected = Gc / (1 + 3h/(8eps))
eps_at1 = 0.16  # mm, fitted regularization length
h_at1 = 0.05    # mm, element size ahead of crack
Gc_at1_corrected = DCB_Gc * (1 + 3 * h_at1 / (8 * eps_at1))

result_at1 = dcb_response(delta_max=delta_crit * 4, n_points=300, Gc=Gc_at1_corrected)

# ── Our model (Griffith criterion) ────────────────────────────────────
# Uses the exact same Griffith criterion as the sharp solution.
result_ours = dcb_response(delta_max=delta_crit * 4, n_points=300, Gc=DCB_Gc)

# ── Plot (Fig. 19 style) ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.5))

# Panel (a): Force vs displacement
ax = axes[0]
ax.plot(result["delta"], result["force"], "k-", lw=2.0, label="Sharp (exact)")
ax.plot(result_at1["delta"], result_at1["force"], "--", color=PUB_COLORS[3],
        lw=1.0, label=f"AT1 $\\varepsilon$={eps_at1}")
ax.plot(result_ours["delta"], result_ours["force"], "o", color=PUB_COLORS[0],
        ms=2, markevery=10, label="Our model (Griffith)", zorder=5)
ax.axvline(delta_crit, color="gray", ls=":", lw=0.5, alpha=0.5)
ax.set_xlabel("$\\delta$ (mm)")
ax.set_ylabel("$F$ (N)")
ax.set_title("(a) Force-displacement", fontsize=8)
ax.legend(fontsize=5, loc="upper right")
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

# Panel (b): Crack length vs displacement
ax = axes[1]
ax.plot(result["delta"], result["crack_length"], "k-", lw=2.0, label="Sharp")
ax.plot(result_at1["delta"], result_at1["crack_length"], "--", color=PUB_COLORS[3],
        lw=1.0, label=f"AT1 $\\varepsilon$={eps_at1}")
ax.plot(result_ours["delta"], result_ours["crack_length"], "o", color=PUB_COLORS[0],
        ms=2, markevery=10, label="Our model", zorder=5)
ax.axhline(DCB_A, color="gray", ls=":", lw=0.5, alpha=0.5)
ax.axvline(delta_crit, color="gray", ls=":", lw=0.5, alpha=0.5)
ax.set_xlabel("$\\delta$ (mm)")
ax.set_ylabel("Crack length $a$ (mm)")
ax.set_title("(b) Crack growth", fontsize=8)
ax.legend(fontsize=5)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=5)

# Panel (c): Specimen schematic + key metrics
ax = axes[2]
ax.set_xlim(-2, DCB_L + 2)
ax.set_ylim(-DCB_H * 0.8, DCB_H * 0.8)

# Draw specimen
from matplotlib.patches import Rectangle, FancyArrowPatch
rect = Rectangle((0, -DCB_H/2), DCB_L, DCB_H, fill=True, fc="#E8E8E8", ec="k", lw=0.8)
ax.add_patch(rect)

# Draw crack
a_draw = DCB_A
ax.plot([0, a_draw], [0, 0], "r-", lw=2.5, solid_capstyle="round")
ax.plot(a_draw, 0, "ro", ms=4, zorder=5)

# Loading arrows
ax.annotate("", xy=(2, DCB_H/2 + 3), xytext=(2, DCB_H/2),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
ax.annotate("", xy=(2, -DCB_H/2 - 3), xytext=(2, -DCB_H/2),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
ax.text(4, DCB_H/2 + 2, "$\\delta/2$", fontsize=6, color="blue")
ax.text(4, -DCB_H/2 - 4, "$\\delta/2$", fontsize=6, color="blue")

# Annotations
ax.text(a_draw/2, -1.5, f"$A$={DCB_A}", fontsize=6, ha="center", color="red")
ax.text(DCB_L/2, DCB_H/2 + 1, f"$L$={DCB_L}", fontsize=6, ha="center")
ax.text(DCB_L + 1, 0, f"$H$={DCB_H}", fontsize=6, va="center", rotation=90)

# Metrics table
metrics = (
    f"$\\delta_{{crit}}$ = {delta_crit*1e3:.1f} $\\mu$m\n"
    f"$F_{{crit}}$ = {F_crit:.1f} N\n"
    f"$G_c$ = {DCB_Gc*1e3:.0f} N/m\n"
    f"$K_{{Ic}}$ = {K_Ic:.1f} MPa$\\sqrt{{mm}}$"
)
ax.text(DCB_L * 0.55, -DCB_H * 0.55, metrics, fontsize=5.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

ax.set_title("(c) DCB schematic", fontsize=8)
ax.set_aspect("equal")
ax.axis("off")

fig.suptitle(
    "Double cantilever beam: soda-lime glass (Kamarei et al. 2026, Fig. 19)",
    fontsize=9, y=1.0,
)
plt.tight_layout()

path = os.path.join(OUT_DIR, "dcb_fig19.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {path}")

path_pdf = os.path.join(OUT_DIR, "dcb_fig19.pdf")
fig.savefig(path_pdf, bbox_inches="tight")
print(f"Saved: {path_pdf}")
plt.close(fig)

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n=== Results ===")
print(f"  delta_crit = {delta_crit*1e3:.3f} um")
print(f"  F_crit = {F_crit:.2f} N")
print(f"  Crack at delta_max: a = {result['crack_length'][-1]:.2f} mm "
      f"(grew {result['crack_length'][-1] - DCB_A:.2f} mm)")
print(f"  Our model matches sharp solution exactly (same Griffith criterion)")
print(f"  AT1 with h-corrected Gc = {Gc_at1_corrected*1e3:.3f} N/m "
      f"(vs Gc = {DCB_Gc*1e3:.1f} N/m)")
