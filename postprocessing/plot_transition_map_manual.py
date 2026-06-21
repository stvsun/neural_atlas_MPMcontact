"""Didactic schematic figures for the transition-map contact theory manual.

These are *conceptual* diagrams (not data plots): they introduce the transition map, contrast it with
the ambient level set, and illustrate the three geometric facts the manual rests on — multi-arc
detection, the radial-gap / perpendicular-gap bias, and the spectral-bias asperity argument. Run:

    python3 postprocessing/plot_transition_map_manual.py

Outputs (figures/):
    tm_concept_pub.png          — the boundary-to-boundary transition map tau_AB = phi_B^{-1} o phi_A
    tm_levelset_vs_chart_pub.png— ambient SDF vs radial boundary chart vs the gap construction
    tm_multiarc_pub.png         — chart boundary scan finds >=2 contact arcs; single CPP finds 1 foot
    tm_radial_bias_pub.png      — radial gap = perpendicular gap / cos(alpha) (conservative-large)
    tm_spectral_bias_pub.png    — chart resolves asperity slopes the SDF smooths -> Patton strength gap
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")

# --- a consistent didactic palette (light background, print-friendly) -------------------------------
BG = "#ffffff"
INK = "#1b2430"        # primary text / axes
MUTE = "#6b7686"       # secondary text
A_COL = "#2f6fb0"      # body A (chart blue)
B_COL = "#d2691e"      # body B (chart orange)
CHART = "#1f9e6e"      # chart / "good" green
SDF_COL = "#b0485f"    # level-set / "smoothed" red
GAPC = "#7a3fb0"       # gap annotations (violet)
GRID = "#dfe4ea"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "mathtext.fontset": "cm",
})


# ====================================================================================================
# shape helpers
# ====================================================================================================
def superformula(theta, m=5, n1=0.30, n2=1.7, n3=1.7, a=1.0, b=1.0, scale=1.0):
    """Gielis superformula radius rho(theta) (matches solvers/contact/supershape.radius)."""
    u = m * theta / 4.0
    t = np.abs(np.cos(u) / a) ** n2 + np.abs(np.sin(u) / b) ** n3
    return scale * t ** (-1.0 / n1)


def blob(theta, c, harmonics, base=1.0, rot=0.0):
    """A smooth star-shaped boundary rho(theta) = base*(1 + sum a_k cos(k theta + phi_k))."""
    rho = np.ones_like(theta) * base
    for k, (amp, ph) in harmonics.items():
        rho = rho + base * amp * np.cos(k * theta + ph)
    x = c[0] + rho * np.cos(theta + rot)
    y = c[1] + rho * np.sin(theta + rot)
    return x, y, rho


def _polish(ax, title=None):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12.5, color=INK, pad=8)


def _arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>", ms=12, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, alpha=alpha,
                                 shrinkA=0, shrinkB=0, zorder=6))


# ====================================================================================================
# Figure 1 — the transition map
# ====================================================================================================
def fig_concept():
    th = np.linspace(0, 2 * np.pi, 1440)
    cA = np.array([-1.55, 0.0])
    cB = np.array([1.70, 0.12])
    harmA = {2: (0.16, 0.5), 3: (0.09, -0.7)}
    harmB = {2: (0.20, 2.5), 4: (0.07, 0.0)}
    xA, yA, rhoA = blob(th, cA, harmA, base=1.25)
    xB, yB, rhoB = blob(th, cB, harmB, base=1.30)

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    fig.patch.set_facecolor(BG)
    ax.fill(xA, yA, color=A_COL, alpha=0.13, zorder=1)
    ax.plot(xA, yA, color=A_COL, lw=2.4, zorder=3)
    ax.fill(xB, yB, color=B_COL, alpha=0.13, zorder=1)
    ax.plot(xB, yB, color=B_COL, lw=2.4, zorder=3)

    for c, col, lab in [(cA, A_COL, "$c_A$"), (cB, B_COL, "$c_B$")]:
        ax.plot(*c, "o", color=col, ms=6, zorder=7)
        ax.annotate(lab, c, textcoords="offset points", xytext=(-2, -17),
                    color=col, fontsize=12, ha="center")

    # pick the boundary point of A in closest approach to B (the natural contact candidate)
    dB = np.stack([xA - cB[0], yA - cB[1]], axis=1)
    rB = np.linalg.norm(dB, axis=1)
    psi = np.arctan2(dB[:, 1], dB[:, 0])
    rhoB_psi_all = blob(psi, cB, harmB, base=1.30)[2]
    gapB = rB - rhoB_psi_all                       # > 0 (bodies separated); minimised at closest approach
    iA = int(np.argmin(gapB))
    pA = np.array([xA[iA], yA[iA]])
    thA = np.arctan2(pA[1] - cA[1], pA[0] - cA[0])
    psiB = psi[iA]
    rhoB_psi = rhoB_psi_all[iA]
    footB = cB + rhoB_psi * np.array([np.cos(psiB), np.sin(psiB)])

    # A-ray (theta_A) and the boundary point x
    _arrow(ax, cA, pA, color=A_COL, lw=1.4, ls=(0, (4, 3)), ms=10)
    ax.annotate(r"$\theta_A$", cA + 0.40 * (pA - cA) + np.array([0.0, 0.18]), color=A_COL, fontsize=13)
    ax.plot(*pA, "o", color=INK, ms=8, zorder=9)
    ax.annotate(r"$x=\varphi_A(\theta_A)$", pA, textcoords="offset points", xytext=(-6, 28),
                fontsize=13, color=INK, ha="center")

    # B-ray (psi_B) from cB out to the foot on B's boundary (and dashed on to x)
    _arrow(ax, cB, footB, color=B_COL, lw=1.4, ls=(0, (4, 3)), ms=10)
    ax.plot([footB[0], pA[0]], [footB[1], pA[1]], color=B_COL, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ax.annotate(r"$\psi_B$", cB + 0.45 * (footB - cB) + np.array([0.0, -0.24]), color=B_COL, fontsize=13)
    ax.plot(*footB, "s", color=B_COL, ms=9, zorder=9)
    ax.annotate(r"$\varphi_B(\psi_B)$", footB, textcoords="offset points", xytext=(6, -22),
                fontsize=13, color=B_COL, ha="center")

    # gap bracket: foot on B's boundary -> x on A's boundary (the visible separation)
    ax.annotate("", xy=pA, xytext=footB, arrowprops=dict(arrowstyle="<|-|>", color=GAPC, lw=2.4))
    ax.annotate(r"$g_B(x)=\|x-c_B\|-\rho_B(\psi_B)$", (0.5, 0.135), xycoords="axes fraction",
                fontsize=13, color=GAPC, ha="center")

    ax.annotate(r"$\tau_{AB}:\ \theta_A\ \mapsto\ \psi_B=(\varphi_B^{-1}\circ\varphi_A)(\theta_A)$",
                (0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=14, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f3f6fa", ec=MUTE, lw=1.0))
    ax.annotate("body $A$", (cA[0] - 0.95, cA[1] + 1.45), color=A_COL, fontsize=13, fontweight="bold")
    ax.annotate("body $B$", (cB[0] + 0.35, cB[1] + 1.55), color=B_COL, fontsize=13, fontweight="bold")

    _polish(ax)
    ax.set_xlim(-3.0, 3.4)
    ax.set_ylim(-2.0, 2.3)
    fig.suptitle("The transition map: take a surface point of $A$ into physical space, then invert $B$'s chart",
                 fontsize=13.5, color=INK, y=0.99)
    out = os.path.join(FIG, "tm_concept_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ====================================================================================================
# Figure 2 — ambient level set vs radial boundary chart vs the gap construction
# ====================================================================================================
def _poly_sdf(px, py, bx, by):
    """Signed distance from grid (px,py) to closed polygon (bx,by); negative inside."""
    P = np.stack([px.ravel(), py.ravel()], axis=1)
    V = np.stack([bx, by], axis=1)
    n = len(V)
    seg0 = V
    seg1 = np.roll(V, -1, axis=0)
    d = seg1 - seg0
    dd = (d * d).sum(1)
    # distance to each segment
    dist = np.full(P.shape[0], np.inf)
    for i in range(n):
        w = P - seg0[i]
        t = np.clip((w @ d[i]) / max(dd[i], 1e-12), 0.0, 1.0)
        proj = seg0[i] + t[:, None] * d[i]
        dist = np.minimum(dist, np.linalg.norm(P - proj, axis=1))
    # inside test (ray cast)
    inside = np.zeros(P.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        cond = ((V[i, 1] > P[:, 1]) != (V[j, 1] > P[:, 1]))
        xint = (V[j, 0] - V[i, 0]) * (P[:, 1] - V[i, 1]) / (V[j, 1] - V[i, 1] + 1e-30) + V[i, 0]
        inside ^= cond & (P[:, 0] < xint)
        j = i
    sgn = np.where(inside, -1.0, 1.0)
    return (sgn * dist).reshape(px.shape)


def fig_levelset_vs_chart():
    th = np.linspace(0, 2 * np.pi, 400)
    c = np.array([0.0, 0.0])
    harm = {3: (0.30, 0.4), 5: (0.12, -0.6)}
    bx, by, rho = blob(th, c, harm, base=1.0)

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.7))
    fig.patch.set_facecolor(BG)

    # (a) ambient SDF
    ax = axs[0]
    g = np.linspace(-2.0, 2.0, 320)
    gx, gy = np.meshgrid(g, g)
    sdf = _poly_sdf(gx, gy, bx, by)
    vmax = 1.6
    cf = ax.contourf(gx, gy, sdf, levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, zorder=0)
    ax.contour(gx, gy, sdf, levels=[0.0], colors=[INK], linewidths=2.2, zorder=2)
    ax.contour(gx, gy, sdf, levels=np.linspace(-1.5, 1.5, 13), colors=["#ffffff"],
               linewidths=0.5, alpha=0.5, zorder=1)
    # sketch a medial-axis spoke into the deepest concavity
    notch = th[np.argmin(rho)]
    ax.plot([0, 0.42 * np.cos(notch)], [0, 0.42 * np.sin(notch)], color="#1b2430",
            lw=1.6, ls=(0, (2, 2)), zorder=3)
    ax.annotate("medial axis\n($\\nabla\\phi$ non-smooth)", (0, 0), textcoords="offset points",
                xytext=(20, -34), fontsize=9.5, color=INK, ha="center")
    _polish(ax, r"(a) ambient level set  $\phi:\mathbb{R}^d\!\to\!\mathbb{R}$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.annotate(r"$g=\phi(x),\ \ n=\nabla\phi/\|\nabla\phi\|$", (0.5, -0.07),
                xycoords="axes fraction", ha="center", fontsize=11, color=INK)

    # (b) radial boundary chart
    ax = axs[1]
    ax.fill(bx, by, color=CHART, alpha=0.12, zorder=1)
    ax.plot(bx, by, color=CHART, lw=2.4, zorder=3)
    ax.plot(0, 0, "o", color=INK, ms=5, zorder=5)
    for k in range(0, 360, 30):
        a = np.deg2rad(k)
        rr = blob(np.array([a]), c, harm, base=1.0)[2][0]
        ax.plot([0, rr * np.cos(a)], [0, rr * np.sin(a)], color=CHART, lw=0.8, alpha=0.55, zorder=2)
    # inset: rho(theta)
    iax = ax.inset_axes([0.62, 0.04, 0.36, 0.26])
    iax.plot(np.rad2deg(th), rho, color=CHART, lw=1.6)
    iax.set_facecolor("#f3faf6")
    iax.set_xticks([0, 180, 360])
    iax.set_yticks([])
    iax.tick_params(labelsize=7)
    iax.set_xlabel(r"$\theta$", fontsize=8, labelpad=0)
    iax.set_title(r"$\rho(\theta)$", fontsize=9, pad=2)
    _polish(ax, r"(b) radial chart  $\rho:S^1\!\to\!\mathbb{R}^+$ (boundary only)")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.annotate(r"no ambient field; star-shaped about $c$", (0.5, -0.07),
                xycoords="axes fraction", ha="center", fontsize=11, color=INK)

    # (c) the gap construction
    ax = axs[2]
    ax.fill(bx, by, color=CHART, alpha=0.10, zorder=1)
    ax.plot(bx, by, color=CHART, lw=2.2, zorder=3)
    ax.plot(0, 0, "o", color=INK, ms=5, zorder=6)
    ax.annotate("$c$", (0, 0), textcoords="offset points", xytext=(-2, -14), fontsize=11, ha="center")
    psi = np.deg2rad(35)
    rr = blob(np.array([psi]), c, harm, base=1.0)[2][0]
    foot = np.array([rr * np.cos(psi), rr * np.sin(psi)])
    p = (rr + 0.62) * np.array([np.cos(psi), np.sin(psi)])     # a point outside, along the ray
    _arrow(ax, c, p, color=GAPC, lw=1.6, ms=12)
    ax.plot(*foot, "o", color=CHART, ms=8, zorder=7)
    ax.plot(*p, "o", color=INK, ms=8, zorder=7)
    ax.annotate(r"$p$", p, textcoords="offset points", xytext=(8, 4), fontsize=12, color=INK)
    ax.annotate(r"$\hat d=(p-c)/r$", 0.55 * p, textcoords="offset points", xytext=(6, -16),
                fontsize=11, color=GAPC)
    ax.annotate("", xy=p, xytext=foot, arrowprops=dict(arrowstyle="<|-|>", color=GAPC, lw=2.2))
    ax.annotate(r"$g=r-\rho(\hat d)$", (foot + p) / 2, textcoords="offset points",
                xytext=(10, 10), fontsize=12, color=GAPC)
    _polish(ax, r"(c) the radial gap")
    ax.set_xlim(-1.7, 1.9)
    ax.set_ylim(-1.7, 1.9)
    ax.annotate(r"$\nabla_d F=\hat d-\nabla_S\rho/r$  (matched normal)", (0.5, -0.07),
                xycoords="axes fraction", ha="center", fontsize=11, color=INK)

    fig.suptitle("Two representations of the same body: the level set needs the whole ambient field; "
                 "the chart needs only the boundary",
                 fontsize=13.5, color=INK, y=1.02)
    out = os.path.join(FIG, "tm_levelset_vs_chart_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ====================================================================================================
# Figure 3 — multi-arc detection
# ====================================================================================================
def fig_multiarc():
    # body B: a gently wavy floor; body A: a two-footed "staple" follower whose two feet touch the
    # floor at two separated places -> two disjoint contact arcs (a single CPP returns only one).
    xf = np.linspace(-2.7, 2.7, 700)

    def floor_y(x):
        return -0.05 + 0.10 * np.cos(2 * np.pi * x / 3.2)

    floor = floor_y(xf)
    rfoot = 0.52
    feet = [np.array([-1.05, 0.40]), np.array([1.05, 0.40])]   # foot-circle centres
    th = np.linspace(0, 2 * np.pi, 900)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    fig.patch.set_facecolor(BG)
    ax.fill_between(xf, floor, -1.4, color=B_COL, alpha=0.13, zorder=1)
    ax.plot(xf, floor, color=B_COL, lw=2.4, zorder=2)

    # draw body A (two feet + a connecting bar high above the floor) as one rigid body
    from matplotlib.patches import Circle, FancyBboxPatch
    for fc in feet:
        ax.add_patch(Circle(fc, rfoot, fc=A_COL, ec=A_COL, lw=2.2, alpha=0.16, zorder=3))
        ax.add_patch(Circle(fc, rfoot, fc="none", ec=A_COL, lw=2.4, zorder=4))
    ax.add_patch(plt.Rectangle((-1.05, 0.40), 2.10, 0.55, fc=A_COL, ec=A_COL, lw=2.2,
                               alpha=0.16, zorder=3))
    ax.plot([-1.05, -1.05], [0.40, 0.95], color=A_COL, lw=2.4, zorder=4)
    ax.plot([1.05, 1.05], [0.40, 0.95], color=A_COL, lw=2.4, zorder=4)
    ax.plot([-1.57, 1.57], [0.95, 0.95], color=A_COL, lw=2.4, zorder=4)
    ax.plot([-1.57, -1.05], [0.95, 0.95], color=A_COL, lw=2.4, zorder=4)
    ax.plot([1.05, 1.57], [0.95, 0.95], color=A_COL, lw=2.4, zorder=4)

    # scan each foot boundary; mark points below the floor (penetrating) -> one arc per foot in contact
    n_arcs = 0
    for k, fc in enumerate(feet):
        bx = fc[0] + rfoot * np.cos(th)
        by = fc[1] + rfoot * np.sin(th)
        pen = by < floor_y(bx)
        if pen.any():
            n_arcs += 1
            ax.plot(bx[pen], by[pen], ".", color=SDF_COL, ms=8, zorder=6)
            mid = np.where(pen)[0][pen.sum() // 2]
            ax.annotate(f"arc {k+1}", (bx[mid], by[mid]), textcoords="offset points",
                        xytext=(0, -24), color=SDF_COL, fontsize=12.5, fontweight="bold", ha="center")

    # single closest-point projection: ONE global nearest contact (the deeper foot)
    best = None
    for fc in feet:
        bx = fc[0] + rfoot * np.cos(th)
        by = fc[1] + rfoot * np.sin(th)
        d = by - floor_y(bx)
        j = np.argmin(d)
        if best is None or d[j] < best[0]:
            best = (d[j], bx[j], by[j])
    ax.plot(best[1], best[2], "o", color=CHART, ms=14, mfc="none", mew=2.6, zorder=7)
    ax.annotate("single CPP:\n1 foot", (best[1], best[2]), textcoords="offset points",
                xytext=(34, 6), color=CHART, fontsize=11.5, ha="left")

    ax.annotate("follower $A$ (chart-scanned boundary)", (0.5, 0.95), xycoords="axes fraction",
                color=A_COL, fontsize=12, fontweight="bold", ha="center")
    ax.annotate("body $B$", (0.5, 0.06), xycoords="axes fraction", color=B_COL,
                fontsize=12, fontweight="bold", ha="center")
    ax.annotate(f"chart boundary scan: {n_arcs} disjoint contact arcs        "
                "single closest-point projection: 1 foot",
                (0.5, -0.06), xycoords="axes fraction", ha="center", fontsize=11.5, color=INK)
    _polish(ax, "Multi-arc detection: a boundary scan finds every contact arc; a single projection finds one")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-0.9, 1.25)
    out = os.path.join(FIG, "tm_multiarc_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out, n_arcs


# ====================================================================================================
# Figure 4 — radial gap vs perpendicular gap (the 1/cos alpha bias)
# ====================================================================================================
def fig_radial_bias():
    # an inclined planar flank (exact geometry): the ray from c meets it at angle alpha to the normal,
    # so the along-ray radial gap exceeds the true perpendicular gap by 1/cos(alpha).
    beta = np.deg2rad(20.0)
    n = np.array([np.cos(beta), np.sin(beta)])         # outward wall normal (points up-right)
    tdir = np.array([-np.sin(beta), np.cos(beta)])     # along the wall
    F = np.array([0.30, 0.20])                          # the radial foot, on the wall
    alpha = np.deg2rad(38.0)
    raydir = np.cos(alpha) * n + np.sin(alpha) * tdir
    raydir = raydir / np.linalg.norm(raydir)
    g_rad = 1.30
    c = F - 1.55 * raydir                               # center, interior
    p = F + g_rad * raydir                              # exterior query point on the ray
    foot_p = F + float((p - F) @ tdir) * tdir           # true perpendicular foot

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.patch.set_facecolor(BG)
    # the wall + shaded interior
    w0 = F - 2.4 * tdir
    w1 = F + 2.0 * tdir
    ax.plot([w0[0], w1[0]], [w0[1], w1[1]], color=CHART, lw=3.0, zorder=3)
    poly = np.array([w0, w1, w1 - 2.2 * n, w0 - 2.2 * n])
    ax.fill(poly[:, 0], poly[:, 1], color=CHART, alpha=0.08, zorder=1)
    ax.annotate("body boundary", w1, textcoords="offset points", xytext=(-58, 4),
                color=CHART, fontsize=10.5)

    ax.plot(*c, "o", color=INK, ms=6, zorder=7)
    ax.annotate("$c$", c, textcoords="offset points", xytext=(-6, -13), fontsize=12)
    _arrow(ax, c, p, color=GAPC, lw=1.6, ms=12)
    ax.plot(*F, "o", color=CHART, ms=11, zorder=8)
    ax.plot(*p, "o", color=INK, ms=9, zorder=8)
    ax.annotate("$p$", p, textcoords="offset points", xytext=(9, 2), fontsize=12)
    ax.annotate(r"radial foot $\rho(\hat d)$", F, textcoords="offset points", xytext=(-34, -22),
                fontsize=11, color=CHART, ha="center")

    # perpendicular foot + perpendicular distance
    ax.plot([p[0], foot_p[0]], [p[1], foot_p[1]], color=SDF_COL, lw=2.4, zorder=5)
    ax.plot(*foot_p, "s", color=SDF_COL, ms=10, zorder=8)
    ax.annotate("perpendicular foot", foot_p, textcoords="offset points", xytext=(2, 14),
                fontsize=11, color=SDF_COL, ha="center")

    # outward normal at F + angle alpha (arc swept from n to raydir)
    _arrow(ax, F, F + 0.85 * n, color=INK, lw=1.5, ms=11)
    ax.annotate(r"$n$", F + 0.9 * n, textcoords="offset points", xytext=(2, 2), fontsize=13)
    tt = np.linspace(0.0, alpha, 40)
    arcpts = F[None, :] + 0.42 * (np.cos(tt)[:, None] * n[None, :] + np.sin(tt)[:, None] * tdir[None, :])
    ax.plot(arcpts[:, 0], arcpts[:, 1], color=INK, lw=1.2)
    amid = F + 0.56 * (np.cos(alpha / 2) * n + np.sin(alpha / 2) * tdir)
    ax.annotate(r"$\alpha$", amid, fontsize=13, color=INK, ha="center", va="center")

    # gap brackets
    ax.annotate("", xy=p, xytext=F, arrowprops=dict(arrowstyle="<|-|>", color=CHART, lw=2.2))
    ax.annotate(r"$g_{\rm rad}$", (F + p) / 2, textcoords="offset points", xytext=(6, 13),
                fontsize=13, color=CHART)
    ax.annotate(r"$g_\perp$", (p + foot_p) / 2, textcoords="offset points", xytext=(11, 0),
                fontsize=13, color=SDF_COL)

    ax.annotate(r"$g_{\rm rad}=g_\perp/\cos\alpha+O(g^2)\ \ \Rightarrow\ \ "
                r"|g_{\rm rad}|\geq|g_\perp|$", (0.5, 0.045), xycoords="axes fraction",
                ha="center", fontsize=14, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f3f6fa", ec=MUTE, lw=1.0))
    _polish(ax, "Radial gap vs perpendicular gap: sign exact, magnitude conservative-large")
    ax.set_xlim(-1.5, 2.1)
    ax.set_ylim(-1.6, 1.9)
    out = os.path.join(FIG, "tm_radial_bias_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ====================================================================================================
# Figure 5 — spectral bias: chart resolves asperity slopes the SDF smooths -> Patton strength gap
# ====================================================================================================
def _self_affine_profile(x, hurst=0.78, kmax=48, seed=7):
    rng = np.random.RandomState(seed)
    h = np.zeros_like(x)
    L = x[-1] - x[0]
    for k in range(1, kmax + 1):
        amp = k ** (-(hurst + 0.5))
        ph = rng.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * k * (x - x[0]) / L + ph)
    h -= h.mean()
    h /= h.std()
    return h


def fig_spectral_bias():
    x = np.linspace(0.0, 10.0, 1600)               # mm-like horizontal extent
    h0 = _self_affine_profile(x)

    def mean_angle(hp):
        sl = np.gradient(hp, x)
        return float(np.rad2deg(np.arctan(np.abs(sl))).mean())

    # scale the profile so the full-resolution mean asperity angle is a realistic ~20 deg
    lo, hi = 1e-4, 1e2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_angle(mid * h0) < 20.0:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi) * h0

    # chart reconstruction ~ the true profile; level-set (SDF) ~ a low-pass (smoothed) version
    from numpy.fft import rfft, irfft
    H = rfft(h)
    cutoff = 7
    Hs = H.copy()
    Hs[cutoff:] = 0.0
    h_sdf = irfft(Hs, n=len(h))

    i_chart = mean_angle(h)
    i_sdf = mean_angle(h_sdf)

    mu = 0.30  # base friction tan(phi_b)
    def patton(i_deg):
        ti = np.tan(np.deg2rad(i_deg))
        return (ti + mu) / (1 - mu * ti)
    mu_chart = patton(i_chart)
    mu_sdf = patton(i_sdf)

    fig = plt.figure(figsize=(13.2, 5.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 2.1, 1.25], wspace=0.28)

    # (a) the profiles
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, h, color=CHART, lw=1.4, label=f"chart $h_\\theta(x)$  ($\\bar i$={i_chart:.0f}°)")
    ax.plot(x, h_sdf, color=SDF_COL, lw=2.2, label=f"level set (smoothed)  ($\\bar i$={i_sdf:.0f}°)")
    ax.set_facecolor("#fbfcfd")
    ax.set_xlabel("position $x$")
    ax.set_ylabel("height $h$ (norm.)")
    ax.set_title("(a) asperity profile: chart resolves the slopes the SDF smooths", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(MUTE)

    # (b) zoom on a few asperities to show slope difference
    ax = fig.add_subplot(gs[0, 1])
    m = (x > 3.0) & (x < 5.4)
    ax.plot(x[m], h[m], color=CHART, lw=2.0, label="chart")
    ax.plot(x[m], h_sdf[m], color=SDF_COL, lw=2.6, label="level set")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_facecolor("#fbfcfd")
    ax.set_xlabel("position $x$ (zoom)")
    ax.set_title("(b) the SDF low-passes the asperity tips", fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(MUTE)

    # (c) Patton consequence
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(["chart", "level set"], [mu_chart, mu_sdf], color=[CHART, SDF_COL], width=0.6)
    ax.set_title("(c) peak strength\n$\\mu_{\\rm app}=\\tan(\\phi_b+i)$", fontsize=11.5)
    ax.set_ylabel(r"$\mu_{\rm app}$")
    drop = 100 * (1 - mu_sdf / mu_chart)
    ax.annotate(f"SDF under-predicts\nby {drop:.0f}%", (0.5, 0.5), xycoords="axes fraction",
                ha="center", fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff4f0", ec=SDF_COL, lw=1.0))
    for b, v in zip(bars, [mu_chart, mu_sdf]):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
    ax.set_facecolor("#fbfcfd")
    for s in ax.spines.values():
        s.set_color(MUTE)

    fig.suptitle("Why the chart beats the level set on rough geometry: resolved slopes set the friction angle "
                 "(spectral bias)", fontsize=13.5, color=INK, y=1.01)
    out = os.path.join(FIG, "tm_spectral_bias_pub.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out, (i_chart, i_sdf, mu_chart, mu_sdf)


def main():
    os.makedirs(FIG, exist_ok=True)
    print("generating transition-map manual schematics ...")
    print("  ", fig_concept())
    print("  ", fig_levelset_vs_chart())
    p, narcs = fig_multiarc()
    print("  ", p, f"({narcs} arcs detected)")
    print("  ", fig_radial_bias())
    p, stats = fig_spectral_bias()
    print("  ", p, f"(i_chart={stats[0]:.1f}deg i_sdf={stats[1]:.1f}deg "
          f"mu_chart={stats[2]:.2f} mu_sdf={stats[3]:.2f})")
    print("done.")


if __name__ == "__main__":
    main()
