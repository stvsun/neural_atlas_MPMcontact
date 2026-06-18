"""CV-6 figures: the Koch-snowflake fractal contact resolution advantage.

Headline (honest axes): the recursive chart stores the IFS at O(1) and resolves contact at ANY
fractal depth (pruned O(depth) descent, ~21 nodes/query), while a uniform SDF grid needs 9^n
cells (adaptive ~4^n) and is capped at its finest resolution. A precomputed SDF is cheaper
per-query; the chart's wins are STORAGE and RESOLUTION-INDEPENDENCE.

Figures (figures/):
  koch_cost_scaling_pub.png         - two panels: storage (O(1) vs 9^n/4^n) | per-query (chart vs SDF)
  koch_geometry_pub.png             - snowflake at levels 1/3/5 + a coarse SDF grid that can't resolve it
  koch_contact_count_pub.png        - distinct contact micro-arcs grow with depth (chart enumerates them)
  koch_spinning_contact.gif         - static-cam schematic: chart-detected contact set (+ net force arrow)
  koch_contact_dynamics.gif         - time-resolved: spinning cam drives a spring-loaded follower (rides it)
  koch_follower_displacement_pub.png- follower position vs cam angle (the fractal cam lift curve)

Run:  python3 postprocessing/plot_koch_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers.contact import koch                                 # noqa: E402
from utils import set_pub_style, PUB_COLORS, SINGLE_COL_W, DOUBLE_COL_W, GOLDEN  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_pub_style()
    return plt


def figure_cost_scaling():
    """Honest two-axis comparison (storage vs per-query), NOT one conflated 'cost' axis.

    LEFT  -- storage/build: the chart's genuine, decisive win (O(1) IFS, flat) vs a uniform
             SDF grid (9^n) and even an adaptive/narrow-band SDF (~boundary cells, O(4^n)).
    RIGHT -- per-query work: the SDF actually WINS (O(1) lookup); the chart pays a small,
             *bounded* O(depth) descent (~21 nodes, measured, resolution-independent).
    The chart's advantage is STORAGE + RESOLUTION-INDEPENDENCE (no max depth baked in), not
    per-query speed -- and a fixed-capacity neural SDF cannot keep refining at all (prose).
    """
    plt = _mpl()
    levels = list(range(1, 13))
    near = np.array([1.03 * v for v in koch.snowflake_vertices(2)[:-1]])
    chart_q = [np.mean([koch.inside_cost(x, n)[1] for x in near]) for n in levels]   # measured
    uniform = [koch.sdf_grid_cells(n) for n in levels]                # 9^n
    adaptive = [float(koch.n_segments(n)) for n in levels]            # ~boundary cells O(4^n)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))

    axL.semilogy(levels, uniform, "s-", color=PUB_COLORS[1], label="uniform SDF grid  $9^n$")
    axL.semilogy(levels, adaptive, "^-", color=PUB_COLORS[3], label="adaptive/narrow-band SDF  $\\sim\\!4^n$")
    axL.semilogy(levels, [4] * len(levels), "o-", color=PUB_COLORS[0], label="IFS chart  $O(1)$ (4 maps)")
    axL.set_xlabel("fractal level $n$"); axL.set_ylabel("storage (cells, log scale)")
    axL.set_title("Storage / build cost")
    axL.legend(loc="center left", fontsize=6.5)
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    axR.plot(levels, chart_q, "o-", color=PUB_COLORS[0], label="recursive chart (nodes/query)")
    axR.plot(levels, [1] * len(levels), "s--", color=PUB_COLORS[1], label="precomputed SDF (O(1) lookup)")
    axR.set_ylim(0, max(chart_q) * 1.4)
    axR.set_xlabel("fractal level $n$"); axR.set_ylabel("per-query work (ops)")
    axR.set_title("Per-query cost (bounded, $\\approx\\!21$ nodes)")
    axR.legend(loc="upper left", fontsize=6.5)
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    fig.suptitle("Chart wins on STORAGE + resolution-independence (any depth on demand); "
                 "SDF wins per-query but is grid-capped", y=1.04, fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "koch_cost_scaling_pub.png")
    fig.savefig(p); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
    return p


def figure_geometry():
    plt = _mpl()
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.3))
    for ax, n in zip(axs[:3], (1, 3, 5)):
        V = koch.snowflake_vertices(n)
        ax.fill(V[:, 0], V[:, 1], fc="#bcd4e6", ec="k", lw=0.7)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"level {n}\n({koch.n_segments(n)} segments)", fontsize=8)
    # level-5 boundary vs a coarse SDF grid that cannot resolve it
    ax = axs[3]
    V = koch.snowflake_vertices(5)
    ax.fill(V[:, 0], V[:, 1], fc="#e8e3c8", ec="k", lw=0.5)
    ng = 24
    g = np.linspace(-1.15, 1.15, ng + 1)
    for gv in g:
        ax.plot([gv, gv], [-1.15, 1.15], color="0.7", lw=0.3)
        ax.plot([-1.15, 1.15], [gv, gv], color="0.7", lw=0.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"level 5 vs {ng}×{ng} SDF grid\n(grid misses the fine spikes)", fontsize=8)
    fig.suptitle("Exact recursive chart (any depth, O(1) storage) vs grid-limited SDF", y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG, "koch_geometry_pub.png")
    fig.savefig(p); plt.close(fig)
    return p


def _contact_arcs(level, dB, R=1.0):
    """Count distinct contact micro-arcs: runs of B-boundary samples inside A (A,B overlapping)."""
    B = koch.snowflake_vertices(level, R=R, center=[dB, 0.0])[:-1]
    ins = np.array([koch.inside_cost(p, level, R=R, center=[0, 0])[0] for p in B], dtype=int)
    arcs = int(np.sum((ins - np.roll(ins, 1)) == 1)) or (1 if ins.any() else 0)
    return arcs, int(ins.sum())


def figure_contact_count():
    plt = _mpl()
    levels = list(range(1, 7))
    dB = 1.7
    arcs = [_contact_arcs(n, dB)[0] for n in levels]
    npts = [_contact_arcs(n, dB)[1] for n in levels]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * GOLDEN))
    ax.plot(levels, arcs, "o-", color=PUB_COLORS[0], label="distinct contact arcs")
    ax.plot(levels, npts, "s--", color=PUB_COLORS[2], label="contacting samples")
    ax.set_xlabel("fractal level $n$"); ax.set_ylabel("count")
    ax.set_title("Fractal contact: more micro-contacts with depth")
    ax.legend(loc="upper left", fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    p = os.path.join(FIG, "koch_contact_count_pub.png")
    fig.savefig(p); plt.close(fig)
    return p


def gif_spinning_contact(level=4, R=1.0, dB=1.55, frames=60):
    """Prescribed spinning cam A; highlight the chart-detected contact set on B (B fixed).
    No free integration (robust); the net contact force arrow shows what WOULD drive B."""
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    th = np.linspace(0, 2 * np.pi, 400)
    Bv = koch.snowflake_vertices(level, R=R, center=[dB, 0.0])
    Bs = Bv[:-1]
    fig, ax = plt.subplots(figsize=(5, 4))

    def draw(fi):
        ax.clear(); ax.set_aspect("equal"); ax.set_xlim(-1.6, dB + 1.6); ax.set_ylim(-1.8, 1.8)
        aA = -2 * np.pi * fi / frames
        Av = koch.snowflake_vertices(level, R=R, center=[0, 0], alpha=aA)
        ax.fill(Av[:, 0], Av[:, 1], fc="#d98e5a", ec="k", lw=0.6, alpha=0.9)
        ax.fill(Bv[:, 0], Bv[:, 1], fc="#5a8fd9", ec="k", lw=0.6, alpha=0.9)
        F = np.zeros(2); pen = []
        for p in Bs:
            ins, _ = koch.inside_cost(p, level, R=R, center=[0, 0], alpha=aA)
            if ins:
                g, _f, n = koch.nearest_boundary(p, level, R=R, center=[0, 0], alpha=aA)
                pen.append(p); F += (-g) * n
        if pen:
            pen = np.array(pen); ax.plot(pen[:, 0], pen[:, 1], "r.", ms=3, zorder=5)
            cB = Bs.mean(0); Fn = F / (np.linalg.norm(F) + 1e-9)
            ax.annotate("", xy=cB + 0.7 * Fn, xytext=cB,
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
        ax.set_title("spinning Koch cam — chart-detected contact (red) + net force on B", fontsize=8)
        ax.axis("off")

    anim = FuncAnimation(fig, draw, frames=frames, interval=80)
    p = os.path.join(FIG, "koch_spinning_contact.gif")
    anim.save(p, writer=PillowWriter(fps=14), dpi=96); plt.close(fig)   # dpi-capped for repo size
    return p


def gif_cam_follower(level=3, R=1.0, omega=2.0, eps_n=1.2e4, k=10.0, c=4.0,
                     x_rest=0.35, x0=0.9, x_stop=0.2, face_h=0.14, n_face=15,
                     T=3.4, nframe=100):
    """Contact mechanics IN TIME: a spinning Koch cam drives a FLAT-FACED follower.

    The follower is a thin 1-DOF plate (a vertical face at x=x_f, |y|<=face_h) pressed onto the
    rotating fractal Koch cam by a return spring; the chart-detected contact force pushes it back
    out. Overdamped (first-order, quasi-static) dynamics:

        c * dx_f/dt = -k (x_f - x_rest) + F_contact_x .

    A flat face (not a second fat snowflake) is the textbook cam-follower geometry: the plate only
    ever touches the cam's RIGHT boundary (outward normals ~ +x), so contact is shallow, there is
    no fat-body overlap, and -- with F_contact_x >= 0 enforced (one-sided plate) plus a slider stop
    x_f >= x_stop -- tunnelling is structurally impossible. The plate position then *traces the
    fractal cam lift curve*: x_f(angle) = the cam's rightmost extent over the face, resolved by the
    recursive chart. Detection is 100% chart-based (koch.inside / koch.nearest_boundary).
    """
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    ys = np.linspace(-face_h, face_h, n_face)              # plate face sample points (in y)
    dy = 2.0 * face_h / (n_face - 1)
    dt = 1.2e-3
    nsteps = int(T / dt)
    rec_every = max(1, nsteps // nframe)
    xf = x0
    frames = []                                            # (alpha_A, x_f, contact_pts)
    disp = []                                              # (alpha_A_deg, x_f)
    max_pen_global = 0.0
    for i in range(nsteps):
        t = i * dt
        aA = -omega * t
        Fx = 0.0
        pen = []
        max_pen = 0.0
        for y in ys:
            p = np.array([xf, y])
            ins, _ = koch.inside_cost(p, level, R=R, center=[0, 0], alpha=aA)
            if ins:
                g, _f, n = koch.nearest_boundary(p, level, R=R, center=[0, 0], alpha=aA)
                Fx += eps_n * (-g) * max(n[0], 0.0) * dy   # one-sided plate: only +x push
                pen.append(p)
                max_pen = max(max_pen, -g)
        # overdamped (first-order) relaxation -- unconditionally stable for stiff contact
        xf += dt * (-k * (xf - x_rest) + Fx) / c
        xf = max(xf, x_stop)                               # physical slider stop (never reached in practice)
        max_pen_global = max(max_pen_global, max_pen)
        if i % rec_every == 0:
            frames.append((aA, xf, np.array(pen) if pen else None))
            disp.append((np.degrees(-aA) % 360.0, xf))
    print("    [cam-follower] x_f in [%.3f, %.3f], max penetration %.4f, hit-stop=%s"
          % (min(d[1] for d in disp), max(d[1] for d in disp), max_pen_global,
             any(abs(d[1] - x_stop) < 1e-6 for d in disp)))

    fig, ax = plt.subplots(figsize=(5, 4))
    plate_w = 0.32

    def draw(fi):
        ax.clear(); ax.set_aspect("equal"); ax.set_xlim(-1.4, 2.4); ax.set_ylim(-1.5, 1.5)
        aA, xx, pen = frames[fi]
        Av = koch.snowflake_vertices(level, R=R, center=[0, 0], alpha=aA)
        ax.fill(Av[:, 0], Av[:, 1], fc="#d98e5a", ec="k", lw=0.6, alpha=0.92, zorder=2)
        # flat-faced follower plate (face at x=xx, extends to the right)
        ax.add_patch(plt.Rectangle((xx, -face_h), plate_w, 2 * face_h,
                                    fc="#5a8fd9", ec="k", lw=0.7, alpha=0.92, zorder=3))
        ax.plot([xx, xx], [-face_h, face_h], color="k", lw=1.2, zorder=4)   # contact face
        if pen is not None and len(pen):
            ax.plot(pen[:, 0], pen[:, 1], "r.", ms=4, zorder=6)            # chart-detected contact
        # return spring: pushes the follower toward the cam (-x), so the arrow points LEFT
        ax.annotate("", xy=(xx + plate_w + 0.12, 0), xytext=(xx + plate_w + 0.62, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.4))
        ax.text(xx + plate_w + 0.37, 0.13, "spring", fontsize=6.5, ha="center", color="0.4")
        ax.set_title("Spinning Koch cam drives a flat-faced follower\n"
                     "chart-detected contact (red);  $x_f$ = %.3f" % xx, fontsize=8)
        ax.axis("off")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=70)
    p = os.path.join(FIG, "koch_contact_dynamics.gif")
    anim.save(p, writer=PillowWriter(fps=15), dpi=96); plt.close(fig)   # dpi-capped for repo size

    # supporting displacement-vs-angle plot = the fractal cam lift curve
    d = np.array(disp)
    order = np.argsort(d[:, 0])
    fig2, ax2 = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * GOLDEN))
    ax2.plot(d[order, 0], d[order, 1], "-", lw=0.9, color=PUB_COLORS[0])
    ax2.set_xlabel("cam angle (deg)"); ax2.set_ylabel("follower position $x_f$")
    ax2.set_title("Fractal cam lift curve (chart-traced)")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG, "koch_follower_displacement_pub.png")); plt.close(fig2)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    for fn in (figure_cost_scaling, figure_geometry, figure_contact_count,
               gif_spinning_contact, gif_cam_follower):
        try:
            print("  Saved:", fn())
        except Exception as exc:                                   # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
