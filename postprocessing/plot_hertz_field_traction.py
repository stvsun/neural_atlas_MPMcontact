#!/usr/bin/env python3
"""Figure: measure-coupling FIELD traction vs closed-form contact mechanics.

Three panels from the benchmark JSON in ``runs/``:
  (a) CV-1b 2-D line Hertz — recovered pressure FIELD p(x) vs the Hertz half-ellipse;
  (b) CV-1c 3-D axisymmetric Hertz — p(r) (BEM penalty, TractionField) vs Hertz;
  (c) CV-2b Cattaneo-Mindlin — tangential traction FIELD q(r) vs the partial-slip closed form.

Run the drivers first (cv1b_hertz_field.py, cv1c_hertz3d_field.py, cv2b_cattaneo_field.py), then
``python3 postprocessing/plot_hertz_field_traction.py`` -> figures/measure_coupling_field_traction.{png,pdf}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing.utils import set_pub_style, PUB_COLORS                  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(_ROOT, "runs")
FIG_DIR = os.path.join(_ROOT, "figures")


def _load(name):
    p = os.path.join(RUNS, name, "history.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _ensure_runs():
    if _load("cv1b_hertz_field") is None:
        from benchmarks.contact.cv_numerical import cv1b_hertz_field as m
        os.makedirs(m.RUN_DIR, exist_ok=True)
        _, hist, _ = m.run(verbose=False)
        json.dump(hist, open(os.path.join(m.RUN_DIR, "history.json"), "w"))
    if _load("cv1c_hertz3d_field") is None:
        from benchmarks.contact.cv_numerical import cv1c_hertz3d_field as m
        os.makedirs(m.RUN_DIR, exist_ok=True)
        _, hist = m.run(verbose=False)
        json.dump(hist, open(os.path.join(m.RUN_DIR, "history.json"), "w"))
    if _load("cv2b_cattaneo_field") is None:
        from benchmarks.contact.cv_numerical import cv2b_cattaneo_field as m
        os.makedirs(m.RUN_DIR, exist_ok=True)
        _, hist = m.run(verbose=False)
        json.dump(hist, open(os.path.join(m.RUN_DIR, "history.json"), "w"))


def main():
    set_pub_style(fontsize=9)
    import matplotlib.pyplot as plt

    _ensure_runs()
    h2d = _load("cv1b_hertz_field")
    h3d = _load("cv1c_hertz3d_field")
    hca = _load("cv2b_cattaneo_field")
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    blue, verm, grn = PUB_COLORS[0], PUB_COLORS[1], PUB_COLORS[2]

    # (a) 2-D line Hertz pressure field
    x = np.array(h2d["x_q"]); p = np.array(h2d["pN_q"]); a = h2d["a_ana"]; p0 = h2d["p0_ana"]
    xs = np.linspace(-a, a, 200)
    ax[0].plot(xs, p0 * np.sqrt(np.clip(1 - (xs / a) ** 2, 0, None)), "-", color="k", lw=1.4,
               label="Hertz $p_0\\sqrt{1-(x/a)^2}$")
    m = p > 0
    ax[0].plot(x[m], p[m], "o", color=blue, ms=3.5, mfc="none", label="field traction (FEM)")
    ax[0].set_title("(a) CV-1b  2-D line Hertz")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("contact pressure $p_N$")
    ax[0].set_xlim(-1.4 * a, 1.4 * a); ax[0].legend(loc="upper right")

    # (b) 3-D axisymmetric Hertz
    r = np.array(h3d["r"]); pr = np.array(h3d["p_penalty"]); a3 = h3d["a_ana"]; p03 = h3d["p0_ana"]
    rs = np.linspace(0, a3, 200)
    ax[1].plot(rs, p03 * np.sqrt(np.clip(1 - (rs / a3) ** 2, 0, None)), "-", color="k", lw=1.4,
               label="Hertz $p(r)$")
    mm = (pr > 0) & (r <= 1.3 * a3)
    ax[1].plot(r[mm][::6], pr[mm][::6], "s", color=verm, ms=3.2, mfc="none",
               label="field traction (BEM)")
    ax[1].set_title("(b) CV-1c  3-D axisymmetric Hertz")
    ax[1].set_xlabel("$r$"); ax[1].set_ylabel("contact pressure $p(r)$")
    ax[1].set_xlim(0, 1.3 * a3); ax[1].legend(loc="upper right")

    # (c) Cattaneo tangential traction field
    rc = np.array(hca["r"]); q = np.array(hca["q"]); qa = np.array(hca["q_cattaneo"])
    ac = hca["a"]; cc = hca["c"]
    ax[2].plot(rc, qa, "-", color="k", lw=1.4, label="Cattaneo $q(r)$")
    mq = rc <= 1.05 * ac
    ax[2].plot(rc[mq][::6], q[mq][::6], "^", color=grn, ms=3.2, mfc="none",
               label="field traction (BEM)")
    ax[2].axvline(cc, ls=":", color="0.5", lw=1.0)
    ax[2].text(cc, ax[2].get_ylim()[1] * 0.05, " stick $c$", fontsize=7, color="0.4")
    ax[2].set_title("(c) CV-2b  Cattaneo partial slip")
    ax[2].set_xlabel("$r$"); ax[2].set_ylabel("tangential traction $q(r)$")
    ax[2].set_xlim(0, 1.05 * ac); ax[2].legend(loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"measure_coupling_field_traction.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved -> {os.path.join(FIG_DIR, 'measure_coupling_field_traction.png')}")


if __name__ == "__main__":
    main()
