"""CV-6 Koch fractal contact — OT measure-coupling gap field, resolution-independence.

Focused checks on ``benchmarks/contact/cv_numerical/cv6_ot_gap.py``:
  * (A) the IFS chart geometry is self-similar to machine precision (segments 3*4^n, perimeter
        ratio (4/3)^n) — the resolution-independent reference;
  * (B) the OT consistent contact integral CONVERGES across IFS levels (Cauchy / geometric) with a
        monotone exterior gap — level-independent once the contacting asperities are resolved;
  * (C) the SAME OT integral fed a FIXED level-set grid SDF is resolution-bound (a finite contact-
        force error vs the exact chart that the chart does not have) — the measured chart-beats-grid.

No closed form exists for the fractal contact field; these are the honest acceptance targets.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact.cv_numerical import cv6_ot_gap as cv6  # noqa: E402


def test_geometry_self_similarity_exact():
    rows, seg_ok, max_per_err = cv6.geometry_self_similarity([0, 1, 2, 3, 4, 5])
    assert seg_ok, "segment count must equal 3*4^n at every level"
    assert max_per_err < 1e-9, f"perimeter ratio must match (4/3)^n to machine eps, got {max_per_err}"


def test_ot_field_converges_across_levels():
    rows, summary = cv6.ot_field_convergence([2, 3, 4, 5, 6])
    # the consistent contact integral is Cauchy: the last cross-level increment is tiny and the
    # increments shrink geometrically.
    assert summary["last_increment_rel"] < 0.02, summary["last_increment_rel"]
    assert summary["cauchy_ratio_last_over_first"] < 1.0, summary["cauchy_ratio_last_over_first"]
    # refining the Koch boundary only ADDS outward bumps -> exterior gap is non-increasing in n.
    assert summary["gap_monotonicity_max_violation"] < 1e-6, summary["gap_monotonicity_max_violation"]
    # the converged contact force is positive and finite.
    assert summary["F_converged"] > 0.0


def test_chart_beats_fixed_levelset_grid():
    rows, meta = cv6.chart_vs_grid([2, 3, 4, 5, 6])
    # the exact chart force converges across levels (it resolves every new asperity).
    F_chart = [r["F_chart"] for r in rows]
    assert abs(F_chart[-1] - F_chart[-2]) / F_chart[-1] < 0.02
    # a FIXED-resolution level-set grid SDF on the SAME OT integral is resolution-bound: a finite
    # contact-force error vs the exact chart (here large because the coarse grid smooths the spike).
    max_relerr = max(r["F_relerr_grid_vs_chart"] for r in rows)
    assert max_relerr > 0.02, f"grid SDF should show a resolution-bound force error, got {max_relerr}"


def test_driver_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cv6, "RUN_DIR", str(tmp_path / "cv6_ot_gap"))
    m = cv6.run(verbose=False)
    assert m["overall_status"] in ("pass", "partial")
    assert m["A_geometry"]["PASS"]
    assert os.path.isfile(os.path.join(cv6.RUN_DIR, "metrics.json"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
