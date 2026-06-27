"""Focused test for the CV-7 OT-gap rock-joint direct-shear driver.

Validates that the optimal-transport (measure-coupling) gap reproduces the Patton closed form
``mu_app = tan(phi_b + i)`` within tolerance, recovers the dilatancy rate ``tan(i)``, and that the
rigid-block net force is integration-scheme-invariant (OT vs vertical-ray lumped coincide)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.contact.cv_numerical.cv7_ot_gap import run


def test_cv7_ot_gap_patton():
    m = run(angle_deg=25.0, mu=0.3, wavelength=5.0)
    # OT gap reproduces Patton to (near machine) precision; pass tol is 2% (lumped-driver tol).
    assert m["ot_relerr_vs_patton"] < 0.02
    assert m["ray_relerr_vs_patton"] < 0.02
    assert m["passed"] is True
    # emergent dilatancy rate dy/dux == tan(i).
    assert m["ot_dilation_rate_relerr_vs_tan_i"] < 0.02
    # rigid-block net-force invariance: OT and vertical-ray peak shear coincide.
    assert m["net_force_rel_diff_ot_vs_ray"] < 0.01
    # OT keeps penetration much smaller than the lumped ray scheme (field-quality benefit).
    assert m["ot_pen_max"] < m["ray_pen_max"]


def test_cv7_ot_gap_second_angle():
    m = run(angle_deg=20.0, mu=0.3, wavelength=5.0)
    patton = math.tan(math.radians(math.degrees(math.atan(0.3)) + 20.0))
    assert abs(m["patton_mu_pred"] - patton) < 1e-9
    assert m["ot_relerr_vs_patton"] < 0.02
