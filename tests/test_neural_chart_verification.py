"""Executable skeleton for verifying NEURAL coordinate charts against the analytical
contact benchmarks (CV-1..CV-5).

This is the harness referenced in docs/contact_verification_manual.md §10. Today every
test SKIPS because no neural chart is trained yet. To activate, implement the two loaders
below to return a trained neural SDF / ChartDecoder for the named shape; the L0
(geometry/kinematics) comparisons against the analytical references in
postprocessing/contact_fields.py and solvers/contact/supershape.py are already wired.

Compare like-for-like (manual §10.2):
  - a neural SDF returns the EUCLIDEAN gap/normal  -> compare to the Euclidean reference;
  - a neural ChartDecoder is compared BOUNDARY-TO-BOUNDARY (phi_nn(theta) vs phi_ana(theta)).

Neural tolerances (manual §10.3): gap RMSE ~1e-3*L, normal angle ~1-2 deg, fields ~5%.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocessing import contact_fields as cf      # analytical references (numpy)
from solvers.contact import supershape as ss

# Neural tolerances (manual §10.3)
TAU_GAP_REL = 2e-3      # gap RMSE / body size
TAU_NORMAL_DEG = 2.0    # normal angle error
TAU_FIELD = 0.05        # stress/field relative error


# ---------------------------------------------------------------------------
# Plug-in points: implement these when neural charts are trained.
# Return None to keep the corresponding tests skipped.
# ---------------------------------------------------------------------------

def load_neural_sdf(shape):
    """Return a trained neural SDF (torch.nn.Module: x(N,3)->(N,) or (N,1)) for *shape*,
    or None if not available. shape in {'sphere','cylinder','disc','supershape'}."""
    return None


def load_neural_decoder(shape):
    """Return a trained ChartDecoder + frame for *shape*'s boundary, or None."""
    return None


def _angle_deg(a, b):
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


# ---------------------------------------------------------------------------
# CV-1 — Hertz: neural sphere SDF, L0 gap/normal vs analytical Euclidean
# ---------------------------------------------------------------------------

def test_cv1_hertz_neural_sdf_L0():
    sdf = load_neural_sdf("sphere")
    if sdf is None:
        pytest.skip("no neural sphere SDF yet — see contact_verification_manual.md §10")
    import torch
    from solvers.contact.gap import evaluate_gap
    R, L = 1.0, 1.0
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.5 * R, 1.5 * R, size=(2000, 3))
    g_nn, n_nn = evaluate_gap(torch.tensor(x, dtype=torch.float64), sdf)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    r = np.linalg.norm(x, axis=1)
    g_ana = r - R                                   # Euclidean SDF of a sphere
    n_ana = x / r[:, None]
    assert np.sqrt(np.mean((g_nn - g_ana) ** 2)) / L < TAU_GAP_REL
    assert np.max(_angle_deg(n_nn, n_ana)) < TAU_NORMAL_DEG


def test_cv1_hertz_neural_L1():
    pytest.skip("L1: run the contact solve with the neural SDF; assert a, p0, F(delta) "
                "to ~3-5% vs cf.hertz_3d_params (manual §10.3). Enable when solver wired.")


# ---------------------------------------------------------------------------
# CV-2..CV-4 — same pattern (skeletons)
# ---------------------------------------------------------------------------

def test_cv2_cattaneo_neural_L1():
    pytest.skip("L1: with the neural normal-force, assert c/a=(1-Q/muP)^(1/3) and q(r) "
                "via cf.cattaneo_* to ~5%.")


def test_cv3_brazilian_neural_L1():
    sdf = load_neural_sdf("disc")
    if sdf is None:
        pytest.skip("no neural disc chart yet — L1: center sxx=+2P/piDt, syy=-6P/piDt to ~5%.")


def test_cv4_nine_disc_neural_L1():
    sdf = load_neural_sdf("disc")
    if sdf is None:
        pytest.skip("no neural disc chart yet — L1: equibiaxial center -2N/piRt; N(d) to ~5%.")


# ---------------------------------------------------------------------------
# CV-5 — superformula: the discriminating L0 test (concavities, cusps)
# ---------------------------------------------------------------------------

def test_cv5_supershape_neural_sdf_L0():
    """Neural SDF (Euclidean) vs the dense Euclidean reference of the superformula —
    discriminates medial-axis normal degradation in the concave valleys (manual §10.4)."""
    sdf = load_neural_sdf("supershape")
    if sdf is None:
        pytest.skip("no neural supershape SDF yet — see contact_verification_manual.md §10.4")
    import torch
    from solvers.contact.gap import evaluate_gap
    p = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, scale=1.0)
    c = np.array([0.0, 0.0])
    L = ss.radius(np.linspace(0, 2 * np.pi, 400), p).max()
    rng = np.random.RandomState(1)
    xy = rng.uniform(-1.6 * L, 1.6 * L, size=(1500, 2))
    # dense Euclidean reference: gap = signed nearest-foot distance; normal = the
    # SDF-GRADIENT normal (foot direction), like-for-like with evaluate_gap (NOT the
    # surface normal at the nearest boundary sample).
    thb = np.linspace(0, 2 * np.pi, 8000, endpoint=False)
    B = ss.boundary(thb, c, 0.0, p)
    d2 = np.sum((xy[:, None, :] - B[None, :, :]) ** 2, axis=2)
    k = np.argmin(d2, axis=1)
    foot = B[k]
    dist = np.sqrt(d2[np.arange(len(xy)), k])
    inside = ss.inside(xy, c, 0.0, p)
    g_ana = np.where(inside, -dist, dist)
    v = xy - foot
    n_ana = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    n_ana[inside] *= -1.0                              # outward SDF gradient
    x3 = np.column_stack([xy, np.zeros(len(xy))])
    g_nn, n_nn = evaluate_gap(torch.tensor(x3, dtype=torch.float64), sdf)
    n_nn = n_nn.numpy()
    assert np.max(np.abs(n_nn[:, 2])) < 0.05          # 2D shape at z=0: z-normal ~ 0
    assert np.sqrt(np.mean((g_nn.numpy() - g_ana) ** 2)) / L < TAU_GAP_REL
    assert np.max(_angle_deg(n_nn[:, :2], n_ana)) < TAU_NORMAL_DEG


def test_cv5_supershape_neural_decoder_L0():
    """Neural ChartDecoder boundary vs analytical superformula boundary (boundary-to-boundary)."""
    dec = load_neural_decoder("supershape")
    if dec is None:
        pytest.skip("no neural supershape decoder yet — compare phi_nn(theta) vs phi_ana(theta).")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs"]))
