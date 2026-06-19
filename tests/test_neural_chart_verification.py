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

# Neural tolerances (manual §11.3) — MEASURED from atlas/sdf/train_analytical_sdf.py
# (pure-regression fit of the exact analytical SDF; see runs/neural_sdf/*_meta.json).
TAU_GAP_REL = 4e-3      # sphere gap RMSE / body size (measured ~1.6e-3 near contact)
TAU_GAP_DISC = 7e-3     # disc gap RMSE / body size (measured ~5e-3 over a +/-0.15 band)
TAU_GAP_SUPERSHAPE = 2e-2  # superformula: a HIGH-CAPACITY neural SDF (width 192) reaches gap ~8e-3
                           # = ~2x the smooth disc — usable but cusp-relaxed (the spectral-bias
                           # difficulty CV-5 exposes, manual §11.4). The degradation is sharper in
                           # the NORMAL (|n_z| ~ 0.5 vs ~0.1 disc). ACCURATE CV-5 path = radial chart (M3).
TAU_NORMAL_DEG = 2.5    # normal angle error (median; smooth shapes ~0.5-1.5 deg)
TAU_FIELD = 0.05        # stress/field relative error
# A 2-D shape is embedded as a 3-D z-prism (z=0 slice) so the 3-D evaluate_gap/SDFNet can
# consume it.  Contact uses the IN-PLANE normal n[:2] (its angle is the primary metric, ~0.5 deg);
# the residual z-tilt of the prism gradient is benign for contact (|n[:2]| stays > 97%).  We bound
# the TAIL of |n_z| (90th percentile) rather than the median, so the prism leak is disclosed, not
# hidden — the review flagged that a median bound let 27.5% of points exceed it silently.
TAU_NZ = 0.25           # 90th-percentile |n_z| for a 2-D shape embedded at z=0 (disc measured ~0.18)


# ---------------------------------------------------------------------------
# Plug-in points: implement these when neural charts are trained.
# Return None to keep the corresponding tests skipped.
# ---------------------------------------------------------------------------

def load_neural_sdf(shape):
    """Return a trained neural SDF (torch.nn.Module: x(N,3)->(N,) or (N,1)) for *shape*,
    or None if not available. shape in {'sphere','cylinder','disc','supershape'}.

    Loads a cached checkpoint trained by atlas/sdf/train_analytical_sdf.py (run
    `python3 atlas/sdf/train_analytical_sdf.py --all` to produce them); returns None (skip)
    if absent, matching the original skeleton behaviour."""
    try:
        from atlas.sdf.train_analytical_sdf import load_trained_sdf
    except Exception:
        return None
    name = "disc" if shape == "cylinder" else shape
    return load_trained_sdf(name)


def load_neural_decoder(shape):
    """Return a trained ChartDecoder + frame for *shape*'s boundary, or None."""
    return None


def load_neural_rho(shape):
    """Return a trained 2-D neural RADIAL chart (NeuralRho2D) for *shape*, or None.
    Trained by atlas/charts/train_radial_chart.py; the transition-map detector for star-shaped
    bodies (the accurate CV-5 path)."""
    try:
        from atlas.charts.train_radial_chart import load_trained_radial_chart
    except Exception:
        return None
    return load_trained_radial_chart(shape)


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
    # sample the NEAR-CONTACT shell |gap| <~ 0.15 R (manual §11.2: query near the contact
    # region), not the whole box — far-field SDF accuracy is irrelevant to contact.
    v = rng.randn(2000, 3); v /= np.linalg.norm(v, axis=1, keepdims=True)
    x = v * (R + rng.uniform(-0.15, 0.15, size=(2000, 1)))
    g_nn, n_nn = evaluate_gap(torch.tensor(x, dtype=torch.float64), sdf)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    r = np.linalg.norm(x, axis=1)
    g_ana = r - R                                   # Euclidean SDF of a sphere
    n_ana = x / r[:, None]
    assert np.sqrt(np.mean((g_nn - g_ana) ** 2)) / L < TAU_GAP_REL
    assert np.median(_angle_deg(n_nn, n_ana)) < TAU_NORMAL_DEG


def test_cv1_hertz_neural_L1():
    """L1 MECHANICS: drive a 2-D FEM Hertz line contact with the trained NEURAL disc SDF as the
    rigid cylinder, and verify the recovered contact half-width a and peak pressure p0 satisfy the
    Hertz relations a=2sqrt(FR/piE*), p0=2F/pia for the measured line load F (manual §11.2 L1).
    Coarse mesh for test speed; the full-resolution result is in cv1_hertz_fem.py."""
    if load_neural_sdf("disc") is None:
        pytest.skip("no neural disc SDF yet — run atlas/sdf/train_analytical_sdf.py --shape disc")
    from benchmarks.contact.cv_numerical.cv1_hertz_fem import run
    m, _ = run(indenter="neural", n_x=80, n_y=40, verbose=False)
    assert m["a_relerr"] < 0.06        # contact half-width vs Hertz(F)
    assert m["p0_relerr"] < 0.06       # peak pressure vs Hertz(F)


# ---------------------------------------------------------------------------
# CV-2..CV-4 — same pattern (skeletons)
# ---------------------------------------------------------------------------

def test_cv2_cattaneo_neural_L1():
    pytest.skip("L1: with the neural normal-force, assert c/a=(1-Q/muP)^(1/3) and q(r) "
                "via cf.cattaneo_* to ~5%.")


def test_cv3_disc_neural_sdf_L0():
    """Neural disc SDF (the CV-3/CV-4 body), Euclidean gap/normal vs the closed-form circle
    SDF g=|xy|-R, embedded at z=0 (so the in-plane normal has n_z~0)."""
    sdf = load_neural_sdf("disc")
    if sdf is None:
        pytest.skip("no neural disc SDF yet — run atlas/sdf/train_analytical_sdf.py --shape disc")
    import torch
    from solvers.contact.gap import evaluate_gap
    R, L = 1.0, 1.0
    rng = np.random.RandomState(2)
    th = rng.uniform(0, 2 * np.pi, 2000)
    v = np.column_stack([np.cos(th), np.sin(th)])
    xy = v * (R + rng.uniform(-0.15, 0.15, size=(2000, 1)))   # near-contact shell
    x3 = np.column_stack([xy, np.zeros(len(xy))])
    r = np.linalg.norm(xy, axis=1)
    g_ana = r - R
    n_ana = xy / r[:, None]
    g_nn, n_nn = evaluate_gap(torch.tensor(x3, dtype=torch.float64), sdf)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    assert np.median(_angle_deg(n_nn[:, :2], n_ana)) < TAU_NORMAL_DEG   # PRIMARY: in-plane normal
    assert np.sqrt(np.mean((g_nn - g_ana) ** 2)) / L < TAU_GAP_DISC
    assert np.percentile(np.abs(n_nn[:, 2]), 90) < TAU_NZ   # disclose the prism z-leak tail


def test_cv3_brazilian_neural_L1():
    sdf = load_neural_sdf("disc")
    if sdf is None:
        pytest.skip("no neural disc chart yet — L1: center sxx=+2P/piDt, syy=-6P/piDt to ~5%.")


def test_cv4_nine_disc_neural_L1():
    """L1: the per-disc unit cell (D4 symmetry: 4 equal diametral neighbour loads) solved by the
    2-D FEM reproduces the analytical equibiaxial centre stress -2N/piRt, isotropic (sxx=syy,
    sxy=0).  Contact-free Neumann verification (the disc geometry is the neural disc SDF, L0-tested
    in test_cv3_disc_neural_sdf_L0); the full N-body contact array is a larger extension."""
    from benchmarks.contact.cv_numerical.cv4_nine_disc_fem import run
    m, _ = run(n_rings=48, verbose=False)
    assert m["sxx_relerr"] < 0.05 and m["syy_relerr"] < 0.05   # equibiaxial center vs -2N/piRt
    assert m["equibiaxial_anisotropy"] < 0.02                  # genuinely isotropic
    assert m["shear_rel"] < 0.02
    assert m["reaction_max"] < 1e-8                            # rigid-body pins are reaction-free


# ---------------------------------------------------------------------------
# CV-5 — superformula: the discriminating L0 test (concavities, cusps)
# ---------------------------------------------------------------------------

def test_cv5_supershape_neural_sdf_L0():
    """CV-5 L0 with CUSP-RELAXED, MEASURED tolerances. A smooth neural SDF *can* represent the
    cusped, concave superformula's gap and inside/outside, but only to looser bounds than a smooth
    shape — the spectral-bias difficulty CV-5 is designed to expose (manual §11.4). We assert the
    measured-achievable quality with honest UPPER bounds (a better chart still passes — this is NOT
    'must be worse than the disc'):
      * gap RMSE / L bounded and usable (the SDF is functional, not garbage);
      * inside/outside sign error low (it tells in from out);
      * in-plane normal angle bounded (degraded vs the ~0.5deg smooth shapes, but not random).
    The cusp degradation shows up most in the OUT-OF-PLANE normal tilt (|n_z| ~ 0.5, vs ~0.1 for
    the disc), reported but not gated; the ACCURATE CV-5 path is the neural RADIAL chart (M3)."""
    sdf = load_neural_sdf("supershape")
    if sdf is None:
        pytest.skip("no neural supershape SDF — run atlas/sdf/train_analytical_sdf.py --shape supershape")
    import torch
    from solvers.contact.gap import evaluate_gap
    p = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, scale=1.0)
    c = np.array([0.0, 0.0])
    L = ss.radius(np.linspace(0, 2 * np.pi, 400), p).max()
    rng = np.random.RandomState(1)
    thq = rng.uniform(0, 2 * np.pi, 1500)
    xy = ss.boundary(thq, c, 0.0, p) + rng.randn(1500, 2) * 0.12
    # dense Euclidean reference (gap + SDF-gradient normal), like-for-like with evaluate_gap
    thb = np.linspace(0, 2 * np.pi, 8000, endpoint=False)
    B = ss.boundary(thb, c, 0.0, p)
    d2 = np.sum((xy[:, None, :] - B[None, :, :]) ** 2, axis=2)
    k = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(xy)), k])
    inside = ss.inside(xy, c, 0.0, p)
    g_ana = np.where(inside, -dist, dist)
    v = xy - B[k]
    n_ana = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    n_ana[inside] *= -1.0
    x3 = np.column_stack([xy, np.zeros(len(xy))])
    g_nn, n_nn = evaluate_gap(torch.tensor(x3, dtype=torch.float64), sdf)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    gap_rel = float(np.sqrt(np.mean((g_nn - g_ana) ** 2)) / L)
    sign_err = float(np.mean((g_nn < 0) != inside))
    inplane_angle_med = float(np.median(_angle_deg(n_nn[:, :2], n_ana)))
    assert gap_rel < TAU_GAP_SUPERSHAPE              # usable gap (measured ~8e-3; cusp-relaxed)
    assert sign_err < 0.05                           # tells inside from outside (measured ~2%)
    assert inplane_angle_med < 12.0                  # in-plane normal degraded but not random (~2.5deg)


def test_cv5_supershape_neural_radial_chart_L0():
    """CV-5 — the ACCURATE chart path (the resolution of the SDF's cusp weakness above).

    A 2-D NEURAL RADIAL chart rho_theta(psi) (transition-map detector, Fourier-feature MLP) is
    compared to the analytical RADIAL reference supershape.radial_gap (like-for-like; the radial
    gap is biased 1/cos(alpha) vs Euclidean, so the radial chart is verified against the radial
    chart — manual §7/§11.2).  Fitting the 1-D boundary radius carries no ambient spectral bias and
    no medial axis, so the chart reproduces the analytical gap/normal to ~4e-3 with a sub-degree
    MEDIAN normal — markedly better than the neural SDF on the same cusped shape (the chart-over-
    level-set advantage, MEASURED)."""
    rho = load_neural_rho("supershape")
    if rho is None:
        pytest.skip("no neural radial chart yet — run atlas/charts/train_radial_chart.py")
    import torch
    from solvers.contact.radial_chart_2d import evaluate_radial_gap_2d
    p = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, a=1.0, b=1.0, scale=1.0)
    c = np.array([0.0, 0.0])
    L = ss.radius(np.linspace(0, 2 * np.pi, 1024), p).max()
    rng = np.random.RandomState(3)
    xy = ss.boundary(rng.uniform(0, 2 * np.pi, 1500), c, 0.0, p) + rng.randn(1500, 2) * 0.1
    g_ana, grad_ana = ss.radial_gap(xy, c, 0.0, p)
    n_ana = grad_ana / np.clip(np.linalg.norm(grad_ana, axis=1, keepdims=True), 1e-12, None)
    g_nn, n_nn = evaluate_radial_gap_2d(torch.tensor(xy, dtype=torch.float64), rho, center=(0, 0), alpha=0.0)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    gap_rel = float(np.sqrt(np.mean((g_nn - g_ana) ** 2)) / L)
    ang_med = float(np.median(_angle_deg(n_nn, n_ana)))
    assert gap_rel < 6e-3                             # accurate radial gap (measured ~3.8e-3)
    assert ang_med < 2.0                              # sub-degree median matched normal (measured ~0.4deg)
    assert np.mean((g_nn < 0) != ss.inside(xy, c, 0.0, p)) < 0.02   # active set matches


def test_cv5_supershape_neural_decoder_L0():
    """Neural ChartDecoder boundary vs analytical superformula boundary (boundary-to-boundary)."""
    dec = load_neural_decoder("supershape")
    if dec is None:
        pytest.skip("no neural supershape decoder yet — compare phi_nn(theta) vs phi_ana(theta).")


# ---------------------------------------------------------------------------
# CV-6 — Koch snowflake: the MEASURED neural-SDF refinement ceiling
# ---------------------------------------------------------------------------
# Unlike CV-1/CV-5 (which skip until a neural chart is supplied), CV-6 is RUNNABLE: the
# analytical Koch SDF is available on demand (sign from koch.inside, magnitude from
# koch.nearest_boundary), so the harness trains a fixed-capacity SDFNet on it directly.
# This turns the CV-6 prose claim ("a fixed net cannot keep resolving self-similar detail")
# from an assertion into a measurement — the experiment of manual §11.6, driven by
# benchmarks/contact/koch_neural_ceiling.py (which also produces figures/koch_neural_ceiling_pub.png).
#
# Tolerances are CV-6-specific and deliberately LOOSER than the smooth-shape neural targets
# above: the Koch boundary is all corners, so even a representable level carries larger gap /
# normal error than a sphere (this is exactly the spectral-bias mechanism CV-6 demonstrates).
TAU_GAP_KOCH = 0.07         # gap RMSE / L at a representable level (fixed net, corner-rich shape)
TAU_NORMAL_KOCH_DEG = 18.0  # median in-plane normal angle error at a representable level
TAU_NZ_KOCH = 0.30          # |n_z| (2-D shape embedded at z=0; in-plane normal)


@pytest.fixture(scope="module")
def koch_ceiling_runs():
    """Train ONE fixed-capacity SDFNet (shared across the CV-6 tests) on the exact Koch SDF at
    a representable level (n=1) and a fine level (n=4). Small dataset + full epochs keeps the
    representable level converged to its capacity floor (so the ceiling is visible) while the
    test stays a few minutes. Returns {level: metrics}."""
    pytest.importorskip("torch")
    from benchmarks.contact import koch_neural_ceiling as kc
    common = dict(width=64, depth=4, epochs=3000, n_near=600, n_bulk=300, n_eval=600, seed=0)
    return {n: kc.train_level(n, **common) for n in (1, 4)}


def test_cv6_koch_neural_sdf_L0(koch_ceiling_runs):
    """CV-6 L0 (runnable; mirrors the CV-1/CV-5 SDF L0 path): a FIXED-capacity neural SDF
    trained on the exact level-1 Koch signed distance reproduces koch.nearest_boundary on a
    held-out near-boundary set — zero-level-set deviation, near-band gap RMSE, in-plane normal
    angle, and a near-in-plane normal (the 2-D shape is embedded at z=0)."""
    m = koch_ceiling_runs[1]
    assert m["boundary_rmse_rel"] < TAU_GAP_KOCH             # net surface hugs the true boundary
    assert m["gap_rmse_rel"] < TAU_GAP_KOCH                  # representable level fits the SDF
    assert m["normal_angle_median_deg"] < TAU_NORMAL_KOCH_DEG
    assert m["max_abs_nz"] < TAU_NZ_KOCH                     # normal is essentially in-plane


def test_cv6_refinement_ceiling(koch_ceiling_runs):
    """CV-6 headline (MEASURED, not argued): with network capacity held FIXED, the zero-level-set
    deviation, the near-band gap RMSE, and the Eikonal residual vs the exact Koch SDF all RISE as
    the fractal level grows past what the net can represent. This earns the prose claim in
    koch.py / manual §8 & §11.6 (figure via benchmarks/contact/koch_neural_ceiling.py)."""
    lo, hi = koch_ceiling_runs[1], koch_ceiling_runs[4]
    assert hi["n_params"] == lo["n_params"]                 # capacity really was held fixed
    assert hi["n_segments"] > lo["n_segments"]              # while boundary detail grew (3*4^n)
    assert hi["boundary_rmse_rel"] > lo["boundary_rmse_rel"]  # the refinement ceiling (surface)
    assert hi["gap_rmse_rel"] > lo["gap_rmse_rel"]          # ... near-band gap
    assert hi["eik_residual_mean"] > lo["eik_residual_mean"]  # ... and the Eikonal residual


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs"]))
