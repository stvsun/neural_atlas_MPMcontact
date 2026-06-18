"""Pure-numpy analytical contact field evaluators (verification reference).

This module is the single numpy source of truth for the closed-form contact
solutions derived symbolically (and adversarially verified) in
``docs/hertz_derivation/{hertz_transition_map,brazilian_disc_atlas,nine_disc_atlas}.py``.
Those files stay SymPy-only (they assert the symbolic identities); this module
mirrors the same results in vectorized numpy so the plotting scripts can render
fields without importing SymPy.

All functions are vectorized over numpy arrays and have NO sympy / torch /
pyvista dependency.

References
----------
- K. L. Johnson, "Contact Mechanics" (1985): Hertz (Ch.3-4), 2D line-contact
  subsurface stress / McEwen (eq 4.49), Cattaneo-Mindlin (Ch.7).
- Timoshenko & Goodier, "Theory of Elasticity" Art.41: Flamant / Brazilian disc.
- Liu & Sun (2020), "ILS-MPM", CMAME 369:113168 (the figures being reproduced).

Caveat carried from the symbolic verification: the paper's Eq.46 line-contact
width is a factor of 4 low (16x in area); the standard Hertz width is used here.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Hertz normal contact (3D axisymmetric)
# ---------------------------------------------------------------------------

def hertz_3d_params(F, R, Estar):
    """Contact radius a, peak pressure p0, approach delta for a sphere pair.

    a = (3 F R / (4 E*))^{1/3},  p0 = 2 E* a/(pi R) = 3F/(2 pi a^2),  delta = a^2/R.
    1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2 ;  1/R = 1/R1 + 1/R2.
    """
    a = (3.0 * F * R / (4.0 * Estar)) ** (1.0 / 3.0)
    p0 = 2.0 * Estar * a / (np.pi * R)
    delta = a ** 2 / R
    return a, p0, delta


def hertz_pressure(r, a, p0):
    """Axisymmetric Hertz pressure p(r) = p0 sqrt(1 - r^2/a^2), zero for r>a."""
    r = np.asarray(r, dtype=float)
    inside = np.abs(r) <= a
    p = np.zeros_like(r)
    p[inside] = p0 * np.sqrt(np.clip(1.0 - (r[inside] / a) ** 2, 0.0, None))
    return p


def hertz_subsurface_axis(z, a, p0, nu):
    """On-axis subsurface stresses (Johnson eq 3.45), returned in stress units.

    Returns (sigma_z, sigma_r) with sigma_r = sigma_theta on the symmetry axis.
    Max principal shear ~0.31 p0 at z/a ~ 0.48 (nu=0.3); first yield p0 ~ 1.61 sigma_Y.
    """
    zeta = np.asarray(z, dtype=float) / a
    sig_z = -p0 / (1.0 + zeta ** 2)
    sig_r = -p0 * ((1.0 + nu) * (1.0 - zeta * np.arctan2(1.0, zeta))
                   - 0.5 / (1.0 + zeta ** 2))
    return sig_z, sig_r


# ---------------------------------------------------------------------------
# Hertz line contact (2D plane strain) + subsurface field (McEwen, Johnson 4.49)
# ---------------------------------------------------------------------------

def line_contact_params(Pline, R, Estar):
    """2D line-contact half-width a and peak pressure p0 (load per thickness Pline).

    a = 2 sqrt(Pline R/(pi E*)),  p0 = 2 Pline/(pi a) = sqrt(Pline E*/(pi R)).
    """
    a = 2.0 * np.sqrt(Pline * R / (np.pi * Estar))
    p0 = 2.0 * Pline / (np.pi * a)
    return a, p0


def line_contact_subsurface(x, z, a, p0):
    """2D Hertz line-contact subsurface stresses (McEwen; Johnson eq 4.49).

    Half-plane z>=0 loaded by a Hertzian pressure over [-a, a] at z=0.
    Returns (sigma_x, sigma_z, tau_xz). Finite everywhere in z>0; at the surface
    centre sigma_z = -p0. Use mirror z->|z| to fill the opposing body.
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(np.asarray(z, dtype=float))
    A = a ** 2 - x ** 2 + z ** 2
    root = np.sqrt(A ** 2 + 4.0 * x ** 2 * z ** 2)
    m = np.sqrt(np.clip(0.5 * (root + A), 0.0, None))
    n = np.sqrt(np.clip(0.5 * (root - A), 0.0, None)) * np.sign(x)
    den = m ** 2 + n ** 2
    den = np.where(den == 0.0, 1e-30, den)
    fac = p0 / a
    sig_x = -fac * (m * (1.0 + (z ** 2 + n ** 2) / den) - 2.0 * z)
    sig_z = -fac * (m * (1.0 - (z ** 2 + n ** 2) / den))
    tau_xz = -fac * (n * (m ** 2 - z ** 2) / den)
    return sig_x, sig_z, tau_xz


# ---------------------------------------------------------------------------
# Cattaneo-Mindlin frictional partial slip
# ---------------------------------------------------------------------------

def cattaneo_stick_radius(Q, mu, P, a):
    """Stick-zone radius c = a (1 - Q/(mu P))^{1/3} for Q < mu P."""
    return a * (1.0 - Q / (mu * P)) ** (1.0 / 3.0)


def cattaneo_traction(r, a, c, mu, p0):
    """Tangential traction q(r): full-slip annulus + inner stick correction."""
    r = np.asarray(r, dtype=float)
    q = np.zeros_like(r)
    out = np.abs(r) <= a
    q[out] = mu * p0 * np.sqrt(np.clip(1.0 - (r[out] / a) ** 2, 0.0, None))
    inner = np.abs(r) <= c
    q[inner] -= mu * p0 * (c / a) * np.sqrt(np.clip(1.0 - (r[inner] / c) ** 2, 0.0, None))
    return q


# ---------------------------------------------------------------------------
# Flamant point-load chart and disc superpositions (plane stress)
# ---------------------------------------------------------------------------

def flamant_tensor(X, Y, ax, ay, fx, fy, p, r_floor):
    """Flamant radial-stress tensor of an inward point load p at (ax, ay).

    Direction (fx, fy) is the (unit) inward force direction. Returns
    (sxx, syy, sxy). r is clamped to r_floor to keep the 1/r singularity finite.
    """
    dx = X - ax
    dy = Y - ay
    r2 = dx * dx + dy * dy
    r2 = np.maximum(r2, r_floor ** 2)
    fd = fx * dx + fy * dy                      # |d| cos(theta)
    srr_over_r2 = -(2.0 * p / np.pi) * fd / (r2 * r2)   # sigma_rr / r2
    sxx = srr_over_r2 * dx * dx
    syy = srr_over_r2 * dy * dy
    sxy = srr_over_r2 * dx * dy
    return sxx, syy, sxy


def brazilian_field(X, Y, R, P, t, r_floor=None):
    """Brazilian disc stress field: two Flamant loads + uniform biaxial tension.

    Loads P at (0, +-R) pointing inward; uniform tension T = 2P/(pi D t),
    D = 2R, makes the rim traction-free. Center: sxx = +2P/(pi D t),
    syy = -6P/(pi D t). Returns (sxx, syy, sxy).
    """
    if r_floor is None:
        r_floor = 0.01 * R
    p = P / t
    sxx_a, syy_a, sxy_a = flamant_tensor(X, Y, 0.0, R, 0.0, -1.0, p, r_floor)
    sxx_b, syy_b, sxy_b = flamant_tensor(X, Y, 0.0, -R, 0.0, 1.0, p, r_floor)
    T = p / (np.pi * R)                          # = 2P/(pi D t)
    sxx = sxx_a + sxx_b + T
    syy = syy_a + syy_b + T
    sxy = sxy_a + sxy_b
    return sxx, syy, sxy


def nine_disc_unit_cell_field(X, Y, R, N, t, r_floor=None):
    """Unit-cell field: disc under 4 equal inward loads N at N,S,E,W.

    = two perpendicular Brazilian solutions (4 Flamant + 2 uniform-tension).
    Center: sxx = syy = -2N/(pi R t) (equibiaxial in-plane compression).
    """
    if r_floor is None:
        r_floor = 0.01 * R
    p = N / t
    sxx = np.zeros_like(X)
    syy = np.zeros_like(X)
    sxy = np.zeros_like(X)
    loads = [(R, 0.0, -1.0, 0.0), (-R, 0.0, 1.0, 0.0),   # E, W
             (0.0, R, 0.0, -1.0), (0.0, -R, 0.0, 1.0)]    # N, S
    for ax, ay, fx, fy in loads:
        a, b, c = flamant_tensor(X, Y, ax, ay, fx, fy, p, r_floor)
        sxx = sxx + a
        syy = syy + b
        sxy = sxy + c
    T = 2.0 * p / (np.pi * R)                     # two perpendicular pairs
    return sxx + T, syy + T, sxy


def principal_stresses(sxx, syy, sxy):
    """In-plane principal stresses (S1>=S2) and max in-plane shear."""
    mean = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    return mean + rad, mean - rad, rad


# ---------------------------------------------------------------------------
# Brazilian a-F curve (contact half-width vs load) -- Fig.22
# ---------------------------------------------------------------------------

def brazilian_contact_halfwidth(F, R, E, nu, t):
    """Disc-on-rigid-platen Hertz contact half-width a(F) (standard 2D form).

    a = 2 sqrt(P' R/(pi E*)),  P' = F/t,  E* = E/(1-nu^2)  (one elastic body).
    NB: the paper's Eq.46 is a factor 4 low; this is the physically correct width.
    """
    Estar = E / (1.0 - nu ** 2)
    Pline = F / t
    return 2.0 * np.sqrt(Pline * R / (np.pi * Estar))


# ---------------------------------------------------------------------------
# Self-test against the verified symbolic targets
# ---------------------------------------------------------------------------

def _self_test():
    ok = True

    def check(name, got, want, tol=1e-9):
        nonlocal ok
        rel = abs(got - want) / (abs(want) + 1e-30)
        flag = "OK " if rel < tol else "FAIL"
        if rel >= tol:
            ok = False
        print(f"  [{flag}] {name}: got {got:.8g}  want {want:.8g}  (rel {rel:.1e})")

    print("Hertz 3D (F=5, R=2, E*=100):")
    a, p0, delta = hertz_3d_params(5.0, 2.0, 100.0)
    check("a = (3FR/4E*)^1/3", a, (3 * 5 * 2 / 400) ** (1 / 3))
    check("p0 = 3F/(2 pi a^2)", p0, 3 * 5 / (2 * np.pi * a ** 2))
    check("delta = a^2/R", delta, a ** 2 / 2.0)
    F_int = np.trapz(hertz_pressure(np.linspace(0, a, 20001), a, p0)
                     * 2 * np.pi * np.linspace(0, a, 20001), np.linspace(0, a, 20001))
    check("int p dA = F", F_int, 5.0, tol=1e-3)

    print("Hertz subsurface (nu=0.3): max shear depth & magnitude")
    zz = np.linspace(1e-4, 3, 400000)
    sz, sr = hertz_subsurface_axis(zz, 1.0, 1.0, 0.3)
    tau = 0.5 * np.abs(sz - sr)
    i = int(np.argmax(tau))
    check("z/a of max shear", zz[i], 0.481, tol=2e-2)
    check("tau_max/p0", tau[i], 0.31, tol=2e-2)

    print("Hertz line contact (P'=10, R=2, E*=100):")
    aL, p0L = line_contact_params(10.0, 2.0, 100.0)
    check("a^2 = 4 P R/(pi E*)", aL ** 2, 4 * 10 * 2 / (np.pi * 100))
    check("p0 = 2P/(pi a)", p0L, 2 * 10 / (np.pi * aL))
    # surface centre sigma_z = -p0
    _, szc, _ = line_contact_subsurface(np.array([0.0]), np.array([0.0]), aL, p0L)
    check("surface centre sigma_z", szc[0], -p0L, tol=1e-6)

    print("Brazilian disc centre (P=1, R=1, t=1):")
    sxx, syy, sxy = brazilian_field(np.array([0.0]), np.array([0.0]), 1.0, 1.0, 1.0)
    D = 2.0
    check("sxx = +2P/(pi D t)", sxx[0], 2 / (np.pi * D))
    check("syy = -6P/(pi D t)", syy[0], -6 / (np.pi * D))
    check("sxy = 0", sxy[0] + 1.0, 1.0)            # offset to avoid /0

    print("Nine-disc unit cell centre (N=1, R=1, t=1):")
    sxx, syy, sxy = nine_disc_unit_cell_field(np.array([0.0]), np.array([0.0]), 1.0, 1.0, 1.0)
    check("sxx = -2N/(pi R t)", sxx[0], -2 / np.pi)
    check("syy = -2N/(pi R t)", syy[0], -2 / np.pi)
    check("equibiaxial sxx=syy", sxx[0], syy[0])

    print("Cattaneo-Mindlin stick radius (Q=0.25, mu=0.5, P=1, a=1):")
    c = cattaneo_stick_radius(0.25, 0.5, 1.0, 1.0)
    check("c/a = (1-Q/muP)^1/3", c, (1 - 0.25 / 0.5) ** (1 / 3))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    _self_test()
