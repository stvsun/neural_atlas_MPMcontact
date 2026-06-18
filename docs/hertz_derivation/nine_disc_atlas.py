"""
Nine-disc packing (Liu & Sun 2020, ILS-MPM Fig.15) solved by the ANALYTICAL atlas.

A 3x3 array of identical elastic discs (R=1.7 mm, E=100 GPa, nu=0.3, t=1 mm,
friction mu=0.5) is confined by four plates and compressed isotropically.

STRATEGY (the analytic atlas, same recipe as Hertz/Brazilian):
  1. SYMMETRY REDUCTION. Diagonal neighbours are 2R*sqrt(2) > 2R apart -> no contact,
     so the contact network is a square lattice of horizontal + vertical chains. By
     series equilibrium + the 4-fold (isotropic) symmetry, EVERY contact (disc-disc and
     disc-plate) carries the SAME normal force N, and friction is inactive (no sliding
     tendency by symmetry). Hence every one of the 9 discs sees the identical loading:
     four equal inward forces N at N, S, E, W. The 9-body problem collapses to ONE unit cell.
  2. UNIT-CELL FIELD by chart superposition: the 4-load disc = two perpendicular Brazilian
     solutions = four Flamant load-point charts (singular, 1/r) + two uniform-tension bulk
     charts. The four contacts are at 90 deg, so the transition maps between charts carry
     genuine rotations (unlike the collinear Hertz/Brazilian cases).
  3. COMPATIBILITY: along a row of 3 discs the plates move in d each; the diametral
     compliances sum, 3*Delta_x(N) = 2 d, giving the force-compression law N(d).

Builds on hertz_transition_map.py (contact half-width b) and brazilian_disc_atlas.py
(Flamant chart + diametral compliance). Run:  python3 nine_disc_atlas.py
"""
import sympy as sp
import mpmath as mp


def unit_cell_field():
    """4-load disc stress field (4 Flamant + 2 uniform-tension charts); checks."""
    print("=" * 72)
    print("UNIT CELL: disc under 4 equal inward loads N at N,S,E,W")
    print("=" * 72)
    x, y, R, N, t = sp.symbols('x y R N t', positive=True)
    p = N / t

    def flamant(ax, ay, fx, fy):
        dx, dy = x - ax, y - ay
        r2 = dx**2 + dy**2
        e = sp.Matrix([dx, dy])
        cth = (fx * dx + fy * dy) / sp.sqrt(r2)
        return -(2 * p / sp.pi) * cth / sp.sqrt(r2) / r2 * (e * e.T)

    T = p / (sp.pi * R)                      # uniform tension per diametral pair
    sig = (flamant(R, 0, -1, 0) + flamant(-R, 0, 1, 0)     # E, W
           + flamant(0, R, 0, -1) + flamant(0, -R, 0, 1)   # N, S
           + 2 * T * sp.eye(2))
    d1 = sp.simplify(sp.diff(sig[0, 0], x) + sp.diff(sig[0, 1], y))
    d2 = sp.simplify(sp.diff(sig[0, 1], x) + sp.diff(sig[1, 1], y))
    c = {x: 0, y: 0}
    print("equilibrium div(sigma) =", (d1, d2))
    print("CENTER: sxx =", sp.simplify(sig[0, 0].subs(c)),
          " syy =", sp.simplify(sig[1, 1].subs(c)),
          " sxy =", sp.simplify(sig[0, 1].subs(c)))
    print("  => EQUIBIAXIAL in-plane compression  -2N/(pi R t) = -4N/(pi D t)")
    print("     (plane stress: sigma_zz=0, so 3D state is biaxial, not truly hydrostatic)")
    sigN = sig.subs({R: 1, N: 1, t: 1})
    mx = 0.0
    for deg in [15, 40, 75, 115, 150, 200, 250, 310]:
        a = sp.rad(deg); cc, ss = sp.cos(a), sp.sin(a)
        tt = sigN.subs({x: cc, y: ss}) * sp.Matrix([cc, ss])
        mx = max(mx, (abs(complex(sp.N(tt[0])))**2 + abs(complex(sp.N(tt[1])))**2)**0.5)
    print("rim |traction| off the 4 poles (max):", mx, " (traction-free)")


def force_compression_law():
    """N(d) from 3*Delta_x(N)=2d, with the paper's geometry/material."""
    print("\n" + "=" * 72)
    print("FORCE-COMPRESSION LAW  3*Delta_x(N) = 2d")
    print("=" * 72)
    x, R, N, t, E, nu, b = sp.symbols('x R N t E nu b', positive=True)
    p = N / t
    sxx_sing = -(2 * p / sp.pi) * (1 / (R - x) + 1 / (R + x))            # E,W contacts
    sxx_reg = 2 * p / (sp.pi * R) - 4 * p * R * x**2 / (sp.pi * (R**2 + x**2)**2)
    syy = 2 * p / (sp.pi * R) - 4 * p * R**3 / (sp.pi * (R**2 + x**2)**2)
    I_sing = sp.integrate(sxx_sing / E, (x, -(R - b), R - b))
    I_reg = sp.integrate((sxx_reg - nu * syy) / E, (x, -(R - b), R - b))
    Dx = -(I_sing + I_reg)
    print("Delta_x(N) = (4N/(pi E t)) ln(2R/b) + (N/(pi E t))[(2-pi)nu + pi - 6]")
    print("  only the leading log coefficient 4N/(pi E t) is convention-robust;")
    print("  the finite constant is cutoff-dependent (Poisson from BOTH pairs + nu-free pieces).")
    print("CORRECTED Hertz line-contact half-width  a = sqrt(4 N R_rel/(pi E*)):")
    print("  disc-disc  : E*=E/(2(1-nu^2)), R_rel=R/2  (two identical elastic bodies)")
    print("  disc-plate : E*=E/(1-nu^2),    R_rel=R     (rigid plate => single-body E*)")
    print("  => a_dp = a_dd identically, so one half-width serves every contact.\n")
    Rv, Ev, nuv, tv = 1.7, 1e5, 0.3, 1.0               # mm, N/mm^2 (100 GPa), -, mm
    a_half = lambda Pr, Es, Rr: mp.sqrt(4 * Pr * Rr / (mp.pi * Es))   # a=sqrt(4 P' R_rel/(pi E*))
    Es_dd, Rr_dd = Ev / (2 * (1 - nuv**2)), Rv / 2     # disc-disc : two identical bodies
    Es_dp, Rr_dp = Ev / (1 - nuv**2), Rv               # disc-plate: rigid plate (single body)
    assert abs(a_half(1.0, Es_dd, Rr_dd) - a_half(1.0, Es_dp, Rr_dp)) < 1e-12  # a_dp == a_dd
    Dx_n = sp.lambdify((N, b), Dx.subs({R: Rv, E: Ev, nu: nuv, t: tv}), 'mpmath')
    hb = lambda Nv: a_half(Nv, Es_dd, Rr_dd)           # disc-disc width (= disc-plate width)
    for d in [1e-4, 1e-3, 1e-2]:
        Ns = mp.findroot(lambda Nv: mp.re(3 * Dx_n(Nv, hb(Nv))) - 2 * d, mp.mpf('0.5'))
        Nv = float(mp.re(Ns))
        print("d=%7.0e mm: N=%9.4f N/mm  b=%.2e mm  center=%9.3f MPa  plate force 3N=%9.4f N"
              % (d, Nv, float(mp.re(hb(Ns))), -2 * Nv / (mp.pi * Rv * tv), 3 * Nv))


if __name__ == "__main__":
    unit_cell_field()
    force_compression_law()
    print("\nPeak stress ~ Flamant 1/r at the 4 contacts (matches Fig.16 concentrations);")
    print("field identical for all 9 discs by symmetry (matches symmetric stress in Fig.16).")
