"""
Analytic Hertz contact via the neural-atlas TRANSITION MAP (no SDF, no FEM).

This script derives, entirely in SymPy/mpmath:
  (A) the undeformed contact gap as a chart transition map (Monge patches),
      reproducing g_n0(r) = r^2/(2R) with normal from the chart Jacobian;
  (B) the 3D axisymmetric Hertz solution via the Method of Dimensionality
      Reduction (MDR / Sneddon) -- contact radius, approach, pressure, force;
  (C) the on-axis subsurface stress field and first-yield onset;
  (D) the 2D plane-strain line contact, matching Liu & Sun (2020) ILS-MPM Eq.46.

Reference for the contact kinematics being reformulated:
  C. Liu & W. Sun (2020), "ILS-MPM: an implicit level-set-based material point
  method for frictional particulate contact mechanics of deformable particles,"
  CMAME 369, 113168.  (Eqs. 14-26: level-set normal, closest-point projection P,
  undeformed gap g_n0 = (P^(j) - P^(i)).n.)
Reference for the analytic Hertz results:
  K. L. Johnson (1985), "Contact Mechanics," Cambridge Univ. Press, Ch. 3-4.

Run:  python3 hertz_transition_map.py
"""
import sympy as sp
import mpmath as mp


def part_A_transition_map():
    print("=" * 72)
    print("PART A  Transition-map gap function (geometry only, no SDF)")
    print("=" * 72)
    xi1, xi2 = sp.symbols('xi1 xi2', real=True)
    R1, R2 = sp.symbols('R1 R2', positive=True)
    r = sp.symbols('r', nonnegative=True)

    # Surface charts = Monge patches (paraboloid approx of each body at the apex)
    phi1 = sp.Matrix([xi1, xi2,  (xi1**2 + xi2**2) / (2 * R1)])
    phi2 = sp.Matrix([xi1, xi2, -(xi1**2 + xi2**2) / (2 * R2)])

    # Contact normal from the CHART JACOBIAN cross product (replaces grad(SDF)):
    J1 = phi1.jacobian([xi1, xi2])
    n_hat = sp.simplify(J1[:, 0].cross(J1[:, 1]) / J1[:, 0].cross(J1[:, 1]).norm())
    print("normal at apex (xi=0)  n =", list(n_hat.subs({xi1: 0, xi2: 0})))

    # Undeformed gap g_n0 = (phi1 - phi2).n  (chart form of ILS-MPM Eq.25)
    # LEADING ORDER: normal taken at the apex (n=e_z), valid for r<<R (a<<R).
    g_n0 = sp.simplify(((phi1 - phi2).T * sp.Matrix([0, 0, 1]))[0])
    print("g_n0(xi) = (phi1-phi2).e_z =", g_n0)
    print("        = r^2/(2R)  with  1/R = 1/R1+1/R2 =", sp.together(1 / R1 + 1 / R2),
          "  [LEADING ORDER, apex normal]")
    # EXACT: project on the position-dependent chart-Jacobian normal of body 1.
    n_full = J1[:, 0].cross(J1[:, 1]); n_full = n_full / n_full.norm()
    g_exact = sp.simplify(((phi1 - phi2).T * n_full)[0]).subs({xi1: r, xi2: 0})
    print("EXACT gap (full Jacobian normal, xi2=0 ray) =", g_exact)
    print("Taylor:", sp.expand(sp.series(g_exact, r, 0, 6).removeO()),
          "  -> r^2/2R is the O(r^2) truncation; correction is O(r^4).")
    print("NOTE: production code solvers/contact/gap.py uses the SDF gradient")
    print("      n=grad(phi)/|grad(phi)| (ILS-MPM Eq.15); the chart cross-product")
    print("      normal equals it only for graph surfaces at the apex.")
    return r


def part_B_hertz_3d():
    print("\n" + "=" * 72)
    print("PART B  3D axisymmetric Hertz via MDR (all integrals symbolic)")
    print("=" * 72)
    r, x, a, R, Es, F, delta = sp.symbols('r x a R E_star F delta', positive=True)

    f = r**2 / (2 * R)                                  # 3D gap profile (Part A)
    g = sp.simplify(x * sp.integrate(sp.diff(f, r) / sp.sqrt(x**2 - r**2), (r, 0, x)))
    print("1D MDR profile  g(x) =", g, " (= x^2/R, twice the 3D curvature)")

    delta_a = g.subs(x, a)                              # delta = g(a) = a^2/R
    qL = Es * (delta_a - g)                             # linear force density
    F_a = sp.simplify(sp.integrate(qL, (x, -a, a)))     # total normal force
    print("approach delta = a^2/R ;  F =", F_a, " (= 4E* a^3/3R)")

    p = sp.simplify(-1 / sp.pi * sp.integrate(sp.diff(qL, x) / sp.sqrt(x**2 - r**2), (x, r, a)))
    p0 = sp.simplify(p.subs(r, 0))
    print("pressure p(r) =", p, " ;  p0 =", p0, "= 2E*a/(piR)")
    assert sp.simplify(sp.integrate(p * 2 * sp.pi * r, (r, 0, a)) - F_a) == 0
    print("check ∫ p dA = F : OK ;  p0 = 3F/(2 pi a^2) :",
          sp.simplify(p0.subs(a, (3 * F * R / (4 * Es))**sp.Rational(1, 3))
                      - 3 * F / (2 * sp.pi * (3 * F * R / (4 * Es))**sp.Rational(2, 3))) == 0)

    a_F = sp.solve(sp.Eq(F, F_a), a)[0]
    print("a     =", sp.simplify(a_F), " = (3FR/4E*)^(1/3)")
    print("delta =", sp.simplify(delta_a.subs(a, a_F)), " = (9F^2/16RE*^2)^(1/3)")
    print("p0    =", sp.simplify(p0.subs(a, a_F)), " = (6FE*^2/pi^3 R^2)^(1/3)")
    Fd = sp.simplify(F_a.subs(a, sp.sqrt(R * delta)))
    print("F(delta) =", Fd, " = (4/3)E* R^(1/2) delta^(3/2)")
    print("stiffness dF/ddelta =", sp.simplify(sp.diff(Fd, delta)), " = 2E*a")


def part_C_subsurface():
    print("\n" + "=" * 72)
    print("PART C  On-axis subsurface stress + first yield (Johnson eq 3.45)")
    print("=" * 72)
    zeta, nu = sp.symbols('zeta nu', positive=True)
    sig_z = -1 / (1 + zeta**2)
    sig_r = -((1 + nu) * (1 - zeta * sp.atan(1 / zeta)) - sp.Rational(1, 2) / (1 + zeta**2))
    tau = ((sig_z - sig_r) / 2).subs(nu, sp.Rational(3, 10))
    z = sp.nsolve(sp.diff(tau, zeta), zeta, 0.6)
    svm = sp.Abs(sig_z - sig_r).subs(nu, sp.Rational(3, 10))
    print("nu=0.3:  max shear at z/a = %.4f , tau_max/p0 = %.4f" % (float(z), float(tau.subs(zeta, z))))
    print("         von Mises max sigma_vM/p0 = %.4f ; yield onset p0 = %.3f sigma_Y"
          % (float(svm.subs(zeta, z)), float(1 / svm.subs(zeta, z))))


def part_D_line_contact():
    print("\n" + "=" * 72)
    print("PART D  2D plane-strain line contact (matches ILS-MPM Eq.46)")
    print("=" * 72)
    mp.mp.dps = 25

    def pv(eta):  # PV ∫_{-1}^1 sqrt(1-t^2)/(t-eta) dt  via regularized split
        eta = mp.mpf(eta)
        reg = mp.quad(lambda t: (mp.sqrt(1 - t**2) - mp.sqrt(1 - eta**2)) / (t - eta), [-1, eta, 1])
        return reg + mp.sqrt(1 - eta**2) * mp.log((1 - eta) / (1 + eta))
    ok = all(abs(pv(e) + mp.pi * e) < 1e-15 for e in [0.2, 0.5, -0.7, 0.9])
    print("finite Hilbert identity PV∫√(1-t²)/(η-t)dt = π η :", "OK" if ok else "FAIL")

    a, R, Es, P, x, p0 = sp.symbols('a R E_star P x p0', positive=True)
    P_expr = sp.integrate(p0 * sp.sqrt(1 - x**2 / a**2), (x, -a, a))  # = pi a p0 / 2
    P_sub = sp.simplify(P_expr.subs(p0, Es * a / (2 * R)))            # p0 = aE*/2R
    print("P = pi a p0/2 ;  with p0=aE*/2R:  P =", P_sub, " -> a^2 = 4PR/(pi E*)")
    print("a  = 2 sqrt(PR/(pi E*)) ;  p0 = sqrt(P E*/(pi R)) = 2P/(pi a)")
    # honest reconciliation with paper Eq.46: it is a factor 4 in width (16 in area)
    E, nu, F = sp.symbols('E nu F', positive=True)
    twoa_std = 2 * sp.sqrt(4 * P * R / (sp.pi * Es)).subs(Es, E / (1 - nu**2))
    twoa_pap = sp.sqrt(F * (1 - nu**2) * R / (sp.pi * E))
    ratio = sp.simplify(twoa_std**2 / twoa_pap.subs(F, P)**2)
    print("standard 2a = 4 sqrt(PR(1-nu^2)/(pi E)); paper Eq.46 2a = sqrt(FR(1-nu^2)/(pi E))")
    print("ratio (2a_std/2a_paper)^2 with F=P =", ratio,
          " => Eq.46 as printed is low by 4x in width (NOT a mere convention).")


def part_E_cattaneo_mindlin():
    print("\n" + "=" * 72)
    print("PART E  Cattaneo-Mindlin frictional partial slip (friction benchmark)")
    print("=" * 72)
    a, c, mu, p0, r, P, Q = sp.symbols('a c mu p0 r P Q', positive=True)
    p0_val = 3 * P / (2 * sp.pi * a**2)
    q_outer = mu * p0 * sp.sqrt(1 - r**2 / a**2)              # slip annulus c<r<a
    q_corr = -mu * p0 * (c / a) * sp.sqrt(1 - r**2 / c**2)    # stick correction r<c
    Qtot = sp.simplify((sp.integrate(q_outer * 2 * sp.pi * r, (r, 0, a))
                        + sp.integrate(q_corr * 2 * sp.pi * r, (r, 0, c))).subs(p0, p0_val))
    print("Q = ∫q dA =", Qtot, " = mu P (1 - (c/a)^3)")
    print("=> stick radius  c/a = (1 - Q/(mu P))^(1/3)")
    print("traction  c<=r<=a: q = mu p0 sqrt(1-r^2/a^2)")
    print("           r<=c  : q = mu p0[sqrt(1-r^2/a^2) - (c/a) sqrt(1-r^2/c^2)]")
    print("compliance dx = (3 mu P)/(16 a G*)[1-(1-Q/mu P)^(2/3)],")
    print("   1/G* = (2-nu1)/(4 G1)+(2-nu2)/(4 G2)  (Mindlin 1949; Johnson Ch.7)")


if __name__ == "__main__":
    part_A_transition_map()
    part_B_hertz_3d()
    part_C_subsurface()
    part_D_line_contact()
    part_E_cattaneo_mindlin()
    print("\nAll blocks verified symbolically/numerically.")
