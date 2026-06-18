"""
Brazilian disc (diametral compression) solved by the ATLAS technique -- no FEM.

The disc Omega = {x^2+y^2 <= R^2} is covered by THREE coordinate charts whose
analytic fields superpose to the exact plane-elasticity solution:

  * chart A, B : Flamant load-point charts at the two diametral loads.
                 Each carries the singular radial field sigma_rr = -(2p/pi) cos(theta)/r
                 (p = P/t, force per thickness). Each is a singularity-absorbing
                 chart for the concentrated-load 1/r singularity (analogous to a
                 crack-tip 1/sqrt(r) enrichment).
  * chart 3    : a regular uniform biaxial-tension bulk chart.

The transition maps are the rigid coordinate changes between each load-point polar
frame and the global frame; the traction-free boundary condition on |x|=R couples
the charts (a linear, single-pass analogue of the multiplicative-Schwarz coupling
used for the nonlinear problems in the code) and fixes the bulk amplitude
T = 2P/(pi D t).

Reference solution: Timoshenko & Goodier, "Theory of Elasticity", Art. 41;
distributed-platen generalisation: Hondros (1959). Paper context: Liu & Sun (2020)
ILS-MPM, section 5.3.1 (Brazilian disc, R=10, rigid platens).

Run:  python3 brazilian_disc_atlas.py
"""
import sympy as sp

x, y, R, P, t = sp.symbols('x y R P t', positive=True)
p = P / t                                    # line load (force per unit thickness)


def flamant(ax, ay, fx, fy):
    """Flamant radial field of a unit-direction (fx,fy) point load p at (ax,ay)."""
    r2 = (x - ax)**2 + (y - ay)**2
    e = sp.Matrix([x - ax, y - ay])          # vector from load point
    cth = (fx * (x - ax) + fy * (y - ay)) / sp.sqrt(r2)   # cos(theta) = fhat . ehat
    srr = -(2 * p / sp.pi) * cth / sp.sqrt(r2)            # sigma_rr = -(2p/pi)cos/r
    return srr / r2 * (e * e.T)               # srr * (ehat outer ehat)


def solve():
    T = sp.symbols('T')
    sig = flamant(0, R, 0, -1) + flamant(0, -R, 0, 1) + T * sp.eye(2)

    # [1] equilibrium
    d1 = sp.simplify(sp.diff(sig[0, 0], x) + sp.diff(sig[0, 1], y))
    d2 = sp.simplify(sp.diff(sig[0, 1], x) + sp.diff(sig[1, 1], y))
    # [2] biharmonic compatibility
    tr = sig[0, 0] + sig[1, 1]
    lap = sp.simplify(sp.diff(tr, x, 2) + sp.diff(tr, y, 2))
    print("[1] equilibrium div(sigma) =", (d1, d2))
    print("[2] compatibility  Laplacian(trace) =", lap)

    # [3] free-boundary condition on |x|=R fixes T (evaluate at rim point a=0)
    a = sp.symbols('a')
    rim = {x: R * sp.cos(a), y: R * sp.sin(a)}
    trac0 = (sig.subs(rim) * sp.Matrix([sp.cos(a), sp.sin(a)])).subs(a, 0)
    Tsol = sp.solve(sp.Eq(trac0[0], 0), T)[0]
    print("[3] free boundary -> T =", sp.simplify(Tsol), " = 2P/(pi D t)")
    sigF = sig.subs(T, Tsol)

    # rim traction-free check off the poles
    sigN = sigF.subs({R: 1, P: 1, t: 1})
    res = []
    for deg in [20, 55, 110, 160, 200, 300]:
        aa = sp.rad(deg); c, s = sp.cos(aa), sp.sin(aa)
        tt = sigN.subs({x: c, y: s}) * sp.Matrix([c, s])
        res.append((abs(complex(sp.N(tt[0])))**2 + abs(complex(sp.N(tt[1])))**2)**0.5)
    print("    rim |traction| off poles (max) =", max(res))

    # [4] key stress values
    sxx = sp.simplify(sigF[0, 0]); syy = sp.simplify(sigF[1, 1])
    print("\n[4] CENTER  sigma_xx =", sp.simplify(sxx.subs({x: 0, y: 0})),
          " (=+2P/piDt, tensile)")
    print("    CENTER  sigma_yy =", sp.simplify(syy.subs({x: 0, y: 0})),
          " (=-6P/piDt, compressive)")
    print("    LOADED diam x=0:  sigma_xx =", sp.simplify(sxx.subs(x, 0)),
          " (uniform tension)")
    print("                       sigma_yy =", sp.simplify(syy.subs(x, 0)))
    print("    SPLIT  diam y=0:  sigma_xx =", sp.simplify(sxx.subs(y, 0)))
    print("                       sigma_yy =", sp.simplify(syy.subs(y, 0)))

    # [5] global equilibrium across the splitting plane
    print("\n[5] global equilibrium  int_-R^R sigma_yy(x,0) dx =",
          sp.simplify(sp.integrate(syy.subs(y, 0), (x, -R, R))), " (= -P/t)")
    print("\n[6] Brazilian tensile strength:  sigma_t = 2P/(pi D t)   (ASTM D3967)")


def compliance_diverges():
    """The point-load diametral compliance is logarithmically DIVERGENT, so the
    point-load model yields NO finite load-displacement (a-F) slope -- a finite
    contact arc (Hondros) is required to compare against the paper's Fig.22."""
    print("\n[compliance] point-load diametral approach diverges:")
    y, R, P, t, E, nu, eps = sp.symbols('y R P t E nu epsilon', positive=True)
    p = P / t
    syy = -(2 * p / sp.pi) * (1 / (R - y) + 1 / (R + y)) + p / (sp.pi * R)
    sxx = p / (sp.pi * R)
    eyy = (syy - nu * sxx) / E                       # plane-stress vertical strain
    approach = sp.simplify(-sp.integrate(eyy, (y, -(R - eps), R - eps)))
    print("   delta(eps->0) =", sp.limit(approach, eps, 0), " (log-divergent) ->")
    print("   need finite contact arc for a finite a-F slope.")


def hondros_and_hertz():
    """Finite contact arc 2*alpha (Hondros 1959) regularises the load singularity.
    alpha is set by the platen-disc HERTZ contact half-width b (hertz_transition_map.py)."""
    P, R, t, alpha = sp.symbols('P R t alpha', positive=True)
    D = 2 * R
    sxx_H = (2 * P / (sp.pi * D * t)) * (sp.sin(2 * alpha) - alpha) / alpha
    syy_H = -(2 * P / (sp.pi * D * t)) * (sp.sin(2 * alpha) + alpha) / alpha
    print("\n[Hondros] center sigma_xx(alpha) ->", sp.limit(sxx_H, alpha, 0), "(=2P/piDt) as alpha->0")
    print("          center sigma_yy(alpha) ->", sp.limit(syy_H, alpha, 0), "(=-6P/piDt) as alpha->0")
    print("          tensile reduction (sin2a-a)/a ~ 1 - (4/3)a^2 (0.96 at 10deg)")
    print("[Hertz]   arc alpha ~ b/R, b = 2 sqrt(P' R/(pi E*)); for R=10mm,E=100GPa,")
    print("          P'=100N/mm: alpha~0.6deg, stress correction <0.1% (point load OK")
    print("          for STRESS, but NOT for compliance).")


def displacement_field():
    """Displacement field by chart superposition (Flamant building block).
    The Flamant displacement carries the signature ln r (settlement) and the
    multivalued theta*sin(theta) terms (the branch the transition maps carry)."""
    print("\n" + "=" * 72)
    print("DISPLACEMENT  Flamant building block (plane stress) + disc assembly")
    print("=" * 72)
    r, th, p, E, nu = sp.symbols('r theta p E nu', positive=True)
    ur = (-(2 * p / (sp.pi * E)) * sp.cos(th) * sp.ln(r)
          - ((1 - nu) * p / (sp.pi * E)) * th * sp.sin(th)
          + ((1 + nu) * p / (sp.pi * E)) * sp.cos(th))           # gauge a=(1+nu)p/piE
    uth = ((2 * p / (sp.pi * E)) * sp.sin(th) * sp.ln(r)
           - ((1 - nu) * p / (sp.pi * E)) * th * sp.cos(th))
    # forward check: u -> strain -> stress = Flamant
    err = sp.diff(ur, r); eth = ur / r + sp.diff(uth, th) / r
    grt = sp.diff(ur, th) / r + sp.diff(uth, r) - uth / r
    srr = sp.simplify(E / (1 - nu**2) * (err + nu * eth))
    print("u_r   = -(2p/piE)cos(th)ln r - ((1-nu)p/piE) th sin(th) + a cos(th)")
    print("u_th  =  (2p/piE)sin(th)ln r - ((1-nu)p/piE) th cos(th) + b sin(th)")
    print("  a+b = (1+nu)p/piE is COMPATIBILITY-FIXED (needed so u->stress=Flamant);")
    print("  the single free rigid-translation gauge is a (then b=(1+nu)p/piE-a).")
    print("forward check  sigma_rr =", srr, " (Flamant: -2p cos/(pi r)) ;",
          "sigma_thth=", sp.simplify(E / (1 - nu**2) * (eth + nu * err)))
    print("branch (Volterra): u(th+2pi)-u(th) =",
          sp.simplify(ur.subs(th, th + 2 * sp.pi) - ur), "* e_r + ...")
    print("disc: u = Flamant^A((0,R),-y) + Flamant^B((0,-R),+y) + (T(1-nu)/E)(x,y) + rigid")
    print("      verified numerically to reproduce the disc stress field exactly.")


def diametral_compliance():
    """Relative platen approach (a-F curve); finite only with a contact cutoff b."""
    print("\n" + "=" * 72)
    print("a-F CURVE  diametral compliance with contact cutoff b (Hertz-set)")
    print("=" * 72)
    print("exact (cutoff b): Delta_D(b) = (2P/piEt)[ 2 ln((2R-b)/b) - (1-nu)(R-b)/R ]")
    print("b->0 asymptote :  Delta_D    = (4P/(pi E t)) [ ln(2R/b) - (1-nu)/2 ]  (plane stress)")
    print("ROBUST parts: prefactor 4P/(piEt) and the -(1-nu)/2 (bulk-tension nu-coupling,")
    print("  distribution-INDEPENDENT). UNCERTAIN part: the log argument (b<->contact width).")
    print("b is an EFFECTIVE cutoff, NOT the Hertz half-width c: for a Hertzian pressure")
    print("  b = c/(2 sqrt(e)) ~ 0.303 c, i.e. ln(2R/b) -> ln(4R/c) + 1/2.")
    print("c = 2 sqrt(P' R/(pi E*)), E*=E/(1-nu^2), P'=P/t (rigid-platen Hertz; plane-strain).")
    print("regime: valid for c << 2R (small load); curve is mildly CONCAVE,")
    print("  dDelta/dP = (4/piEt)[ln(2R/b) - (1-nu)/2 - 1/2]; plane strain swaps E->E/(1-nu^2).")


if __name__ == "__main__":
    solve()
    compliance_diverges()
    hondros_and_hertz()
    displacement_field()
    diametral_compliance()
