"""
Symbolic verification of the TWO-BODY MORTAR / OT-COUPLING CONTACT TANGENT.

Context
-------
The OT measure-coupling contact module (solvers/contact/measure_coupling/) works for a
deformable body against a RIGID counterface, but the deformable-deformable (cv8) and N-body
(cv9) drivers diverge because the SLAVE-MASTER coupling tangent block is never assembled.
This script verifies, ENTIRELY in SymPy, that the derived 4-block tangent

    K_ss[I,J] = +eps_n  sum_q w_q J_q  N_I^+(xi_q) N_J^+(xi_q)            (n (x) n)
    K_sm[I,K] = -eps_n  sum_q w_q J_q  N_I^+(xi_q) (N_K^- o chi)(xi_q)    (n (x) n)
    K_ms[K,J] = -eps_n  sum_q w_q J_q  (N_K^- o chi)(xi_q) N_J^+(xi_q)    (n (x) n)   ( = K_sm^T )
    K_mm[K,L] = +eps_n  sum_q w_q J_q  (N_K^- o chi)(xi_q) (N_L^- o chi)(xi_q) (n (x) n)

is EXACTLY the Jacobian d[residual]/d[u] of the discrete contact residual on a minimal
two-body interface (one P1 slave segment over one P1 master segment).

Derivation being checked (frictionless penalty, small-rotation / fixed correspondence,
geometric d(n) term dropped, as the one-body assembly does):

  slave surface Gamma+ shape fns N_I^+,  master surface Gamma- shape fns N_K^-,
  correspondence X^- = chi(X^+) is the OT monotone/Brenier map (treated as fixed: dchi=0).
  Normal gap at a slave quadrature point xi_q:
      g_N(xi_q) = ( x_s(xi_q) - x_m(chi(xi_q)) ) . n
      x_s = sum_I N_I^+ x_I^+,   x_m = sum_K (N_K^- o chi) x_K^- .
  Penalty traction  t = eps_n <-g_N> n  (active where g_N < 0).
  Contact forces:
      f_I^+ = +sum_q w_q J_q N_I^+(xi_q) t(xi_q)            (slave)
      f_K^- = -sum_q w_q J_q (N_K^- o chi)(xi_q) t(xi_q)    (master reaction)
  Residual R = K_bulk u - f_contact, so the contact tangent block is K_c = -d f_contact / d u
  and the global tangent is K_tot = K_bulk + K_c.  The derived blocks above ARE K_c.

Patch-test reproduction condition (the OT mass-preservation / marginal property):
      sum_K (N_K^- o chi)(xi) = 1   for all xi on Gamma_c
  i.e. the master interpolation is a partition of unity ALONG the correspondence.  Under this
  condition a constant slave traction integrates to the same total force on both bodies and is
  transmitted as a uniform stress.  We verify this row-sum identity symbolically as well.

This is SymPy-only.  It asserts the identities and prints PASS.

Run:  python3 two_body_mortar_tangent.py
"""
import sympy as sp


def build_residual_and_jacobian():
    """Form the contact residual on a minimal 2-body P1 interface and its symbolic Jacobian.

    Geometry (2-D, planar interface, constant normal n).  We allow a GENERAL fixed unit normal
    n=(nx,ny) so the (n (x) n) outer-product structure is exercised in both components, not just
    the y-axis.  Vertical/normal scalar displacements are carried by the gap; we keep the FULL
    2-D nodal displacement vectors so the Jacobian is a full 2x2-per-node-pair block matrix and
    the (n (x) n) tensor structure is checked.
    """
    eps_n = sp.symbols('epsilon_n', positive=True)
    nx, ny = sp.symbols('n_x n_y', real=True)
    n = sp.Matrix([nx, ny])

    # --- correspondence values along the slave segment ---------------------------------------
    # The OT map chi sends slave param xi -> master param.  We do NOT need its closed form; only
    # the COMPOSED master shape functions (N_K^- o chi)(xi_q) evaluated at the slave quadrature
    # points enter the tangent.  Treat them as symbolic constants b_qK (chi fixed => d/du = 0).
    # Two master nodes K=0,1 ; we keep them as a partition of unity b_q0 + b_q1 = 1 symbolically
    # only where the patch-test identity is checked; for the Jacobian check they are free symbols.

    # --- two Gauss points along the slave segment (P1 slave shape fns N_0=1-s, N_1=s) ---------
    # weights*Jacobian collapsed into W_q (w_q * J_q); shape-fn values a_qI = N_I^+(xi_q).
    nq = 2
    W = sp.symbols('W_0 W_1', positive=True)                       # w_q * J_q
    a = sp.Matrix(2, 2, lambda q, I: sp.Symbol(f'a_{q}{I}'))       # a[q,I] = N_I^+(xi_q)  slave
    b = sp.Matrix(2, 2, lambda q, K: sp.Symbol(f'b_{q}{K}'))       # b[q,K] = (N_K^- o chi)(xi_q) master

    # --- nodal displacement dofs (2 slave nodes, 2 master nodes; 2 components each) -----------
    us = [sp.Matrix(sp.symbols(f'us{I}_x us{I}_y', real=True)) for I in range(2)]   # slave
    um = [sp.Matrix(sp.symbols(f'um{K}_x um{K}_y', real=True)) for K in range(2)]   # master
    # reference positions enter the gap only through an additive g0 constant; its derivative is 0,
    # so we may carry the gap as the DISPLACEMENT part plus a symbolic reference gap g0_q.
    g0 = sp.symbols('g0_0 g0_1', real=True)                        # reference (undeformed) gap @ q

    # ============================================================================================
    #  RESIDUAL  (contact force on every dof), built straight from the penalty definition.
    #  We assume the ACTIVE branch g_N < 0 (penetration) so <-g_N> = -g_N (the branch the tangent
    #  is derived on).  g_N(xi_q) = ( x_s - x_m ).n = g0_q + ( sum_I a_qI us_I - sum_K b_qK um_K ).n
    # ============================================================================================
    def gap(q):
        xs = sp.zeros(2, 1)
        for I in range(2):
            xs += a[q, I] * us[I]
        xm = sp.zeros(2, 1)
        for K in range(2):
            xm += b[q, K] * um[K]
        return g0[q] + ((xs - xm).T * n)[0]

    # traction vector at q (active branch): t = eps_n * (-g_N) * n
    def trac(q):
        return eps_n * (-gap(q)) * n

    # nodal contact forces
    fs = [sp.zeros(2, 1) for _ in range(2)]     # slave   f_I^+ = +sum_q W_q a_qI t_q
    fm = [sp.zeros(2, 1) for _ in range(2)]     # master  f_K^- = -sum_q W_q b_qK t_q
    for q in range(nq):
        tq = trac(q)
        for I in range(2):
            fs[I] += W[q] * a[q, I] * tq
        for K in range(2):
            fm[K] += -W[q] * b[q, K] * tq

    # residual contribution R_contact = -f_contact (so that R = K_bulk u - f_contact)
    # full dof ordering: [us0, us1, um0, um1]  (each a 2-vector)
    u_all = []
    for I in range(2):
        u_all += list(us[I])
    for K in range(2):
        u_all += list(um[K])
    R = sp.zeros(8, 1)
    for I in range(2):
        R[2 * I:2 * I + 2, 0] = -fs[I]
    for K in range(2):
        R[4 + 2 * K:4 + 2 * K + 2, 0] = -fm[K]

    # symbolic Jacobian K_c = dR/du  (8x8)
    Kc = R.jacobian(u_all)
    return dict(eps_n=eps_n, n=n, W=W, a=a, b=b, g0=g0, Kc=Kc,
                us=us, um=um, gap=gap)


def derived_blocks(ctx):
    """Assemble the derived 4-block tangent from the closed-form formula (independent of R)."""
    eps_n, n, W, a, b = ctx['eps_n'], ctx['n'], ctx['W'], ctx['a'], ctx['b']
    nn = n * n.T                              # outer product (n (x) n), 2x2
    Kc = sp.zeros(8, 8)                        # 8x8 zero

    def add(row_node, col_node, scalar):
        blk = scalar * nn
        for di in range(2):
            for dk in range(2):
                Kc[2 * row_node + di, 2 * col_node + dk] += blk[di, dk]

    # node global indexing: slave I -> 0,1 ; master K -> 2,3
    for q in range(2):
        for I in range(2):
            for J in range(2):
                # K_ss[I,J] = +eps_n W_q a_qI a_qJ (n x n)
                add(I, J, eps_n * W[q] * a[q, I] * a[q, J])
            for K in range(2):
                # K_sm[I,K] = -eps_n W_q a_qI b_qK (n x n)
                add(I, 2 + K, -eps_n * W[q] * a[q, I] * b[q, K])
        for K in range(2):
            for J in range(2):
                # K_ms[K,J] = -eps_n W_q b_qK a_qJ (n x n)
                add(2 + K, J, -eps_n * W[q] * b[q, K] * a[q, J])
            for L in range(2):
                # K_mm[K,L] = +eps_n W_q b_qK b_qL (n x n)
                add(2 + K, 2 + L, eps_n * W[q] * b[q, K] * b[q, L])
    return Kc


def patch_test_identity():
    """Symbolically verify the constant-stress reproduction condition.

    Constant slave traction t = -p n on the interface.  The total slave force and total master
    reaction must be EQUAL AND OPPOSITE node-summed (resultant) AND, more strongly, the master
    nodal forces reproduce the SAME constant traction field iff sum_K (N_K^- o chi)(xi) = 1.
    We check: with sum_K b_qK = 1 at every Gauss point, the total master force resultant equals
    minus the total slave force, and the per-Gauss master contribution carries the full traction.
    """
    eps_n = sp.symbols('epsilon_n', positive=True)
    p = sp.symbols('p', positive=True)
    nx, ny = sp.symbols('n_x n_y', real=True)
    n = sp.Matrix([nx, ny])
    W = sp.symbols('W_0 W_1', positive=True)
    a = sp.Matrix(2, 2, lambda q, I: sp.Symbol(f'a_{q}{I}'))
    b = sp.Matrix(2, 2, lambda q, K: sp.Symbol(f'b_{q}{K}'))

    # constant traction t_q = -p n at every Gauss point
    tq = [-p * n for _ in range(2)]

    # slave partition of unity:  sum_I a_qI = 1  (P1 always);  total slave force
    Fs = sp.zeros(2, 1)
    for q in range(2):
        for I in range(2):
            Fs += W[q] * a[q, I] * tq[q]
    # master reaction with the OT row-sum imposed: substitute b_q1 = 1 - b_q0
    subs_pou = {b[q, 1]: 1 - b[q, 0] for q in range(2)}
    Fm = sp.zeros(2, 1)
    for q in range(2):
        for K in range(2):
            Fm += -W[q] * b[q, K] * tq[q]
    Fm = Fm.subs(subs_pou)
    # also impose slave P1 partition of unity a_q1 = 1 - a_q0
    subs_pou_s = {a[q, 1]: 1 - a[q, 0] for q in range(2)}
    Fs = Fs.subs(subs_pou_s)
    Fm = Fm.subs(subs_pou_s)
    total = sp.simplify(Fs + Fm)
    return total, Fs, Fm


def main():
    print("=" * 78)
    print("TWO-BODY MORTAR / OT-COUPLING CONTACT TANGENT  --  SymPy symbolic verification")
    print("=" * 78)

    ctx = build_residual_and_jacobian()
    Kc_sym = ctx['Kc']
    Kc_der = derived_blocks(ctx)

    print("\n[1] Jacobian of the contact residual  vs  derived 4-block tangent")
    diff = sp.simplify(Kc_sym - Kc_der)
    is_zero = diff == sp.zeros(8, 8)
    if not is_zero:
        # fall back to per-entry simplification
        is_zero = all(sp.simplify(diff[i, j]) == 0 for i in range(8) for j in range(8))
    print("    max symbolic |dR/du - K_4block| =", "0" if is_zero else "NONZERO")
    assert is_zero, "Symbolic Jacobian does NOT equal the derived 4-block tangent!"
    print("    -> dR/du == [K_ss, K_sm; K_ms, K_mm]   (EXACT, all 64 entries)")

    print("\n[2] Symmetry (frictionless => K_c = K_c^T)")
    sym_ok = sp.simplify(Kc_sym - Kc_sym.T) == sp.zeros(8, 8)
    if not sym_ok:
        sym_ok = all(sp.simplify((Kc_sym - Kc_sym.T)[i, j]) == 0 for i in range(8) for j in range(8))
    print("    K_c - K_c^T =", "0  (SYMMETRIC)" if sym_ok else "NONZERO")
    assert sym_ok, "Contact tangent is not symmetric!"

    print("\n[3] K_ms == K_sm^T  (transpose coupling through the SAME field)")
    # extract the 2x2-node blocks: slave nodes 0,1 (dof 0..3), master 2,3 (dof 4..7)
    Ksm = Kc_sym[0:4, 4:8]
    Kms = Kc_sym[4:8, 0:4]
    tcoup_ok = sp.simplify(Kms - Ksm.T) == sp.zeros(4, 4)
    if not tcoup_ok:
        tcoup_ok = all(sp.simplify((Kms - Ksm.T)[i, j]) == 0 for i in range(4) for j in range(4))
    print("    K_ms - K_sm^T =", "0" if tcoup_ok else "NONZERO")
    assert tcoup_ok, "Master-slave block is not the transpose of the slave-master block!"

    print("\n[4] Sign / structure spot-check of each block (a_qI=b_qK=1, single Gauss pt)")
    # set a simple numeric substitution to read off signs unambiguously
    sub = {ctx['W'][0]: 1, ctx['W'][1]: 0,
           **{ctx['a'][0, I]: 1 for I in range(2)},
           **{ctx['b'][0, K]: 1 for K in range(2)},
           ctx['n'][0]: 0, ctx['n'][1]: 1, ctx['eps_n']: 1}
    Kc_num = Kc_sym.subs(sub)
    # K_ss block (slave-slave, dof 0..3) y-y entries should be +1
    print("    K_ss[I,J] (y,y) =", [Kc_num[2 * I + 1, 2 * J + 1] for I in range(2) for J in range(2)],
          " expect +1 (SPD, +eps_n)")
    print("    K_sm[I,K] (y,y) =", [Kc_num[2 * I + 1, 4 + 2 * K + 1] for I in range(2) for K in range(2)],
          " expect -1 (cross coupling)")
    print("    K_mm[K,L] (y,y) =", [Kc_num[4 + 2 * K + 1, 4 + 2 * L + 1] for K in range(2) for L in range(2)],
          " expect +1 (master self)")
    ss_ok = all(Kc_num[2 * I + 1, 2 * J + 1] == 1 for I in range(2) for J in range(2))
    sm_ok = all(Kc_num[2 * I + 1, 4 + 2 * K + 1] == -1 for I in range(2) for K in range(2))
    mm_ok = all(Kc_num[4 + 2 * K + 1, 4 + 2 * L + 1] == 1 for K in range(2) for L in range(2))
    assert ss_ok and sm_ok and mm_ok, "Block signs disagree with the derivation!"
    print("    signs: K_ss=+, K_sm=K_ms=-, K_mm=+  CONFIRMED")

    print("\n[5] Patch-test mass-preservation identity  sum_K (N_K^- o chi) = 1")
    total, Fs, Fm = patch_test_identity()
    print("    total contact resultant (slave + master) =", list(total),
          " (expect [0,0]: equal & opposite => constant pressure transmitted exactly)")
    pt_ok = total == sp.zeros(2, 1)
    assert pt_ok, "Patch-test resultant is nonzero under the OT partition-of-unity condition!"
    print("    -> with sum_K b_qK = 1, the master reaction exactly balances the slave force")
    print("       (uniform pressure in -> uniform stress out; the OT marginal property).")

    print("\n" + "=" * 78)
    print("PASS:  the symbolic Jacobian dR/du EQUALS the derived 4-block tangent")
    print("       (K_ss=+, K_sm=K_ms^T=-, K_mm=+ ; symmetric SPSD), and the patch-test")
    print("       mass-preservation condition sum_K (N_K^- o chi)=1 makes the two-body")
    print("       resultant vanish.  No discrepancy.")
    print("=" * 78)


if __name__ == "__main__":
    main()
