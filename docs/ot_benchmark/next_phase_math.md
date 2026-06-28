# Mathematical verification — OT measure-coupling next phase (T2 / T3 / T4)

Scope: re-derive, from first principles, the mathematics of the NEW formulations the implement agents
build on top of the verified OT measure-coupling core (`solvers/contact/measure_coupling/`, audited in
`docs/ot_benchmark/math_verification.md`), and confirm the code matches the math. Three tracks:

- **T2** — two-body mortar OT coupling: a measure-coupling applied to BOTH non-matching surfaces is
  variationally consistent and passes the **two-body patch test** (constant pressure -> constant stress
  in the receiving body).
- **T3** — semismooth-Newton / augmented-Lagrangian (AL) friction update: the Clarke/B-subdifferential
  (or Uzawa) update is a consistent linearization of the Coulomb complementarity, and the
  energy-balance metric is correctly formed.
- **T4** — N-body array equilibrium (force balance, symmetry) and the deformable-asperity gap-field
  update.

Method: each claim is re-derived here independently, then checked against the as-shipped code. Every
numerical check is executed (numpy/scipy/torch, repo root) and reproducible. Errors are flagged
explicitly; "at floor / honest-partial" is reported where no closed form exists.

Status legend: ✅ verified correct · ⚠️ imprecision flagged (not a correctness bug) · ❌ error.

---

## Reference derivations (target formulas the code must reproduce)

### R-T2  Two-body mortar / OT coupling — variational consistency

Two deformable bodies Ω⁺ (slave, surface Γ⁺) and Ω⁻ (master, Γ⁻) in contact. The contact virtual work
is the single interface integral

    δW_c = ∫_{Γ_c} t_N (δg_N) ds + ∫_{Γ_c} t_T · (δg_T) ds,

with the gap g_N = (φ⁺(X⁺) − φ⁻(X⁻))·n evaluated through a **correspondence** X⁻ = χ(X⁺) between the two
surfaces. In a mortar/OT method χ is the measure-preserving (here monotone-rearrangement / Brenier) map,
NOT independent closest-point projection. Discretely, with slave shape functions N_I⁺ and master shape
functions N_K⁻,

    g_N(ξ) = Σ_I N_I⁺(ξ) x_I⁺ · n  −  Σ_K [Σ_I N_I⁺(ξ) (N_K⁻∘χ)(ξ)] x_K⁻ · n.

The **mortar mass matrices** are
    M_IJ = ∫_{Γ_c} N_I⁺ N_J⁺ ds          (slave–slave, the existing one-body block),
    D_IK = ∫_{Γ_c} N_I⁺ (N_K⁻∘χ) ds      (slave–master coupling — NEW for two bodies).
Force: f_I⁺ = ∫ N_I⁺ t ds = Σ_J M_IJ t_J ; reaction on master f_K⁻ = −∫ (N_K⁻∘χ) t ds = −Σ_J D_KJ^T t_J
(Newton's third law transmitted through the SAME field, i.e. the master force is the transpose-coupled
image of the slave traction — this is what makes the coupling variationally consistent rather than two
independent one-sided projections).

**Two-body patch test (the gate).** Apply a uniform pressure p̄ across the interface between two bodies
with non-matching meshes. PASS requires: (i) partition of unity Σ_I N_I⁺≡1 and Σ_K N_K⁻≡1; (ii) the
coupling reproduces a constant exactly — Σ_K D_IK = ∫ N_I⁺ (Σ_K N_K⁻∘χ) ds = ∫ N_I⁺ ds = M-row-sum,
i.e. the master interpolation is a partition of unity along the correspondence (mass is conserved by χ);
(iii) the transmitted master force equals minus the slave force node-by-node in resultant, so the
receiving body sees the SAME uniform traction → constant stress. The exact-constant-reproduction
condition Σ_K (N_K⁻∘χ) ≡ 1 is precisely the OT marginal/mass-preservation property. **This is the
mathematical heart of T2 and must be checked in code: constant pressure in → constant stress out, to
machine precision, on non-matching meshes.**

### R-T3  Semismooth Newton / AL for Coulomb friction

Regularized/exact Coulomb complementarity at a contact point (pressure p_N ≥ 0, slip rate ṡ_T):
normal:  0 ≤ p_N ⟂ g_N ≥ 0;  friction:  |t_T| ≤ μ p_N, with t_T = −μ p_N ṡ_T/|ṡ_T| when |ṡ_T|>0 (slip),
|t_T| < μ p_N ⇒ ṡ_T = 0 (stick).

**C-function form.** The normal law is equivalent to the nonsmooth equation
    C_N(p_N,g_N) = p_N − max(0, p_N − c_n g_N) = 0,  c_n>0,
and the tangential law (radial-return / projection onto the friction disk) to
    C_T = t_T − Π_{B(μ p_N)}( t_T + c_t ṡ_T ) = 0,
where Π_{B(ρ)}(z)=z·min(1, ρ/|z|) projects onto the disk of radius ρ=μ p_N. These are **semismooth**
(piecewise-smooth, Lipschitz). A semismooth Newton step uses any element of the Clarke / B-subdifferential
∂C; convergence is locally superlinear if the chosen generalized Jacobian is nonsingular. The two
branches give the consistent generalized Jacobian:
  - **stick** (|t_T + c_t ṡ_T| ≤ μ p_N): ∂_{ṡ_T} t_T = c_t I  → full tangential stiffness, no normal coupling;
  - **slip** (|q| > μ p_N, q=t_T + c_t ṡ_T): t_T = μ p_N q/|q|, and
    ∂t_T = μ p_N (c_t/|q|)(I − q̂⊗q̂)  +  μ q̂ ⊗ ∂p_N      — the (I−q̂⊗q̂) projector PLUS the
    p_N-coupling term μ q̂⊗∂p_N (a NON-symmetric contribution — true for frictional sliding).
The AL/Uzawa alternative augments the energy with multipliers (λ_N,λ_T) and updates
    λ_N ← max(0, λ_N + c_n g_N),  λ_T ← Π_{B(μ λ_N)}(λ_T + c_t ṡ_T),
which is the SAME projection structure; at convergence the multipliers equal the tractions and the
inner Newton solves the regularized problem exactly (no penalty bias).

**Energy balance metric (the T3 gate).** Over a load path the discrete balance is
    W_ext = ΔΨ_elastic + D_fric,   D_fric = Σ_steps Σ_pts (t_T · Δs_T^{slip}) A ≥ 0,
with Δs_T^{slip} the *plastic* (slipping) part of the tangential increment. The baseline cyclic FEM
reports ~1.5× imbalance because the Coulomb non-smoothness makes the midpoint t_T·Δs inconsistent with
the energy actually dissipated; a consistent semismooth/AL update + a properly integrated dissipation
(e.g. work conjugate to the *return-mapped* traction over the slip increment, or a midpoint rule on the
projected traction) must drive W_ext/(ΔΨ+D_fric) → 1. **Verify: (a) the generalized Jacobian matches
the stick/slip branches above; (b) the dissipation sum uses the plastic slip increment and the
return-mapped traction (not the trial), so D_fric ≥ 0 and the ratio → 1 within the reported tolerance.**

### R-T4  N-body array equilibrium + deformable-asperity gap update

**N-body equilibrium.** For N bodies with pairwise contact forces f_{ij} (force on i from j),
Newton's third law f_{ij} = −f_{ji} must hold per contact pair, and global equilibrium under no
external load requires Σ_j f_{ij} = 0 for each free body i and Σ_i Σ_j f_{ij} = 0 overall. For a
symmetric array (e.g. a disc lattice under equibiaxial load) the equilibrium configuration must inherit
the symmetry: equal contact forces on symmetry-equivalent pairs, zero net force/moment on each body,
and the per-body force balance closes to the applied boundary traction. The tangent K = K_bulk + Σ_pairs
K_c^{pair} must be symmetric (frictionless) and SPSD on the constrained system.

**Deformable-asperity gap-field update.** When the asperity (master surface) itself deforms, the gap
field g_N(ξ) = (x_s(ξ) − x_m(χ(ξ)))·n must be re-evaluated on the CURRENT deformed master, i.e. the
correspondence χ and the master point x_m(χ) are functions of the master displacement u_m. The
consistent gap linearization is
    δg_N = (δx_s − δx_m)·n + (x_s − x_m)·δn,
where δx_m = Σ_K N_K⁻(χ) δu_K (master nodes carry the variation) and the second term is the geometric
(curvature) contribution. The two-body tangent therefore couples slave AND master dofs (the D_IK block);
dropping δn is the documented small-rotation approximation. **Verify: (a) the gap is recomputed on the
deformed master each Newton iter (not frozen reference); (b) the master-dof coupling appears in the
tangent with the correct sign (−N_K⁻ n on the master, +N_I⁺ n on the slave); (c) emergent dilation > 0
and force balance closes.**

---

## T2 — two-body mortar OT coupling

Files: `benchmarks/contact/cv_numerical/cv8_deformable_ot.py` (`assemble_two_body`, `TwoBlockOT`,
`patch_test`, `hertz_test`). No new module under `solvers/contact/`; the slave side reuses
`measure_coupling.assemble_contact` verbatim and the master reaction is added in the driver.

### T2.1 Slave-side consistent force — ✅ correct

`assemble_two_body` integrates the slave traction with the consistent mortar mass exactly as the
verified one-body `assemble_contact` (R-T2 force `f_I⁺ = Σ_J M_IJ t_J`). This part inherits the
audited assembly and is correct.

### T2.2 Master reaction is NOT the variational transpose coupling — ❌ error

The reference (R-T2) requires the master reaction to be the **transpose-coupled image of the slave
traction through the SAME correspondence**: `f_K⁻ = −Σ_J D_KJ^T t_J` with
`D_IK = ∫_{Γ_c} N_I⁺ (N_K⁻∘χ) ds`. The code instead, per Gauss point, locates the master segment
hosting the mapped point `x_m=T(ξ_q)` and scatters the interpolated slave traction onto its two P1
master nodes (`cv8:200-210`, `_locate_master`). Resultant force is conserved (each Gauss
contribution is split `N_m0+N_m1=1`), so the *total* `Σf` is machine-zero — confirmed
(`force balance |sum f|=1.79e-18`). But this is a one-sided collocation of the slave traction onto
the master, **not** the symmetric `D_IK` mortar block, so it is not variationally consistent.

### T2.3 Two-body tangent omits the master block → Newton diverges — ❌ error (decisive)

`Kc` returned by `assemble_two_body` contains ONLY the slave–slave normal block
`M_IJ ε_n n_J⊗n_J` (`cv8:211-222`). The master-reaction force depends on the master dofs (the
correspondence/host-segment moves with `u_m`), but that dependence is **absent from the tangent**.
The Newton step is therefore taken with a tangent inconsistent with the residual.

**Re-derivation gate (the heart of T2): constant pressure → constant stress. RESULT: FAIL.**
Executed `patch_test()` (non-matching 12 vs 17 jittered interface). The Newton iteration diverges:

```
it=0 resid=1.78e+01   |u|=0
it=1 resid=1.36e+01   |u|=4.4e+01
it=2 resid=1.97e+00   |u|=4.4e+15   <- blow-up
...
it=13 resid=inf       |u|=1.06e+108
```

Driver output: `MORTAR : sigma_yy mean=nan ... non-uniformity=nan`. Heavy damping (factor 0.1, 400
iters) still diverges (`DIVERGED at it 189`). So the **two-body patch test does not pass**; the
decisive T2 consistency claim (uniform pressure transmitted as uniform stress to machine precision
across non-matching meshes) is **not demonstrated** — it returns NaN.

The cylinder/Hertz mode likewise fails to converge on small meshes (seat residual stuck at ~0.25,
1–2 active nodes, `a_fem=nan`); the `--mode all` driver prints `CV-8 two-body OT: CHECK`, not PASS.

**Root cause & fix.** Either (a) assemble the symmetric two-body tangent including the master block
`−N_K⁻ n` coupling and the slave–master cross term `D_IK` (the proper mortar/`D` matrix), or
(b) treat the master surface as the OT *master profile* and assemble the reaction through the SAME
`assemble_contact` machinery on the master side with the transposed correspondence. The current
nearest-segment scatter conserves resultant but cannot give a convergent consistent-tangent Newton.

**Verdict T2: ❌ formulation incomplete / patch test fails (NaN).** The slave force is consistent;
the master coupling and two-body tangent are not, and the decisive constant-stress gate does not run
to a number.

## T3 — semismooth-Newton / AL friction

Files: `solvers/contact/semismooth_friction.py`
(`alart_curnier_residual/_tangent`, `FrictionInterface1D`, `SemismoothBlock1D`),
`benchmarks/contact/cv_numerical/cv7_semismooth_friction_test.py`.

### T3.1 Alart–Curnier C-functions + Patton — ✅ correct

`alart_curnier_residual` implements `C_N = p_N − Π_{≥0}(p_N − r_N g_N)` and
`C_T = p_T − Π_{B(μ Π_{≥0}(p_N^aug))}(p_T − r_T g_T)` exactly as R-T3. The branch masks
(open/stick/slip) are correct. The `SemismoothBlock1D` reproduces the Patton closed form
`μ_app = tan(φ_b+i)` to machine precision (measured: i=0° err 0.0%, i=8° err 1.4e-14%, residual
1.4e-14%). ✅

### T3.2 Consistent (Clarke) tangent — mostly ✅, one sign bug ❌

Finite-difference check of `FrictionInterface1D.tangent` (P points, mixed open/stick/slip):

| block | code | central-diff FD | status |
|---|---|---|---|
| `dt_N/du_N` (closed) | `−r_N` | matches (2e-8) | ✅ |
| `dt_T/du_T` (slip) | `r_T (μ t_N/‖t_T^tr‖)(I−q̂q̂)` | matches (4e-7) | ✅ |
| `dt_T/du_N` (slip) | `+(μ r_N) q̂` | FD gives `−(μ r_N) q̂` | ❌ **sign wrong** |

Re-derivation: on the closed branch `t_N = r_N max(0,−u_N) = −r_N u_N`, so `dt_N/du_N = −r_N`; the
slip traction `t_T = μ t_N q̂` with `q̂` independent of `u_N`, hence
`dt_T/du_N = μ q̂ · dt_N/du_N = −μ r_N q̂`. The code (`semismooth_friction.py:269`) has
`+(self.mu*self.rN)*q` — wrong sign (FD ratio code/FD = −1, exactly). NB the *Alart–Curnier*
`dCT_dgN = +(μ r_N) q` (line 184) IS correct, because there the kinematic variable is the gap
`g_N` (penetration `g_N<0`, `p_N^aug = p_N − r_N g_N`), opposite in sign to the interface's `u_N`.
The bug is confined to `FrictionInterface1D.tangent`; it is **latent** — the shipped drivers
(`SemismoothBlock1D` fixes the normal load and uses only `dt_T/du_T`; the FEM-patch bank drives
displacement with fixed normal penetration) never exercise `dt_T/du_N`, so reported numbers are
unaffected. It would break a fully-coupled normal+tangential global Newton.

### T3.3 Energy-balance metric — formula ✅, but the headline "1.5×→1.0" claim is NOT reproduced ❌

The dissipation `dWp = (t_T·Δg_stick)·A` with `Δg_stick = (t_T^trial − t_T)/r_T` (the plastic slip
increment, return-mapped traction) matches R-T3 `D_fric = Σ t_T·Δs^slip ≥ 0`. The ledger
`W_ext / (ΔΨ + D_fric)` is correctly formed.

But the *claim* the driver advertises — regularized split closes only to ~1.5× while semismooth
fixes it to ~1.0 — is **not what this benchmark shows.** Measured energy ratios on the block:

| n_per_quarter | semismooth | regularized-split |
|---|---|---|
| 8 (quick) | 0.856 | **0.996** |
| 20 | 0.970 | 0.999 |
| 50 | 0.994 | 1.000 |
| 200 | 0.999 | 1.000 |
| 5000 | 0.99998 | 1.000 |

Both methods → 1.0 with step refinement (the 0.856 is **time-discretization error in the
trapezoidal `W_ext` integral**, not a formulation flaw — good news for correctness). However at
EVERY budget the regularized baseline is *closer* to 1.0 than the semismooth one, so on this
testbed semismooth does **not** beat the regularized split. The driver's gates `energy_block_flat`
and `energy_block_incl` FAIL at the quick budget (0.856/0.874 ∉ [0.95,1.05]); the
`semismooth_beats_regularized` gate is hard-wired to pass via `... or True` (`cv7_…:225`). The
~1.5× imbalance is a property of the *FEM* `rock_joint_cyclic_fem.py` driver and is **not
reproduced or fixed here** — the block/bank testbed does not contain the FEM non-smoothness that
caused it.

**Verdict T3:** Alart–Curnier kinematics and Patton ✅ exact; energy-ledger formula ✅ correct and
convergent to 1.0; but ❌ one tangent sign bug (latent) and ❌ the central "semismooth cures the
1.5× imbalance" claim is unsupported by the shipped benchmark (gates fail at budget; the comparison
gate is rigged with `or True`).

## T4 — N-body array + deformable asperities

Files: `benchmarks/contact/cv_numerical/cv9_nbody_ot.py` (`Disc`, `_build_pair_contact`,
`_build_wall_contact`, `run`).

### T4.1 Global force balance / Newton's third law — ✅ by construction

Each slave nodal force gets an equal-and-opposite reaction applied to the nearest master rim node
(`cv9:256-262`), so the per-pair resultant balances and the global wall reactions cancel. Measured
(2×2, n_rings=14): `global_balance_x=7e-5`, `|N−S|=2.7e-3` — closes. ✅ (this is the same
resultant-conserving-but-non-variational scatter as T2.2).

### T4.2 Tangent is inconsistent (master block missing, slave block non-symmetric) — ❌

(a) Only the slave block of each pair's `Kc` is added to the global tangent
(`Kc_tot[oa:oa+A.n_dof,…]+=Kc`, `cv9:247`); the master reaction's dependence on master dofs is
**not** in the tangent. (b) Even the single-body `Kc` from `assemble_contact` is **not symmetric on
a curved rim**: measured `max|Kc−Kcᵀ|=2.4e3` on a disc-pair band. Reason: the off-diagonal block is
`M_IJ ε_n (n_J⊗n_J)`, which is asymmetric in (I,J) when `n_I≠n_J` (curved surface). [This is a
property of the shared `assemble_contact`; it is symmetric only for the flat-facet case its
self-test uses — relevant context for the R-T4 "tangent symmetric/SPSD" claim, which therefore does
NOT hold on the curved N-body lattice.] Consequence: the solve relies on under-relaxation
(`relax=0.6`), not a consistent Newton.

### T4.3 Convergence + closed-form recovery — ❌ fragile, does not converge

R-T4 gate: centre disc `σ_xx=σ_yy=−2N/(πRt)`, D4 symmetry, convergence. Measured:

| n_rings | converged | wall imbalance | centre σ err | anisotropy |
|---|---|---|---|---|
| 14 | **False** (hit max_iter 120) | 0.27% | 1–2% | 2.0% |
| 16 | **False** (hit max_iter 120) | **318%** | **46%** | 2.0% |

The `converged` flag is False in BOTH cases — the iteration never reaches the Newton/relaxation
tolerance, it just stops at `max_iter`. Results are **non-robust**: refining n_rings 14→16 swings
the centre error from 2% to 46% and the wall imbalance from 0.3% to 318% (n_rings=16 also throws
`divide by zero / overflow in matmul` from degenerate stresses). The driver's own gate requires
`m["converged"]`, so it correctly prints `CHECK`, not PASS. The closed-form recovery is therefore
**not verified** — the one good n_rings=14 number is incidental, not a converged solution.

### T4.4 Deformable-asperity gap-field update — ✅ recomputed on deformed master

`_build_pair_contact` re-bakes BOTH rim bands from the CURRENT deformed nodes each Newton iteration
(`xs_cur = nodes + u`, master profile in slave-local frame), so the gap is evaluated on the deformed
master, not a frozen reference (R-T4 requirement (a) ✅). But the linearization of that moving
correspondence (R-T4 requirement (b), the `−N_K⁻ n` master coupling in the tangent) is the missing
block of T4.2, so the consistent-tangent half of the deformable-asperity claim is ❌.

**Verdict T4:** force balance ✅ (by resultant-conserving scatter) and deformed-master gap ✅, but
❌ inconsistent/asymmetric tangent, ❌ does not converge (max_iter in all runs), ❌ closed-form
centre stress not robustly recovered (2%↔46% across one mesh step).

---

## Summary

| track | claim | status |
|---|---|---|
| **T1** (context) | OT-path patch test machine-exact; neural radial L0 < 6e-3 / <2°; disc-SDF floor needs Euler | ✅ verified / honest-partial — tests pass (3/3), reuses the audited module, reports `partial` honestly |
| **T2** | two-body mortar variationally consistent; constant pressure → constant stress to machine precision | ❌ **patch test diverges to NaN**; master reaction is non-variational nearest-node scatter; two-body tangent omits the master block (Newton blows up even with 0.1 damping) |
| **T3** | semismooth/AL = consistent Coulomb linearization; energy ledger closes [0.95,1.05]; beats the 1.5× split | ✅ AC kinematics + Patton exact; ✅ ledger formula correct & → 1.0 with refinement; ❌ `dt_T/du_N` sign bug (latent); ❌ energy gates FAIL at budget and the "beats regularized / cures 1.5×" claim is **unsupported** (regularized is closer to 1.0 here; comparison gate forced with `or True`) |
| **T4** | N-body equilibrium, D4 symmetry, symmetric SPSD tangent, centre = −2N/πRt | ✅ resultant force balance + deformed-master gap; ❌ tangent inconsistent & asymmetric on curved rims; ❌ **does not converge** (max_iter every run); ❌ centre stress non-robust (2%↔46% over one mesh step) |

**Overall: NOT correct as a set.** T1 is sound. T2's decisive patch test fails (NaN). T3's core
math (Alart–Curnier, Patton) is correct but carries a latent tangent sign error and its headline
energy-improvement claim is not reproduced (the gate is rigged). T4 balances forces but does not
converge and does not robustly recover the closed form. The common thread across T2/T4 is the same
unresolved item: **the two-body / multi-body mortar tangent (the master-dof coupling block) is never
assembled**, so any solve that needs it either diverges (T2) or stalls at max_iter (T4).

### Errors flagged (actionable)
1. **T2 cv8 `assemble_two_body`** — master reaction is non-variational (nearest-segment scatter) and
   the two-body tangent omits the master/`D_IK` block ⇒ patch-test Newton diverges (NaN). Decisive
   gate does not pass.
2. **T3 `semismooth_friction.py:269`** — `dtT_duN[slip]` has the wrong sign; should be
   `−(mu*rN)*q` (FD-confirmed, ratio −1). Latent (unused by shipped drivers) but it is an incorrect
   "consistent" tangent block.
3. **T3 `cv7_semismooth_friction_test.py:225`** — `semismooth_beats_regularized` gate hard-wired
   with `... or True`; the claim is not actually tested and is not reproduced (regularized split is
   closer to 1.0 on this testbed). The `energy_block_flat/incl` gates genuinely FAIL at the quick
   budget.
4. **T4 cv9** — relies on under-relaxation, never converges (`converged=False`, max_iter in every
   run); curved-rim `Kc` is asymmetric (`max|Kc−Kcᵀ|≈2.4e3`); closed-form centre stress swings
   2%↔46% across a single n_rings step ⇒ not verified.
5. (context, shared module) `assemble_contact`'s `Kc` off-diagonal `M_IJ n_J⊗n_J` is only symmetric
   for flat facets; on curved surfaces it is asymmetric. Fine for the flat patch tests; it
   undermines the "symmetric SPSD tangent" assumption that T4 inherits.

---

## Two-body tangent FIX — verified (adversarial re-check)

The decisive T2/T4 blocker above ("the two-body / multi-body mortar tangent is never assembled") is
now **resolved**. A new shared assembler `solvers/contact/measure_coupling/two_body.py`
(`assemble_two_body_contact`) builds the full SYMMETRIC SPSD 4-block consistent tangent, and the cv8
(T2) and cv9 (T4) drivers call it. Re-derived independently, SymPy-verified, FD-verified, and run to
numbers below — no NaN-masking, no rigged overall verdict.

### The 4 blocks as shipped (`two_body.py:180-204`)

Per active slave Gauss point ξ_q, with one interpolated unit normal `n_hat` and the four signed
weights `entries = [(N_I⁺,slave,+1), (N_J⁺,slave,+1), (N_K⁻∘χ,master,−1), (N_L⁻∘χ,master,−1)]`, the
code accumulates `coeff = (ε_n w J)·(s_a w_a)·(s_b w_b)` times the SAME `n_hat⊗n_hat` for every
(a,b) pair. This is exactly

    K_ss = +ε_n Σ_q wJ N_I⁺N_J⁺ (n⊗n),   K_sm = −ε_n Σ_q wJ N_I⁺(N_K⁻∘χ)(n⊗n),
    K_ms = K_sm^T,                        K_mm = +ε_n Σ_q wJ (N_K⁻∘χ)(N_L⁻∘χ)(n⊗n),

with the signs (+,−,−,+) coming from `s_a s_b` (slave +1, master −1). K_ms = K_sm^T holds *by
construction* because the same nested loop emits the (a,b) and (b,a) entries with identical `coeff`.

### Gates (all executed, measured)

| gate | result |
|---|---|
| **SymPy** `dR/du == [K_ss,K_sm;K_ms,K_mm]` (all 64 entries, general n) | `two_body_mortar_tangent.py` → **0** (exact); symmetry 0; K_ms−K_sm^T = 0; signs +,−,−,+ confirmed; patch-test resultant [0,0] under Σ_K(N_K⁻∘χ)=1 |
| **FD Jacobian** `max|K_ana − K_fd|/scale` (non-matching 4v4, frozen geometry) | `two_body.py` self-test 2 → **3.3e-11** (≪1e-6) ✅ |
| **flat self-test** force balance / F_line / symmetric SPSD | 4.4e-16 / exact / symmetric SPSD ✅ |
| **curved-rim symmetry** `max|Kc−Kc^T|` (non-flat slave+master, general normals) | **0.0** (resolves the prior 2.4e3 asymmetry); min eig 0.0 (SPSD); |Σf|=0.0 ✅ |
| **T2 patch test** (cv8, non-matching 12 vs 17 jittered) | CONVERGES (was NaN/1e108). σ_yy mean **−0.049992** vs −0.05 (err 1.7e-4), non-uniformity 3.3e-3; net-resultant **3.6e-17**, transmit err **1.4e-16**; LUMPED baseline σ_yy=−0.909, non-unif 67.3 ✅ |
| **T4 N-body** (cv9, 3×3, 12 pairs) | **converged=True** (84 iters; was `converged=False`/max_iter every run); centre MEAN err **0.21%** vs −2N/πRt; global |Σf|=2.9e-15; gate requires `converged` → prints **PASS** ✅ |

### Why the curved-rim asymmetry is fixed

The prior shared `assemble_contact` used per-node normals `M_IJ (n_J⊗n_J)` (asymmetric in I,J on a
curved rim). The new block uses ONE interpolated `n_hat` per Gauss point as both the row and column
normal — `N_I N_J (n_hat⊗n_hat)` — which is manifestly symmetric in (I,J). cv9 also assembles all
four global blocks including `Ksm.T` explicitly (`cv9_nbody_array_ot.py:321-324`), with the local→
global rotation `R K R^T` preserving symmetry.

### Honest caveats (not correctness bugs, flagged so they are not over-read)

- **cv8 `coupling_pou_err`** (`cv8_deformable_ot.py:312-321`) computes `abs((1−tt)+tt−1.0)`, which is
  identically 0 by algebra — it does NOT actually probe Σ_K(N_K⁻∘χ)=1 and always reads 0.0. It is
  cosmetic and NOT load-bearing: the real evidence is the converged interior σ_yy (−0.049992 vs −0.05)
  plus `net_resultant`/`transmit_err` at machine precision and the SymPy patch-test identity. The line
  should be removed or replaced with an actual mass-marginal residual, but it does not inflate the
  verdict.
- **CV-8 overall prints `CHECK`, not `PASS`** — honest: the deformable-Hertz `a_fem` is 37.6% off on
  the coarse 40×10 CST mesh (stress-recovery / penalty bias), independent of the now-correct coupling.
  The verdict was NOT rigged to PASS.
- Scope unchanged: frictionless symmetric block (friction adds force only); small-rotation (d(n),
  d(χ) dropped, documented and consistent with the FD-frozen-geometry check).

### Updated verdict

| track | prior | now |
|---|---|---|
| **T2** | ❌ patch test NaN; master block absent | ✅ **4-block tangent assembled, FD==Jacobian (3.3e-11), patch test CONVERGES, uniform stress transmitted to ~1e-16**; CV-8 still `CHECK` only for the unrelated coarse-CST Hertz `a` |
| **T4** | ❌ asymmetric tangent, never converges | ✅ **symmetric SPSD tangent (curved-rim 0.0), N-body converged=True, centre 0.21%, |Σf|=2.9e-15 → PASS** |

The two errors #1 (T2) and the T4 half of #4/#5 in "Errors flagged" above are **fixed**. The T3 items
(#2 sign bug, #3 rigged `or True` gate) are a different module and remain open.
