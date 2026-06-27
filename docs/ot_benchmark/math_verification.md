# Mathematical verification of the OT measure-coupling contact model

Scope: re-derive the mathematics of `solvers/contact/measure_coupling/` and confirm the code matches
the math. Cross-referenced against `docs/contact_verification_manual.md` §3b (CV-1b/1c/2b). Every
numerical check below was executed (numpy/scipy, repo root) and is reproducible.

Files audited: `coupling.py`, `gap_field.py`, `traction.py`, `assembly.py`, `quadrature.py`,
`halfspace_bem.py`.

Verdict: **all six claimed components are mathematically correct and faithfully implemented.** Two
imprecisions are flagged (neither is a correctness bug): (a) the Sinkhorn→monotone *table* produced by
`measure_coupling_compare` at its default `n_iter=4000` is non-monotone at small `eps` due to
unconverged iterations, not a failure of the limit itself; (b) the documented dropped geometric tangent
term `p_N dn/du` degrades Newton (quadratic→linear) convergence on curved/large-rotation contact, which
is the relevant accuracy lever for CV-1/CV-3.

---

## Component 1 — 1-D monotone rearrangement is the Brenier / OT map for the quadratic cost

**Math.** For two probability measures `a, b` on the line and the convex cost `c(x,y)=½|x−y|²`, the
Monge optimal-transport map is the unique monotone non-decreasing map pushing `a` onto `b`, given by
the rearrangement `T = F_b⁻¹ ∘ F_a` where `F_a, F_b` are the CDFs (Brenier's theorem in 1-D; the
optimal potential is convex so its gradient `T` is monotone). The marginal/quantile identity
`F_b(T(x)) = F_a(x)` characterizes it.

**Code.** `MonotoneCoupling1D` builds normalized cumulative **arclength** measures
`F = ∫√(1+h'²)dx` (`_arclength_cdf`, trapezoid; `√(1+h'²) ≥ 1 > 0` ⇒ strictly increasing ⇒ `F⁻¹`
well-defined), then `map` computes `q = F_s(ξ)` (`np.interp(xi, x, Fs1)`) and
`x_m = F_m⁻¹(q)` (`np.interp(q, Fm1, master_x)`) — exactly `T = F_m⁻¹ ∘ F_s`.

**Verified (executed).** On `h_s=0.3 sin0.7x`, `h_m=0.15 cos0.5x`: `T` is monotone increasing, and
`max|F_m(T(ξ)) − F_s(ξ)| = 1.1e-16` (machine precision). The measure is the **arclength** (not
Lebesgue), so the coupling matches equal cumulative-arclength quantiles — the physically right measure
for surface mass. **CORRECT.**

## Component 2 — OT gap `g_N = (X_s − T)·n_s` and its `d_n f` reading

**Math.** Given the correspondence `T(ξ)=X_m`, the signed normal gap is the normal projection of the
slave→master displacement, `g_N = (X_s − X_m)·n_s`, `g_N<0` = penetration. For the quadratic OT cost
the Kantorovich potential `f` satisfies `∇f = X_s − T` at the optimum (the displacement field), so the
normal gap is its normal derivative `g_N = ∂_n f`; the c-transform pairs `f`, `g` are the Sinkhorn
potentials in the entropic case (Component 5).

**Code.** `GapField.sample`: `d = Xs − Xm`, `gN = (d·ns)`, `gT = (d·ts)` with
`ns = (−h', 1)/√(1+h'²)`. `eval_gap` parameterizes the single-valued slave by its x-coordinate.

**Verified (executed).** Flat pair offset `+0.10` ⇒ `g_N≡0.10`; penetration pair ⇒ `g_N≡−0.05`;
the identity `g_N == (X_s−X_m)·n_s` holds to machine precision; tilted-flat pair (slope `a=0.5`, pure
vertical offset `dz`) ⇒ `g_N = dz/√(1+a²) = dz cosθ`, the correct normal (cos) projection. **CORRECT.**

> Imprecision flagged (radial-gap 1/cos bias). This is **not** present in the measure-coupling gap. The
> 1/cosα bias warned about in CLAUDE.md / manual §11.2 belongs to the *single-body inverse radial
> chart* (`radial_chart_2d`/`supershape`), where the gap is measured along a radius, not the surface
> normal. The measure-coupling `g_N` uses the true surface normal `n_s` and is therefore the unbiased
> normal gap (the tilted-pair check above gives the exact `cosθ`, not `1/cosθ`). The benchmarks must
> keep comparing a neural **SDF** (Euclidean distance) against the Euclidean reference and the
> measure-coupling normal gap against the normal gap — not mix the radial gap in.

## Component 3 — consistent Galerkin assembly `f_I = Σ_J M_IJ t_J`, `M_IJ = ∫ N_I N_J ds`

**Math.** Interpolating the nodal traction `t(x)=Σ_J N_J(x) t_J` and testing against `N_I` gives
`f_I = ∫ N_I t ds = Σ_J (∫ N_I N_J ds) t_J = Σ_J M_IJ t_J` — the **consistent** (non-lumped) contact
force, with `M` the boundary consistent mass matrix. Per P1 segment of length `L`,
`M = (L/6)[[2,1],[1,2]]`. The consistent tangent linearizing the penalty `t_J = ε_n⟨−g_{N,J}⟩₊ n_J` on
the active set, holding `n` fixed, is `K_c = Σ_J M_IJ ε_n (n_J⊗n_J)`.

**Code.** `assemble_contact`: per segment `m = einsum('q,qa,qb->ab', wds, Nref, Nref)` with
`wds = w·L/2` (Gauss) — this *is* `∫N_I N_J ds`; force `f[gid[a]] += Σ_b m[a,b] t_node[b]`; tangent
block `m[a,b]·deps_node[b]·outer(n_b,n_b)` scattered to the 2×2 dof block, assembled to sparse CSR.
`deps_node = ε_n` where `g_N<0` else `0` (active set).

**Verified (executed).** The local `m` equals `(L/6)[[2,1],[1,2]]` exactly; the single-flat-segment
self-test gives a 50/50 force split and `F_line` exact; on a curved 9-node surface `K_c` is symmetric,
SPSD (`min eig = 0`), and carries a **nonzero off-diagonal** node-node coupling
(`K[node0_y,node1_y]=33.7`) — the mortar structure that a lumped/diagonal penalty lacks. **CORRECT.**

> Imprecision flagged (dropped geometric term). The tangent **omits** the geometric term
> `p_N ∂n/∂u` (and the `∂(ds)/∂u` metric term), documented in `assembly.py` and `__init__.py`. For a
> flat/charted interface (`∂n/∂u=0`) this is exact, so the patch test, rock-joint (Patton 0.00%), and
> half-space-BEM checks are unaffected. For a **curved** master or **large rotation** (CV-1 Hertz, CV-3
> Brazilian) the missing term makes the Newton tangent inconsistent → convergence degrades from
> quadratic to linear and the converged active-set/`a(F)` carries a small bias. This is exactly the
> manual's named CV-1/CV-3 accuracy lever ("consistent geometric contact tangent (currently
> dropped)"): adding `p_N ∂n/∂u` is mathematically required for asymptotic quadratic Newton on curved
> contact. The dropped term does **not** affect symmetry or SPSD of the *current* `K_c`.

## Component 4 — contact patch test (partition of unity + measure preservation + Gauss exactness)

**Math.** A uniform pressure must transmit exactly across non-matching/non-uniform nodes. This follows
from three properties: (i) **partition of unity** `Σ_I N_I ≡ 1` (P1, `lagrange_p1` rows sum to 1) ⇒ a
constant field is represented exactly and `Σ_I f_I = ∫ p ds`; (ii) **measure preservation** — the OT
marginal constraint sends the slave arclength measure onto the master's, so no node is double-counted
or starved (the failure mode of independent node-to-surface projection); (iii) **Gauss exactness** —
`order=3` integrates the `N_I N_J·(linear Jacobian)` products and a quadratic traction profile exactly,
so `M_IJ` is integrated with no quadrature error.

**Code/Verified (executed).** Non-uniform slave nodes `x=[0,0.3,0.5,1.1,1.7,2.0]`, flat, uniform
penetration: interpolated Gauss-point pressure field is uniform (`max−min = 0`), `F_line` error
`3.6e-15`, total tangential force `0` and total normal force `= ε_n·pen·L` to `0`. **CORRECT** — the
patch test passes to machine precision, matching manual §3b's `<10⁻¹²` claim.

## Component 5 — Sinkhorn → monotone as `eps → 0`

**Math.** The entropic OT `min ⟨C,π⟩ + ε·KL(π|a⊗b)` has a unique plan whose barycentric map
`T_ε(x)=E_π[X_m|x]` converges to the Monge (monotone) map as `ε→0` (Γ-convergence of entropic to
Monge–Kantorovich; the log-domain Sinkhorn iteration computes the potentials `f,g`).

**Code.** `SinkhornCoupling1D`: log-domain fixed point `f ← −ε·LSE((g−C)/ε + log b)`,
`g ← −ε·LSE((f−C)/ε + log a)` on the squared-distance cost over arclength measures; `map` returns the
softmax barycentric `X_m`. `measure_coupling_compare` tabulates relerr vs the monotone reference.

**Verified (executed).** With a **sufficient** iteration budget the limit holds:
`eps=0.01 → relerr 5.7e-5`, `0.003 → 5.9e-5`, `0.001 → 5.9e-5` (plateau at the ~6e-5 grid-discretization
floor; n_iter 16.6k / 40k / 40k). **CORRECT.**

> Imprecision flagged (the default-budget table is misleading). The shipped
> `measure_coupling_compare` (and the `SinkhornCoupling1D` default `n_iter=4000`) reports a
> **non-monotone** error: `eps=0.01 → 4.3e-5` but then `0.003 → 2.2e-4`, `0.001 → 7.3e-4` — the error
> *rises* at the two smallest `eps`. Cause: at small `eps` the 4000-iteration cap is hit before
> convergence (Sinkhorn needs O(1/ε) iterations), so the reported map is the unconverged iterate, not
> the entropic optimum. The marginal residual confirms this (Component 6: `2e-9`, not machine zero, at
> `eps=0.01, n_iter=4000`). This is a *budget/convergence* artifact, not a violation of the
> `Sinkhorn→monotone` theorem; the manual's "<0.02%" agreement is met at the converged `eps` but a
> reader scanning the table would wrongly infer the limit reverses. Recommended fix for the
> CV-2 "debiased Sinkhorn" lever: scale `n_iter ∝ 1/eps` (or use ε-scaling / Sinkhorn divergence
> debiasing) so the table is monotone, and treat the `2e-12`-class machine-precision claims as
> requiring convergence, not the default cap.

## Component 6 — marginal / mass constraint

**Math.** The transport plan must satisfy the marginals `Σ_j π_ij = a_i`, `Σ_i π_ij = b_j` (mass is
neither created nor destroyed); this is the constraint that forbids the many-slaves-to-one-master
collapse and makes the gap a continuous field. For the monotone map it is the quantile identity of
Component 1; for Sinkhorn it is enforced by the alternating potential updates.

**Code/Verified (executed).** Monotone: marginal = quantile match, machine precision (Component 1).
Sinkhorn (`eps=0.01`): reconstructing `π_ij = exp((f_i+g_j−C_ij)/ε)·a_i·b_j`, the column marginal is
exact to `6.9e-18` (last update), the row marginal `2.2e-9` (one update stale — same
non-convergence note as Component 5), total mass `= 1.0`. **CORRECT** (with the convergence caveat on
the row residual at the default budget).

---

## Cross-checks run

- All module `__main__` self-tests pass: `assembly` (50/50 split, F_line exact, symmetric SPSD mortar
  tangent), `coupling` (identity map + normalized CDFs), `gap_field` (flat ±0.10/−0.05),
  `traction` (penalty pressure + bounded Coulomb **+ torch cross-check: numpy penalty/Coulomb ==
  audited `solvers.contact.{penalty,friction}` kernels bit-for-bit**), `quadrature` (length + linear
  exactness).
- `halfspace_bem.py`: the BEM compliance uses the correct 2-D Flamant log kernel
  `u_z = −(2/πE*)∫p ln|x−ξ|dξ`, with the closed-form antiderivative `(t−x)(ln|t−x|−1)`; the LCP and
  TractionField-penalty solves are independent paths to the same Hertz pressure (manual §3b reports
  0.00% — the clean isolation of the coupling math from the bulk-FEM floor).

## Summary of flagged items (none are correctness bugs)

1. **Dropped geometric tangent `p_N ∂n/∂u`** — exact on flat/charted interfaces; on curved/large-
   rotation contact (CV-1, CV-3) it makes Newton inconsistent (quadratic→linear) and biases the
   converged `a(F)`. Adding it is the documented CV-1/CV-3 accuracy lever.
2. **Sinkhorn default-budget table non-monotonicity** — `measure_coupling_compare` /
   `SinkhornCoupling1D(n_iter=4000)` under-iterate at small `eps`, so the relerr table rises spuriously
   below `eps≈0.01`. The `Sinkhorn→monotone` limit itself is correct (verified to the ~6e-5 grid floor
   with adequate iterations). Scale `n_iter ∝ 1/eps` (debiased/ε-scaled Sinkhorn) for CV-2.
3. **Radial-gap 1/cos bias does NOT apply here** — the measure-coupling gap uses the true surface
   normal and is unbiased; the bias is a property of the separate single-body radial chart. Keep the
   benchmark comparisons normal-gap-vs-normal-gap and SDF-vs-Euclidean.

> **Round-2 correction to item #1.** The round-1 summary asserted the dropped geometric tangent
> `p_N ∂n/∂u` "biases the converged `a(F)`". The round-2 convergence test (below, §R2) **disproves**
> the *biases the converged solution* half of that claim: at a converged Newton solve the residual is
> driven to `‖R‖→0` and the solution is **independent of which tangent** was used to get there. The
> dropped term is purely a **Newton-path** lever (quadratic↔linear convergence rate / iteration count);
> it cannot change the converged `a_fem`. The accurate half of item #1 stands: the term is required for
> *asymptotic quadratic Newton*. It is **not** an accuracy lever for the converged contact metric.

---

## §R2 — Round-2 convergence test (can any round-1-best CV error be beaten by a new lever?)

Goal: re-verify the round-1 driver changes are still mathematically applied correctly, and test whether
any CV beats its round-1-best OT error with a new algorithmic lever. All numbers below were executed
(numpy/scipy, repo root). **Verdict: round-1 math all re-confirmed correct; no CV beats its round-1
best — the loop has CONVERGED.**

### R2.1 Round-1 changes re-verified (all reproduce, all sound)

- **CV-1** `a_relerr = 1.4762%` reproduces bit-for-bit (round-1 best 1.48%). Solve is fully converged
  (identical `a_fem=0.098167` at `max_iter=40` and `120`, both `iters=14`). Patch test, mortar mass
  `(L/6)[[2,1],[1,2]]`, closest-point OT gap `g=|p−c|−R` all unchanged and correct.
- **CV-3** centre `σxx` err `0.26%` reproduces (round-1 best 0.258%). The **quadratic debiasing** is a
  *valid* extrapolation: applied to the EXACT analytic Brazilian field on `r<0.1R`, the quadratic fit
  recovers the closed-form centre point to **0.0008%** while a patch-MEAN is biased **0.98%** low (the
  curvature bias the lever removes). So the recovered FEM `0.26%` is genuine discretization, not a fit
  artefact. **Sound.**
- **CV-4** centre `σxx` err `0.12%` at `n_rings=64` reproduces; the round-1 best `0.077%` is the
  `n_rings=96` symmetric-discretization floor (`--fine`). D4 force imbalance `0.05%`. **Sound.**
- **CV-6** last cross-level relative increment `2.429e-5` reproduces (round-1 best 2.43e-5, round-0
  5.94e-4). Increments `[3.36e-1, 7.09e-2, 8.01e-3, 5.94e-4, 2.43e-5]` form a **monotone geometric**
  sequence (successive ratios `0.227, 0.114, 0.074, 0.041`, decreasing → super-geometric Cauchy). The
  **plateau/tail-median** convergence estimator is unbiased: it measures the increment on the
  active-set-stabilized tail (`level≥3`), which is exactly the regime where the geometric law holds.
  **Sound.**
- **CV-2** Part A c/a-law mean err `0.13%`, field L2 `0.31%` reproduce. The **deformed-config fixed
  point** (Part B) is a *legitimate self-consistent* contact solve: it re-assembles the half-space
  compliance on the current deformed radii, re-solves the Signorini non-penetration LCP against the
  indenter (active-set with `p≥0` and gap≥0 admissibility — a proper complementarity solve), and
  under-relaxes `r_def` until `‖Δr_def‖<1e-6` (converges in 15–16 iters, `res≈7e-7`). It correctly
  collapses the spurious single-pass frozen swell (3.6/10.9/21.8%) onto the indenter-set radius
  (−0.24/0.06/−0.12%) — the physically right answer (lateral stretch balanced by re-contact). The
  **sub-grid stick-radius midpoint** estimator (`c = ½(r_last_stick + r_first_slip)`) is unbiased to
  `O(dr²)` (the true boundary is sub-grid; a symmetric bracket midpoint removes the one-sided `O(dr)`
  node-quantization truncation). **Sound.**

### R2.2 Why no CV beats its round-1 best (the floor proof)

- **CV-1 (the named round-2 lever fails by construction).** The campaign suggested implementing the
  consistent geometric tangent `p_N dn/du` for "true quadratic Newton + a(F) accuracy". But `a_relerr`
  is recovered from the **converged** displacement (`R→1e-9`), and a converged Newton solution is
  **independent of the tangent**. Measured: `a_fem=0.098167` is identical at `max_iter=40` and `120`.
  Adding `p_N dn/du` would cut iteration count but **cannot** move `a_fem`. A contact-edge **mesh
  refinement** sweep (the only real accuracy lever) does **not** beat 1.48% locally — it makes it
  *worse* (n_fine 90→140: 1.48%→1.69%; (160,80,160): 1.53%; (200,100,140): 1.72%), because the
  half-ellipse edge-fit `a_fem` jitters with where the contact edge falls between nodes. The round-1
  single-δ value 1.48% sits at that jitter floor; the honest robust (δ-window-averaged) floor is 2.01%.
  **improved = false (floor confirmed: tangent is path-only; refinement does not help the edge-fit).**
- **CV-3** `0.258%` ≈ quadratic-debiased FEM discretization floor on `n_rings=64`; the patch curvature
  bias is already removed analytically (0.0008% residual). **At floor.**
- **CV-4** `0.077%` is the measured symmetric-discretization floor (`n_rings=128` measured WORSE,
  0.084%, in round-1). **At floor.**
- **CV-6** `2.43e-5` is the converged geometric-Cauchy last increment on the local ladder (2..7); only
  deepening to `n=8` (`--fine`, Euler A100) lowers it further — no new *algorithmic* lever, just more
  levels. **At floor for the local budget.**

**Conclusion.** Round-1 mathematics is correct and faithfully applied; the round-1 improvements are all
mathematically sound (quadratic-extrapolation debias, self-consistent deformed-config fixed point,
unbiased plateau/sub-grid estimators). No CV beats its round-1 best with a new lever. The CV-1 geometric
tangent is a Newton-path (convergence-rate) lever, not a converged-accuracy lever — a confirmed floor,
which proves convergence of the loop. **0 correctness errors found.**

### R2.3 Minor imprecision flagged (not a correctness bug, no reported-error impact)

- **CV-2 Part B pressure–radius pairing.** `largedef_cattaneo` interpolates the converged pressure with
  `np.interp(|x_def|, r0, p)` — using the **reference** abscissa `r0` for a pressure that now lives on
  the **deformed** radii `r_def`. This mildly mis-pairs `p(r)` in the tangential post-processing. It
  affects **only** Part B, which reports qualitative large-deformation *shifts* (no error-vs-closed-form
  claim); Part A (the c/a=0.13% headline) is unaffected. Cosmetic; pair on `r_def` for full rigor.
