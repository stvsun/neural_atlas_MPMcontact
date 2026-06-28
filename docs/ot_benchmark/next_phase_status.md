# Next-Phase Four-Track Status (honest verdict)

A multi-agent team executed the four documented next-steps. A dedicated **math-verification agent**
re-derived each new formulation from first principles, finite-difference-checked the tangents, and ran
the gates — and initially **overturned the builder agents' optimistic self-reports** (`overall_correct =
False` at that stage). The precisely-identified hard problems were then fixed and re-verified: **all four
tracks now pass** (see the UPDATE block and table below). Full derivations in `next_phase_math.md`. This
file is the honest at-a-glance status; the verdicts were independently reproduced by the orchestrator
before recording.

> **UPDATE (T2 & T4 NOW SOLVED — orchestrator re-verified).** The root-cause gap below — the
> never-assembled multi-body mortar master-coupling tangent — has been deployed via a derive → SymPy →
> code → V&V pipeline, and the **two-body deformable BVP (T2) is now genuinely solved** by the
> **closest-point OT transition map** `π_B ∘ φ_A`. The shared helper
> `solvers/contact/measure_coupling/two_body.py::assemble_two_body_contact` assembles the full 4-block
> tangent `[[K_ss, K_sm],[K_ms, K_mm]]` (`K_ms = K_sm^T`, symmetric SPSD). Independently re-verified:
> SymPy proves the symbolic `dR/du` **equals** the 4-block tangent (`docs/hertz_derivation/
> two_body_mortar_tangent.py`, resultant `[0,0]`); the FD check on the closest-point tangent gives
> `K==∂f/∂u` to **3.4e-11** (`two_body.py` self-test 3); the two-body force-balance self-test is
> **4.4e-16**.
>
> **Two prior T2 bugs were fixed:** (1) the **inverted Hertz geometry** (the `cv8` `block_mesh` sign that
> made the indenter convex the wrong way), and (2) the use of the **GLOBAL ARCLENGTH** `MonotoneCoupling1D`
> for a **partial** contact — the arclength map *smears* a partial/non-conforming contact across the whole
> interface (it mis-maps central nodes to the indenter edge). The **closest-point projection is the correct
> OT map for non-conforming/partial contact**, exactly as `cv1_ot_gap.py:15-20` already documents (for a
> localized convex rigid indenter the 1-D convex-cost OT map degenerates to the radial closest-point
> projection `g = |p−c| − R`; arclength-monotone is the closest-point map only for two *mated* rough
> profiles). `p0` is recovered from the Gauss-point pressures.
>
> **T2 (`cv8`) now PASSES with mesh convergence** (closest-point correspondence hardcoded for all
> seat/slide/convergence solves; arclength only used for the full-contact patch test): deformable Hertz
> at the finest mesh (nx=192) **a_relerr = 2.75%** (<10%), **p0_relerr = 5.82%** (<12%); mesh convergence
> a_relerr **5.14 → 4.14 → 2.96 → 2.75%** (strictly down, 96→192) and p0_relerr **6.34 → 5.68 → 5.91 →
> 5.82%** (drops then plateaus ~5.8% — not a lucky point); the non-matching **patch test** still transmits
> a uniform pressure to **1.4e-16** (OT mass-marginal err 0.0, net resultant 3.6e-17; lumped baseline 67×
> worse); large **sliding tracks smoothly** (centroid monotone, max backstep 0.0, no saltation; F_cov
> 0.0017); **force balance** `|Σf|` = **1.27e-19**. Contact localizes physically: 52 active / 161 slave
> nodes, a/W = 0.228.
>
> **T4 (`cv9_nbody_array`) now PASSES at FULL Newton (relax=1.0).** The 3×3 array **converges** (32 iters,
> line search ON, 0 backtracks at default settings — `converged==True`, not max_iter; forcing max_iter=1
> correctly yields `converged=False`), global **force balance 3.71e-15** (genuine Newton's-3rd-law cancel
> of per-pair `|f|≈14.4`, not a trivial zero), centre-disc equibiaxial **MEAN stress 0.58%** (<5%).
> Per-component anisotropy is **0.20–0.22% on the D4-symmetric mesh** (the legacy ring mesh reproduces the
> ~11.6% artifact, so the D4 fix is real, honestly labeled as mesh-symmetry; the MEAN gate passes under
> *both* meshes so the trick does not prop up the gate). Tests: `tests/test_cv9_nbody_ot.py` 5/5 pass;
> `two_body.py` self-tests 1/2/3 pass.
>
> **Residual honest caveats (kept):** discrete contact-edge noise at the half-ellipse edge (infinite Hertz
> edge-pressure slope rounded over one element — this is the residual ~3–6% in a/p0); single-realization;
> plane-strain CST (small-strain Tri2D, frictionless, rigid outer walls in T4); the dropped `d(n)/d(χ)`
> geometric term is the standard small-rotation approximation (recovered across Newton iters). The
> orchestrator independently re-ran every gate above.

| Track | What was built | Honest status |
|-------|----------------|---------------|
| **T1** Neural charts | `cv_neural_chart_verify.py` + test | ✅ **VERIFIED** — OT-path patch test machine-exact; neural radial chart hits L0 (gap RMSE/L < 6e-3, normal median < 2°); disc-SDF floor reported as honest-partial (Euler budget). 3/3 tests pass. Reuses the audited `measure_coupling` module. |
| **T2** Deformable-deformable OT | `cv8_deformable_ot.py`, `measure_coupling/two_body.py` | ✅ **VERIFIED — two-body BVP solved via the closest-point OT transition map.** Two prior bugs fixed (inverted Hertz geometry; arclength-OT *smearing* a partial contact → switched to the closest-point projection `π_B∘φ_A`, the correct OT map for non-conforming/partial contact). Deformable Hertz (finest nx=192): **a_relerr 2.75%** (<10%), **p0_relerr 5.82%** (<12%, `p0` from Gauss-point pressures), with monotone mesh convergence (a: 5.14→2.75%, p0: 6.34→5.82%). Non-matching **patch test** transmits uniform pressure to **1.4e-16** (67× tighter than lumped); large **sliding** tracks smoothly (centroid monotone, max backstep 0.0); **force balance 1.27e-19**; SymPy + FD tangent (3.4e-11). 5/5 cv9 tests + 3/3 two_body self-tests pass. |
| **T3** Semismooth/AL friction | `solvers/contact/semismooth_friction.py`, `cv7_semismooth_friction_test.py` | ⚠️→✅ **CORRECTED.** Alart-Curnier kinematics + Patton exact (1e-14); energy ledger correct and in tolerance (0.97). Two defects fixed: (1) a wrong-sign `dt_T/du_N` tangent block (FD-confirmed, `semismooth_friction.py:269`); (2) a **rigged test gate** (`... or True`) removed. Honest finding: semismooth does **not** beat the regularized split on this block testbed (regularized is closer to 1.0; both → 1.0 with refinement); the ~1.5× imbalance is a property of the FEM `rock_joint_cyclic_fem` driver, **not reproduced here**. |
| **T4** N-body + deformable asperities | `cv9_nbody_array_ot.py`, `cv9_nbody_ot.py`, `cv9_deformable_asperity_ot.py` (+ tests) | ✅ **VERIFIED — N-body array converges at full Newton (relax=1.0).** 3×3 array (12 OT pairs): **converged=True**, 32 iters, line search ON, 0 backtracks (NOT max_iter — forcing max_iter=1 yields converged=False); global **force balance 3.71e-15** (genuine per-pair `|f|≈14.4` cancel); centre-disc equibiaxial **MEAN stress 0.58%** (<5%). Per-component anisotropy **0.20–0.22% on the D4 mesh** (legacy ring mesh reproduces the ~11.6% artifact → fix is real, honestly labeled; MEAN gate passes under both meshes). `tests/test_cv9_nbody_ot.py` 5/5 pass. |

## The former root cause — now RESOLVED (a real research finding)

The original blocker was a **multi-body mortar master-coupling tangent block** — the
`D_IK = ∫ N_I⁺ (N_K⁻∘χ) ds` slave-master coupling and the master-dof linearization `−N_K⁻ n` — that was
**never assembled**. The rigid-counterface CV-1…CV-9 cases never needed it (a rigid master has no dofs),
so it was never built; the deformable-deformable (T2) and N-body (T4) cases were the first to require it,
and without it the consistent-tangent Newton diverged (T2) or stalled at max_iter (T4).

**This is now built and verified.** The shared helper `solvers/contact/measure_coupling/two_body.py::
assemble_two_body_contact` assembles the full 4-block tangent `[[K_ss, K_sm],[K_ms, K_mm]]` with
`K_ms = K_sm^T` (symmetric SPSD). SymPy proves the symbolic `dR/du` **equals** the 4-block tangent
(`docs/hertz_derivation/two_body_mortar_tangent.py`, resultant `[0,0]`); the FD check on the
closest-point tangent gives `K == ∂f/∂u` to **3.45e-11** (`two_body.py` self-test 3); the two-body
force-balance self-test is **4.4e-16**.

**The deformable-deformable BVP itself (T2) needed two further fixes** that the tangent alone did not
expose — both were geometry/correspondence bugs, not tangent bugs:
1. an **inverted Hertz geometry** (the `cv8` `block_mesh` sign that made the indenter convex the wrong
   way); and
2. the use of the **GLOBAL ARCLENGTH** `MonotoneCoupling1D` for a **partial** contact. The arclength
   map *smears* a partial / non-conforming contact across the whole interface (it mis-maps central nodes
   to the indenter edge). The **closest-point projection is the correct OT map for non-conforming /
   partial contact** — exactly as `benchmarks/contact/cv_numerical/cv1_ot_gap.py:15-20` already
   documents: for a localized convex rigid indenter the 1-D convex-cost OT map degenerates to the radial
   closest-point projection `g = |p−c| − R`; arclength-monotone is the closest-point map only for two
   *mated* rough profiles. The driver therefore hardcodes `CORR='closest_point'` for all
   seat/slide/convergence solves and uses the arclength map only for the full-contact patch test.

A second, earlier finding (now contained): `measure_coupling.assemble_contact`'s tangent off-diagonal
`M_IJ εₙ (n_J⊗n_J)` is symmetric **only for flat facets**; on a curved surface it is asymmetric. The
closest-point two-body tangent in `two_body.py` is symmetric to atol 1e-12 with min eigenvalue
`> −1e-9` (SPSD) on the Hertz seat, so the verified T2 path does not rely on the curved-rim symmetry of
the older `assemble_contact`.

## Remaining honest residuals (kept)

- **Discrete contact-edge noise** at the half-ellipse edge: the infinite Hertz edge-pressure slope is
  rounded over one element. This is the residual ~3–6 % in `a` / `p0` (the finest nx=192 gives a 2.75 %
  / 5.82 %; the p0 error drops then plateaus ~5.8 %, i.e. it is a real floor, not a lucky point).
- **Single realization**, plane-strain CST (small-strain Tri2D), frictionless, rigid outer walls in T4.
- The dropped `d(n)/d(χ)` geometric term is the standard small-rotation approximation (recovered across
  Newton iters; the converged solution is tangent-independent).
- T4 per-component anisotropy is mesh-symmetry-controlled (0.20–0.22 % on the D4 mesh vs ~11.6 % on the
  legacy ring mesh); the **MEAN** stress gate — the physical invariant — passes under *both* meshes
  (0.58 % D4, 0.32 % ring), so the D4 mesh choice does not prop up the gate.

The orchestrator independently re-ran every gate above.

## Meta-lesson

The builder agents originally reported T2/T3/T4 as "verified"/"pass"; the adversarial math-verification
agent (and the orchestrator's independent re-run) exposed a NaN-masked false pass, a rigged gate, a
wrong-sign tangent, and a non-convergent solve — and sent them back. **The dedicated math/verification
agent earned its place**: the precisely-identified hard problems (the never-assembled master-coupling
tangent, the arclength-smearing of partial contact) were then fixed for real, and T2/T4 now pass with
the residuals stated honestly rather than papered over.
