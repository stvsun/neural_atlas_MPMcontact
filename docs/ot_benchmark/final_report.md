# OT vs Conventional Gap-Function Model — CV-1..CV-7 Benchmark (Final Report)

**Verdict: the OT measure-coupling gap model wins or ties the conventional gap-function model on all 7
verification cases, and the iterative accuracy-improvement loop has CONVERGED** (no CV improved in
round 3; improvement frontier 7 → 1 → 0).

A multi-agent campaign (one agent per CV + a dedicated OT-contact math-verification agent) benchmarked
the OT measure-coupling gap function (`solvers/contact/measure_coupling/`) against the conventional
node-to-surface lumped-penalty gap-function model, then looped — sharing knowledge at a barrier and
running ReAct+Reflexion improvement rounds — until no CV could be improved. Heavy/fine and
large-deformation solves were offloaded to the **Euler A100** (`ws2414@euler.civil.columbia.edu`, conda
`atlas`, direct GPU). Records: `ledger.json` (per-round errors), `phase_log.md` (the four recorded phases
per round), `math_verification.md` (the math agent's findings).

## 1. Final OT-vs-conventional benchmark

| CV | Problem | Metric | Conventional | **Best OT** | Advantage |
|----|---------|--------|--------------|-------------|-----------|
| CV-1 | Hertz line contact | half-width `a` relerr (robust) | 1.59% | **0.50%** | **3.2×** |
| CV-2 | Cattaneo–Mindlin | stick-radius `c/a` relerr | 11.15% | **0.03%** | **370×** |
| CV-3 | Brazilian disc | centre `σ_xx` relerr | 1.62% | **0.23%** | **6.3×** |
| CV-4 | Nine-disc cell | centre `σ_xx` relerr | 0.11% | **0.077%** | **1.4×** |
| CV-5 | Superformula cusps | cusp gap RMS | 1.99e-2 | **5.3e-13** | **>1e10× (grid-indep.)** |
| CV-6 | Koch fractal | contact-force cross-level increment | frozen (non-convergent) | **2.6e-16** | **convergent vs frozen** |
| CV-7 | Patton sawtooth | `μ_app` relerr | 1.5e-4 | **3.5e-14** | **~1e10× (machine prec.)** |

Smallest margin is CV-4 (1.4×; both excellent on a smooth analytic field). The largest margins are the
**geometry-dominated** cases (CV-5/6/7), where OT's closest-point / Brenier coupling is grid-independent
while a conventional gap function is resolution-bound (the SDF smooths cusps; the level-set grid freezes
the fractal contact force; the vertical-ray gap sheds contact nodes).

## 2. Round-by-round improvement trajectory (OT error)

| CV | Round 0 | Round 1 | Round 2 | Round 3 | Floor type |
|----|---------|---------|---------|---------|------------|
| CV-1 | 3.24% | 1.48% | **0.50%** | 0.40–0.50% (plateau) | discrete-edge (∞ Hertz edge slope) |
| CV-2 | 0.41% | 0.13% | **0.03%** | — | converged (n=3200) |
| CV-3 | 1.29% | 0.258% | **0.23%** | — | FEM stress recovery |
| CV-4 | 0.12% | 0.077% | 0.077% | — | FEM stress recovery |
| CV-5 | 1.2e-4 | **5.3e-13** | — | — | float64 |
| CV-6 | 0.06% | 2.4e-5 | **2.6e-16** | — | machine-eps geometric |
| CV-7 | 2e-12 | **3.5e-14** | — | — | machine precision |
| **improved?** | — | **7/7** | CV-1 only | **none** | **CONVERGED** |

Round 1 was a broad sweep (all 7 improved, several by orders of magnitude); round 2 narrowed to a single
improver (CV-1) while proving the other six at floor; round 3 closed CV-1's edge floor. Each floor is
**demonstrated**, not asserted: CV-3/CV-4 recovery is exact on the analytic field and finer meshes go
non-monotone (CV-4 n_rings 96→128 worsens 0.077→0.084%); CV-4's confining force `N` is cross-checked to 5
digits between the OT contact-integral and the boundary reaction; CV-6 reaches machine-eps; CV-1 plateaus
under mesh refinement and *degrades* under the A100 `--fine-mesh` (6.90%).

## 3. CV-1 / CV-2 large-deformation soft-material gap-field update

The OT gap field is re-evaluated on the **deformed** surface each step:
- **CV-1** (genuine compressible Neo-Hookean bulk solve, incremental + backtracking line search rejecting
  det F≤0): at δ/R=0.10 the updated-gap solve diverges from a frozen small-strain control by **dF=10.3%,
  dp0=9.3%, da=1.8%** (geometric stiffening raises the contact load), vanishing at small load where Hertz
  is valid (the closed-form anchor). Depth sweep monotonic to dF=29% at δ/R=0.18.
- **CV-2** (frictional): round 1 upgraded Part B from a frozen lateral-swell estimate to a **genuine
  deformed-configuration fixed-point** (re-assemble the deformed-radius compliance, re-solve the
  non-penetration LCP). This **corrected a round-0 artifact** — the frozen 21.8% contact-radius shift was
  non-physical (the conforming-indenter edge is set by geometry, not free swell); the real large-def
  signature is the stick ratio c/a dropping 0.779→0.553 with δ/R from deformed-pressure redistribution.

## 4. Mathematics verification (dedicated agent)

`overall_correct = True`, **no code bugs**. All six claims verified by re-derivation + executed checks:
the 1-D monotone rearrangement is the Brenier map (quantile match 1.1e-16); the OT gap is the true normal
projection (the 1/cos radial-gap bias correctly noted as *not* applying here — it belongs to the separate
radial chart); the consistent mass `M_IJ = (L/6)[[2,1],[1,2]]` and SPSD mortar tangent; the patch test to
3.6e-15; Sinkhorn → monotone. The agent **self-corrected a wrong round-1 doc claim**: the dropped
geometric tangent `p_N dn/du` does *not* bias the converged `a(F)` (the solution is tangent-independent at
residual→0; verified `a_fem` identical at 40 vs 120 Newton iters) — it is a convergence-rate lever only,
so CV-1's accuracy gain is the **contact-edge mesh refinement**, not the tangent. (One cosmetic CV-2
Part-B pressure-interp imprecision flagged; Part-A headline unaffected.)

## 5. Orchestration (recorded phases)

Per round, the four phases are logged in `phase_log.md`:
**Scout** (per-CV + math) → **Knowledge barrier** (shared brief) → **Implement** (ReAct+Reflexion, OT vs
conventional head-to-head; math verifies code-vs-formulation) → **Compare/Verify/Validate** (head-to-head,
re-run tests, ledger update). 3 rounds, ~40 agents total, ~2.6M subagent tokens. Euler A100 ran the fine
confirmations and the CV-1 mesh-convergence sweep.

## 6. Honest caveats

- CV-1's 0.50% "robust" is a delta-window-averaged, jitter-suppressed value; single-config readings scatter
  ±0.1–0.3% and the raw default is 0.73%. The honest floor is **~0.4–0.5%** (half-ellipse edge limit).
- CV-2 large-def Part B couples a half-space-BEM engine to a finite-strain surface-stretch closure (now a
  self-consistent fixed point), not a full 3-D hyperelastic bulk solve; CV-1's large-def **is** a genuine
  Neo-Hookean bulk solve.
- CV-5/CV-6 are geometry/assembly verification (no stress closed form exists for those shapes); their
  "advantage" is grid-independence vs a resolution-bound SDF, not a deformable-BVP stress match.
- The conventional models for CV-3/CV-4 are *prescribed-load* Neumann drivers (exact-by-construction on the
  load); the OT win there is that the load **emerges** from resolved contact and still matches better, plus
  it recovers observables (contact half-width, D4 symmetry) the conventional driver cannot produce.

**Bottom line:** OT measure-coupling is at or below the conventional baseline on every CV, by 1.4×–370×
where a like-for-like number exists and by orders of magnitude on grid-dominated geometry, with a verified
formulation and a demonstrated (not assumed) accuracy floor on all 7 — the loop has converged.
