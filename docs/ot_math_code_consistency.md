# OT math ↔ code ↔ Lean consistency map

Maps every optimal-transport (measure-coupling) math statement in `paper/main.tex` (equation/label) to
its code implementation (`file::function`) and, where one exists, its machine-checked Lean lemma. Built
for loop 1 of the OT-reframing campaign; every row was checked against the cited source. Numbers are the
authoritative verified set (never altered).

Scope: `solvers/contact/measure_coupling/` (`coupling.py`, `gap_field.py`, `traction.py`, `assembly.py`,
`two_body.py`), the two-body drivers under `benchmarks/contact/cv_numerical/`, and the mathlib-free Lean
project under `lean/`. Manuscript path: `paper/main.tex` (Sec. 4 `sec:tmap`, Appendix B `app:ot`).

## Status: CONSISTENT (re-verified loop 1)

`tectonic main.tex` exits 0 with 0 undefined references/citations (re-run this loop). `cd lean && lake build`
exits 0 (7 jobs, mathlib-free, packages `[]`); the cited algebraic theorems are sorry-free and depend only on
`propext`/`Quot.sound` (verified by `#print axioms` this loop, including the OT-6 patch-test extension
`patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point`); only the two
`*_proposed` Brenier theorems carry `sorryAx` (build emits exactly the two expected `sorry` warnings at
`BrenierProposed.lean:93,113`) and are labelled "proposed, not machine-checked" in the paper. No math/code
mismatch found.

Loop-1 code-side delta: `solvers/contact/measure_coupling/two_body.py` PATCH TEST docstring (l.29-40) was
corrected (OT-3) to stop equating the P1 partition of unity with the OT mass marginal — it now reads "DISTINCT
FROM, and downstream of, the OT mass marginal `chi_# mu_A = mu_B`", matching the manuscript wording at
`app:ot:discrete` and the abstract/intro/discussion. The docstring also now points at the machine-checked
Lean precondition (`patch_test_resultant`).

## 1. Measure coupling and the Brenier map (`app:ot:gap`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean lemma | Consistent? |
|---|---|---|---|---|
| `eq:ot-measure` | arclength measure `dμ_X = √(1+h'²) dx`; normalised CDF `F̂=F/F[-1]` (equal total mass) | `coupling.py::_arclength_cdf`; `MonotoneCoupling1D.__init__` (`_Fs1=Fs/Fs[-1]`, `_Fm1=Fm/Fm[-1]`) | — | yes — renormalisation now stated in paper (OT-P5) |
| `eq:ot-monge` | quadratic-cost Monge problem, `τ=∇φ` convex; AC hypothesis `√(1+h'²)≥1>0` | `coupling.py::_arclength_cdf` docstring ("strictly increasing ⇒ F⁻¹ well-defined") | — | yes — AC/equal-mass hypotheses now stated (OT-P5) |
| `eq:ot-brenier-1d` | 1-D monotone map `τ=F_B⁻¹∘F_A`, quantile identity `F_B(τ(x))=F_A(x)` | `coupling.py::MonotoneCoupling1D.map` (`q=interp(xi,x,Fs1)`, `x_m=interp(q,Fm1,master_x)`) | `BrenierProposed.quantile_identity` (axiom-free) | yes — quantile to 1.1e-16 (`math_verification.md` C1) |
| `eq:ot-brenier-1d` (existence/uniqueness, pushforward) | Brenier theorem; `τ_#μ_A=μ_B` | (measure-theoretic; not in code) | `BrenierProposed.brenier_existence_uniqueness_proposed`, `monotone_map_pushes_forward_proposed` (`sorry`, prose proof) | yes — honestly labelled "proposed, not machine-checked" |
| `eq:ot-gap` | `g_N=(x−τ(x))·n`; `∇f=x−τ`, `f=½‖x‖²−φ` (potential defined; identity `g_N=∂_n f` scoped to conforming branch) | `gap_field.py::GapField.sample` (`d=Xs−Xm`, `gN=(d·ns)`); `two_body.py` (`gN=(xs−xmaster)@n_hat`) | — | yes — `f` now defined w/ ½‖x‖² shift; identity scoped to Rem `rem:ot-limits` (OT-P4) |
| `eq:ot-gap` (true normal, unbiased) | `n_s=(−h',1)/√(1+h'²)`; no 1/cosα bias | `gap_field.py` (`ns=[-hp/sec,1/sec]`); `two_body.py::assemble_two_body_contact` (`ns`) | `RadialSign.*` (separates magnitude bias from sign) | yes — unbiased normal (`math_verification.md` C2) |

## 2. The two admissible regimes (`app:ot:limits`, Remark `rem:ot-limits`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean | Consistent? |
|---|---|---|---|---|
| `rem:ot-limits` (i) conforming | full support ⇒ Brenier = arclength-monotone | `coupling.py::MonotoneCoupling1D` | — | yes |
| `rem:ot-limits` (ii) partial-support | partial contact ⇒ admissible coupling restricts to closest-point `π_B`; `g_N=‖p−c‖−R` | `coupling.py::ClosestPointCoupling1D` (`_project` orthogonal foot; `map_full`); contact-band mass screen | — | yes — wording changed from "degenerates" to "admissible/partial-support" (OT-P3); `cv1_ot_gap.py:15-20` documents `g=|p−c|−R` |
| `eq:ot-twolimits` | piecewise map; partial limit realised by `mass=0` outside support | `coupling.py::ClosestPointCoupling1D.map_full` (`mass[i]=0` if `d>contact_band`); `two_body.py` (`gN=...*mass+(1−mass)*1e3`) | — | yes — unbalanced/partial OT screen |

## 3. The discrete coupling is a mortar assembly (`app:ot:discrete`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean lemma | Consistent? |
|---|---|---|---|---|
| `eq:ot-force` | `f_I⁺=Σ w J N_I⁺ t`; `f_K⁻=−Σ w J (N_K⁻∘χ) t` | `two_body.py` (`f[sgid]+=wq*Nq*t_q`; `f[master_ids[jm]]+=−wq*Nm0*t_q`) | — | yes |
| `eq:ot-mortar-mass` | consistent mass `M=(L/6)[[2,1],[1,2]]` | `assembly.py::assemble_contact` (`m=einsum('q,qa,qb->ab',wds,Nref,Nref)`) | — | yes — `M=(L/6)[[2,1],[1,2]]` exact (`math_verification.md` C3) |
| `eq:ot-marginal` | host weights partition of unity `Σ_K(N_K⁻∘χ)=1` — PoU, **not** the OT mass marginal | `two_body.py::_locate_master` (returns `(j,1−t,t)`, sum≡1); `coupling.py::ClosestPointCoupling1D.map_full` (`N0=1−t,N1=t`) | `PartitionOfUnity.p1_host_weights`, `marginal_two_host` (sorry-free) | yes — OT-3/P1: relabelled PoU at all ~10 paper sites + the `two_body.py` docstring (loop 1); paper now states "where the foot lands" (measure preservation) vs "how the reaction is split" (PoU, what the patch test exercises) |
| `prop:patch` (boxed Proposition) | constant pressure transmits exactly: `Σ_K f_K⁻ = −Σ_I f_I⁺ = 0` **iff** single-host + PoU + Gauss exactness (NOT the OT marginal) | `two_body.py::assemble_two_body_contact` (constant-`t` master sum; self-test 1) | `PartitionOfUnity.patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point` (sorry-free, `propext`/`Quot.sound`) | yes — OT-6: Proposition + two-line proof added; Lean resultant extension built + `#print axioms`-checked this loop before the paper cites it |
| patch test (measured) | uniform pressure → 1.4e-16 across non-matching mesh; node-lumped non-uniformity 67.3 (structural, mesh-independent) | `two_body.py` self-test 1 (`F_line` exact, force balance) | (PoU + Gauss exactness is the algebraic precondition; `patch_test_resultant`) | yes — 1.4e-16 (`final_report.md` §7); 67.3 (NE-7) |

## 4. The symmetric four-block tangent (`app:ot:tangent`, Proposition `prop:spsd`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean lemma | Consistent? |
|---|---|---|---|---|
| `eq:ot-gapvar` | frozen-geometry variation: drop **both** `∂n` and `∂χ` (`∂(N_K⁻∘χ)/∂u`) | `two_body.py` docstring l.11-14 ("drop the geometric d(n) term"); self-test 2 `resid_frozen` freezes `n_hat` AND `(N_K∘χ)` | — | yes — OT-P2: paper now states both dropped terms, matching `resid_frozen` |
| `eq:twobody-tangent` | 4-block `K_ss,K_sm,K_ms=K_sm^T,K_mm`, each `∝ (w_a w_b)(n⊗n)` | `two_body.py::assemble_two_body_contact` (signed double loop over `entries`, `coeff=wfac*(sa*wa)*(sb*wb)`, `blk=coeff*nn`) | `TangentPSD.four_block_diag_nonneg` | yes |
| `prop:spsd` | symmetric SPSD: `K=ε Σ w J (w⊗w)⊗(n⊗n) ⪰ 0`; `K_ms=K_sm^T` | `two_body.py` self-test 1 (`Kd==Kd.T` to 1e-12; `min eig > −1e-9`) | `TangentPSD.rank1_psd`, `rank1_form`, `block_coeff_symm`, `quad_form_nonneg` (sorry-free) | yes — Proposition + rank-one-sum proof added (OT-P7) |
| `tab:ot:gates` (symbolic `dR/du`, FD `df/du`) | 64 entries exact; FD 3.45e-11 — **on the frozen-geometry residual** | `two_body.py` self-test 2 (FD vs `resid_frozen`, rel<1e-6 → 3.45e-11); self-test 3 (closest-point) | — | yes — table footnote scopes both gates to the frozen residual (OT-P7) |

## 5. Verified two-body / N-body results (`app:ot:results`, `sec:ex_twobody`)

| Paper claim | Value | Source | Consistent? |
|---|---|---|---|
| CV-8 deformable Hertz `a` (nx 24→192) | 5.14→4.14→2.96→2.75% | `cv8_deformable_ot.py`, `runs/cv8_deformable_ot/metrics.json`; `final_report.md` §7 | yes |
| CV-8 `p_0` | 6.34→5.68→5.91→5.82% (plateau ~5.8%) | same | yes |
| CV-8 patch test / force balance | 1.4e-16 / 1.27e-19 | same | yes |
| CV-9a centre mean stress / anisotropy (D4) / force balance | 0.58% / 0.20–0.22% / 3.71e-15 | `cv9_nbody_array_ot.py`; `tests/test_cv9_nbody_ot.py` | yes |
| tangent FD / force balance | 3.45e-11 / 4.4e-16 | `two_body.py` self-tests | yes |
| OT-vs-conventional head-to-head (CV-1..7) | 1.59→0.50 (3.2×), 11.15→0.03 (370×), 1.62→0.23 (6.3×), 0.11→0.077 (1.4×), 1.99e-2→5.3e-13, frozen→2.6e-16, 1.5e-4→3.5e-14 | `final_report.md` §1 Table 1 | yes — `tab:ot_vs_conv` + `fig:ot_advantage` (P1, FIG-OT-ADV-1) |

## 6. Honest residuals / caveats carried in the paper (unchanged)

- CV-8 ~3–6% `a`/`p_0` is contact-edge noise (∞ Hertz edge slope over one element; `p_0` plateaus ~5.8%, a floor). Stated in `sec:ex_twobody`, `tab:ot:gates` caption, abstract (N8 parenthetical), and `sec:verif_summary` closing paragraph (P9).
- Both two-body solves: single-realization, plane-strain CST small-strain, frictionless, rigid outer walls (CV-9a). Stated in the same places + `sec:disc:limits`.
- Brazilian/nine-disc conventional baselines are prescribed-load (Neumann-exact); the OT win there is structural (emergent load, recovered half-width/symmetry), not a like-for-like stress win. Stated in `tab:ot_vs_conv` caption, the nine-disc callout (P5), and `final_report.md` §6.
- Dropped geometric tangent (`∂n`, `∂χ`) is a Newton-path lever only; the converged `a_fem` is tangent-independent (identical at 40 vs 120 iters; `math_verification.md` §R2). Stated in `app:ot:tangent` and `sec:disc:limits`.
- Cattaneo 370× and Hertz 3.2× are reported as same-mesh (Cattaneo on the converged n=3200 mesh; both conventional and OT from `final_report.md` Table 1). The 5–11% Cattaneo figure in `sec:ex_cattaneo` is the coarse-mesh L1-vs-closed-form number and is explicitly distinguished from the converged 0.03%.

## 7. Lean ↔ paper citation map

| Lean file | sorry-free theorems cited | axioms (`#print axioms`) | paper citation site |
|---|---|---|---|
| `PartitionOfUnity.lean` | `p1_host_weights`, `marginal_two_host`, `patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point` | `propext`, `Quot.sound` | `app:ot:discrete` footnote (eq:ot-marginal + Prop `prop:patch`) |
| `TangentPSD.lean` | `rank1_psd`, `rank1_form`, `block_coeff_symm`, `quad_form_nonneg` | `propext`(/`Quot.sound`) | `app:ot:tangent` (Prop `prop:spsd`) footnote |
| `RadialSign.lean` | `active_set_iff`, `radial_gap_sign_agree` | `propext`, `Quot.sound` | `prop:active-set` (Sec. 4) footnote |
| `BrenierProposed.lean` | `quantile_identity` (axiom-free); `*_proposed` carry `sorryAx` | none / `sorryAx` | `app:ot:gap` footnote + `app:repro` |

Reproduce: `cd lean && lake build` (Lean 4 v4.30.0, mathlib-free, ~1 s). The `lean/` source is present in the
worktree (untracked; `.lake/` build artifacts are gitignored via `lean/.gitignore`); committing it is left to
the maintainer per the repo's commit policy.
