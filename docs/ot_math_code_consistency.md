# OT math ↔ code ↔ Lean consistency map

Maps every optimal-transport (measure-coupling) math statement in `paper/main.tex` (equation/label) to
its code implementation (`file::function`) and, where one exists, its machine-checked Lean lemma. Built
for loop 1 of the OT-reframing campaign; every row was checked against the cited source. Numbers are the
authoritative verified set (never altered).

Scope: `solvers/contact/measure_coupling/` (`coupling.py`, `gap_field.py`, `traction.py`, `assembly.py`,
`two_body.py`), the two-body drivers under `benchmarks/contact/cv_numerical/`, and the mathlib-free Lean
project under `lean/`. Manuscript path: `paper/main.tex` (Sec. 4 `sec:tmap`, Appendix B `app:ot`).

## Status: CONSISTENT (re-verified loop 4)

`tectonic main.tex` exits 0 with 0 undefined references/citations (re-run this loop, loop 4; new labels
`eq:ot-kantorovich`, `eq:ot-cyclical`, `fig:ot-coupling`, and the `sec:tmap:ot` cross-references all resolve in
`main.aux`). `cd lean && lake build` exits 0 (mathlib-free, packages `[]`); the cited algebraic theorems are
sorry-free and depend only on `propext`/`Quot.sound` (verified by `#print axioms`, including the loop-3
`MortarMass.lean` SPD set `posdef`, `scaled_posdef`, `eigen_three`, `eigen_one`, `det_pos` and the OT-6 patch-test
extension `patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point`); only the
two `*_proposed` Brenier theorems carry `sorryAx` (build emits exactly the two expected `sorry` warnings at
`BrenierProposed.lean:93,113`) and are labelled "proposed, not machine-checked" in the paper. No math/code
mismatch found.

Loop-4 delta (manuscript-side, additive — no verified number altered):
- **Two genuinely-missing OT equations restored to the appendix proof home** (`app:ot:gap`, after `eq:ot-monge`):
  the convex **Kantorovich relaxation** `eq:ot-kantorovich` (inf over plans `π∈Π(μ̂_A,μ̂_B)` with the prescribed
  marginals) and **c-cyclical monotonicity** `eq:ot-cyclical` (`Σ c(x_i,y_i) ≤ Σ c(x_i,y_{σ(i)})` ∀σ), with the
  tightness statement (a.c. slave marginal ⇒ plan on a graph ⇒ Monge=Kantorovich) and the reduction to
  correspondence monotonicity `(τ(x_1)−τ(x_2))·(x_1−x_2)≥0`. The body (`sec:tmap:ot`, l.975) already promised
  "the Kantorovich relaxation and c-cyclical monotonicity ... in Appendix `app:ot`"; that forward-promise is now
  kept and `\eqref`-linked to the two new labels. HONESTY: c-cyclical monotonicity is scoped to the **conforming**
  Brenier map (no-crossing order-preservation); the prose states it is the structure the partial-support
  restriction *forfeits*, **not** the cause of the partial-load smearing (a separate support/admissibility
  mechanism) — per the applied-math judge's note. No code object (these are measure-theoretic existence
  statements, like the `*_proposed` Brenier theorems); not in Lean.
- **Spine threaded to the body OT subsection** (`sec:tmap:ot`, which a prior loop created but left referenced from
  nowhere): organisation paragraph (P2), contributions item 1 (P3), examples-intro two-body (P4), `sec:ex_twobody`
  detector opener (P5), and discussion opener (P6) now name `Section~\ref{sec:tmap:ot}`; the organisation
  paragraph recasts `app:ot` from "develops in full" to "supplies the measure-theoretic backbone of
  Section~\ref{sec:tmap:ot}". No equation moved — the `-body` display copies and the canonical `app:ot` copies
  both stay (the panel rejected the move-not-duplicate label churn).
- **Repro Lean list completed** (`app:repro`, TEX-2): `MortarMass.lean` added to the machine-checked-module
  enumeration (now five modules: PartitionOfUnity, MortarMass, TangentPSD, RadialSign, BrenierProposed), matching
  the body footnote at `eq:ot-mortar-mass-body` (l.940-945) and the actual `lean/OTContact/` directory.
- **Measure-coupling hero schematic added** to `sec:disc:ot` (`fig:ot-coupling`,
  `\includegraphics{fig_ot_section_loop4}`): the (a) two densities + mass-preserving χ, (b) quantile-matched
  Brenier map, (c) two-limits schematic the existing figures omit. Number-free (caption: "Schematic; no benchmark
  quantity is plotted"). The figure script `postprocessing/fig_ot_section_loop4.py` builds `τ=F_B⁻¹∘F_A` by the
  same arclength-CDF construction `MonotoneCoupling1D` uses.

Loop-3 delta (manuscript-side, additive — no verified number altered):
- **Spectral-bias measurement wired in.** New figure `fig:spectral-bias`
  (`\includegraphics{fig_spectral_bias_pub}`, Sec.~`sec:cv7-headline`) replaces the previously asserted
  "spectral bias" sentence with the MEASURED Welch-PSD comparison produced by
  `postprocessing/fig_spectral_bias.py`. Measured values printed by that script and quoted in the prose/caption:
  asperity band `[1, 14.2]` cyc/mm; chart never rolls off (retains 100% / +0.0 dB of band power); plain MLP
  rolls off at 0.33 cyc/mm (0.027% / −35.7 dB); ambient SDF at 0.25 cyc/mm (0.019% / −37.2 dB); built-in
  honesty gate `chart_wins=True`. The figure's chart uses `n_freq=128` (82,562 params), labelled in the caption
  as DISTINCT from the `n_freq=64` capstone, so the 82k vs 66k counts are never conflated.
- **Parameter-count audit table added** (`tab:capacity-audit`, after Remark `cor:capacity` in Sec.~`sec:fourier`).
  Trainable counts instantiated from the code (`scratchpad/count_params.py`; Fourier bank `freqs` is a
  non-trainable registered buffer, excluded): Fourier height chart `n_freq=64` = 66,178; ambient SDF depth-6 =
  83,073; plain-MLP ablation = 49,922. The SDF carries 25.5% MORE trainable params than the chart and still
  reconstructs 47× worse — the chart's win is capacity-matched-or-better, not bought with parameters. The
  plain-MLP row carries a footnote: it is the width/depth/optimizer-matched encoding-off ablation, NOT a
  total-param-matched comparison (the Fourier encoding necessarily widens the first layer).
- **Two OT rigor one-liners** (minimal, additive) in `app:ot:gap` / Remark `rem:ot-limits`: (1) the
  closest-point = restricted-OT-map claim now carries its well-posedness hypotheses (convex master ⇒ single-valued
  metric projection off the medial axis, `\citep[\S1.3]{santambrogio2015optimal}`; single-valued slave graph
  `z=h_A(x)`) and the partial-transport `\citep{chizat2018unbalanced}` cite; (2) `eq:ot-gap` now states `φ`/`f`
  are scalar potentials on the 1-D slave parameter line `Γ_c^A`, not fields on the ambient `R²`. Point (3)
  (`π_B` not a.e. a gradient across the medial axis) was already present at `app:ot:gap` and was NOT touched.
  All cited bib keys (`santambrogio2015optimal`, `chizat2018unbalanced`) pre-exist in `refs.bib`.

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
| `eq:ot-kantorovich` (loop 4, NEW) | convex Kantorovich relaxation: `inf_{π∈Π(μ̂_A,μ̂_B)} ∫c dπ`, `Π` = plans with the prescribed marginals; tight for a.c. slave marginal (plan on graph ⇒ Monge=Kantorovich) | (measure-theoretic existence; not in code — same status as the Brenier existence statement) | — | yes — body promise (`sec:tmap:ot` l.975) now kept; cited `villani2009optimal`, `santambrogio2015optimal` (pre-exist in `refs.bib`) |
| `eq:ot-cyclical` (loop 4, NEW) | c-cyclical monotonicity `Σ c(x_i,y_i) ≤ Σ c(x_i,y_{σ(i)})` ∀σ; quadratic cost ⇒ `(τ(x_1)−τ(x_2))·(x_1−x_2)≥0` = no-crossing order-preservation of the **conforming** map | (measure-theoretic; not in code) | — | yes — scoped to conforming Brenier map; prose states partial-support restriction *forfeits* it (NOT the cause of smearing), per judge note |
| `eq:ot-brenier-1d` | 1-D monotone map `τ=F_B⁻¹∘F_A`, quantile identity `F_B(τ(x))=F_A(x)` | `coupling.py::MonotoneCoupling1D.map` (`q=interp(xi,x,Fs1)`, `x_m=interp(q,Fm1,master_x)`) | `BrenierProposed.quantile_identity` (axiom-free) | yes — quantile to 1.1e-16 (`math_verification.md` C1) |
| `eq:ot-brenier-1d` (existence/uniqueness, pushforward) | Brenier theorem; `τ_#μ_A=μ_B` | (measure-theoretic; not in code) | `BrenierProposed.brenier_existence_uniqueness_proposed`, `monotone_map_pushes_forward_proposed` (`sorry`, prose proof) | yes — honestly labelled "proposed, not machine-checked" |
| `eq:ot-gap` | `g_N=(x−τ(x))·n`; `∇f=x−τ`, `f=½‖x‖²−φ` (potential defined; identity `g_N=∂_n f` scoped to conforming branch; `φ`,`f` are 1-D potentials on slave line `Γ_c^A`, not fields on `R²`) | `gap_field.py::GapField.sample` (`d=Xs−Xm`, `gN=(d·ns)`); `two_body.py` (`gN=(xs−xmaster)@n_hat`) | — | yes — `f` defined w/ ½‖x‖² shift; identity scoped to Rem `rem:ot-limits` (OT-P4); domain clause added loop 3 |
| `eq:ot-gap` (true normal, unbiased) | `n_s=(−h',1)/√(1+h'²)`; no 1/cosα bias | `gap_field.py` (`ns=[-hp/sec,1/sec]`); `two_body.py::assemble_two_body_contact` (`ns`) | `RadialSign.*` (separates magnitude bias from sign) | yes — unbiased normal (`math_verification.md` C2) |

## 2. The two admissible regimes (`app:ot:limits`, Remark `rem:ot-limits`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean | Consistent? |
|---|---|---|---|---|
| `rem:ot-limits` (i) conforming | full support ⇒ Brenier = arclength-monotone | `coupling.py::MonotoneCoupling1D` | — | yes |
| `rem:ot-limits` (ii) partial-support | partial contact ⇒ admissible coupling restricts to closest-point `π_B`; `g_N=‖p−c‖−R`; well-posed under convex master (single-valued metric projection off medial axis) + single-valued slave graph `z=h_A(x)` | `coupling.py::ClosestPointCoupling1D` (`_project` orthogonal foot; `map_full`); contact-band mass screen | — | yes — wording "admissible/partial-support" (OT-P3); uniqueness hypotheses + `chizat2018unbalanced`/`santambrogio2015optimal §1.3` cites added loop 3; `cv1_ot_gap.py:15-20` documents `g=|p−c|−R` |
| `eq:ot-twolimits` | piecewise map; partial limit realised by `mass=0` outside support | `coupling.py::ClosestPointCoupling1D.map_full` (`mass[i]=0` if `d>contact_band`); `two_body.py` (`gN=...*mass+(1−mass)*1e3`) | — | yes — unbalanced/partial OT screen |

## 3. The discrete coupling is a mortar assembly (`app:ot:discrete`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean lemma | Consistent? |
|---|---|---|---|---|
| `eq:ot-force` | `f_I⁺=Σ w J N_I⁺ t`; `f_K⁻=−Σ w J (N_K⁻∘χ) t` | `two_body.py` (`f[sgid]+=wq*Nq*t_q`; `f[master_ids[jm]]+=−wq*Nm0*t_q`) | — | yes |
| `eq:ot-mortar-mass` (`eq:ot-mortar-mass-body`) | consistent mass `M=(L/6)[[2,1],[1,2]]`, SPD (eigenvalues `L/2`,`L/6`; `xᵀMx=(L/3)(x₀²+x₀x₁+x₁²)>0`) | `assembly.py::assemble_contact` (`m=einsum('q,qa,qb->ab',wds,Nref,Nref)`) | `MortarMass.posdef`, `scaled_posdef`, `eigen_three`, `eigen_one`, `det_pos` (sorry-free) | yes — `M=(L/6)[[2,1],[1,2]]` exact (`math_verification.md` C3); SPD machine-checked (loop 3 module), footnote at `eq:ot-mortar-mass-body` + repro list (loop 4) |
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
| `MortarMass.lean` | `posdef`, `scaled_posdef`, `eigen_three`, `eigen_one`, `det_pos` (+`quadform`, `psd`, `symm`) | `posdef`/`scaled_posdef`: `propext`,`Quot.sound`; `eigen_three`/`eigen_one`/`det_pos`: axiom-free | `eq:ot-mortar-mass-body` footnote (Sec. 4 `sec:tmap:ot`) + `app:repro` (loop 4) |
| `TangentPSD.lean` | `rank1_psd`, `rank1_form`, `block_coeff_symm`, `quad_form_nonneg` | `propext`(/`Quot.sound`) | `app:ot:tangent` (Prop `prop:spsd`) footnote |
| `RadialSign.lean` | `active_set_iff`, `radial_gap_sign_agree` | `propext`, `Quot.sound` | `prop:active-set` (Sec. 4) footnote |
| `BrenierProposed.lean` | `quantile_identity` (axiom-free); `*_proposed` carry `sorryAx` | none / `sorryAx` | `app:ot:gap` footnote + `app:repro` |

Reproduce: `cd lean && lake build` (Lean 4 v4.30.0, mathlib-free, ~1 s). The `lean/` source is present in the
worktree (untracked; `.lake/` build artifacts are gitignored via `lean/.gitignore`); committing it is left to
the maintainer per the repo's commit policy.
