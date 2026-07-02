# OT math ↔ code ↔ Lean consistency map

Maps every optimal-transport (measure-coupling) math statement in `paper/main.tex` (equation/label) to
its code implementation (`file::function`) and, where one exists, its machine-checked Lean lemma. Built
for loop 1 of the OT-reframing campaign; every row was checked against the cited source. Numbers are the
authoritative verified set (never altered).

Scope: `solvers/contact/measure_coupling/` (`coupling.py`, `gap_field.py`, `traction.py`, `assembly.py`,
`two_body.py`), the two-body drivers under `benchmarks/contact/cv_numerical/`, and the mathlib-free Lean
project under `lean/`. Manuscript path: `paper/main.tex` (Sec. 4 `sec:tmap`, Appendix B `app:ot`).

## Status: CONSISTENT (re-verified loop 12; CV-8 HALF-PLANE RE-REGIME — root-cause fix, numbers CHANGED. A multi-agent root-cause panel traced the CV-8 "poor" deformable-Hertz result (a 2.75%, p0 5.82%, non-refining) to a FINITE-SIZE / half-plane REFERENCE-regime mismatch, NOT a solver bug: the shallow clamped blocks (W=1,H=0.5 → a/H≈0.43) are ~10% stiffer than the elastic half-plane the closed form assumes (E*_eff/E*_hp≈1.10, measured 4FR/πa²=0.607 vs 0.549), biasing a low / p0 high by ~5-6% (one coupled effect; force balance was machine-zero and the patch test 1.4e-16 throughout — the elasticity + OT coupling were always correct). FIX: run CV-8 in the half-plane regime (deep/wide blocks W=2,H=2 → a/H≤0.14), where E*_eff/E*_hp≈1.0 and the errors converge to the ~2% discrete-edge + CST stress-recovery floor. NEW numbers: finest nx=260 → a **1.64%**, p0 **2.26%**; ensemble (nx=220, 5 seeds) a **1.42±0.32%**, p0 **2.31±0.07%** (seed 7 now the MEDIAN, not best-of-five). Panel refuted penalty-stiffness (flat ~6% over eps_fac∈[300,1200]), recovery-operator bias (<0.1% on synthetic Hertz), and plane-stress/strain mismatch (already consistent). Driver `cv8_deformable_ot.py` parametrised by (W,H,δ,grade); `metrics.json`/`ensemble.json` regenerated; `fig:cv8-hertz` regenerated; all manuscript sites updated (abstract, intro, sec:ex_twobody + tab:cv8-ensemble, tab:twobody:headline, app:ot tab:ot:gates, conclusion, sec:disc, capability matrices). `tectonic main.tex` exits 0, 0 undefined refs/cites, 0 multiply-defined labels. Prior: loop 11 editorial dedup; loop 10 CV-9a mesh-convergence sweep; loop 9 CV-8 five-seed ensemble; loop 8 coherence; below unchanged — historical loop entries preserve the record of the pre-re-regime numbers.)

Loop-11 delta (EDITORIAL ONLY — repetition dedup + prose; no math/number/code/Lean touched; `tectonic main.tex` exits 0, 0 undefined references/citations, 0 multiply-defined labels, re-verified in `main.log`):
- **Mandate:** writing-quality pass on `paper/main.tex`. The measured repetition was severe (2.75% restated 28×, 5.82% 21×, 67× 21×, 3.45e-11 16×, 1.4e-16 14×, 370× 8×, 61%/98% 11/10×). Dedup REMOVES redundant restatements (or points them to the canonical anchor); it never alters a value. Each over-restated fact is retained at its load-bearing anchors: abstract headline, intro first-statement, results tables (`tab:twobody:headline`, `tab:cv8-ensemble`, `tab:ot_vs_conv`, `tab:cv_summary`, appendix `tab:ot:gates`), figure captions, and the conclusion.
- **Number-dedup applied (8 majority-accepted edits, ~10 number restatements removed, 0 values altered, 0 disclosures touched):**
  1. CV-8 second sweep restatement (sec:ex_twobody): cut the two duplicate 4-point convergence strings `5.14→4.14→2.96→2.75%` / `6.34→5.68→5.91→5.82%` (verbatim repeat one sentence after their defining statement; survive at the defining sentence + gate table) → "decreases monotonically … plateaus near 5.8%".
  2. `fig:cv8-hertz` caption: cut the same two parenthetical 4-point strings (the figure plots them); endpoints 2.75%/5.82%/near-5.8% retained.
  3–4. `sec:disc:ot` (the 4th narrative ledger): the patch-test (`1.4e-16`, `67×`) and two-body-tangent (`3.45e-11`, `4.4e-16`) restatements → pointers to `tab:twobody:headline` and `prop:spsd`; conceptual three-consequences framing kept.
  5. `rem:ot-verdict`: Cattaneo `11.15%→0.03%` restatement dropped, `370×` class label kept (defining cell remains in `tab:ot_vs_conv`).
  6. CV-7 scope paragraph: end-of-subsection `61%/35%/98%` restatement → pointer to `sec:cv7-headline` + `tab:cv7-atlas-sdf`; the surrounding scope/caveat sentences and the disambiguation paragraph (`61%` = real Inada, `98%/35%` = synthetic decoder) PRESERVED.
  7. Appendix patch-test proof: dropped the `67×` cross-comparison clause (proof keeps its own measured `1.4e-16`).
  8. Appendix tangent paragraph: dropped the duplicate `3.45e-11` value (restated 23 lines later + in `tab:ot:gates`) → table pointer.
- **Prose tightenings applied (11 majority-accepted, meaning-preserving, away from protected anchors):** abstract self-echo split (l.119); intro empty-intensifier removed ("We take that pairing literally", l.157); dangling-antecedent bug fixed ("On geometrically complex bodies it breaks down" → "this fails", l.190–191); OT-opener "two ways to get (g,n)" middle restatement deleted (l.737) and "it is/it is/it returns" triad de-parallelised (l.741–743); OT-identification preview collapsed to one `sec:tmap:ot` pointer (l.754–759); `sec:tmap:ot` opener de-hedged ("is not an ad hoc construction but" → declarative, trailing fields-clause dropped, l.878); OT-section "resolves the apparent tension" editorial restatement cut after the proof (l.960); two-body opener three-phrasings tightened (l.1896); conclusion roadmap-of-the-roadmap scaffolding sentence deleted (l.2705).
- **Items SKIPPED to honor the hard constraints (disclosure/anchor protection overrides bare majority):** the conclusion plateau-diagnosis sentence ("The 3.8–4.8% plateau is thus a diagnosed property of constraint enforcement, orthogonal to the … representation result", l.2879–2881) was a 2–1 accept, but it is a measured-limitation DISCLOSURE that bounds the claim (residual is enforcement, not representation) — KEPT intact per the "preserve every honest residual/disclosure" constraint and the editor judge's grounded objection. The Cattaneo edit (item 5) kept format parallel with its Hertz sibling per the skeptic's parallelism note.
- **Figures:** `fig_ot_advantage_loop1.py` is ALREADY in the audit's recommended clean state (in-axes fonts 8–9 pt, panel (a) carries only the four advantage-factor chips + one floor bracket and no per-bar rotated numbers, panel (c) shows only the two endpoint labels with no ledger box — the figures audit described a stale earlier render); regenerated this loop and verified (`2.75% / 5.82% / force balance 1.3e-19`). The remaining figure-polish items (rock_joint_capstone panel-(f) raster-table + super-title removal; cv9_nbody on-field ledger → caption; jet→perceptual colormaps on the four PyVista renders; multiarc label collision) require regenerating from `runs/rock_joint_capstone/`, `downloads/inada_granite/`, and PyVista — none present in this checkout (gitignored / cluster-bound). They are DEFERRED to the Euler regeneration workflow; the committed PNGs remain valid embeddings and the document compiles. No figure value changed.

Loop-9 delta (CV-8 ensemble disclosure + reproducibility, additive — the headline 2.75%/5.82% is unchanged and re-derived exactly; `tectonic main.tex` exits 0, 0 undefined references/citations, 0 multiply-defined labels):
- **Framing (judge consensus, 3 panels):** KEEP `2.75%`/`5.82%` (mesh seed 7) as the labeled single-realization headline gate; carry the five-seed ensemble alongside; do NOT substitute the ensemble mean. The cherry-picking concern is neutralized by *disclosure* (label seed 7 as best-of-five, report mean±sd and range), not substitution. Honesty enforced everywhere: seed 7 is the **best of five on BOTH metrics** (`a` 2.75% = min; `p_0` 5.82% = low end of the band); only the `p_0` floor is called realization-robust, the half-width is explicitly realization-dependent.
- **Measured ensemble (nx=192, R=2.0, δ=0.02, n_load=6, graded mesh, jitter 0.03; mesh seeds 7/11/17/23/31; all force balances ~1e-18..1e-19), re-run this loop and matching the prior values EXACTLY:** `a_relerr` {2.75, 3.52, 4.30, 3.71, 3.09}% → mean **3.47%**, population sd **0.53%** (sample sd 0.59%), range [2.75, 4.30]% — realization-DEPENDENT; `p0_relerr` {5.82, 5.96, 6.00, 6.02, 5.90}% → mean **5.94%**, population sd **0.07%** (sample sd 0.08%), range [5.82, 6.02]% — realization-INDEPENDENT (genuine plane-strain CST + finite-penalty systematic floor). sd convention disclosed as population (ddof=0) in the table/figure captions.
- **Manuscript edits (all additive, headline numbers preserved):** new `\paragraph{Ensemble disclosure across realisations.}` + `tab:cv8-ensemble` + `fig:cv8-ensemble` in `sec:ex_twobody` (l.2058–2114); headline-table caption/rows labeled best-of-five-on-both-metrics + ensemble (`tab:twobody:headline`, l.2145–2158); abstract (l.134–136 and l.379–381), conclusion (l.2890–2891), `fig:ot_advantage` caption (l.2297–2298), `rem:ot-verdict` floor sentence (l.2322–2326), the in-section "single-realisation" caveat (l.2179–2180 now points to the ensemble), and the appendix prose + `tab:ot:gates` caption (l.3612–3615, l.3625–3629). Every pre-existing residual caveat (~3–6% contact-edge floor, single-realisation/now-evidenced, plane-strain CST small-strain, frictionless, rigid outer walls in CV-9a) is PRESERVED, not deleted.
- **Reproducibility hygiene (judge required-fix):** the five non-headline values previously lived only hardcoded in the figure generator. Now `cv8_deformable_ot.py` threads `mesh_seed` through `hertz_test`→`hertz_convergence` to the curved-block `block_mesh(..., seed=mesh_seed)`, adds a `--mesh-seed` CLI flag and a `--mode ensemble` sweep (`hertz_ensemble(seeds=...)`) that writes `runs/cv8_deformable_ot/ensemble.json` (per-seed `a/p0_relerr`, force balance, and population+sample sd stats); `postprocessing/fig_cv8_ensemble.py` now READS that JSON (committed-fallback to the same MEASURED arrays if `runs/` is absent, since it is gitignored). A referee regenerates the ensemble with `python3 benchmarks/contact/cv_numerical/cv8_deformable_ot.py --mode ensemble` — no source edit. The default `cv8_deformable_ot.py` gate run is unchanged (seed 7).

Loop-8 delta (manuscript-side, additive — no verified number altered; `tectonic main.tex` exits 0, 0 undefined references/citations, 0 multiply-defined labels):
- **Item 1 — abstract parenthetical reordered** (l.142): `(Cattaneo stick radius, Hertz half-width)` → `(Hertz half-width, Cattaneo stick radius)`. The adjacent conventional range `1.59\%`--`11.15\%` is Hertz-then-Cattaneo (Table `tab:ot_vs_conv` l.2197 Hertz `a`=1.59%, l.2198 Cattaneo `c/a`=11.15%; contributions bullet l.391 pairs Cattaneo with 11.15%); the parenthetical was reversed, inverting the value→benchmark mapping in the abstract. Reorder aligns it with the nearer conventional list. No number moved.
- **Item 2 — undefined acronym expanded** (l.2224, Fig. `fig:ot_advantage` caption): `symmetric SPSD four-block mortar tangent` → `symmetric positive-semidefinite four-block mortar tangent`. "SPSD" occurred exactly once and was never expanded; "symmetric positive-semidefinite" is the house form used throughout. ("symmetric SPSD" was also redundant — the S already means symmetric.)
- **Item 3 — CST defined at first use** (l.2115): `plane-strain CST small-strain` → `plane-strain constant-strain-triangulation (CST) small-strain`. The expansion previously appeared only later (l.3042); the three later "CST" instances (l.2256, 2766, 3548) are left as-is.
- **Item 4 — orphan float closed** (Appendix A, acceptance-tolerance sentence ~l.2942): appended `(these acceptance tolerances are collected in Table~\ref{tab:closedforms})`. `tab:closedforms` carried a `\label` (l.3143) but had zero `\ref` anywhere; it is now cited exactly where the tolerances it consolidates are introduced.
- Item 5 (retarget the friction-plateau cross-reference at l.3942 from `sec:discr` to `sec:disc:limits`) was REJECTED by both judges — the current `sec:discr` target is defensible (the plateau is named there) and the "discussed there" anchor ties the reference to the cited section; left as-is.

`tectonic main.tex` exits 0 with 0 undefined references/citations and 0 multiply-defined labels (re-run this loop, loop 7; `grep -icE 'undefined (reference|citation)' main.log` = 0, `grep -icE 'multiply.defined' main.log` = 0; no rerun/labels-changed warning; the regenerated `cv8_hertz_convergence_pub.png` loads from `../figures/`). The loop-6 state below is otherwise unchanged: `grep -ci undefined
main.log` = 0; labels `eq:ot-kantorovich-body`/`eq:ot-cyclical-body` (body), `eq:ot-kantorovich`/`eq:ot-cyclical`
(appendix proof home), `eq:ot-unbalanced` (Fig 37/page 51, co-located with the new schematic Eq 82),
`fig:ot-coupling`, `fig:ot-unbalanced` (loop-6 NEW), `sec:ot`/`sec:tmap:ot`, and the
`TranslationInvariance` footnote cross-references all resolve in `main.aux`; no duplicate `\label`). `cd lean &&
lake build` exits 0 (mathlib-free, packages `[]`, 9 jobs); the cited algebraic theorems are sorry-free and depend
only on `propext`/`Quot.sound` (verified by `#print axioms` this loop, including the loop-3 `MortarMass.lean` SPD
set `posdef`, `scaled_posdef`, `eigen_three`, `eigen_one`, `det_pos`, the OT-6 patch-test extension
`patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point`, and the **new
`TranslationInvariance.lean` set `patch_resultant_list`, `tangent_translation_null`,
`rigid_translation_gap_invariant`, `foot_resultant_zero`, `rigid_translation_balanced` — all `[propext,
Quot.sound]`**); only the two `*_proposed` Brenier theorems carry `sorryAx` (build emits exactly the two expected
`sorry` warnings at `BrenierProposed.lean:93,113`) and are labelled "proposed, not machine-checked" in the paper.
No math/code mismatch found.

Loop-4 delta (manuscript-side, additive — no verified number altered):
- **Two core OT equations PROMOTED into the main-body OT subsection** (EQ1/EQ2/X1; `sec:tmap:ot`, after
  `eq:ot-monge-body`): the convex **Kantorovich relaxation** `eq:ot-kantorovich-body` (inf over plans
  `π∈Π(μ̂_A,μ̂_B)` with the prescribed marginals) and **c-cyclical monotonicity** `eq:ot-cyclical-body`
  (`Σ c(x_i,y_i) ≤ Σ c(x_i,y_{σ(i)})` ∀σ), with the tightness statement (a.c. slave marginal ⇒ plan on a graph ⇒
  Monge=Kantorovich) and the reduction to correspondence monotonicity `(τ(x_1)−τ(x_2))·(x_1−x_2)≥0`. This makes
  the existence chain Monge→(infeasible)→Kantorovich→cyclical-monotone→Brenier self-contained in the body; the
  body now `\eqref`s the two `-body` labels and the closing-sentence pointer (`sec:tmap:ot`, formerly l.975) is
  reframed from "the Kantorovich relaxation … in Appendix" to "stated here … their measure-theoretic existence and
  pushforward proof … are given in Appendix~`app:ot`". The appendix copies `eq:ot-kantorovich`/`eq:ot-cyclical`
  (proof home) are KEPT — they carry the appendix's own internal `\eqref`s (l.3280, 3282, 3345, 3354) and the
  detailed tightness/Brenier flow — so this is a PROMOTE with distinct labels, not a move (no appendix `\ref`
  breaks; no duplicate `\label`). HONESTY: c-cyclical monotonicity is scoped to the **conforming** Brenier map
  (no-crossing order-preservation); the body block ties it to `eq:ot-brenier-1d-body`, and the appendix prose
  states it is the structure the partial-support restriction *forfeits*, **not** the cause of the partial-load
  smearing — per the applied-math judge's note. No code object (these are measure-theoretic existence statements,
  like the `*_proposed` Brenier theorems); not in Lean.
- **Spine threaded to the body OT subsection** (`sec:tmap:ot`, which a prior loop created but left referenced from
  nowhere): organisation paragraph (P2), contributions item 1 (P3), examples-intro two-body (P4), `sec:ex_twobody`
  detector opener (P5), and discussion opener (P6) now name `Section~\ref{sec:tmap:ot}`; the organisation
  paragraph recasts `app:ot` from "develops in full" to "supplies the measure-theoretic backbone of
  Section~\ref{sec:tmap:ot}". No equation moved — the `-body` display copies and the canonical `app:ot` copies
  both stay (the panel rejected the move-not-duplicate label churn).
- **Repro Lean list completed** (`app:repro`, TEX-2): the machine-checked-module enumeration now lists all SIX
  modules — PartitionOfUnity, MortarMass, TangentPSD, **TranslationInvariance** (loop-4 NEW), RadialSign,
  BrenierProposed — matching the body footnotes (`eq:ot-mortar-mass-body` l.954, the `app:ot:tangent`
  rigid-translation footnote l.3458) and the actual `lean/OTContact/` directory.
- **Measure-coupling hero schematic added** to `sec:disc:ot` (`fig:ot-coupling`,
  `\includegraphics{fig_ot_section_loop4}`): the (a) two densities + mass-preserving χ, (b) quantile-matched
  Brenier map, (c) two-limits schematic the existing figures omit. Number-free (caption: "Schematic; no benchmark
  quantity is plotted"). The figure script `postprocessing/fig_ot_section_loop4.py` builds `τ=F̂_B⁻¹∘F̂_A` by the
  same arclength-CDF construction `MonotoneCoupling1D` uses (`_Fs1=Fs/Fs[-1]`, max|F̂_B(τ)−F̂_A|=1.1e-16).
- **OT-MATH-2 normalisation fix (Brenier display + propagated to EVERY symbol site).** The appendix display
  `eq:ot-brenier-1d` (l.3273) and the body display `eq:ot-brenier-1d-body` (l.902) both carry the normalised
  `τ=F̂_B⁻¹∘F̂_A` with `F̂_X=F_X/F_X(θ_max)` and `F̂_B(τ(θ_A))=F̂_A(θ_A)`, matching the code
  (`coupling.py::MonotoneCoupling1D`, `_Fs1=Fs/Fs[-1]`, `_Fm1=Fm/Fm[-1]`). The false "both sides scale by the same
  total mass" parenthetical is GONE, replaced by the correct admissibility argument (the unnormalised composition
  would query `F_B⁻¹` at a quantile up to `F_A(θ_max)` outside `F_B`'s range; the renormalisation by the in general
  unequal totals `F_A(θ_max)≠F_B(θ_max)` is what makes `χ_#μ̂_A=μ̂_B` admissible). This loop the `F̂` convention
  was made UNIFORM across all eight Brenier-map symbol sites: abstract (l.125), `fig:tm-ot-limits` node label +
  caption (l.1905, 1925), CV-5 conforming-limit prose (l.2240), `fig:ot-coupling` caption (l.2606-2608),
  `sec:disc:ot` prose (l.2622), and the appendix display + two-limits + `rem:ot-limits` (l.3269, 3273, 3316, 3336).
  The only remaining bare `F_B⁻¹` (l.3279) is deliberate — it is the explanatory clause stating WHAT goes wrong
  WITHOUT normalisation. Type-row in §1 updated.
- **OT-MATH-1 type-consistency fix (conforming-regime potential).** `app:ot:gap` (l.3292-3306) no longer claims
  `f=½‖x‖²−φ` is simultaneously a 1-D scalar potential on `Γ_c^A` and satisfies the `R²` gradient identity
  `∇f(x)=x−τ(x)`. It now states the Brenier gradient is the PARAMETER-space map `τ=φ'` (monotone, conservative, no
  circulation), and routes the physical-gap energy-consistency to the matched-normal property `∇_x g_N=n` of
  Remark `prop:matched-normal` (already proven; `RadialSign.lean`). No `g_N=∂_n f` claim remains. §1 `eq:ot-gap`
  row updated to reflect the parameter-space-gradient / physical-gap split.
- **OT-LEAN-3 rigid-translation invariance machine-checked (NEW module `TranslationInvariance.lean`).** The
  `app:ot:tangent` claim was tightened from the imprecise "master row-sum of `K_ms` vanishes" (false in isolation:
  `Σ_J K_ms[K,J]=−ε Σ_q w_q J (N_K⁻∘χ)(n⊗n)≠0`) to "the COMBINED slave–master gap variation vanishes under a
  uniform translation, `Σ_I N_I⁺=Σ_K(N_K⁻∘χ)=1`, so `ε(Σ N_I⁺−Σ N_K⁻∘χ)=0` and the net contact force is null."
  Three theorems now certify this sorry-free: `tangent_translation_null` (`ε d+(−(ε((d−k)+k)))=0`),
  `rigid_translation_gap_invariant` (`d−((d−k)+k)=0`), and `patch_resultant_list` (the patch resultant of
  `prop:patch` summed over the FULL Gauss-foot list, by list induction, not one foot). §4 + §7 Lean-citation map
  updated (now SIX modules).
- **Two-limits ordering harmonised.** Body `eq:ot-twolimits-body` (l.918) and appendix `eq:ot-twolimits` (l.3333)
  now both list the arclength-monotone (conforming) branch FIRST, then the closest-point (partial-support) branch —
  resolving the prior body-vs-appendix ordering mismatch. No content change.

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
| `eq:ot-kantorovich-body` (body, loop 4) / `eq:ot-kantorovich` (appendix proof home) | convex Kantorovich relaxation: `inf_{π∈Π(μ̂_A,μ̂_B)} ∫c dπ`, `Π` = plans with the prescribed marginals; tight for a.c. slave marginal (plan on graph ⇒ Monge=Kantorovich) | (measure-theoretic existence; not in code — same status as the Brenier existence statement) | — | yes — promoted to body (EQ1) with distinct `-body` label; appendix copy kept as proof home; cited `villani2009optimal`, `santambrogio2015optimal` (pre-exist in `refs.bib`) |
| `eq:ot-cyclical-body` (body, loop 4) / `eq:ot-cyclical` (appendix proof home) | c-cyclical monotonicity `Σ c(x_i,y_i) ≤ Σ c(x_i,y_{σ(i)})` ∀σ; quadratic cost ⇒ `(τ(x_1)−τ(x_2))·(x_1−x_2)≥0` = no-crossing order-preservation of the **conforming** map | (measure-theoretic; not in code) | — | yes — promoted to body (EQ2) with distinct `-body` label; scoped to conforming Brenier map; appendix prose states partial-support restriction *forfeits* it (NOT the cause of smearing), per judge note |
| `eq:ot-brenier-1d` | NORMALISED 1-D monotone map `τ=F̂_B⁻¹∘F̂_A`, `F̂_X=F_X/F_X(θ_max)`, quantile identity `F̂_B(τ(x))=F̂_A(x)` | `coupling.py::MonotoneCoupling1D.map` (`q=interp(xi,x,Fs1)`, `x_m=interp(q,Fm1,master_x)`; `_Fs1=Fs/Fs[-1]`) | `BrenierProposed.quantile_identity` (axiom-free) | yes — OT-MATH-2 (loop 4): display + ALL 8 symbol sites normalised to `F̂`; false "same total mass" parenthetical removed; quantile to 1.1e-16 (`math_verification.md` C1) |
| `eq:ot-brenier-1d` (existence/uniqueness, pushforward) | Brenier theorem; `τ_#μ_A=μ_B` | (measure-theoretic; not in code) | `BrenierProposed.brenier_existence_uniqueness_proposed`, `monotone_map_pushes_forward_proposed` (`sorry`, prose proof) | yes — honestly labelled "proposed, not machine-checked" |
| `eq:ot-gap` | `g_N=(x−τ(x))·n`; conforming branch: Brenier gradient is the PARAMETER-space map `τ=φ'` (monotone, conservative, no circulation), NOT an `R²` gradient of `f`; the physical-gap energy consistency is the matched-normal property `∇_x g_N=n` (Rem `prop:matched-normal`) | `gap_field.py::GapField.sample` (`d=Xs−Xm`, `gN=(d·ns)`); `two_body.py` (`gN=(xs−xmaster)@n_hat`) | `RadialSign.*` (matched-normal sign/magnitude) | yes — OT-MATH-1 (loop 4): dropped the type-inconsistent `g_N=∂_n f` / `∇f(x)=x−τ∈R²` claim; `τ=φ'` is the 1-D param-space gradient, physical-gap conservativeness via `prop:matched-normal` |
| `eq:ot-gap` (true normal, unbiased) | `n_s=(−h',1)/√(1+h'²)`; no 1/cosα bias | `gap_field.py` (`ns=[-hp/sec,1/sec]`); `two_body.py::assemble_two_body_contact` (`ns`) | `RadialSign.*` (separates magnitude bias from sign) | yes — unbiased normal (`math_verification.md` C2) |

## 2. The two admissible regimes (`app:ot:limits`, Remark `rem:ot-limits`)

| Paper (eq/label) | Statement | Code (`file::symbol`) | Lean | Consistent? |
|---|---|---|---|---|
| `rem:ot-limits` (i) conforming | full support ⇒ Brenier = arclength-monotone | `coupling.py::MonotoneCoupling1D` | — | yes |
| `rem:ot-limits` (ii) partial-support | partial contact ⇒ admissible coupling restricts to closest-point `π_B`; `g_N=‖p−c‖−R`; well-posed under convex master (single-valued metric projection off medial axis) + single-valued slave graph `z=h_A(x)` | `coupling.py::ClosestPointCoupling1D` (`_project` orthogonal foot; `map_full`); contact-band mass screen | — | yes — wording "admissible/partial-support" (OT-P3); uniqueness hypotheses + `chizat2018unbalanced`/`santambrogio2015optimal §1.3` cites added loop 3; `cv1_ot_gap.py:15-20` documents `g=|p−c|−R` |
| `eq:ot-unbalanced` | marginal-relaxed (KL-penalised) Kantorovich functional `inf_{π≥0} ∫c dπ + λKL((P_A)_#π‖μ̂_A) + λKL((P_B)_#π‖μ̂_B)`; `λ→∞` recovers hard marginals, finite `λ` lets the plan create/destroy mass; minimiser support → touching band; convex master ⇒ restriction is `π_B`, well-posed off the medial axis | `coupling.py::ClosestPointCoupling1D` (contact-band mass screen = finite-`λ` discrete analogue) | — (measure-theoretic; not in code) | yes — body state-and-cites it (`sec:tmap:ot` l.952-958: `\eqref{eq:ot-unbalanced}` + `chizat2018unbalanced` + medial-axis single-valuedness); full display + hypotheses in `app:ot:limits`/`rem:ot-limits`; `chizat2018unbalanced`/`santambrogio2015optimal §1.3` cited |
| `fig:ot-unbalanced` (loop-6 NEW) | partial-OT schematic: (a) balanced map smears the untouched flanks onto the indenter edge, (b) KL-relaxed plan support collapses onto the touching band → `π_B`, `g_N=‖p−c‖−R`, (c) single-valued off medial axis vs multivalued at a re-entrant feature | `postprocessing/fig_ot_section_loop6.py` (closest-point feet + gap computed analytically; self-check `g_N==‖p−c‖−R` to 0.0; schematic, no benchmark number) | — | yes — placed in `app:ot:limits` after `rem:ot-limits`, referenced once from the remark; illustrates the balanced-map failure mode + medial-axis multivaluedness not in `fig:ot-coupling`/`fig:tm-ot-limits` |
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
| CV-8 deformable Hertz `a` (nx 140→260, half-plane regime, the GATE) | 1.81→6.05→1.54→**1.64%** (finest nx=260; sits at the ~2% edge/CST floor, jitters by 1-node edge quantisation — the nx=180 spike) | `cv8_deformable_ot.py`, `runs/cv8_deformable_ot/metrics.json` | yes — loop-12 half-plane regime (W=2,H=2,δ=0.05,grade=1.5); SUPERSEDES the shallow-block 2.75% |
| CV-8 `p_0` (nx 140→260, half-plane regime) | 2.62→1.67→2.38→**2.26%** (tight realization-independent CST floor) | same as gate row | yes — loop-12; SUPERSEDES the shallow-block 5.82% |
| CV-8 effective modulus check | E*_eff = 4FR/πa² = **0.607** (half-plane E* = 0.549), ratio **1.10** shallow (a/H≈0.43) → **~1.0** deep (a/H≤0.14) | root-cause panel `exp_effE.py`; `metrics.json` (a_fem, F, R) | yes — the finite-size diagnosis; recomputing a_ana with E*_eff reproduces a_fem to <0.1% |
| CV-8 `a` ENSEMBLE (nx 220, seeds 7/11/17/23/31, loop 12) | {1.54, 1.21, 1.54, 1.87, 0.94}% → mean **1.42%**, pop. sd **0.32%**, range [0.94, 1.87]% (realization-DEPENDENT; seed 7 = **median**, not best) | `cv8_deformable_ot.py --mode ensemble` → `runs/cv8_deformable_ot/ensemble.json`; `tab:cv8-ensemble` | yes — half-plane regime; seed 7 labeled median realization |
| CV-8 `p_0` ENSEMBLE (nx 220, same seeds, loop 12) | {2.38, 2.36, 2.21, 2.23, 2.35}% → mean **2.31%**, pop. sd **0.07%**, range [2.21, 2.38]% (realization-INDEPENDENT CST floor) | same as `a` ensemble row | yes — `p_0` floor realization-robust (sd 0.07%) |
| CV-8 patch test / force balance | 1.4e-16 / 1.7e-18 | same | yes |
| CV-9a centre mean stress / anisotropy (D4) / force balance | 0.58% / 0.20–0.22% / 3.71e-15 | `cv9_nbody_array_ot.py`; `tests/test_cv9_nbody_ot.py` | yes |
| tangent FD / force balance | 3.45e-11 / 4.4e-16 | `two_body.py` self-tests | yes |
| OT-vs-conventional head-to-head (CV-1..7) | 1.59→0.50 (3.2×), 11.15→0.03 (370×), 1.62→0.23 (7.0×), 0.11→0.077 (1.4×), 1.99e-2→5.3e-13, frozen→2.6e-16, 1.5e-4→3.5e-14 | `final_report.md` §1 Table 1 | yes — `tab:ot_vs_conv` + `fig:ot_advantage` (P1, FIG-OT-ADV-1) |

## 6. Honest residuals / caveats carried in the paper (unchanged)

- CV-8 ~2% `a`/`p_0` is the discrete contact-edge quantisation (∞ Hertz edge slope over one element) + the constant-strain-triangle stress-recovery floor. Stated in `sec:ex_twobody`, `tab:ot:gates` caption, abstract, and `sec:verif_summary`. **Loop 12 ROOT-CAUSED and re-regimed** (supersedes the loops 7/9 shallow-block reading): the earlier ~3–6% residual was NOT purely contact-edge noise — it was dominated by a FINITE-SIZE compliance offset (shallow clamped blocks a/H≈0.43 are ~10% stiffer than the half-plane the closed form assumes, E*_eff/E*_hp≈1.10), a reference-regime mismatch that biased `a` low / `p_0` high by ~5–6% while force balance stayed machine-zero. Running CV-8 in the half-plane regime (deep/wide blocks, a/H≤0.14, E*_eff/E*_hp≈1.0) removes that offset and exposes the true ~2% floor: `a` jitters 1.81→6.05→1.54→1.64% (edge quantisation; the nx=180 spike is one surface node), `p_0` is a tight realization-independent 2.26% CST floor. The half-plane regime is the honest reading; the shallow-block numbers are a modelling-gap artefact, preserved only in the historical loop log.
- Both two-body solves report **single-realization gate values**, plane-strain CST small-strain, frictionless, rigid outer walls (CV-9a). Stated in the same places + `sec:disc:limits`. **Loop 12 five-seed ensemble at the half-plane headline mesh** (seeds 7/11/17/23/31, nx=220): the `a`-spread (0.32% pop. sd, range [0.94,1.87]%) is the edge quantisation made visible — the reported seed 7 (1.54%) is the *median* of five, neither best nor worst — while `p_0` is realization-robust at 2.31±0.07% (pop. sd 0.07%), confirming the ~2% floor is a genuine CST stress-recovery systematic, not a lucky seed. CV-9a remains single-realization (no ensemble run).
- Brazilian/nine-disc conventional baselines are prescribed-load (Neumann-exact); the OT win there is structural (emergent load, recovered half-width/symmetry), not a like-for-like stress win. Stated in `tab:ot_vs_conv` caption, the nine-disc callout (P5), and `final_report.md` §6.
- Dropped geometric tangent (`∂n`, `∂χ`) is a Newton-path lever only; the converged `a_fem` is tangent-independent (identical at 40 vs 120 iters; `math_verification.md` §R2). Stated in `app:ot:tangent` and `sec:disc:limits`.
- Cattaneo 370× and Hertz 3.2× are reported as same-mesh (Cattaneo on the converged n=3200 mesh; both conventional and OT from `final_report.md` Table 1). The 5–11% Cattaneo figure in `sec:ex_cattaneo` is the coarse-mesh L1-vs-closed-form number and is explicitly distinguished from the converged 0.03%.

## 7. Lean ↔ paper citation map

| Lean file | sorry-free theorems cited | axioms (`#print axioms`) | paper citation site |
|---|---|---|---|
| `PartitionOfUnity.lean` | `p1_host_weights`, `marginal_two_host`, `patch_test_resultant`, `constant_pressure_master_reaction`, `constant_pressure_balance_point` | `propext`, `Quot.sound` | `app:ot:discrete` footnote (eq:ot-marginal + Prop `prop:patch`) |
| `MortarMass.lean` | `posdef`, `scaled_posdef`, `eigen_three`, `eigen_one`, `det_pos` (+`quadform`, `psd`, `symm`) | `posdef`/`scaled_posdef`: `propext`,`Quot.sound`; `eigen_three`/`eigen_one`/`det_pos`: axiom-free | `eq:ot-mortar-mass-body` footnote (Sec. 4 `sec:tmap:ot`) + `app:repro` (loop 4) |
| `TangentPSD.lean` | `rank1_psd`, `rank1_form`, `block_coeff_symm`, `quad_form_nonneg` | `propext`(/`Quot.sound`) | `app:ot:tangent` (Prop `prop:spsd`) footnote |
| `TranslationInvariance.lean` (loop 4, NEW) | `patch_resultant_list`, `tangent_translation_null`, `rigid_translation_gap_invariant` (+`foot_resultant_zero`, `rigid_translation_balanced`) | `propext`, `Quot.sound` | `app:ot:tangent` rigid-translation footnote (combined slave−master PoU difference + full-list patch resultant) |
| `RadialSign.lean` | `active_set_iff`, `radial_gap_sign_agree` | `propext`, `Quot.sound` | `prop:active-set` (Sec. 4) footnote |
| `BrenierProposed.lean` | `quantile_identity` (axiom-free); `*_proposed` carry `sorryAx` | none / `sorryAx` | `app:ot:gap` footnote + `app:repro` |

Reproduce: `cd lean && lake build` (Lean 4 v4.30.0, mathlib-free, ~1 s). The `lean/` source is present in the
worktree (untracked; `.lake/` build artifacts are gitignored via `lean/.gitignore`); committing it is left to
the maintainer per the repo's commit policy.

## Loop-6 delta (substantive-refinement pass; manuscript-side only, no verified number altered, no equation moved)

Structure and equation-completeness are SATURATED (the dedicated top-level OT section `sec:ot`/`sec:tmap`
exists with the full body equation chain; the appendix `app:ot` carries the measure-theoretic proofs and the
`eq:ot-unbalanced` functional). Loop 6 made exactly two manuscript edits plus one consistency-map row:

- **Introduction two-limits dedup + ordering harmonised** (`sec:intro`, l.158-162). The intro previously
  stated the two limits TWICE — once conforming-first then again closest-point-first (flipped) — with a
  redundant restatement. Collapsed to a single "two familiar limits of one coupling" sentence ordered
  **arclength-monotone/conforming FIRST, closest-point/partial-support SECOND**, matching the abstract
  (l.124), the body `eq:ot-twolimits-body` (l.943-944), and the appendix `eq:ot-twolimits` (l.3395-3396).
  All four sites now order conforming→partial identically. No new equation; pure prose harmonisation.
- **Partial-OT schematic `fig:ot-unbalanced`** inserted in `app:ot:limits` after `rem:ot-limits` (Fig 37,
  page 51, beside `eq:ot-unbalanced` = Eq 82). Illustrates the balanced-map smearing failure mode and the
  medial-axis multivaluedness — content NOT in the two existing OT figures (`fig:ot-coupling`,
  `fig:tm-ot-limits`). Honest: schematic, no benchmark quantity plotted, closest-point feet/gap computed
  analytically with a self-check `g_N==‖p−c‖−R` to 0.0 (`postprocessing/fig_ot_section_loop6.py`).
  Referenced once from `rem:ot-limits`; no body cross-reference added (kept the saturated body subsection
  state-and-cite intact).

Audited-and-left-unchanged this loop (verified already-present / measured, per the HARD RULE — Read the live
target before proposing; do not re-propose live text):

- **Partial-OT functional** (item 1): `eq:ot-unbalanced` (KL-relaxed Kantorovich) + the off-medial-axis
  single-valuedness argument are STATED in full in `app:ot:limits`/`rem:ot-limits` (l.3354-3369) and
  state-and-cited in the body `sec:tmap:ot` (l.952-958). No re-derivation needed; the body forward-references
  the appendix equation rather than duplicating it.
- **ML/capacity/spectral** (item 2): MEASURED, not asserted. Chart `n_freq=64` = 66,178 params; plain-MLP
  ablation = 49,922; ambient SDF = 83,073 (25.5% surplus); spectral-figure chart `n_freq=128` = 82,562,
  matched to the SDF's 83,073 — the `fig:spectral-bias` caption (l.2453-2457) states this matched-capacity
  fact so the roll-off separation is charged to the representation, not a capacity edge. `n_freq=128-vs-64`
  disambiguation intact; no implied equal-capacity where there is none.
- **Newton-path lever** (item 3): the dropped `d(n)/d(χ)` frozen-geometry terms are framed as a
  convergence-rate lever (NOT an accuracy lever) at every occurrence (`sec:weak`, `sec:disc:limits`
  "identical `a_fem` at 40 and 120 iterations", `app:ot:tangent`, `app:repro`); none claims they can move the
  converged CV-8 1.64%/2.26% accuracy (half-plane regime, loop 12).
- **Lean** (item e): `lake build` exits 0 in this worktree with exactly the two documented `BrenierProposed`
  sorries (`brenier_existence_uniqueness_proposed`, `monotone_map_pushes_forward_proposed`), both labelled
  "proposed, not machine-checked" and never cited as proven; all other cited theorems sorry-free. No new
  load-bearing algebraic claim was introduced by loop 6, so no Lean extension was required.
