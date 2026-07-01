# CMAME Submission-Readiness Mind-Map

**North star:** `paper/main.tex` ready to submit to *Computer Methods in Applied Mechanics and
Engineering* — correct, concise, well-organised, referee-proof, and free of AI-language tells.

Living planning doc for the multi-agent campaign. Updated every loop.

---

## Current state (baseline, loop 0)

- 3915 lines, **61 pages**, 35 figures, 11 tables, 2 algorithms (both now in the main text), 87 numbered eqs.
- **211 `---` em-dashes** (AI-tell; CMAME prefers restraint), "robust" ×10.
- Structure (body sections): 1 Intro · 2 Neural coordinate atlas · 3 Governing eqs/variational ·
  4 Contact as OT (transition-map measure coupling) · 5 OT measure coupling + two-body tangent (ex-App C) ·
  6 FE formulation · 7 Numerical examples (Hertz, Cattaneo, nine-disc, superformula, Koch, two-body, perf
  summary, CV-7 rock joint) · 8 Discussion/limits/conclusions.
  Appendices: closed forms · chart training · data/code · supplementary · linearisation.

## Workstreams

| # | Stream | Owner lens | Status |
|---|--------|-----------|--------|
| A | Reorganise (section order, §4/§5 merge?, examples→supp) | editor + contact | open |
| B | Algorithms in main text | — | DONE (Alg 1 §4, Alg 2 §5.5) |
| C | Concision (cut length toward ~50 pp; kill repetition) | editor + English | open |
| D | Anti-AI-language (em-dash ↓, "robust"/tricolon/signposting ↓) | English | open |
| E | Figures (35 → fewer; quality/clarity) | sci-viz + editor | open |
| F | Math/notation consistency + proof rigor | mathematician | open |
| G | V&V honesty (claims=evidence, reproducibility) | V&V | open |
| H | References + CMAME submission checklist | editor | open |

## Protected numbers (MUST NOT DRIFT — verify each loop by grep)

- CV-8: a **1.64%**, p0 **2.26%**; ens a **1.42±0.32%**, p0 **2.31±0.07%**; a/W 0.137; patch 1.4e-16; FD 3.45e-11; force bal 1.7e-18.
- CV-9a: mean **0.58%**, anisotropy 0.20–0.22%, force bal 3.71e-15.
- OT-vs-conv: 1.59→0.50 (3.2×), 11.15→0.03 (370×), 1.62→0.23 (6.3×).
- CV-7: strength −**61%** (Inada), dilatancy −**98%** (synthetic), recon 2.3µm vs 107µm; active set 98.9% vs 95.8%.
- Fourier recon 2.2%/48%/15.7%.

## Anti-AI-language checklist (hunt & destroy)

- em-dash `---` overuse → commas/parens/periods where cleaner (target: cut ≥40%).
- tricolons ("X, Y, and Z" chains as rhetoric), "not only … but also".
- signposting: "Importantly", "Notably", "Crucially", "It is worth noting", "We emphasise".
- inflated adjectives: "robust", "seamless", "powerful", "rich", "novel", "key".
- hedge+boast pairs; sentence-initial "This" with vague referent; "leverage", "showcase", "underscore".
- over-parallel triadic sentence rhythm; restatement of the same result 3+ times.

## Panel findings (loop 1) — the referee-proofing program

**Correctness / honesty handles a referee could exploit (fix these):**
1. **mortar-vs-penalty over-claim** (HIGH): title/abstract sell "mortar" (dual-multiplier, variationally
   consistent), but the unilateral constraint is enforced by penalty/AL node-to-surface. Add a scope sentence
   (§5.5/App) stating this + the penalty stiffness value/rule + error-insensitivity + inf-sup/LBB positioning
   (cite Wohlmuth 2011) + symmetric-tangent vs Laursen/Wriggers non-symmetry. → loop 3.
2. **CV-8 provenance mismatch** (MED): 1.64% = nx=260 finest single seed; 1.42±0.32% = nx=220 ensemble (median
   1.54%). Decouple point-value from band at all sites (abstract ✔done, contributions, conclusion, tab). → loop 4.
3. **ML spectral-bias overreach** (MED): "defeats the spectral bias" (×4) → "shifts the cutoff above the asperity
   band"; fix k^-(d+1) NTK exponent citation (Bietti-Mairal 2019 / Geifman 2020, ReLU-on-sphere; charts are
   tanh/softplus) or soften; "cannot resolve"→"does not resolve at a practical budget". → loops 1-2.
4. **Notation collisions** (MED): Brenier potential φ collides with chart φ_i → rename ψ; unbalanced-OT KL weight
   λ collides with AL multiplier/NTK eigenvalue → rename τ. Promote 3 Lean-backed Remarks (active-set, matched-
   normal, multiarc) to Lemma. → loop 2.
5. **Rock-joint provenance** (LOW): tag -61% (real Inada 1-D) vs -98% (synthetic 3-D block); add evidence pointer
   for the real-Inada 3-D decoder sub-result (3.2%/15.8%, 58±35%). → loops 4-5.

**Structural (length drivers):**
- **Doubled OT derivation** (§5.2 main + §6/app:ot): measure/Monge/Kantorovich/Brenier derived twice, ~330 lines.
  Keep §5.2 compact, demote §6 to a real appendix (non-duplicated: unbalanced-OT, SPSD proof, patch proof, gates). ~4-5pp. → loop 3.
- **One-canonical-home rule** for every headline number (patch 1.4e-16 ×15, -61/-98 ×10/9, 67× ×8, FD ×8): full
  number once at its table/section, pointers elsewhere; delete rem:ot-verdict / "two honest boundaries" dup. → loops 4/6.
- **Rock-joint Phases 1-5 → online Supplementary** (~3-4 figs).

**Figure plan (35 → ~26-28):** merge cvX-bvp schematics (cv1/cv2/cv4/cv6) into results captions; Koch 4→2
(merge cv6-bvp+cv6-geom, cv6-cost+cv6-ceiling); Phases 1-5 → supp; cut/fold fig:ot-coupling (dup of tm-concept/
ot-unbalanced); remove dead `\label{fig:cv4}`. Keep novel-geometry schematics + all results figs. → loop 5.

## Multi-loop plan (panel-chair)

- **loop 1** ✔ Sec 1-3 anti-AI-language + concision (abstract done; intro+§2-3 in progress).
- **loop 2** — §4 (tmap): notation renames ψ/τ; c-cyclical-monotonicity on scalar param; promote Remarks→Lemma;
  de-repeat "one coupling / two limits" to one statement + pointers.
- **loop 3** — §5-6: collapse doubled OT (demote §6 to appendix); mortar-vs-penalty scope + penalty stiffness +
  inf-sup clause; "consistent tangent"→"frozen-geometry (algorithmic) tangent".
- **loop 4** — §7 results/two-body: CV-8 provenance at every site; one-canonical-home for CV-8/patch/FD; penalty
  1/cosα caveat; trim 55 em-dashes in two-body block; CV-5 SDF param count; tag rock-joint numbers.
- **loop 5** — §8-9 + figures: Koch 4→2, BVP schematics merged, Phases 1-5→supp; real-Inada 3-D pointer; Welch-PSD
  ">99.9% (0.1 dB)" precision.
- **loop 6** — §10 discussion/concl: delete rem:ot-verdict dup + 3rd contributions re-listing; vary triadic openers;
  cut "resolves what the level set smooths" motif to ≤2; trim genuine/decisive/honest/unlock; em-dash <100 whole-paper.
- **loop 7** — whole-paper QA: protected-number audit, \ref/\cite/\label integrity, figure/table reconciliation,
  read-through for refs broken by §6 demotion + figure merges. Final compile + submission checklist.

## Decisions log

- (loop 0) Keep both algorithms in body. Do NOT regenerate figures that retrain models (number drift risk):
  rock_joint_capstone, p1_refinement, PyVista renders (cluster-bound).
- (loop 1) Notation renames for loop 2 — VERIFIED collisions: Brenier potential is bare `\varphi` (L910,1156,
  1179,1182) but `\map{}`=`\varphi_i`=chart AND `\psi` is already the chart transition map (L501 `\psi_{ij}`) and
  slave-param image (L1182 `\psi_B`). So Brenier potential must NOT go to ψ (panel's suggestion collides) →
  use **`\Phi`** (capital; standard Kantorovich/Brenier potential, free). KL weight: bare `\lambda` (L1212-1239)
  collides with the bare AL multiplier `\lambda` (L1054-1055) → rename KL weight **`→ \tau`** (AL multiplier stays
  `\lambda`; NTK `\lambda(k)` and SDF `\lambda_{s,n,e}` are subscripted/local, leave). Do the Φ/τ renames only in §4-6.

## Prep intelligence (loop-0 recon, read-only)

**Figure inventory (35 figs, ref counts):** most refs=1 (typical for V&V). Levers:
- 7 `cvX-bvp` schematics (cv1/cv2/cv4/cv5/cv6/cv7/cv8) → consolidate/shrink to insets. Top figure cut.
- `fig:cv4` line 1894 has THREE stacked labels (`fig:cv4`/`-bvp`/`-ninedisc`); bare `fig:cv4` is DEAD → remove.
- Koch: `fig:koch_ceiling` + `fig:koch_cost` (2 figs) → candidate merge.
- Two-body: `fig:cv8-bvp` + `fig:cv8-patch` + `fig:cv8-hertz` + `fig:cv9-array` — check if bvp can inset.

**Repetition map (restatement counts — concision target: keep each at ~4 anchors):**
- patch `1.4e-16`: **15×** (worst) · `-61%`: 10× · `-98%`: 9× · `67×`: 8× · `370×`: 8× · spectral-bias/low-pass: 23×.
- Rule: each headline number lives at abstract + intro first-use + its table + conclusion. Cut the rest to `\ref`.

## Per-loop log

- **loop 0** — baseline + recon captured; mind-map created. Figure inventory + repetition map above.
- **loop 1** ✔ DONE — 6-lens panel diagnose → directive → editor anti-AI-language + concision pass on abstract
  + Sec 1-3. 22 edits. Em-dashes whole-file 211→185 (in-scope prose 28→6). "defeats spectral bias"→"shifts
  cutoff"; "genuine/decisive/unlock" removed in scope; thesis de-repeated; contributions CV-8 numbers→pointers;
  rock-joint bullet→§cv7 pointer. Compile clean (0 undef, 61 pp); all protected numbers + \ref/\cite/\label
  byte-identical. Sec 1-3 words 5075→4948. Verified independently. (Note: builder editor agent died mid-run once;
  finished by a retry agent + my abstract fixes — always independently verify builder claims.)
