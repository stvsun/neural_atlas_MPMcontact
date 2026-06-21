# CMAME manuscript blueprint — transition-map contact mechanics on a neural atlas

**Target journal:** Computer Methods in Applied Mechanics and Engineering (CMAME).
**Class:** Springer `svjour3` (matching the group's CMAME submission template). Build: `latexmk -pdf main.tex` (or upload `paper/` to Overleaf).
**Selling points:** (1) **the transition map** $\tmap=\varphi_B^{-1}\!\circ\varphi_A$ as the contact detector
[MAJOR]; (2) **Fourier-feature coordinate charts** [enabling]. The **level set (neural SDF) is a comparison
baseline only**, not a contribution.

**Thesis (drives every section).** *The geometric representation, not the constraint-enforcement scheme,
controls contact fidelity on complex geometry. Reading contact from the chart transition map with
band-limited Fourier-feature charts resolves slopes, cusps and fractal detail that a fixed-capacity level
set smooths, so the level set under-predicts contact strength and dilatancy — demonstrated on CV-1..CV-6
closed forms and a real granite joint (CV-7).*

## Authoritative-source rule (prevents number drift)
Use as **authoritative (post-execution, MEASURED):** `docs/cv7_session_results.md`,
`docs/contact_verification_manual.md` §11.8–§11.12, `runs/cv7_decoder/manuscript_table.md`, and the three
chart manuals. Treat `contact_atlas/cv7_formulation_brief.md` and `02_implementation_plan.md` as **stale
pre-execution planning docs** — their "not built / target" labels are superseded by completed runs.

---

## Structure (sections/*.tex; titles/labels already in the stubs)

| § | File | Lead content | Main figures | Key `% code:` |
|---|---|---|---|---|
| 1 | `01_introduction` | motivation; level set as incumbent + its 2 weaknesses; related work; contributions (TM first) | — | — |
| 2 | `02_transition_map` *(MAJOR)* | $\tmap$ def; matched conservative normal; multi-arc; inversion forms; $1/\cos\alpha$ bound | `tm_levelset_vs_chart`, `tm_concept`, `tm_multiarc`, `tm_radial_bias` | `chart_gap.py::evaluate_gap_chart`, `supershape.py::radial_gap`, `geometry.py::invert_decoder` |
| 3 | `03_fourier_charts` | spectral bias; 3 banks; architecture; train+verify-first; `plain` ablation | `fourier_training_architecture`, `fourier_training_curves` | `profile_chart_2d.py::NeuralHeight1D`, `surface_chart_3d.py::NeuralHeight2D`, `radial_chart_2d.py::NeuralRho2D` |
| 4 | `04_formulation` | Signorini/KKT; penalty/AL; conservativity; friction; consistent tangent; chart-FEM/MPM coupling; dispatch | — | `penalty.py`, `augmented_lagrangian.py`, `friction.py`, `chart_vector_fem.py`, `contact_manager.py::body_gap_normal` |
| 5 | `05_verification` | CV-1..CV-4 equivalence; CV-5 discriminator; CV-6 SDF ceiling; V&V table | `supershape_chart_vs_sdf`, `koch_neural_ceiling`, `koch_cost_scaling`, `numerical_cv_summary` | `cv{1,2,3,4}_*_fem.py`, `supershape_cam_drive.py`, koch drivers |
| 6 | `06_validation_cv7` *(tight)* | Inada chart-vs-SDF; P1 two-block; P2 roughness law; atlas-vs-SDF payoff | `rock_joint_capstone`, `rock_joint_3d_twoblock_modes`, `cv7_roughness_law`, `rock_joint_atlas_vs_sdf` | `cv7_real_inada_decoder.py`, `rock_joint_two_block.py`, `cv7_roughness_sweep.py`, `cv7_atlas_vs_sdf_shear.py` |
| 7 | `07_discussion` | when charts beat level sets; preconditions; honest limitations | — (Table T1 capability matrix) | — |
| 8 | `08_conclusions` | outlook (non-star-shaped atlas; 3-D Hertz; large-def; cyclic/dynamic) | — | — |
| A | `A_appendix_closedforms` | Hertz/Cattaneo/Brazilian/nine-disc formulas | — | `postprocessing/contact_fields.py`, `docs/hertz_derivation/` |
| B | `B_appendix_training` | chart hyperparameters; verify-first protocol | — | the three chart modules + `rough_block_decoder.py` |
| C | `C_appendix_reproducibility` | drivers, Inada DOI, `pip install -e .`, figure scripts | — | `benchmarks/contact/cv_numerical/*` |
| S | `S_supplementary` | P3 detector-in-loop; P4 cyclic ledger; P5 MPM | `rock_joint_cyclic_energy`, `cv7_traction_history` | `cv7_transition_map_contact.py`, `rock_joint_decoder_cyclic.py`, `rock_joint_mpm_xcheck.py` |

## Headline numbers (authoritative)
- **CV-5:** chart gap $3.8\times10^{-3}L$ / normal $0.42^\circ$ vs SDF $8\times10^{-3}$ / degraded; multi-arc
  $\ge2$ vs 1; rigid dynamics match analytic chart to **0.04%**.
- **CV-6:** $O(1)$ chart vs $O(9^n)$/$O(4^n)$ SDF storage; per-query $O(\text{depth})\approx21$ nodes;
  fixed-capacity SDF refinement ceiling (normal angle $7.8^\circ\!\to\!\sim45^\circ$, plateaus at $n\!\ge\!2$).
- **CV-7 Inada:** chart **2.3 µm** vs SDF **107 µm** (47×); asperity angle $19.4^\circ\!\to\!12.5^\circ$;
  Patton-law strength **−61%**; Patton anchor **0.00%**.
- **CV-7 P1:** mesh-converged peak $\muapp\approx0.53\,(\pm2\%)$; dilation $\to0.063$ mm; MMS $O(h^2)$.
- **CV-7 P2:** peak $\muapp$ **0.61→1.20** as RMS 0.022→0.066 mm.
- **CV-7 atlas-vs-SDF:** SDF under-predicts dilatancy **−98%** (frictionless) / strength **−35%**.
- **Fourier vs plain (Inada, this paper's run):** chart recon **0.003% of RMS**, angle **19.9°**; plain
  **7.6% / 136 µm**, angle **11.5°**.
- **Supplementary:** P3 active-set 98.9% (chart) vs 95.8% (SDF), gap RMSE 4.2% vs 44.7%; P4 per-cycle ledger
  1.09→1.01; P5 ChartMPM $\muapp\approx0.34$ vs base 0.4.

## Tables
- **T1** capability / which CV exercises the transition map (Discussion).
- **T2** CV-1..CV-6 verification summary (target / measured / chart-vs-SDF) (§5).
- **T3** chart families & Fourier banks (§3 or Appendix B).

## Limitations (must appear, §7)
Radial gap is conservative-large ($g_{\rm rad}=g_\perp/\cos\alpha$, sign-exact → active-set equivalence);
star-shapedness (radial) / single-valuedness (height) preconditions; two-block friction residual 1.5–6%
(moving-master node-to-surface, not mortar); single band-limited realization; small-strain; the contact gap
is a small-slope proxy (fidelity is in slopes/normals, not gap magnitude); the production MPM oracle still
uses the SDF gradient (chart detector is opt-in).

## Build status
The svjour3 manuscript (`main.tex` + `refs.bib` + `sections/*.tex` + `svjour3.cls`/`svglov3.clo`) compiles
to a 43-page PDF locally (TeX Live; `unsrtnat` bib style, `newtx` fonts). Schematic figures are native
TikZ; data figures resolve from `../figures/`. Revised per *The Craft of Research*: patient per-CV
exposition, reduced math density, em-dashes cut, AI/odd diction and the word "honest" removed,
adversarial rigor + English passes applied.

## Style
WaiChing Sun voice: declarative, equation-dense, explicit sign conventions, measured-result tables, limitations; **no AI-language** ("delve/leverage/crucial/it is worth noting/in conclusion/tapestry"). Heavy
subsectioning. `\citep/\citet` against `refs.bib`; figures `\ref{fig:...}`; macros from `main.tex`
(`\tmap,\muapp,\norm,\nbf,\dhat`).
