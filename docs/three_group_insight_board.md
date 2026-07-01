# Three-Group Campaign — Shared Insight Board

Cross-group communication log for the Figures / Mathematics / Writing groups running a
Reasoning + Action + Reflexion routine. Updated after every round. Groups read this before acting.

## Ground rules (all groups)

- **Protected numbers (never drift):** CV-8 a 1.64% / p0 2.26% / ens 1.42±0.32% / 2.31±0.07% / a/W 0.137 /
  patch 1.4e-16 / FD 3.45e-11 / force bal 1.7e-18; CV-9a 0.58% / 0.20–0.22% / 3.71e-15; OT-vs-conv 3.2×/370×/6.3×;
  CV-7 −61% / −98% / recon 2.3µm/107µm / active 98.9%/95.8%; Fourier recon 2.2%/48%/15.7%.
- **Figures — SAFE to regenerate** (deterministic, read committed `runs/*.json` or pure schematic/TikZ):
  `plot_two_body_ot.py` (cv8_hertz, cv9_nbody, cv8_patch), `plot_numerical_cv_summary.py`,
  `plot_transition_map_composite.py`, `plot_fourier_mechanism.py`, `plot_liusun_*.py`,
  `plot_transition_map_manual.py`, and the 6 inline TikZ figures. Use `/usr/bin/python3`.
- **Figures — OFF-LIMITS** (retrain models / need cluster data → number drift): `plot_rock_joint_capstone.py`,
  `plot_fourier_training.py`, `plot_rock_joint_3d*.py`, `plot_levelset_vs_atlas.py`, `plot_supershape_demo.py`
  (verify before touching; if a generator trains or samples randomly, do NOT regenerate).
- **File-conflict rule:** figures group edits `postprocessing/*.py` + `figures/*.png|pdf` ONLY (never main.tex).
  All `.tex` edits (math fixes + writing + captions) go through the single main.tex actor, serialized.
- Every round: compile must stay clean (0 undefined refs), protected numbers byte-identical.

## Baseline (start of this campaign)

60 pp, 33 figures, 11 tables, 2 algorithms, em-dashes 105, 0 AI signposting tells. Prior 7-loop readiness
campaign already polished Figs 2 (transition_map_composite), 3 (fourier_mechanism), 20 (cv8_hertz).

## Round log

- **round 0** — board created. Next: diagnose figures/math/writing in parallel.
- **round 1** ✔ — 9 agents (3 diagnose + synth + 2 act + 3 reflect). Cross-group insight board built.
  - **FIGURES**: fixed `fig_ot_advantage_loop1` — panel (a) tick-label collision AND its stale CV-8 fallback
    data (was nx 96-192 / 5.14→2.75%, now nx 140-260 / 1.64% half-plane); added dpi=300 + vector PDF output to
    `plot_two_body_ot.py` and de-emphasised the CV-9 mesh overlay → regenerated cv8_patch/cv8_hertz/cv9 (numbers
    intact, PDFs produced; metrics.json repopulated via `--mode all`, deterministic, no drift). JET→perceptual
    swap for the two Liu–Sun pyvista field maps SKIPPED (no vtk/pyvista locally) → carried to human/next.
  - **MATHEMATICS**: core math re-verified symbolically + vs runs JSON. One precision fix applied+grammar-cleaned
    (main.tex:872: the quadratic cost yields a *closest-point (metric) projection only in the partial-support
    limit*, not the arclength/Brenier map — corrected an overclaim). Two DEFERRED (see below).
  - **WRITING**: 10 consistency fixes (CST spell-out moved to first use; SDF/BVP glossed at first use;
    realization→realisation UK spelling). Reflexion notes ~15 more US -ize words remain (next round).
  - Compile clean 0 undefined, 60 pp. All protected numbers byte-identical.

## DEFERRED TO HUMAN (need a decision — do not auto-change)

1. **Brazilian OT-vs-conv ratio "6.3×" is arithmetically wrong.** The row (main.tex:2343) shows conv 1.62%,
   OT 0.23%, ratio **6.3×**, but 1.62/0.23 = **7.04×**. Same triple hardcoded in `fig_ot_advantage_loop1.py`.
   6.3× is a "protected" number, so I did NOT change it. FIX OPTIONS: (a) correct the ratio to ~7.0×, or
   (b) the true underlying OT relerr is ~0.257% (1.62/0.257=6.3) and 0.23% is the rounded/typo'd value — check
   the CV-3 driver output and correct whichever is wrong. Needs the source number confirmed.
2. **CV-8 slave/master A/B convention flips.** OT theory: A=slave, B=master. The CV-8 figure labels
   Ω_A='master'/Ω_B='slave' (and the Koch figure follows the figure's assignment). Pick one direction:
   relabel the figure to A=slave, or add a one-line CV-8 caption note that A plays the master role there.
3. **JET→perceptual colormap** on liusun_fig13 (Hertz σ_yy) + fig16 (nine-disc) — needs vtk/pyvista (absent
   locally); real CMAME red-flag. Regenerate on a machine with pyvista, or leave.
