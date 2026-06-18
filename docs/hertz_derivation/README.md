# Analytical Atlas Contact Benchmarks — artifact index

Closed-form (SymPy-derived, adversarially verified) contact-mechanics solutions, recast in
the neural-atlas / coordinate-chart framework. These are the **verification references** for a
future neural-coordinate-chart contact implementation: when a trained neural SDF / `ChartDecoder`
replaces an analytical chart, its output is checked against the closed forms collected here.

**Read [`../contact_verification_manual.md`](../contact_verification_manual.md) first** — it is the
detailed reference (benchmarks CV-1..CV-5, pass criteria, the **neural-chart verification protocol**
in §10, the master table, and the figure index). This file is just the artifact map.

## Symbolic derivations (this directory)

| Script | Derives | Benchmark | Run |
|---|---|---|---|
| `hertz_transition_map.py` | Hertz 3D (MDR) + 2D line contact; on-axis subsurface stress & yield; Cattaneo–Mindlin partial slip | CV-1, CV-2 | `python3 docs/hertz_derivation/hertz_transition_map.py` |
| `brazilian_disc_atlas.py` | Brazilian disc via Flamant-chart superposition; stress field, $\sigma_t$, displacement, diametral compliance, Hondros | CV-3 | `python3 docs/hertz_derivation/brazilian_disc_atlas.py` |
| `nine_disc_atlas.py` | 9-disc packing: symmetry reduction → 4-load unit cell; equibiaxial center; force–compression law | CV-4 | `python3 docs/hertz_derivation/nine_disc_atlas.py` |

Each script is SymPy/mpmath only and **prints** its verified identities and closed forms to stdout
(self-checking). They are the provenance for the formulas; they are not imported at runtime.

## Numpy evaluators (single source of truth for runtime/verification)

| Module | Provides |
|---|---|
| `postprocessing/contact_fields.py` | Vectorized numpy evaluators mirroring the symbolic forms (Hertz params/pressure/subsurface, line contact, Cattaneo–Mindlin, Brazilian field, nine-disc unit cell, principal stresses). Self-test: `python3 postprocessing/contact_fields.py`. **The neural-chart verification harness compares against these.** |
| `solvers/contact/supershape.py` | Gielis superformula boundary chart (CV-5): boundary/tangent/normal, inverse radial gap (matched `(g, ∇g)`), bounded closest-point refine, area/inertia. |

## Dynamics + figures

| Artifact | Benchmark | Run |
|---|---|---|
| `benchmarks/contact/supershape_cam_drive.py` | CV-5 rigid-body cam-drive demo (+ `--free-A` momentum control) | `python3 benchmarks/contact/supershape_cam_drive.py` |
| `postprocessing/plot_liusun_all.py` | Liu & Sun figures 12–16, 21–23 (paper style) | `python3 postprocessing/plot_liusun_all.py` |
| `postprocessing/plot_supershape_demo.py` | CV-5 GIF + summary + chart-vs-SDF comparison | `python3 postprocessing/plot_supershape_demo.py` |

## Tests

| Test | Covers |
|---|---|
| `tests/test_supershape_contact.py` | CV-5 geometry, dynamics, momentum, multi-arc vs single-CPP, bias/refine, cusp regression (9 tests) |
| `tests/test_neural_chart_verification.py` | **Skeleton** harness for verifying neural charts against CV-1..CV-6 (skipped until neural charts are trained; see manual §11) |

## How it all fits (verification workflow)

```
analytical shape/chart  --(train)-->  neural SDF / ChartDecoder
        |                                     |
   closed form (this dir)              neural chart output
        |                                     |
   contact_fields.py  <----- compare ------>  tests/test_neural_chart_verification.py
        |                                     |
   CV-1..CV-6 pass criteria  ===  acceptance for the neural chart  (manual §9, §11)
```
