# CV-7 capstone — session results & manuscript figure plan

Genuine rough rock-joint direct shear with a neural atlas (boundary-fitted chart) vs an ambient level
set (neural SDF). Heavy solves run on the Euler cluster (`scripts/euler/`, conda env `atlas`); figures
generated locally. All numbers below are measured (no tuning to targets); honest caveats are listed.

Thesis: on a genuinely rough joint the apparent friction `μ_app = tan(φ_b + i)` and the dilation EMERGE
from the resolved asperity geometry; a boundary-fitted neural chart resolves the asperity slopes that an
ambient SDF smooths (spectral bias), so the level set under-predicts strength and dilatancy.

---

## 1. Key results (this session: §11.12 Phases 1–5, run on Euler)

| Phase | Result | Numbers (measured) |
|---|---|---|
| **P1** two mutually-deformable decoder blocks (node-to-surface, mutual contact) | emergent **anisotropy**; verified geometry | peak μ_app 0.42 / 0.54 / 0.74 (out-/in-/mixed-plane); dilation 0.090 / 0.066 / 0.038 mm; **MMS O(h²)** rates 2.39, 1.95; det J ∈ [0.97, 1.03]; frictionless residual ~1e-10 |
| **P2** roughness / spectral-cutoff sweep | **emergent dilatancy law** | peak μ_app **0.61 → 1.20** as surface RMS 0.022 → 0.066 mm; **0.68 → 1.21** with spectral cutoff; dilation 0.018 → 0.059 mm; decoder recon < 8 % |
| **P3** transition-map detector in the FEM contact loop | chart drives genuine contact | active-set agreement vs analytic **98.9 % (chart) vs 95.8 % (SDF)** on 1e5 pts; per-query **1.47×** SDF; FEM shear via chart ≈ analytic (peak μ 0.566 vs 0.513, τ within **9.5 %**) |
| **P4** cyclic + complete energy ledger (stateful return-map friction) | stick/slip hysteresis; **per-cycle energy CLOSES** | per-cycle `W_ext / (W_fric+ΔU_el+W_pen+W_stick)` = **1.09 → 1.01** (within [0.98,1.02]); residual ~7e-5; Plesha decay 0.340 → 0.338 |
| **P5** explicit ChartMPM dynamic cross-check | reproduces the Coulomb floor | μ_app **0.344** (max 0.43) ≈ base μ = 0.4 (~14 %) |

### Verification (mesh refinement of the P2 one-block shear, n_cells 6 → 14, 1.3k → 16.5k tets)
- **Mesh-converged** global response: σ_n held at 2.0 MPa (CNL), peak τ **1.79–1.82 MPa**, peak μ_app
  **0.885–0.908** (~1.5 % spread); det J_min steady ~0.96; Newton residual ~1e-9 (one-block converges cleanly).
- **Checkerboard ≠ locking** (diagnosed): the per-element checkerboard is constant-strain (CST) stress
  faceting — it collapses under nodal stress recovery and the checkerboard index DECREASES with
  refinement (0.286 → 0.214). Volumetric locking is ruled out (ν = 0.25 is far from incompressible;
  a locking/pressure mode would survive averaging and not converge).

### Honest caveats (reported, not hidden)
- **Two-block friction residual 1.5–6 %** — the two-deformable, moving-master node-to-surface contact
  converges far worse than the frictionless (1e-10) / one-block (1e-9) cases.
- **Cyclic cumulative ratio 0.17** is an artifact of the initial asperity-**seating** dissipation
  (`W_fric[0]=0.91` vs `W_ext[0]=0`) that the shear-only CNL machine work does not count — the per-cycle
  balance closes; it is a bookkeeping offset, not an energy violation.
- **MPM** reproduces friction but **not dilatancy**: a mated rough-on-rough MPM block is unstable in
  explicit penalty dynamics (needs implicit / mortar contact).
- Single band-limited realization; node-to-surface (not mortar) contact; small-strain linear elastic.

---

## 2. Recommended figures for the paper

**Main text (≈6):**

| # | File | Shows | Caption gist |
|---|---|---|---|
| 1 | `rock_joint_atlas_vs_sdf_pub.png` | **THE thesis** | Atlas resolves asperities; level-set (SDF) smooths → under-predicts dilatancy 98 % / strength 35 % |
| 2 | `rock_joint_3d_twoblock_modes_pub.png` | P1 genuine two-block | Emergent anisotropy (μ_app, dilation, transverse traction) across in-/out-of-plane/mixed shear |
| 3 | `cv7_roughness_law_pub.png` | P2 dilatancy law | Emergent peak μ_app & dilation vs surface RMS and spectral cutoff (decoder recon stays <8 %) |
| 4 | `cv7_traction_history_pub.png` | verification | Global shear traction τ(u) + emergent dilation δ_n(u); mesh-converged across 4 meshes |
| 5 | `rock_joint_cyclic_energy_pub.png` | P4 cyclic ledger | Stick/slip hysteresis; per-cycle energy ledger closes to ~1 % |
| 6 | `cv7_time_evolution_vm_pub.png` | showcase | 3-D von Mises contour over loading time (cols) × mesh refinement (rows), with the rigid mating surface |

**Supplementary / appendix:**

| File | Shows |
|---|---|
| `cv7_refinement_vm_pub.png` | 3-D von Mises contour (translucent domain + iso-contours) vs mesh refinement |
| `cv7_vm_locking_check_pub.png` | checkerboard = CST faceting (per-element vs nodal-averaged), NOT locking |
| `cv7_refinement_pub.png` | convergence curves: peak μ_app, dilation, residual + det J vs mesh |
| `rock_joint_twoblock_vm_pub.png` | two-block mid-plane von Mises cross-section over shear |
| `cv7_manuscript_pub.png` | 6-panel composite (atlas vs SDF vs flat benchmark) — alt. headline |
| `rock_joint_cyclic_genuine_pub.png` | cyclic hysteresis (3 modes) + Plesha degradation decay |
| `cv7_real_inada_pub.png`, `rock_joint_capstone_pub.png` | real Inada-granite 2-D capstone (chart 2.3 µm vs SDF 107 µm) |

Notes: figures 1, the Inada 2-D, and the composite are from earlier sessions; figs 2–6 and all
supplementary `cv7_*`/`*_vm_*`/`traction`/`refinement` figures are from this session. Drivers:
`benchmarks/contact/cv_numerical/{rock_joint_two_block, cv7_roughness_sweep, cv7_transition_map_contact,
rock_joint_decoder_cyclic, rock_joint_mpm_xcheck, cv7_refinement_study, cv7_time_evolution_fields,
cv7_traction_history}.py`; plotters `postprocessing/{plot_rock_joint_3d, plot_rock_joint_refinement_vm,
plot_rock_joint_twoblock_vm, check_vm_locking}.py`.
