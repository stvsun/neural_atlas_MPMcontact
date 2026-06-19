# CLAUDE.md

Onboarding guide for Claude Code agents working on this codebase.

## Project Overview

**Neural Atlas for Contact Mechanics** — a meshfree framework for contact problems on complex 3D
geometries using learned coordinate charts and neural SDFs, a chart-based Material Point Method
(MPM), and persistent-homology-based contact detection. The contact stack and a closed-form
analytical verification suite (CV-1..CV-6) are in place; **training the neural coordinate charts is
the next step**, and the analytical benchmarks are written to verify them.

The earlier Nine-Circles brittle-fracture work is **archived** under `archive/` (code, tests, docs,
figures) — preserved for history, not maintained. New work should target contact only.

Key techniques: neural SDF, coordinate charts/atlas, Material Point Method (MPM), penalty &
augmented-Lagrangian contact, regularized Coulomb friction, persistent homology (combined-SDF
contact events), multiplicative Schwarz coupling.

- **Language**: Python with PyTorch
- **Hardware**: CUDA GPU, Apple MPS (M1-M4), or CPU (auto-detected via `resolve_device()`)

## Repository Structure

```
atlas/              # Geometry & chart infrastructure (SHARED)
  sdf/              #   Neural SDF training (Eikonal loss)
  charts/           #   Chart/atlas construction (ChartDecoder training)
  topo/             #   Persistent homology (used for contact-topology events)

solvers/
  mpm/              #   Chart-based MPM: ChartMPMSolver, particles, grid, transfers, constitutive, schwarz_mpm
  contact/          #   gap, penalty, augmented_lagrangian, friction, contact_topology,
                    #     contact_chart_spawn, self_contact, contact_manager, supershape

common/             # ChartDecoder/MaskNet/MLP (models.py), geometry.py (Jacobians, invert_decoder),
                    #   schwarz.py (atlas loading/coloring), utils.py (device resolution, plotting)

benchmarks/
  contact/          #   ball-drop, two-sphere, sliding-block, folding-slab, topology, supershape cam-drive
  mpm_basic/        #   (placeholder for MPM core benchmarks)

postprocessing/     # contact_fields.py (numpy CV references), pyvista_field2d.py,
                    #   plot_liusun_*.py, plot_supershape_demo.py, utils.py

docs/               # contact_theory_manual.md, contact_verification_manual.md,
                    #   hertz_derivation/ (SymPy derivations + index), mpm_velocity_gradient_audit.md
contact_atlas/      # Design docs: brainstorm, implementation plan, variational theory
tests/              # Contact + core-MPM tests (test_neural_chart_verification.py = neural-chart harness)
figures/            # Contact figures (embedded in the verification manual)
archive/            # Legacy Nine-Circles fracture work — DO NOT extend; not maintained
```

## Common Commands

```bash
pip install -e .                                  # editable install

pytest -q                                         # active suite: 122 passed, 7 skipped
pytest tests/test_supershape_contact.py -v        # CV-5 geometry + dynamics
pytest tests/test_neural_chart_verification.py -v # CV-6 neural-SDF ceiling (trains; ~5 min)

python3 postprocessing/contact_fields.py          # numpy CV evaluators self-test
python3 docs/hertz_derivation/hertz_transition_map.py   # CV-1/CV-2 symbolic self-check

python3 benchmarks/contact/supershape_cam_drive.py            # CV-5 rigid-body demo (+ --free-A)
python3 postprocessing/plot_liusun_all.py                     # regenerate CV figures

# --- numerical CV suite (branch numerical-cv-suite) ---
pytest tests/test_chart_fem.py -v                            # chart-FEM port: patch + MMS O(h^2)
python3 -m pytest tests/test_neural_chart_verification.py -k "not cv6" -q   # L0/L1 vs analytical (fast)
python3 atlas/sdf/train_analytical_sdf.py --all              # train sphere/disc/supershape neural SDFs
python3 atlas/charts/train_radial_chart.py                   # train the CV-5 neural radial chart
python3 benchmarks/contact/cv_numerical/cv3_brazilian_fem.py # CV-3 FEM vs closed form (also cv1/cv2/cv4/cv5)
python3 postprocessing/plot_numerical_cv_summary.py          # numerical CV results summary figure
```

## Code Patterns

- **Package imports**: `from common.models import ChartDecoder`, `from solvers.contact.gap import evaluate_gap`, `from solvers.mpm.chart_mpm_solver import ChartMPMSolver`.
- Device/dtype via `resolve_device()` / `resolve_dtype()` in `common.utils` (auto CUDA/MPS/CPU).
- MPM and contact use PyTorch throughout (GPU + autograd). The analytical CV evaluators (`postprocessing/contact_fields.py`, `solvers/contact/supershape.py`) are pure numpy.
- Contact forces are **per-particle physical-space forces** passed to `particle_to_grid(..., contact_force=...)`; penalty, AL, and friction all share this API (`solvers/contact/{penalty,augmented_lagrangian,friction}.py`).
- Constitutive models follow `compute_stress(F, state) -> (sigma, state)`.
- Benchmarks write JSON to `runs/<name>/` (gitignored); figures go to `figures/`.

## Architecture Notes

### MPM Solver (`solvers/mpm/`)
- `ChartMPMSolver`: explicit time stepper; particles store position (xi), velocity, deformation gradient F, stress; linear B-spline P2G/G2P; Neo-Hookean and J2 constitutive models.
- `SchwarzMPMSolver`: multi-chart orchestration + `configure_contact()` + `observe_contact_topology()`.
- Contact force scatters through the **same P2G channel as gravity, with no Jacobian pull-back** (everything is physical space). See `docs/contact_theory_manual.md §1.2`.

### Contact Framework (`solvers/contact/`)
- `gap.py::evaluate_gap` — SDF gap + autograd normal (forces `torch.enable_grad()` internally).
- `penalty.py` / `augmented_lagrangian.py` — normal force (Uzawa multiplier persists across steps; needs stable input shape).
- `friction.py` — regularized Coulomb, stateless, composes with any normal force.
- `contact_topology.py` — combined-SDF $H_0$ persistent-homology events via `atlas/topo`.
- `supershape.py` — Gielis superformula boundary chart + inverse radial gap (CV-5).
- `chart_gap.py` — 3-D level-set-free radial-chart detector (`RadialChart`, `evaluate_gap_chart`).
- `radial_chart_2d.py` — 2-D NEURAL radial chart (`NeuralRho2D`, Fourier-feature $\rho_\theta$) + gap/normal; the transition-map detector that gives the accurate CV-5 path (numerical-cv-suite branch).

### Atlas-FEM (`solvers/fem/`, branch `numerical-cv-suite`)
- Chart-based elastostatic FEM ported from `archive/`: `chart_vector_fem.py` (3-D P1/P2 tet, ChartDecoder-Jacobian pushforward, Newton; SVD-stabilized via `common.geometry.stabilized_jacobian_ops`), `tri2d.py` (2-D plane-stress CST, sparse), `linear_elastic.py`, `schwarz_vector_fem.py`. Static penalty contact lives in the CV-1/CV-2 drivers (`benchmarks/contact/cv_numerical/`).

### Topology Pipeline (`atlas/topo/`)
- `filtration.py` / `persistence.py` — sublevel-set filtration + persistent homology (GUDHI).
- `chart_spawn.py` — converts topology events into chart-pair spawn requests (contact reuses this).

## V&V Status

Active suite: **122 passed, 7 skipped** on the original contact suite; the **numerical-CV work is on
branch `numerical-cv-suite`** (not yet merged to main) and adds the chart-FEM + neural-chart pipeline.

**Analytical contact verification (CV-1..CV-6)** — closed-form acceptance targets:
`docs/contact_verification_manual.md` (Hertz, Cattaneo–Mindlin, Brazilian disc, nine-disc, superformula,
Koch; **§11 = the neural-chart protocol**, **§11.8 = measured numerical status + capability matrix**).
Closed forms: `docs/hertz_derivation/` (SymPy) + `postprocessing/contact_fields.py` (numpy).

**Numerical CV suite (branch `numerical-cv-suite`)** — trains neural charts on the CV shapes and solves
numerically vs the closed forms (drivers in `benchmarks/contact/cv_numerical/`; summary figure
`figures/numerical_cv_summary_pub.png`; full status + capability matrix in manual §11.8). VERIFIED:
chart-FEM port (`solvers/fem/`, patch test + MMS $O(h^2)$); **CV-1** Hertz line contact (FEM + penalty
contact, neural disc SDF indenter, $a(F)$–$E^*$ ~1.6%); **CV-2** Cattaneo stick/slip ($c/a$ law ~5–11%,
deep half-plane); **CV-3** Brazilian (centre 1.62%); **CV-4** nine-disc unit cell (equibiaxial centre
0.15%); neural-SDF L0 (sphere 1.6e-3, disc 3.5e-3); **CV-5 the chart-over-SDF advantage MEASURED** —
neural SDF degrades on cusps (8e-3) while the neural **radial chart** (`solvers/contact/radial_chart_2d.py`,
Fourier-feature $\rho_\theta$) reaches 3.8e-3 / 0.42°, and drives the cam-drive dynamics to a 0.04% match
vs the analytical chart; **CV-6** refinement ceiling measured. NOT BUILT: full N-body disc-array contact
(only the unit cell), ChartDecoder trained on CV shapes (the FEM is verified on a ChartDecoder), 3-D
Hertz, MPM cross-checks. Harness `tests/test_neural_chart_verification.py` + `tests/test_chart_fem.py`
pass against trained `.pt` charts (gitignored, so a fresh checkout skips until retrained); the slow
contact sweeps (CV-1/CV-2/CV-5-dynamics) run as benchmark drivers, not routine tests.

## Gotchas

- **MPS (Apple Silicon)** requires float32. Use `resolve_dtype("auto", device)`.
- **Local pytest is old (5.x)** — it ignores `[tool.pytest.ini_options]` in `pyproject.toml`; collection exclusion of `archive/` is handled by the root `conftest.py`.
- **Large binary files** — do not read `.pt` checkpoints directly; `runs/` JSON is gitignored.
- **GUDHI overhead** — use a 16³ grid for contact-topology monitoring and check every 50+ steps, not every step.
- **`archive/` is frozen** — legacy fracture code with imports to other archived modules; do not extend it or wire active code to it.
- **CV-5 radial gap ≠ Euclidean distance** — it's a single-body inverse radial chart (biased ~1/cosα on flanks); compare a neural *SDF* against the Euclidean reference, not the radial gap (see verification manual §11.2).
