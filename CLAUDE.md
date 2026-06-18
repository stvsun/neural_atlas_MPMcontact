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

pytest -q                                         # active suite: 120 passed, 7 skipped
pytest tests/test_supershape_contact.py -v        # CV-5 geometry + dynamics

python3 postprocessing/contact_fields.py          # numpy CV evaluators self-test
python3 docs/hertz_derivation/hertz_transition_map.py   # CV-1/CV-2 symbolic self-check

python3 benchmarks/contact/supershape_cam_drive.py            # CV-5 rigid-body demo (+ --free-A)
python3 postprocessing/plot_liusun_all.py                     # regenerate CV figures
python3 atlas/sdf/train_sdf.py                               # train a neural SDF (when bringing up charts)
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

### Topology Pipeline (`atlas/topo/`)
- `filtration.py` / `persistence.py` — sublevel-set filtration + persistent homology (GUDHI).
- `chart_spawn.py` — converts topology events into chart-pair spawn requests (contact reuses this).

## V&V Status

Active suite: **120 passed, 7 skipped** (`pytest`; the 7 skips are the neural-chart harness, awaiting trained charts).

**Analytical contact verification (CV-1..CV-6)** — closed-form acceptance targets for the neural
charts: `docs/contact_verification_manual.md` (Hertz, Cattaneo–Mindlin, Brazilian disc, nine-disc,
nonconvex superformula; **§11 = the neural-chart verification protocol**). Closed forms:
`docs/hertz_derivation/` (SymPy) + `postprocessing/contact_fields.py` (numpy). Harness skeleton:
`tests/test_neural_chart_verification.py` (skipped until neural charts exist).

## Gotchas

- **MPS (Apple Silicon)** requires float32. Use `resolve_dtype("auto", device)`.
- **Local pytest is old (5.x)** — it ignores `[tool.pytest.ini_options]` in `pyproject.toml`; collection exclusion of `archive/` is handled by the root `conftest.py`.
- **Large binary files** — do not read `.pt` checkpoints directly; `runs/` JSON is gitignored.
- **GUDHI overhead** — use a 16³ grid for contact-topology monitoring and check every 50+ steps, not every step.
- **`archive/` is frozen** — legacy fracture code with imports to other archived modules; do not extend it or wire active code to it.
- **CV-5 radial gap ≠ Euclidean distance** — it's a single-body inverse radial chart (biased ~1/cosα on flanks); compare a neural *SDF* against the Euclidean reference, not the radial gap (see verification manual §11.2).
