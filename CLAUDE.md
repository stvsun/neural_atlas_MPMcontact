# CLAUDE.md

Onboarding guide for Claude Code agents working on this codebase.

## Project Overview

**Topology-Aware Neural Atlas for FEM/MPM on Complex 3D Geometries** — a meshfree framework using learned SDFs and coordinate charts to solve boundary value problems via FEM or MPM on complex 3D geometries, with topology-aware chart construction via persistent homology.

Key techniques: Neural SDF, coordinate charts/atlas, multiplicative Schwarz domain decomposition, P1 tetrahedral FEM, Material Point Method (MPM), persistent homology, Lusternik-Schnirelmann category.

- **Language**: Python with PyTorch
- **Hardware**: CUDA GPU, Apple MPS (M1-M4), or CPU (auto-detected via `resolve_device()`)

## Repository Structure

```
atlas/                          # Geometry & chart infrastructure
  sdf/                         #   Neural SDF training (Eikonal loss)
  charts/                      #   Chart/atlas construction
  topo/                        #   Topology-aware construction (persistent homology)

solvers/                        # PDE solvers on mapped charts
  fem/                         #   FEM: ChartFEMSolver (scalar), ChartVectorFEMSolver (vector)
  mpm/                         #   MPM: ChartMPMSolver, particles, grid, transfers, constitutive

common/                         # Shared utilities
  models.py                    #   ChartDecoder, MaskNet, MLP
  geometry.py                  #   Jacobian ops, local coords, metric tensor
  schwarz.py                   #   Atlas loading, chart coloring
  utils.py                     #   Device resolution, seeding, plotting

benchmarks/                     # Validated numerical examples
  rabbit_poisson_fem/          #   Poisson on Stanford rabbit via FEM Schwarz
  mpm_basic/                   #   MPM verification benchmarks

tests/                          # Test suite
  test_chart_fem_solver.py     #   FEM solver tests
  test_chart_mpm_solver.py     #   MPM solver tests

topo_atlas/                     # Topology pipeline (docs, CI, phase plan)
configs/                        # YAML experiment configs
postprocessing/                 # Visualization & analysis
```

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run MPM tests only
pytest tests/test_chart_mpm_solver.py -v

# Run FEM tests only
pytest tests/test_chart_fem_solver.py -v

# Train SDF
python atlas/sdf/train_sdf.py

# Train atlas
python atlas/charts/train_atlas.py
```

## Code Patterns

- **Proper package imports**: Use `from common.models import ChartDecoder`, `from solvers.fem.chart_fem_solver import ChartFEMSolver`, etc.
- Device selection uses `resolve_device()` from `common.utils` to auto-detect CUDA/MPS/CPU.
- FEM solvers use NumPy/SciPy for assembly and sparse linear solves.
- MPM solvers use PyTorch throughout for GPU support and differentiability.
- Constitutive models follow the interface: `compute_stress(F, state) -> (sigma, state)`.
- All scripts write outputs to the `runs/` directory.

## Architecture Notes

### FEM Solver
- `ChartFEMSolver`: P1 tets on structured hex grid, Freudenthal decomposition, SDF-filtered.
- Mapped diffusion tensor: A_i = |det J_i| * J_i^{-1} J_i^{-T} from chart Jacobian.
- Vectorized assembly using einsum. Dirichlet BCs via row/column elimination.

### MPM Solver
- `ChartMPMSolver`: explicit time stepper in chart-local coordinates.
- Particles store position (xi), velocity, deformation gradient F, stress.
- Linear B-spline shape functions for P2G/G2P transfers.
- Neo-Hookean and J2 elastoplastic constitutive models.

### Topology Pipeline (atlas/topo/)
- `filtration.py`: sublevel-set filtration of neural SDF on cubical grid.
- `persistence.py`: persistent homology via GUDHI (birth-death pairs for H0, H1, H2).
- `ls_category.py`: Lusternik-Schnirelmann category -> minimum chart count M_min.
- `monitor.py`: tracks persistence diagrams across load steps, detects topology changes.
- `chart_spawn.py`: converts topology events into chart pair spawn requests.

## V&V Status

All 5 phases pass verification and validation (78 tests, 1 xpassed):
- Phase 1-3: Persistent homology pipeline, certification, dynamic monitoring
- Phase 4: Fracture benchmarks (Mode-I K_I extraction, crack topology detection)
- Phase 5: GUDHI overhead 1.1% on 16^3 grid with monitoring every 50 steps

Run full suite: `pytest tests/ topo_atlas/tests/ -v`
See `topo_atlas/docs/VandV.md` for per-phase exercise details.

**Analytical contact verification** (for verifying neural coordinate charts later): see
`docs/contact_verification_manual.md` (benchmarks CV-1..CV-5 = Hertz, Cattaneo–Mindlin, Brazilian
disc, nine-disc, nonconvex superformula; closed-form acceptance targets; §10 neural-chart protocol).
Closed forms: `docs/hertz_derivation/` (SymPy) + `postprocessing/contact_fields.py` (numpy). Harness
skeleton: `tests/test_neural_chart_verification.py` (skipped until neural charts exist).

## Gotchas

- **MPS (Apple Silicon)** requires float32 instead of float64. Use `resolve_dtype("auto", device)`.
- **Large binary files in `runs/`** — do not try to read `.pt` checkpoint files directly.
- **Checkpoint compatibility**: renaming modules (e.g., `core.train_sdf_rabbit.SDFNet` -> `atlas.sdf.train_sdf.SDFNet`) breaks `torch.load` on old checkpoints.
- **Older `manuscript_experiments/`** directory still contains the original FEM runner — canonical FEM code is in `solvers/fem/`.
- **GUDHI overhead**: Use 16^3 grid (not 32^3) for topology monitoring in production. Monitor every 50+ steps, not every step.
- **Edge cracks vs topology**: A partial edge crack does not create an H1 loop — it only changes H0 (connected components) when the crack fully severs the domain.
