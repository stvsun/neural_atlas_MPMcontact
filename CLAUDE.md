# CLAUDE.md

Onboarding guide for Claude Code agents working on this codebase.

## Project Overview

**Mapped Sphere Method for Complex 3D Geometry** — a meshfree atlas-based PINN framework for solving forward and inverse PDEs on complex 3D geometries. Target journal: CMAME.

Key techniques: Neural SDF, coordinate charts/atlas, multiplicative Schwarz domain decomposition, PCGrad gradient surgery, CompactChartNet.

- **Language**: Python with PyTorch
- **Hardware**: CUDA GPU, Apple MPS (M1-M4), or CPU (auto-detected via `resolve_device()`)

## Repository Structure

```
manuscript_experiments/     # 4 validated numerical examples from the CMAME paper
  example1_forward_poisson/ #   Poisson on 3D ellipsoid + star domain
  example2_rabbit_poisson/  #   Poisson on Stanford rabbit (Schwarz decomposition)
  example3_torus_inverse_original/ # Inverse Neo-Hookean on torus (original atlas)
  example4_torus_inverse_schwarz_dual/ # Inverse Neo-Hookean on torus (Schwarz dual)

core/                       # Shared training modules (SDF, atlas, gradient surgery)
  train_sdf_rabbit.py       #   Neural SDF training (with PLY support)
  train_mapping_from_sdf.py #   Chart decoder training from SDF
  train_rabbit_atlas.py     #   Atlas construction
  build_rabbit_atlas_poissondisk.py # Poisson-disk atlas seeding
  pinn_gradient_surgery.py  #   PCGrad implementation
  build_mesh_sdf.py         #   Mesh-based SDF construction
  train_sdf_chartwise.py    #   Chart-wise SDF training

experiments/                # Exploratory/non-manuscript experiments
  rabbit_elder/             #   Inverse Elder problem on rabbit (moved from manuscript)
  rabbit_inverse_neohookean/ #  Neo-Hookean/Arruda-Boyce rabbit variants
  torus_variants/           #   Torus geometry variations
  rabbit_atlas_variants/    #   Atlas construction experiments + CompactChartNet
  paraview_exporters/       #   ParaView visualization exporters
  experimental_ideas/       #   Atlas splitting, fixed focus refinement

postprocessing/             # Figure generation and convergence plotting
manuscript/                 # LaTeX paper, figures, tables, figure-generation scripts
configs/                    # YAML configuration files
scripts/                    # Shell scripts (experimental, successful, release)
runs/                       # Output data from experiments
figures/                    # Generated figures
```

## Key Architectural Concepts

1. **Neural SDF** — Geometry encoded as a learned signed distance function (Eikonal-trained).
2. **Coordinate Charts** — Overlapping balls on the surface, each with a TNB frame and decoder mapping local 2D coords to 3D.
3. **PINN per chart** — Each chart has a local MLP (or CompactChartNet) solving the PDE in local coordinates.
4. **Multiplicative Schwarz** — Charts coupled through alternating value+flux matching at overlap interfaces.
5. **PCGrad** — Gradient surgery resolves L_pde vs L_bc conflicts per chart.

## Common Commands

```bash
# Run a manuscript experiment
python manuscript_experiments/example1_forward_poisson/pinn_3d_ellipsoid_mapped_sphere.py

# Train SDF
python core/train_sdf_rabbit.py

# Train atlas
python core/train_rabbit_atlas.py

# Generate paper figures
python manuscript/scripts_figures/example1_forward_poisson.py

# Run tests
pytest test_chart_fem_solver.py
```

## Code Patterns

- Most solver scripts are **self-contained** — they redefine MLP, ChartDecoder, etc. locally rather than importing from a shared module.
- The main cross-file dependency: `run_poisson_rabbit_atlas_schwarz_fem.py` imports from `chart_fem_solver.py`, `run_poisson_rabbit_atlas_schwarz.py`, and `core.train_sdf_rabbit`.
- Scripts use `sys.path` manipulation for imports; the repo root must be findable.
- Device selection uses `resolve_device()` to auto-detect CUDA/MPS/CPU.
- All scripts write outputs to the `runs/` directory.

## Gotchas

- **Duplicated classes**: MLP, SDFNet, ChartDecoder, MaskNet are copy-pasted across scripts, not shared. Changes to one copy do not propagate.
- **`experiments/` has older copies** of some manuscript scripts (with `_experimental` suffix). These may be stale.
- **Large binary files in `runs/`** — do not try to read `.pt` checkpoint files.
- **MPS (Apple Silicon)** sometimes requires float32 instead of float64. Watch for dtype errors on M-series Macs.
- **`configs/` YAML files** may reference old paths from before the repo reorganization.
