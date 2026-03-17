# CODEX.md — Onboarding for Codex Agents

## Project Overview
Mapped Sphere Method: meshfree PDE solver using neural SDFs and coordinate charts.
Python + PyTorch. Target: CMAME journal paper with 4 numerical examples.

## Architecture Pipeline
1. Train neural SDF
2. Build overlapping chart atlas
3. Train chart decoders
4. Solve PDE per chart with PINN
5. Couple via multiplicative Schwarz iteration

## Key Directories
```
manuscript_experiments/example{1-4}_*/   # 4 validated paper experiments (self-contained)
core/                                    # Shared: SDF training, atlas construction, gradient surgery
experiments/                             # Exploratory (not in paper), organized by topic
postprocessing/                          # Plotting/visualization scripts
manuscript/                              # LaTeX paper + scripts_figures/
configs/                                 # YAML configs
runs/                                    # Output data (large .pt checkpoints — don't read)
figures/                                 # Output plots
```

## Running
```bash
python manuscript_experiments/example1_forward_poisson/pinn_3d_ellipsoid_mapped_sphere.py
python core/train_sdf_rabbit.py
pytest test_chart_fem_solver.py
```

## Import Patterns
- Most scripts are **self-contained** (copy-paste MLP/ChartDecoder/SDFNet definitions).
- MLP, SDFNet, ChartDecoder are **duplicated** across files, not shared from `core/`.
- `manuscript_experiments/example2_*/run_poisson_rabbit_atlas_schwarz_fem.py` is an exception:
  imports from sibling modules + `core.train_sdf_rabbit`.
- Path setup uses `sys.path.insert(0, ...)`.

## Device Handling
```python
resolve_device()  # CUDA → MPS → CPU auto-detect
# float32 on MPS, float64 on CUDA/CPU
```

## Code Patterns
- Each `manuscript_experiments/example*` directory is a standalone experiment script.
- `experiments/` contains older `_experimental` copies of some scripts.
- YAML configs in `configs/` drive hyperparameters.
- Schwarz coupling: charts overlap, solutions enforced at boundaries via iteration.

## Testing
```bash
pytest test_chart_fem_solver.py
```

## Warnings
- `runs/` contains large `.pt` checkpoint files — do not read them.
- `experiments/` has exploratory code that may be outdated.
- No centralized module registry; grep for class names to find definitions.
