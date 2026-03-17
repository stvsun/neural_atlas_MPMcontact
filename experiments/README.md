# Experimental / Exploratory Simulations

This folder contains **exploratory experiments** that are not part of the validated
manuscript benchmark set. These scripts were used during development to test
alternative formulations, material models, and atlas construction strategies.

For the five validated manuscript experiments, see `manuscript_experiments/`.

---

## Subfolders

### `rabbit_elder/`
Inverse Elder problem on the rabbit geometry:
- `run_rabbit_inverse_elder_atlas_schwarz.py` — Recover permeability tensor from Elder-type data

### `rabbit_inverse_neohookean/`
Inverse elasticity problems on the rabbit geometry with various material models:
- `run_rabbit_inverse_neohookean_mapped.py` — Single-chart mapped-sphere inverse
- `run_rabbit_inverse_neohookean_atlas_schwarz_normal_disp.py` — Normal displacement BC variant
- `run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz.py` — Arruda-Boyce material model
- `run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz_mms.py` — MMS verification
- `run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz_normal_ramp.py` — Normal ramp loading

### `torus_variants/`
Alternative torus geometry configurations:
- `run_torus_inverse_neohookean_atlas_cube.py` — Cube-domain torus embedding
- `run_torus_inverse_neohookean_atlas_fillet_inn.py` — Fillet INN variant

### `rabbit_atlas_variants/`
Atlas construction and PINN architecture experiments:
- `compact_chart_net.py` — CompactChartNet (Voronoi sub-atlas) module
- `build_rabbit_atlas_volumetric.py` — Volumetric atlas builder
- `build_rabbit_atlas_poissondisk.py` — Poisson-disk seeding (experimental copy)
- `run_poisson_rabbit_atlas_schwarz_experimental.py` — Extended Schwarz solver with CompactChartNet support
- `run_poisson_rabbit_atlas_schwarz_fem_experimental.py` — FEM reference (experimental copy)
- `chart_fem_solver_experimental.py` — FEM solver (experimental copy)
- `train_rabbit_atlas.py` — Atlas training (experimental copy)
- `postprocess_rabbit_poisson_dense_fields.py` — Dense field post-processing

### `paraview_exporters/`
Scripts for exporting results to ParaView VTK format:
- `export_fem_paraview.py` — FEM solution export
- `export_rabbit_atlas_paraview.py` — Atlas geometry export
- `export_rabbit_error_paraview.py` — Error field export
- `export_rabbit_elder_inverse_paraview.py` — Elder inverse results
- `export_rabbit_ti_inverse_interface_diagnostics.py` — Interface diagnostics

### `experimental_ideas/`
Early-stage experimental approaches:
- `atlas_split/` — Adaptive atlas splitting
- `fixed_focus/` — Fixed-focus chart refinement
