# Paper Starter (One Page)

## Working Title
Meshfree Atlas-Based Mapped-Domain PINNs for Forward and Inverse PDEs on Complex 3D Geometry

## Abstract Draft
We present a meshfree physics-informed framework for solving forward and inverse boundary-value problems on complex 3D geometries by learning and using mapped coordinate charts. The method combines a reference-domain formulation, multi-chart atlas representation, and interface coupling to solve PDEs without volumetric meshing. We evaluate the approach on manufactured and inverse benchmarks, including Neo-Hookean elasticity on torus domains and buoyancy-driven inverse flow on rabbit geometry. The strongest results are obtained for torus inverse elasticity, where both original-atlas and Schwarz-coupled variants recover material parameters with high accuracy. On rabbit geometry, inverse Elder-flow cases are consistently successful, while hyperelastic Arruda–Boyce identification remains challenging due to interface sensitivity and identifiability limits. We provide implementation-level diagnostics, backend stability considerations for Apple MPS execution, and a reproducible benchmark protocol for publication-ready comparison.

## Main Contributions (Draft Claims)
1. A meshfree mapped-domain PINN workflow using atlas charts for complex 3D domains.
2. A practical Schwarz-coupled multi-chart training strategy for inverse elasticity.
3. A reproducible benchmark suite spanning forward Poisson and inverse parameter identification.
4. A diagnostic protocol for interface artifacts, identifiability, and backend stability (MPS/CPU fallback).

## Method Section Skeleton
1. Geometry and mapping representation:
   - Sphere/reference chart coordinates, rabbit/torus physical domains.
   - Atlas chart construction and overlap handling.
2. Mapped PDE formulation:
   - Jacobian-transformed residuals and boundary conditions.
3. Multi-objective optimization:
   - PDE, BC/data, interface value/flux, regularization terms.
4. Inverse parameterization:
   - Positive transforms, staged optimization, trust/guard controls.
5. Implementation details:
   - Backend policy (MPS float32 + CPU fallback for fragile operations).

## Benchmark Table Skeleton (Fill/Update)
| Family | Script | Best Run Path | Key Metric | Best Value | Status |
|---|---|---|---|---:|---|
| Torus inverse (original atlas) | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_atlas.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/torus_inverse_mps/torus_inverse_neohookean_atlas_metrics.json` | composite score | `3.96e-07` | strong |
| Torus inverse (Schwarz, displacement) | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_schwarz_dual.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/torus_schwarz_dual_accurate_displacement_20260215_140200/torus_inverse_neohookean_schwarz_displacement_accurate_metrics.json` | composite score | `6.73e-03` | strong |
| Torus inverse (Schwarz, traction) | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_schwarz_dual.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/torus_schwarz_dual_accurate_traction_20260215_140800/torus_inverse_neohookean_schwarz_traction_accurate_metrics.json` | composite score | `1.02e-02` | strong |
| Rabbit mapping | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/train_mapping_from_sdf.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/examples/meshfree_accel_20260212_194730/map_rabbit_short/mapping_rabbit_metrics_relaxed.json` | mapping rel metric | `2.72e-02` | good |
| Rabbit inverse Elder | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_rabbit_inverse_elder_atlas_schwarz.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/rabbit_inverse_elder_globalfield_small/rabbit_inverse_elder_atlas_schwarz_globalfield_small_metrics.json` | param metric | `4.31e-02` | good |
| Rabbit Poisson atlas | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/experiments/run_poisson_rabbit_atlas_schwarz.py` | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/atlas_schwarz_adaptive_fastmps_20260214_210804/seed_42/rabbit_poisson_schwarz_adaptive_refined_split_s42_metrics.json` | rel metric | `7.43e-02` | mixed |

Source for ranking values: `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/run_ranking_latest.json`

## Figure Checklist (Publication Set)
1. Pipeline overview: mapping + atlas + mapped PDE + inverse loop.
2. Torus inverse results:
   - Traction and displacement variants (pred vs true, error maps, parameter convergence).
3. Rabbit inverse Elder:
   - Field plots + parameter recovery curve + interface diagnostics.
4. Rabbit Poisson:
   - Pressure quality and seam-aware velocity visualization policy.
5. Ablation:
   - Schwarz interface weighting effect and deterministic eval/checkpoint policy.

## Risks/Limits Paragraph Starter
Performance differs substantially across geometry/problem classes: torus inverse elasticity is well-conditioned and stable, while rabbit hyperelastic inversion is sensitive to chart interfaces, load informativeness, and backend numerical limitations. On Apple MPS, float32-only execution and fallback behavior can affect stability and reproducibility for second-order derivatives and certain linear algebra kernels. These constraints motivate reporting both accuracy and robustness diagnostics, including interface mismatch, sensitivity rank/conditioning, and backend-specific safeguards.

## Immediate Writing To-Do
1. Replace table placeholders with 3-seed mean/std for primary benchmarks.
2. Add exact loss definitions used in final selected runs.
3. Add hardware/runtime details (M4, MPS/CPU policy) in reproducibility appendix.
4. Add failure analysis subsection for rabbit Arruda–Boyce and planned remedies.

