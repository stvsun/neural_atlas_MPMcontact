# CMAME Claim-to-Evidence Matrix

Last updated: 2026-03-02

## Scope
Main-paper evidence is restricted to validated core benchmarks listed in:
- `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_benchmarks_registry.json`

Normalized metrics source:
- `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_metrics_collected.json`

## Claims Matrix
| ID | Claim (paper-facing wording) | Status | Evidence Script | Evidence Metrics | Figure / Artifact |
|---|---|---|---|---|---|
| C1 | The atlas-based mapped PINN framework solves 3D forward Poisson problems on nontrivial geometries without volumetric meshing. | Supported | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/pinn_3d_ellipsoid_mapped_sphere.py`, `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_poisson_star3d_mapped.py` | Ellipsoid best relative L2 = `2.26e-3`; Star best relative L2 = `2.59e-1` | Ellipsoid run log and Star metrics in registry records |
| C2 | Schwarz-coupled chart training produces accurate rabbit Poisson solutions and bounded interface mismatch. | Supported | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/experiments/run_poisson_rabbit_atlas_schwarz.py` | Best relative L2 = `2.21e-2`; best interface flux = `1.16e-2`; 3-run mean relative L2 = `6.00e-2 ± 5.13e-2` | `attempt20c_compact` artifacts and `core_submission_tables.md` |
| C3 | Inverse Neo-Hookean parameter recovery on torus is accurate for the original atlas setup. | Supported | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_atlas.py` | Best `mu` error = `9.27e-6%`; best `K` error = `2.29e-5%`; best traction rel-L2 = `2.35e-7` | `torus_inverse_mps` metrics + checkpoint |
| C4 | Schwarz dual inverse torus solver works for both traction and displacement observation modes. | Qualified | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_schwarz_dual.py` | Traction mode best obs rel-L2 = `5.80e-3`; Displacement mode best obs rel-L2 = `1.74e-3`, but displacement 3-run variance is high | Dual-mode runs under `runs/torus_schwarz_dual_accurate_*` |
| C5 | Rabbit inverse Elder benchmark robustly recovers permeability parameters on fixed atlas. | Supported | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_rabbit_inverse_elder_atlas_schwarz.py` | `k0` rel. error `3.00e-2`; mean eigenvalue rel. error `2.97e-2`; mean axis error `2.70°` | `runs/rabbit_inverse_elder_globalfield_small/*` |
| C6 | The method is broadly stable across all exploratory tracks (Arruda-Boyce rabbit, torus cube/fillet, adaptive split). | Not supported | Experimental scripts in `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/experiments` | Multiple divergence/non-convergence logs; MPS backend fragility in several runs | Must remain in limitations/appendix |

## Strong Claims Removed or Downgraded
| Original strong claim | Action |
|---|---|
| “Consistently accurate across all complex geometries and inverse models” | Downgraded to “validated on selected core benchmarks; exploratory tracks reported as limitations.” |
| “Schwarz universally improves all inverse tasks” | Downgraded to benchmark-specific statement; include displacement-mode variance caveat. |
| “Single-load inverse identification is robust for high-dimensional material models” | Removed from main claims; retained as future work note. |

## Mandatory Citation-to-Artifact Mapping for Main Text
| Benchmark label in TeX | Script path | Primary run / checkpoint |
|---|---|---|
| Ellipsoid Poisson | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/pinn_3d_ellipsoid_mapped_sphere.py` | `runs/examples/ellipsoid3d_20260212_180204/run.log` |
| Star Poisson | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_poisson_star3d_mapped.py` | `runs/examples/meshfree_accel_20260212_194730/poisson/poisson_star3d_mapped_metrics.json` |
| Rabbit Poisson Schwarz | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/experiments/run_poisson_rabbit_atlas_schwarz.py` | `runs/attempt20c_compact/rabbit_poisson_schwarz_attempt20c_compact_metrics.json` |
| Torus inverse (original atlas) | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_atlas.py` | `runs/torus_inverse_mps/torus_inverse_neohookean_atlas_metrics.json` |
| Torus inverse (Schwarz traction/displacement) | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_torus_inverse_neohookean_schwarz_dual.py` | `runs/torus_schwarz_dual_accurate_traction_*/...`, `runs/torus_schwarz_dual_accurate_displacement_*/...` |
| Rabbit inverse Elder | `/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/src/run_rabbit_inverse_elder_atlas_schwarz.py` | `runs/rabbit_inverse_elder_globalfield_small/*` |

