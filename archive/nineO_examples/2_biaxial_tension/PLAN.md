# Challenge 2: Biaxial Tension Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 2.2)

Circular plate (R=5mm, L=0.25mm) under equi-biaxial tension. Tests fracture nucleation governed by biaxial tensile strength sigma_bs.

**Geometry:** Thin circular plate.
**Loading:** Affine displacement delta at lateral boundary.
**Exact solution:** Eq. (6) for glass, Eq. (7) for PU. Fracture at S = sigma_bs.
**Materials:** Glass (sigma_bs=27MPa), PU (sigma_bs=0.27MPa).

## Current Status: VALIDATED

### Completed Work
- **Real chart FEM solve**: 1331 nodes, 6000 tets, 32 load steps
- **Nucleation at S=28.09 MPa** (expected 27.03, error 3.9%)
- Stress-strain matches exact solution across all pre-nucleation steps
- VTU time series with displacement, stress, DP criterion
- Comparison figures with AT1 model (Figs. 5, 6)

### Existing Code
- `benchmarks/fracture/run_biaxial_chart_fem.py` — real chart FEM solve
- `benchmarks/fracture/run_biaxial_nucleation.py` — DP nucleation with 80 steps
- `benchmarks/fracture/run_biaxial_propagation.py` — comparison with paper
- `benchmarks/fracture/biaxial_tension.py` — geometry, material constants, SDF
- `postprocessing/plot_biaxial_nucleation_detail.py` — detailed Fig. 5 comparison

### Remaining Work
- PU elastomer version (requires Neo-Hookean in FEM solver)
- Multi-chart solve (currently single chart)
- Post-nucleation topology monitoring and chart spawning on real FEM solve

### Priority: LOW (already validated for glass)
