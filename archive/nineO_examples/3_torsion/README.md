# Challenge 3: Torsion Test

## Problem Description (Kamarei et al. 2026, Section 2.3)

Thin-walled circular tube under torsion. Tests fracture nucleation governed by the shear strength.

**Geometry:** Tube with length L=5mm, inner radius r=2.85mm, outer radius R=3mm, wall thickness t=0.15mm.

**Loading:** Angle of twist alpha applied at one end, other end fixed.

**Material (soda-lime glass):** E=70GPa, nu=0.22, sigma_ss=44.4MPa, G_c=10 N/m.

**Exact solution (Eq. 10):** S = mu * alpha * (r+R) / (2L) until S = sigma_ss, then S = 0.

**Crack pattern:** 45-degree helical crack (perpendicular to max principal stress direction).

## Current Status: PARTIALLY READY

### What works
- **Single-chart FEM solve is exact**: tau = 33.57 MPa vs analytical 33.57 MPa (0.00% error)
- **TubeSectorDecoder** provides correct cylindrical-to-Cartesian Jacobian
- **Multi-chart Schwarz** converges with under-relaxation (omega=0.3)
- **Parallel execution**: 4 threads, 2.1x speedup

### What doesn't work
- **Nucleation triggers too early**: tau = 33.7 MPa instead of sigma_ss = 44.4 MPa
- **Root cause**: The thin wall (t=0.15mm) with ~2 effective elements across the thickness creates highly anisotropic elements (aspect ratio ~40:1). Boundary constraints at z-faces produce compatibility stresses (sigma_rr ~ 14 MPa) that push the Drucker-Prager criterion above the threshold prematurely.
- This is a **mesh resolution limitation**, not a formulation error.

### Key findings
1. Both St. Venant-Kirchhoff and small-strain formulations give identical results
2. The spurious normal stresses persist in cylindrical coordinates — they are real compatibility stresses, not coordinate artifacts
3. Volume-averaged DP is too conservative (never nucleates); element-wise is too aggressive

### What's needed
- Anisotropic mesh: n_r=20+, n_theta=6, n_z=6 (currently isotropic n=12)
- Or: separate radial-refinement charts covering the thin wall
- Or: cylindrical-coordinate-native FEM formulation

## Existing Code
- `benchmarks/fracture/run_torsion.py` — analytical stress-strain (Fig. 8 style)
- `benchmarks/fracture/run_torsion_chart_fem.py` — real chart FEM solve
- `postprocessing/export_torsion_vtu.py` — VTU time series export
- `solvers/fem/analytic_decoders.py::TubeSectorDecoder` — cylindrical decoder

## Figures
- `figures/torsion_fig8.png` — stress-strain comparison with AT1
- `figures/torsion_fig8c.png` — contour plots at nucleation
- `figures/torsion_chart_fem.png` — chart FEM results
- `figures/torsion_chart_fem_3d.png` — PyVista 3D rendering
