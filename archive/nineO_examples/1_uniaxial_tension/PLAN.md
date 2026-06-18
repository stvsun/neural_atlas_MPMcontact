# Challenge 1: Uniaxial Tension Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 2.1)

Circular rod (L=15mm, R=2mm) under uniaxial tension. Tests fracture nucleation governed by the uniaxial tensile strength sigma_ts.

**Geometry:** Cylindrical rod, axisymmetric.
**Loading:** Axial displacement delta applied at ends.
**Exact solution (Eq. 5):** S = E * 2*delta/L for glass; Neo-Hookean for PU elastomer. Fracture at S = sigma_ts.
**Materials:** Glass (sigma_ts=40MPa), PU elastomer (sigma_ts=0.3MPa).

## Chart FEM Approach

### Decoder
Use `CylinderDecoder` (or a new `RodDecoder`) mapping [-1,1]^3 to a cylindrical rod sector. 2-4 charts around the circumference, each covering a 120-degree sector with full axial extent.

### Boundary Conditions
- z=0 face: u_z=0 (fixed)
- z=L face: u_z=delta (prescribed)
- Lateral surface: traction-free

### Expected Challenges
- **Low risk**: Geometry is simple, uniform stress state, mild aspect ratio (~7:1)
- Should work with n_cells=10, 4 charts
- Both glass (linear) and PU (Neo-Hookean) should be validated

### Validation Criteria
- sigma_zz error < 1% vs exact E*epsilon (glass)
- Nucleation at sigma_ts within 5%
- Crack perpendicular to axis (correct orientation)

### Tasks
1. Create `CylinderDecoder` or adapt existing `CylinderDecoder` for a rod
2. Build single-chart FEM solve on rod
3. Verify stress against analytical
4. Multi-chart Schwarz solve
5. Incremental loading with DP nucleation check
6. Compare stress-strain to Fig. 2 (glass) and Fig. 3 (PU)

### Priority: HIGH (simplest benchmark, should pass easily)
