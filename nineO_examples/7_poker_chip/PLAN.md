# Challenge 7: Poker-Chip Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 4.3)

Circular disk bonded between a flat substrate and a spherical fixture, pulled apart. Tests strength-Griffith mediated nucleation under near-hydrostatic tension.

**Geometry:** Disk diameter D=10mm, thickness varying from L_min=1mm (center) to L_max=1.7mm (edge) due to spherical fixture of radius R=18.2mm. Bottom bonded to flat substrate, top bonded to spherical fixture.
**Loading:** Vertical displacement delta of the spherical fixture.
**Key physics:** Near-incompressible PU elastomer develops near-hydrostatic tension near the centerline. Crack nucleates perpendicular to the applied displacement within the disk. This is a 3D axisymmetric problem.
**Materials:** PU elastomer only (near-incompressible, Lambda/mu = 165).

## Chart FEM Approach

### Key Challenges
- **Near-incompressibility**: Lambda/mu = 165 causes volumetric locking with standard P1 elements. Needs mixed formulation (u-p) or Crouzeix-Raviart elements.
- **Axisymmetric geometry**: Variable thickness disk bonded to spherical fixture.
- **Hydrostatic tension**: The DP criterion must correctly handle triaxial states.
- **Neo-Hookean material**: Finite-strain constitutive model needed for PU.

### Expected Difficulty: VERY HARD
- Near-incompressibility is a fundamental limitation of the current P1 tet formulation
- Variable-thickness geometry requires careful SDF construction
- Contact with spherical fixture adds complexity

### Validation Criteria
- Crack nucleates at center of disk (perpendicular to pulling direction)
- Nucleation displacement matches phase-field prediction
- Crack pattern matches Fig. 17

### Tasks
1. Implement mixed u-p formulation or Crouzeix-Raviart elements for near-incompressibility
2. Create disk SDF with variable thickness (spherical cap geometry)
3. Neo-Hookean constitutive for FEM (port from MPM)
4. Axisymmetric or 3D multi-chart atlas
5. Incremental loading until nucleation
6. Compare to Fig. 17

### Dependencies
- Mixed formulation for near-incompressibility (NOT AVAILABLE)
- Neo-Hookean for vector FEM (NOT AVAILABLE — exists in MPM only)
- Axisymmetric FEM (NOT AVAILABLE)

### Priority: LOW (requires substantial new infrastructure)
