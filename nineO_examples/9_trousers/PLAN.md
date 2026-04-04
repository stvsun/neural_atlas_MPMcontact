# Challenge 9: Trousers Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 5.2)

Sheet with pre-existing edge crack, legs bent in opposite directions and pulled apart. Tests Mode III (tearing) fracture propagation.

**Geometry:** Sheet L=100mm, W=40mm, B=1mm, pre-existing crack A=50mm. Legs bent 180 degrees, grips pulled apart.
**Loading:** Separation delta between grips (after bending legs into same plane).
**Key physics:** Mode III (out-of-plane tearing) crack propagation. Legs stretch, bend, and twist elastically before crack grows. Self-similar steady-state propagation.
**Materials:** PU elastomer only (glass is too stiff to bend into trouser shape).

## Chart FEM Approach

### Key Challenges
- **Large deformation**: Legs fold 180 degrees — requires finite-strain kinematics with large rotations
- **Neo-Hookean material**: PU elastomer with finite-strain constitutive model
- **Near-incompressibility**: Lambda/mu = 165 causes volumetric locking
- **Mode III**: Out-of-plane tearing, not opening (Mode I) — different crack mechanics
- **Self-similar propagation**: Crack grows steadily under constant normalized force 2F/B
- **3D required**: Cannot be simplified to 2D (paper explicitly states this)

### Expected Difficulty: VERY HARD

This is the most demanding benchmark. It requires:
1. Finite-strain Neo-Hookean FEM (not just linear elastic)
2. Large-rotation kinematics (legs fold 180 degrees)
3. Near-incompressibility handling (mixed formulation)
4. Mode III crack propagation
5. True 3D with many elements

### Validation Criteria
- Normalized force 2F/B vs grip separation delta matches Fig. 22
- Crack length a vs delta matches Fig. 22
- Self-similar propagation (constant 2F/B during growth)
- Phase-field contour matches Fig. 22(c,f) at delta = 106mm

### Tasks
1. Implement Neo-Hookean for ChartVectorFEMSolver
2. Implement mixed u-p formulation for near-incompressibility
3. Large-rotation kinematics (finite strain F with updated Lagrangian)
4. Create sheet SDF with pre-existing crack (A=50mm)
5. Simulate leg bending (large displacement loading)
6. Incremental loading: bending -> tearing -> crack growth
7. Compare to Fig. 22

### Dependencies
- Neo-Hookean for vector FEM (NOT AVAILABLE)
- Mixed u-p formulation (NOT AVAILABLE)
- Large-rotation support (partially available — F is computed correctly)
- Mode III crack mechanics (NOT AVAILABLE)

### Priority: LOW (most complex benchmark, requires substantial new capability)
