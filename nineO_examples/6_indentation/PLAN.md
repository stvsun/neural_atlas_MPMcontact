# Challenge 6: Indentation Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 4.2)

Cylindrical block indented by a flat-ended cylindrical punch. Tests strength-Griffith mediated nucleation with complex stress state.

**Geometry:** Block L=25mm, R=25mm (cylinder), indenter radius a=1mm (flat-ended, rigid, frictionless). Bottom bonded to rigid substrate.
**Loading:** Prescribed indenter displacement delta.
**Key physics:** Ring crack nucleates near the edge of the indenter (not underneath it). The ring crack then turns and grows into a cone crack. This is a 3D axisymmetric problem.
**Materials:** Glass only (PU has different failure mode — internal penny-shaped crack).

## Chart FEM Approach

### Decoder
- Axisymmetric: can use `CylinderDecoder` for the block
- Need fine resolution near the indenter edge (r ~ a = 1mm)
- `CrackTipDecoder` for the ring crack region (circular crack front)

### Key Challenges
- **Contact**: Rigid flat punch indenting the block — need contact BCs (prescribed displacement in a circular region on the top surface)
- **Axisymmetry**: True 3D is expensive; 2D axisymmetric formulation would be ideal but not yet implemented
- **Cone crack**: The crack turns from a ring to a cone — requires 3D crack path tracking
- **Fine resolution**: Element size h=0.005mm near the indenter (paper uses this)

### Expected Difficulty: HARD
- Axisymmetric FEM not yet available (would need new element formulation)
- Full 3D requires many charts and fine mesh near the indenter
- Contact BC implementation (flat punch) adds complexity
- Ring-to-cone crack transition requires dynamic crack path update

### Validation Criteria
- Ring crack nucleates at correct displacement delta_crit
- Ring crack radius > indenter radius a
- Crack turns into cone with correct angle (~22 degrees for glass)
- Phase-field contour matches Fig. 15

### Tasks
1. Implement axisymmetric FEM (or use full 3D with many charts)
2. Create cylindrical block SDF + flat punch contact BC
3. Build multi-chart atlas with refinement near indenter edge
4. Incremental loading until ring crack nucleation
5. Track crack path (ring -> cone transition)
6. Compare to Fig. 15

### Dependencies
- Axisymmetric FEM formulation (NOT YET AVAILABLE)
- Contact BC handling (NOT YET AVAILABLE)
- 3D crack path tracking (partially available via TopologyMonitor)

### Priority: LOW (requires significant new infrastructure)
