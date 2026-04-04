# Challenge 8: Double Cantilever Beam Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 5.1)

Prismatic bar with pre-existing edge crack, arms pulled apart through pinholes. Tests Mode I fracture propagation governed by Griffith energy competition.

**Geometry:** Bar L=55mm, H=20mm, B=2.5mm, pre-existing crack A=25mm. Pinholes at 1.5mm from left edge, spacing H/3=6.667mm.
**Loading:** Displacement delta applied through pinholes in y-direction.
**Exact solution:** Eqs. (17)-(18) from Gross-Srawley / Wiederhorn beam theory. Force F(delta) and crack length a(delta) in closed form.
**Materials:** Glass (linear elastic) and PU (Neo-Hookean).

## Current Status: PARTIALLY VALIDATED

### Completed Work
- Analytical F vs delta and a vs delta curves (beam theory + Griffith)
- Comparison figure (Fig. 19 style) with AT1 model
- F_crit = 24.15 N at delta_crit = 17.3 um (glass)

### Remaining Work: Chart FEM Solve

### Decoder
- `BoxDecoder` for the bar body
- `CrackTipDecoder` at the crack front (x = -L/2 + A)
- SDF-filtered crack slit

### Charts
- 2-3 charts along bar length + 1 crack-tip chart
- Pre-existing crack handled by SDF filtering

### Boundary Conditions
- Pinhole loading: prescribed displacement at pinhole node locations
- Symmetry: solve half-specimen (top arm only)

### Expected Challenges
- **Medium risk**: Geometry is simple (rectangular bar), beam theory gives good reference
- Pinhole BCs are point loads — need careful handling in FEM
- Crack growth tracking: a(delta) as delta increases
- Need to rebuild mesh (SDF update) as crack advances

### Validation Criteria
- F vs delta curve matches Eq. (17) within 5%
- a vs delta curve matches Eq. (18) within 5%
- Crack propagation is stable (F decreases with increasing delta after onset)
- Phase-field contour matches Fig. 19(c) at delta = 0.03mm

### Tasks
1. Create bar SDF with pre-existing edge crack
2. Build multi-chart atlas: 2 BoxDecoder charts + 1 CrackTipDecoder
3. Pinhole BCs (concentrated displacement)
4. Incremental loading: solve, extract K_I, advance crack if K_I >= K_Ic
5. Record F(delta) and a(delta)
6. Compare to Eqs. (17)-(18) and Fig. 19

### Dependencies
- CrackTipDecoder (DONE)
- K_I extraction (DONE)
- Crack propagation driver (DONE)
- SDF update for crack growth (DONE — MultiCrackSDFOracle)

### Priority: HIGH (all infrastructure exists, just needs assembly)
