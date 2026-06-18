# Challenge 4: Pure-Shear Fracture Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 3.1)

Thin strip with a pre-existing edge crack under uniform grip separation. Tests fracture nucleation from a large pre-existing crack governed by the Griffith energy competition.

**Geometry:** Strip L=50mm, H=5mm, B=0.5mm, pre-existing edge crack A=10mm.
**Loading:** Prescribed separation h between top and bottom grips (clamped).
**Key physics:** The energy release rate is uniform along the crack front: G = mu * h^2 / H (Rivlin-Thomas for soft materials), or G = E*h^2/(4H(1-nu^2)) for glass.
**Exact solution:** Crack grows when G = G_c. Critical grip separation h_crit = sqrt(G_c * H / (E/(4(1-nu^2)))).
**Materials:** Glass (h_crit ~ 0.036mm), PU elastomer (h_crit ~ 0.628mm).

## Chart FEM Approach

### Decoder
Use `BoxDecoder` for the strip. The pre-existing crack is handled by SDF filtering (CrackedPlateSDFOracle adapted for the strip geometry).

### Charts
- 2-4 charts along the strip length
- CrackTipDecoder at the crack front for singularity absorption
- SDF-filtered elements remove the crack region

### Boundary Conditions
- Top face: u_y = +h/2
- Bottom face: u_y = -h/2
- Left/right: traction-free

### Expected Challenges
- **Medium risk**: Simple geometry but crack-tip stress field needs proper resolution
- The crack is semi-infinite (A/L = 0.2) — standard LEFM applies
- K_I extraction should work with existing infrastructure
- Plane-strain assumption simplifies to 2D-like problem

### Validation Criteria
- Critical grip separation h_crit within 5% of analytical
- K_I = G_c relationship satisfied: K_I^2 = E * G_c / (1-nu^2)
- Crack grows straight ahead (Mode I)

### Tasks
1. Create strip SDF (thin rectangle with edge crack)
2. Single-chart FEM solve with BoxDecoder + CrackTipDecoder
3. Extract K_I from displacement field
4. Verify G = G_c at critical load
5. Compare to Fig. 10 (glass) and Fig. 10 (PU)

### Priority: HIGH (Griffith nucleation — different from strength tests)
