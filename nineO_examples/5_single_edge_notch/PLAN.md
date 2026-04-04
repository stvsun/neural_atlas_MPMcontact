# Challenge 5: Single Edge Notch Test — Implementation Plan

## Problem (Kamarei et al. 2026, Section 4.1)

Strips with pre-existing edge cracks of varying lengths under tension. Tests the transition from strength-dominated to Griffith-dominated nucleation.

**Geometry:** Strip L=25mm, W=5mm, B=0.25mm, edge crack A = 0.025, 0.1, 0.5, 1.0, 1.5mm (5 configurations).
**Loading:** Displacement delta applied at left/right boundaries (clamped).
**Key physics:** For small cracks (A << internal length), nucleation is strength-governed (sigma_crit = sigma_ts). For large cracks, nucleation is Griffith-governed (K_I = K_Ic). The transition occurs at an intermediate crack length.
**Critical stress:** sigma_crit = min(sigma_ts, K_Ic / sqrt(pi*A) / F(A/W)).
**Materials:** Glass and PU elastomer.

## Chart FEM Approach

### Decoder
- `BoxDecoder` for the main strip body
- `CrackTipDecoder` at the crack tip for singularity absorption
- 2-3 charts along the strip + 1 crack-tip chart

### Key Challenge: Size Effect
This benchmark probes the **competition between strength and toughness**. For A=0.025mm, the crack is tiny relative to the strip — nucleation is governed by sigma_ts. For A=1.5mm, classical LEFM applies. The model must handle both regimes with the same material constants.

### Validation Criteria
- sigma_crit vs A curve matches Fig. 12 (glass) and Fig. 13 (PU)
- For large A: sigma_crit ~ K_Ic / (sqrt(pi*A) * F(A/W))
- For small A: sigma_crit ~ sigma_ts
- Transition region: correct interpolation

### Tasks
1. Create strip SDF with parameterized crack length A
2. For each A in [0.025, 0.1, 0.5, 1.0, 1.5]:
   a. Build chart FEM with CrackTipDecoder
   b. Incremental loading until nucleation
   c. Record sigma_crit
3. Plot sigma_crit vs A and compare to Figs. 12-13
4. Verify both strength and Griffith limits are recovered

### Dependencies
- Requires CrackTipDecoder (DONE)
- Requires working K_I extraction (DONE)
- Requires Drucker-Prager nucleation (DONE)
- Requires small crack resolution (CrackTipDecoder handles this)

### Priority: HIGH (tests the strength-toughness transition — unique to this framework)
