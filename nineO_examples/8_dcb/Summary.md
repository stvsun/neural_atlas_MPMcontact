# Challenge 8: DCB (Double Cantilever Beam) — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 90 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 5.1)
Griffith propagation under Mode I loading. A double cantilever beam specimen with a pre-crack is loaded to verify stable crack growth, force-displacement response, and stress intensity factor extraction.

## Material
- Glass: E = 70 GPa, nu = 0.22, Gc = 10 N/m, K_Ic derived from Gc
- Specimen: L = 55 mm, H = 20 mm, B = 2.5 mm, pre-crack A = 25 mm

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C8.1 Force vs displacement curve | ~14 | PASS |
| C8.2 Crack growth detection | ~14 | PASS |
| C8.3 Critical displacement delta_crit | ~14 | PASS |
| C8.4 K_Ic extraction | ~14 | PASS |
| C8.5 Chart FEM solve | ~14 | PASS |
| C8.6 FEM K_I accuracy | 10-15 | **NEEDS IMPROVEMENT** |
| C8.7 Stable growth criterion | ~14 | PASS |

**Note:** C8.6 currently scores 10-15 out of 20 points due to K_I extraction error of 15-50%. This is the only check across all 9 challenges not at full marks.

## TODO
- [ ] Stage 2: J-integral implementation for improved K_I extraction
- [ ] Stage 2: P2 enrichment near crack tip
- [ ] Stage 2: Mesh refinement with n_cells = 14
- [ ] Stage 2: Closed-loop FEM crack propagation
- [ ] Stage 2: Topology monitoring during propagation
