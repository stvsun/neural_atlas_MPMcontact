# Challenge 6: Indentation — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 4.2)
Ring/cone crack formation under indentation. A cylindrical glass block is loaded by a rigid cylindrical punch, producing a ring crack at the surface that propagates into a cone crack at depth.

## Material
- Glass: E = 70 GPa, nu = 0.22, Gc = 10 N/m
- Cylindrical block, punch radius R = 1 mm

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C6.1 Cylinder decoder | ~17 | PASS |
| C6.2 3D mesh generation | ~17 | PASS |
| C6.3 Contact boundary conditions | ~17 | PASS |
| C6.4 Ring crack nucleation | ~17 | PASS |
| C6.5 Ring crack radius | ~16 | PASS |
| C6.6 Cone angle | ~16 | PASS |

## TODO
- [ ] Stage 2: Hertzian pressure distribution
- [ ] Stage 2: Ring crack location accuracy
- [ ] Stage 2: Multi-decoder domain decomposition
