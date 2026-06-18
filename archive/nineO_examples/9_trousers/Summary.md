# Challenge 9: Trousers — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 5.2)
Mode III crack propagation in a trousers test. A PU elastomer sheet with a pre-crack is torn under large deformation, verifying steady-state tearing force and energy release rate.

## Material
- PU elastomer: mu = 0.52 MPa, Lambda = 85.77 MPa, Gc = 41 N/m
- Neo-Hookean constitutive model, sheet with pre-crack

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C9.1 Sheet SDF | ~17 | PASS |
| C9.2 Neo-Hookean stress | ~17 | PASS |
| C9.3 F-bar anti-locking | ~17 | PASS |
| C9.4 Large rotation handling | ~17 | PASS |
| C9.5 Mode III energy release G | ~16 | PASS |
| C9.6 Normalized force plateau | ~16 | PASS |

## TODO
- [ ] Stage 2: K_III extraction
- [ ] Stage 2: Neo-Hookean convergence study
- [ ] Stage 2: Steady-state tearing force accuracy
- [ ] Stage 2: Topology detection during tearing
