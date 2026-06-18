# Challenge 3: Torsion — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 2.3)
Mode III strength nucleation under torsion. A thin-walled glass tube is twisted until the Drucker-Prager criterion triggers shear-mode crack nucleation at 45 degrees.

## Material
- Glass: E = 70 GPa, nu = 0.22, sigma_ss = 44.4 MPa
- Thin-walled tube geometry

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C3.1 Decoder roundtrip | ~17 | PASS |
| C3.2 Shear stress accuracy | ~17 | PASS |
| C3.3 Multi-chart Schwarz DD | ~17 | PASS |
| C3.4 Drucker-Prager nucleation | ~17 | PASS |
| C3.5 45-degree crack orientation | ~16 | PASS |
| C3.6 Parallel speedup | ~16 | PASS |

## TODO
- [ ] Stage 2: Jacobian quality assessment
- [ ] Stage 2: Analytical stress match
- [ ] Stage 2: Domain decomposition convergence rate
