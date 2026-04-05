# Challenge 1: Uniaxial Tension — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 2.1)
Mode I strength nucleation under uniaxial tension. A glass specimen is loaded in tension until the Drucker-Prager nucleation criterion is met, verifying crack initiation stress and orientation.

## Material
- Glass: E = 70 GPa, nu = 0.22, sigma_ts = 40 MPa

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C1.1 Mesh build | ~17 | PASS |
| C1.2 Newton convergence | ~17 | PASS |
| C1.3 Stress accuracy | ~17 | PASS |
| C1.4 Drucker-Prager nucleation | ~17 | PASS |
| C1.5 Crack orientation | ~16 | PASS |
| C1.6 Multi-chart Robin DD | ~16 | PASS |

## TODO
- [ ] Stage 2: Convergence order test
- [ ] Stage 2: Patch test
- [ ] Stage 2: Manufactured solution
