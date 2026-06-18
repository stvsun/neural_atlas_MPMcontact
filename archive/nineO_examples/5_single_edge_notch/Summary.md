# Challenge 5: Single Edge Notch — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 4.1)
Strength-Griffith mediation for a single edge notch specimen. Five crack lengths A = [0.025, 0.1, 0.5, 1.0, 1.5] mm are tested to verify the transition from strength-dominated (small crack) to Griffith-dominated (large crack) failure.

## Material
- Glass: E = 70 GPa, nu = 0.22, Gc = 10 N/m, sigma_ts = 40 MPa
- 5 crack lengths: A = [0.025, 0.1, 0.5, 1.0, 1.5] mm

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C5.1 SDF at 5 crack lengths | ~17 | PASS |
| C5.2 CrackTipDecoder | ~17 | PASS |
| C5.3 Small crack limit (strength) | ~17 | PASS |
| C5.4 Large crack limit (Griffith) | ~17 | PASS |
| C5.5 K_I in transition zone | ~16 | PASS |
| C5.6 sigma_crit curve | ~16 | PASS |

## TODO
- [ ] Stage 2: K_I convergence study
- [ ] Stage 2: K_I accuracy assessment
- [ ] Stage 2: Transition curve match to Kamarei reference
- [ ] Stage 2: Spawned chart pair roundtrip verification
