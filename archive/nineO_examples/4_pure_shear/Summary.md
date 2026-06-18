# Challenge 4: Pure Shear — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 3.1)
Griffith nucleation under Mode I pure shear. A glass strip with an edge crack of length a = 10 mm is loaded to verify critical grip separation and energy release rate matching the Griffith criterion G = Gc.

## Material
- Glass: E = 70 GPa, nu = 0.22, Gc = 10 N/m
- Strip with edge crack: a = 10 mm

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C4.1 Cracked SDF | ~17 | PASS |
| C4.2 CrackTipDecoder | ~17 | PASS |
| C4.3 K_I Williams extraction | ~17 | PASS |
| C4.4 Critical grip separation | ~17 | PASS |
| C4.5 Mode I straight crack | ~16 | PASS |
| C4.6 G = Gc verification | ~16 | PASS |

## TODO
- [ ] Stage 2: Energy release rate convergence
- [ ] Stage 2: CrackTipDecoder smoothness assessment
- [ ] Stage 2: GUDHI Betti number verification
