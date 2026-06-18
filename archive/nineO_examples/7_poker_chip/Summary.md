# Challenge 7: Poker Chip — Status Report

## Scores
| Stage | Score | Max |
|-------|-------|-----|
| Stage 1 | 100 | 100 |
| Stage 2 | - | TBD |

## Problem (Kamarei et al. Section 4.3)
Strength-Griffith mediation in a poker chip test. A thin disk of PU elastomer is loaded in confined tension, producing high hydrostatic stress at the center that leads to cavitation and crack nucleation perpendicular to the loading axis.

## Material
- PU elastomer: mu = 0.52 MPa, Lambda = 85.77 MPa, Gc = 41 N/m
- Neo-Hookean constitutive model

## Current Stage 1 Checks
| Check | Pts | Status |
|-------|-----|--------|
| C7.1 Disk geometry | ~17 | PASS |
| C7.2 Neo-Hookean stress | ~17 | PASS |
| C7.3 F-bar anti-locking | ~17 | PASS |
| C7.4 Hydrostatic stress at center | ~17 | PASS |
| C7.5 Crack perpendicular to load | ~16 | PASS |
| C7.6 Nucleation criterion | ~16 | PASS |

## TODO
- [ ] Stage 2: F-bar volumetric locking verification
- [ ] Stage 2: Cavitation pressure accuracy
- [ ] Stage 2: Overlap coverage assessment
- [ ] Stage 2: Decoder inverse quality
