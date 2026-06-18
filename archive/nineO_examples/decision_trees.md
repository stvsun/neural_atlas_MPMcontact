# Per-Challenge Decision Trees: 3-Stage Score Analysis

Generated: 2026-04-05, ReAct Iteration 12

## Scoring Baseline

| # | Challenge | S1 | S2 | S3 | Combined | Priority |
|---|-----------|----|----|----|---------|---------:|
| 1 | Uniaxial | 100% | 100% | 40% | 80% | 4 |
| 2 | Biaxial | 90% | 100% | 60% | 83% | 5 |
| 3 | Torsion | 90% | 100% | 43% | 78% | 3 |
| 4 | Pure-shear | 100% | 100% | 41% | 80% | 2 |
| 5 | SEN | 100% | 93% | 45% | 79% | 1 |
| 6 | Indentation | 100% | 93% | 57% | 83% | 6 |
| 7 | Poker-chip | 100% | 100% | 40% | 80% | 7 |
| 8 | DCB | 95% | 100% | 42% | 79% | 1 |
| 9 | Trousers | 100% | 100% | 18% | 73% | 1 |

## Decision Tree per Challenge

### C1: Uniaxial Tension (S1=100, S2=100, S3=40%)
```
S3 gaps:
  X6 PASS (cond OK) ──> No action needed
  X8 FAIL (nucleation scatter=3.2mm) ──> Decision:
      ├─ Implement nonlocal damage? [HIGH effort, +5 pts]
      └─ Accept pointwise DP? [0 effort, 0 pts]
Recommendation: Defer. S1+S2 are perfect.
```

### C2: Biaxial Tension (S1=90, S2=100, S3=60%)
```
S1 gap: C2.6 GUDHI env ──> Install GUDHI [trivial, +10 S1]
S3 gaps:
  X6 PASS ──> No action
  X8 PASS (scatter=0.23mm) ──> Already good!
  X8 nonlocal FAIL ──> Structural gap
Recommendation: Install GUDHI for +10 S1. Best ROI.
```

### C3: Torsion (S1=90, S2=100, S3=43%)
```
S1 gap: C3.6 parallel speedup 0.6x ──> Decision:
      ├─ Use multiprocessing instead of threading? [MED effort, +10 S1]
      └─ Accept GIL limitation? [0 effort, 0 pts]
S3 gaps:
  X5 DD discontinuity FAIL ──> Structural: need Heaviside enrichment
  X5 PoU PASS ──> Good
  X6 PASS ──> No action
  X8 FAIL (scatter, nonlocal) ──> Structural gap
Recommendation: multiprocessing for +10 S1. X5 Heaviside is long-term.
```

### C4: Pure-Shear (S1=100, S2=100, S3=41%)
```
S3 gaps (all structural XFEM features):
  X1 Williams angular ──> Need angular enrichment functions
  X2 Traction-free ──> Need Heaviside enrichment
  X3 K_II FAIL ──> Need interaction integral
  X4 Curved path FAIL ──> Need max_hoop_stress in propagation loop
  X6 PASS ──> Good!
  X7 PASS ──> Good!
  X9 K_I 109% err ──> Need better extraction
Decision tree:
  Fastest S3 gain: X4 curved path [MED effort]
    ├─ Wire max_hoop_stress_angle into propagate_crack
    └─ Update crack_direction each step
  Highest impact: X3 interaction integral [HIGH effort]
    ├─ Implement M-integral in j_integral.py
    └─ Separate K_I and K_II
```

### C5: SEN (S1=100, S2=93, S3=45%)
```
Same as C4 + nucleation mesh independence:
  X1, X2, X3, X4 ──> Same structural gaps as C4
  X6 PASS, X7 PASS ──> Good
  X8 PASS (scatter=0.04mm!) ──> Excellent convergence
  X8 nonlocal FAIL ──> Structural
  X9 K_I 109% err ──> Need better extraction
S2 gap: C5.S2.2 K_I convergence 83% err ──> Related to X9
Recommendation: Focus on X3 (M-integral) + X4 (curved path).
```

### C6: Indentation (S1=100, S2=93, S3=57%)
```
S3 (best S3 performer!):
  X6 PASS (cond OK) ──> Good
  X7 PASS (elements OK) ──> Good
  X8 FAIL (scatter=4.6mm, not converging) ──> Mesh-dep nucleation
Recommendation: Indentation nucleation is geometrically sensitive.
  Nonlocal regularization would help most here.
```

### C7: Poker-Chip (S1=100, S2=100, S3=40%)
```
Same as C1: X6 PASS, X8 FAIL (nucleation scatter).
Recommendation: Defer. S1+S2 perfect.
```

### C8: DCB (S1=95, S2=100, S3=42%)
```
S1 gap: C8.6 K_I 26% err ──> Decision:
      ├─ Improve Schwarz convergence [MED, +5 S1]
      ├─ Implement M-integral [HIGH, +10 S3]
      └─ Add displacement enrichment from DENNs [MED, +? S1+S3]
S3 gaps:
  X1 Williams angular ──> Need angular enrichment
  X2 Traction-free σ_yy/max = 1.0 ──> Crack faces NOT traction-free!
  X3 K_II FAIL ──> Need M-integral
  X4 Curved path FAIL ──> Need direction update
  X5 DD discontinuity FAIL ──> Need Heaviside
  X6 PASS, X7 PASS ──> Good
  X9 K_I 109% err ──> Same as S1 gap
Decision tree:
  Fastest combined gain: Wire max_hoop_stress_angle (+10 X4)
  Highest impact: M-integral + displacement enrichment
```

### C9: Trousers (S1=100, S2=100, S3=18%)
```
S3 gaps (lowest S3 score):
  X3 K_II FAIL ──> Need Mode III K_III extraction
  X4 Curved path FAIL ──> Less relevant for Mode III tearing
  X5 DD discontinuity FAIL ──> Need Heaviside
  X6 cond=3.5e8 FAIL ──> BoxDecoder aspect ratio (50:20:0.5) is extreme
Decision tree:
  X6 is fixable: reduce BoxDecoder aspect ratio or use better scaling
  X3 needs K_III extraction (not just K_II)
```

## Cross-Challenge Impact Analysis

| Fix | Challenges Affected | S3 pts Gained | Effort |
|-----|--------------------:|:-------------:|:------:|
| **Wire max_hoop_stress_angle** | C4,C5,C8,C9 | +40 (4x10) | S |
| **Implement M-integral** | C4,C5,C8,C9 | +40 (4x10) | M |
| **Nonlocal nucleation** | C1,C2,C3,C5,C6,C7 | +30 (6x5) | L |
| **Heaviside enrichment** | C3,C4,C5,C8,C9 | +25 (5x5) | L |
| **BoxDecoder scaling for C9** | C9 | +8 | S |
| **Install GUDHI** | C2 | +10 (S1) | S |

## Recommended Next 3 Actions (by ROI)

1. **Wire max_hoop_stress_angle into propagate_crack** — S effort, +40 S3 pts
2. **Implement interaction integral (M-integral)** — M effort, +40 S3 pts
3. **Install GUDHI + multiprocessing** — S effort, +20 S1 pts
