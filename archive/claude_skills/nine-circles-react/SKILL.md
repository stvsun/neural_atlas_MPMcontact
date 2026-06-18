---
name: nine-circles-react
version: 2.0.0
description: 3-stage ReAct loop for Nine Circles fracture benchmark
triggers:
  - "nine circles"
  - "react loop"
  - "fracture benchmark"
  - "improve score"
  - "stage 3"
  - "xfem critique"
---

# Nine Circles ReAct Loop Skill v2.0

## Purpose
Systematic improvement of the Nine Circles elastic brittle fracture benchmark using a 3-stage Reasoning-Action loop with XFEM-critic scoring.

## Current State (v2.0 — 2026-04-05, 12 iterations)
```
Stage 1 (Functional):   875/900  (97.2%)
Stage 2 (Mathematical): 665/675  (98.5%)
Stage 3 (XFEM-Critic):  197/465  (42.4%)
```

## 3-Stage ReAct Protocol

### Step 1: SCORE ALL 3 STAGES
```bash
P="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
$P nineO_examples/score.py           # Stage 1
$P nineO_examples/score_stage2.py    # Stage 2
$P nineO_examples/score_stage3.py    # Stage 3
```

### Step 2: COMPUTE COMBINED SCORE & CONSULT DECISION TREE
- Read `nineO_examples/decision_trees.md` for per-challenge priorities
- Identify the challenge with lowest combined (S1+S2+S3)/3 score
- Find the highest-ROI fix from the cross-challenge impact table

### Step 3: SELECT ACTION FROM DECISION TREE
Priority order for action selection:
1. **S effort fixes affecting 4+ challenges** (e.g., wire max_hoop_stress_angle)
2. **Bug fixes that unlock test categories** (e.g., dtype, SDF oracle wrapping)
3. **M effort structural improvements** (e.g., M-integral, displacement enrichment)
4. **L effort architectural changes** (e.g., Heaviside enrichment, nonlocal damage)

### Step 4: IMPLEMENT & TEST
```bash
$P -m pytest tests/ -v                    # Unit tests
$P nineO_examples/score.py N              # Stage 1 for challenge N
$P nineO_examples/score_stage3.py N       # Stage 3 for challenge N
```

### Step 5: UPDATE JOURNAL + DECISION TREE + SKILL
- Log in `nineO_examples/ReAct_journal.md`
- Update `nineO_examples/decision_trees.md` with new scores
- Increment skill version if strategy ranking changed

## Stage 3 Test Catalog (XFEM-Critic)
| Test | Pts | What it checks | Current |
|------|-----|----------------|---------|
| X1 | 15 | Williams angular modes (4 branch functions) | FAIL: only radial |
| X2 | 10 | Crack-face traction-free enforcement | FAIL: no Heaviside |
| X3 | 10 | Mixed-mode K_II + interaction integral | FAIL: not implemented |
| X4 | 10 | Curved crack path + branching | FAIL: direction never updated |
| X5 | 10 | Displacement discontinuity at overlaps | PARTIAL: PoU exists |
| X6 | 10 | Stiffness conditioning | PASS after diag scaling |
| X7 | 10 | Integration near singularity | PASS: no inverted elements |
| X8 | 15 | Nucleation mesh independence | PARTIAL: some converge |
| X9 | 10 | K_I accuracy vs XFEM (<1%) | FAIL: 26-109% error |

## Highest-ROI Actions (from decision_trees.md)
| # | Action | S3 Gain | Effort | Challenges |
|---|--------|---------|--------|------------|
| 1 | Wire max_hoop_stress_angle | +40 | S | C4,C5,C8,C9 |
| 2 | Implement M-integral | +40 | M | C4,C5,C8,C9 |
| 3 | Install GUDHI + multiprocessing | +20 S1 | S | C2,C3 |
| 4 | Nonlocal nucleation | +30 | L | C1-C3,C5-C7 |
| 5 | Heaviside enrichment | +25 | L | C3-C5,C8,C9 |

## Key Files
| File | Purpose |
|------|---------|
| `nineO_examples/score.py` | Stage 1 master |
| `nineO_examples/score_stage2.py` | Stage 2 master |
| `nineO_examples/score_stage3.py` | Stage 3 master |
| `nineO_examples/decision_trees.md` | Per-challenge decision trees |
| `nineO_examples/ReAct_journal.md` | Iteration log (12 entries) |
| `nineO_examples/common_stage3/CRITIQUE.md` | XFEM critique analysis |
| `solvers/fem/j_integral.py` | J-integral (needs M-integral extension) |
| `solvers/fem/denns_enrichment.py` | DENNs SDF enrichment |
| `solvers/fem/train_denns.py` | DENNs training script |
| `solvers/fem/crack_propagation.py` | Needs curved path support |
| `solvers/fracture_criteria.py` | Has max_hoop_stress_angle (unused!) |

## Key Learnings (12 iterations)
1. Robin DD Dirichlet approx attenuates 99.93% → use Multiplicative Schwarz
2. BoxDecoder.inverse() was missing → blocked Schwarz for box charts
3. P2 + evaluate_at incompatible → P1 n_cells=14 optimal for CrackTipDecoder
4. Reference-space testing needed for non-affine decoders
5. RPROP converges well for DENNs Williams field training
6. Diagonal scaling preconditioner reduces cond(K) from 1e18 to 1e2-1e6
7. Stage 3 XFEM gaps are structural, not parametric — need new capabilities

## Environment
```bash
PYTHON="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
# Python 3.10.18, torch 2.2.2
```
