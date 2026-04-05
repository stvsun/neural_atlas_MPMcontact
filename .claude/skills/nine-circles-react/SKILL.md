---
name: nine-circles-react
version: 1.1.0
description: ReAct loop for Nine Circles fracture benchmark improvement
triggers:
  - "nine circles"
  - "react loop"
  - "fracture benchmark"
  - "improve score"
---

# Nine Circles ReAct Loop Skill

## Purpose
Systematic improvement of the Nine Circles elastic brittle fracture benchmark (Kamarei et al., CMAME 448, 2026) using a Reasoning-Action loop.

## Current State (v1.1 — 2026-04-05)
- Stage 1: 870/900 (C2/C3 env-specific -10 each; C8 DCB = 90/100)
- Stage 2: 315/675 (46.7%) — global tests pass, per-challenge placeholders
- Blocking: C8.6 K_I extraction — root cause is Dirichlet approx of Robin BCs

## ReAct Loop Protocol

### Step 1: SCORE & DOCUMENT
```bash
PYTHON="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
$PYTHON nineO_examples/score.py          # Stage 1 (all 9)
$PYTHON nineO_examples/score_stage2.py   # Stage 2 (all 9)
$PYTHON nineO_examples/score.py 8        # Single challenge
```
- Update nineO_examples/{N}_*/Summary.md
- Update TODO lists

### Step 2: BRAINSTORM & DECIDE
- Identify lowest-scoring check
- Trace code path for failing check
- Formulate hypothesis
- List decisions needing human input

### Step 3: UPDATE PLAN
- Design bounded code change (one file, one function)
- Estimate effort: S (<1hr), M (1-4hr), L (4+hr)

### Step 4: RECORD IN JOURNAL
- Log in nineO_examples/ReAct_journal.md
- Format: REASON / ACT / RESULT / SKILL UPDATE

### Step 5: IMPLEMENT
- Write code changes
- Run: $PYTHON -m pytest tests/ -v

### Step 6: DEBUG & VALIDATE
- Run scoring, fix failures
- Commit on improvement

### Step 7: UPDATE THIS SKILL
- Encode new learnings
- Increment version
- Log skill diff in journal

## Key Files
| File | Purpose |
|------|---------|
| nineO_examples/score.py | Master Stage 1 scorer |
| nineO_examples/score_stage2.py | Master Stage 2 scorer |
| nineO_examples/ReAct_journal.md | Iteration log |
| nineO_examples/{1..9}_*/score.py | Per-challenge Stage 1 |
| nineO_examples/{1..9}_*/score_stage2.py | Per-challenge Stage 2 |
| nineO_examples/{1..9}_*/Summary.md | Progress docs |
| solvers/fem/j_integral.py | J-integral K_I extraction |
| solvers/fem/k_extraction.py | K_I extraction (dispatch) |
| solvers/fem/chart_vector_fem.py | FEM solver (P1/P2) |
| solvers/fem/robin_schwarz.py | Robin DD (with relaxation) |
| solvers/fem/crack_propagation.py | Crack propagation driver |
| solvers/fem/analytic_decoders.py | CrackTipDecoder |
| atlas/topo/monitor.py | TopologyMonitor |
| atlas/topo/chart_spawn.py | ChartSpawner |

## Environment
```bash
PYTHON="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
# Python 3.10.18, torch 2.2.2
```

## Strategy Rankings (empirical, updated each iteration)
| Rank | Strategy | Expected Impact | Status |
|------|----------|----------------|--------|
| 1 | **Fix Robin DD: proper weak Robin BC** | CRITICAL | Next iteration |
| 2 | Multiplicative Schwarz for DCB (bypass Robin) | HIGH | Alternative to #1 |
| 3 | J-integral + P2 (implemented) | HIGH (blocked by #1) | Done but ineffective until DD fixed |
| 4 | DENNs SDF enrichment (full) | MEDIUM | Planned |
| 5 | Live topology-solver loop | MEDIUM | Planned |
| 6 | NeuralCrackDecoder | LOW-MEDIUM | Deferred |

## Key Learnings (v1.1)
1. **Robin DD oscillates for DCB**: max_change swings 0.07-14.0 without relaxation
2. **Under-relaxation (omega=0.4) stabilizes convergence** but doesn't fix accuracy
3. **Root cause**: `u = lambda + g/delta` Dirichlet approximation (robin_schwarz.py:253) loses 99.93% of displacement at interface
4. **CrackTipDecoder radius** significantly affects DD convergence but NOT K_I accuracy
5. **P2 elements on crack tip chart work correctly** (mesh builds, assembly verified)
6. **J-integral module is correct** for smooth q-function but needs accurate FEM solution to work

## Convergence Criteria
Stop when: Stage 1 = 900/900 AND Stage 2 > 600/900, OR 3 consecutive iterations with zero improvement.

## Materials Reference
- Glass: E=70GPa, nu=0.22, sigma_ts=40MPa, sigma_bs=27MPa, sigma_ss=44.4MPa, Gc=10N/m
- PU elastomer: mu=0.52MPa, Lambda=85.77MPa, sigma_ts=0.3MPa, Gc=41N/m
