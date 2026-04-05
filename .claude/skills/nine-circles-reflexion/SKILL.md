---
name: nine-circles-reflexion
version: 1.0.0
description: Reflexion agent for Nine Circles fracture benchmark (Shinn et al. 2023)
triggers:
  - "nine circles"
  - "reflexion"
  - "fracture benchmark"
  - "improve score"
  - "stage 3"
  - "trial"
---

# Nine Circles Reflexion Agent Skill

## Architecture (Shinn et al., 2023)

Three roles:
- **Actor (Mₐ)**: Plans + executes code changes, conditioned on reflexion_memory.md
- **Evaluator (Mₑ)**: Runs 3-stage scoring, computes deltas
- **Self-Reflection (Mₛᵣ)**: Writes verbal lessons explaining WHY tests passed/failed

## Memory
- **Short-term**: Current trial plan + actions + test outputs (in ReAct_journal.md)
- **Long-term**: Sliding window of last 3 reflections (`reflexion_memory.md`)
- **Archive**: Older reflections (`reflexion_archive.md`)

## Current Scores (after Trial 1)
```
Stage 1: 875/900  (97.2%)
Stage 2: 665/675  (98.5%)
Stage 3: 237/465  (51.0%)
```

## Trial Protocol

```
TRIAL N:
  1. ACTOR reads reflexion_memory.md
  2. ACTOR reads decision_trees.md
  3. ACTOR plans action conditioned on both
  4. ACTOR implements
  5. EVALUATOR:
     P="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
     $P nineO_examples/score.py
     $P nineO_examples/score_stage2.py
     $P nineO_examples/score_stage3.py
  6. EVALUATOR computes delta
  7. SELF-REFLECTION writes to reflexion_memory.md:
     "I attempted X because Y. Result was Z.
      This failed/succeeded because [root cause].
      Next trial I should [corrective strategy]."
  8. Keep 3 reflections in memory; archive older ones
  9. Update decision_trees.md
  10. Commit + push
```

## Key Files
| File | Role |
|------|------|
| `nineO_examples/reflexion_memory.md` | Actor reads BEFORE every trial |
| `nineO_examples/reflexion_archive.md` | Historical reflections |
| `nineO_examples/decision_trees.md` | ROI-prioritized actions |
| `nineO_examples/ReAct_journal.md` | Trial log |
| `nineO_examples/score.py` | Stage 1 |
| `nineO_examples/score_stage2.py` | Stage 2 |
| `nineO_examples/score_stage3.py` | Stage 3 XFEM-critic |

## Environment
```bash
PYTHON="/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
```
