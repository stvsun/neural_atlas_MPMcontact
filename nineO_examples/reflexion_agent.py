#!/usr/bin/env python3
"""Reflexion Agent for Nine Circles Benchmark (Shinn et al., NeurIPS 2023).

Implements Algorithm 1 from "Reflexion: Language Agents with Verbal
Reinforcement Learning" adapted for computational mechanics code improvement.

Algorithm 1:
  Initialize Actor, Evaluator, Self-Reflection: Ma, Me, Msr
  Initialize policy pi_theta(a|s), theta = {Ma, mem}
  Generate initial trajectory tau_0 using pi_theta
  Evaluate tau_0 using Me
  Generate initial self-reflection sr_0 using Msr
  Set mem <- [sr_0]
  Set t = 0
  while Me not pass or t < max_trials do
      Generate tau_t using pi_theta
      Evaluate tau_t using Me
      Generate self-reflection sr_t using Msr
      Append sr_t to mem
      Increment t
  end while

Components:
  Actor (Ma):     Reads reflexion_memory.md + decision_trees.md -> selects action
  Evaluator (Me): Runs score.py + score_stage2.py + score_stage3.py
  Self-Reflection (Msr): Generates verbal lesson from evaluation results

Memory:
  Short-term: Current trial trajectory (plan + actions + scores)
  Long-term:  Sliding window of last Omega=3 reflections in reflexion_memory.md
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
PYTHON = "/Users/wsun/opt/anaconda3/envs/reactmesh/bin/python"
OMEGA = 3  # Sliding window size for reflections


# ═══════════════════════════════════════════════════════════════
# EVALUATOR (Me): Runs all 3 scoring stages
# ═══════════════════════════════════════════════════════════════

def evaluate() -> dict:
    """Run all 3 scoring stages and return structured results.

    Returns dict with keys:
      s1_total, s1_max, s1_pct,
      s2_total, s2_max, s2_pct,
      s3_total, s3_max, s3_pct,
      combined_pct,
      per_challenge: {1..9: {s1, s2, s3}},
      passed: bool (True if all stages >= target)
    """
    results = {}

    for stage, script, key in [
        (1, "nineO_examples/score.py", "s1"),
        (2, "nineO_examples/score_stage2.py", "s2"),
        (3, "nineO_examples/score_stage3.py", "s3"),
    ]:
        try:
            out = subprocess.run(
                [PYTHON, str(ROOT / script)],
                capture_output=True, text=True, timeout=600,
                cwd=str(ROOT),
            )
            # Parse the TOTAL line
            for line in out.stdout.split("\n"):
                if "TOTAL" in line:
                    # Extract percentage
                    parts = line.strip().split()
                    for p in parts:
                        if "%" in p:
                            results[f"{key}_pct"] = float(p.replace("%", ""))
                            break
        except Exception as e:
            results[f"{key}_pct"] = 0.0
            print(f"  [Evaluator] Stage {stage} error: {e}")

    results["passed"] = (
        results.get("s1_pct", 0) >= 100.0
        and results.get("s2_pct", 0) >= 100.0
        and results.get("s3_pct", 0) >= 100.0
    )

    return results


def evaluate_quick(challenges=None) -> dict:
    """Quick evaluation on specific challenges only."""
    results = {}
    for stage, script, key in [
        (3, "nineO_examples/score_stage3.py", "s3"),
    ]:
        args = [PYTHON, str(ROOT / script)]
        if challenges:
            args.extend(str(c) for c in challenges)
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=300,
                cwd=str(ROOT),
            )
            for line in out.stdout.split("\n"):
                if "TOTAL" in line:
                    parts = line.strip().split()
                    for p in parts:
                        if "%" in p:
                            results[f"{key}_pct"] = float(p.replace("%", ""))
                            break
            results[f"{key}_output"] = out.stdout
        except Exception as e:
            results[f"{key}_pct"] = 0.0
    return results


# ═══════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def read_memory() -> str:
    """Read the reflexion memory (last Omega reflections)."""
    mem_path = ROOT / "nineO_examples" / "reflexion_memory.md"
    if mem_path.exists():
        return mem_path.read_text()
    return ""


def read_decision_tree() -> str:
    """Read the decision tree for ROI-based action selection."""
    dt_path = ROOT / "nineO_examples" / "decision_trees.md"
    if dt_path.exists():
        return dt_path.read_text()
    return ""


def append_reflection(reflection_text: str):
    """Append a reflection to memory. Maintain sliding window of Omega=3."""
    mem_path = ROOT / "nineO_examples" / "reflexion_memory.md"
    archive_path = ROOT / "nineO_examples" / "reflexion_archive.md"

    content = mem_path.read_text() if mem_path.exists() else ""

    # Count existing reflections (## Reflection N)
    import re
    reflections = re.findall(r"## Reflection \d+", content)
    n = len(reflections)

    # If we have Omega reflections, archive the oldest
    if n >= OMEGA:
        # Find the first reflection block and move it to archive
        first_ref_match = re.search(r"(## Reflection \d+.*?)(?=## Reflection \d+|\Z)",
                                     content, re.DOTALL)
        if first_ref_match:
            archived_text = first_ref_match.group(1).strip()
            # Append to archive
            archive_content = archive_path.read_text() if archive_path.exists() else ""
            archive_content += f"\n\n{archived_text}\n"
            archive_path.write_text(archive_content)
            # Remove from memory
            content = content[:first_ref_match.start()] + content[first_ref_match.end():]

    # Append new reflection
    new_n = n + 1
    content += f"\n\n## Reflection {new_n} (Trial {new_n} — {datetime.now().strftime('%Y-%m-%d')})\n\n"
    content += reflection_text + "\n"

    mem_path.write_text(content)


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def log_trial(trial_num, action, scores_before, scores_after, reflection):
    """Log a trial to the ReAct journal."""
    journal_path = ROOT / "nineO_examples" / "ReAct_journal.md"
    content = journal_path.read_text() if journal_path.exists() else ""

    entry = f"""
## Reflexion Trial {trial_num}: {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Actor Plan
{action}

### Evaluator Scores
| Stage | Before | After | Delta |
|-------|--------|-------|-------|
| S1 | {scores_before.get('s1_pct', '?'):.1f}% | {scores_after.get('s1_pct', '?'):.1f}% | {scores_after.get('s1_pct', 0) - scores_before.get('s1_pct', 0):+.1f}% |
| S2 | {scores_before.get('s2_pct', '?'):.1f}% | {scores_after.get('s2_pct', '?'):.1f}% | {scores_after.get('s2_pct', 0) - scores_before.get('s2_pct', 0):+.1f}% |
| S3 | {scores_before.get('s3_pct', '?'):.1f}% | {scores_after.get('s3_pct', '?'):.1f}% | {scores_after.get('s3_pct', 0) - scores_before.get('s3_pct', 0):+.1f}% |

### Self-Reflection
{reflection}
"""
    content += entry
    journal_path.write_text(content)


# ═══════════════════════════════════════════════════════════════
# ALGORITHM 1: REFLEXION MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def reflexion_loop(max_trials: int = 10):
    """Execute the Reflexion algorithm (Shinn et al., 2023, Algorithm 1).

    Initialize Actor, Evaluator, Self-Reflection: Ma, Me, Msr
    Initialize policy pi_theta, theta = {Ma, mem}
    while Me not pass or t < max_trials:
        Generate trajectory tau_t using pi_theta
        Evaluate tau_t using Me
        Generate self-reflection sr_t using Msr
        Append sr_t to mem
        t += 1
    """
    print("=" * 70)
    print("  REFLEXION AGENT — Nine Circles Benchmark")
    print("  (Shinn et al., NeurIPS 2023, Algorithm 1)")
    print("=" * 70)

    # Step 0: Initial evaluation
    print("\n[Evaluator] Running initial 3-stage evaluation...")
    scores = evaluate()
    print(f"  S1={scores.get('s1_pct', 0):.1f}%  "
          f"S2={scores.get('s2_pct', 0):.1f}%  "
          f"S3={scores.get('s3_pct', 0):.1f}%")

    if scores.get("passed"):
        print("\n[Reflexion] ALL STAGES PASS! Nothing to do.")
        return

    t = 0
    while not scores.get("passed") and t < max_trials:
        t += 1
        print(f"\n{'='*70}")
        print(f"  TRIAL {t} / {max_trials}")
        print(f"{'='*70}")

        # Step 1: Actor reads memory and decision tree
        memory = read_memory()
        decision_tree = read_decision_tree()

        print(f"\n[Actor] Reading reflexion_memory.md "
              f"({memory.count('## Reflection')} reflections)")
        print(f"[Actor] Reading decision_trees.md")

        # Step 2: Actor selects action (printed for human review)
        print(f"\n[Actor] Based on memory + decision tree, the next action is:")
        print(f"  >>> See decision_trees.md 'Recommended Next 3 Actions'")
        print(f"  >>> Human implements the action, then re-runs this script")

        # Step 3: Placeholder for action execution
        # In the automated version, this would call Claude to implement code.
        # For now, we assume the human has made changes between runs.
        print(f"\n[Actor] Waiting for code changes...")
        print(f"  (In the full agent, Claude would implement changes here)")

        # Step 4: Evaluate
        print(f"\n[Evaluator] Running 3-stage evaluation...")
        scores_before = dict(scores)
        scores = evaluate()
        print(f"  S1={scores.get('s1_pct', 0):.1f}%  "
              f"S2={scores.get('s2_pct', 0):.1f}%  "
              f"S3={scores.get('s3_pct', 0):.1f}%")

        # Step 5: Compute delta
        delta_s1 = scores.get('s1_pct', 0) - scores_before.get('s1_pct', 0)
        delta_s2 = scores.get('s2_pct', 0) - scores_before.get('s2_pct', 0)
        delta_s3 = scores.get('s3_pct', 0) - scores_before.get('s3_pct', 0)
        print(f"  Delta: S1={delta_s1:+.1f}%  S2={delta_s2:+.1f}%  S3={delta_s3:+.1f}%")

        # Step 6: Self-reflection (template for human/LLM to fill)
        if delta_s1 + delta_s2 + delta_s3 > 0:
            outcome = "improvement"
        elif delta_s1 + delta_s2 + delta_s3 == 0:
            outcome = "no change"
        else:
            outcome = "regression"

        reflection = (
            f"**Trial {t} outcome: {outcome}**\n\n"
            f"Scores: S1={scores.get('s1_pct', 0):.1f}%, "
            f"S2={scores.get('s2_pct', 0):.1f}%, "
            f"S3={scores.get('s3_pct', 0):.1f}%\n\n"
            f"Delta: S1={delta_s1:+.1f}%, S2={delta_s2:+.1f}%, S3={delta_s3:+.1f}%\n\n"
            f"*(Fill in: I attempted X because Y. Result was Z. "
            f"This succeeded/failed because [...]. Next trial I should [...])*"
        )

        # Step 7: Append reflection to memory
        append_reflection(reflection)
        print(f"\n[Self-Reflection] Written to reflexion_memory.md")

        # Step 8: Log trial
        log_trial(t, f"Trial {t} action (see journal)", scores_before, scores, reflection)
        print(f"[Logger] Trial {t} logged to ReAct_journal.md")

        # Check if passed
        if scores.get("passed"):
            print(f"\n{'='*70}")
            print(f"  ALL STAGES PASS! Reflexion converged in {t} trials.")
            print(f"{'='*70}")
            return

        # If no improvement after this trial, the next Actor will read
        # the reflection and try a different strategy.
        if delta_s1 + delta_s2 + delta_s3 <= 0:
            print(f"\n[Reflexion] No improvement — reflection will guide next trial")

    print(f"\n{'='*70}")
    print(f"  MAX TRIALS ({max_trials}) REACHED.")
    print(f"  Final: S1={scores.get('s1_pct', 0):.1f}%  "
          f"S2={scores.get('s2_pct', 0):.1f}%  "
          f"S3={scores.get('s3_pct', 0):.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    max_t = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    reflexion_loop(max_trials=max_t)
