#!/usr/bin/env python3
"""Scoring framework for the Nine Circles challenge problems.

Runs each benchmark's score module and produces a consolidated report
with quantitative pass/fail and accuracy metrics.

Usage:
    python nineO_examples/score.py          # run all
    python nineO_examples/score.py 1 2 3    # run specific problems
"""

import importlib
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBLEM_NAMES = {
    1: "Uniaxial tension",
    2: "Biaxial tension",
    3: "Torsion",
    4: "Pure-shear fracture",
    5: "Single edge notch",
    6: "Indentation",
    7: "Poker-chip",
    8: "Double cantilever beam",
    9: "Trousers",
}


def run_all_scores(problems=None):
    if problems is None:
        problems = list(range(1, 10))

    results = {}
    for p in problems:
        folder_map = {
            1: "1_uniaxial_tension",
            2: "2_biaxial_tension",
            3: "3_torsion",
            4: "4_pure_shear",
            5: "5_single_edge_notch",
            6: "6_indentation",
            7: "7_poker_chip",
            8: "8_dcb",
            9: "9_trousers",
        }
        folder = folder_map[p]
        module_path = f"nineO_examples.{folder}.score"

        print(f"\n{'='*60}")
        print(f"  Challenge {p}: {PROBLEM_NAMES[p]}")
        print(f"{'='*60}")

        try:
            mod = importlib.import_module(module_path)
            t0 = time.time()
            result = mod.run_score()
            result["time_s"] = time.time() - t0
            results[p] = result
        except ModuleNotFoundError:
            results[p] = {"status": "NOT_IMPLEMENTED", "score": 0.0,
                          "checks": [], "time_s": 0}
            print(f"  score.py not found — NOT IMPLEMENTED")
        except Exception as e:
            results[p] = {"status": "ERROR", "score": 0.0,
                          "checks": [], "error": str(e), "time_s": 0}
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Nine Circles Scorecard")
    print(f"{'='*60}")
    print(f"{'#':>3} {'Problem':<25} {'Status':<18} {'Score':>6} {'Time':>7}")
    print(f"{'-'*3} {'-'*25} {'-'*18} {'-'*6} {'-'*7}")

    total_score = 0
    total_possible = 0
    for p in sorted(results.keys()):
        r = results[p]
        status = r.get("status", "UNKNOWN")
        score = r.get("score", 0.0)
        t = r.get("time_s", 0)
        total_score += score
        total_possible += 100.0
        print(f"{p:3d} {PROBLEM_NAMES[p]:<25} {status:<18} {score:5.1f}% {t:6.1f}s")

    print(f"{'-'*3} {'-'*25} {'-'*18} {'-'*6} {'-'*7}")
    pct = total_score / total_possible * 100 if total_possible > 0 else 0
    print(f"    {'TOTAL':<25} {'':18} {total_score/len(results):5.1f}% ")
    print()

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problems = [int(x) for x in sys.argv[1:]]
    else:
        problems = None
    run_all_scores(problems)
