#!/usr/bin/env python3
"""Master runner for all 9 challenge problems with CrackTipDecoder enrichment.

Each problem uses:
  - BoxDecoder (or TubeSectorDecoder) bulk charts
  - CrackTipDecoder enrichment chart at the expected crack nucleation/tip site
  - Robin parallel DD for chart coupling
  - PyVista rendering of von Mises stress on deformed configuration

Usage:
    python nineO_examples/run_all.py          # run all
    python nineO_examples/run_all.py 1 2 4    # run specific problems
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBLEM_MAP = {
    1: "1_uniaxial_tension.run_challenge",
    2: "2_biaxial_tension.run_challenge",
    3: "3_torsion.run_challenge",
    4: "4_pure_shear.run_challenge",
    5: "5_single_edge_notch.run_challenge",
    6: "6_indentation.run_challenge",
    7: "7_poker_chip.run_challenge",
    8: "8_dcb.run_challenge",
    9: "9_trousers.run_challenge",
}

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problems = [int(x) for x in sys.argv[1:]]
    else:
        problems = list(range(1, 10))

    for p in problems:
        print(f"\n{'='*70}")
        print(f"  Challenge {p}: {PROBLEM_NAMES[p]}")
        print(f"{'='*70}")

        mod_name = f"nineO_examples.{PROBLEM_MAP[p]}"
        try:
            mod = importlib.import_module(mod_name)
            mod.run()
        except ModuleNotFoundError as e:
            print(f"  Module not found: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
