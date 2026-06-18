"""Master Stage 3 scorer for Nine Circles benchmark.

Runs all Stage 3 XFEM-critic tests across 9 challenges and produces a consolidated scorecard.
Stage 3 tests evaluate fracture mechanics rigor against XFEM standards:
  - Williams enrichment completeness (X1)
  - Crack-face traction-free enforcement (X2)
  - Mixed-mode extraction (X3)
  - Curved crack path capability (X4)
  - Displacement discontinuity handling (X5)
  - Stiffness matrix conditioning (X6)
  - Integration near singularity (X7)
  - Nucleation mesh independence (X8)
  - K_I accuracy vs XFEM (X9)

Usage:
    python nineO_examples/score_stage3.py          # all 9
    python nineO_examples/score_stage3.py 1 4 8    # specific challenges
"""
import sys, os, time, importlib, importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHALLENGES = {
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

DIRS = {
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


def main():
    selected = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 10))

    results = {}
    for n in selected:
        name = CHALLENGES.get(n, f"Challenge {n}")
        dirname = DIRS.get(n)

        print(f"\n{'='*60}")
        print(f"  Stage 3: Challenge {n}: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            mod_path = os.path.join(os.path.dirname(__file__), dirname, "score_stage3.py")
            if not os.path.exists(mod_path):
                print(f"  [SKIP] No score_stage3.py found")
                results[n] = {"status": "NOT_IMPLEMENTED", "score": 0, "time": 0}
                continue

            spec = importlib.util.spec_from_file_location(f"score_stage3_{n}", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            res = mod.run_score()
            dt = time.time() - t0
            res["time"] = dt
            results[n] = res
        except Exception as e:
            dt = time.time() - t0
            print(f"  [ERROR] {e}")
            results[n] = {"status": "ERROR", "score": 0, "time": dt, "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY -- Stage 3 XFEM-Critic Scorecard")
    print(f"{'='*60}")
    print(f"  # {'Problem':<25s} {'Status':<18s} {'Score':>6s} {'Time':>7s}")
    print(f"--- {'-'*25} {'-'*18} {'-'*6} {'-'*7}")

    total_score = 0
    total_max = 0
    for n in selected:
        name = CHALLENGES.get(n, f"?")
        r = results.get(n, {})
        status = r.get("status", "?")
        score = r.get("score", 0)
        mx = r.get("max_score", 100)
        dt = r.get("time", 0)
        total_score += score
        total_max += mx
        pct = score / mx * 100 if mx > 0 else 0
        print(f"  {n} {name:<25s} {status:<18s} {pct:5.1f}% {dt:6.1f}s")

    pct = total_score / total_max * 100 if total_max > 0 else 0
    print(f"--- {'-'*25} {'-'*18} {'-'*6} {'-'*7}")
    print(f"    {'TOTAL':<25s} {'':18s} {pct:5.1f}%")


if __name__ == "__main__":
    main()
