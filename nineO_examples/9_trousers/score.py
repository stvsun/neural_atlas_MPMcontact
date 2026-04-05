"""Score for Challenge 9: Trousers Test.

Checks:
  C9.1: Sheet SDF with pre-existing crack                        [10 pts]
  C9.2: Neo-Hookean finite-strain FEM                            [20 pts]
  C9.3: Near-incompressibility (no volumetric locking)           [15 pts]
  C9.4: Large-rotation kinematics (legs fold 180 deg)            [15 pts]
  C9.5: Mode III crack propagation                               [20 pts]
  C9.6: Normalized force 2F/B matches reference                  [20 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    checks = []
    total = 0

    # C9.1: Sheet SDF
    try:
        import numpy as np
        from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
        v = sdf_cracked_plate(np.array([[0, 10, 0]]), a=50.0, W=50.0, H=40.0, T=1.0)
        ok = v[0] < 0
        checks.append({"id": "C9.1", "name": "Sheet SDF", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C9.1", "name": "Sheet SDF", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C9.2-C9.6: All require finite-strain + large rotation + Mode III
    for cid, name, pts, note in [
        ("C9.2", "Neo-Hookean FEM", 20, "Exists in MPM, not ported to vector FEM"),
        ("C9.3", "Near-incompressibility", 15, "Requires mixed u-p formulation"),
        ("C9.4", "Large rotation (180 deg)", 15, "Requires updated Lagrangian or co-rotational"),
        ("C9.5", "Mode III propagation", 20, "Out-of-plane tearing not implemented"),
        ("C9.6", "2F/B vs delta curve", 20, "Requires complete trousers simulation"),
    ]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts, "note": note})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
