"""Score for Challenge 6: Indentation Test.

Checks:
  C6.1: Cylindrical block SDF                                    [10 pts]
  C6.2: Axisymmetric or 3D mesh builds                           [15 pts]
  C6.3: Contact BC (flat punch on top surface)                   [15 pts]
  C6.4: Ring crack nucleates near indenter edge                  [20 pts]
  C6.5: Ring crack radius > indenter radius                      [20 pts]
  C6.6: Cone crack forms upon further loading                    [20 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    checks = []
    total = 0

    # C6.1: Block SDF
    try:
        import numpy as np
        from solvers.fem.analytic_decoders import CylinderDecoder
        import torch
        dec = CylinderDecoder(R=25.0, z_center=12.5, L_half=12.5)
        xi = torch.tensor([[0, 0, 0.5]], dtype=torch.float64)
        x = dec(xi)
        ok = torch.isfinite(x).all()
        checks.append({"id": "C6.1", "name": "Block decoder", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C6.1", "name": "Block decoder", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C6.2-C6.6: All require axisymmetric FEM + contact — not implemented
    for cid, name, pts in [
        ("C6.2", "Axisymmetric/3D mesh", 15),
        ("C6.3", "Flat punch contact BC", 15),
        ("C6.4", "Ring crack nucleation", 20),
        ("C6.5", "Ring radius > punch radius", 20),
        ("C6.6", "Cone crack formation", 20),
    ]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts,
                        "note": "Requires axisymmetric FEM and contact BCs"})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
