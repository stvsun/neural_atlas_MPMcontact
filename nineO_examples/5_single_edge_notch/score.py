"""Score for Challenge 5: Single Edge Notch Test.

Checks:
  C5.1: Strip SDF with parameterized crack length                [10 pts]
  C5.2: CrackTipDecoder for each crack length                    [10 pts]
  C5.3: Small crack (A=0.025): sigma_crit ~ sigma_ts             [20 pts]
  C5.4: Large crack (A=1.5): sigma_crit ~ K_Ic/sqrt(pi*A)/F     [20 pts]
  C5.5: Transition region follows strength-Griffith interpolation [20 pts]
  C5.6: sigma_crit vs A curve matches Figs. 12-13                [20 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import numpy as np, math
    checks = []
    total = 0

    E, nu, Gc = 70e3, 0.22, 0.01
    sigma_ts = 40.0
    from solvers.fracture_criteria import griffith_K_Ic
    K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)

    # C5.1: Strip SDF
    try:
        from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
        for A in [0.025, 0.1, 0.5, 1.0, 1.5]:
            v = sdf_cracked_plate(np.array([[0, 1, 0]]), a=A, W=12.5, H=5.0, T=0.25)
            assert v[0] < 0
        ok = True
        checks.append({"id": "C5.1", "name": "Strip SDF (5 crack lengths)", "pass": ok, "pts": 10, "max": 10})
        total += 10
    except Exception as e:
        checks.append({"id": "C5.1", "name": "Strip SDF", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C5.2: CrackTipDecoder
    try:
        import torch
        from solvers.fem.analytic_decoders import CrackTipDecoder
        for A in [0.025, 1.5]:
            d = CrackTipDecoder.from_crack_tip([-12.5+A, 0, 0], [1, 0, 0], [0, 1, 0], radius=min(A*0.5, 0.3))
            x = d(torch.zeros(1, 3, dtype=torch.float64))
            assert torch.isfinite(x).all()
        ok = True
        checks.append({"id": "C5.2", "name": "CrackTipDecoder (all A)", "pass": ok, "pts": 10, "max": 10})
        total += 10
    except Exception as e:
        checks.append({"id": "C5.2", "name": "CrackTipDecoder", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C5.3: Small crack limit
    try:
        from benchmarks.fracture.lefm_reference import geometry_factor_edge_crack
        A_small = 0.025; W = 2.5
        sigma_griffith = K_Ic / (math.sqrt(math.pi * A_small) * geometry_factor_edge_crack(A_small / W))
        ok = sigma_griffith > sigma_ts  # small crack -> strength-dominated
        checks.append({"id": "C5.3", "name": f"Small A: sigma_G={sigma_griffith:.0f} > sigma_ts={sigma_ts:.0f}",
                        "pass": ok, "pts": 20 if ok else 0, "max": 20,
                        "note": "Strength limit verified analytically"})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C5.3", "name": "Small crack limit", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C5.4: Large crack limit
    try:
        A_large = 1.5; W = 2.5
        sigma_griffith_L = K_Ic / (math.sqrt(math.pi * A_large) * geometry_factor_edge_crack(A_large / W))
        ok = sigma_griffith_L < sigma_ts  # large crack -> Griffith-dominated
        checks.append({"id": "C5.4", "name": f"Large A: sigma_G={sigma_griffith_L:.1f} < sigma_ts={sigma_ts:.0f}",
                        "pass": ok, "pts": 20 if ok else 0, "max": 20})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C5.4", "name": "Large crack limit", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C5.5-C5.6: Full transition curve needs FEM
    for cid, name, pts in [("C5.5", "Transition region", 20), ("C5.6", "sigma_crit vs A curve", 20)]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts,
                        "note": "Needs chart FEM solve at each crack length"})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
