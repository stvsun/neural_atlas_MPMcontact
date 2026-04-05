"""Stage 3 XFEM-critic tests for Challenge 9: Trousers Tear (Mode III).

Has crack (Mode III) -- limited fracture test subset:
  X3: Mixed-mode extraction (10 pts)
  X4: Curved crack path (10 pts)
  X5: Displacement discontinuity handling (10 pts)
  X6: Stiffness conditioning (10 pts)

Max score: 40
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    import numpy as np
    import math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import BoxDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_mixed_mode_extraction,
        test_curved_crack_path,
        test_displacement_discontinuity_handling,
        test_stiffness_conditioning,
    )

    checks = []
    total = 0

    # Material: PU elastomer
    mu_pu, lam_pu = 0.52, 85.77
    E_eff = mu_pu * (3 * lam_pu + 2 * mu_pu) / (lam_pu + mu_pu)
    nu_eff = lam_pu / (2 * (lam_pu + mu_pu))
    Gc = 0.041
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E_eff, nu_eff)

    # Trousers geometry: crack at center of specimen
    crack_tip = [0, 0, 0]
    crack_dir = [0, 1, 0]   # crack propagates along y
    opening_dir = [1, 0, 0]  # Mode III: out-of-plane tearing

    # Build solver with BoxDecoder for trousers specimen
    dec = BoxDecoder(center=(0, 0, 0), half_extents=(50, 20, 0.5)).double()
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=dec,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )

    # --- X3: Mixed-mode extraction (10 pts) ---
    try:
        res = test_mixed_mode_extraction(solver, E_eff, nu_eff, crack_tip, crack_dir, opening_dir)
        for c in res["checks"]:
            c["id"] = c.get("id", "X3")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X3", "name": f"Mixed-mode extraction ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X4: Curved crack path (10 pts) ---
    try:
        res = test_curved_crack_path()
        for c in res["checks"]:
            c["id"] = c.get("id", "X4")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X4", "name": f"Curved crack path ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X5: Displacement discontinuity handling (10 pts) ---
    try:
        res = test_displacement_discontinuity_handling()
        for c in res["checks"]:
            c["id"] = c.get("id", "X5")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X5", "name": f"Displacement discontinuity ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X6: Stiffness conditioning (10 pts) ---
    try:
        res = test_stiffness_conditioning(solver, stress_fn, tangent_fn)
        for c in res["checks"]:
            c["id"] = c.get("id", "X6")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X6", "name": f"Stiffness conditioning ({e})",
                        "pass": False, "pts": 0, "max": 10})

    max_score = 40
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
