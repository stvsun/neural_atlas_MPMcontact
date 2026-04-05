"""Stage 2 V&V tests for Challenge 4: Pure-Shear Fracture.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C4.S2.1: Convergence order >= 1.8 (15 pts)
  C4.S2.2: CrackTipDecoder radial mesh concentration (15 pts)
  C4.S2.3: CrackTipDecoder roundtrip error < 1e-8 (15 pts)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import CrackTipDecoder

    checks = []
    total = 0
    E, nu, Gc = 70e3, 0.22, 0.01
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # G1: Patch test (using CrackTipDecoder)
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = CrackTipDecoder.from_crack_tip(
            tip_position=[-15, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=0.5, power=2.0,
        ).double()
        solver = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec,
                                       decoder_kwargs={}, device="cpu", dtype=torch.float64)
        res = run_patch_test(solver, dec, stress_fn, tangent_fn, E, nu)
        for c in res["checks"]:
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "G1", "name": f"Patch test ({e})", "pass": False, "pts": 0, "max": 15})

    # G2: Decoder quality
    try:
        from nineO_examples.common_stage2.test_decoder_quality import run_decoder_quality_test
        res = run_decoder_quality_test(solver, dec)
        for c in res["checks"]:
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "G2", "name": f"Decoder quality ({e})", "pass": False, "pts": 0, "max": 15})

    # C4.S2.1: Convergence order (placeholder)
    checks.append({"id": "C4.S2.1", "name": "Convergence order (not yet impl)", "pass": False, "pts": 0, "max": 15})

    # C4.S2.2: CrackTipDecoder radial mesh concentration (placeholder)
    checks.append({"id": "C4.S2.2", "name": "Radial mesh concentration (not yet impl)", "pass": False, "pts": 0, "max": 15})

    # C4.S2.3: CrackTipDecoder roundtrip
    try:
        # Sample points away from crack tip (avoid singularity at xi_2=-1)
        xi = torch.rand(100, 3, dtype=torch.float64) * 1.6 - 0.8  # [-0.8, 0.8]
        xi[:, 2] = xi[:, 2].abs()  # positive xi_2 to stay away from tip
        x = dec(xi)
        xi_back = dec.inverse(x)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-8
        pts = 15 if ok else 0
        checks.append({"id": "C4.S2.3", "name": f"CrackTip roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C4.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
