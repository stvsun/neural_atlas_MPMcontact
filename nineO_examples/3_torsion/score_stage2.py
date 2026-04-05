"""Stage 2 V&V tests for Challenge 3: Torsion.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C3.S2.1: Convergence order >= 1.8 (15 pts)
  C3.S2.2: TubeSectorDecoder Jacobian analytical vs AD match (15 pts)
  C3.S2.3: TubeSectorDecoder roundtrip error < 1e-10 (15 pts)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch, math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import TubeSectorDecoder

    checks = []
    total = 0
    E, nu = 70e3, 0.22
    r_mid = 2.925; t_wall = 0.15; L = 5.0
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # G1: Patch test
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = TubeSectorDecoder(theta_center=0.0, theta_span=math.pi / 1.5,
                                r_mid=r_mid, t_half=t_wall / 2,
                                z_center=L / 2, L_half=L / 2).double()
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

    # C3.S2.1: Convergence order (placeholder)
    checks.append({"id": "C3.S2.1", "name": "Convergence order (not yet impl)", "pass": False, "pts": 0, "max": 15})

    # C3.S2.2: TubeSectorDecoder Jacobian analytical vs AD (placeholder)
    checks.append({"id": "C3.S2.2", "name": "Jacobian analytical vs AD (not yet impl)", "pass": False, "pts": 0, "max": 15})

    # C3.S2.3: TubeSectorDecoder roundtrip
    try:
        xi = torch.rand(100, 3, dtype=torch.float64) * 2 - 1  # [-1, 1]^3
        x = dec(xi)
        xi_back = dec.inverse(x)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-10
        pts = 15 if ok else 0
        checks.append({"id": "C3.S2.3", "name": f"TubeSector roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C3.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
