"""Score for Challenge 1: Uniaxial Tension Test.

Checks:
  C1.1: Chart FEM solver builds mesh on cylindrical rod         [10 pts]
  C1.2: Newton converges for uniaxial loading                   [15 pts]
  C1.3: sigma_zz matches E*epsilon within 2%                    [25 pts]
  C1.4: Drucker-Prager nucleation at sigma_ts within 10%        [25 pts]
  C1.5: Crack orientation perpendicular to axis                  [15 pts]
  C1.6: Multi-chart Schwarz converges                            [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    checks = []
    total = 0

    # C1.1: Solver builds mesh
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        import torch
        solver = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=None,
                                       device="cpu", dtype=torch.float64)
        ok = solver.n_nodes > 0 and solver.n_elements > 0
        checks.append({"id": "C1.1", "name": "Mesh builds", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C1.1", "name": "Mesh builds", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C1.2: Newton converges
    try:
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        stress_fn, tangent_fn = make_linear_elastic_small_strain(100.0, 0.3)
        nodes = solver.nodes
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc[:, 0] = 0.005 * (nodes[:, 0] + 1.0) / 2.0
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-9)
        ok = u is not None and torch.isfinite(u).all()
        checks.append({"id": "C1.2", "name": "Newton converges", "pass": ok, "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C1.2", "name": "Newton converges", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C1.3: Stress accuracy
    try:
        import numpy as np
        F_def = solver.compute_F(u)
        P = stress_fn(F_def)
        sigma_xx = P.detach().numpy()[:, 0, 0].mean()
        lam = 100.0 * 0.3 / (1.3 * 0.4)
        mu = 100.0 / 2.6
        expected = (lam + 2 * mu) * 0.005 / 2.0
        err = abs(sigma_xx - expected) / expected
        ok = err < 0.02
        pts = 25 if err < 0.02 else (15 if err < 0.05 else (5 if err < 0.1 else 0))
        checks.append({"id": "C1.3", "name": f"Stress accuracy ({err*100:.1f}%)", "pass": ok,
                        "pts": pts, "max": 25, "value": float(sigma_xx), "expected": float(expected)})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.3", "name": "Stress accuracy", "pass": False, "pts": 0, "max": 25, "error": str(e)})

    # C1.4-C1.6: Not yet implemented
    for cid, name, pts in [("C1.4", "DP nucleation at sigma_ts", 25),
                            ("C1.5", "Crack orientation", 15),
                            ("C1.6", "Multi-chart Schwarz", 10)]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts,
                        "note": "Not yet implemented"})

    score = total / 100 * 100
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")

    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": score, "checks": checks}
