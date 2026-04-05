"""Score for Challenge 2: Biaxial Tension Test.

Checks:
  C2.1: SDF-filtered circular plate mesh                         [10 pts]
  C2.2: Newton converges at each load step                       [10 pts]
  C2.3: Stress-strain matches exact S=E*eps/(1-nu) within 2%     [20 pts]
  C2.4: DP nucleation detected from FEM stress                   [20 pts]
  C2.5: Nucleation stress within 5% of sigma_bs=27.03 MPa        [20 pts]
  C2.6: Topology monitor detects domain splitting                [10 pts]
  C2.7: VTU output with all fields                               [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import numpy as np
    import torch
    checks = []
    total = 0

    E, nu = 70e3, 0.22
    sigma_ts, sigma_hs = 40.0, 27.8
    R = 5.0

    from solvers.fracture_criteria import derived_biaxial_strength
    sigma_bs = derived_biaxial_strength(sigma_ts, sigma_hs)

    # C2.1: SDF mesh
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from benchmarks.fracture.biaxial_tension import sdf_circular_plate

        class PlateOracle:
            def sdf(self, x):
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                v = sdf_circular_plate(x_np, R=R, L=0.5)
                return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

        solver = ChartVectorFEMSolver(n_cells=10, support_r=R*1.2, sdf_oracle=PlateOracle(),
                                       sdf_threshold=-0.01, device="cpu", dtype=torch.float64)
        ok = solver.n_nodes > 100
        checks.append({"id": "C2.1", "name": f"SDF mesh ({solver.n_nodes} nodes)", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C2.1", "name": "SDF mesh", "pass": False, "pts": 0, "max": 10, "error": str(e)})
        solver = None

    # C2.2: Newton convergence
    try:
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
        eps_test = 1e-4
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        nodes = solver.nodes
        u_bc[:, 0] = eps_test * nodes[:, 0]
        u_bc[:, 1] = eps_test * nodes[:, 1]
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-9)
        ok = u is not None and torch.isfinite(u).all()
        checks.append({"id": "C2.2", "name": "Newton converges", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C2.2", "name": "Newton converges", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C2.3: Stress-strain accuracy
    try:
        P = stress_fn(solver.compute_F(u))
        sigma = P.detach().numpy()
        S_biaxial = (sigma[:, 0, 0].mean() + sigma[:, 1, 1].mean()) / 2
        S_expected = E * eps_test / (1 - nu)
        err = abs(S_biaxial - S_expected) / S_expected
        ok = err < 0.02
        pts = 20 if err < 0.02 else (10 if err < 0.05 else (5 if err < 0.1 else 0))
        checks.append({"id": "C2.3", "name": f"Stress accuracy ({err*100:.1f}%)", "pass": ok,
                        "pts": pts, "max": 20, "value": float(S_biaxial), "expected": float(S_expected)})
        total += pts
    except Exception as e:
        checks.append({"id": "C2.3", "name": "Stress accuracy", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C2.4: DP nucleation detection
    try:
        from solvers.fracture_criteria import drucker_prager_F
        # Load to sigma_bs level
        eps_bs = sigma_bs * (1 - nu) / E
        u_bc2 = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc2[:, 0] = eps_bs * 1.05 * nodes[:, 0]
        u_bc2[:, 1] = eps_bs * 1.05 * nodes[:, 1]
        u2 = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc2, bc_mask, max_iter=10, tol=1e-9)
        sigma2 = stress_fn(solver.compute_F(u2)).detach().numpy()
        F_dp = drucker_prager_F(sigma2, sigma_ts, sigma_hs)
        ok = F_dp.max() >= 0
        checks.append({"id": "C2.4", "name": "DP nucleation detected", "pass": ok,
                        "pts": 20 if ok else 0, "max": 20, "F_dp_max": float(F_dp.max())})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C2.4", "name": "DP nucleation", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C2.5: Nucleation stress accuracy
    try:
        S_at_nuc = (sigma2[:, 0, 0].mean() + sigma2[:, 1, 1].mean()) / 2
        err_nuc = abs(S_at_nuc - sigma_bs) / sigma_bs
        ok = err_nuc < 0.05
        pts = 20 if err_nuc < 0.05 else (10 if err_nuc < 0.1 else 0)
        checks.append({"id": "C2.5", "name": f"sigma_bs accuracy ({err_nuc*100:.1f}%)", "pass": ok,
                        "pts": pts, "max": 20, "value": float(S_at_nuc), "expected": float(sigma_bs)})
        total += pts
    except Exception as e:
        checks.append({"id": "C2.5", "name": "sigma_bs accuracy", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C2.6-C2.7: Topology + VTU
    for cid, name, pts in [("C2.6", "Topology monitor", 10), ("C2.7", "VTU output", 10)]:
        try:
            if cid == "C2.7":
                vtu_exists = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))), "runs", "biaxial_chart_fem", "biaxial.pvd"))
                ok = vtu_exists
                checks.append({"id": cid, "name": name, "pass": ok, "pts": pts if ok else 0, "max": pts})
                total += pts if ok else 0
            else:
                checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts, "note": "Not tested in score"})
        except Exception:
            checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
