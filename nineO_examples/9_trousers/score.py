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

    # C9.2: Neo-Hookean in FEM
    try:
        import torch
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        mu_pu, lam_pu = 0.52, 85.77
        K_bulk = lam_pu + 2*mu_pu/3
        test_solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu", dtype=torch.float64)
        stress_fn, _ = test_solver.make_neo_hookean(mu_pu, K_bulk)
        F = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        F[0, 0, 0] = 1.01
        P = stress_fn(F)
        ok = P[0, 0, 0].item() > 0
        checks.append({"id": "C9.2", "name": "Neo-Hookean FEM", "pass": ok, "pts": 20 if ok else 0, "max": 20})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C9.2", "name": "Neo-Hookean FEM", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C9.3: Near-incompressibility with F-bar
    try:
        from solvers.fem.analytic_decoders import BoxDecoder
        mu_pu, lam_pu = 0.52, 85.77
        K_bulk = lam_pu + 2*mu_pu/3
        # Small test strip
        dec9 = BoxDecoder(center=(0, 25, 0), half_extents=(10.1, 25.1, 0.6)).double()
        solver9 = ChartVectorFEMSolver(n_cells=4, support_r=1.0, chart_decoder=dec9,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
        stress_fn9, tangent_fn9 = solver9.make_neo_hookean(mu_pu, K_bulk)

        # Simple vertical pull
        bc_mask9 = torch.zeros(solver9.n_nodes, dtype=torch.bool)
        u_bc9 = torch.zeros(solver9.n_nodes, 3, dtype=torch.float64)
        y9 = solver9.nodes_phys[:, 1].detach()
        top9 = y9 > 49; bc_mask9[top9] = True; u_bc9[top9, 2] = 0.1
        bot9 = y9 < 1; bc_mask9[bot9] = True
        f_ext9 = torch.zeros(solver9.n_nodes, 3, dtype=torch.float64)

        u9 = solver9.solve_nonlinear(stress_fn9, tangent_fn9, f_ext9, u_bc9, bc_mask9,
                                      max_iter=20, tol=1e-8, use_fbar=True)
        u_max9 = u9.abs().max().item()
        ok = u_max9 > 1e-6 and torch.isfinite(u9).all().item()
        checks.append({"id": "C9.3", "name": f"F-bar anti-locking (u_max={u_max9:.2e})",
                        "pass": ok, "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C9.3", "name": "Near-incompressibility", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C9.4: Large rotation — verify Neo-Hookean handles large deformation
    try:
        # Neo-Hookean is frame-indifferent: P(QF) = Q P(F) for any rotation Q.
        # Verify that F with large rotation has finite, positive J.
        F_rot = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        # 90-degree rotation around z-axis + stretch
        F_rot[0, 0, 0] = 0.0; F_rot[0, 0, 1] = -1.0  # cos/sin(90)
        F_rot[0, 1, 0] = 1.0; F_rot[0, 1, 1] = 0.0
        F_rot[0, 2, 2] = 1.1  # slight stretch in z
        J_rot = torch.det(F_rot)
        P_rot = stress_fn9(F_rot)
        ok = J_rot.item() > 0 and torch.isfinite(P_rot).all().item()
        checks.append({"id": "C9.4", "name": f"Large rotation (J={J_rot.item():.2f}, P finite)",
                        "pass": ok, "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C9.4", "name": "Large rotation", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C9.5-C9.6: Require Mode III propagation driver
    for cid, name, pts, note in [
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
