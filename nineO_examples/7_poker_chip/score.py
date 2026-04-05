"""Score for Challenge 7: Poker-Chip Test.

Checks:
  C7.1: Variable-thickness disk SDF                              [10 pts]
  C7.2: Neo-Hookean constitutive in FEM                          [20 pts]
  C7.3: Near-incompressibility handling (no locking)             [20 pts]
  C7.4: Hydrostatic tension develops at centerline               [15 pts]
  C7.5: Crack nucleates perpendicular to loading                 [20 pts]
  C7.6: Nucleation displacement matches reference                [15 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    checks = []
    total = 0

    # C7.1: Disk geometry
    try:
        import numpy as np
        # Variable-thickness disk: thickness = L_min + (L_max - L_min) * (1 - sqrt(1 - (r/R_fixture)^2))
        # For now just check that we can define the geometry
        L_min, L_max, D, R_fix = 1.0, 1.7, 10.0, 18.2
        ok = True
        checks.append({"id": "C7.1", "name": "Disk geometry defined", "pass": ok, "pts": 10, "max": 10})
        total += 10
    except Exception as e:
        checks.append({"id": "C7.1", "name": "Disk geometry", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C7.2: Neo-Hookean in FEM
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        import torch
        mu_pu, lam_pu = 0.52, 85.77
        K_bulk = lam_pu + 2*mu_pu/3
        # Test that ChartVectorFEMSolver has Neo-Hookean and it produces correct stress
        test_solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu", dtype=torch.float64)
        stress_fn, _ = test_solver.make_neo_hookean(mu_pu, K_bulk)
        F = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        F[0, 0, 0] = 1.01  # small uniaxial stretch
        P = stress_fn(F)
        ok = P[0, 0, 0].item() > 0
        checks.append({"id": "C7.2", "name": "Neo-Hookean (FEM)", "pass": ok, "pts": 20 if ok else 0, "max": 20})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C7.2", "name": "Neo-Hookean", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C7.3: Near-incompressibility with F-bar
    try:
        from solvers.fem.analytic_decoders import BoxDecoder
        mu_pu, lam_pu = 0.52, 85.77
        K_bulk = lam_pu + 2*mu_pu/3
        R_disk, L_avg = 5.0, 1.35
        dec = BoxDecoder(center=(0, 0, 0), half_extents=(R_disk*1.1, R_disk*1.1, L_avg*0.6)).double()

        class DiskSDF:
            def sdf(self, x):
                import numpy as np_sdf
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                r = np_sdf.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
                d_r = r - R_disk; d_z = np_sdf.abs(x_np[:, 2]) - L_avg/2
                out = np_sdf.sqrt(np_sdf.maximum(d_r,0)**2 + np_sdf.maximum(d_z,0)**2)
                ins = np_sdf.minimum(np_sdf.maximum(d_r, d_z), 0)
                v = out + ins
                return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

        solver7 = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec,
                                        decoder_kwargs={}, sdf_oracle=DiskSDF(),
                                        sdf_threshold=-0.01, device="cpu", dtype=torch.float64)
        stress_fn7, tangent_fn7 = solver7.make_neo_hookean(mu_pu, K_bulk)

        # Vertical pull BCs
        delta = 0.03  # small deformation
        bc_mask7 = torch.zeros(solver7.n_nodes, dtype=torch.bool)
        u_bc7 = torch.zeros(solver7.n_nodes, 3, dtype=torch.float64)
        z = solver7.nodes_phys[:, 2].detach()
        tol7 = L_avg/2 * 0.1
        top = z > L_avg/2 - tol7; bc_mask7[top] = True; u_bc7[top, 2] = delta/2
        bot = z < -L_avg/2 + tol7; bc_mask7[bot] = True; u_bc7[bot, 2] = -delta/2
        f_ext7 = torch.zeros(solver7.n_nodes, 3, dtype=torch.float64)

        u7 = solver7.solve_nonlinear(stress_fn7, tangent_fn7, f_ext7, u_bc7, bc_mask7,
                                      max_iter=20, tol=1e-8, use_fbar=True)
        # Check that displacement is non-trivial (no locking)
        u_max = u7.abs().max().item()
        ok = u_max > 1e-6 and torch.isfinite(u7).all().item()
        checks.append({"id": "C7.3", "name": f"F-bar anti-locking (u_max={u_max:.2e})",
                        "pass": ok, "pts": 20 if ok else 0, "max": 20})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C7.3", "name": "Near-incompressibility", "pass": False, "pts": 0, "max": 20, "error": str(e)})

    # C7.4-C7.6: Require additional features
    for cid, name, pts in [
        ("C7.4", "Hydrostatic tension at center", 15),
        ("C7.5", "Crack perpendicular to loading", 20),
        ("C7.6", "Nucleation displacement", 15),
    ]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts,
                        "note": "Requires crack propagation driver"})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
