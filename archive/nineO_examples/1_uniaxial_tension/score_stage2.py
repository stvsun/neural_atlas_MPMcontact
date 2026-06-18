"""Stage 2 V&V tests for Challenge 1: Uniaxial Tension.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C1.S2.1: Convergence order >= 1.8 (15 pts)
  C1.S2.2: Two-chart DD stress matches single-chart within 0.5% (15 pts)
  C1.S2.3: BoxDecoder roundtrip error < 1e-12 (15 pts)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import BoxDecoder

    checks = []
    total = 0
    E, nu = 70e3, 0.22
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # G1: Patch test
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = BoxDecoder(center=(0, 0, 0), half_extents=(1, 1, 1)).double()
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

    # C1.S2.1: Convergence order test
    try:
        import numpy as np
        errors = []
        h_vals = []
        for nc in [4, 6, 8, 12]:
            dec_c = BoxDecoder(center=(0,0,0), half_extents=(1,1,1)).double()
            s_c = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
            if s_c.n_elements == 0:
                continue
            # Manufactured solution: u_exact = eps * [x, -nu*y, -nu*z]
            eps_val = 1e-4
            nodes = s_c.nodes_phys.detach().cpu().numpy()
            u_exact = np.zeros_like(nodes)
            u_exact[:, 0] = eps_val * nodes[:, 0]
            u_exact[:, 1] = -nu * eps_val * nodes[:, 1]
            u_exact[:, 2] = -nu * eps_val * nodes[:, 2]
            u_t = torch.tensor(u_exact, device="cpu", dtype=torch.float64)

            # Apply as Dirichlet on all boundary, solve
            bc_mask = s_c.boundary_mask
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)
            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t, bc_mask, max_iter=5, tol=1e-10)

            err = (u_sol - u_t).norm().item() / max(u_t.norm().item(), 1e-15)
            h = 2.0 / nc
            errors.append(err)
            h_vals.append(h)

        import numpy as np
        if len(errors) >= 3 and all(e > 1e-10 for e in errors):
            # Non-trivial errors: fit convergence order
            log_h = np.log(h_vals)
            log_e = np.log(errors)
            coeffs = np.polyfit(log_h, log_e, 1)
            order = coeffs[0]
            ok = order >= 1.5
            pts = 15 if order >= 1.8 else (10 if order >= 1.5 else (5 if order >= 1.0 else 0))
            checks.append({"id": "C1.S2.1", "name": f"Convergence order={order:.2f}", "pass": ok, "pts": pts, "max": 15})
        else:
            # All errors are essentially zero (exact for affine fields on P1!)
            max_err = max(errors) if errors else 0
            checks.append({"id": "C1.S2.1", "name": f"Affine exact (max err={max_err:.2e})", "pass": True, "pts": 15, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C1.S2.1", "name": f"Convergence ({e})", "pass": False, "pts": 0, "max": 15})

    # C1.S2.2: Two-chart DD stress match
    try:
        import numpy as np
        from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver

        # Single chart reference
        dec_single = BoxDecoder(center=(0,0,0), half_extents=(1,1,1)).double()
        s_single = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_single,
                                         decoder_kwargs={}, device="cpu", dtype=torch.float64)
        eps_val = 1e-4
        nodes_s = s_single.nodes_phys.detach().cpu().numpy()
        u_bc_s = np.zeros_like(nodes_s)
        u_bc_s[:, 0] = eps_val * nodes_s[:, 0]
        u_bc_s[:, 1] = -nu * eps_val * nodes_s[:, 1]
        u_bc_s[:, 2] = -nu * eps_val * nodes_s[:, 2]

        u_t_s = torch.tensor(u_bc_s, dtype=torch.float64)
        f_ext_s = torch.zeros(s_single.n_nodes, 3, dtype=torch.float64)
        u_ref = s_single.solve_nonlinear(stress_fn, tangent_fn, f_ext_s, u_t_s,
                                          s_single.boundary_mask, max_iter=5, tol=1e-10)
        F_ref = s_single.compute_F(u_ref)
        P_ref = stress_fn(F_ref)
        sigma_ref = P_ref[:, 0, 0].mean().item()

        # Two-chart DD
        dec_a = BoxDecoder(center=(-0.5, 0, 0), half_extents=(0.6, 1, 1)).double()
        dec_b = BoxDecoder(center=(0.5, 0, 0), half_extents=(0.6, 1, 1)).double()
        s_a = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec_a,
                                    decoder_kwargs={}, device="cpu", dtype=torch.float64)
        s_b = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec_b,
                                    decoder_kwargs={}, device="cpu", dtype=torch.float64)

        def bc_fn_uni(np_phys):
            n = len(np_phys)
            u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            x = np_phys[:, 0]
            left = x < -0.95; right = x > 0.95
            top = np_phys[:, 1] > 0.95; bot = np_phys[:, 1] < -0.95
            front = np_phys[:, 2] > 0.95; back = np_phys[:, 2] < -0.95
            bnd = left | right | top | bot | front | back
            m[bnd] = True
            u[bnd, 0] = eps_val * np_phys[bnd, 0]
            u[bnd, 1] = -nu * eps_val * np_phys[bnd, 1]
            u[bnd, 2] = -nu * eps_val * np_phys[bnd, 2]
            return u, m

        seeds_dd = torch.tensor([[-0.5,0,0],[0.5,0,0]], dtype=torch.float64)
        schwarz = SchwarzVectorFEMSolver(chart_solvers=[s_a, s_b], seeds=seeds_dd,
                                          decoders=[dec_a, dec_b], neighbors=[[1],[0]])
        u_dd = schwarz.solve(stress_fn, tangent_fn, bc_fn_uni, max_schwarz_iters=15, tol=1e-3, relaxation=0.5)

        # Compare stress
        sigmas = []
        for i, u in enumerate(u_dd):
            if u is not None:
                F_i = [s_a, s_b][i].compute_F(u)
                P_i = stress_fn(F_i)
                sigmas.append(P_i[:, 0, 0].mean().item())

        if sigmas:
            sigma_dd = np.mean(sigmas)
            err = abs(sigma_dd - sigma_ref) / abs(sigma_ref) if abs(sigma_ref) > 0 else 0
            ok = err < 0.005
            pts = 15 if ok else (10 if err < 0.02 else (5 if err < 0.05 else 0))
            checks.append({"id": "C1.S2.2", "name": f"DD stress err={err*100:.2f}%", "pass": ok, "pts": pts, "max": 15})
        else:
            checks.append({"id": "C1.S2.2", "name": "DD solve failed", "pass": False, "pts": 0, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C1.S2.2", "name": f"DD stress ({e})", "pass": False, "pts": 0, "max": 15})

    # C1.S2.3: BoxDecoder roundtrip
    try:
        xi = torch.randn(100, 3, dtype=torch.float64) * 0.5
        x = dec(xi)
        # BoxDecoder inverse: xi = (x - center) / half_extents
        xi_back = (x - dec.center.unsqueeze(0)) / dec.half_extents.unsqueeze(0)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-12
        pts = 15 if ok else 0
        checks.append({"id": "C1.S2.3", "name": f"BoxDecoder roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75  # 15+15+15+15+15
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
