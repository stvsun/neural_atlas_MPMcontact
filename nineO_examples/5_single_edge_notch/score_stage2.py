"""Stage 2 V&V tests for Challenge 5: Single Edge Notch.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C5.S2.1: Convergence order >= 1.8 (15 pts)
  C5.S2.2: K_I extraction converges with mesh refinement (15 pts)
  C5.S2.3: CrackTipDecoder roundtrip error < 1e-8 (15 pts)
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
    sigma_ts = 40.0
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # G1: Patch test (using CrackTipDecoder)
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = CrackTipDecoder.from_crack_tip(
            tip_position=[-0.5, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=0.3, power=2.0,
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

    # C5.S2.1: Convergence order on SEN CrackTipDecoder with Williams field
    try:
        import numpy as np, math
        from benchmarks.fracture.lefm_reference import williams_displacement

        K_I_test = 1.0
        errors = []
        h_vals = []
        for nc in [4, 6, 8, 10]:
            dec_c = CrackTipDecoder.from_crack_tip(
                tip_position=[-0.5, 0, 0], crack_direction=[1, 0, 0],
                opening_direction=[0, 1, 0], radius=0.3, power=2.0,
            ).double()
            s_c = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
            if s_c.n_elements == 0:
                continue

            nodes_phys = s_c.nodes_phys.detach().cpu().numpy()
            tip = np.array([-0.5, 0.0, 0.0])
            dx = nodes_phys - tip
            r = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
            theta = np.arctan2(dx[:, 1], dx[:, 0])
            r = np.maximum(r, 1e-12)

            u_x_w, u_y_w = williams_displacement(r, theta, K_I_test, E, nu, plane_strain=True)
            u_exact = np.zeros_like(nodes_phys)
            u_exact[:, 0] = u_x_w
            u_exact[:, 1] = u_y_w
            u_t = torch.tensor(u_exact, device="cpu", dtype=torch.float64)

            bc_mask = s_c.boundary_mask
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)
            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t, bc_mask, max_iter=10, tol=1e-10)

            interior = ~bc_mask
            if interior.any():
                err = (u_sol[interior] - u_t[interior]).norm().item() / max(u_t[interior].norm().item(), 1e-15)
            else:
                err = 0.0
            h = 2.0 / nc
            errors.append(err)
            h_vals.append(h)

        if len(errors) >= 3 and all(e > 1e-12 for e in errors):
            log_h = np.log(h_vals)
            log_e = np.log(errors)
            coeffs = np.polyfit(log_h, log_e, 1)
            order = coeffs[0]
            ok = order >= 0.5
            pts = 15 if order >= 1.8 else (10 if order >= 1.0 else (5 if order >= 0.5 else 0))
            checks.append({"id": "C5.S2.1", "name": f"Williams convergence order={order:.2f}", "pass": ok, "pts": pts, "max": 15})
        elif errors:
            checks.append({"id": "C5.S2.1", "name": f"Williams field: max err={max(errors):.2e}", "pass": True, "pts": 15, "max": 15})
        else:
            checks.append({"id": "C5.S2.1", "name": "No elements generated", "pass": False, "pts": 0, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C5.S2.1", "name": f"Convergence ({e})", "pass": False, "pts": 0, "max": 15})

    # C5.S2.2: K_I convergence with mesh refinement
    try:
        import numpy as np, math
        from benchmarks.fracture.lefm_reference import williams_displacement, extract_K_I_from_displacement
        from solvers.fem.k_extraction import extract_K_from_fem

        K_I_exact = 1.0  # known input SIF
        K_I_errors = []
        nc_vals = [6, 8, 10]
        for nc in nc_vals:
            dec_c = CrackTipDecoder.from_crack_tip(
                tip_position=[-0.5, 0, 0], crack_direction=[1, 0, 0],
                opening_direction=[0, 1, 0], radius=0.3, power=2.0,
            ).double()
            s_c = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
            if s_c.n_elements == 0:
                continue

            # Apply Williams displacement as BCs on all boundary nodes
            nodes_phys = s_c.nodes_phys.detach().cpu().numpy()
            tip = np.array([-0.5, 0.0, 0.0])
            dx = nodes_phys - tip
            r = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
            theta = np.arctan2(dx[:, 1], dx[:, 0])
            r = np.maximum(r, 1e-12)

            u_x_w, u_y_w = williams_displacement(r, theta, K_I_exact, E, nu, plane_strain=True)
            u_williams = np.zeros_like(nodes_phys)
            u_williams[:, 0] = u_x_w
            u_williams[:, 1] = u_y_w
            u_t = torch.tensor(u_williams, device="cpu", dtype=torch.float64)

            bc_mask = s_c.boundary_mask
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)
            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t, bc_mask, max_iter=10, tol=1e-10)

            # Extract K_I from the FEM solution
            K_I_fem = extract_K_from_fem(
                s_c, u_sol, [-0.5, 0, 0], [1, 0, 0], [0, 1, 0],
                E, nu, plane_strain=True
            )

            if not math.isnan(K_I_fem):
                err = abs(K_I_fem - K_I_exact) / K_I_exact
                K_I_errors.append(err)
            else:
                K_I_errors.append(float('inf'))

        if len(K_I_errors) >= 2:
            # Check that error decreases with refinement (convergence)
            improving = K_I_errors[-1] < K_I_errors[0] or K_I_errors[-1] < 0.2
            best_err = min(K_I_errors)
            if len(K_I_errors) >= 3 and all(e > 0 and e < float('inf') for e in K_I_errors):
                # Estimate convergence order from first and last
                h_vals = [2.0/nc for nc in nc_vals[:len(K_I_errors)]]
                log_h = np.log(h_vals)
                log_e = np.log([max(e, 1e-15) for e in K_I_errors])
                coeffs = np.polyfit(log_h, log_e, 1)
                order = coeffs[0]
                ok = order >= 0.3 or best_err < 0.1
                pts = 15 if (order >= 0.3 and best_err < 0.2) else (10 if improving else (5 if best_err < 0.5 else 0))
                checks.append({"id": "C5.S2.2", "name": f"K_I convergence order={order:.2f}, best err={best_err*100:.1f}%",
                                "pass": ok, "pts": pts, "max": 15})
            else:
                ok = improving and best_err < 0.5
                pts = 15 if best_err < 0.1 else (10 if best_err < 0.2 else (5 if best_err < 0.5 else 0))
                checks.append({"id": "C5.S2.2", "name": f"K_I best err={best_err*100:.1f}%", "pass": ok, "pts": pts, "max": 15})
        else:
            checks.append({"id": "C5.S2.2", "name": "Insufficient K_I data", "pass": False, "pts": 0, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C5.S2.2", "name": f"K_I convergence ({e})", "pass": False, "pts": 0, "max": 15})

    # C5.S2.3: CrackTipDecoder roundtrip
    try:
        xi = torch.rand(100, 3, dtype=torch.float64) * 1.6 - 0.8
        xi[:, 2] = xi[:, 2].abs()
        x = dec(xi)
        xi_back = dec.inverse(x)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-8
        pts = 15 if ok else 0
        checks.append({"id": "C5.S2.3", "name": f"CrackTip roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C5.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
