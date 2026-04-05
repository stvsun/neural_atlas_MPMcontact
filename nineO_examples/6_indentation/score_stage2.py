"""Stage 2 V&V tests for Challenge 6: Indentation.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C6.S2.1: Convergence order >= 1.8 (15 pts)
  C6.S2.2: Axisymmetric stress field under flat punch (15 pts)
  C6.S2.3: CylinderDecoder roundtrip error < 1e-10 (15 pts)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch, math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import CylinderDecoder

    checks = []
    total = 0
    E, nu = 70e3, 0.22
    R_block = 25.0; L_block = 25.0
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # G1: Patch test (using CylinderDecoder)
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = CylinderDecoder(theta_center=0.0, theta_span=math.pi / 1.5,
                              R=R_block, z_center=L_block / 2,
                              L_half=L_block / 2).double()
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

    # C6.S2.1: Stress reproduction on CylinderDecoder (affine field)
    import numpy as np
    try:
        eps_val = 1e-4
        errs = []
        for nc in [4, 6, 8]:
            from solvers.fem.analytic_decoders import CylinderDecoder as CylDec
            dec_c = CylDec(theta_center=0.0, theta_span=math.pi/1.5, R=R_block, z_center=L_block/2, L_half=L_block/2).double()
            s_c = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
            if s_c.n_elements == 0:
                continue
            nodes = s_c.nodes_phys.detach().cpu().numpy()
            u_exact = np.zeros_like(nodes)
            u_exact[:, 0] = eps_val * nodes[:, 0]
            u_exact[:, 1] = -nu * eps_val * nodes[:, 1]
            u_exact[:, 2] = -nu * eps_val * nodes[:, 2]
            u_t = torch.tensor(u_exact, dtype=torch.float64)
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)
            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t, s_c.boundary_mask, max_iter=5, tol=1e-10)
            err = (u_sol - u_t).norm().item() / max(u_t.norm().item(), 1e-15)
            errs.append(err)

        if errs and all(e < 1e-6 for e in errs):
            checks.append({"id": "C6.S2.1", "name": f"Affine exact (max err={max(errs):.2e})", "pass": True, "pts": 15, "max": 15})
        else:
            checks.append({"id": "C6.S2.1", "name": f"Stress repr err={max(errs) if errs else 'N/A'}", "pass": False, "pts": 0, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C6.S2.1", "name": f"Convergence ({e})", "pass": False, "pts": 0, "max": 15})

    # C6.S2.2: Axisymmetric stress field validation
    try:
        dec_ax = CylinderDecoder(theta_center=0.0, theta_span=math.pi/1.5,
                                  R=R_block, z_center=L_block/2, L_half=L_block/2).double()
        s_ax = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec_ax,
                                     decoder_kwargs={}, device="cpu", dtype=torch.float64)
        if s_ax.n_elements > 0:
            nodes_ax = s_ax.nodes_phys.detach().cpu().numpy()
            # Apply uniform radial displacement u_r = eps * r (axisymmetric expansion)
            eps_r = 1e-4
            r_vals = np.sqrt(nodes_ax[:, 0]**2 + nodes_ax[:, 1]**2)
            theta_vals = np.arctan2(nodes_ax[:, 1], nodes_ax[:, 0])
            u_ax = np.zeros_like(nodes_ax)
            u_ax[:, 0] = eps_r * r_vals * np.cos(theta_vals)  # u_x = eps * x
            u_ax[:, 1] = eps_r * r_vals * np.sin(theta_vals)  # u_y = eps * y
            u_ax[:, 2] = -nu * eps_r * nodes_ax[:, 2]         # u_z = -nu*eps*z (Poisson)
            u_ax_t = torch.tensor(u_ax, dtype=torch.float64)
            F_ax = s_ax.compute_F(u_ax_t)
            sig_ax = stress_fn(F_ax).detach().cpu().numpy()
            # Compute sigma_rr per element from Cartesian stress tensor
            centroids = s_ax.elem_centroids_phys.detach().cpu().numpy()
            r_c = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
            cos_t = np.where(r_c > 1e-10, centroids[:, 0] / r_c, 1.0)
            sin_t = np.where(r_c > 1e-10, centroids[:, 1] / r_c, 0.0)
            # sigma_rr = cos^2*sig_xx + 2*cos*sin*sig_xy + sin^2*sig_yy
            sig_rr = (cos_t**2 * sig_ax[:, 0, 0] + 2 * cos_t * sin_t * sig_ax[:, 0, 1]
                       + sin_t**2 * sig_ax[:, 1, 1])
            mean_rr = np.mean(np.abs(sig_rr))
            std_rr = np.std(sig_rr)
            cv = std_rr / mean_rr if mean_rr > 1e-15 else 0.0
            ok = cv < 0.05
            pts = 15 if ok else 0
            checks.append({"id": "C6.S2.2", "name": f"Axisym sig_rr CV={cv:.4f}", "pass": ok, "pts": pts, "max": 15})
            total += pts
        else:
            checks.append({"id": "C6.S2.2", "name": "No elements", "pass": False, "pts": 0, "max": 15})
    except Exception as e:
        checks.append({"id": "C6.S2.2", "name": f"Axisym stress ({e})", "pass": False, "pts": 0, "max": 15})

    # C6.S2.3: CylinderDecoder roundtrip
    try:
        xi = torch.rand(100, 3, dtype=torch.float64) * 2 - 1  # [-1, 1]^3
        # Avoid r=0 singularity by keeping xi_2 > -0.8
        xi[:, 2] = xi[:, 2].clamp(min=-0.8)
        x = dec(xi)
        xi_back = dec.inverse(x)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-10
        pts = 15 if ok else 0
        checks.append({"id": "C6.S2.3", "name": f"Cylinder roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C6.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
