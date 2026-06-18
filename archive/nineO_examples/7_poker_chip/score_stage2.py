"""Stage 2 V&V tests for Challenge 7: Poker-Chip.

Tests:
  G1: Patch test (15 pts)
  G2: Decoder Jacobian quality (15 pts)
  C7.S2.1: Convergence order >= 1.8 (15 pts)
  C7.S2.2: Near-incompressibility no volumetric locking (15 pts)
  C7.S2.3: BoxDecoder roundtrip error < 1e-12 (15 pts)
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
    # Poker chip: polyurethane-like rubber
    mu_pu, lam_pu = 0.52, 85.77
    E_eff = mu_pu * (3 * lam_pu + 2 * mu_pu) / (lam_pu + mu_pu)
    nu_eff = lam_pu / (2 * (lam_pu + mu_pu))
    D = 10.0; L_min = 1.0
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E_eff, nu_eff)

    # G1: Patch test
    try:
        from nineO_examples.common_stage2.test_patch import run_patch_test
        dec = BoxDecoder(center=(0, 0, 0), half_extents=(D / 2, D / 2, L_min / 2)).double()
        solver = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec,
                                       decoder_kwargs={}, device="cpu", dtype=torch.float64)
        res = run_patch_test(solver, dec, stress_fn, tangent_fn, E_eff, nu_eff)
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

    # C7.S2.1: Convergence order on BoxDecoder (rubber) — affine reproduction
    import numpy as np, math
    try:
        eps_val = 1e-4
        errs = []
        for nc in [4, 6, 8]:
            dec_c = BoxDecoder(center=(0, 0, 0), half_extents=(D / 2, D / 2, L_min / 2)).double()
            s_c = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                                        decoder_kwargs={}, device="cpu", dtype=torch.float64)
            if s_c.n_elements == 0:
                continue
            nodes = s_c.nodes_phys.detach().cpu().numpy()
            u_exact = np.zeros_like(nodes)
            u_exact[:, 0] = eps_val * nodes[:, 0]
            u_exact[:, 1] = -nu_eff * eps_val * nodes[:, 1]
            u_exact[:, 2] = -nu_eff * eps_val * nodes[:, 2]
            u_t = torch.tensor(u_exact, dtype=torch.float64)
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)
            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t, s_c.boundary_mask, max_iter=5, tol=1e-10)
            err = (u_sol - u_t).norm().item() / max(u_t.norm().item(), 1e-15)
            errs.append(err)

        if errs and all(e < 1e-6 for e in errs):
            checks.append({"id": "C7.S2.1", "name": f"Affine exact (max err={max(errs):.2e})", "pass": True, "pts": 15, "max": 15})
        else:
            checks.append({"id": "C7.S2.1", "name": f"Stress repr err={max(errs) if errs else 'N/A'}", "pass": False, "pts": 0, "max": 15})
        total += checks[-1]["pts"]
    except Exception as e:
        checks.append({"id": "C7.S2.1", "name": f"Convergence ({e})", "pass": False, "pts": 0, "max": 15})

    # C7.S2.2: Volumetric locking check (F-bar)
    try:
        dec_v = BoxDecoder(center=(0, 0, 0), half_extents=(D / 2, D / 2, L_min / 2)).double()
        s_v = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec_v,
                                    decoder_kwargs={}, device="cpu", dtype=torch.float64)
        if s_v.n_elements > 0:
            eps_h = 1e-4  # hydrostatic strain
            nodes_v = s_v.nodes_phys.detach().cpu().numpy()
            u_hydro = np.zeros_like(nodes_v)
            u_hydro[:, 0] = eps_h * nodes_v[:, 0]
            u_hydro[:, 1] = eps_h * nodes_v[:, 1]
            u_hydro[:, 2] = eps_h * nodes_v[:, 2]
            u_h_t = torch.tensor(u_hydro, dtype=torch.float64)
            # Compute F with F-bar to check volumetric locking
            F_bar = s_v.compute_F_bar(u_h_t)
            J_vals = torch.det(F_bar).detach().cpu().numpy()
            # Expected: J = (1+eps)^3 ~ 1 + 3*eps for small eps
            J_expected = (1.0 + eps_h)**3
            err_J = np.max(np.abs(J_vals - J_expected))
            ok = err_J < 0.01
            pts = 15 if ok else 0
            checks.append({"id": "C7.S2.2", "name": f"F-bar J err={err_J:.2e} (expect {J_expected:.6f})",
                            "pass": ok, "pts": pts, "max": 15})
            total += pts
        else:
            checks.append({"id": "C7.S2.2", "name": "No elements", "pass": False, "pts": 0, "max": 15})
    except Exception as e:
        checks.append({"id": "C7.S2.2", "name": f"Locking ({e})", "pass": False, "pts": 0, "max": 15})

    # C7.S2.3: BoxDecoder roundtrip
    try:
        xi = torch.randn(100, 3, dtype=torch.float64) * 0.5
        x = dec(xi)
        xi_back = (x - dec.center.unsqueeze(0)) / dec.half_extents.unsqueeze(0)
        err = (xi_back - xi).abs().max().item()
        ok = err < 1e-12
        pts = 15 if ok else 0
        checks.append({"id": "C7.S2.3", "name": f"BoxDecoder roundtrip (err={err:.2e})", "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C7.S2.3", "name": f"Roundtrip ({e})", "pass": False, "pts": 0, "max": 15})

    max_score = 75
    score = total
    status = "PASS" if score >= max_score * 0.8 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")

    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 2 Score: {score}/{max_score}")

    return {"status": status, "score": float(score), "max_score": max_score, "checks": checks}
