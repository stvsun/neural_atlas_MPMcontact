"""Score for Challenge 1: Uniaxial Tension Test.

Checks:
  C1.1: Chart FEM solver builds mesh on cylindrical rod         [10 pts]
  C1.2: Newton converges for uniaxial loading                   [15 pts]
  C1.3: sigma_zz matches E*epsilon within 2%                    [25 pts]
  C1.4: Drucker-Prager nucleation at sigma_ts within 10%        [25 pts]
  C1.5: Crack orientation perpendicular to axis                  [15 pts]
  C1.6: Multi-chart Robin DD converges                           [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    checks = []
    total = 0

    # C1.1: Solver builds mesh with BoxDecoder + SDF
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.analytic_decoders import BoxDecoder
        import torch, numpy as np

        R = 2.0; L = 15.0

        class RodSDF:
            def sdf(self, x):
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
                d_radial = r - R
                d_axial = np.maximum(-x_np[:, 2], x_np[:, 2] - L)
                outside = np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_axial, 0)**2)
                inside = np.minimum(np.maximum(d_radial, d_axial), 0)
                vals = outside + inside
                if isinstance(x, torch.Tensor):
                    return torch.tensor(vals, dtype=x.dtype, device=x.device)
                return vals

        sdf = RodSDF()
        dec = BoxDecoder(center=(0, 0, L/2), half_extents=(R*1.2, R*1.2, L/2)).double()
        solver = ChartVectorFEMSolver(
            n_cells=8, support_r=1.0, chart_decoder=dec, decoder_kwargs={},
            sdf_oracle=sdf, sdf_threshold=-0.01,
            device="cpu", dtype=torch.float64,
        )
        ok = solver.n_nodes > 0 and solver.n_elements > 0
        checks.append({"id": "C1.1", "name": f"Mesh builds ({solver.n_nodes} nodes)", "pass": ok,
                        "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C1.1", "name": "Mesh builds", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C1.2: Newton converges for uniaxial loading
    try:
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        E = 70e3; nu = 0.22
        stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

        eps = 1e-4
        nodes_phys = solver.nodes_phys.detach().cpu().numpy()
        u_bc = np.zeros((solver.n_nodes, 3))
        bc_mask = np.ones(solver.n_nodes, dtype=bool)
        u_bc[:, 0] = -nu * eps * nodes_phys[:, 0]
        u_bc[:, 1] = -nu * eps * nodes_phys[:, 1]
        u_bc[:, 2] = eps * nodes_phys[:, 2]

        u_bc_t = torch.tensor(u_bc, dtype=torch.float64)
        bc_mask_t = torch.tensor(bc_mask, dtype=torch.bool)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc_t, bc_mask_t,
                                    max_iter=10, tol=1e-9)
        ok = u is not None and torch.isfinite(u).all()
        checks.append({"id": "C1.2", "name": "Newton converges", "pass": ok,
                        "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C1.2", "name": "Newton converges", "pass": False,
                        "pts": 0, "max": 15, "error": str(e)})

    # C1.3: Stress accuracy — sigma_zz = E * eps
    try:
        F_def = solver.compute_F(u)
        P = stress_fn(F_def)
        sigma_zz = P.detach().numpy()[:, 2, 2].mean()
        expected = E * eps
        err = abs(sigma_zz - expected) / expected
        ok = err < 0.02
        pts = 25 if err < 0.02 else (15 if err < 0.05 else (5 if err < 0.1 else 0))
        checks.append({"id": "C1.3", "name": f"Stress accuracy ({err*100:.1f}%)",
                        "pass": ok, "pts": pts, "max": 25,
                        "value": float(sigma_zz), "expected": float(expected)})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.3", "name": "Stress accuracy", "pass": False,
                        "pts": 0, "max": 25, "error": str(e)})

    # C1.4: Drucker-Prager nucleation at sigma_ts within 10%
    try:
        from solvers.fracture_criteria import drucker_prager_F
        sigma_ts = 40.0; sigma_hs = 27.8

        # Test: at sigma_ts, F should be ~0
        sigma_test = np.zeros((1, 3, 3))
        sigma_test[0, 2, 2] = sigma_ts
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]
        ok_at_ts = abs(F_dp) < 1.0  # should be near zero

        # Run incremental to find nucleation
        eps_ts = sigma_ts / E
        n_steps = 10
        eps_vals = np.linspace(eps_ts * 0.5, eps_ts * 1.2, n_steps)
        nuc_sigma = None

        for eps_i in eps_vals:
            u_bc_i = np.zeros((solver.n_nodes, 3))
            u_bc_i[:, 0] = -nu * eps_i * nodes_phys[:, 0]
            u_bc_i[:, 1] = -nu * eps_i * nodes_phys[:, 1]
            u_bc_i[:, 2] = eps_i * nodes_phys[:, 2]
            u_bc_t_i = torch.tensor(u_bc_i, dtype=torch.float64)

            u_i = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc_t_i, bc_mask_t,
                                          max_iter=10, tol=1e-9)
            F_i = solver.compute_F(u_i)
            P_i = stress_fn(F_i)
            szz_i = P_i.detach().numpy()[:, 2, 2].mean()

            sig_i = np.zeros((1, 3, 3))
            sig_i[0, 2, 2] = szz_i
            F_dp_i = drucker_prager_F(sig_i, sigma_ts, sigma_hs)[0]
            if F_dp_i >= 0:
                nuc_sigma = szz_i
                break

        if nuc_sigma is not None:
            err_nuc = abs(nuc_sigma - sigma_ts) / sigma_ts
            ok = err_nuc < 0.10
            pts = 25 if err_nuc < 0.05 else (15 if err_nuc < 0.10 else 5)
        else:
            ok = False; pts = 0; err_nuc = 999

        checks.append({"id": "C1.4", "name": f"DP nucleation ({err_nuc*100:.1f}%)",
                        "pass": ok, "pts": pts, "max": 25})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.4", "name": "DP nucleation", "pass": False,
                        "pts": 0, "max": 25, "error": str(e)})

    # C1.5: Crack orientation perpendicular to loading axis
    try:
        from solvers.fracture_criteria import crack_normal_from_stress
        sigma_uniaxial = np.zeros((3, 3))
        sigma_uniaxial[2, 2] = sigma_ts
        normal = crack_normal_from_stress(sigma_uniaxial)
        # Normal should be parallel to z-axis (loading direction)
        dot_z = abs(normal[2])
        ok = dot_z > 0.95
        pts = 15 if ok else 0
        checks.append({"id": "C1.5", "name": f"Crack orientation (n·z={dot_z:.3f})",
                        "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.5", "name": "Crack orientation", "pass": False,
                        "pts": 0, "max": 15, "error": str(e)})

    # C1.6: Multi-chart Robin DD converges
    try:
        from solvers.fem.robin_schwarz import RobinSchwarzSolver

        # Build 2-chart Robin DD
        n_charts = 2; overlap = 0.3
        z_span = L / n_charts * (1 + overlap)
        solvers_mc = []; decoders_mc = []; seeds_mc = []

        for ci in range(n_charts):
            z_c = L * (ci + 0.5) / n_charts
            d = BoxDecoder(center=(0, 0, z_c), half_extents=(R*1.2, R*1.2, z_span/2)).double()
            s = ChartVectorFEMSolver(
                n_cells=8, support_r=1.0, chart_decoder=d, decoder_kwargs={},
                sdf_oracle=sdf, sdf_threshold=-0.01,
                device="cpu", dtype=torch.float64,
            )
            solvers_mc.append(s); decoders_mc.append(d); seeds_mc.append([0, 0, z_c])

        seeds_mc_t = torch.tensor(seeds_mc, dtype=torch.float64)
        nbrs = [[1], [0]]

        robin = RobinSchwarzSolver(
            chart_solvers=solvers_mc, seeds=seeds_mc_t,
            decoders=decoders_mc, neighbors=nbrs,
            robin_delta=E * 0.5, parallel=True, n_workers=2,
        )

        eps_test = 1e-4
        def test_bc(nodes_phys):
            n = len(nodes_phys)
            u_bc = np.zeros((n, 3)); mask = np.ones(n, dtype=bool)
            u_bc[:, 0] = -nu * eps_test * nodes_phys[:, 0]
            u_bc[:, 1] = -nu * eps_test * nodes_phys[:, 1]
            u_bc[:, 2] = eps_test * nodes_phys[:, 2]
            return u_bc, mask

        u_charts = robin.solve(stress_fn, tangent_fn, test_bc, max_iters=15, tol=1e-3)
        ok = all(u is not None for u in u_charts) and all(torch.isfinite(u).all() for u in u_charts)

        # Check stress accuracy across charts
        if ok:
            szz_all = []
            for ci in range(n_charts):
                F_i = solvers_mc[ci].compute_F(u_charts[ci])
                P_i = stress_fn(F_i)
                szz_all.extend(P_i.detach().numpy()[:, 2, 2].tolist())
            szz_mean = np.mean(szz_all)
            err_mc = abs(szz_mean - E * eps_test) / (E * eps_test)
            ok = err_mc < 0.05

        pts = 10 if ok else 0
        checks.append({"id": "C1.6", "name": "Robin DD converges", "pass": ok,
                        "pts": pts, "max": 10})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.6", "name": "Robin DD converges", "pass": False,
                        "pts": 0, "max": 10, "error": str(e)})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")

    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
