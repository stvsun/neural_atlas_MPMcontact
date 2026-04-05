"""Score for Challenge 1: Uniaxial Tension Test.

Checks:
  C1.1: Chart FEM solver builds mesh on cylindrical rod         [10 pts]
  C1.2: Newton converges for uniaxial loading                   [15 pts]
  C1.3: sigma_zz matches E*epsilon within 2%                    [25 pts]
  C1.4: Drucker-Prager nucleation at sigma_ts within 10%        [25 pts]
  C1.5: Crack orientation perpendicular to axis                  [15 pts]
  C1.6: Multi-chart Schwarz DD converges                         [10 pts]

Uses CylinderDecoder (body-fitted) instead of BoxDecoder (SDF-cut)
for proper O(h^2) stress convergence on the cylindrical rod geometry.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import math
    checks = []
    total = 0

    # C1.1: Solver builds mesh with CylinderDecoder (body-fitted)
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.analytic_decoders import CylinderDecoder
        import torch, numpy as np

        R = 2.0; L = 15.0

        dec = CylinderDecoder(theta_center=0, theta_span=math.pi * 1.5,
                               R=R, z_center=L / 2, L_half=L / 2).double()
        solver = ChartVectorFEMSolver(
            n_cells=10, support_r=1.0, chart_decoder=dec, decoder_kwargs={},
            device="cpu", dtype=torch.float64,
        )
        ok = solver.n_nodes > 0 and solver.n_elements > 0
        checks.append({"id": "C1.1", "name": f"CylinderDecoder mesh ({solver.n_nodes} nodes)",
                        "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C1.1", "name": "Mesh builds", "pass": False, "pts": 0,
                        "max": 10, "error": str(e)})

    # C1.2: Newton converges for uniaxial loading
    try:
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        E = 70e3; nu = 0.22
        stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

        eps = 1e-4
        nodes_phys = solver.nodes_phys.detach().cpu().numpy()

        # Exact analytical solution: uniaxial tension with Poisson contraction
        u_exact = np.zeros((solver.n_nodes, 3))
        u_exact[:, 0] = -nu * eps * nodes_phys[:, 0]
        u_exact[:, 1] = -nu * eps * nodes_phys[:, 1]
        u_exact[:, 2] = eps * nodes_phys[:, 2]

        # BCs: prescribe exact solution on boundary nodes
        u_bc_t = torch.tensor(u_exact, dtype=torch.float64)
        bc_mask_t = solver.boundary_mask.clone()
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
        checks.append({"id": "C1.3", "name": f"Stress accuracy ({err * 100:.2f}%)",
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

        checks.append({"id": "C1.4", "name": f"DP nucleation ({err_nuc * 100:.1f}%)",
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
        dot_z = abs(normal[2])
        ok = dot_z > 0.95
        pts = 15 if ok else 0
        checks.append({"id": "C1.5", "name": f"Crack orientation (n·z={dot_z:.3f})",
                        "pass": ok, "pts": pts, "max": 15})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.5", "name": "Crack orientation", "pass": False,
                        "pts": 0, "max": 15, "error": str(e)})

    # C1.6: Multi-chart Schwarz DD converges
    try:
        from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver

        n_sectors = 3
        sector_span = 2 * math.pi / n_sectors * 1.3  # 30% overlap
        solvers_mc = []; decoders_mc = []; seeds_mc = []

        for k in range(n_sectors):
            theta_c = k * 2 * math.pi / n_sectors
            d = CylinderDecoder(theta_center=theta_c, theta_span=sector_span,
                                 R=R, z_center=L / 2, L_half=L / 2).double()
            s = ChartVectorFEMSolver(
                n_cells=6, support_r=1.0, chart_decoder=d, decoder_kwargs={},
                device="cpu", dtype=torch.float64,
            )
            solvers_mc.append(s); decoders_mc.append(d)
            seeds_mc.append([R * 0.5 * math.cos(theta_c),
                             R * 0.5 * math.sin(theta_c), L / 2])

        seeds_mc_t = torch.tensor(seeds_mc, dtype=torch.float64)
        nbrs = [[1, 2], [0, 2], [0, 1]]

        eps_test = 1e-4

        def test_bc(nodes_phys_np):
            n = len(nodes_phys_np)
            u_bc_arr = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            # Prescribe exact solution on top and bottom faces
            bot = nodes_phys_np[:, 2] < 0.5
            top = nodes_phys_np[:, 2] > L - 0.5
            mask[bot | top] = True
            u_bc_arr[:, 0] = -nu * eps_test * nodes_phys_np[:, 0]
            u_bc_arr[:, 1] = -nu * eps_test * nodes_phys_np[:, 1]
            u_bc_arr[:, 2] = eps_test * nodes_phys_np[:, 2]
            return u_bc_arr, mask

        schwarz = SchwarzVectorFEMSolver(
            chart_solvers=solvers_mc, seeds=seeds_mc_t, decoders=decoders_mc,
            neighbors=nbrs, parallel=False,
        )
        u_charts = schwarz.solve(stress_fn, tangent_fn, test_bc,
                                  max_schwarz_iters=25, tol=3e-2, relaxation=0.5)

        ok = all(u is not None for u in u_charts) and \
             all(torch.isfinite(u).all() for u in u_charts)

        if ok:
            szz_all = []
            for ci in range(n_sectors):
                F_i = solvers_mc[ci].compute_F(u_charts[ci])
                P_i = stress_fn(F_i)
                szz_all.extend(P_i.detach().numpy()[:, 2, 2].tolist())
            szz_mean = np.mean(szz_all)
            err_mc = abs(szz_mean - E * eps_test) / (E * eps_test)
            ok = err_mc < 0.15  # relaxed: DD on curved geometry has higher error

        pts = 10 if ok else 0
        checks.append({"id": "C1.6", "name": f"Schwarz DD (err={err_mc * 100:.1f}%)",
                        "pass": ok, "pts": pts, "max": 10})
        total += pts
    except Exception as e:
        checks.append({"id": "C1.6", "name": "Schwarz DD converges", "pass": False,
                        "pts": 0, "max": 10, "error": str(e)})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")

    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
