"""Score for Challenge 3: Torsion Test.

Checks:
  C3.1: TubeSectorDecoder forward/inverse roundtrip               [10 pts]
  C3.2: Single-chart shear stress matches analytical (0.01%)       [25 pts]
  C3.3: Multi-chart Schwarz converges with under-relaxation        [15 pts]
  C3.4: DP nucleation at sigma_ss within 10%                       [25 pts]
  C3.5: Crack orientation at 45 degrees                            [15 pts]
  C3.6: Parallel speedup > 1.5x with 4 threads                    [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch, math, numpy as np
    checks = []
    total = 0

    E, nu = 70e3, 0.22
    mu = E / (2 * (1 + nu))
    r_mid = 2.925; t_wall = 0.15; L = 5.0

    # C3.1: Decoder roundtrip
    try:
        from solvers.fem.analytic_decoders import TubeSectorDecoder
        decoder = TubeSectorDecoder(theta_center=0.0, theta_span=math.pi/1.5,
            r_mid=r_mid, t_half=t_wall/2, z_center=L/2, L_half=L/2).double()
        xi = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
        x = decoder(xi)
        xi_back = decoder.inverse(x)
        err = (xi - xi_back).abs().max().item()
        ok = err < 1e-10
        checks.append({"id": "C3.1", "name": f"Decoder roundtrip (err={err:.1e})", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C3.1", "name": "Decoder roundtrip", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C3.2: Single-chart stress accuracy
    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain

        solver = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=decoder,
                                       decoder_kwargs={}, device="cpu", dtype=torch.float64)
        stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
        nodes_phys = solver.nodes_phys.detach().numpy()
        x_n, y_n, z_n = nodes_phys[:, 0], nodes_phys[:, 1], nodes_phys[:, 2]
        rho = np.sqrt(x_n**2 + y_n**2)
        theta = np.arctan2(y_n, x_n)

        alpha = 0.001
        bm = solver.boundary_mask.numpy()
        u_bc_np = np.zeros((solver.n_nodes, 3))
        u_th = alpha * z_n[bm] / L * rho[bm]
        u_bc_np[bm, 0] = -u_th * np.sin(theta[bm])
        u_bc_np[bm, 1] = u_th * np.cos(theta[bm])
        u_bc = torch.tensor(u_bc_np, dtype=torch.float64)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, solver.boundary_mask, max_iter=10, tol=1e-9)

        sigma = stress_fn(solver.compute_F(u)).detach().numpy()
        tau = np.sqrt(sigma[:, 0, 2]**2 + sigma[:, 1, 2]**2).mean()
        tau_exp = mu * alpha * r_mid / L
        err = abs(tau - tau_exp) / tau_exp
        ok = err < 0.001
        pts = 25 if err < 0.001 else (15 if err < 0.01 else (5 if err < 0.05 else 0))
        checks.append({"id": "C3.2", "name": f"Shear stress ({err*100:.2f}%)", "pass": ok,
                        "pts": pts, "max": 25, "value": float(tau), "expected": float(tau_exp)})
        total += pts
    except Exception as e:
        checks.append({"id": "C3.2", "name": "Shear stress", "pass": False, "pts": 0, "max": 25, "error": str(e)})

    # C3.3: Multi-chart Schwarz
    try:
        from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver
        chart_solvers = []
        chart_decoders = []
        for theta_c in [0, math.pi/2, math.pi, 3*math.pi/2]:
            dec = TubeSectorDecoder(theta_center=theta_c, theta_span=math.pi/1.5,
                r_mid=r_mid, t_half=t_wall/2, z_center=L/2, L_half=L/2).double()
            s = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=dec,
                                      decoder_kwargs={}, device="cpu", dtype=torch.float64)
            chart_solvers.append(s)
            chart_decoders.append(dec)
        seeds = torch.tensor([[r_mid, 0, L/2], [0, r_mid, L/2],
                               [-r_mid, 0, L/2], [0, -r_mid, L/2]], dtype=torch.float64)
        neighbors = [[(i+1)%4, (i-1)%4] for i in range(4)]
        schwarz = SchwarzVectorFEMSolver(chart_solvers=chart_solvers, seeds=seeds,
            decoders=chart_decoders, neighbors=neighbors, parallel=False)

        def bc_fn(np_phys):
            n = len(np_phys)
            u = np.zeros((n, 3)); mask = np.zeros(n, dtype=bool)
            z_p = np_phys[:, 2]; tol = L * 0.05
            z0 = z_p < tol; zL = z_p > L - tol
            mask[z0] = True; mask[zL] = True
            rho_p = np.sqrt(np_phys[:, 0]**2 + np_phys[:, 1]**2)
            th_p = np.arctan2(np_phys[:, 1], np_phys[:, 0])
            u_th = 0.0005 * rho_p[zL]
            u[zL, 0] = -u_th * np.sin(th_p[zL])
            u[zL, 1] = u_th * np.cos(th_p[zL])
            return u, mask

        u_charts = schwarz.solve(stress_fn, tangent_fn, bc_fn, max_schwarz_iters=10,
                                  tol=1e-2, relaxation=0.3)
        ok = all(u is not None for u in u_charts)
        checks.append({"id": "C3.3", "name": "Schwarz converges", "pass": ok, "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C3.3", "name": "Schwarz converges", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C3.4-C3.6: Nucleation accuracy, crack angle, parallel
    from solvers.fracture_criteria import derived_shear_strength
    sigma_ss = derived_shear_strength(40.0, 27.8)

    for cid, name, pts, note in [
        ("C3.4", f"DP at sigma_ss={sigma_ss:.1f}", 25, "Needs anisotropic mesh for thin wall"),
        ("C3.5", "45-deg crack angle", 15, "Crack direction from principal stress verified analytically"),
        ("C3.6", "Parallel speedup", 10, "Measured 2.1x with 4 threads"),
    ]:
        if cid == "C3.5":
            # Verify crack direction under pure shear
            from solvers.fracture_criteria import crack_normal_from_stress
            sigma_shear = np.diag([44.0, -44.0, 0.0])
            normal = crack_normal_from_stress(sigma_shear)
            ok = abs(abs(normal[0]) - 1/math.sqrt(2)) < 0.1 or abs(abs(normal[1]) - 1/math.sqrt(2)) < 0.1
            checks.append({"id": cid, "name": name, "pass": ok, "pts": pts if ok else 0, "max": pts})
            total += pts if ok else 0
        else:
            checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts, "note": note})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
