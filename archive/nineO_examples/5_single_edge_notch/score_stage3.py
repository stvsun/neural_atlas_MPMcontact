"""Stage 3 XFEM-critic tests for Challenge 5: Single Edge Notch.

Has crack -- full fracture test battery:
  X1: Williams angular modes (15 pts)
  X2: Crack-face traction-free (10 pts)
  X3: Mixed-mode extraction (10 pts)
  X4: Curved crack path (10 pts)
  X6: Stiffness conditioning (10 pts)
  X7: Integration near singularity (10 pts)
  X8: Nucleation mesh independence (15 pts)
  X9: K_I accuracy vs XFEM (10 pts)

Max score: 100
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    import numpy as np
    import math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import CrackTipDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_williams_angular_modes,
        test_crack_face_traction_free,
        test_mixed_mode_extraction,
        test_curved_crack_path,
        test_stiffness_conditioning,
        test_integration_near_singularity,
        test_ki_accuracy_vs_xfem,
    )

    checks = []
    total = 0

    # Material: glass
    E, nu, Gc = 70e3, 0.22, 0.01
    sigma_ts, sigma_hs = 40.0, 27.8
    K_Ic = math.sqrt(E * Gc / (1 - nu**2))
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # Crack geometry
    crack_tip = [-0.5, 0, 0]
    crack_dir = [1, 0, 0]
    opening_dir = [0, 1, 0]

    # Build solver with CrackTipDecoder
    dec = CrackTipDecoder.from_crack_tip(
        tip_position=crack_tip, crack_direction=crack_dir,
        opening_direction=opening_dir, radius=0.3,
    ).double()
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=dec,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )

    # Pre-solve Williams displacement for X1, X2, X9
    u_sol = None
    try:
        from benchmarks.fracture.lefm_reference import williams_displacement
        nodes = solver.nodes_phys.detach().cpu().numpy()
        tip = np.array(crack_tip)
        dx = nodes - tip
        r = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
        theta = np.arctan2(dx[:, 1], dx[:, 0])
        r = np.maximum(r, 1e-12)
        ux_w, uy_w = williams_displacement(r, theta, K_Ic, E, nu, plane_strain=True)
        u_bc = np.zeros_like(nodes)
        u_bc[:, 0] = ux_w; u_bc[:, 1] = uy_w
        u_t = torch.tensor(u_bc, dtype=torch.float64)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_sol = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t,
                                        solver.boundary_mask, max_iter=10, tol=1e-10)
    except Exception:
        pass

    # --- X1: Williams angular modes (15 pts) ---
    try:
        res = test_williams_angular_modes(
            solver, dec, K_Ic, E, nu, crack_tip, crack_dir, opening_dir,
        )
        for c in res["checks"]:
            c["id"] = c.get("id", "X1")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X1", "name": f"Williams angular modes ({e})",
                        "pass": False, "pts": 0, "max": 15})

    # --- X2: Crack-face traction-free (10 pts) ---
    try:
        if u_sol is not None:
            res = test_crack_face_traction_free(
                solver, stress_fn, u_sol, crack_tip, crack_dir, opening_dir,
            )
        else:
            res = {"pts": 0, "checks": [{"name": "No solution available", "pass": False, "pts": 0}]}
        for c in res["checks"]:
            c["id"] = c.get("id", "X2")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X2", "name": f"Crack-face traction-free ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X3: Mixed-mode extraction (10 pts) ---
    try:
        res = test_mixed_mode_extraction(solver, E, nu, crack_tip, crack_dir, opening_dir)
        for c in res["checks"]:
            c["id"] = c.get("id", "X3")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X3", "name": f"Mixed-mode extraction ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X4: Curved crack path (10 pts) ---
    try:
        res = test_curved_crack_path()
        for c in res["checks"]:
            c["id"] = c.get("id", "X4")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X4", "name": f"Curved crack path ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X6: Stiffness conditioning (10 pts) ---
    try:
        res = test_stiffness_conditioning(solver, stress_fn, tangent_fn)
        for c in res["checks"]:
            c["id"] = c.get("id", "X6")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X6", "name": f"Stiffness conditioning ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X7: Integration near singularity (10 pts) ---
    try:
        res = test_integration_near_singularity(solver)
        for c in res["checks"]:
            c["id"] = c.get("id", "X7")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X7", "name": f"Integration near singularity ({e})",
                        "pass": False, "pts": 0, "max": 10})

    # --- X8: Nucleation mesh independence (15 pts) ---
    # Uses same CrackTipDecoder as S1, tight BCs (Williams), split into X8a/X8b/X8c
    try:
        from solvers.fracture_criteria import drucker_prager_F
        from benchmarks.fracture.lefm_reference import williams_displacement

        eps_load = sigma_ts / E * 0.95  # just below nucleation

        def bc_fn_nuc(nodes_np):
            """SEN: prescribe Williams displacement on boundary nodes."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            tip = np.array(crack_tip)
            dx = nodes_np - tip
            r = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
            theta = np.arctan2(dx[:, 1], dx[:, 0])
            r_safe = np.maximum(r, 1e-12)
            # Scale K_I so stress is just below nucleation
            K_load = K_Ic * eps_load * E / sigma_ts
            ux_w, uy_w = williams_displacement(r_safe, theta, K_load, E, nu, plane_strain=True)
            # Boundary: nodes on the surface of the CrackTipDecoder domain
            # Use distance from crack tip > 0.28 (near boundary of radius=0.3 domain)
            on_boundary = r > 0.28
            mask[on_boundary] = True
            u[on_boundary, 0] = ux_w[on_boundary]
            u[on_boundary, 1] = uy_w[on_boundary]
            return u, mask

        nuc_locs_pointwise = []
        nuc_locs_nonlocal = []

        for nc in [6, 8, 10, 12]:
            dec_c = CrackTipDecoder.from_crack_tip(
                tip_position=crack_tip, crack_direction=crack_dir,
                opening_direction=opening_dir, radius=0.3,
            ).double()
            s_c = ChartVectorFEMSolver(
                n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                decoder_kwargs={}, device="cpu", dtype=torch.float64,
            )
            if s_c.n_elements == 0:
                continue

            nodes_np = s_c.nodes_phys.detach().cpu().numpy()
            u_bc_vals, mask = bc_fn_nuc(nodes_np)
            u_bc = torch.tensor(u_bc_vals, dtype=torch.float64)
            bc_mask = torch.tensor(mask, dtype=torch.bool)
            f_ext = torch.zeros(s_c.n_nodes, 3, dtype=torch.float64)

            u_sol = s_c.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
                                         max_iter=10, tol=1e-8)
            F = s_c.compute_F(u_sol)
            P = stress_fn(F)
            P_np = P.detach().cpu().numpy()
            centroids = s_c.elem_centroids_phys.detach().cpu().numpy()

            # Pointwise DP
            F_dp_vals = np.array([
                drucker_prager_F(
                    torch.tensor(P_np[e]).unsqueeze(0), sigma_ts, sigma_hs
                ).item()
                for e in range(len(P_np))
            ])
            max_idx_pw = np.argmax(F_dp_vals)
            nuc_locs_pointwise.append(centroids[max_idx_pw].copy())

            # Nonlocal-smoothed DP
            try:
                from solvers.fem.nonlocal_damage import (
                    compute_local_equivalent_strain, solve_nonlocal_strain
                )
                e_local = compute_local_equivalent_strain(F)
                e_nl = solve_nonlocal_strain(s_c, e_local, length_scale=0.1)
                elements = s_c.elements[:, :4]
                e_nl_elem = e_nl[elements].mean(dim=1).detach().cpu().numpy()
                max_idx_nl = np.argmax(e_nl_elem)
                nuc_locs_nonlocal.append(centroids[max_idx_nl].copy())
            except Exception:
                nuc_locs_nonlocal.append(centroids[max_idx_pw].copy())

        # X8a: Pointwise scatter
        if len(nuc_locs_pointwise) >= 3:
            dists_pw = [np.linalg.norm(nuc_locs_pointwise[i] - nuc_locs_pointwise[i - 1])
                        for i in range(1, len(nuc_locs_pointwise))]
            max_scatter_pw = max(dists_pw)
            converging_pw = dists_pw[-1] < dists_pw[0] if len(dists_pw) >= 2 else False
            pts_pw = 5 if max_scatter_pw < 0.2 else (3 if max_scatter_pw < 0.5 else (1 if max_scatter_pw < 1.0 else 0))
            checks.append({
                "id": "X8a", "max": 5,
                "name": f"Pointwise DP scatter: {max_scatter_pw:.3f}mm, conv={converging_pw}",
                "pass": max_scatter_pw < 0.5, "pts": pts_pw,
            })
            total += pts_pw
        else:
            checks.append({"id": "X8a", "name": "Not enough meshes", "pass": False, "pts": 0, "max": 5})

        # X8b: Nonlocal scatter
        if len(nuc_locs_nonlocal) >= 3:
            dists_nl = [np.linalg.norm(nuc_locs_nonlocal[i] - nuc_locs_nonlocal[i - 1])
                        for i in range(1, len(nuc_locs_nonlocal))]
            max_scatter_nl = max(dists_nl)
            converging_nl = dists_nl[-1] < dists_nl[0] if len(dists_nl) >= 2 else False
            pts_nl = 5 if max_scatter_nl < 0.2 else (3 if max_scatter_nl < 0.5 else (1 if max_scatter_nl < 1.0 else 0))
            checks.append({
                "id": "X8b", "max": 5,
                "name": f"Nonlocal DP scatter: {max_scatter_nl:.3f}mm, conv={converging_nl}",
                "pass": max_scatter_nl < 0.5, "pts": pts_nl,
            })
            total += pts_nl
            if max_scatter_pw > 0:
                improvement = (max_scatter_pw - max_scatter_nl) / max_scatter_pw * 100
                print(f"    [INFO] Nonlocal vs pointwise scatter: "
                      f"{max_scatter_pw:.3f} -> {max_scatter_nl:.3f}mm "
                      f"({improvement:+.1f}% improvement)")
        else:
            checks.append({"id": "X8b", "name": "Not enough meshes", "pass": False, "pts": 0, "max": 5})

        # X8c: Nonlocal regularization exists
        has_nonlocal = False
        try:
            from solvers.fem import nonlocal_damage
            has_nonlocal = True
        except ImportError:
            pass
        checks.append({
            "id": "X8c", "max": 5,
            "name": f"Nonlocal/gradient regularization: {has_nonlocal}",
            "pass": has_nonlocal, "pts": 5 if has_nonlocal else 0,
        })
        total += 5 if has_nonlocal else 0

    except Exception as e:
        checks.append({"id": "X8", "name": f"Nucleation mesh independence ({e})",
                        "pass": False, "pts": 0, "max": 15})

    # --- X9: K_I accuracy vs XFEM (10 pts) ---
    try:
        if u_sol is not None:
            from solvers.fem.k_extraction import extract_K_from_fem
            from solvers.fem.j_integral import extract_K_via_J_integral
            K_I_extracted = float('nan')
            try:
                K_I_extracted = extract_K_via_J_integral(
                    [solver], [u_sol], crack_tip, crack_dir, opening_dir,
                    stress_fn, E, nu, plane_strain=True,
                )
            except Exception:
                pass
            if math.isnan(K_I_extracted) or K_I_extracted <= 0:
                K_I_extracted = extract_K_from_fem(
                    solver, u_sol, crack_tip, crack_dir, opening_dir,
                    E, nu, plane_strain=True,
                )
            res = test_ki_accuracy_vs_xfem(K_I_extracted, K_Ic)
        else:
            res = {"pts": 0, "checks": [{"name": "No solution for K_I", "pass": False, "pts": 0}]}
        for c in res["checks"]:
            c["id"] = c.get("id", "X9")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X9", "name": f"K_I accuracy ({e})",
                        "pass": False, "pts": 0, "max": 10})

    max_score = 100
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
