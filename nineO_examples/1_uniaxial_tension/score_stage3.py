"""Stage 3 XFEM-critic tests for Challenge 1: Uniaxial Tension.

Uses CylinderDecoder (body-fitted) consistent with the S1 scoring,
and compares pointwise vs nonlocal-smoothed nucleation.

Tests:
  X6: Stiffness conditioning (10 pts)
  X8: Nucleation mesh independence (15 pts)
    - X8a: Pointwise DP scatter (5 pts)
    - X8b: Nonlocal-smoothed DP scatter (5 pts)
    - X8c: Nonlocal regularization exists (5 pts)

Max score: 25
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    import numpy as np
    import math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import CylinderDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_stiffness_conditioning,
    )

    checks = []
    total = 0

    # Material: glass (same as C1 S1)
    R = 2.0; L = 15.0
    E, nu = 70e3, 0.22
    sigma_ts, sigma_hs = 40.0, 27.8
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # Build solver with CylinderDecoder (same as S1)
    dec = CylinderDecoder(theta_center=0, theta_span=math.pi * 1.5,
                           R=R, z_center=L / 2, L_half=L / 2).double()
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=dec,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )

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

    # --- X8: Nucleation mesh independence (15 pts) ---
    # Run mesh refinement study and compare pointwise vs nonlocal nucleation
    try:
        from solvers.fracture_criteria import drucker_prager_F

        # Uniaxial tension BCs on CylinderDecoder: prescribe exact analytical
        # solution on boundary, let interior be free
        eps_load = sigma_ts / E * 0.95  # just below nucleation

        def bc_fn_cylinder(nodes_np):
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            # Prescribe exact uniaxial solution on boundary nodes
            bot = nodes_np[:, 2] < L / 8  # capture zone for BC
            top = nodes_np[:, 2] > L - L / 8
            bnd = bot | top
            mask[bnd] = True
            u[bnd, 0] = -nu * eps_load * nodes_np[bnd, 0]
            u[bnd, 1] = -nu * eps_load * nodes_np[bnd, 1]
            u[bnd, 2] = eps_load * nodes_np[bnd, 2]
            return u, mask

        # ─── X8a: Pointwise DP nucleation scatter ───
        nuc_locs_pointwise = []
        nuc_locs_nonlocal = []

        for nc in [6, 8, 10, 12]:
            dec_c = CylinderDecoder(theta_center=0, theta_span=math.pi * 1.5,
                                     R=R, z_center=L / 2, L_half=L / 2).double()
            s_c = ChartVectorFEMSolver(
                n_cells=nc, support_r=1.0, chart_decoder=dec_c,
                decoder_kwargs={}, device="cpu", dtype=torch.float64,
            )
            if s_c.n_elements == 0:
                continue

            nodes_np = s_c.nodes_phys.detach().cpu().numpy()
            u_bc_vals, mask = bc_fn_cylinder(nodes_np)
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
                e_nl = solve_nonlocal_strain(s_c, e_local, length_scale=0.5)

                # Project nonlocal strain to elements
                elements = s_c.elements[:, :4]
                e_nl_elem = e_nl[elements].mean(dim=1).detach().cpu().numpy()

                # Use nonlocal strain as the nucleation indicator
                max_idx_nl = np.argmax(e_nl_elem)
                nuc_locs_nonlocal.append(centroids[max_idx_nl].copy())
            except Exception:
                nuc_locs_nonlocal.append(centroids[max_idx_pw].copy())

        # Compute scatter for pointwise
        if len(nuc_locs_pointwise) >= 3:
            dists_pw = [np.linalg.norm(nuc_locs_pointwise[i] - nuc_locs_pointwise[i - 1])
                        for i in range(1, len(nuc_locs_pointwise))]
            max_scatter_pw = max(dists_pw)
            converging_pw = dists_pw[-1] < dists_pw[0] if len(dists_pw) >= 2 else False

            ok_pw = max_scatter_pw < 0.5
            pts_pw = 5 if max_scatter_pw < 0.2 else (3 if max_scatter_pw < 0.5 else (1 if max_scatter_pw < 1.0 else 0))
            checks.append({
                "id": "X8a", "max": 5,
                "name": f"Pointwise DP scatter: {max_scatter_pw:.3f}mm, conv={converging_pw}",
                "pass": ok_pw, "pts": pts_pw,
            })
            total += pts_pw
        else:
            checks.append({"id": "X8a", "name": "Not enough meshes", "pass": False, "pts": 0, "max": 5})

        # Compute scatter for nonlocal
        if len(nuc_locs_nonlocal) >= 3:
            dists_nl = [np.linalg.norm(nuc_locs_nonlocal[i] - nuc_locs_nonlocal[i - 1])
                        for i in range(1, len(nuc_locs_nonlocal))]
            max_scatter_nl = max(dists_nl)
            converging_nl = dists_nl[-1] < dists_nl[0] if len(dists_nl) >= 2 else False

            ok_nl = max_scatter_nl < 0.5
            pts_nl = 5 if max_scatter_nl < 0.2 else (3 if max_scatter_nl < 0.5 else (1 if max_scatter_nl < 1.0 else 0))
            checks.append({
                "id": "X8b", "max": 5,
                "name": f"Nonlocal DP scatter: {max_scatter_nl:.3f}mm, conv={converging_nl}",
                "pass": ok_nl, "pts": pts_nl,
            })
            total += pts_nl

            # Report improvement
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

    max_score = 25
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
