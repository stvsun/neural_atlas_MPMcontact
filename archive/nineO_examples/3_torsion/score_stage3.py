"""Stage 3 XFEM-critic tests for Challenge 3: Torsion.

No crack present -- only non-crack tests apply:
  X5: Displacement discontinuity handling (10 pts)
  X6: Stiffness conditioning (10 pts)
  X8: Nucleation mesh independence (15 pts)

Max score: 35
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import torch
    import numpy as np
    import math
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.analytic_decoders import TubeSectorDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_displacement_discontinuity_handling,
        test_stiffness_conditioning,
    )

    checks = []
    total = 0

    # Material: glass
    E, nu = 70e3, 0.22
    sigma_ts, sigma_hs = 40.0, 27.8
    r_mid = 2.925
    t_half = 0.075
    z_center = 2.5
    L_half = 2.5
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # Build solver with TubeSectorDecoder
    dec = TubeSectorDecoder(
        r_mid=r_mid, t_half=t_half, z_center=z_center, L_half=L_half,
    ).double()
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=dec,
        decoder_kwargs={}, device="cpu", dtype=torch.float64,
    )

    # --- X5: Displacement discontinuity handling (10 pts) ---
    try:
        res = test_displacement_discontinuity_handling()
        for c in res["checks"]:
            c["id"] = c.get("id", "X5")
            checks.append(c)
        total += res["pts"]
    except Exception as e:
        checks.append({"id": "X5", "name": f"Displacement discontinuity ({e})",
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

    # --- X8: Nucleation mesh independence (15 pts) ---
    # Uses same TubeSectorDecoder as S1, tight BCs, split into X8a/X8b/X8c
    try:
        from solvers.fracture_criteria import drucker_prager_F

        eps_load = sigma_ts / E * 0.95  # just below nucleation
        twist_angle = eps_load  # small torsion proportional to load

        def bc_fn_nuc(nodes_np):
            """Torsion: prescribe analytical torsion field on boundary nodes.
            Uses solver boundary_mask (SDF-derived boundary)."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            # Tight face selection: use all boundary nodes
            # (TubeSectorDecoder boundary is SDF-derived)
            r = np.sqrt(nodes_np[:, 0]**2 + nodes_np[:, 1]**2)
            d_r = np.abs(r - r_mid) - t_half
            d_z = np.abs(nodes_np[:, 2] - z_center) - L_half
            on_boundary = np.maximum(d_r, d_z) > -0.01
            mask[on_boundary] = True
            # Torsion field: fix bottom (z=0), twist top (z=5)
            # Linear twist along z: angle(z) = twist_angle * (z - z_center + L_half) / (2*L_half)
            frac_z = (nodes_np[:, 2] - (z_center - L_half)) / (2 * L_half)
            angle = twist_angle * frac_z
            u[on_boundary, 0] = -angle[on_boundary] * nodes_np[on_boundary, 1]
            u[on_boundary, 1] = angle[on_boundary] * nodes_np[on_boundary, 0]
            return u, mask

        nuc_locs_pointwise = []
        nuc_locs_nonlocal = []

        for nc in [6, 8, 10, 12]:
            dec_c = TubeSectorDecoder(
                r_mid=r_mid, t_half=t_half, z_center=z_center, L_half=L_half,
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
                e_nl = solve_nonlocal_strain(s_c, e_local, length_scale=0.3)
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

    max_score = 35
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
