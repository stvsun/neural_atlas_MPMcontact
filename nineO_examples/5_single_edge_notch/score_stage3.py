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
        test_nucleation_mesh_independence,
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
    try:
        def sdf_oracle(x):
            """SEN specimen: approximate as box, negative inside."""
            he = torch.tensor([1.0, 1.0, 0.5], dtype=x.dtype, device=x.device)
            d = torch.abs(x) - he
            return torch.max(d, dim=-1).values

        def bc_fn(nodes_np):
            """SEN tension: fix bottom, pull top."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            eps_val = 1e-3
            bot = nodes_np[:, 1] < -0.95
            top = nodes_np[:, 1] > 0.95
            mask[bot | top] = True
            u[top, 1] = eps_val * 2.0
            return u, mask

        res = test_nucleation_mesh_independence(
            decoder_cls=CrackTipDecoder,
            decoder_args={},
            sdf_oracle=sdf_oracle,
            stress_fn=stress_fn,
            tangent_fn=tangent_fn,
            bc_fn=bc_fn,
            sigma_ts=sigma_ts,
            sigma_hs=sigma_hs,
        )
        for c in res["checks"]:
            c["id"] = c.get("id", "X8")
            checks.append(c)
        total += res["pts"]
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
