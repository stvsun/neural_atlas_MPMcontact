"""Stage 3 XFEM-critic tests for Challenge 6: Indentation.

No crack present -- only non-crack tests apply:
  X6: Stiffness conditioning (10 pts)
  X7: Integration near singularity (10 pts)
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
    from solvers.fem.analytic_decoders import CylinderDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_stiffness_conditioning,
        test_integration_near_singularity,
        test_nucleation_mesh_independence,
    )

    checks = []
    total = 0

    # Material: glass
    E, nu = 70e3, 0.22
    sigma_ts, sigma_hs = 40.0, 27.8
    R_block = 25.0
    L_block = 25.0
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # Build solver with CylinderDecoder
    dec = CylinderDecoder(R=R_block, z_center=L_block / 2, L_half=L_block / 2).double()
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
            """Cylinder: negative inside, positive outside."""
            r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
            d_r = r - R_block
            d_z = torch.abs(x[:, 2] - L_block / 2) - L_block / 2
            return torch.max(d_r, d_z)

        def bc_fn(nodes_np):
            """Indentation: fix bottom face, apply compressive displacement on top."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            indent = 0.1
            bot = nodes_np[:, 2] < 0.05
            top = nodes_np[:, 2] > (L_block - 0.05)
            # Only indent near center of top face (r < indenter radius)
            r_top = np.sqrt(nodes_np[top, 0]**2 + nodes_np[top, 1]**2)
            indent_zone = r_top < 5.0  # 5mm indenter radius
            mask[bot] = True
            top_idx = np.where(top)[0]
            mask[top_idx[indent_zone]] = True
            u[top_idx[indent_zone], 2] = -indent
            return u, mask

        res = test_nucleation_mesh_independence(
            decoder_cls=CylinderDecoder,
            decoder_args={"R": R_block, "z_center": L_block / 2,
                          "L_half": L_block / 2},
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

    max_score = 35
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
