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
        test_nucleation_mesh_independence,
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
    try:
        def sdf_oracle(x):
            """Tube sector: approximate SDF as distance from tube wall."""
            r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
            d_r = torch.abs(r - r_mid) - t_half
            d_z = torch.abs(x[:, 2] - z_center) - L_half
            return torch.max(d_r, d_z)

        def bc_fn(nodes_np):
            """Torsion: fix bottom face, twist top face."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            twist_angle = 1e-3  # small twist
            bot = nodes_np[:, 2] < (z_center - L_half + 0.05)
            top = nodes_np[:, 2] > (z_center + L_half - 0.05)
            mask[bot | top] = True
            # Top face: rotate by twist_angle about z-axis
            x_top = nodes_np[top, 0]
            y_top = nodes_np[top, 1]
            u[top, 0] = -twist_angle * y_top
            u[top, 1] = twist_angle * x_top
            return u, mask

        res = test_nucleation_mesh_independence(
            decoder_cls=TubeSectorDecoder,
            decoder_args={"r_mid": r_mid, "t_half": t_half,
                          "z_center": z_center, "L_half": L_half},
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
