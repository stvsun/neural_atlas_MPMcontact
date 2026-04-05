"""Stage 3 XFEM-critic tests for Challenge 7: Poker-Chip.

No crack present -- only non-crack tests apply:
  X6: Stiffness conditioning (10 pts)
  X8: Nucleation mesh independence (15 pts)

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
    from solvers.fem.analytic_decoders import BoxDecoder
    from nineO_examples.common_stage3.test_xfem_critique import (
        test_stiffness_conditioning,
        test_nucleation_mesh_independence,
    )

    checks = []
    total = 0

    # Material: PU elastomer
    mu_pu, lam_pu = 0.52, 85.77
    E_eff = mu_pu * (3 * lam_pu + 2 * mu_pu) / (lam_pu + mu_pu)
    nu_eff = lam_pu / (2 * (lam_pu + mu_pu))
    sigma_ts, sigma_hs = 40.0, 27.8
    Gc = 0.041
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E_eff, nu_eff)

    # Build solver with BoxDecoder for poker chip
    dec = BoxDecoder(center=(0, 0, 0), half_extents=(5, 5, 1)).double()
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
    try:
        def sdf_oracle(x):
            """Poker chip box: negative inside, positive outside."""
            he = torch.tensor([5.0, 5.0, 1.0], dtype=x.dtype, device=x.device)
            d = torch.abs(x) - he
            return torch.max(d, dim=-1).values

        def bc_fn(nodes_np):
            """Poker chip: fix bottom, pull top (hydrostatic tension)."""
            n = len(nodes_np)
            u = np.zeros((n, 3))
            mask = np.zeros(n, dtype=bool)
            eps_val = 1e-3
            bot = nodes_np[:, 2] < -0.95
            top = nodes_np[:, 2] > 0.95
            mask[bot | top] = True
            u[top, 2] = eps_val * 2.0
            return u, mask

        res = test_nucleation_mesh_independence(
            decoder_cls=BoxDecoder,
            decoder_args={"center": (0, 0, 0), "half_extents": (5, 5, 1)},
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

    max_score = 25
    status = "PASS" if total >= max_score * 0.5 else ("PARTIAL" if total > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c.get("pass", False) else "FAIL"
        print(f"  [{sym:4s}] {c.get('id', '')}: {c['name']} ({c.get('pts', 0)}/{c.get('max', '?')})")
    print(f"  Stage 3 Score: {total}/{max_score}")
    return {"status": status, "score": float(total), "max_score": max_score, "checks": checks}
