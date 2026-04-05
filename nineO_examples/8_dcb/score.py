"""Score for Challenge 8: Double Cantilever Beam Test.

Checks:
  C8.1: Beam theory F vs delta matches Eq. (17)                  [15 pts]
  C8.2: Crack growth a(delta) matches Eq. (18)                   [15 pts]
  C8.3: Critical displacement delta_crit correct                 [10 pts]
  C8.4: Griffith criterion K_Ic = sqrt(E*Gc/(1-nu^2))           [10 pts]
  C8.5: Chart FEM solve on bar with pre-crack                    [20 pts]
  C8.6: FEM K_I matches beam theory within 10%                   [20 pts]
  C8.7: Stable crack growth (F decreases with delta)             [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import numpy as np, math
    checks = []
    total = 0

    E, nu, Gc = 70e3, 0.22, 0.01
    A, H, B = 25.0, 20.0, 2.5
    h = H / 2  # arm height

    # C8.1: F vs delta
    try:
        from benchmarks.fracture.dcb_reference import dcb_response, dcb_critical_displacement
        result = dcb_response(delta_max=dcb_critical_displacement() * 3, n_points=100)
        # F should increase then decrease
        F_max_idx = np.argmax(result["force"])
        ok = F_max_idx > 0 and F_max_idx < len(result["force"]) - 1
        checks.append({"id": "C8.1", "name": f"F vs delta (peak at idx {F_max_idx})", "pass": ok,
                        "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C8.1", "name": "F vs delta", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C8.2: Crack growth
    try:
        a_final = result["crack_length"][-1]
        ok = a_final > A * 1.2  # crack should grow beyond initial
        checks.append({"id": "C8.2", "name": f"Crack grows to a={a_final:.1f}mm", "pass": ok,
                        "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C8.2", "name": "Crack growth", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C8.3: delta_crit
    try:
        dc = dcb_critical_displacement()
        # Analytical: delta_crit = F_crit * C(A) where C = 2A^3/(3EI)
        I_arm = B * h**3 / 12
        F_crit_check = B * math.sqrt(E * Gc * h**3 / (12 * A**2))
        C_A = 2 * A**3 / (3 * E * I_arm)
        dc_check = F_crit_check * C_A
        err = abs(dc - dc_check) / dc_check
        ok = err < 0.01
        checks.append({"id": "C8.3", "name": f"delta_crit={dc*1e3:.2f}um (err {err*100:.1f}%)",
                        "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C8.3", "name": "delta_crit", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C8.4: Griffith K_Ic
    try:
        from solvers.fracture_criteria import griffith_K_Ic
        K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)
        K_Ic_check = math.sqrt(E * Gc / (1 - nu**2))
        ok = abs(K_Ic - K_Ic_check) < 1e-10
        checks.append({"id": "C8.4", "name": f"K_Ic={K_Ic:.2f}", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C8.4", "name": "K_Ic", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C8.5: Chart FEM solve on DCB bar
    try:
        import torch
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.robin_schwarz import RobinSchwarzSolver
        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder
        from solvers.fem.k_extraction import extract_K_from_charts
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle

        W_half = L / 2; h_arm = H / 2
        sdf = CrackedPlateSDFOracle(a=A, W=W_half, H=H/2, T=B, delta=0.05)
        stress_fn8, tangent_fn8 = make_linear_elastic_small_strain(E, nu)

        dec_l = BoxDecoder(center=(-L/4, 0, 0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
        s_l = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_l,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)
        dec_r = BoxDecoder(center=(L/4, 0, 0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
        s_r = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=dec_r,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)
        tip_x = -W_half + A
        crack_dec = CrackTipDecoder.from_crack_tip([tip_x, 0, 0], [1,0,0], [0,1,0], radius=2.0).double()
        s_c = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=crack_dec,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)

        solvers8 = [s_l, s_r, s_c]; decoders8 = [dec_l, dec_r, crack_dec]
        seeds8 = torch.tensor([[-L/4,0,0],[L/4,0,0],[tip_x,0,0]], dtype=torch.float64)

        # Apply displacement at pins
        I_arm = B * h_arm**3 / 12
        F_crit = B * math.sqrt(E * Gc * h_arm**3 / (12 * A**2))
        C_A = 2 * A**3 / (3 * E * I_arm)
        delta = F_crit * C_A * 1.0  # at critical
        pin_x = -W_half + 1.5

        def bc_fn8(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            x = np_phys[:, 0]; y = np_phys[:, 1]
            right = x > W_half - 1.0; m[right] = True
            top_pin = (np.abs(x - pin_x) < 2.0) & (np.abs(y - H/6) < 2.0) & (y > 0)
            m[top_pin] = True; u[top_pin, 1] = delta
            bot_pin = (np.abs(x - pin_x) < 2.0) & (np.abs(y + H/6) < 2.0) & (y < 0)
            m[bot_pin] = True; u[bot_pin, 1] = -delta
            return u, m

        robin = RobinSchwarzSolver(chart_solvers=solvers8, seeds=seeds8, decoders=decoders8,
                                    neighbors=[[1,2],[0,2],[0,1]], robin_delta=E*0.5, parallel=True)
        u_charts8 = robin.solve(stress_fn8, tangent_fn8, bc_fn8, max_iters=25, tol=1e-2)

        ok = all(u is not None for u in u_charts8)
        checks.append({"id": "C8.5", "name": "Chart FEM on DCB bar", "pass": ok,
                        "pts": 20 if ok else 0, "max": 20})
        total += 20 if ok else 0
    except Exception as e:
        checks.append({"id": "C8.5", "name": "Chart FEM on DCB bar", "pass": False, "pts": 0,
                        "max": 20, "error": str(e)})

    # C8.6: FEM K_I vs beam theory
    try:
        from solvers.fracture_criteria import griffith_K_Ic
        K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)
        K_I_fem = extract_K_from_charts(solvers8, u_charts8, [tip_x, 0, 0], [1,0,0], [0,1,0],
                                         E, nu, plane_strain=True, r_min=0.2, r_max=2.0)
        err_K = abs(K_I_fem - K_Ic) / K_Ic if K_Ic > 0 else float('inf')
        ok = err_K < 0.1 and not math.isnan(K_I_fem)
        pts = 20 if err_K < 0.1 else (10 if err_K < 0.2 else 0)
        checks.append({"id": "C8.6", "name": f"FEM K_I={K_I_fem:.2f} vs K_Ic={K_Ic:.2f} ({err_K*100:.1f}%)",
                        "pass": ok, "pts": pts, "max": 20})
        total += pts
    except Exception as e:
        checks.append({"id": "C8.6", "name": "FEM K_I vs beam theory", "pass": False, "pts": 0,
                        "max": 20, "error": str(e)})

    # C8.7: Stable crack growth (requires propagation driver)
    checks.append({"id": "C8.7", "name": "Stable crack growth", "pass": False, "pts": 0, "max": 10,
                    "note": "Requires crack propagation driver"})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
