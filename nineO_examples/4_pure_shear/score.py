"""Score for Challenge 4: Pure-Shear Fracture Test.

Checks:
  C4.1: Strip SDF with edge crack builds correctly                [10 pts]
  C4.2: CrackTipDecoder at crack front                            [15 pts]
  C4.3: K_I extraction from FEM displacement within 10%           [25 pts]
  C4.4: Critical grip separation h_crit within 10%                [25 pts]
  C4.5: Crack propagates straight ahead (Mode I)                  [15 pts]
  C4.6: G = G_c at crack onset                                    [10 pts]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_score():
    import numpy as np, math
    checks = []
    total = 0

    E, nu, Gc = 70e3, 0.22, 0.01

    # C4.1: Strip SDF
    try:
        from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
        x_inside = np.array([[0.0, 1.0, 0.0]])
        x_crack = np.array([[-20.0, 0.0, 0.0]])
        v_in = sdf_cracked_plate(x_inside, a=10.0, W=25.0, H=5.0, T=0.5)
        v_cr = sdf_cracked_plate(x_crack, a=10.0, W=25.0, H=5.0, T=0.5)
        ok = v_in[0] < 0 and v_cr[0] > 0
        checks.append({"id": "C4.1", "name": "Strip SDF", "pass": ok, "pts": 10 if ok else 0, "max": 10})
        total += 10 if ok else 0
    except Exception as e:
        checks.append({"id": "C4.1", "name": "Strip SDF", "pass": False, "pts": 0, "max": 10, "error": str(e)})

    # C4.2: CrackTipDecoder
    try:
        import torch
        from solvers.fem.analytic_decoders import CrackTipDecoder
        decoder = CrackTipDecoder.from_crack_tip([-15, 0, 0], [1, 0, 0], [0, 1, 0], radius=0.5)
        xi = torch.tensor([[0, 0, 0.5]], dtype=torch.float64)
        x = decoder(xi)
        xi_back = decoder.inverse(x)
        ok = (xi - xi_back).abs().max().item() < 1e-8
        checks.append({"id": "C4.2", "name": "CrackTipDecoder", "pass": ok, "pts": 15 if ok else 0, "max": 15})
        total += 15 if ok else 0
    except Exception as e:
        checks.append({"id": "C4.2", "name": "CrackTipDecoder", "pass": False, "pts": 0, "max": 15, "error": str(e)})

    # C4.3: K_I extraction (from exact Williams data)
    try:
        from benchmarks.fracture.lefm_reference import stress_intensity_factor, williams_displacement, extract_K_I_from_displacement
        a, W = 10.0, 25.0
        K_I_exact = stress_intensity_factor(1.0, a, W)
        r_vals = np.linspace(0.05, 0.5*a, 15)
        theta_vals = np.linspace(-math.pi, math.pi, 24, endpoint=False)
        rr, tt = np.meshgrid(r_vals, theta_vals)
        _, u_y = williams_displacement(rr.ravel(), tt.ravel(), K_I_exact, E, nu, True)
        K_rec = extract_K_I_from_displacement(u_y, rr.ravel(), tt.ravel(), E, nu, True)
        err = abs(K_rec - K_I_exact) / K_I_exact
        ok = err < 0.1
        pts = 25 if err < 0.05 else (15 if err < 0.1 else 0)
        checks.append({"id": "C4.3", "name": f"K_I extraction ({err*100:.1f}%)", "pass": ok, "pts": pts, "max": 25})
        total += pts
    except Exception as e:
        checks.append({"id": "C4.3", "name": "K_I extraction", "pass": False, "pts": 0, "max": 25, "error": str(e)})

    # C4.4-C4.6: Not yet implemented with FEM
    H = 5.0
    h_crit = math.sqrt(Gc * H * 4 * (1 - nu**2) / E)
    for cid, name, pts, note in [
        ("C4.4", f"h_crit={h_crit*1e3:.2f}um", 25, "Needs chart FEM on strip"),
        ("C4.5", "Mode I straight ahead", 15, "Needs crack growth driver on strip"),
        ("C4.6", "G=Gc at onset", 10, "Needs energy release rate computation"),
    ]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts, "note": note})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
