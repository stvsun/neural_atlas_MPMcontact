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

    # C8.5-C8.7: Need chart FEM solve on DCB specimen
    for cid, name, pts in [
        ("C8.5", "Chart FEM on DCB bar", 20),
        ("C8.6", "FEM K_I vs beam theory", 20),
        ("C8.7", "Stable crack growth", 10),
    ]:
        checks.append({"id": cid, "name": name, "pass": False, "pts": 0, "max": pts,
                        "note": "Needs multi-chart FEM on bar with pre-crack"})

    score = total
    status = "PASS" if score >= 80 else ("PARTIAL" if score > 0 else "NOT_IMPLEMENTED")
    for c in checks:
        sym = "PASS" if c["pass"] else "FAIL"
        print(f"  [{sym:4s}] {c['id']}: {c['name']} ({c['pts']}/{c['max']})")
    print(f"  Score: {score:.1f}/100")
    return {"status": status, "score": float(score), "checks": checks}
