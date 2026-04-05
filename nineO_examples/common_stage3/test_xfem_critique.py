"""XFEM-critic tests for fracture mechanics rigor (Stage 3).

Each test targets a specific limitation that a mature XFEM code would not have.
Tests are designed to FAIL on the current implementation, revealing gaps.

Point allocation per challenge: up to 100 pts across ~10 checks.
"""
import math
import numpy as np
import torch
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# X1: Williams enrichment completeness
# ---------------------------------------------------------------------------

def test_williams_angular_modes(solver, decoder, K_I, E, nu, crack_tip, crack_dir, opening_dir):
    """Test that ALL 4 Williams branch functions are captured, not just sqrt(r).

    XFEM enriches with: {sqrt(r)*cos(theta/2), sqrt(r)*sin(theta/2),
                          sqrt(r)*sin(theta/2)*sin(theta), sqrt(r)*cos(theta/2)*sin(theta)}

    The CrackTipDecoder only captures the radial sqrt(r) part. This test checks
    whether the angular modes cos(theta/2), sin(theta/2) are resolved.

    Returns dict with pts (max 15).
    """
    result = {"id": "X1", "max": 15, "pts": 0, "checks": []}

    try:
        from benchmarks.fracture.lefm_reference import williams_displacement

        nodes = solver.nodes_phys.detach().cpu().numpy()
        tip = np.array(crack_tip, dtype=np.float64)
        cd = np.array(crack_dir, dtype=np.float64)
        od = np.array(opening_dir, dtype=np.float64)
        cd /= np.linalg.norm(cd); od /= np.linalg.norm(od)

        dx = nodes - tip
        r = np.sqrt((dx @ cd)**2 + (dx @ od)**2)
        theta = np.arctan2(dx @ od, dx @ cd)

        # Williams displacement
        ux_w, uy_w = williams_displacement(r, theta, K_I, E, nu)
        u_bc = np.zeros_like(nodes)
        u_bc[:, 0] = ux_w * cd[0] + uy_w * od[0]
        u_bc[:, 1] = ux_w * cd[1] + uy_w * od[1]
        u_target = torch.tensor(u_bc, dtype=solver.dtype, device=solver.device)

        from solvers.fem.linear_elastic import make_linear_elastic_small_strain
        stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

        # Solve with Williams BCs on boundary
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=solver.dtype, device=solver.device)
        u_sol = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_target,
                                        solver.boundary_mask, max_iter=10, tol=1e-10)

        interior = ~solver.boundary_mask
        if interior.sum() == 0:
            result["checks"].append({"name": "No interior nodes", "pass": False, "pts": 0})
            return result

        # Error at interior nodes (where enrichment matters)
        err_int = (u_sol[interior] - u_target[interior]).norm().item()
        ref_int = u_target[interior].norm().item()
        rel_err = err_int / max(ref_int, 1e-15)

        # Check angular mode resolution:
        # Near theta=0 (ahead of crack): u_y should be 0 (symmetry)
        # Near theta=pi/2 (perpendicular): u_y should be maximal
        # If only sqrt(r) is captured (no angular), u_y would be constant in theta
        int_np = interior.cpu().numpy().astype(bool)
        near_ahead = (np.abs(theta) < 0.3) & (r > 0.1 * r.max()) & int_np
        near_perp = (np.abs(theta - np.pi/2) < 0.3) & (r > 0.1 * r.max()) & int_np

        theta_variation_ok = False
        if near_ahead.sum() > 0 and near_perp.sum() > 0:
            u_ahead = u_sol[near_ahead, 1].abs().mean().item()
            u_perp = u_sol[near_perp, 1].abs().mean().item()
            ratio = u_perp / max(u_ahead, 1e-15)
            theta_variation_ok = ratio > 2.0  # perpendicular should be >> ahead
            result["checks"].append({
                "name": f"Angular variation ratio={ratio:.2f} (need >2)",
                "pass": theta_variation_ok, "pts": 5 if theta_variation_ok else 0
            })
        else:
            result["checks"].append({"name": "Not enough angular samples", "pass": False, "pts": 0})

        # X1.2: Interior error < 5% (XFEM standard on comparable mesh)
        ok_err = rel_err < 0.05
        pts_err = 5 if rel_err < 0.05 else (3 if rel_err < 0.10 else (1 if rel_err < 0.30 else 0))
        result["checks"].append({
            "name": f"Williams interior err={rel_err*100:.1f}% (XFEM target <5%)",
            "pass": ok_err, "pts": pts_err
        })

        # X1.3: Check all 4 branch functions are non-zero
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu
        # Branch function 1: sqrt(r)*cos(theta/2)
        bf1 = np.sqrt(r) * np.cos(theta / 2)
        # Branch function 2: sqrt(r)*sin(theta/2)
        bf2 = np.sqrt(r) * np.sin(theta / 2)
        # Check that both contribute to the solution
        u_y_sol = u_sol[:, 1].detach().cpu().numpy()
        corr_bf1 = np.abs(np.corrcoef(bf1[int_np], u_y_sol[int_np])[0, 1])
        corr_bf2 = np.abs(np.corrcoef(bf2[int_np], u_y_sol[int_np])[0, 1])
        both_active = corr_bf1 > 0.3 and corr_bf2 > 0.3
        result["checks"].append({
            "name": f"Branch functions: corr(bf1)={corr_bf1:.2f}, corr(bf2)={corr_bf2:.2f}",
            "pass": both_active, "pts": 5 if both_active else 0
        })

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X2: Crack-face traction-free enforcement
# ---------------------------------------------------------------------------

def test_crack_face_traction_free(solver, stress_fn, u_sol, crack_tip, crack_dir, opening_dir):
    """Test that traction vanishes on crack faces (theta = ±pi).

    XFEM automatically enforces this via Heaviside enrichment.
    The neural atlas relies on SDF filtering but doesn't explicitly enforce it.

    Returns dict with pts (max 10).
    """
    result = {"id": "X2", "max": 10, "pts": 0, "checks": []}

    try:
        nodes = solver.nodes_phys.detach().cpu().numpy()
        tip = np.array(crack_tip, dtype=np.float64)
        cd = np.array(crack_dir, dtype=np.float64)
        od = np.array(opening_dir, dtype=np.float64)
        cd /= np.linalg.norm(cd); od /= np.linalg.norm(od)

        dx = nodes - tip
        x1 = dx @ cd; x2 = dx @ od
        r = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)

        # Elements near crack faces: |theta| close to pi, r > small
        F = solver.compute_F(u_sol)
        P = stress_fn(F)
        P_np = P.detach().cpu().numpy()

        centroids = solver.elem_centroids_phys.detach().cpu().numpy()
        dc = centroids - tip
        c_x1 = dc @ cd; c_x2 = dc @ od
        c_r = np.sqrt(c_x1**2 + c_x2**2)
        c_theta = np.arctan2(c_x2, c_x1)

        # Crack face elements: |theta| > 2.5 rad (close to pi), r not too small
        r_max = c_r.max()
        crack_face = (np.abs(c_theta) > 2.5) & (c_r > 0.05 * r_max) & (c_r < 0.5 * r_max)

        if crack_face.sum() < 3:
            result["checks"].append({"name": "Not enough crack-face elements", "pass": False, "pts": 0})
            return result

        # Traction on crack face: sigma_yy should be ~0
        sigma_yy = P_np[crack_face, 1, 1]
        sigma_ref = np.abs(P_np[:, 1, 1]).max()
        traction_ratio = np.abs(sigma_yy).max() / max(sigma_ref, 1e-15)

        ok = traction_ratio < 0.05
        pts = 10 if traction_ratio < 0.05 else (5 if traction_ratio < 0.15 else (2 if traction_ratio < 0.30 else 0))
        result["checks"].append({
            "name": f"Crack-face |sigma_yy|/max = {traction_ratio:.3f} (XFEM target <0.05)",
            "pass": ok, "pts": pts
        })
        result["pts"] = pts

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X3: Mixed-mode capability
# ---------------------------------------------------------------------------

def test_mixed_mode_extraction(solver, E, nu, crack_tip, crack_dir, opening_dir):
    """Test K_II extraction capability.

    XFEM codes routinely extract K_I and K_II via interaction integral.
    The neural atlas only extracts K_I.

    Returns dict with pts (max 10).
    """
    result = {"id": "X3", "max": 10, "pts": 0, "checks": []}

    try:
        # Check if K_II extraction function exists
        from solvers.fem import k_extraction
        has_kii = hasattr(k_extraction, 'extract_K_II_from_fem') or \
                  hasattr(k_extraction, 'extract_K_II_from_charts')
        result["checks"].append({
            "name": f"K_II extraction function exists: {has_kii}",
            "pass": has_kii, "pts": 5 if has_kii else 0
        })

        # Check if interaction integral (M-integral) exists
        has_m_integral = False
        try:
            from solvers.fem.j_integral import compute_interaction_integral
            has_m_integral = True
        except ImportError:
            pass
        result["checks"].append({
            "name": f"Interaction integral (M-integral) exists: {has_m_integral}",
            "pass": has_m_integral, "pts": 5 if has_m_integral else 0
        })

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X4: Curved crack path capability
# ---------------------------------------------------------------------------

def test_curved_crack_path():
    """Test whether crack propagation supports curved/kinked paths.

    XFEM: crack propagates in max hoop stress direction, path updates each step.
    Neural atlas: crack direction is fixed at initialization.

    Returns dict with pts (max 10).
    """
    result = {"id": "X4", "max": 10, "pts": 0, "checks": []}

    try:
        from solvers.fem.crack_propagation import propagate_crack
        import inspect
        src = inspect.getsource(propagate_crack)

        # Check 1: Does propagation update crack direction?
        updates_dir = "crack_direction" in src and ("rotate" in src or "angle" in src or "theta_c" in src)
        result["checks"].append({
            "name": f"Propagation updates crack direction: {updates_dir}",
            "pass": updates_dir, "pts": 5 if updates_dir else 0
        })

        # Check 2: Does it call max_hoop_stress_angle?
        uses_mhs = "max_hoop_stress_angle" in src
        result["checks"].append({
            "name": f"Uses max_hoop_stress_angle criterion: {uses_mhs}",
            "pass": uses_mhs, "pts": 3 if uses_mhs else 0
        })

        # Check 3: Does it support branching?
        has_branching = "branch" in src.lower()
        result["checks"].append({
            "name": f"Branching criterion exists: {has_branching}",
            "pass": has_branching, "pts": 2 if has_branching else 0
        })

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X5: Displacement discontinuity at chart overlaps
# ---------------------------------------------------------------------------

def test_displacement_discontinuity_handling():
    """Test whether Schwarz handles displacement discontinuity across cracks.

    XFEM: Heaviside enrichment provides jump in u across crack face.
    Neural atlas: Schwarz interpolation assumes continuous displacement.

    Returns dict with pts (max 10).
    """
    result = {"id": "X5", "max": 10, "pts": 0, "checks": []}

    try:
        from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver
        import inspect
        src = inspect.getsource(SchwarzVectorFEMSolver._interpolate_from_neighbors)

        # Check 1: Does interpolation check for crack face crossing?
        checks_crack = "crack" in src.lower() or "discontinu" in src.lower() or "heaviside" in src.lower()
        result["checks"].append({
            "name": f"Checks crack-face crossing in interpolation: {checks_crack}",
            "pass": checks_crack, "pts": 5 if checks_crack else 0
        })

        # Check 2: Does it use partition of unity?
        has_pou = "partition" in src.lower() or "pou" in src.lower()
        # Check for proper PoU: sum of weights = 1
        has_weight_norm = "weight" in src and ("/ weight" in src or "normalize" in src.lower())
        result["checks"].append({
            "name": f"Partition of unity blending: {has_pou or has_weight_norm}",
            "pass": has_pou or has_weight_norm, "pts": 5 if (has_pou or has_weight_norm) else 0
        })

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X6: Stiffness matrix conditioning
# ---------------------------------------------------------------------------

def test_stiffness_conditioning(solver, stress_fn, tangent_fn):
    """Test stiffness matrix condition number.

    XFEM: condition number is monitored; preconditioning applied.
    Neural atlas: no conditioning check.

    Returns dict with pts (max 10).
    """
    result = {"id": "X6", "max": 10, "pts": 0, "checks": []}

    try:
        u_zero = torch.zeros(solver.n_nodes, 3, dtype=solver.dtype, device=solver.device)
        K = solver.tangent_stiffness(u_zero, tangent_fn)

        # Check condition number with diagonal scaling (standard preconditioner)
        n_dof = K.shape[0]
        if n_dof <= 3000:
            # Apply diagonal scaling: K_scaled = D^{-1/2} K D^{-1/2}
            diag_K = torch.diag(K).abs().clamp(min=1e-15)
            D_inv_sqrt = 1.0 / torch.sqrt(diag_K)
            K_scaled = K * D_inv_sqrt.unsqueeze(0) * D_inv_sqrt.unsqueeze(1)

            s = torch.linalg.svdvals(K_scaled)
            s_pos = s[s > 1e-15]
            if len(s_pos) > 0:
                cond = (s_pos[0] / s_pos[-1]).item()
            else:
                cond = float('inf')
        else:
            cond = float('nan')

        # After diagonal scaling, condition number is more meaningful
        ok = cond < 1e6
        pts = 10 if cond < 1e6 else (5 if cond < 1e8 else (2 if cond < 1e10 else 0))
        result["checks"].append({
            "name": f"cond(D^-½KD^-½) = {cond:.2e} (XFEM target <1e6)",
            "pass": ok, "pts": pts
        })
        result["pts"] = pts

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X7: Integration accuracy near singularity
# ---------------------------------------------------------------------------

def test_integration_near_singularity(solver):
    """Test whether quadrature is adequate near crack tip.

    XFEM: uses enriched quadrature or subtriangulation for cut elements.
    Neural atlas: standard Gauss quadrature on all elements.

    Returns dict with pts (max 10).
    """
    result = {"id": "X7", "max": 10, "pts": 0, "checks": []}

    try:
        # Check 1: Are there elements with very small volume (degenerate)?
        vol = solver.vol.detach().cpu().numpy()
        vol_ratio = vol.max() / max(vol.min(), 1e-30)

        ok_vol = vol_ratio < 1e4
        result["checks"].append({
            "name": f"Element volume ratio = {vol_ratio:.2e} (target <1e4)",
            "pass": ok_vol, "pts": 5 if ok_vol else (2 if vol_ratio < 1e6 else 0)
        })

        # Check 2: Are there elements with inverted Jacobian?
        detJ = solver.geom_detJ.detach().cpu().numpy()
        n_inverted = (detJ <= 0).sum()
        ok_inv = n_inverted == 0
        result["checks"].append({
            "name": f"Inverted elements: {n_inverted}/{len(detJ)} (target: 0)",
            "pass": ok_inv, "pts": 5 if ok_inv else 0
        })

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X8: Nucleation mesh independence
# ---------------------------------------------------------------------------

def test_nucleation_mesh_independence(decoder_cls, decoder_args, sdf_oracle,
                                       stress_fn, tangent_fn, bc_fn, sigma_ts, sigma_hs):
    """Test that nucleation location is mesh-independent.

    XFEM with nonlocal/gradient damage: nucleation location converges with refinement.
    Pointwise Drucker-Prager: nucleation may jump between elements.

    sdf_oracle can be either:
      - An object with .sdf(x) method (CrackedPlateSDFOracle)
      - A callable function x -> d (will be wrapped)
      - None (skip SDF filtering)

    Returns dict with pts (max 15).
    """
    result = {"id": "X8", "max": 15, "pts": 0, "checks": []}

    try:
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fracture_criteria import drucker_prager_F

        # Wrap callable SDF into an object with .sdf() method if needed
        actual_sdf = sdf_oracle
        if sdf_oracle is not None and not hasattr(sdf_oracle, 'sdf'):
            class _SDFWrap:
                def __init__(self, fn):
                    self._fn = fn
                def sdf(self, x):
                    return self._fn(x)
            actual_sdf = _SDFWrap(sdf_oracle)

        nuc_locations = []
        for nc in [6, 8, 10, 12]:
            dec = decoder_cls(**decoder_args).double()
            s = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec,
                                      decoder_kwargs={}, sdf_oracle=actual_sdf,
                                      sdf_threshold=-0.01, device="cpu", dtype=torch.float64)
            if s.n_elements == 0:
                continue

            nodes_np = s.nodes_phys.detach().cpu().numpy()
            u_bc_vals, mask = bc_fn(nodes_np)
            u_bc = torch.tensor(u_bc_vals, dtype=torch.float64)
            bc_mask = torch.tensor(mask, dtype=torch.bool)
            f_ext = torch.zeros(s.n_nodes, 3, dtype=torch.float64)

            u_sol = s.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
                                       max_iter=10, tol=1e-8)
            F = s.compute_F(u_sol)
            P = stress_fn(F)

            # Find max Drucker-Prager location
            P_np = P.detach().cpu().numpy()
            centroids = s.elem_centroids_phys.detach().cpu().numpy()

            F_dp_vals = []
            for e in range(len(P_np)):
                sigma = P_np[e]
                val = drucker_prager_F(
                    torch.tensor(sigma).unsqueeze(0), sigma_ts, sigma_hs
                ).item()
                F_dp_vals.append(val)
            F_dp_vals = np.array(F_dp_vals)

            max_idx = np.argmax(F_dp_vals)
            nuc_locations.append(centroids[max_idx])

        if len(nuc_locations) >= 3:
            # Check convergence: distance between consecutive nucleation locations
            dists = []
            for i in range(1, len(nuc_locations)):
                d = np.linalg.norm(nuc_locations[i] - nuc_locations[i-1])
                dists.append(d)

            max_dist = max(dists)
            mean_dist = np.mean(dists)
            converging = dists[-1] < dists[0] if len(dists) >= 2 else False

            ok = max_dist < 0.5  # nucleation location converges to within 0.5 mm
            pts = 10 if max_dist < 0.2 else (5 if max_dist < 0.5 else (2 if max_dist < 1.0 else 0))
            result["checks"].append({
                "name": f"Nucleation location scatter: max_d={max_dist:.3f}mm, converging={converging}",
                "pass": ok, "pts": pts
            })

            # Check: is there gradient/nonlocal regularization?
            has_nonlocal = False
            try:
                from solvers.fem import nonlocal_damage
                has_nonlocal = True
            except ImportError:
                pass
            result["checks"].append({
                "name": f"Nonlocal/gradient regularization: {has_nonlocal}",
                "pass": has_nonlocal, "pts": 5 if has_nonlocal else 0
            })
        else:
            result["checks"].append({"name": "Not enough meshes generated", "pass": False, "pts": 0})

        result["pts"] = sum(c.get("pts", 0) for c in result["checks"])

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result


# ---------------------------------------------------------------------------
# X9: K_I accuracy vs XFEM benchmark
# ---------------------------------------------------------------------------

def test_ki_accuracy_vs_xfem(K_I_extracted, K_I_analytical):
    """Compare K_I accuracy against XFEM standards.

    XFEM achieves <1% K_I error on comparable meshes.
    Neural atlas: currently ~26% error (C8).

    Returns dict with pts (max 10).
    """
    result = {"id": "X9", "max": 10, "pts": 0, "checks": []}

    try:
        if math.isnan(K_I_extracted) or K_I_analytical <= 0:
            result["checks"].append({"name": "K_I invalid", "pass": False, "pts": 0})
            return result

        err = abs(K_I_extracted - K_I_analytical) / K_I_analytical
        # XFEM standard: <1% on mesh with similar DOF count
        pts = 10 if err < 0.01 else (7 if err < 0.05 else (4 if err < 0.10 else (2 if err < 0.25 else 0)))
        result["checks"].append({
            "name": f"K_I err = {err*100:.1f}% (XFEM target <1%)",
            "pass": err < 0.01, "pts": pts
        })
        result["pts"] = pts

    except Exception as e:
        result["checks"].append({"name": f"Error: {e}", "pass": False, "pts": 0})

    return result
