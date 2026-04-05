"""Decoder Jacobian quality tests (Stage 2 global test G2)."""
import numpy as np
import torch

def run_decoder_quality_test(solver, decoder):
    """Test decoder Jacobian: condition number, positivity, anisotropy.

    Returns dict with pts (out of 15).
    """
    results = {"pts": 0, "max": 15, "checks": []}

    try:
        # Sample points in reference domain
        xi = solver.elem_centroids_ref
        if len(xi) == 0:
            results["checks"].append({"id": "G2", "name": "No elements", "pass": False, "pts": 0})
            return results

        J = solver.decoder_jacobian(xi)  # (M, 3, 3)

        # G2.1: Condition number < 100
        s = torch.linalg.svdvals(J)
        kappa = s[:, 0] / s[:, -1].clamp(min=1e-15)
        max_kappa = kappa.max().item()
        ok = max_kappa < 100
        pts = 5 if ok else 0
        results["checks"].append({"id": "G2.1", "name": f"kappa(J)={max_kappa:.1f}", "pass": ok, "pts": pts})
        results["pts"] += pts

        # G2.2: det(J) > 0 everywhere
        detJ = torch.det(J)
        min_det = detJ.min().item()
        ok = min_det > 0
        pts = 5 if ok else 0
        results["checks"].append({"id": "G2.2", "name": f"min det(J)={min_det:.4e}", "pass": ok, "pts": pts})
        results["pts"] += pts

        # G2.3: Metric tensor eigenvalue ratio < 50
        G = torch.einsum("eij,eik->ejk", J, J)  # metric tensor
        eigvals = torch.linalg.eigvalsh(G)
        ratio = eigvals[:, -1] / eigvals[:, 0].clamp(min=1e-15)
        max_ratio = ratio.max().item()
        ok = max_ratio < 50
        pts = 5 if ok else 0
        results["checks"].append({"id": "G2.3", "name": f"Metric ratio={max_ratio:.1f}", "pass": ok, "pts": pts})
        results["pts"] += pts

    except Exception as e:
        results["checks"].append({"id": "G2", "name": f"Error: {e}", "pass": False, "pts": 0})

    return results
