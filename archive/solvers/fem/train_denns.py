#!/usr/bin/env python3
"""Train DENNs SDF enrichment on fracture problems.

Uses direct supervision from the Williams asymptotic solution to train
the SDF-enriched decoder's enrichment MLP to represent crack-opening
displacement within the coordinate chart.

Training approach:
1. Sample points in reference space ξ
2. Map through base decoder to physical space x = decoder(ξ)
3. Compute Williams displacement u_Williams(x) at those points
4. Compute SDF embedding [d(x), H(d), |d|]
5. Train enrichment MLP: minimize ||MLP(ξ, sdf_embed) - u_Williams||²
6. The trained enrichment captures the crack-opening displacement

This gives the FEM solver a better coordinate mapping that pre-encodes
the crack displacement field, improving K_I extraction accuracy.

Reference:
    Zhao & Shao (CMAME 446, 2025): DENNs SDF enrichment
    Manav et al. (CMAME 429, 2024): RPROP optimizer for fracture energy landscapes
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.analytic_decoders import CrackTipDecoder
from solvers.fem.denns_enrichment import (
    SDFEnrichedDecoder, enrich_decoder, compute_sdf_embedding
)
from benchmarks.fracture.lefm_reference import williams_displacement
from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle


def make_crack_sdf_fn(sdf_oracle):
    """Create a torch-compatible SDF function from the oracle."""
    def sdf_fn(x_phys):
        with torch.no_grad():
            if isinstance(x_phys, torch.Tensor):
                return sdf_oracle.sdf(x_phys)
            x_t = torch.tensor(x_phys, dtype=torch.float64)
            return sdf_oracle.sdf(x_t)
    return sdf_fn


def train_denns_williams(
    crack_tip=(0.0, 0.0, 0.0),
    crack_dir=(1.0, 0.0, 0.0),
    opening_dir=(0.0, 1.0, 0.0),
    radius: float = 1.0,
    K_I: float = 1.0,
    E: float = 70e3,
    nu: float = 0.22,
    n_samples: int = 2000,
    n_epochs: int = 200,
    lr: float = 1e-3,
    verbose: bool = True,
) -> SDFEnrichedDecoder:
    """Train DENNs enrichment using Williams displacement as supervision.

    Parameters
    ----------
    crack_tip : tuple
        Crack tip position.
    crack_dir, opening_dir : tuple
        Crack frame.
    radius : float
        CrackTipDecoder radius.
    K_I : float
        Target stress intensity factor.
    E, nu : float
        Material properties.
    n_samples : int
        Number of training points per epoch.
    n_epochs : int
        Training iterations.
    lr : float
        Learning rate.

    Returns
    -------
    enriched : SDFEnrichedDecoder
        Trained enriched decoder.
    """
    dtype = torch.float64
    tip = np.array(crack_tip, dtype=np.float64)
    cd = np.array(crack_dir, dtype=np.float64); cd /= np.linalg.norm(cd)
    od = np.array(opening_dir, dtype=np.float64); od /= np.linalg.norm(od)

    # SDF oracle for crack
    sdf_oracle = CrackedPlateSDFOracle(
        a=radius * 2, W=radius * 3, H=radius * 3, T=radius * 2, delta=0.02
    )
    sdf_fn = make_crack_sdf_fn(sdf_oracle)

    # Base decoder
    base_dec = CrackTipDecoder.from_crack_tip(
        list(crack_tip), list(crack_dir), list(opening_dir), radius=radius
    ).double()

    # DENNs enriched decoder
    enriched = enrich_decoder(base_dec, sdf_fn, epsilon=0.05,
                               enrichment_width=48, enrichment_depth=3)

    # Optimizer: RPROP (Manav et al. 2024)
    optimizer = torch.optim.Rprop(
        list(enriched.enrichment_net.parameters()) + [enriched.raw_amplitude],
        lr=lr
    )

    if verbose:
        print(f"[DENNs] Training: K_I={K_I:.2f}, radius={radius:.1f}, "
              f"{n_samples} pts/epoch, {n_epochs} epochs")

    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sample random points in reference space [-0.9, 0.9]^3
        xi = (torch.rand(n_samples, 3, dtype=dtype) * 2 - 1) * 0.9

        # Map to physical space via base decoder
        with torch.no_grad():
            x_phys = base_dec(xi)
        x_np = x_phys.detach().cpu().numpy()

        # Compute Williams displacement at physical points
        dx = x_np - tip
        x1 = dx @ cd
        x2 = dx @ od
        r = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)

        ux_w, uy_w = williams_displacement(r, theta, K_I, E, nu, plane_strain=True)
        u_williams = np.zeros_like(x_np)
        u_williams[:, 0] = ux_w * cd[0] + uy_w * od[0]
        u_williams[:, 1] = ux_w * cd[1] + uy_w * od[1]
        u_williams[:, 2] = ux_w * cd[2] + uy_w * od[2]
        u_target = torch.tensor(u_williams, dtype=dtype)

        # Compute SDF embedding
        sdf_embed = compute_sdf_embedding(sdf_fn, x_phys, epsilon=0.05)

        # Forward pass through enrichment MLP
        enrichment_input = torch.cat([xi, sdf_embed.to(dtype)], dim=-1)
        correction = enriched.enrichment_net(enrichment_input)
        correction = enriched.amplitude * correction

        # Loss: ||correction - u_williams||² weighted by 1/r (emphasize near-tip)
        r_t = torch.tensor(r, dtype=dtype).clamp(min=1e-6)
        weights = 1.0 / r_t  # Weight near-tip points more
        weights = weights / weights.sum()

        diff = correction - u_target
        loss = (weights.unsqueeze(1) * diff**2).sum()

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                'net': {k: v.clone() for k, v in enriched.enrichment_net.state_dict().items()},
                'amp': enriched.raw_amplitude.data.clone(),
            }

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            amp = enriched.amplitude.item()
            max_corr = correction.abs().max().item()
            max_tgt = u_target.abs().max().item()
            ratio = max_corr / max_tgt if max_tgt > 0 else 0
            print(f"  epoch {epoch:4d}: loss={loss.item():.6e}, amp={amp:.4f}, "
                  f"max|corr|/max|target|={ratio:.3f}")

    # Restore best
    if best_state is not None:
        enriched.enrichment_net.load_state_dict(best_state['net'])
        enriched.raw_amplitude.data = best_state['amp']

    dt = time.time() - t0
    if verbose:
        print(f"[DENNs] Done in {dt:.1f}s. Best loss={best_loss:.6e}, "
              f"amp={enriched.amplitude.item():.4f}")

    return enriched


def train_denns_dcb(
    E=70e3, nu=0.22, Gc=0.01,
    L=55.0, A=25.0, H=20.0, B=2.5,
    n_epochs=200, verbose=True,
):
    """Train DENNs for DCB problem."""
    K_Ic = math.sqrt(E * Gc / (1 - nu**2))
    tip_x = -L / 2 + A

    if verbose:
        print(f"[DENNs-DCB] K_Ic={K_Ic:.2f}, tip_x={tip_x:.1f}")

    enriched = train_denns_williams(
        crack_tip=(tip_x, 0.0, 0.0),
        crack_dir=(1.0, 0.0, 0.0),
        opening_dir=(0.0, 1.0, 0.0),
        radius=5.0, K_I=K_Ic, E=E, nu=nu,
        n_samples=3000, n_epochs=n_epochs, lr=5e-4,
        verbose=verbose,
    )
    return enriched


if __name__ == "__main__":
    print("=" * 60)
    print("  DENNs SDF Enrichment Training")
    print("=" * 60)

    # 1. Train on pure-shear Williams field
    print("\n--- C4: Williams field (pure-shear) ---")
    enriched_c4 = train_denns_williams(
        crack_tip=(0.0, 0.0, 0.0),
        crack_dir=(1.0, 0.0, 0.0),
        opening_dir=(0.0, 1.0, 0.0),
        radius=1.0, K_I=27.12, E=70e3, nu=0.22,
        n_samples=2000, n_epochs=200, lr=1e-3,
    )

    os.makedirs("runs/denns_training", exist_ok=True)
    torch.save(enriched_c4.state_dict(), "runs/denns_training/enriched_c4_williams.pt")
    print(f"Saved: runs/denns_training/enriched_c4_williams.pt")

    # 2. Train on DCB
    print("\n--- C8: DCB geometry ---")
    enriched_c8 = train_denns_dcb(n_epochs=200)
    torch.save(enriched_c8.state_dict(), "runs/denns_training/enriched_c8_dcb.pt")
    print(f"Saved: runs/denns_training/enriched_c8_dcb.pt")

    # 3. Validate: use enriched decoder in FEM solve
    print("\n--- Validation: FEM with enriched decoder ---")
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic_small_strain
    from solvers.fem.k_extraction import extract_K_from_fem

    E, nu = 70e3, 0.22
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
    K_Ic = math.sqrt(E * 0.01 / (1 - nu**2))

    # Build solver with enriched decoder
    sdf = CrackedPlateSDFOracle(a=10, W=3, H=3, T=2, delta=0.02)
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=enriched_c4,
        decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
        device="cpu", dtype=torch.float64,
    )

    # Apply Williams BCs and solve
    nodes = solver.nodes_phys.detach().cpu().numpy()
    r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    ux, uy = williams_displacement(r, theta, K_Ic, E, nu)
    u_bc = np.zeros_like(nodes)
    u_bc[:, 0] = ux; u_bc[:, 1] = uy
    u_t = torch.tensor(u_bc, dtype=torch.float64)

    f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
    u_sol = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_t,
                                    solver.boundary_mask, max_iter=10, tol=1e-10)

    K_I_extracted = extract_K_from_fem(solver, u_sol, [0,0,0], [1,0,0], [0,1,0], E, nu)
    err = abs(K_I_extracted - K_Ic) / K_Ic * 100
    print(f"K_I extracted: {K_I_extracted:.2f} vs K_Ic={K_Ic:.2f} (err={err:.1f}%)")
    print(f"Enrichment amplitude: {enriched_c4.amplitude.item():.4f}")
