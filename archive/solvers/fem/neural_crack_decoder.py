#!/usr/bin/env python3
"""Neural network crack-tip decoder: pre-trained to absorb the sqrt(r) singularity.

Instead of using the analytical CrackTipDecoder (r_phys = radius * t^power),
this module trains a small MLP to learn the optimal coordinate mapping from
the Williams asymptotic displacement field. The trained decoder can then
be used as a drop-in replacement in ChartVectorFEMSolver.

Training objective:
    Given Williams displacement u(r, theta) = K_I/(2mu) * sqrt(r/(2pi)) * f(theta),
    find a mapping phi: xi -> x such that u(phi(xi)) is maximally smooth in xi-space.

The MLP learns a residual on top of the analytical power-law mapping:
    x = analytical_base(xi) + MLP(xi)

This allows the network to adapt to the specific crack geometry while
preserving the singularity absorption from the power-law base.

Usage:
    # Train a decoder for a specific crack tip
    decoder = train_crack_decoder(
        tip=[0, 0, 0], crack_dir=[1, 0, 0], opening_dir=[0, 1, 0],
        radius=0.5, K_I=1.0, E=200.0, nu=0.3
    )

    # Use in FEM solver
    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0,
        chart_decoder=decoder, decoder_kwargs={},
    )
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.analytic_decoders import CrackTipDecoder
from benchmarks.fracture.lefm_reference import williams_displacement


class NeuralCrackDecoder(nn.Module):
    """Neural network crack-tip decoder with analytical base + learned residual.

    Architecture:
        x = CrackTipDecoder_base(xi) + amplitude * MLP(xi)

    The base provides the power-law singularity absorption (r ~ xi^2).
    The MLP learns corrections that further smooth the Williams field
    in reference space.

    Parameters
    ----------
    base_decoder : CrackTipDecoder
        Analytical base mapping (provides the singularity absorption).
    width : int
        MLP hidden layer width.
    depth : int
        MLP number of hidden layers.
    amplitude : float
        Scale of the learned residual (default 0.1 * radius).
    """

    def __init__(
        self,
        base_decoder: CrackTipDecoder,
        width: int = 32,
        depth: int = 3,
        amplitude: Optional[float] = None,
    ):
        super().__init__()
        self.base = base_decoder
        self.amplitude = amplitude if amplitude is not None else 0.1 * base_decoder.radius

        # MLP: 3 -> width -> ... -> width -> 3
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 3))
        self.mlp = nn.Sequential(*layers)

        # Initialize to near-zero output
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, xi, **kwargs):
        """Map reference coords to physical space: base + residual."""
        x_base = self.base(xi)
        residual = self.amplitude * self.mlp(xi)
        return x_base + residual

    def inverse(self, x_phys):
        """Approximate inverse via base decoder inverse (ignores residual)."""
        return self.base.inverse(x_phys)

    def jacobian(self, xi, **kwargs):
        """Analytical base Jacobian (autograd handles residual in FEM)."""
        return self.base.jacobian(xi)

    @property
    def radius(self):
        return self.base.radius


def train_crack_decoder(
    tip=(0, 0, 0),
    crack_dir=(1, 0, 0),
    opening_dir=(0, 1, 0),
    radius: float = 0.5,
    K_I: float = 1.0,
    E: float = 200.0,
    nu: float = 0.3,
    power: float = 2.0,
    width: int = 32,
    depth: int = 3,
    n_train: int = 2000,
    n_epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> NeuralCrackDecoder:
    """Train a neural crack-tip decoder on the Williams asymptotic field.

    The training minimizes the smoothness of the Williams displacement
    when mapped through the decoder: min || d^2 u(phi(xi)) / dxi^2 ||.

    Parameters
    ----------
    tip, crack_dir, opening_dir : array-like
        Crack tip geometry.
    radius : float
        Support radius.
    K_I : float
        Stress intensity factor for training data.
    E, nu : float
        Elastic constants for Williams field.
    power : float
        Base power-law exponent.
    width, depth : int
        MLP architecture.
    n_train : int
        Number of training points.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.

    Returns
    -------
    decoder : NeuralCrackDecoder
        Trained decoder ready for ChartVectorFEMSolver.
    """
    dtype = torch.float64

    # Build base decoder
    base = CrackTipDecoder.from_crack_tip(
        tip, crack_dir, opening_dir, radius=radius, power=power,
    ).double().to(device)

    # Build neural decoder
    decoder = NeuralCrackDecoder(base, width=width, depth=depth).double().to(device)

    # Training points: sample xi in [-0.95, 0.95]^3
    # Avoid xi_2 = -1 (singular tip)
    optimizer = torch.optim.Adam(decoder.mlp.parameters(), lr=lr)

    mu = E / (2 * (1 + nu))
    kappa = 3 - 4 * nu  # plane strain

    if verbose:
        print(f"  Training NeuralCrackDecoder: {width}x{depth} MLP, "
              f"{n_train} points, {n_epochs} epochs")

    for epoch in range(n_epochs):
        # Random training points
        xi = torch.rand(n_train, 3, dtype=dtype, device=device) * 1.8 - 0.9
        xi[:, 2] = xi[:, 2].clamp(min=-0.9)  # avoid tip
        xi.requires_grad_(True)

        # Forward map to physical space
        x = decoder(xi)

        # Compute Williams displacement at physical points
        dx = x[:, 0].detach() - base.center_np[0]
        dy = x[:, 1].detach() - base.center_np[1]
        # Project onto crack-plane coords
        t1 = torch.tensor(base.tangent1_np, dtype=dtype, device=device)
        t2 = torch.tensor(base.tangent2_np, dtype=dtype, device=device)
        n_vec = torch.tensor(base.normal_np, dtype=dtype, device=device)

        dx_vec = x.detach() - torch.tensor(base.center_np, dtype=dtype, device=device)
        x_local = torch.sum(dx_vec * t1, dim=1)
        y_local = torch.sum(dx_vec * n_vec, dim=1)

        r = torch.sqrt(x_local**2 + y_local**2).clamp(min=1e-10)
        theta = torch.atan2(y_local, x_local)

        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()

        u_x_w, u_y_w = williams_displacement(r_np, theta_np, K_I, E, nu, True)
        u_williams = torch.tensor(
            np.column_stack([u_x_w, u_y_w, np.zeros_like(u_x_w)]),
            dtype=dtype, device=device,
        )

        # Loss: smoothness of u_williams in xi-space
        # Approximate via finite differences in xi
        h = 0.02
        loss = torch.tensor(0.0, dtype=dtype, device=device)
        for d in range(3):
            xi_plus = xi.detach().clone()
            xi_plus[:, d] += h
            xi_minus = xi.detach().clone()
            xi_minus[:, d] -= h

            x_plus = decoder(xi_plus)
            x_minus = decoder(xi_minus)

            # Physical points at xi +/- h
            dx_p = x_plus.detach() - torch.tensor(base.center_np, dtype=dtype, device=device)
            dx_m = x_minus.detach() - torch.tensor(base.center_np, dtype=dtype, device=device)

            r_p = torch.sqrt(torch.sum(dx_p * t1, dim=1)**2 + torch.sum(dx_p * n_vec, dim=1)**2).clamp(min=1e-10)
            r_m = torch.sqrt(torch.sum(dx_m * t1, dim=1)**2 + torch.sum(dx_m * n_vec, dim=1)**2).clamp(min=1e-10)

            # Smoothness: penalize variation in sqrt(r) mapping
            # We want sqrt(r(xi)) to be smooth => penalize d^2/dxi^2 of sqrt(r)
            sqrt_r_center = torch.sqrt(r)
            sqrt_r_plus = torch.sqrt(r_p)
            sqrt_r_minus = torch.sqrt(r_m)

            # Second derivative: (f+ - 2f + f-) / h^2
            d2_sqrt_r = (sqrt_r_plus - 2 * sqrt_r_center + sqrt_r_minus) / h**2
            loss = loss + torch.mean(d2_sqrt_r**2)

        # Regularization: keep residual small
        xi_reg = torch.rand(200, 3, dtype=dtype, device=device) * 1.8 - 0.9
        xi_reg[:, 2] = xi_reg[:, 2].clamp(min=-0.9)
        residual = decoder.mlp(xi_reg)
        loss = loss + 0.1 * torch.mean(residual**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss = {loss.item():.6f}")

    if verbose:
        print(f"  Training complete. Final loss = {loss.item():.6f}")

    decoder.eval()
    for p in decoder.mlp.parameters():
        p.requires_grad_(False)

    return decoder


def save_crack_decoder(decoder: NeuralCrackDecoder, path: str):
    """Save trained decoder to disk."""
    torch.save({
        "mlp_state": decoder.mlp.state_dict(),
        "amplitude": decoder.amplitude,
        "base_params": {
            "center": decoder.base.center_np.tolist(),
            "normal": decoder.base.normal_np.tolist(),
            "tangent1": decoder.base.tangent1_np.tolist(),
            "tangent2": decoder.base.tangent2_np.tolist(),
            "radius": decoder.base.radius,
            "power": decoder.base.power,
            "in_plane_scale": decoder.base.in_plane_scale,
        },
        "width": decoder.mlp[0].in_features,  # first layer input
        "depth": sum(1 for m in decoder.mlp if isinstance(m, nn.Linear)) - 1,
    }, path)
    print(f"  Saved: {path}")


def load_crack_decoder(path: str, device: str = "cpu") -> NeuralCrackDecoder:
    """Load a pre-trained decoder from disk."""
    ckpt = torch.load(path, map_location=device)

    bp = ckpt["base_params"]
    base = CrackTipDecoder(
        center=bp["center"], normal=bp["normal"],
        tangent1=bp["tangent1"], tangent2=bp["tangent2"],
        radius=bp["radius"], power=bp["power"],
        in_plane_scale=bp["in_plane_scale"],
    ).double()

    # Infer width from first linear layer
    first_linear = None
    for key in ckpt["mlp_state"]:
        if "weight" in key:
            first_linear = ckpt["mlp_state"][key]
            break
    width = first_linear.shape[0] if first_linear is not None else 32
    depth = ckpt.get("depth", 3)

    decoder = NeuralCrackDecoder(base, width=width, depth=depth,
                                  amplitude=ckpt["amplitude"]).double()
    decoder.mlp.load_state_dict(ckpt["mlp_state"])
    decoder.eval()
    for p in decoder.mlp.parameters():
        p.requires_grad_(False)

    return decoder


if __name__ == "__main__":
    # Train and save a default crack-tip decoder
    print("=== Training Default NeuralCrackDecoder ===")
    decoder = train_crack_decoder(
        tip=[0, 0, 0], crack_dir=[1, 0, 0], opening_dir=[0, 1, 0],
        radius=0.5, K_I=1.0, E=200.0, nu=0.3,
        width=32, depth=3, n_epochs=500,
    )

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "runs", "pretrained_decoders")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "crack_tip_decoder_default.pt")
    save_crack_decoder(decoder, save_path)

    # Verify roundtrip
    xi = torch.tensor([[0.3, 0.1, 0.5]], dtype=torch.float64)
    x = decoder(xi)
    print(f"  xi = {xi.numpy()[0]}")
    print(f"  x  = {x.detach().numpy()[0]}")

    # Verify FEM compatibility
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    solver = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0,
        chart_decoder=decoder, decoder_kwargs={},
        device="cpu", dtype=torch.float64,
    )
    print(f"  FEM mesh: {solver.n_nodes} nodes, {solver.n_elements} elements")
    print("  Done.")
