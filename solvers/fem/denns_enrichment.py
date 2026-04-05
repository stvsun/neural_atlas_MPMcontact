"""DENNs-style SDF enrichment for chart decoders.

Implements the Discontinuity-Embedded Neural Network (DENNs) approach from
Zhao & Shao (CMAME 446, 2025) adapted for the neural atlas framework.

Key idea: Embed signed distance field (SDF) derived functions directly into
the chart decoder's input space to explicitly capture strong discontinuities
(crack surfaces) and weak discontinuities (material interfaces) within a
single chart, without requiring chart boundaries to align with crack surfaces.

The SDF embedding consists of:
  - d(x): signed distance to crack surface
  - H(d): smoothed Heaviside function for jump enrichment
  - |d|: absolute distance for kink enrichment

These are appended to the reference coordinates xi before passing through
the decoder, allowing the decoder to learn displacement fields with jump
discontinuities across crack surfaces.

Reference:
    Luyang Zhao, Qian Shao. "DENNs: Discontinuity-Embedded Neural Networks
    for fracture mechanics." CMAME 446, 118184 (2025).
"""

import math
from typing import Optional, Callable

import torch


def smooth_heaviside(d: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """Smooth Heaviside function: 0 for d << -eps, 1 for d >> eps.

    H_eps(d) = 0.5 * (1 + tanh(d / epsilon))

    This provides a C^inf approximation to the Heaviside step function,
    capturing the displacement jump across the crack surface.
    """
    return 0.5 * (1.0 + torch.tanh(d / epsilon))


def smooth_abs(d: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """Smooth absolute value: |d|_eps = sqrt(d^2 + eps^2) - eps.

    This captures the kink (gradient discontinuity) at d=0 while being
    differentiable everywhere. Used for weak discontinuity enrichment.
    """
    return torch.sqrt(d * d + epsilon * epsilon) - epsilon


def compute_sdf_embedding(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    x_phys: torch.Tensor,
    epsilon: float = 0.01,
    include_raw: bool = True,
    include_heaviside: bool = True,
    include_abs: bool = True,
) -> torch.Tensor:
    """Compute SDF-derived embedding features at physical points.

    Parameters
    ----------
    sdf_fn : callable
        Maps physical coordinates x (N, 3) -> signed distance d (N,).
    x_phys : torch.Tensor (N, 3)
        Physical-space query points.
    epsilon : float
        Smoothing parameter for Heaviside and absolute value.
    include_raw : bool
        Include raw SDF value d(x).
    include_heaviside : bool
        Include smoothed Heaviside H(d).
    include_abs : bool
        Include smoothed |d|.

    Returns
    -------
    embedding : torch.Tensor (N, K)
        K features where K = sum of included channels (up to 3).
    """
    with torch.no_grad():
        d = sdf_fn(x_phys)  # (N,) or (N, 1)
        if d.dim() > 1:
            d = d.squeeze(-1)

    channels = []
    if include_raw:
        channels.append(d.unsqueeze(-1))
    if include_heaviside:
        channels.append(smooth_heaviside(d, epsilon).unsqueeze(-1))
    if include_abs:
        channels.append(smooth_abs(d, epsilon).unsqueeze(-1))

    if not channels:
        return torch.zeros(x_phys.shape[0], 0, device=x_phys.device, dtype=x_phys.dtype)

    return torch.cat(channels, dim=-1)


class SDFEnrichedDecoder(torch.nn.Module):
    """Wraps a base decoder with SDF-derived discontinuity embedding.

    The enriched decoder's forward pass:
      1. Compute physical coordinates: x = base_decoder(xi)
      2. Evaluate SDF embedding: e = [d(x), H(d(x)), |d(x)|]
      3. Compute enrichment: u_enrich = enrichment_net(cat(xi, e))
      4. Return: x + u_enrich (or just x if in geometry-only mode)

    For fracture problems, this allows the decoder to represent crack-opening
    displacement within a single chart, without requiring chart boundaries
    to coincide with the crack surface.

    Parameters
    ----------
    base_decoder : torch.nn.Module
        Underlying coordinate chart decoder (BoxDecoder, CrackTipDecoder, etc.).
    sdf_fn : callable
        Crack SDF: maps x (N, 3) -> d (N,) signed distance to crack surface.
    epsilon : float
        Smoothing parameter for Heaviside/abs.
    enrichment_width : int
        Hidden layer width of enrichment MLP.
    enrichment_depth : int
        Number of hidden layers in enrichment MLP.
    """

    def __init__(
        self,
        base_decoder: torch.nn.Module,
        sdf_fn: Callable,
        epsilon: float = 0.01,
        enrichment_width: int = 32,
        enrichment_depth: int = 2,
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.sdf_fn = sdf_fn
        self.epsilon = epsilon

        # Count SDF embedding channels: d, H(d), |d| = 3
        n_sdf_channels = 3

        # Enrichment MLP: maps (xi, sdf_embedding) -> displacement correction
        from common.models import MLP
        self.enrichment_net = MLP(
            in_dim=3 + n_sdf_channels,  # xi (3) + sdf embedding (3)
            out_dim=3,                   # displacement correction (3)
            width=enrichment_width,
            depth=enrichment_depth,
        )

        # Zero-initialize output layer (no initial perturbation)
        torch.nn.init.zeros_(self.enrichment_net.out.weight)
        torch.nn.init.zeros_(self.enrichment_net.out.bias)

        # Learnable enrichment amplitude (starts at 0)
        self.raw_amplitude = torch.nn.Parameter(torch.tensor(-5.0, dtype=torch.float64))

    @property
    def amplitude(self) -> torch.Tensor:
        """Current enrichment amplitude (sigmoid-bounded to [0, 0.5])."""
        return 0.5 * torch.sigmoid(self.raw_amplitude)

    def forward(self, xi: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass: base decoder + SDF enrichment.

        Parameters
        ----------
        xi : torch.Tensor (N, 3)
            Reference coordinates.
        **kwargs : dict
            Passed to base decoder.

        Returns
        -------
        x : torch.Tensor (N, 3)
            Physical coordinates with enrichment.
        """
        # 1. Base decoder: xi -> x_base
        x_base = self.base_decoder(xi, **kwargs)

        # 2. SDF embedding at physical locations
        sdf_embed = compute_sdf_embedding(
            self.sdf_fn, x_base, self.epsilon,
            include_raw=True, include_heaviside=True, include_abs=True,
        )

        # 3. Enrichment: MLP(cat(xi, sdf_embed)) * amplitude
        enrichment_input = torch.cat([xi, sdf_embed.to(xi.dtype)], dim=-1)
        correction = self.enrichment_net(enrichment_input)
        correction = self.amplitude * correction

        return x_base + correction

    def inverse(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximate inverse via base decoder's inverse.

        The enrichment is small (amplitude starts at 0), so the base
        decoder's inverse is a good approximation.
        """
        if hasattr(self.base_decoder, 'inverse'):
            return self.base_decoder.inverse(x, **kwargs)
        raise NotImplementedError("Base decoder has no inverse()")

    def jacobian(self, xi: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute Jacobian dx/dxi via autograd (accounts for enrichment)."""
        xi_var = xi.detach().clone().requires_grad_(True)
        x = self.forward(xi_var, **kwargs)
        rows = []
        for d in range(3):
            grad_out = torch.zeros_like(x)
            grad_out[:, d] = 1.0
            g = torch.autograd.grad(
                x, xi_var, grad_outputs=grad_out,
                retain_graph=(d < 2), create_graph=False,
            )[0]
            rows.append(g)
        return torch.stack(rows, dim=1)

    def get_sdf_embedding(self, xi: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the SDF embedding for diagnostics/visualization."""
        with torch.no_grad():
            x = self.base_decoder(xi, **kwargs)
            return compute_sdf_embedding(self.sdf_fn, x, self.epsilon)


def enrich_decoder(
    decoder: torch.nn.Module,
    sdf_fn: Callable,
    epsilon: float = 0.01,
    enrichment_width: int = 32,
    enrichment_depth: int = 2,
) -> SDFEnrichedDecoder:
    """Convenience function to wrap any decoder with SDF enrichment.

    Parameters
    ----------
    decoder : torch.nn.Module
        Base decoder to enrich.
    sdf_fn : callable
        Crack SDF function.
    epsilon : float
        Smoothing parameter.
    enrichment_width, enrichment_depth : int
        MLP architecture for enrichment network.

    Returns
    -------
    SDFEnrichedDecoder
        Enriched decoder (drop-in replacement for the original).
    """
    enriched = SDFEnrichedDecoder(
        base_decoder=decoder,
        sdf_fn=sdf_fn,
        epsilon=epsilon,
        enrichment_width=enrichment_width,
        enrichment_depth=enrichment_depth,
    )
    # Match dtype of base decoder
    if hasattr(decoder, 'center'):
        enriched = enriched.to(decoder.center.dtype)
    else:
        enriched = enriched.double()
    return enriched
