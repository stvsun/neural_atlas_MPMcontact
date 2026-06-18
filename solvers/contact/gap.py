"""SDF gap function evaluation for contact detection.

Evaluates the signed distance field and its gradient at candidate contact
points.  The gradient gives the outward contact normal via the Eikonal
property |nabla phi| ~ 1 enforced during SDF training.

The autograd pattern mirrors atlas/sdf/train_sdf.py (Eikonal loss).
"""

from typing import Tuple

import torch


def evaluate_gap(
    x_candidates: torch.Tensor,
    sdf_net: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate gap function and contact normal via neural SDF.

    This function forces ``torch.enable_grad()`` internally, so it is
    safe to call from within an outer ``torch.no_grad()`` block — the
    autograd machinery still runs for the SDF gradient.

    Both ``(N,)`` and ``(N, 1)`` SDF network output shapes are
    accepted; the output is always a flat ``(N,)`` gap tensor.

    Parameters
    ----------
    x_candidates : torch.Tensor
        (N, 3) candidate contact points in physical space.  May be any
        floating dtype; the result dtype matches ``x_candidates``.
    sdf_net : torch.nn.Module
        Neural SDF network.  ``sdf_net(x)`` must return ``(N,)`` or
        ``(N, 1)`` scalar signed distances (negative inside the body).

    Returns
    -------
    gap : torch.Tensor
        (N,) signed distance values.  ``gap < 0`` means penetration.
        Always 1-D regardless of the SDF's output shape.
    normal : torch.Tensor
        (N, 3) unit outward normals ``grad(phi) / |grad(phi)|``.
    """
    # Empty input: return empty results without touching autograd.
    if x_candidates.numel() == 0:
        return (
            x_candidates.new_zeros(0),
            x_candidates.new_zeros(0, 3),
        )

    # Force grad-recording even if the caller is in no_grad context.
    with torch.enable_grad():
        x = x_candidates.clone().detach().requires_grad_(True)
        phi = sdf_net(x)
        # Accept (N,) or (N, 1); flatten to (N,)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)
        grad_phi = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=False,
        )[0]                                  # (N, 3)
        # dtype-aware epsilon for the norm clamp
        eps = max(torch.finfo(grad_phi.dtype).eps, 1e-12)
        norm = grad_phi.norm(dim=1, keepdim=True).clamp(min=eps)
        normal = grad_phi / norm
    return phi.detach(), normal.detach()
