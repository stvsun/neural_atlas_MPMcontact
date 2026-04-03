"""Geometry utilities for mapped coordinate charts.

Functions for computing local coordinates, Jacobians, and metric tensors
on chart-mapped domains.
"""

from typing import Optional, Tuple

import torch

from common.models import ChartDecoder


def normalize_rows_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / torch.clamp(n, min=eps)


def local_coords(
    x: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    d = x - seed.unsqueeze(0)
    return torch.stack(
        [
            torch.sum(d * t1.unsqueeze(0), dim=1),
            torch.sum(d * t2.unsqueeze(0), dim=1),
            torch.sum(d * n.unsqueeze(0), dim=1),
        ],
        dim=1,
    )


def chart_map_and_jacobian(
    decoder: ChartDecoder,
    xi_in: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: torch.Tensor,
    chart_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xi = xi_in.clone().detach().requires_grad_(True)
    x = decoder(xi, seed=seed, t1=t1, t2=t2, n=n, chart_scale=chart_scale)
    grads = []
    for i in range(3):
        gi = torch.autograd.grad(
            x[:, i],
            xi,
            grad_outputs=torch.ones_like(x[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(gi)
    jac = torch.stack(grads, dim=1)
    return x, xi, jac


def stabilized_jacobian_ops(
    jac: torch.Tensor,
    sigma_floor: float,
    det_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(jac)
    s_safe = torch.clamp(s, min=sigma_floor)
    inv_s = torch.diag_embed(1.0 / s_safe)
    inv_j = torch.bmm(vh.transpose(1, 2), torch.bmm(inv_s, u.transpose(1, 2)))

    raw_det_abs = torch.abs(torch.det(jac))
    det_abs = torch.clamp(raw_det_abs, min=det_floor)
    kappa = s_safe[:, 0] / torch.clamp(s_safe[:, -1], min=sigma_floor)
    valid = raw_det_abs > det_floor
    valid = valid & torch.isfinite(kappa) & torch.isfinite(det_abs)
    return inv_j, det_abs, kappa, valid


def invert_decoder(
    decoder: ChartDecoder,
    x_target: torch.Tensor,
    seed: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    n_vec: torch.Tensor,
    chart_scale: torch.Tensor,
    xi_init: Optional[torch.Tensor] = None,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Find xi* such that decoder(xi*) = x_target using Newton iteration.

    Parameters
    ----------
    decoder : ChartDecoder
        Chart decoder mapping xi -> x.
    x_target : torch.Tensor
        (N, 3) target physical coordinates.
    seed, t1, t2, n_vec : torch.Tensor
        Chart frame vectors (each shape (3,)).
    chart_scale : torch.Tensor
        Chart support radius (scalar).
    xi_init : torch.Tensor, optional
        (N, 3) initial guess. Defaults to linear local_coords projection.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on max |decoder(xi) - x_target|.

    Returns
    -------
    xi_star : torch.Tensor
        (N, 3) inverted coordinates such that decoder(xi_star) ~ x_target.
    """
    if xi_init is None:
        xi_init = local_coords(x_target, seed, t1, t2, n_vec)

    xi = xi_init.clone().detach()

    for it in range(max_iter):
        xi_var = xi.clone().detach().requires_grad_(True)
        x_pred = decoder(xi_var, seed=seed, t1=t1, t2=t2, n=n_vec, chart_scale=chart_scale)

        residual = x_pred - x_target
        res_norm = torch.linalg.norm(residual, dim=1)

        if res_norm.max().item() < tol:
            break

        grads = []
        for i in range(3):
            gi = torch.autograd.grad(
                x_pred[:, i], xi_var,
                grad_outputs=torch.ones_like(x_pred[:, i]),
                create_graph=False, retain_graph=True,
            )[0]
            grads.append(gi)
        J = torch.stack(grads, dim=1)  # (N, 3, 3)

        try:
            delta = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
        except Exception:
            delta = torch.bmm(torch.linalg.pinv(J), residual.unsqueeze(-1)).squeeze(-1)

        xi = xi - delta

    return xi.detach()
