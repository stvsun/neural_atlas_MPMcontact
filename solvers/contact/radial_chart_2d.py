"""2-D neural radial boundary chart (the transition-map detector for CV-5).

The 2-D companion to ``chart_gap.py`` (which is 3-D, S^2 -> R+).  A star-shaped 2-D body is
carried level-set-free by a radial function on the circle of directions

    rho : S^1 -> R+,        rho(psi) = NeuralRho2D(psi),

an MLP of ``(cos psi, sin psi)`` (periodic by construction).  With center ``c`` and orientation
``alpha`` the gap and matched/conservative normal are read EXACTLY as the analytical inverse
radial chart ``solvers/contact/supershape.py::radial_gap`` — only ``rho`` and its derivative
``rho'(psi)`` are replaced by the network and its autograd gradient:

    d = R(-alpha)(p - c),  r = |d|,  psi = atan2(d_y, d_x),
    gap = r - rho(psi),
    grad_d F = d/r - rho'(psi) * grad psi,   grad psi = (-d_y, d_x)/r^2,
    normal  = normalize( R(alpha) grad_d F ).

WHY THIS IS THE ACCURATE CV-5 PATH.  Fitting a 1-D function rho(psi) is far easier than fitting a
2-D ambient SDF of the cusped, concave superformula: there is no medial axis, the boundary
parametrization carries no ambient spectral bias, and the normal is the conservative matched
normal at every point.  So where a neural SDF degrades on cusps/concavities (CV-5's documented
SDF weakness, manual §11.4), the neural radial chart reproduces the analytical radial gap/normal
to ~1e-3.  Compare it to the RADIAL reference (``supershape.radial_gap``), NOT the Euclidean one
(the radial gap is biased 1/cos(alpha); manual §7/§11.2).

Scope: star-shaped bodies only (every ray from c hits the boundary once) — so the superformula
(CV-5), not the Koch snowflake (CV-6, not star-shaped for level>=2).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from common.models import MLP


def _dtype_eps(dtype: torch.dtype) -> float:
    return max(torch.finfo(dtype).eps, 1e-12)


class NeuralRho2D(torch.nn.Module):
    """Trained radial function rho : S^1 -> R+ as an MLP of FOURIER FEATURES of psi.

    The input is the harmonic encoding [cos(k psi), sin(k psi)]_{k=1..n_freq} (periodic by
    construction).  Fourier features hand the network the high frequencies directly, defeating the
    1-D spectral bias that a plain (cos psi, sin psi) input suffers — essential for the sharp,
    cusped superformula radius (m lobes, n1<1).  A softplus output keeps rho > 0; a learnable
    ``base`` (init to the mean target radius) lets the net learn only the shape deviation.
    """

    def __init__(self, width: int = 96, depth: int = 3, base: float = 1.0, n_freq: int = 16):
        super().__init__()
        self.n_freq = int(n_freq)
        self.net = MLP(in_dim=2 * self.n_freq, out_dim=1, width=width, depth=depth)
        self.raw_base = torch.nn.Parameter(torch.tensor(float(base), dtype=torch.float64))

    def _features(self, psi: torch.Tensor) -> torch.Tensor:
        k = torch.arange(1, self.n_freq + 1, dtype=psi.dtype, device=psi.device)
        ang = psi.unsqueeze(-1) * k                          # (N, K)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)   # (N, 2K)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        delta = self.net(self._features(psi)).squeeze(-1)
        return torch.nn.functional.softplus(self.raw_base + delta)


class SuperformulaRho2D(torch.nn.Module):
    """Analytic Gielis superformula radius rho(psi) in torch (reference / dataset target).

    rho(psi) = ( |cos(m psi/4)/a|^n2 + |sin(m psi/4)/b|^n3 )^(-1/n1) * scale .
    Mirrors solvers/contact/supershape.py::radius (numpy) for like-for-like comparison."""

    def __init__(self, m=6, n1=0.7, n2=0.7, n3=0.7, a=1.0, b=1.0, scale=1.0):
        super().__init__()
        self.m, self.n1, self.n2, self.n3 = m, n1, n2, n3
        self.a, self.b, self.scale = a, b, scale

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        t = self.m * psi / 4.0
        term = (torch.abs(torch.cos(t) / self.a) ** self.n2
                + torch.abs(torch.sin(t) / self.b) ** self.n3)
        return self.scale * term.clamp_min(1e-12) ** (-1.0 / self.n1)


def _to_body(p_pts: torch.Tensor, c: torch.Tensor, alpha: float):
    d = p_pts - c
    ca, sa = math.cos(alpha), math.sin(alpha)
    dx = d[:, 0] * ca + d[:, 1] * sa            # R(-alpha)(p-c)
    dy = -d[:, 0] * sa + d[:, 1] * ca
    return dx, dy


def evaluate_radial_gap_2d(
    p_pts: torch.Tensor,
    rho_module: torch.nn.Module,
    center=(0.0, 0.0),
    alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gap + matched (conservative) normal of a 2-D radial chart.  Drop-in 2-D twin of
    ``supershape.radial_gap``: returns (gap (N,), normal (N,2)); gap<0 => inside.  Safe inside an
    outer ``no_grad`` block; returns detached tensors."""
    if p_pts.numel() == 0:
        return p_pts.new_zeros(0), p_pts.new_zeros(0, 2)
    eps = _dtype_eps(p_pts.dtype)
    c = torch.as_tensor(center, dtype=p_pts.dtype, device=p_pts.device)
    dx, dy = _to_body(p_pts.detach(), c, alpha)
    r = torch.sqrt(dx * dx + dy * dy).clamp_min(eps)
    psi = torch.atan2(dy, dx)

    with torch.enable_grad():
        psi_g = psi.detach().requires_grad_(True)
        rho = rho_module(psi_g)
        if rho.dim() > 1:
            rho = rho.squeeze(-1)
        drho = torch.autograd.grad(rho, psi_g, grad_outputs=torch.ones_like(rho),
                                   create_graph=False, retain_graph=False)[0]
    rho = rho.detach()
    gap = r - rho
    # body-frame matched gradient: d/r - rho'(psi) * grad psi,  grad psi = (-dy, dx)/r^2
    gd_x = dx / r - drho * (-dy) / (r ** 2)
    gd_y = dy / r - drho * (dx) / (r ** 2)
    ca, sa = math.cos(alpha), math.sin(alpha)
    gw_x = gd_x * ca - gd_y * sa                 # rotate body -> world (+alpha)
    gw_y = gd_x * sa + gd_y * ca
    grad = torch.stack([gw_x, gw_y], dim=-1)
    normal = grad / grad.norm(dim=1, keepdim=True).clamp_min(eps)
    return gap.detach(), normal.detach()


if __name__ == "__main__":
    # self-test: the ANALYTIC torch radius matches the numpy supershape.radius, and the torch
    # radial gap matches supershape.radial_gap to machine precision (validates the 2-D evaluator
    # before any training).
    import numpy as np
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from solvers.contact import supershape as ss
    p = ss.SuperParams(m=6, n1=0.7, n2=0.7, n3=0.7, a=1.0, b=1.0, scale=1.0)
    rho_t = SuperformulaRho2D(m=6, n1=0.7, n2=0.7, n3=0.7, a=1.0, b=1.0, scale=1.0).double()
    th = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    r_np = ss.radius(th, p)
    r_t = rho_t(torch.tensor(th)).numpy()
    print(f"  analytic radius torch-vs-numpy max err = {np.max(np.abs(r_np - r_t)):.2e}")
    assert np.max(np.abs(r_np - r_t)) < 1e-12
    rng = np.random.RandomState(0)
    xy = ss.boundary(rng.uniform(0, 2 * np.pi, 400), np.zeros(2), 0.0, p) + rng.randn(400, 2) * 0.1
    g_np, grad_np = ss.radial_gap(xy, np.zeros(2), 0.0, p)
    g_t, n_t = evaluate_radial_gap_2d(torch.tensor(xy), rho_t, center=(0.0, 0.0), alpha=0.0)
    n_np = grad_np / np.clip(np.linalg.norm(grad_np, axis=1, keepdims=True), 1e-12, None)
    print(f"  radial gap torch-vs-numpy max err = {np.max(np.abs(g_np - g_t.numpy())):.2e}")
    print(f"  matched normal max angle err = "
          f"{np.degrees(np.max(np.arccos(np.clip(np.sum(n_np * n_t.numpy(), 1), -1, 1)))):.2e} deg")
    assert np.max(np.abs(g_np - g_t.numpy())) < 1e-10
    print("  radial_chart_2d self-test PASSED")
