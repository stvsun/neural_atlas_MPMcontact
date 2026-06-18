"""Manufactured solution for the Poisson benchmark on Stanford rabbit.

u(x) = sin(πx₁) sin(πx₂) sin(πx₃)
f(x) = -Δu = 3π² sin(πx₁) sin(πx₂) sin(πx₃)
"""

import math

import torch


def manufactured_u(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(math.pi * x[:, 0:1])
        * torch.sin(math.pi * x[:, 1:2])
        * torch.sin(math.pi * x[:, 2:3])
    )


def manufactured_grad_u(x: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    du1 = pi * torch.cos(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    du2 = pi * torch.sin(pi * x1) * torch.cos(pi * x2) * torch.sin(pi * x3)
    du3 = pi * torch.sin(pi * x1) * torch.sin(pi * x2) * torch.cos(pi * x3)
    return torch.cat([du1, du2, du3], dim=1)


def forcing_f(x: torch.Tensor) -> torch.Tensor:
    return 3.0 * (math.pi ** 2) * manufactured_u(x)
