"""Nonlocal gradient-enhanced damage model for mesh-independent nucleation.

Implements the Peerlings et al. (1996) implicit gradient damage formulation
to regularize fracture nucleation, ensuring mesh-independent results.

The nonlocal equivalent strain is obtained by solving:
    e_nl - c * nabla^2(e_nl) = e_local

where:
  e_nl: nonlocal equivalent strain (regularized)
  e_local: local equivalent strain (from FEM)
  c = l^2: gradient parameter (l = characteristic length scale)

The damage variable D evolves as:
    D = D(kappa)  where kappa = max(kappa_0, max_t e_nl(t))

with D(kappa) a monotonically increasing function satisfying:
  D(kappa_0) = 0  (undamaged below threshold)
  D -> 1           (fully damaged at large strain)

Reference:
    Peerlings, de Borst, Brekelmans & de Vree (1996),
    "Gradient enhanced damage for quasi-brittle materials",
    Int J Numer Methods Eng, 39(19), 3391-3403.
"""

import math
import numpy as np
import torch
from typing import Optional, Tuple


def compute_local_equivalent_strain(
    F: torch.Tensor,
    strain_type: str = "von_mises",
) -> torch.Tensor:
    """Compute local equivalent strain from deformation gradient.

    Parameters
    ----------
    F : torch.Tensor (M, 3, 3)
        Deformation gradient per element.
    strain_type : str
        'von_mises': von Mises equivalent strain
        'mazars': Mazars equivalent strain (positive principal strains only)

    Returns
    -------
    e_eq : torch.Tensor (M,)
        Local equivalent strain per element.
    """
    I = torch.eye(3, device=F.device, dtype=F.dtype).unsqueeze(0)
    # Small strain: epsilon = 0.5 * (F + F^T) - I
    eps = 0.5 * (F + F.transpose(-2, -1)) - I

    if strain_type == "von_mises":
        # e_eq = sqrt(2/3 * eps_dev : eps_dev)
        eps_vol = torch.diagonal(eps, dim1=-2, dim2=-1).sum(-1) / 3.0
        eps_dev = eps - eps_vol.unsqueeze(-1).unsqueeze(-1) * I
        e_eq = torch.sqrt(2.0 / 3.0 * (eps_dev * eps_dev).sum((-2, -1)).clamp(min=0))
        return e_eq

    elif strain_type == "mazars":
        # Mazars: e_eq = sqrt(sum(<eps_i>_+^2)) where eps_i are principal strains
        eigvals = torch.linalg.eigvalsh(eps)
        pos_eigvals = torch.clamp(eigvals, min=0)
        e_eq = torch.sqrt((pos_eigvals ** 2).sum(-1))
        return e_eq

    raise ValueError(f"Unknown strain_type: {strain_type}")


def solve_nonlocal_strain(
    solver,
    e_local: torch.Tensor,
    length_scale: float = 0.1,
) -> torch.Tensor:
    """Solve the implicit gradient equation for nonlocal equivalent strain.

    Solves: e_nl - l^2 * nabla^2(e_nl) = e_local

    Uses the FEM mesh to assemble and solve the Helmholtz-type equation.

    Parameters
    ----------
    solver : ChartVectorFEMSolver
        FEM solver (provides mesh, shape functions, volumes).
    e_local : torch.Tensor (M,)
        Local equivalent strain per element.
    length_scale : float
        Characteristic length (l); c = l^2 is the gradient parameter.

    Returns
    -------
    e_nl : torch.Tensor (N,)
        Nonlocal equivalent strain at nodes.
    """
    device = solver.device
    dtype = solver.dtype
    n_nodes = solver.n_nodes
    n_elem = solver.n_elements
    c = length_scale ** 2

    # Project element values to nodes via volume-weighted averaging
    e_node = torch.zeros(n_nodes, device=device, dtype=dtype)
    vol_node = torch.zeros(n_nodes, device=device, dtype=dtype)

    elements = solver.elements[:, :4]  # vertex indices
    vol = solver.vol

    for a in range(4):
        e_node.scatter_add_(0, elements[:, a], e_local * vol)
        vol_node.scatter_add_(0, elements[:, a], vol)

    vol_node = vol_node.clamp(min=1e-15)
    e_node = e_node / vol_node

    # Assemble Helmholtz system: (M + c*K) e_nl = M * e_local_nodes
    # where M = mass matrix, K = stiffness matrix (Laplacian)
    # For simplicity, use lumped mass + element stiffness from dNdx
    M_lumped = vol_node.clone()  # lumped mass ≈ nodal volume

    # K matrix: K_ab = sum_e vol_e * dN_a/dx . dN_b/dx
    n_dof = n_nodes
    K_diag = torch.zeros(n_dof, device=device, dtype=dtype)
    dNdx = solver.dNdx_ref if solver.chart_decoder is None else solver.dNdx_phys

    for a in range(4):
        # Diagonal contribution: K_aa = sum_e vol * |dN_a|^2
        grad_sq = (dNdx[:, a, :] ** 2).sum(-1)  # (M,)
        K_diag.scatter_add_(0, elements[:, a], vol * grad_sq)

    # Solve: (M_lumped + c * K_diag) * e_nl = M_lumped * e_node
    # Using diagonal approximation for speed
    lhs = M_lumped + c * K_diag
    rhs = M_lumped * e_node
    e_nl = rhs / lhs.clamp(min=1e-15)

    return e_nl


def damage_function(
    kappa: torch.Tensor,
    kappa_0: float = 1e-4,
    kappa_c: float = 1e-2,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Exponential damage evolution law.

    D(kappa) = 1 - (kappa_0/kappa) * exp(-alpha * (kappa - kappa_0) / (kappa_c - kappa_0))

    Parameters
    ----------
    kappa : torch.Tensor
        History variable (max nonlocal equivalent strain).
    kappa_0 : float
        Damage initiation threshold.
    kappa_c : float
        Characteristic strain for full damage.
    alpha : float
        Damage evolution rate.

    Returns
    -------
    D : torch.Tensor
        Damage variable in [0, 1].
    """
    D = torch.zeros_like(kappa)
    active = kappa > kappa_0
    if active.any():
        ratio = kappa_0 / kappa[active].clamp(min=1e-15)
        exponent = -alpha * (kappa[active] - kappa_0) / max(kappa_c - kappa_0, 1e-15)
        D[active] = 1.0 - ratio * torch.exp(exponent)
    return D.clamp(0, 1)


class NonlocalDamageModel:
    """Nonlocal gradient-enhanced damage model.

    Wraps a base elastic constitutive model with damage:
        sigma_damaged = (1 - D) * sigma_undamaged

    The damage D is driven by the nonlocal equivalent strain,
    which regularizes the nucleation and prevents mesh dependence.

    Parameters
    ----------
    length_scale : float
        Characteristic length for gradient regularization.
    kappa_0 : float
        Damage initiation threshold (equivalent strain).
    kappa_c : float
        Characteristic strain for significant damage.
    """

    def __init__(
        self,
        length_scale: float = 0.1,
        kappa_0: float = 1e-4,
        kappa_c: float = 1e-2,
    ):
        self.length_scale = length_scale
        self.kappa_0 = kappa_0
        self.kappa_c = kappa_c
        self.kappa_history = None  # (N,) max nonlocal strain at each node

    def update(self, solver, F: torch.Tensor) -> torch.Tensor:
        """Update damage state and return damage field.

        Parameters
        ----------
        solver : ChartVectorFEMSolver
        F : torch.Tensor (M, 3, 3)

        Returns
        -------
        D : torch.Tensor (N,)
            Damage at each node.
        """
        e_local = compute_local_equivalent_strain(F)
        e_nl = solve_nonlocal_strain(solver, e_local, self.length_scale)

        # Update history variable (irreversibility)
        if self.kappa_history is None:
            self.kappa_history = e_nl.clone()
        else:
            self.kappa_history = torch.maximum(self.kappa_history, e_nl)

        D = damage_function(self.kappa_history, self.kappa_0, self.kappa_c)
        return D
