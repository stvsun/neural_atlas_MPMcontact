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

    Solves: (M + c*K) e_nl = M * e_local_nodes

    where:
      M = consistent mass matrix (P1 tetrahedral)
      K = stiffness (Laplacian) matrix: K_ab = sum_e vol_e * dN_a/dx . dN_b/dx
      c = l^2 (gradient parameter, l = length_scale)
      e_local_nodes = volume-weighted projection of element strains to nodes

    Uses proper sparse assembly + scipy.sparse.linalg.spsolve for the
    Helmholtz-type equation, providing genuine spatial smoothing.

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
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    device = solver.device
    dtype_torch = solver.dtype
    n_nodes = solver.n_nodes
    n_elem = solver.n_elements
    c = length_scale ** 2

    elements = solver.elements[:, :4].cpu().numpy()  # (M, 4) vertex indices
    vol = solver.vol.detach().cpu().numpy()           # (M,)
    dNdx = (solver.dNdx_phys if solver.chart_decoder is not None
            else solver.dNdx_ref).detach().cpu().numpy()  # (M, 4, 3)

    # ── Project element strains to nodes (volume-weighted) ──
    e_local_np = e_local.detach().cpu().numpy()       # (M,)
    e_node = np.zeros(n_nodes, dtype=np.float64)
    vol_node = np.zeros(n_nodes, dtype=np.float64)
    for a in range(4):
        np.add.at(e_node, elements[:, a], e_local_np * vol)
        np.add.at(vol_node, elements[:, a], vol)
    vol_node = np.maximum(vol_node, 1e-15)
    e_node /= vol_node

    # ── Assemble sparse M (lumped mass) + K (stiffness) ──
    # For P1 tets: element mass matrix M_e = vol/20 * (ones(4,4) + eye(4))
    # Element stiffness: K_e[a,b] = vol * dN_a . dN_b

    rows = []
    cols = []
    m_vals = []  # mass entries
    k_vals = []  # stiffness entries

    for a in range(4):
        for b in range(4):
            r = elements[:, a]  # (M,)
            cc = elements[:, b]  # (M,)
            rows.append(r)
            cols.append(cc)

            # Consistent mass: M_e[a,b] = vol/20 * (1 + delta_ab)
            if a == b:
                m_vals.append(vol / 10.0)  # vol/20 * 2
            else:
                m_vals.append(vol / 20.0)

            # Stiffness: K_e[a,b] = vol * dot(dN_a, dN_b)
            dot_ab = np.sum(dNdx[:, a, :] * dNdx[:, b, :], axis=1)  # (M,)
            k_vals.append(vol * dot_ab)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    m_data = np.concatenate(m_vals)
    k_data = np.concatenate(k_vals)

    M_sparse = sp.csr_matrix((m_data, (rows, cols)), shape=(n_nodes, n_nodes))
    K_sparse = sp.csr_matrix((k_data, (rows, cols)), shape=(n_nodes, n_nodes))

    # ── Solve (M + c*K) e_nl = M * e_node ──
    A = M_sparse + c * K_sparse
    rhs = M_sparse @ e_node

    try:
        e_nl_np = spla.spsolve(A, rhs)
    except Exception:
        # Fallback to diagonal if sparse solve fails
        diag_A = np.array(A.diagonal()).flatten()
        diag_A = np.maximum(diag_A, 1e-15)
        e_nl_np = rhs / diag_A

    e_nl = torch.tensor(e_nl_np, device=device, dtype=dtype_torch)
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
