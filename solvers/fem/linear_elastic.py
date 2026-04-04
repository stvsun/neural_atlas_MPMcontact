"""Linear elastic material for ChartVectorFEMSolver.

Provides stress_fn and tangent_fn callables that implement St. Venant-
Kirchhoff (linear elastic) constitutive response for use with
ChartVectorFEMSolver.solve_nonlinear().

For small strains, this is equivalent to Hooke's law:
    sigma = lambda * tr(eps) * I + 2 * mu * eps
where eps = (F^T F - I) / 2 is the Green-Lagrange strain.

The first Piola-Kirchhoff stress is:
    P = F * S,  S = lambda * tr(E) * I + 2 * mu * E
"""

import torch


def make_linear_elastic(E: float, nu: float, device="cpu", dtype=torch.float64):
    """Create stress_fn and tangent_fn for linear elasticity.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    stress_fn : callable
        F (M,3,3) -> P (M,3,3) first Piola-Kirchhoff stress.
    tangent_fn : callable
        F (M,3,3) -> C (M,9,9) material tangent dP/dF.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    I3 = torch.eye(3, device=device, dtype=dtype)

    def stress_fn(F):
        """P = F @ S, where S = lam * tr(E) * I + 2 * mu * E."""
        # Green-Lagrange strain: E = (F^T F - I) / 2
        C_tensor = torch.bmm(F.transpose(1, 2), F)  # (M, 3, 3)
        E_gl = 0.5 * (C_tensor - I3.unsqueeze(0))    # (M, 3, 3)

        tr_E = E_gl[:, 0, 0] + E_gl[:, 1, 1] + E_gl[:, 2, 2]  # (M,)

        # Second Piola-Kirchhoff: S = lam * tr(E) * I + 2 * mu * E
        S = lam * tr_E.view(-1, 1, 1) * I3.unsqueeze(0) + 2.0 * mu * E_gl

        # First Piola-Kirchhoff: P = F @ S
        P = torch.bmm(F, S)
        return P

    def tangent_fn(F):
        """Material tangent dP_{iJ}/dF_{kL} in (M, 9, 9) row-major format.

        For St. Venant-Kirchhoff:
        dP_{iJ}/dF_{kL} = S_{JL} * delta_{ik} + F_{iA} * C_{AJBL} * F_{kB}
        where C_{AJBL} = lam * delta_{AJ} * delta_{BL} + mu * (delta_{AB} * delta_{JL} + delta_{AL} * delta_{JB})

        For small strains (F ~ I), this simplifies considerably.
        """
        M = F.shape[0]

        # Green-Lagrange strain
        C_tensor = torch.bmm(F.transpose(1, 2), F)
        E_gl = 0.5 * (C_tensor - I3.unsqueeze(0))
        tr_E = E_gl[:, 0, 0] + E_gl[:, 1, 1] + E_gl[:, 2, 2]
        S = lam * tr_E.view(-1, 1, 1) * I3.unsqueeze(0) + 2.0 * mu * E_gl

        # Build 9x9 tangent per element
        # dP_{iJ}/dF_{kL} indexed as row = 3*i+J, col = 3*k+L
        dPdF = torch.zeros(M, 9, 9, device=F.device, dtype=F.dtype)

        for i in range(3):
            for J in range(3):
                row = 3 * i + J
                for k in range(3):
                    for L in range(3):
                        col = 3 * k + L
                        # Term 1: S_{JL} * delta_{ik}
                        val = S[:, J, L] * (1.0 if i == k else 0.0)

                        # Term 2: F_{iA} * C_{AJBL} * F_{kB}
                        # C_{AJBL} = lam*d_{AJ}*d_{BL} + mu*(d_{AB}*d_{JL} + d_{AL}*d_{JB})
                        for A in range(3):
                            for B in range(3):
                                c_AJBL = 0.0
                                if A == J and B == L:
                                    c_AJBL += lam
                                if A == B and J == L:
                                    c_AJBL += mu
                                if A == L and J == B:
                                    c_AJBL += mu
                                if c_AJBL != 0.0:
                                    val = val + c_AJBL * F[:, i, A] * F[:, k, B]

                        dPdF[:, row, col] = val

        return dPdF

    return stress_fn, tangent_fn
