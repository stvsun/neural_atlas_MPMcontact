"""Constitutive models for MPM solvers.

Material models compute Cauchy stress from the deformation gradient F
and optional internal state variables (e.g., plastic strain).
"""

from typing import Dict, Optional, Tuple

import torch


class ConstitutiveModel:
    """Base class for constitutive models."""

    def compute_stress(
        self,
        F: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Cauchy stress from deformation gradient.

        Parameters
        ----------
        F : torch.Tensor
            (N, 3, 3) deformation gradient.
        state : dict, optional
            Internal state variables (modified in-place for plasticity).

        Returns
        -------
        sigma : torch.Tensor
            (N, 3, 3) Cauchy stress.
        state : dict
            Updated internal state variables.
        """
        raise NotImplementedError


class NeoHookeanModel(ConstitutiveModel):
    """Compressible Neo-Hookean hyperelastic material.

    Strain energy density:
        Ψ = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2)(ln J)²

    where I₁ = tr(F^T F), J = det(F).

    Cauchy stress:
        σ = (1/J)[μ(B - I) + λ ln(J) I]

    where B = F F^T (left Cauchy-Green tensor).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    """

    def __init__(self, E: float = 1e5, nu: float = 0.3):
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def compute_stress(
        self,
        F: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = F.shape[0]
        device = F.device
        dtype = F.dtype

        J = torch.det(F)  # (N,)
        J_safe = torch.clamp(J.abs(), min=1e-10)
        log_J = torch.log(J_safe)  # (N,)

        B = torch.bmm(F, F.transpose(1, 2))  # (N, 3, 3)
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)

        # σ = (1/J)[μ(B - I) + λ ln(J) I]
        sigma = (1.0 / J_safe.view(-1, 1, 1)) * (
            self.mu * (B - I)
            + self.lam * log_J.view(-1, 1, 1) * I
        )

        return sigma, state if state is not None else {}


class ElastoplasticModel(ConstitutiveModel):
    """J2 elastoplastic model with isotropic hardening.

    Uses a multiplicative decomposition F = F_e F_p and
    radial return mapping for the plastic corrector.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    yield_stress : float
        Initial yield stress σ_y.
    hardening : float
        Isotropic hardening modulus H.
    """

    def __init__(
        self,
        E: float = 1e5,
        nu: float = 0.3,
        yield_stress: float = 1e3,
        hardening: float = 1e3,
    ):
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.yield_stress = yield_stress
        self.hardening = hardening

    def compute_stress(
        self,
        F: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = F.shape[0]
        device = F.device
        dtype = F.dtype

        if state is None:
            state = {}

        # Get or initialize plastic deformation gradient
        if "Fp" not in state:
            state["Fp"] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1).clone()
        if "equiv_plastic_strain" not in state:
            state["equiv_plastic_strain"] = torch.zeros(n, device=device, dtype=dtype)

        Fp = state["Fp"]
        alpha = state["equiv_plastic_strain"]

        # Trial elastic deformation: F_e_trial = F @ Fp^{-1}
        Fp_inv = torch.linalg.inv(Fp)
        Fe_trial = torch.bmm(F, Fp_inv)

        # SVD of trial elastic deformation
        U, S, Vh = torch.linalg.svd(Fe_trial)

        # Clamp singular values for stability
        S = torch.clamp(S, min=1e-10)
        log_S = torch.log(S)  # (N, 3) Hencky strains

        # Trial Kirchhoff stress (in principal space)
        J = S.prod(dim=1)  # (N,)
        tau_diag = 2.0 * self.mu * log_S + self.lam * log_S.sum(dim=1, keepdim=True)  # (N, 3)

        # Deviatoric part
        tau_mean = tau_diag.mean(dim=1, keepdim=True)  # (N, 1)
        tau_dev = tau_diag - tau_mean  # (N, 3)
        tau_dev_norm = torch.linalg.norm(tau_dev, dim=1)  # (N,)

        # Yield function: f = ||dev(τ)|| - sqrt(2/3) * (σ_y + H * α)
        sqrt_23 = (2.0 / 3.0) ** 0.5
        yield_val = sqrt_23 * (self.yield_stress + self.hardening * alpha)
        f_trial = tau_dev_norm - yield_val

        # Return mapping for yielding particles
        yielding = f_trial > 0.0
        if yielding.any():
            delta_gamma = f_trial[yielding] / (2.0 * self.mu + 2.0 / 3.0 * self.hardening)
            # Update Hencky strains
            direction = tau_dev[yielding] / torch.clamp(tau_dev_norm[yielding].unsqueeze(1), min=1e-12)
            log_S[yielding] = log_S[yielding] - delta_gamma.unsqueeze(1) * direction
            # Update equivalent plastic strain
            alpha[yielding] = alpha[yielding] + sqrt_23 * delta_gamma

        # Reconstruct elastic deformation from corrected singular values
        S_corrected = torch.exp(log_S)
        Fe_new = torch.bmm(U * S_corrected.unsqueeze(1), Vh)

        # Update plastic deformation gradient: Fp_new = Fe_new^{-1} @ F
        Fe_inv = torch.linalg.inv(Fe_new)
        state["Fp"] = torch.bmm(Fe_inv, F)
        state["equiv_plastic_strain"] = alpha

        # Cauchy stress from corrected elastic deformation
        J_new = S_corrected.prod(dim=1)
        tau_corrected = 2.0 * self.mu * log_S + self.lam * log_S.sum(dim=1, keepdim=True)
        # Rotate to spatial frame: σ = (1/J) U diag(τ) U^T
        sigma = (1.0 / J_new.view(-1, 1, 1)) * torch.bmm(
            U * tau_corrected.unsqueeze(1), U.transpose(1, 2)
        )

        return sigma, state
