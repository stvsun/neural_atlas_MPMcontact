"""Material point cloud for MPM on mapped coordinate charts.

Stores particle state in chart-local coordinates (ξ-space) and provides
methods for pushing to physical space via chart decoders.
"""

from typing import Optional

import torch


class MaterialPointCloud:
    """Collection of material points living in a chart's local coordinate system.

    Each particle carries:
        - position ξ in chart-local coords (N, 3)
        - velocity v in physical space (N, 3)
        - deformation gradient F (N, 3, 3)
        - Cauchy stress σ (N, 3, 3)
        - volume V₀ in physical space (N,)
        - mass m (N,)
        - internal state variables for constitutive models (dict of tensors)

    Parameters
    ----------
    positions : torch.Tensor
        (N, 3) initial positions in chart-local coordinates.
    velocities : torch.Tensor
        (N, 3) initial velocities in physical space.
    volumes : torch.Tensor
        (N,) initial volumes in physical space.
    masses : torch.Tensor
        (N,) particle masses.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        volumes: torch.Tensor,
        masses: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.device = device
        self.dtype = dtype
        n = positions.shape[0]

        self.xi = positions.to(device=device, dtype=dtype)          # (N, 3)
        self.v = velocities.to(device=device, dtype=dtype)          # (N, 3)
        self.V0 = volumes.to(device=device, dtype=dtype)            # (N,)
        self.mass = masses.to(device=device, dtype=dtype)           # (N,)

        # Deformation gradient — initialized to identity
        self.F = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1).clone()

        # Cauchy stress — initialized to zero
        self.stress = torch.zeros(n, 3, 3, device=device, dtype=dtype)

        # Internal state variables (e.g., plastic strain for elastoplasticity)
        self.state: dict = {}

    @property
    def n_particles(self) -> int:
        return self.xi.shape[0]

    @property
    def current_volume(self) -> torch.Tensor:
        """Current volume = V₀ * det(F)."""
        return self.V0 * torch.det(self.F).abs()

    def push_to_physical(self, decoder: torch.nn.Module, seed, t1, t2, n_vec, chart_scale) -> torch.Tensor:
        """Map particle positions from chart-local to physical space via decoder.

        Returns
        -------
        x_phys : torch.Tensor
            (N, 3) physical-space positions.
        """
        with torch.no_grad():
            return decoder(self.xi, seed=seed, t1=t1, t2=t2, n=n_vec, chart_scale=chart_scale)

    def update_deformation_gradient(self, vel_grad: torch.Tensor, dt: float) -> None:
        """Update F via F_{n+1} = (I + dt * ∇v) @ F_n.

        Parameters
        ----------
        vel_grad : torch.Tensor
            (N, 3, 3) velocity gradient at particle positions.
        dt : float
            Time step.
        """
        eye = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.F = torch.bmm(eye + dt * vel_grad, self.F)

    @classmethod
    def create_uniform(
        cls,
        n_per_axis: int,
        extent: float,
        density: float,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> "MaterialPointCloud":
        """Create a uniform grid of particles in [-extent, extent]^3.

        Parameters
        ----------
        n_per_axis : int
            Number of particles per axis.
        extent : float
            Half-width of the particle domain.
        density : float
            Material density (mass = density * volume).
        """
        lin = torch.linspace(-extent, extent, n_per_axis, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(lin, lin, lin)
        positions = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

        n = positions.shape[0]
        cell_vol = (2.0 * extent / n_per_axis) ** 3
        volumes = torch.full((n,), cell_vol, device=device, dtype=dtype)
        masses = density * volumes
        velocities = torch.zeros(n, 3, device=device, dtype=dtype)

        return cls(positions, velocities, volumes, masses, device=device, dtype=dtype)
