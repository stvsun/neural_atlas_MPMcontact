"""Background grid for MPM in chart-local coordinates.

Provides a regular Cartesian grid in ξ-space with node data for
mass, momentum, and force accumulation during P2G/G2P transfers.
"""

from typing import Optional, Tuple

import torch


class ChartGrid:
    """Regular background grid for MPM in a chart's local coordinate system.

    The grid spans [-extent, extent]^3 in chart-local coordinates with
    (n_cells+1)^3 nodes.

    Parameters
    ----------
    n_cells : int
        Number of grid cells per axis.
    extent : float
        Half-width of the grid domain.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision.
    """

    def __init__(
        self,
        n_cells: int = 16,
        extent: float = 1.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.n_cells = n_cells
        self.extent = extent
        self.device = device
        self.dtype = dtype

        self.n_nodes_per_axis = n_cells + 1
        self.n_nodes = self.n_nodes_per_axis ** 3
        self.h = 2.0 * extent / n_cells

        # Node positions in ξ-space
        lin = torch.linspace(-extent, extent, self.n_nodes_per_axis, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(lin, lin, lin)
        self.positions = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)  # (N, 3)

        # Node data — allocated once, reset each step
        self.mass = torch.zeros(self.n_nodes, device=device, dtype=dtype)
        self.momentum = torch.zeros(self.n_nodes, 3, device=device, dtype=dtype)
        self.force = torch.zeros(self.n_nodes, 3, device=device, dtype=dtype)
        self.velocity = torch.zeros(self.n_nodes, 3, device=device, dtype=dtype)

        # Boundary mask: nodes on the cube faces
        tol = self.h * 0.01
        on_face = torch.any(torch.abs(torch.abs(self.positions) - extent) < tol, dim=1)
        self.boundary_mask = on_face  # (N,) bool

    def reset(self) -> None:
        """Zero all node data for a new time step."""
        self.mass.zero_()
        self.momentum.zero_()
        self.force.zero_()
        self.velocity.zero_()

    def compute_velocity(self, eps: float = 1e-12) -> None:
        """Compute grid velocity from momentum: v = p / m."""
        safe_mass = torch.clamp(self.mass, min=eps)
        self.velocity = self.momentum / safe_mass.unsqueeze(1)
        # Zero velocity at nodes with negligible mass
        zero_mask = self.mass < eps
        self.velocity[zero_mask] = 0.0

    def apply_boundary_conditions(self, bc_type: str = "fixed") -> None:
        """Apply boundary conditions on grid faces.

        Parameters
        ----------
        bc_type : str
            "fixed" — zero velocity on all boundary nodes.
            "free" — no constraint (do nothing).
            "slip" — zero normal component only.
        """
        if bc_type == "fixed":
            self.velocity[self.boundary_mask] = 0.0
            self.momentum[self.boundary_mask] = 0.0
        elif bc_type == "slip":
            # Zero normal component on each face
            npa = self.n_nodes_per_axis
            tol = self.h * 0.01
            for axis in range(3):
                lo = self.positions[:, axis] < (-self.extent + tol)
                hi = self.positions[:, axis] > (self.extent - tol)
                face = lo | hi
                self.velocity[face, axis] = 0.0
                self.momentum[face, axis] = 0.0

    def node_index(self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        """Convert 3D grid indices to flat node index."""
        npa = self.n_nodes_per_axis
        return ix * npa * npa + iy * npa + iz
