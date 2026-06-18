"""P2G and G2P transfer functions for MPM on mapped coordinate charts.

Implements particle-to-grid (scatter) and grid-to-particle (gather)
transfers using linear B-spline shape functions in chart-local ξ-space.
The metric tensor from the chart Jacobian is used for volume scaling.
"""

from typing import Optional, Tuple

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.grid import ChartGrid


def _linear_bspline(x: torch.Tensor) -> torch.Tensor:
    """Linear (hat) B-spline shape function: N(x) = max(0, 1 - |x|)."""
    return torch.clamp(1.0 - torch.abs(x), min=0.0)


def _shape_functions_and_indices(
    particles: MaterialPointCloud,
    grid: ChartGrid,
    J_inv_T: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute shape function values and grid node indices for each particle.

    For linear B-splines, each particle interacts with at most 2^3 = 8 nodes.

    Parameters
    ----------
    particles : MaterialPointCloud
        Particle data.
    grid : ChartGrid
        Background grid.
    J_inv_T : torch.Tensor, optional
        (N_particles, 3, 3) transpose of the inverse chart Jacobian
        ``J^{-T}`` at each particle.  When supplied, the returned
        ``grad_weights`` are transformed to **physical space** via the
        chain rule

            ∇_x N_I(x_p) = J_p^{-T} ∇_ξ N_I(ξ_p).

        When ``None`` (the default, identity-chart path), ``grad_weights``
        is left in ξ-space.

    Returns
    -------
    node_indices : torch.Tensor
        (N_particles, 8) flat node indices into the grid.
    weights : torch.Tensor
        (N_particles, 8) shape function values.
    grad_weights : torch.Tensor
        (N_particles, 8, 3) shape function gradients — in ξ-space when
        ``J_inv_T`` is None, in physical space when ``J_inv_T`` is
        supplied.
    """
    h = grid.h
    npa = grid.n_nodes_per_axis
    nc = grid.n_cells
    extent = grid.extent

    # Particle positions in grid-index space
    xi_rel = (particles.xi + extent) / h  # (N, 3) in [0, n_cells]
    base_idx = torch.floor(xi_rel).long()  # (N, 3) base grid cell
    frac = xi_rel - base_idx.to(particles.dtype)  # (N, 3) fractional position in [0, 1)

    # Clamp to valid range
    base_idx = torch.clamp(base_idx, 0, nc - 1)

    n_particles = particles.n_particles

    # 8 corner offsets: (8, 3)
    offsets = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
         [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=particles.device, dtype=torch.long,
    )

    # Node indices: (N, 8)
    node_ijk = base_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 8, 3)
    node_ijk = torch.clamp(node_ijk, 0, npa - 1)
    node_indices = node_ijk[:, :, 0] * npa * npa + node_ijk[:, :, 1] * npa + node_ijk[:, :, 2]

    # Shape function values: product of 1D linear B-splines
    # For corner offset (dx, dy, dz): N = N_x(frac_x - dx) * N_y(frac_y - dy) * N_z(frac_z - dz)
    # where N_i(t) = 1 - |t| for |t| <= 1, else 0
    frac_expanded = frac.unsqueeze(1)  # (N, 1, 3)
    offsets_f = offsets.unsqueeze(0).to(particles.dtype)  # (1, 8, 3)
    dist = frac_expanded - offsets_f  # (N, 8, 3)

    w_1d = _linear_bspline(dist)  # (N, 8, 3)
    weights = w_1d[:, :, 0] * w_1d[:, :, 1] * w_1d[:, :, 2]  # (N, 8)

    # Shape function gradients: dN/dξ_i = (dN_i/dξ_i) * prod_{j≠i} N_j
    # dN_i/dξ_i = -sign(dist_i) / h  for |dist_i| < 1
    sign_dist = torch.sign(dist)  # (N, 8, 3)
    dw_1d = -sign_dist / h  # (N, 8, 3) derivative of each 1D shape fn
    # Zero gradient where shape function is zero
    dw_1d[w_1d == 0.0] = 0.0

    grad_weights = torch.zeros(n_particles, 8, 3, device=particles.device, dtype=particles.dtype)
    for d in range(3):
        other_dims = [i for i in range(3) if i != d]
        grad_weights[:, :, d] = dw_1d[:, :, d] * w_1d[:, :, other_dims[0]] * w_1d[:, :, other_dims[1]]

    # Chain-rule transform to physical-space gradients for curved charts.
    # With grad_weights_ξ of shape (N, 8, 3), and J_inv_T of shape
    # (N, 3, 3), we want
    #     grad_weights_phys[p, i, k] = Σ_j J_inv_T[p, k, j] * grad_weights_ξ[p, i, j]
    # (subscripts must be lowercase a-z for torch einsum)
    if J_inv_T is not None:
        grad_weights = torch.einsum("pkj,pij->pik", J_inv_T, grad_weights)

    return node_indices, weights, grad_weights


def particle_to_grid(
    particles: MaterialPointCloud,
    grid: ChartGrid,
    gravity: Optional[torch.Tensor] = None,
    contact_force: Optional[torch.Tensor] = None,
    J_inv_T: Optional[torch.Tensor] = None,
) -> None:
    """Scatter particle mass, momentum, and force to grid nodes (P2G).

    Implements the standard MPM transfer:
        m_I = Σ_p m_p N_I(ξ_p)
        (mv)_I = Σ_p m_p v_p N_I(ξ_p)
        f_I = -Σ_p V_p σ_p ∇_x N_I(x_p) + Σ_p m_p g N_I(ξ_p)
              + Σ_p f_contact_p N_I(ξ_p)

    Parameters
    ----------
    particles : MaterialPointCloud
        Particle data.
    grid : ChartGrid
        Background grid (must be reset before calling).
    gravity : torch.Tensor, optional
        (3,) gravity vector in physical space.
    contact_force : torch.Tensor, optional
        (N_particles, 3) per-particle contact force in physical space.
        Unlike gravity (acceleration scaled by m_p), this is already a
        force and scatters as ``f_I += f_p * N_I(ξ_p)`` without the
        mass factor.
    J_inv_T : torch.Tensor, optional
        (N_particles, 3, 3) transpose inverse chart decoder Jacobian.
        When supplied, the shape-function gradients are chain-ruled
        into physical space so the internal-force scatter
        ``σ_p · ∇_x N_I`` is dimensionally correct on curved charts.
        Leave ``None`` for identity-chart use.
    """
    grid.reset()

    node_indices, weights, grad_weights = _shape_functions_and_indices(
        particles, grid, J_inv_T=J_inv_T,
    )
    # node_indices: (N, 8), weights: (N, 8), grad_weights: (N, 8, 3)

    n_particles = particles.n_particles

    # Scatter mass: m_I += m_p * N_I(ξ_p)
    mass_contrib = particles.mass.unsqueeze(1) * weights  # (N, 8)
    grid.mass.scatter_add_(0, node_indices.reshape(-1), mass_contrib.reshape(-1))

    # Scatter momentum: (mv)_I += m_p * v_p * N_I(ξ_p)
    for d in range(3):
        mom_contrib = particles.mass.unsqueeze(1) * particles.v[:, d:d+1] * weights  # (N, 8)
        grid.momentum[:, d].scatter_add_(0, node_indices.reshape(-1), mom_contrib.reshape(-1))

    # Internal force: f_I -= V_p * σ_p @ ∇N_I(ξ_p)
    Vp = particles.current_volume  # (N,)
    # stress_grad: (N, 8, 3) = V_p * σ_p @ ∇N_I
    stress_grad = torch.einsum("pij, pnj -> pni", particles.stress, grad_weights)  # (N, 8, 3)
    stress_grad = Vp.unsqueeze(1).unsqueeze(2) * stress_grad  # (N, 8, 3)

    for d in range(3):
        grid.force[:, d].scatter_add_(0, node_indices.reshape(-1), -stress_grad[:, :, d].reshape(-1))

    # External force (gravity)
    if gravity is not None:
        for d in range(3):
            grav_contrib = particles.mass.unsqueeze(1) * gravity[d] * weights  # (N, 8)
            grid.force[:, d].scatter_add_(0, node_indices.reshape(-1), grav_contrib.reshape(-1))

    # External force (contact penalty)
    if contact_force is not None:
        for d in range(3):
            cf_contrib = contact_force[:, d:d+1] * weights  # (N, 8)
            grid.force[:, d].scatter_add_(
                0, node_indices.reshape(-1), cf_contrib.reshape(-1)
            )


def grid_to_particle(
    particles: MaterialPointCloud,
    grid: ChartGrid,
    dt: float,
    update_position: bool = True,
    pic_flip_ratio: float = 0.95,
    J_inv_T: Optional[torch.Tensor] = None,
) -> None:
    """Gather grid velocities back to particles (G2P).

    Updates particle velocities (PIC/FLIP blend), positions, and
    deformation gradients.

    When ``J_inv_T`` is supplied, the shape-function gradients are
    transformed to physical space via ``∇_x N_I = J^{-T} ∇_ξ N_I``,
    and the resulting ``vel_grad`` is the true physical spatial
    velocity gradient ``L = ∂v_phys/∂x_phys`` that
    ``update_deformation_gradient`` requires.  When ``J_inv_T`` is
    ``None``, ``vel_grad`` is the ξ-space gradient ``∂v_phys/∂ξ``,
    which equals ``L`` only for identity decoders (the legacy path
    that every existing test exercises).

    Parameters
    ----------
    particles : MaterialPointCloud
        Particle data (modified in-place).
    grid : ChartGrid
        Grid with solved velocities.
    dt : float
        Time step.
    update_position : bool
        If True, update particle positions using grid velocity.
    pic_flip_ratio : float
        Blending ratio: 1.0 = pure FLIP, 0.0 = pure PIC. Default 0.95.
    J_inv_T : torch.Tensor, optional
        (N_particles, 3, 3) transpose inverse chart decoder Jacobian.
        Must be supplied whenever the containing ``ChartMPMSolver`` is
        constructed with a curved decoder; ``None`` for the identity
        path.
    """
    node_indices, weights, grad_weights = _shape_functions_and_indices(
        particles, grid, J_inv_T=J_inv_T,
    )

    # Gather grid velocity to particles (PIC velocity)
    v_pic = torch.zeros_like(particles.v)  # (N, 3)
    for d in range(3):
        v_grid_d = grid.velocity[:, d]  # (n_nodes,)
        v_at_nodes = v_grid_d[node_indices]  # (N, 8)
        v_pic[:, d] = (weights * v_at_nodes).sum(dim=1)

    # FLIP velocity: v_flip = v_old + dv_grid
    # dv_grid = interpolated acceleration * dt
    # For simplicity, compute as: v_flip = v_old + (v_pic_new - v_pic_old)
    # where v_pic_old is the PIC velocity before the grid solve
    # In standard MPM, FLIP = v_old + dt * interpolated_force / interpolated_mass
    # Here we use the simplified version: blend PIC and old velocity + change
    v_new = (1.0 - pic_flip_ratio) * v_pic + pic_flip_ratio * (particles.v + (v_pic - particles.v))
    # Note: when pic_flip_ratio=0.95, this is mostly FLIP with 5% PIC damping
    particles.v = v_pic  # Use PIC for stability in initial implementation

    # Compute velocity gradient at particles: ∇v = Σ_I v_I ⊗ ∇N_I
    vel_grad = torch.zeros(particles.n_particles, 3, 3, device=particles.device, dtype=particles.dtype)
    for d in range(3):
        v_at_nodes = grid.velocity[:, d][node_indices]  # (N, 8)
        # vel_grad[:, d, :] = Σ_I v_I^d * ∇N_I  → (N, 3)
        vel_grad[:, d, :] = (v_at_nodes.unsqueeze(2) * grad_weights).sum(dim=1)

    # Update deformation gradient
    particles.update_deformation_gradient(vel_grad, dt)

    # Update positions
    if update_position:
        particles.xi = particles.xi + dt * v_pic
