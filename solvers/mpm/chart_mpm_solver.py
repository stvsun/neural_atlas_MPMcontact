"""MPM solver on a single mapped coordinate chart.

Implements the Material Point Method (MPM) time stepper on a chart's
local coordinate system. The chart decoder φ: ξ→x provides the mapping
to physical space, and the chart Jacobian supplies the metric tensor
for correct volume/force computations.

The solver follows the standard MPM cycle:
    1. P2G: scatter particle data to grid
    2. Grid solve: apply forces, update grid momentum, apply BCs
    3. G2P: gather grid velocities to particles, update F and positions
    4. Stress update: evaluate constitutive model at new F
"""

from typing import Dict, List, Optional

import torch

from solvers.mpm.particles import MaterialPointCloud
from solvers.mpm.grid import ChartGrid
from solvers.mpm.transfers import particle_to_grid, grid_to_particle
from solvers.mpm.constitutive import ConstitutiveModel, NeoHookeanModel


class ChartMPMSolver:
    """Single-chart MPM time stepper.

    Parameters
    ----------
    n_cells : int
        Grid resolution per axis.
    extent : float
        Half-width of the grid domain in chart-local coords.
    constitutive : ConstitutiveModel
        Material model for stress computation.
    gravity : tuple or torch.Tensor, optional
        Gravity vector in physical space, e.g. (0, 0, -9.81).
    bc_type : str
        Boundary condition type: "fixed", "free", or "slip".
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision.
    decoder : torch.nn.Module, optional
        Chart decoder φ: ξ → x.  When supplied together with the five
        frame parameters below, the solver computes the per-particle
        chart Jacobian J = ∂φ/∂ξ once per step and passes its
        transpose-inverse into the P2G/G2P transfers so that the
        internal-force scatter and velocity gradient are in physical
        space (see ``docs/mpm_velocity_gradient_audit.md``).  Leave as
        ``None`` (default) for the legacy identity-chart path.
    seed, t1, t2, n_vec : torch.Tensor, optional
        (3,) chart frame vectors passed to ``decoder``.
    chart_scale : torch.Tensor, optional
        Scalar chart support radius passed to ``decoder``.
    """

    def __init__(
        self,
        n_cells: int = 16,
        extent: float = 1.0,
        constitutive: Optional[ConstitutiveModel] = None,
        gravity: Optional[tuple] = None,
        bc_type: str = "fixed",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        decoder: Optional[torch.nn.Module] = None,
        seed: Optional[torch.Tensor] = None,
        t1: Optional[torch.Tensor] = None,
        t2: Optional[torch.Tensor] = None,
        n_vec: Optional[torch.Tensor] = None,
        chart_scale: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.bc_type = bc_type

        self.grid = ChartGrid(n_cells=n_cells, extent=extent, device=device, dtype=dtype)
        self.constitutive = constitutive or NeoHookeanModel()

        if gravity is not None:
            self.gravity = torch.tensor(gravity, device=device, dtype=dtype)
        else:
            self.gravity = None

        # Optional chart decoder bundle — used for physical-space
        # shape-function gradient computation on curved charts.
        bundle_fields = (decoder, seed, t1, t2, n_vec, chart_scale)
        n_present = sum(1 for f in bundle_fields if f is not None)
        if 0 < n_present < 6:
            raise ValueError(
                "ChartMPMSolver: decoder bundle must be fully specified "
                "(all of decoder, seed, t1, t2, n_vec, chart_scale) or "
                "fully absent.  Got {} of 6 fields.".format(n_present)
            )
        if n_present == 6:
            self._decoder_bundle = {
                "decoder": decoder,
                "seed": seed,
                "t1": t1,
                "t2": t2,
                "n_vec": n_vec,
                "chart_scale": chart_scale,
            }
        else:
            self._decoder_bundle = None

        self.time = 0.0
        self.step_count = 0

    def _compute_J_inv_T(
        self, particles: MaterialPointCloud,
    ) -> Optional[torch.Tensor]:
        """Compute the transpose-inverse chart Jacobian at every particle.

        Returns ``None`` when no decoder bundle is configured (identity
        path) or when there are no particles to evaluate.
        """
        if self._decoder_bundle is None or particles.n_particles == 0:
            return None

        # Reuse the existing autograd-based chart-Jacobian helper.
        from common.geometry import (
            chart_map_and_jacobian, stabilized_jacobian_ops,
        )
        _, _, J = chart_map_and_jacobian(
            decoder=self._decoder_bundle["decoder"],
            xi_in=particles.xi,
            seed=self._decoder_bundle["seed"],
            t1=self._decoder_bundle["t1"],
            t2=self._decoder_bundle["t2"],
            n=self._decoder_bundle["n_vec"],
            chart_scale=self._decoder_bundle["chart_scale"],
        )
        # We only need J numerically — autograd through the time
        # integration is not desired.
        J_inv, _, _, _ = stabilized_jacobian_ops(
            J.detach(), sigma_floor=1e-8, det_floor=1e-10,
        )
        return J_inv.transpose(-1, -2)

    def step(
        self,
        particles: MaterialPointCloud,
        dt: float,
        contact_force: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Advance one MPM time step.

        Parameters
        ----------
        particles : MaterialPointCloud
            Particle state (modified in-place).
        dt : float
            Time step size.
        contact_force : torch.Tensor, optional
            (N_particles, 3) per-particle contact force in physical
            space.  Scattered to the grid alongside gravity.

        Returns
        -------
        diagnostics : dict
            Step diagnostics: kinetic energy, max velocity, etc.
        """
        # 0. (curved-chart only) Compute J^{-T} at every particle so
        #    shape-function gradients land in physical space.  For the
        #    identity-chart path this is None and the transfers use
        #    the legacy ξ-space gradients.
        J_inv_T = self._compute_J_inv_T(particles)

        # 1. P2G: scatter particle data to grid
        particle_to_grid(
            particles, self.grid,
            gravity=self.gravity,
            contact_force=contact_force,
            J_inv_T=J_inv_T,
        )

        # 2. Grid solve: update momentum with forces, compute velocity
        self.grid.momentum += dt * self.grid.force
        self.grid.compute_velocity()
        self.grid.apply_boundary_conditions(self.bc_type)

        # 3. G2P: gather grid velocities, update positions and F
        grid_to_particle(particles, self.grid, dt, J_inv_T=J_inv_T)

        # 4. Stress update from new deformation gradient
        sigma, particles.state = self.constitutive.compute_stress(particles.F, particles.state)
        particles.stress = sigma

        self.time += dt
        self.step_count += 1

        # Diagnostics
        ke = 0.5 * (particles.mass * (particles.v ** 2).sum(dim=1)).sum().item()
        v_max = particles.v.norm(dim=1).max().item() if particles.n_particles > 0 else 0.0

        return {
            "kinetic_energy": ke,
            "max_velocity": v_max,
            "time": self.time,
            "step": self.step_count,
        }

    def run(
        self,
        particles: MaterialPointCloud,
        dt: float,
        n_steps: int,
        output_interval: int = 10,
    ) -> List[Dict[str, float]]:
        """Run multiple MPM time steps.

        Parameters
        ----------
        particles : MaterialPointCloud
            Particle state (modified in-place).
        dt : float
            Time step size.
        n_steps : int
            Number of steps to run.
        output_interval : int
            Print diagnostics every this many steps.

        Returns
        -------
        history : list of dict
            Diagnostics at each step.
        """
        history = []
        for i in range(n_steps):
            diag = self.step(particles, dt)
            history.append(diag)
            if (i + 1) % output_interval == 0:
                print(
                    f"  [MPM] step {diag['step']:5d} | "
                    f"t={diag['time']:.4e} | "
                    f"KE={diag['kinetic_energy']:.4e} | "
                    f"v_max={diag['max_velocity']:.4e}"
                )
        return history
