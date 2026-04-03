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

        self.time = 0.0
        self.step_count = 0

    def step(self, particles: MaterialPointCloud, dt: float) -> Dict[str, float]:
        """Advance one MPM time step.

        Parameters
        ----------
        particles : MaterialPointCloud
            Particle state (modified in-place).
        dt : float
            Time step size.

        Returns
        -------
        diagnostics : dict
            Step diagnostics: kinetic energy, max velocity, etc.
        """
        # 1. P2G: scatter particle data to grid
        particle_to_grid(particles, self.grid, gravity=self.gravity)

        # 2. Grid solve: update momentum with forces, compute velocity
        self.grid.momentum += dt * self.grid.force
        self.grid.compute_velocity()
        self.grid.apply_boundary_conditions(self.bc_type)

        # 3. G2P: gather grid velocities, update positions and F
        grid_to_particle(particles, self.grid, dt)

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
