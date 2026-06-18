"""Mode-I edge crack benchmark with topology-aware atlas.

Validates the topology-aware atlas against the analytical LEFM solution
for a single edge crack in a finite-width plate under uniform tension.

Geometry:
    Plate: [-W, W] x [-H/2, H/2] x [-T/2, T/2]
    Edge crack: from x1 = -W to x1 = -W + a, along x2 = 0
    Loading: uniform tension sigma_inf on y-faces

Reference:
    K_I = sigma_inf * sqrt(pi * a) * F(a/W)
    F(a/W) from Tada, Paris & Irwin (2000)

V&V-4.1 success criterion:
    K_I relative error < 5% for a/W in [0.1, 0.5]
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from benchmarks.fracture.plate_crack_sdf import (
    CrackedPlateSDFOracle,
    sdf_cracked_plate,
)
from benchmarks.fracture.lefm_reference import (
    stress_intensity_factor,
    williams_displacement,
    extract_K_I_from_displacement,
    geometry_factor_edge_crack,
    mode_i_reference_table,
)


class ModeIEdgeCrackBenchmark:
    """Mode-I edge crack benchmark harness.

    Sets up the cracked-plate geometry, defines boundary conditions,
    and provides methods to extract K_I from numerical solutions.

    Parameters
    ----------
    W : float
        Plate half-width.
    H : float
        Plate full height.
    T : float
        Plate full thickness.
    a : float
        Initial crack length.
    sigma_inf : float
        Far-field tensile stress.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    """

    def __init__(
        self,
        W: float = 1.0,
        H: float = 4.0,
        T: float = 0.5,
        a: float = 0.2,
        sigma_inf: float = 1.0,
        E: float = 200.0,
        nu: float = 0.3,
    ):
        self.W = W
        self.H = H
        self.T = T
        self.a = a
        self.sigma_inf = sigma_inf
        self.E = E
        self.nu = nu

        self.sdf_oracle = CrackedPlateSDFOracle(
            a=a, W=W, H=H, T=T, delta=0.02,
        )

    @property
    def crack_tip(self) -> np.ndarray:
        """Physical coordinates of the crack tip."""
        return np.array([-self.W + self.a, 0.0, 0.0])

    @property
    def K_I_analytical(self) -> float:
        """Analytical stress intensity factor."""
        return stress_intensity_factor(self.sigma_inf, self.a, self.W)

    @property
    def a_over_W(self) -> float:
        return self.a / self.W

    def displacement_bc(self, x: np.ndarray) -> np.ndarray:
        """Dirichlet BC: Williams asymptotic displacement at boundaries.

        For validation, we impose the exact Williams displacement on the
        plate boundaries so the FEM solution should recover K_I exactly
        (up to discretization error).

        Parameters
        ----------
        x : ndarray (N, 3)
            Physical coordinates of boundary nodes.

        Returns
        -------
        u : ndarray (N, 3)
            Prescribed displacement vector.
        """
        # Distance and angle from crack tip
        dx = x[:, 0] - self.crack_tip[0]
        dy = x[:, 1] - self.crack_tip[1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        K_I = self.K_I_analytical
        u_x, u_y = williams_displacement(
            r, theta, K_I, self.E, self.nu, plane_strain=True,
        )

        u = np.zeros_like(x)
        u[:, 0] = u_x
        u[:, 1] = u_y
        # u[:, 2] = 0 (plane strain)
        return u

    def displacement_bc_scalar(self, x: np.ndarray) -> np.ndarray:
        """Scalar version: returns u_y component for Poisson-like testing."""
        u_full = self.displacement_bc(x)
        return u_full[:, 1]

    def extract_K_I(
        self,
        u_numerical: np.ndarray,
        x_eval: np.ndarray,
        r_min: float = 0.05,
        r_max: float = 0.5,
    ) -> float:
        """Extract K_I from numerical displacement field.

        Uses points in an annular region around the crack tip to
        fit K_I via the Williams expansion.

        Parameters
        ----------
        u_numerical : ndarray (N, 3) or (N,)
            Numerical displacement (full vector or y-component only).
        x_eval : ndarray (N, 3)
            Evaluation point coordinates.
        r_min, r_max : float
            Annular region for K_I extraction (avoid singularity and
            boundary effects).

        Returns
        -------
        K_I_numerical : float
        """
        dx = x_eval[:, 0] - self.crack_tip[0]
        dy = x_eval[:, 1] - self.crack_tip[1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        # Select points in the annular extraction region
        mask = (r >= r_min) & (r <= r_max)
        if mask.sum() < 3:
            return float("nan")

        # Get u_y
        if u_numerical.ndim == 2:
            u_y = u_numerical[mask, 1]
        else:
            u_y = u_numerical[mask]

        return extract_K_I_from_displacement(
            u_y, r[mask], theta[mask], self.E, self.nu, plane_strain=True,
        )

    def run_validation(
        self,
        a_values: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run validation across a range of crack lengths.

        For each crack length, computes the analytical K_I and sets up
        the benchmark geometry. This method only generates reference data;
        numerical solution must be provided separately.

        Parameters
        ----------
        a_values : ndarray, optional
            Crack lengths to test. Default: a/W in [0.1, 0.5].

        Returns
        -------
        results : dict
            Reference table with a, a_over_W, K_I_analytical, F.
        """
        if a_values is None:
            a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * self.W

        return mode_i_reference_table(
            W=self.W, H=self.H,
            sigma_inf=self.sigma_inf, E=self.E, nu=self.nu,
            a_values=a_values,
        )

    def topology_check(self, resolution: int = 32) -> Dict:
        """Run topology certification on the cracked plate SDF.

        Returns M_min and Betti numbers. For a cracked plate, the crack
        introduces an H1 loop, so M_min >= 2.

        Parameters
        ----------
        resolution : int
            Grid resolution for persistence computation.

        Returns
        -------
        report : dict with M_min, betti, has_crack_topology.
        """
        from atlas.topo.filtration import clip_to_interior

        grid_vals = self.sdf_oracle.sdf_grid(resolution=resolution)
        grid_clipped = clip_to_interior(grid_vals)

        try:
            from atlas.topo.persistence import (
                compute_persistence_diagrams,
                filter_by_lifetime,
                betti_numbers_at,
            )
            from atlas.topo.ls_category import compute_m_min

            diagrams = compute_persistence_diagrams(grid_clipped, max_dimension=2)
            filtered = filter_by_lifetime(
                diagrams, threshold=0.05, relative=True,
                filtration_range=(float(grid_clipped.min()), 0.0),
            )
            betti = betti_numbers_at(filtered, t=-1e-6)
            m_min = compute_m_min(betti)

            return {
                "M_min": m_min,
                "betti": betti,
                "has_crack_topology": betti.get(1, 0) > 0,
                "has_gudhi": True,
            }
        except ImportError:
            return {
                "M_min": 1,
                "betti": {0: 1},
                "has_crack_topology": False,
                "has_gudhi": False,
            }
