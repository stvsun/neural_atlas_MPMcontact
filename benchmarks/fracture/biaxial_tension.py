"""Biaxial tension test on a circular plate — Challenge Problem 2.

Adapted from Kamarei, Zeng, Dolbow & Lopez-Pamies (2026),
"Nine circles of elastic brittle fracture", CMAME 448, 118449.

Geometry:
    Circular plate: radius R, thickness L
    Equi-biaxial tension via affine displacement delta at lateral boundary

Exact solution (linear elastic, soda-lime glass):
    S = E * delta / ((1 - nu) * R)  for 0 <= delta < delta_bs
    S = 0                            for delta >= delta_bs
    where delta_bs = sigma_bs * (1 - nu) * R / E

Fracture criterion:
    Crack nucleates when biaxial stress S reaches sigma_bs.
    A through-thickness crack severs the plate (H0 topology change).

Material constants (soda-lime glass, Table 2 of Kamarei et al.):
    E = 70 GPa, nu = 0.22
    sigma_ts = 40 MPa (uniaxial tensile strength)
    sigma_bs = 27 MPa (biaxial tensile strength)
    G_c = 10 N/m (critical energy release rate)

Data repositories:
    https://databank.illinois.edu/datasets/IDB-6684845
    https://research.repository.duke.edu/record/401
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Material constants (soda-lime glass)
# ---------------------------------------------------------------------------
GLASS = {
    "E": 70e3,           # Young's modulus (MPa)
    "nu": 0.22,          # Poisson's ratio
    "mu": 70e3 / (2 * (1 + 0.22)),      # shear modulus
    "lam": 70e3 * 0.22 / ((1 + 0.22) * (1 - 2 * 0.22)),  # first Lame constant
    "sigma_ts": 40.0,    # uniaxial tensile strength (MPa)
    "sigma_hs": 27.8,    # hydrostatic strength (MPa)
    "sigma_bs": 27.0,    # biaxial tensile strength (MPa)
    "sigma_ss": 44.4,    # shear strength (MPa)
    "G_c": 0.01,         # critical energy release rate (N/mm = MPa*mm)
}

PU_ELASTOMER = {
    "mu": 0.52,          # shear modulus (MPa)
    "lam": 85.77,        # first Lame constant (MPa)
    "sigma_ts": 0.3,     # uniaxial tensile strength (MPa)
    "sigma_hs": 1.0,     # hydrostatic strength (MPa)
    "sigma_bs": 0.27,    # biaxial tensile strength (MPa)
    "sigma_ss": 0.19,    # shear strength (MPa)
    "G_c": 0.041,        # critical energy release rate (N/mm)
}


# ---------------------------------------------------------------------------
# SDF for circular plate
# ---------------------------------------------------------------------------
def sdf_circular_plate(
    x: np.ndarray,
    R: float = 5.0,
    L: float = 0.25,
) -> np.ndarray:
    """SDF of a circular plate: radius R in x1-x2, thickness L in x3.

    The plate occupies {x : x1^2 + x2^2 <= R^2, |x3| <= L/2}.

    Parameters
    ----------
    x : ndarray (N, 3)
    R : float
        Plate radius.
    L : float
        Plate thickness.

    Returns
    -------
    sdf : ndarray (N,)
        Negative inside, positive outside.
    """
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    d_radial = r - R
    d_axial = np.abs(x[:, 2]) - L / 2.0

    # Intersection of cylinder and slab
    outside = np.sqrt(
        np.maximum(d_radial, 0) ** 2 + np.maximum(d_axial, 0) ** 2
    )
    inside = np.minimum(np.maximum(d_radial, d_axial), 0)
    return outside + inside


def sdf_cracked_circular_plate(
    x: np.ndarray,
    R: float = 5.0,
    L: float = 0.25,
    crack_angle: float = 0.0,
    delta: float = 0.01,
) -> np.ndarray:
    """SDF of a circular plate with a through-thickness crack.

    The crack is a diametral line at angle crack_angle from x1-axis,
    modeled as a thin slit of half-opening delta extending the full
    diameter (2R).

    Parameters
    ----------
    x : ndarray (N, 3)
    R : float
        Plate radius.
    L : float
        Plate thickness.
    crack_angle : float
        Crack orientation in radians (0 = along x1-axis).
    delta : float
        Crack slit half-opening.

    Returns
    -------
    sdf : ndarray (N,)
        Negative inside cracked plate, positive outside.
    """
    d_plate = sdf_circular_plate(x, R=R, L=L)

    # Crack slit: rotate coordinates to align with crack
    cos_a = math.cos(crack_angle)
    sin_a = math.sin(crack_angle)
    x_rot = x[:, 0] * cos_a + x[:, 1] * sin_a   # along crack
    y_rot = -x[:, 0] * sin_a + x[:, 1] * cos_a   # normal to crack

    # Crack slit: |y_rot| <= delta, |x_rot| <= R
    d_slit_x = np.abs(x_rot) - R
    d_slit_y = np.abs(y_rot) - delta

    outside_slit = np.sqrt(
        np.maximum(d_slit_x, 0) ** 2 + np.maximum(d_slit_y, 0) ** 2
    )
    inside_slit = np.minimum(np.maximum(d_slit_x, d_slit_y), 0)
    d_crack = outside_slit + inside_slit

    # CSG subtraction: plate \ crack
    return np.maximum(d_plate, -d_crack)


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------
def stress_strain_response(
    delta_values: np.ndarray,
    R: float = 5.0,
    material: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact stress-strain response for the biaxial tension test.

    Parameters
    ----------
    delta_values : ndarray (N,)
        Applied displacement values.
    R : float
        Plate radius.
    material : dict
        Material constants (must have E, nu, sigma_bs).

    Returns
    -------
    strain : ndarray (N,)
        Biaxial strain delta/R.
    stress : ndarray (N,)
        Biaxial stress S (MPa).
    """
    if material is None:
        material = GLASS

    E = material["E"]
    nu = material["nu"]
    sigma_bs = material["sigma_bs"]

    strain = delta_values / R
    delta_bs = sigma_bs * (1 - nu) * R / E
    stress = np.where(
        delta_values < delta_bs,
        E * delta_values / ((1 - nu) * R),
        0.0,
    )
    return strain, stress


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
class BiaxialTensionBenchmark:
    """Biaxial tension benchmark harness.

    Parameters
    ----------
    R : float
        Plate radius (mm).
    L : float
        Plate thickness (mm).
    material : dict
        Material constants.
    """

    def __init__(
        self,
        R: float = 5.0,
        L: float = 0.25,
        material: dict = None,
    ):
        self.R = R
        self.L = L
        self.material = material or GLASS

    @property
    def sigma_bs(self) -> float:
        return self.material["sigma_bs"]

    @property
    def delta_bs(self) -> float:
        """Critical displacement at fracture."""
        E = self.material["E"]
        nu = self.material["nu"]
        return self.sigma_bs * (1 - self.material["nu"]) * self.R / E

    @property
    def strain_at_fracture(self) -> float:
        return self.delta_bs / self.R

    def exact_stress_strain(self, n_points: int = 200) -> Dict:
        """Generate the exact stress-strain curve.

        Returns
        -------
        result : dict with keys: strain, stress, delta, delta_bs, sigma_bs
        """
        delta_max = self.delta_bs * 1.5
        delta_values = np.linspace(0, delta_max, n_points)
        strain, stress = stress_strain_response(
            delta_values, R=self.R, material=self.material,
        )
        return {
            "strain": strain,
            "stress": stress,
            "delta": delta_values,
            "delta_bs": self.delta_bs,
            "sigma_bs": self.sigma_bs,
        }

    def sdf_intact(self) -> np.ndarray:
        """SDF grid for the intact plate (pre-fracture)."""
        return self._sdf_grid(cracked=False)

    def sdf_cracked(self, crack_angle: float = 0.0, delta: float = 0.02) -> np.ndarray:
        """SDF grid for the cracked plate (post-fracture)."""
        return self._sdf_grid(cracked=True, crack_angle=crack_angle, delta=delta)

    def _sdf_grid(
        self, cracked: bool = False,
        crack_angle: float = 0.0, delta: float = 0.02,
        resolution: int = 48,
    ) -> np.ndarray:
        extent = max(self.R, self.L) * 1.3
        lin_xy = np.linspace(-extent, extent, resolution)
        lin_z = np.linspace(-self.L, self.L, resolution)
        gx, gy, gz = np.meshgrid(lin_xy, lin_xy, lin_z, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        if cracked:
            vals = sdf_cracked_circular_plate(
                coords, R=self.R, L=self.L,
                crack_angle=crack_angle, delta=delta,
            )
        else:
            vals = sdf_circular_plate(coords, R=self.R, L=self.L)

        return vals.reshape(resolution, resolution, resolution).astype("float32")

    def topology_check(
        self, cracked: bool = False, resolution: int = 48,
    ) -> Dict:
        """Check topology of intact vs cracked plate.

        Intact plate: beta_0 = 1 (one component)
        Cracked plate: beta_0 = 2 (two components)
        """
        from atlas.topo.filtration import clip_to_interior

        grid = self.sdf_cracked(delta=0.1) if cracked else self.sdf_intact()
        grid_clipped = clip_to_interior(grid)

        try:
            from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
            from atlas.topo.ls_category import compute_m_min

            diagrams = compute_persistence_diagrams(grid_clipped, max_dimension=1)
            betti = betti_numbers_at(diagrams, t=-1e-6)
            m_min = compute_m_min(betti)

            return {
                "betti": betti,
                "M_min": m_min,
                "domain_split": betti.get(0, 0) >= 2,
                "has_gudhi": True,
            }
        except ImportError:
            return {"betti": {}, "M_min": 1, "domain_split": False, "has_gudhi": False}
