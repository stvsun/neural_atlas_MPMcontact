"""Analytical SDF for a rectangular plate with an edge crack.

Provides an SDF oracle for a 2D plate (extruded to 3D) with a Mode-I
edge crack, suitable for topology-aware atlas construction.

Geometry (plane-strain in x1-x2, extruded along x3):
    Plate: [-W, W] x [-H/2, H/2] x [-T/2, T/2]
    Crack: line segment from (-W, 0, z) to (-W + a, 0, z) for all z
           represented as a thin slit with half-opening delta

The SDF is negative inside the cracked plate and positive outside.
The crack introduces an H1 topological feature detectable by persistent
homology when the crack length a is sufficiently large.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch


def sdf_plate(
    x: np.ndarray,
    W: float = 1.0,
    H: float = 4.0,
    T: float = 0.5,
) -> np.ndarray:
    """SDF of a rectangular plate [-W, W] x [-H/2, H/2] x [-T/2, T/2].

    Parameters
    ----------
    x : ndarray (N, 3)
    W, H, T : float
        Half-width, full height, full thickness.

    Returns
    -------
    sdf : ndarray (N,)
        Negative inside, positive outside.
    """
    dx = np.abs(x[:, 0]) - W
    dy = np.abs(x[:, 1]) - H / 2.0
    dz = np.abs(x[:, 2]) - T / 2.0

    # Outside distance = max(dx, dy, dz, 0)
    outside = np.sqrt(
        np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2
    )
    # Inside distance = min(max(dx, dy, dz), 0)
    inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)

    return outside + inside


def sdf_edge_crack(
    x: np.ndarray,
    a: float,
    W: float = 1.0,
    delta: float = 0.02,
) -> np.ndarray:
    """SDF of a thin crack slit extending from x1=-W to x1=-W+a along x2=0.

    The crack is modeled as a thin rectangle of half-opening delta.

    Parameters
    ----------
    x : ndarray (N, 3)
    a : float
        Crack length.
    W : float
        Plate half-width.
    delta : float
        Half-opening of the crack slit.

    Returns
    -------
    sdf : ndarray (N,)
        Negative inside the crack slit, positive outside.
    """
    # Crack centerline: x1 in [-W, -W+a], x2 = 0
    # Transform to crack-local coordinates
    x1_local = x[:, 0] - (-W + a / 2.0)  # center of crack
    x2_local = x[:, 1]

    # Rectangular slit in x1-x2: [-a/2, a/2] x [-delta, delta]
    dx = np.abs(x1_local) - a / 2.0
    dy = np.abs(x2_local) - delta

    outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0)

    return outside + inside


def sdf_cracked_plate(
    x: np.ndarray,
    a: float,
    W: float = 1.0,
    H: float = 4.0,
    T: float = 0.5,
    delta: float = 0.02,
) -> np.ndarray:
    """SDF of plate with edge crack: plate minus crack slit.

    The cracked plate is the set-difference: plate \ crack_slit.
    SDF is computed as max(sdf_plate, -sdf_crack).

    Parameters
    ----------
    x : ndarray (N, 3)
    a : float
        Crack length.
    W : float
        Plate half-width (plate spans [-W, W] in x1).
    H : float
        Plate full height.
    T : float
        Plate full thickness.
    delta : float
        Crack slit half-opening.

    Returns
    -------
    sdf : ndarray (N,)
        Negative inside the cracked plate, positive outside or in crack.
    """
    d_plate = sdf_plate(x, W=W, H=H, T=T)
    d_crack = sdf_edge_crack(x, a=a, W=W, delta=delta)

    # CSG subtraction: plate \ crack = intersection(plate, complement(crack))
    # SDF(A \ B) = max(SDF_A, -SDF_B)
    return np.maximum(d_plate, -d_crack)


class CrackedPlateSDFOracle:
    """Torch-compatible SDF oracle for the cracked plate.

    Implements the .sdf(x) interface expected by ChartFEMSolver.

    Parameters
    ----------
    a : float
        Current crack length.
    W, H, T : float
        Plate dimensions.
    delta : float
        Crack slit half-opening.
    """

    def __init__(
        self,
        a: float = 0.5,
        W: float = 1.0,
        H: float = 4.0,
        T: float = 0.5,
        delta: float = 0.02,
    ):
        self.a = a
        self.W = W
        self.H = H
        self.T = T
        self.delta = delta

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        vals = sdf_cracked_plate(
            x_np, a=self.a, W=self.W, H=self.H, T=self.T, delta=self.delta,
        )
        return torch.tensor(vals, dtype=x.dtype, device=x.device)

    def sdf_grid(self, resolution: int = 32) -> np.ndarray:
        """Evaluate SDF on a regular grid for topology analysis.

        Returns
        -------
        grid_vals : ndarray (resolution, resolution, resolution)
        """
        extent = max(self.W, self.H / 2, self.T / 2) * 1.2
        lin = np.linspace(-extent, extent, resolution)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = sdf_cracked_plate(
            coords, a=self.a, W=self.W, H=self.H, T=self.T, delta=self.delta,
        )
        return vals.reshape(resolution, resolution, resolution).astype("float32")

    def update_crack_length(self, a_new: float) -> None:
        """Update the crack length (for quasi-static crack growth)."""
        self.a = a_new
