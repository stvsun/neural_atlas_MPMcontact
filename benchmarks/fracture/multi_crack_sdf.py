"""Multi-crack SDF oracle for arbitrary crack nucleation and propagation.

Extends the single-crack SDF to support multiple cracks at arbitrary
locations and orientations, nucleated from the stress field via the
Drucker-Prager criterion.

Each crack is represented as a thin planar slit (center, normal, half_length)
subtracted from the base domain via CSG.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


class MultiCrackSDFOracle:
    """SDF oracle with support for multiple arbitrarily-oriented cracks.

    Parameters
    ----------
    base_sdf_fn : callable
        base_sdf_fn(x: ndarray (N,3)) -> ndarray (N,) for the intact domain.
    bbox : tuple of ndarray
        (bbox_min (3,), bbox_max (3,)) bounding box for grid evaluation.
    delta : float
        Half-opening of each crack slit.
    """

    def __init__(
        self,
        base_sdf_fn,
        bbox: Tuple[np.ndarray, np.ndarray],
        delta: float = 0.02,
    ):
        self.base_sdf_fn = base_sdf_fn
        self.bbox_min = np.asarray(bbox[0], dtype=np.float64)
        self.bbox_max = np.asarray(bbox[1], dtype=np.float64)
        self.delta = delta

        # List of cracks: (center (3,), normal (3,), tangent (3,), half_length)
        self.cracks: List[dict] = []

    @property
    def n_cracks(self) -> int:
        return len(self.cracks)

    def add_crack(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        half_length: float,
        tangent: Optional[np.ndarray] = None,
    ) -> int:
        """Add a new crack to the domain.

        Parameters
        ----------
        center : ndarray (3,)
            Crack center in physical coordinates.
        normal : ndarray (3,)
            Unit normal to the crack plane (crack opens in this direction).
        half_length : float
            Half-length of the crack slit.
        tangent : ndarray (3,), optional
            In-plane tangent direction (crack extends along this). If None,
            computed from normal via Gram-Schmidt.

        Returns
        -------
        crack_id : int
            Index of the new crack.
        """
        normal = np.asarray(normal, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        center = np.asarray(center, dtype=np.float64)

        if tangent is None:
            # Build tangent perpendicular to normal
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(normal, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            tangent = np.cross(normal, ref)
            tangent = tangent / np.linalg.norm(tangent)

        tangent = np.asarray(tangent, dtype=np.float64)
        tangent = tangent / np.linalg.norm(tangent)

        self.cracks.append({
            "center": center.copy(),
            "normal": normal.copy(),
            "tangent": tangent.copy(),
            "half_length": float(half_length),
        })
        return len(self.cracks) - 1

    def advance_crack(self, crack_id: int, da: float, direction: Optional[np.ndarray] = None) -> None:
        """Advance a crack tip by da in the given direction.

        Parameters
        ----------
        crack_id : int
            Index of the crack to advance.
        da : float
            Increment in crack half-length.
        direction : ndarray (3,), optional
            Propagation direction. If None, uses the existing tangent.
        """
        crack = self.cracks[crack_id]
        if direction is not None:
            direction = np.asarray(direction, dtype=np.float64)
            direction = direction / np.linalg.norm(direction)
            # Shift center forward by da/2 and increase half_length
            crack["center"] = crack["center"] + (da / 2.0) * direction
            crack["tangent"] = direction.copy()
        crack["half_length"] += da

    def crack_tips(self, crack_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the two tip positions of a crack.

        Returns
        -------
        tip_plus, tip_minus : ndarray (3,)
        """
        c = self.cracks[crack_id]
        offset = c["half_length"] * c["tangent"]
        return c["center"] + offset, c["center"] - offset

    def _sdf_single_crack(self, x: np.ndarray, crack: dict) -> np.ndarray:
        """SDF of a single planar crack slit."""
        center = crack["center"]
        normal = crack["normal"]
        tangent = crack["tangent"]
        a = crack["half_length"]

        # Project onto crack-local coordinates
        dx = x - center[None, :]
        d_normal = np.dot(dx, normal)           # distance along normal
        d_tangent = np.dot(dx, tangent)          # distance along tangent
        # Third axis (binormal) not needed for 2D slit

        # Slit: |d_tangent| <= a, |d_normal| <= delta
        dx_slit = np.abs(d_tangent) - a
        dy_slit = np.abs(d_normal) - self.delta

        outside = np.sqrt(np.maximum(dx_slit, 0)**2 + np.maximum(dy_slit, 0)**2)
        inside = np.minimum(np.maximum(dx_slit, dy_slit), 0)
        return outside + inside

    def sdf_np(self, x: np.ndarray) -> np.ndarray:
        """Evaluate SDF at points (numpy interface).

        SDF = max(base_sdf, -min(crack_sdf_1, crack_sdf_2, ...))
        i.e., domain = base ∩ complement(union(cracks))
        """
        d_base = self.base_sdf_fn(x)

        if len(self.cracks) == 0:
            return d_base

        # CSG subtraction: for each crack, d_domain = max(d_domain, -d_crack)
        d_domain = d_base.copy()
        for crack in self.cracks:
            d_crack = self._sdf_single_crack(x, crack)
            d_domain = np.maximum(d_domain, -d_crack)

        return d_domain

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate SDF at points (torch interface for ChartFEMSolver)."""
        x_np = x.detach().cpu().numpy()
        vals = self.sdf_np(x_np)
        return torch.tensor(vals, dtype=x.dtype, device=x.device)

    def sdf_grid(self, resolution: int = 32) -> np.ndarray:
        """Evaluate SDF on a regular grid for topology analysis."""
        lin = [np.linspace(self.bbox_min[i], self.bbox_max[i], resolution)
               for i in range(3)]
        gx, gy, gz = np.meshgrid(*lin, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = self.sdf_np(coords)
        return vals.reshape(resolution, resolution, resolution).astype("float32")
