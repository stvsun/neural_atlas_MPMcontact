"""
atlas/topo/filtration.py
Sublevel-set filtration of the neural SDF for persistent homology.

The filtration {Omega_t = {x : s_theta(x) <= t}, t in [-inf, 0]} encodes
the topology of the domain Omega = {s_theta < 0}.  Topological events
(connected components born/merged, loops formed/filled, voids
created/destroyed) occur at the critical values of s_theta and are
recorded by persistent homology as birth-death pairs in Dgm_k.

Verification contract
---------------------
For an analytic SDF of a ball of radius r centered at the origin,
  s(x) = ||x|| - r
the interior filtration has exactly one H_0 birth at t = -r and no
H_1 or H_2 pairs — since a ball is contractible. Any implementation
that produces additional long-lived pairs has a bug.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional


def sample_sdf_on_grid(
    sdf_net: torch.nn.Module,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    resolution: int = 32,
    batch_size: int = 4096,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the neural SDF on a regular cubical grid.

    Parameters
    ----------
    sdf_net    : callable  R^3 -> R, the trained SDF network s_theta
    bbox_min   : [3] lower corner of bounding box
    bbox_max   : [3] upper corner of bounding box
    resolution : number of grid points per axis (total = resolution^3)
    batch_size : evaluation chunk size (avoid OOM on large grids)
    device     : torch device string

    Returns
    -------
    grid_vals : (resolution, resolution, resolution) float32 ndarray
                of SDF values — negative inside, positive outside
    coords    : (resolution^3, 3) float32 ndarray of grid coordinates
    """
    lin = [
        torch.linspace(float(bbox_min[i]), float(bbox_max[i]), resolution)
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*lin)
    coords_t = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1).to(device)

    vals_list = []
    sdf_net.eval()
    with torch.no_grad():
        for start in range(0, coords_t.shape[0], batch_size):
            chunk = coords_t[start : start + batch_size]
            v = sdf_net(chunk).squeeze(-1)
            vals_list.append(v.cpu().float())

    vals = torch.cat(vals_list, dim=0).numpy()
    grid_vals = vals.reshape(resolution, resolution, resolution)
    return grid_vals, coords_t.cpu().numpy()


def clip_to_interior(
    grid_vals: np.ndarray,
    t_max: float = 0.0,
    t_min: Optional[float] = None,
) -> np.ndarray:
    """
    Clip filtration values so that only the interior sublevel set matters.

    GUDHI's CubicalComplex sees the full real-valued function on the grid.
    Features that are born and die entirely in {s_theta > 0} (i.e., outside
    the body) are irrelevant.  Clipping at t_max=0 removes them.

    Parameters
    ----------
    grid_vals : (N, N, N) SDF values
    t_max     : upper clip threshold (0.0 = body boundary)
    t_min     : optional lower clip for numerical normalization

    Returns
    -------
    clipped : (N, N, N) array with values in [t_min, t_max]

    Notes
    -----
    After clipping, the filtration starts at the most negative SDF value
    (deep interior) and ends at 0 (boundary). GUDHI's sublevel-set convention
    is that simplex sigma enters the filtration at value f(sigma).
    """
    lo = t_min if t_min is not None else float(grid_vals.min())
    return np.clip(grid_vals, lo, t_max).astype(np.float32)


def filtration_value_range(grid_vals: np.ndarray) -> Tuple[float, float]:
    """Return (min, max) of filtration values — useful for setting thresholds."""
    return float(grid_vals.min()), float(grid_vals.max())


# ---------------------------------------------------------------------------
# Analytic SDFs for verification and unit tests
# ---------------------------------------------------------------------------

def sdf_ball(
    coords: np.ndarray,
    center: np.ndarray = None,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Analytic SDF for a solid ball: s(x) = ||x - center|| - radius.

    Topology: contractible => beta_0=1, beta_1=0, beta_2=0, cat=0, M_min=1.
    """
    if center is None:
        center = np.zeros(3)
    return np.linalg.norm(coords - center, axis=-1) - radius


def sdf_solid_torus(
    coords: np.ndarray,
    R: float = 1.0,
    r: float = 0.35,
) -> np.ndarray:
    """
    Analytic SDF for a solid torus with major radius R, minor radius r.

    Topology: homotopy-equivalent to S^1 =>
        beta_0=1, beta_1=1, beta_2=0, cat=1, M_min=2.

    Used in Sun (2026) Examples 3-5.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    q = np.sqrt(x**2 + y**2) - R
    return np.sqrt(q**2 + z**2) - r


def sdf_thick_spherical_shell(
    coords: np.ndarray,
    R_inner: float = 0.5,
    R_outer: float = 1.0,
) -> np.ndarray:
    """
    Analytic SDF for a thick spherical shell.

    Topology: homotopy-equivalent to S^2 =>
        beta_0=1, beta_1=0, beta_2=1, cat=1, M_min=2.
    """
    r = np.linalg.norm(coords, axis=-1)
    return np.maximum(R_inner - r, r - R_outer)
