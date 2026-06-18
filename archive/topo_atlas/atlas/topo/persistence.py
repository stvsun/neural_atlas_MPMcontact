"""
atlas/topo/persistence.py
Persistent homology of the SDF sublevel-set filtration.

Computes birth-death pairs (b, d) for each homological dimension k.
The persistence lifetime |d - b| measures feature robustness:
  - short-lived pairs (|d-b| small) = numerical noise from SDF approximation
  - long-lived pairs (|d-b| large)  = genuine topological handles of Omega

Key reference quantities
------------------------
Dgm_0  :  connected components  (beta_0 = 1 for a connected body)
Dgm_1  :  loops / tunnels       (beta_1 = 1 for the solid torus tunnel)
Dgm_2  :  enclosed voids        (beta_2 = 1 for a spherical shell)

Bottleneck stability theorem guarantees:
  d_B(Dgm_k(f), Dgm_k(g)) <= ||f - g||_inf
so small SDF approximation errors produce only small perturbations of
long-lived pairs.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

PersistenceDiagram = List[Tuple[float, float]]   # list of (birth, death) pairs

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def _require_gudhi() -> None:
    if not GUDHI_AVAILABLE:
        raise ImportError(
            "GUDHI is required for persistent homology computation.\n"
            "Install with:  pip install gudhi\n"
            "Documentation: https://gudhi.inria.fr/python/latest/"
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_persistence_diagrams(
    grid_vals: np.ndarray,
    max_dimension: int = 2,
) -> Dict[int, PersistenceDiagram]:
    """
    Compute persistent homology of the sublevel-set filtration via GUDHI.

    Uses GUDHI's CubicalComplex which operates directly on a regular grid,
    making it exact (no Rips/Cech approximation) and efficient O(N^3 log N).

    Parameters
    ----------
    grid_vals     : (N, N, N) filtration values (SDF clipped at t_max=0)
    max_dimension : highest homological dimension to compute (default 2)

    Returns
    -------
    diagrams : dict   k -> list of (birth, death) pairs
               death = np.inf means the feature is still alive at t_max

    Verification contract
    ---------------------
    Ball (contractible):
        diagrams[0] should have exactly 1 pair with death=inf
        diagrams[1] should be empty  (no loops)
        diagrams[2] should be empty  (no voids)

    Solid torus (homotopy ~ S^1):
        diagrams[0]: 1 essential pair (death=inf)
        diagrams[1]: 1 long-lived pair + possible short noise pairs
        diagrams[2]: empty

    These contracts are enforced in tests/test_topo_pipeline.py::TestVerification.
    """
    _require_gudhi()

    cc = gudhi.CubicalComplex(top_dimensional_cells=grid_vals)
    cc.compute_persistence(min_persistence=0.0, homology_coeff_field=2)

    diagrams: Dict[int, PersistenceDiagram] = {k: [] for k in range(max_dimension + 1)}
    for dim, (birth, death) in cc.persistence():
        if dim <= max_dimension:
            diagrams[dim].append((float(birth), float(death)))

    return diagrams


# ---------------------------------------------------------------------------
# Filtering and analysis
# ---------------------------------------------------------------------------

def filter_by_lifetime(
    diagrams: Dict[int, PersistenceDiagram],
    threshold: float,
    relative: bool = True,
    filtration_range: Optional[Tuple[float, float]] = None,
) -> Dict[int, PersistenceDiagram]:
    """
    Retain only persistence pairs with lifetime |d - b| > threshold.

    Parameters
    ----------
    diagrams         : raw persistence diagrams from compute_persistence_diagrams
    threshold        : minimum lifetime to keep
    relative         : if True, threshold is fraction of total filtration range
    filtration_range : (f_min, f_max) needed when relative=True

    Returns
    -------
    filtered : persistence diagrams with only significant pairs retained

    Notes
    -----
    The choice threshold ~ 0.05 * |f_max - f_min| is empirically robust
    for SDFs normalized so the domain diameter ~ 1. For raw medical scan
    SDFs or unnormalized point clouds, prefer absolute thresholds.
    """
    if relative:
        if filtration_range is None:
            all_births = [b for pairs in diagrams.values() for (b, _) in pairs]
            lo = min(all_births) if all_births else 0.0
            hi = 0.0   # clipped at boundary
            filtration_range = (lo, hi)
        span = max(abs(filtration_range[1] - filtration_range[0]), 1e-8)
        abs_threshold = threshold * span
    else:
        abs_threshold = threshold

    filtered: Dict[int, PersistenceDiagram] = {}
    for dim, pairs in diagrams.items():
        filtered[dim] = [
            (b, d) for (b, d) in pairs
            if (d == np.inf) or ((d - b) > abs_threshold)
        ]
    return filtered


def betti_numbers_at(
    diagrams: Dict[int, PersistenceDiagram],
    t: float = -1e-6,
) -> Dict[int, int]:
    """
    Evaluate Betti numbers beta_k at filtration parameter t.

    beta_k(t) = |{(b,d) in Dgm_k : b <= t < d}|

    For atlas construction, use t = 0^- (just inside the body boundary)
    to get the Betti numbers of the domain Omega itself.

    Parameters
    ----------
    diagrams : persistence diagrams (may be filtered or raw)
    t        : evaluation parameter (default -1e-6, just below the boundary)

    Returns
    -------
    betti : dict  k -> beta_k(t)

    Examples (Sun 2026 benchmarks)
    -------------------------------
    Ellipsoid (simply connected):  {0:1, 1:0, 2:0}
    Solid torus:                   {0:1, 1:1, 2:0}
    Stanford Bunny solid:          {0:1, 1:0, 2:0}  (genus-0 surface)
    """
    betti: Dict[int, int] = {}
    for dim, pairs in diagrams.items():
        betti[dim] = sum(
            1 for (b, d) in pairs
            if (b <= t) and ((d > t) or (d == np.inf))
        )
    return betti


# ---------------------------------------------------------------------------
# Bottleneck distance (for dynamic topology monitoring)
# ---------------------------------------------------------------------------

def bottleneck_distance(
    dgm_a: PersistenceDiagram,
    dgm_b: PersistenceDiagram,
) -> float:
    """
    Approximate bottleneck distance between two persistence diagrams.

    d_B(A, B) = inf_{matching gamma} max_{p in A} ||p - gamma(p)||_inf

    This implementation uses a simple greedy nearest-neighbor matching
    with diagonal projections.  For a certified computation, use
    gudhi.bottleneck_distance(dgm_a, dgm_b).

    Parameters
    ----------
    dgm_a, dgm_b : persistence diagrams as lists of (birth, death) pairs

    Returns
    -------
    distance : approximate bottleneck distance

    Notes
    -----
    The Bottleneck Stability Theorem guarantees:
        d_B(Dgm_k(f), Dgm_k(g)) <= ||f - g||_inf
    so for the atlas application, a bottleneck distance exceeding
    the SDF approximation error indicates a genuine topology change.
    """
    if GUDHI_AVAILABLE:
        try:
            return gudhi.bottleneck_distance(dgm_a, dgm_b)
        except Exception:
            pass  # fall back to greedy

    # Greedy fallback: O(n^2), sufficient for small diagrams
    if not dgm_a and not dgm_b:
        return 0.0
    if not dgm_a or not dgm_b:
        # Distance to diagonal
        pts = dgm_a if dgm_a else dgm_b
        return max(abs(d - b) / 2.0 for (b, d) in pts if d != np.inf)

    cost = np.zeros((len(dgm_a), len(dgm_b)))
    for i, (b1, d1) in enumerate(dgm_a):
        for j, (b2, d2) in enumerate(dgm_b):
            if d1 == np.inf or d2 == np.inf:
                cost[i, j] = np.inf if (d1 != d2) else abs(b1 - b2)
            else:
                cost[i, j] = max(abs(b1 - b2), abs(d1 - d2))

    return float(cost.min(axis=1).max())
