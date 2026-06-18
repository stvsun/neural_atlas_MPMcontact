"""
atlas/topo/ls_category.py
Lusternik-Schnirelmann category and atlas chart count bounds.

The Lusternik-Schnirelmann (LS) category cat(X) of a topological space X
is the minimum k such that X can be covered by k+1 open sets, each
contractible in X.  By the nerve theorem, the minimum number of
contractible charts in a good cover is exactly M_min = cat(X) + 1.

Key values for mechanics domains
---------------------------------
Domain                  |  cat  |  M_min  |  Notes
------------------------|-------|---------|---------------------------
Ball / convex body      |   0   |    1    |  simply connected
Ellipsoid               |   0   |    1    |  Sun (2026) Example 1
Stanford Bunny solid    |   0   |    1    |  genus-0 surface boundary
Solid torus             |   1   |    2    |  homotopy ~ S^1; paper M=8
Spherical shell         |   1   |    2    |  homotopy ~ S^2
Genus-g handlebody      |   1   |    2    |  all g >= 1
Torus surface T^2       |   2   |    3    |  beta_1=2 drives cup-length
S^1 x S^1 x S^1         |   3   |    4    |

Note: M_practical >> M_min in general due to Jacobian conditioning
and Schwarz convergence requirements.  This module computes only the
topological floor M_min; the quality-gate machinery (atlas.quality_gates)
provides the numerical ceiling.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# LS category from Betti numbers
# ---------------------------------------------------------------------------

def cup_length_lower_bound(betti: Dict[int, int]) -> int:
    """
    Lower bound on cat(Omega) from Betti numbers (simplified cup-length).

    The true cup-length requires the multiplicative structure of the
    cohomology ring H*(Omega; Z_2), which is not recoverable from Betti
    numbers alone.  This simplified bound counts the number of non-zero
    higher Betti numbers:

        cat(Omega) >= |{k >= 1 : beta_k > 0}|

    This bound is tight for all domains in the Sun (2026) benchmark suite
    and for all orientable surfaces.  It may underestimate cat for products
    of spheres (e.g., S^1 x S^2 has cat=2 but this bound gives 1).

    Parameters
    ----------
    betti : dict  k -> beta_k  from persistence.betti_numbers_at()

    Returns
    -------
    lower_bound : non-negative integer, lower bound on cat(Omega)

    Verification table
    ------------------
    Manifold          betti              bound  true cat  correct?
    ---------------   -----------------  -----  --------  ---------
    Ball              {0:1,1:0,2:0}        0       0       YES
    Solid torus       {0:1,1:1,2:0}        1       1       YES
    Spherical shell   {0:1,1:0,2:1}        1       1       YES
    Torus surface     {0:1,1:2,2:1}        2       2       YES
    S^1 x S^2         {0:1,1:1,2:1}        2       2       YES
    """
    return sum(1 for k, b in betti.items() if k >= 1 and b > 0)


def compute_m_min(betti: Dict[int, int]) -> int:
    """
    Topological minimum chart count: M_min = cat(Omega) + 1.

    This is the hard topological floor — any atlas with fewer charts
    cannot form a good cover of Omega by the nerve theorem, regardless
    of how the charts are shaped.

    Parameters
    ----------
    betti : Betti numbers of Omega from persistence.betti_numbers_at()

    Returns
    -------
    M_min : positive integer, minimum number of charts

    Examples
    --------
    >>> betti_ball = {0:1, 1:0, 2:0}
    >>> compute_m_min(betti_ball)
    1
    >>> betti_torus = {0:1, 1:1, 2:0}
    >>> compute_m_min(betti_torus)
    2
    """
    return cup_length_lower_bound(betti) + 1


# ---------------------------------------------------------------------------
# Practical chart count recommendation
# ---------------------------------------------------------------------------

def quality_driven_m_practical(
    betti: Dict[int, int],
    kappa_target: float = 10.0,
    alpha_overlap: float = 0.20,
    domain_diameter: float = 1.0,
    target_chart_diameter: Optional[float] = None,
) -> Tuple[int, int, str]:
    """
    Recommend a practical chart count balancing topology and numerics.

    Two independent constraints determine M:
    1. Topological:  M >= M_min = cat(Omega) + 1
    2. Numerical:    M >= volume(Omega) / (pi/6 * r_chart^3) where r_chart
       is the maximum radius achieving kappa_95 < kappa_target

    Parameters
    ----------
    betti                : Betti numbers of Omega
    kappa_target         : target 95th-percentile Jacobian condition number
                          (paper Table 4: observed max 9.49, target <= 10)
    alpha_overlap        : Schwarz overlap fraction alpha (Eq. 37, paper)
    domain_diameter      : approximate diameter of Omega for scaling
    target_chart_diameter: if given, directly compute M from volume ratio

    Returns
    -------
    M_min        : topological floor
    M_practical  : recommended practical count
    explanation  : human-readable justification string

    Notes
    -----
    The gap M_practical / M_min can be large:
      Solid torus benchmark: M_min=2, M_practical=8 (paper uses 8)
    This gap is entirely due to numerical requirements.
    """
    M_min = compute_m_min(betti)

    # Heuristic numerical estimate
    if target_chart_diameter is not None:
        import math
        vol_domain_approx = (4.0 / 3.0) * math.pi * (domain_diameter / 2.0) ** 3
        vol_chart = (4.0 / 3.0) * math.pi * (target_chart_diameter / 2.0) ** 3
        M_numerical = max(1, int(math.ceil(vol_domain_approx / vol_chart)))
    else:
        # Rule of thumb: r_chart ~ domain_diameter / (2 * sqrt(kappa_target))
        import math
        r_chart = domain_diameter / (2.0 * math.sqrt(kappa_target))
        vol_domain = (4.0 / 3.0) * math.pi * (domain_diameter / 2.0) ** 3
        vol_chart = (4.0 / 3.0) * math.pi * r_chart ** 3
        M_numerical = max(1, int(math.ceil(vol_domain / vol_chart * (1.0 + alpha_overlap))))

    M_practical = max(M_min, M_numerical)

    explanation = (
        f"M_min={M_min} (from cat(Omega)+1, Betti={betti}); "
        f"M_numerical~{M_numerical} (kappa_target={kappa_target}, "
        f"alpha={alpha_overlap}); "
        f"M_practical={M_practical} (max of both constraints)"
    )
    return M_min, M_practical, explanation


# ---------------------------------------------------------------------------
# Certification check (run after atlas is built)
# ---------------------------------------------------------------------------

def certify_atlas(
    M_actual: int,
    betti: Dict[int, int],
    quality_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Check whether a built atlas satisfies both topological and quality constraints.

    Parameters
    ----------
    M_actual         : number of charts in the built atlas
    betti            : Betti numbers of Omega
    quality_metrics  : optional dict with keys from the paper's quality gates:
                       'g_cov'   : coverage fraction (should be ~1.0)
                       'g_ov'    : mean transition map error (should be < 0.05)
                       'g_fold'  : foldover fraction (must be 0.0 on interior)
                       'g_rmse'  : PoU-weighted reconstruction RMSE

    Returns
    -------
    report : dict with keys:
             'topology_pass' : bool  (M_actual >= M_min)
             'M_min'         : int
             'M_actual'      : int
             'quality_pass'  : bool or None if metrics not provided
             'messages'      : list of str
    """
    M_min = compute_m_min(betti)
    topo_pass = M_actual >= M_min
    messages = []

    if topo_pass:
        messages.append(f"PASS: M_actual={M_actual} >= M_min={M_min}")
    else:
        messages.append(
            f"FAIL: M_actual={M_actual} < M_min={M_min}. "
            f"The atlas cannot form a good cover by the nerve theorem."
        )

    quality_pass = None
    if quality_metrics is not None:
        quality_pass = True
        if quality_metrics.get("g_fold", 0.0) > 0.0:
            messages.append(f"FAIL: g_fold={quality_metrics['g_fold']:.4f} > 0 (foldover on interior)")
            quality_pass = False
        if quality_metrics.get("g_cov", 1.0) < 0.99:
            messages.append(f"FAIL: g_cov={quality_metrics['g_cov']:.4f} < 0.99 (incomplete coverage)")
            quality_pass = False
        if quality_metrics.get("g_ov", 0.0) > 0.05:
            messages.append(f"WARN: g_ov={quality_metrics['g_ov']:.4f} > 0.05 (high transition error)")
        if quality_pass and topo_pass:
            messages.append("PASS: All quality gates satisfied.")

    return {
        "topology_pass": topo_pass,
        "M_min": M_min,
        "M_actual": M_actual,
        "quality_pass": quality_pass,
        "messages": messages,
    }
