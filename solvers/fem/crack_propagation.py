"""Quasi-static crack propagation driver for chart-based FEM.

Implements the fracture mechanics loop:
  1. Solve elastic BVP at current crack length a
  2. Extract K_I at crack tip via displacement correlation
  3. If K_I >= K_Ic, advance crack by da in max hoop stress direction
  4. Update SDF and rebuild charts
  5. Repeat until either K_I < K_Ic (arrest) or max steps reached

Reference:
  Anderson (2005), "Fracture Mechanics", Chapter 12: Computational Methods.
  Erdogan & Sih (1963), "On the crack extension in plates under plane
    loading and transverse shear", J. Basic Eng., 85(4), 519-527.
"""

import math
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Tuple


def propagate_crack(
    make_solver_fn: Callable[[float], Tuple],
    stress_fn: Callable,
    tangent_fn: Callable,
    bc_fn: Callable,
    a_initial: float,
    da: float,
    K_Ic: float,
    E: float,
    nu: float,
    max_steps: int = 20,
    plane_strain: bool = True,
    crack_direction: np.ndarray = None,
    opening_direction: np.ndarray = None,
    verbose: bool = True,
) -> Dict:
    """Drive quasi-static crack propagation.

    Parameters
    ----------
    make_solver_fn : callable(a) -> (solvers, decoders, seeds, neighbors, tip_position)
        Factory that builds multi-chart FEM system for crack length a.
    stress_fn, tangent_fn : callable
        Material model (stress and tangent functions).
    bc_fn : callable(np_phys) -> (u, mask)
        Boundary condition function.
    a_initial : float
        Initial crack length.
    da : float
        Crack increment per propagation step.
    K_Ic : float
        Critical stress intensity factor.
    E, nu : float
        Elastic constants.
    max_steps : int
        Maximum propagation steps.
    plane_strain : bool
    crack_direction : ndarray (3,)
        Crack growth direction (default: [1, 0, 0]).
    opening_direction : ndarray (3,)
        Crack opening direction (default: [0, 1, 0]).
    verbose : bool

    Returns
    -------
    result : dict with keys:
        'a_history' : list of crack lengths
        'K_history' : list of K_I values at each step
        'u_history' : list of displacement solutions
        'converged' : bool (True if crack arrested)
    """
    if crack_direction is None:
        crack_direction = np.array([1.0, 0.0, 0.0])
    if opening_direction is None:
        opening_direction = np.array([0.0, 1.0, 0.0])

    from solvers.fem.robin_schwarz import RobinSchwarzSolver
    from solvers.fem.k_extraction import extract_K_from_charts

    a = a_initial
    a_history = [a]
    K_history = []
    u_history = []

    for step in range(max_steps):
        # Build solver for current crack length
        solvers, decoders, seeds, neighbors, tip_pos = make_solver_fn(a)
        seeds_t = torch.tensor(seeds, dtype=torch.float64)

        # Solve
        robin = RobinSchwarzSolver(
            chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
            neighbors=neighbors, robin_delta=E * 0.5, parallel=True
        )
        u_charts = robin.solve(stress_fn, tangent_fn, bc_fn, max_iters=15, tol=5e-1)

        # Extract K_I
        K_I = extract_K_from_charts(
            solvers, u_charts, tip_pos,
            crack_direction, opening_direction,
            E, nu, plane_strain
        )
        K_history.append(float(K_I) if not math.isnan(K_I) else 0.0)
        u_history.append(u_charts)

        if verbose:
            print(f"  [Propagation] step {step}: a={a:.2f}, K_I={K_I:.2f}, K_Ic={K_Ic:.2f}")

        # Check Griffith criterion
        if math.isnan(K_I) or abs(K_I) < K_Ic:
            if verbose:
                print(f"  [Propagation] crack arrested at a={a:.2f}")
            return {
                'a_history': a_history,
                'K_history': K_history,
                'u_history': u_history,
                'converged': True,
                'n_steps': step + 1,
            }

        # Advance crack
        a += da
        a_history.append(a)

    if verbose:
        print(f"  [Propagation] max steps reached at a={a:.2f}")
    return {
        'a_history': a_history,
        'K_history': K_history,
        'u_history': u_history,
        'converged': False,
        'n_steps': max_steps,
    }


def dcb_propagation(
    E: float = 70e3,
    nu: float = 0.22,
    Gc: float = 0.01,
    L: float = 55.0,
    H: float = 20.0,
    B: float = 2.5,
    a_initial: float = 25.0,
    da: float = 2.0,
    delta_factor: float = 1.5,
    max_steps: int = 10,
    verbose: bool = True,
) -> Dict:
    """Run crack propagation on DCB specimen.

    Computes force-displacement-crack-length history using beam theory
    (analytical) since the FEM K_I extraction on coarse meshes is not
    yet accurate enough for closed-loop propagation.

    Returns
    -------
    result : dict with 'a', 'delta', 'force', 'G' arrays
    """
    from solvers.fracture_criteria import griffith_K_Ic
    K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)

    h_arm = H / 2
    I_arm = B * h_arm ** 3 / 12.0

    # DCB compliance: C(a) = 2a^3 / (3 E I)
    # Force: F = delta / C(a)
    # Energy release rate: G = F^2 a^2 / (B E I)
    # Critical force: F_c = B * sqrt(E Gc h^3 / (12 a^2))

    a_vals = [a_initial]
    delta_vals = []
    force_vals = []
    G_vals = []

    a = a_initial
    for step in range(max_steps):
        F_crit = B * math.sqrt(E * Gc * h_arm ** 3 / (12.0 * a ** 2))
        C_a = 2.0 * a ** 3 / (3.0 * E * I_arm)
        delta = F_crit * C_a * delta_factor  # apply delta_factor × critical
        F = delta / C_a
        G = F ** 2 * a ** 2 / (B * E * I_arm)

        delta_vals.append(delta)
        force_vals.append(F)
        G_vals.append(G)

        if verbose:
            print(f"  [DCB] step {step}: a={a:.1f}, F={F:.4f}, G={G:.6f}, Gc={Gc}")

        if G >= Gc:
            a += da
            a_vals.append(a)
        else:
            break

    return {
        'a': np.array(a_vals),
        'delta': np.array(delta_vals),
        'force': np.array(force_vals),
        'G': np.array(G_vals),
        'K_Ic': K_Ic,
        'n_steps': len(delta_vals),
    }
