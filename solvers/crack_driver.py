"""Crack propagation driver for quasi-static fracture simulation.

Implements the load-solve-check-grow loop:
    1. Solve elasticity BVP on the current cracked geometry
    2. Extract K_I at the crack tip
    3. If K_I >= K_Ic: advance crack by da, update SDF
    4. Optionally: run TopologyMonitor to detect domain splitting

Usage:
    driver = CrackPropagationDriver(sdf_oracle, solver_factory, ...)
    history = driver.run(n_steps=20)
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch


class CrackPropagationDriver:
    """Quasi-static crack propagation with Griffith criterion.

    Parameters
    ----------
    sdf_oracle : CrackedPlateSDFOracle
        SDF oracle with .update_crack_length(a) and .sdf(x).
    solver_factory : callable
        (sdf_oracle) -> configured solver with .solve() returning displacement.
        Called each time the crack advances to rebuild the mesh.
    extract_K_I_fn : callable
        (u, nodes, crack_tip, E, nu) -> float. Extracts K_I from displacement.
    crack_tip_fn : callable
        (a, W) -> (3,) crack tip coordinates.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    W : float
        Plate half-width.
    a_init : float
        Initial crack length.
    K_Ic : float
        Critical stress intensity factor (fracture toughness).
    da : float
        Crack increment per propagation step.
    sigma_inf : float
        Applied far-field stress.
    monitor : TopologyMonitor, optional
        If provided, checks topology at each step.
    spawner : ChartSpawner, optional
        If provided with monitor, spawns charts on topology events.
    """

    def __init__(
        self,
        sdf_oracle,
        solver_factory: Callable,
        extract_K_I_fn: Callable,
        crack_tip_fn: Callable,
        E: float,
        nu: float,
        W: float,
        a_init: float,
        K_Ic: float,
        da: float,
        sigma_inf: float = 1.0,
        monitor=None,
        spawner=None,
    ):
        self.sdf_oracle = sdf_oracle
        self.solver_factory = solver_factory
        self.extract_K_I_fn = extract_K_I_fn
        self.crack_tip_fn = crack_tip_fn
        self.E = E
        self.nu = nu
        self.W = W
        self.a = a_init
        self.K_Ic = K_Ic
        self.da = da
        self.sigma_inf = sigma_inf
        self.monitor = monitor
        self.spawner = spawner

        self.history: List[Dict] = []
        self.topology_events: List = []

    @property
    def crack_tip(self):
        return self.crack_tip_fn(self.a, self.W)

    @property
    def a_over_W(self):
        return self.a / self.W

    def step(self) -> Dict:
        """Execute one quasi-static step: solve -> extract K_I -> grow if critical.

        Returns
        -------
        result : dict with keys: step, a, a_over_W, K_I, K_Ic, propagated, events.
        """
        step_num = len(self.history)

        # 1. Build solver on current geometry
        solver, u, nodes, bc_mask = self.solver_factory(self.sdf_oracle)

        # 2. Extract K_I
        crack_tip = self.crack_tip
        K_I = self.extract_K_I_fn(u, nodes, crack_tip, self.E, self.nu, bc_mask)

        # 3. Check Griffith criterion
        propagated = False
        if K_I >= self.K_Ic and self.a + self.da <= 2 * self.W:
            self.a += self.da
            self.sdf_oracle.update_crack_length(self.a)
            propagated = True

        # 4. Topology monitoring
        events = []
        if self.monitor is not None and propagated:
            from atlas.topo.filtration import clip_to_interior
            grid = self.sdf_oracle.sdf_grid(resolution=32)
            grid_clipped = clip_to_interior(grid)
            events = self.monitor.update(grid_clipped, load_step=step_num)
            self.topology_events.extend(events)

        result = {
            "step": step_num,
            "a": self.a,
            "a_over_W": self.a_over_W,
            "K_I": K_I,
            "K_Ic": self.K_Ic,
            "propagated": propagated,
            "n_events": len(events),
        }
        self.history.append(result)
        return result

    def run(self, n_steps: int, verbose: bool = True) -> List[Dict]:
        """Run n_steps of quasi-static crack propagation.

        Returns
        -------
        history : list of step dicts.
        """
        for i in range(n_steps):
            result = self.step()
            if verbose:
                status = "GROW" if result["propagated"] else "hold"
                topo = f" [{result['n_events']} topo events]" if result["n_events"] > 0 else ""
                print(
                    f"  Step {result['step']:3d} | "
                    f"a/W={result['a_over_W']:.3f} | "
                    f"K_I={result['K_I']:.4f} | "
                    f"K_Ic={result['K_Ic']:.4f} | "
                    f"{status}{topo}"
                )
            # Stop if crack reached full width
            if self.a >= 2 * self.W:
                if verbose:
                    print("  Crack reached full width — stopping.")
                break
        return self.history
