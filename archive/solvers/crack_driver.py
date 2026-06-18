"""Crack propagation driver for quasi-static fracture simulation.

Implements the full nucleation-propagation loop:
    1. Solve elasticity BVP on the current geometry
    2. If no crack: check Drucker-Prager nucleation criterion pointwise
       -> if F(sigma) >= 0: nucleate crack at critical location
       -> crack direction = eigenvector of max principal stress
    3. If crack exists: extract K_I, check Griffith criterion K_I >= K_Ic
       -> advance crack by da in max hoop stress direction
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

        # Nucleation parameters (optional — if set, enables Drucker-Prager check)
        self.sigma_ts: Optional[float] = None
        self.sigma_hs: Optional[float] = None
        self.nucleation_check_fn: Optional[Callable] = None
        self.crack_nucleated = a_init > 0  # True if crack already exists

        self.history: List[Dict] = []
        self.topology_events: List = []

    @property
    def crack_tip(self):
        return self.crack_tip_fn(self.a, self.W)

    @property
    def a_over_W(self):
        return self.a / self.W

    def enable_nucleation(
        self,
        sigma_ts: float,
        sigma_hs: float,
        nucleation_check_fn: Optional[Callable] = None,
    ) -> None:
        """Enable Drucker-Prager nucleation criterion.

        When enabled, the driver checks the strength surface at all
        elements before checking the Griffith propagation criterion.

        Parameters
        ----------
        sigma_ts : float
            Uniaxial tensile strength.
        sigma_hs : float
            Hydrostatic tensile strength.
        nucleation_check_fn : callable, optional
            Custom nucleation checker. Default uses
            fracture_criteria.check_nucleation_pointwise.
        """
        self.sigma_ts = sigma_ts
        self.sigma_hs = sigma_hs
        self.nucleation_check_fn = nucleation_check_fn
        self.crack_nucleated = self.a > 0

    def step(self) -> Dict:
        """Execute one quasi-static step.

        If nucleation is enabled and no crack exists:
            Check Drucker-Prager -> nucleate if F(sigma) >= 0
        If crack exists:
            Extract K_I -> propagate if K_I >= K_Ic

        Returns
        -------
        result : dict with keys: step, a, a_over_W, K_I, K_Ic,
                 propagated, nucleated, nucleation_site, events.
        """
        step_num = len(self.history)

        # 1. Build solver on current geometry
        solver, u, nodes, bc_mask = self.solver_factory(self.sdf_oracle)

        nucleated = False
        propagated = False
        K_I = 0.0
        nucleation_site = None
        events = []

        # 2. Nucleation check (if enabled and no crack yet)
        if self.sigma_ts is not None and not self.crack_nucleated:
            from solvers.fracture_criteria import (
                check_nucleation_pointwise, cauchy_from_first_piola,
                crack_normal_from_stress,
            )

            # Evaluate stress at all elements if solver is available
            if solver is not None and hasattr(solver, 'compute_F') and u is not None:
                import torch as _torch
                u_t = _torch.tensor(u, dtype=_torch.float64) if not isinstance(u, _torch.Tensor) else u
                F_all = solver.compute_F(u_t).detach().cpu().numpy()
                # Use linear elastic stress
                from solvers.fem.linear_elastic import make_linear_elastic
                stress_fn, _ = make_linear_elastic(self.E, self.nu)
                P_all = stress_fn(_torch.tensor(F_all)).detach().cpu().numpy()
                sigma_all = cauchy_from_first_piola(P_all, F_all)

                centroids = nodes  # approximate: use node positions
                if hasattr(solver, 'nodes_phys'):
                    centroids_t = solver.nodes_phys
                    if centroids_t is not None and centroids_t.shape[0] > 0:
                        centroids = centroids_t.detach().cpu().numpy()

                sites = check_nucleation_pointwise(
                    sigma_all, centroids[:len(sigma_all)],
                    self.sigma_ts, self.sigma_hs,
                )
                if sites:
                    nucleation_site = sites[0]
                    nucleated = True
                    self.crack_nucleated = True
                    self.a = self.da  # initial crack size

                    # Update SDF with the new crack
                    if hasattr(self.sdf_oracle, 'add_crack'):
                        self.sdf_oracle.add_crack(
                            center=np.array(nucleation_site["center"]),
                            normal=np.array(nucleation_site["crack_normal"]),
                            half_length=self.da,
                        )
                    elif hasattr(self.sdf_oracle, 'update_crack_length'):
                        self.sdf_oracle.update_crack_length(self.a)

        # 3. Propagation check (if crack exists)
        if self.crack_nucleated and not nucleated:
            crack_tip = self.crack_tip
            K_I = self.extract_K_I_fn(u, nodes, crack_tip, self.E, self.nu, bc_mask)

            if K_I >= self.K_Ic and self.a + self.da <= 2 * self.W:
                self.a += self.da
                if hasattr(self.sdf_oracle, 'advance_crack'):
                    self.sdf_oracle.advance_crack(0, self.da)
                elif hasattr(self.sdf_oracle, 'update_crack_length'):
                    self.sdf_oracle.update_crack_length(self.a)
                propagated = True

        # 4. Topology monitoring
        if self.monitor is not None and (propagated or nucleated):
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
            "nucleated": nucleated,
            "nucleation_site": nucleation_site,
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
                if result["nucleated"]:
                    status = "NUCLEATE"
                elif result["propagated"]:
                    status = "GROW"
                else:
                    status = "hold"
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
