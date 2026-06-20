#!/usr/bin/env python3
"""Finite-deformation ChartMPM dynamic cross-check of the rough-joint shear (Phase 5).

An INDEPENDENT solver check of the quasi-static chart-FEM result: a deformable block (explicit MPM,
Neo-Hookean) is confined onto the SAME band-limited rough rigid surface and sheared by sliding that
surface underneath it (direct-shear kinematics).  The contact uses the existing MPM channel
(``evaluate_gap`` -> penalty ``compute_contact_force`` -> regularized Coulomb ``compute_friction_force``
-> ``solver.step(contact_force=...)``, all physical-space per-particle forces).  We measure the EMERGENT
apparent friction mu_app = tau/sigma_n and the dilation (block centroid rise) and compare to the FEM.

The rough floor is the SAME surface family as the FEM/decoder (``band_limited_rough_surface``); the MPM
sees it as an analytic SDF phi(x) = z - h(x - u_x, y) (the floor slides at constant rate -> direct shear).

HONEST scope (measured): the explicit MPM independently reproduces the COULOMB FRICTION FLOOR
(mu_app -> base mu) of a flat-bottomed block sliding on the rough floor, converging stably under dynamic
relaxation — a dynamic cross-check of the friction channel.  It does NOT reproduce the dilatant
interlock (emergent dilation + mu_app enhancement) the rough-on-rough FEM shows: a MATED rough-bottomed
MPM block is unstable in explicit penalty dynamics (full-surface conformal contact -> NaN) without an
implicit/mortar contact treatment.  So this phase validates friction but not dilatancy (documented limit),
exactly the kind of issue to surface rather than hide.

Run:  python3 benchmarks/contact/cv_numerical/rock_joint_mpm_xcheck.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.mpm.particles import MaterialPointCloud                    # noqa: E402
from solvers.mpm.chart_mpm_solver import ChartMPMSolver                 # noqa: E402
from solvers.mpm.constitutive import NeoHookeanModel                    # noqa: E402
from solvers.contact.gap import evaluate_gap                            # noqa: E402
from solvers.contact.penalty import compute_contact_force              # noqa: E402
from solvers.contact.friction import compute_friction_force            # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(_ROOT, "runs", "cv7_mpm_xcheck")
DT = torch.float64
torch.set_default_dtype(DT)


def _rough_modes(n_modes=6, k_min=0.6, k_max=2.2, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.uniform(k_min, k_max, n_modes), rng.uniform(k_min, k_max, n_modes),
            rng.uniform(0, 2 * np.pi, n_modes))


class RoughFloorSDF(torch.nn.Module):
    """phi(x) = z - h(x - u_x, y), the SAME band-limited surface as band_limited_rough_surface(amp);
    floor slides at u_x (direct shear).  Solid below the surface; particles above -> gap = phi."""

    def __init__(self, amp=0.10, n_modes=6, k_min=0.6, k_max=2.2, seed=0):
        super().__init__()
        kx, ky, ph = _rough_modes(n_modes, k_min, k_max, seed)
        self.kx = torch.tensor(kx); self.ky = torch.tensor(ky); self.ph = torch.tensor(ph)
        self.amp = amp
        self.u_x = 0.0
        # fixed normalization (max |f| over a dense grid) so the surface matches the FEM's
        gx = np.linspace(-1, 1, 200); X, Y = np.meshgrid(gx, gx)
        f = sum(np.sin(np.pi * (a * X + b * Y) + p) for a, b, p in zip(kx, ky, ph))
        self.fmax = float(np.abs(f).max())

    def height(self, x, y):
        xs = x - self.u_x
        f = sum(torch.sin(np.pi * (self.kx[i] * xs + self.ky[i] * y) + self.ph[i])
                for i in range(len(self.kx)))
        return self.amp * f / self.fmax

    def forward(self, P):
        return P[:, 2] - self.height(P[:, 0], P[:, 1])


def make_block(nx, nz, spacing, z0, density, floor=None):
    """A deformable block of particles spanning [-W,W]^2 in x,y and z in [z0, z0+H].  If ``floor`` is
    given, the block bottom CONFORMS to the rough floor (z += h(x,y)) -> a MATED rough-on-rough joint
    (so shearing produces genuine asperity interlock + dilation, like the FEM), not flat-on-rough."""
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing
    zs = z0 + np.arange(nz) * spacing
    P = np.array([[x, y, z] for z in zs for y in xs for x in xs], float)
    if floor is not None:
        h = floor.height(torch.from_numpy(P[:, 0]), torch.from_numpy(P[:, 1])).numpy()
        P[:, 2] = P[:, 2] + h                                          # conform block bottom to the floor
    pos = torch.tensor(P, dtype=DT)
    n = pos.shape[0]
    vol = torch.full((n,), spacing ** 3, dtype=DT)
    mass = density * vol
    vel = torch.zeros(n, 3, dtype=DT)
    return MaterialPointCloud(pos, vel, vol, mass)


def run_mpm_shear(amp=0.10, mu=0.4, E=2.0e5, nu=0.25, sigma_n=2000.0, v_shear=0.3, shear_total=0.14,
                  nx=9, nz=4, spacing=0.08, eps_n=None, density=1.0e3, verbose=True):
    """Gravity-loaded direct shear (the robust sliding-block pattern + a ROUGH floor): the block rests
    under gravity on the rough asperity tops (its weight -> normal stress sigma_n), its TOP layer is held
    in x (the reaction frame that lets a shear traction build) but FREE in z (so dilation is emergent),
    and the rough floor slides underneath at v_shear.  Emergent apparent friction = the NET contact FORCE
    RATIO mu_app=|F_x|/|F_z| (robust; folds in the dilatant tilted-normal contribution); dilation = block
    centroid rise.  Explicit MPM with dynamic relaxation = the dynamic analog of the quasi-static FEM."""
    floor = RoughFloorSDF(amp=amp)
    W = (nx - 1) / 2 * spacing
    A = (2 * W) ** 2
    z0 = amp + 0.3 * spacing                                            # start just above the asperity tops
    block = make_block(nx, nz, spacing, z0, density)                     # flat-bottomed probe (stable)
    n_part = block.xi.shape[0]
    # gravity for the target normal stress; E is large so sigma_n/E stays small-strain (block rides, no crush)
    g = sigma_n * A / (density * n_part * spacing ** 3)
    solver = ChartMPMSolver(n_cells=32, extent=1.8, gravity=(0.0, 0.0, -g), bc_type="free",
                            constitutive=NeoHookeanModel(E=E, nu=nu))
    if eps_n is None:
        eps_n = 15.0 * E / spacing
    from solvers.contact.penalty import contact_stable_dt
    dt = min(2e-4, 0.5 * contact_stable_dt(eps_n, block.mass.min().item()))
    top = torch.arange((nz - 1) * nx * nx, nz * nx * nx)                 # top layer (held in x), by build index
    n_fall, n_quench, n_shear = 4500, 4500, int(shear_total / (v_shear * dt))

    def contact_forces(v_floor_x=0.0):
        gap, normal = evaluate_gap(block.xi.clone(), floor)
        fN = compute_contact_force(gap, normal, block.current_volume, eps_n)
        v_rel = block.v.clone(); v_rel[:, 0] = v_rel[:, 0] - v_floor_x   # slip RELATIVE to the moving floor
        fT = compute_friction_force(v_rel, normal, fN.norm(dim=1), mu=mu, epsilon_t=1e-3)
        return gap, fN, fT
    hist = {k: [] for k in ("u", "tau", "sigma_n", "mu_app", "dilation", "ke")}
    z_cm0 = None
    for phase, nstep in (("fall", n_fall), ("quench", n_quench), ("shear", n_shear)):
        for it in range(nstep):
            if phase == "shear":
                floor.u_x += v_shear * dt
            gap, fN, fT = contact_forces(v_floor_x=(v_shear if phase == "shear" else 0.0))
            solver.step(block, dt, contact_force=fN + fT)
            block.v[top, 0] = 0.0                                        # hold top in x -> shear reaction frame
            damp = {"fall": 1.0, "quench": 0.80, "shear": 0.98}[phase]   # free-fall, then quench, then quasi-static
            block.v *= damp
            if (phase == "shear" and it % 25 == 0) or (phase == "quench" and it == nstep - 1):
                act = gap < 0
                Fc = (fN[act] + fT[act]).sum(0) if act.any() else torch.zeros(3)
                Fx, Fz = float(Fc[0]), float(Fc[2])
                z_cm = float((block.mass * block.xi[:, 2]).sum() / block.mass.sum())
                if z_cm0 is None:
                    z_cm0 = z_cm
                if phase == "shear":
                    hist["u"].append(float(floor.u_x)); hist["tau"].append(Fx / A)
                    hist["sigma_n"].append(abs(Fz) / A)
                    hist["mu_app"].append(abs(Fx) / max(abs(Fz), 1e-9))   # net contact force ratio
                    hist["dilation"].append(z_cm - z_cm0)
                    hist["ke"].append(0.5 * float((block.mass * (block.v ** 2).sum(1)).sum()))
    for k in hist:
        hist[k] = np.asarray(hist[k])
    m = hist["u"] > 0.5 * hist["u"].max() if hist["u"].size else np.array([False])
    summary = dict(amp=amp, mu=mu, sigma_n_target=sigma_n, dt=float(dt), n_shear=int(n_shear),
                   sigma_n_meas=float(np.median(hist["sigma_n"][m])) if m.any() else float("nan"),
                   mu_app_steady=float(np.median(hist["mu_app"][m])) if m.any() else float("nan"),
                   dilation_final=float(hist["dilation"][-1]) if hist["dilation"].size else float("nan"),
                   mu_app_max=float(np.max(hist["mu_app"])) if hist["mu_app"].size else float("nan"))
    if verbose:
        print(f"  MPM steady mu_app={summary['mu_app_steady']:.3f}  dilation={summary['dilation_final']:+.4f}"
              f"  sigma_n_meas={summary['sigma_n_meas']:.3f}")
    return hist, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=0.10)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--sigma_n", type=float, default=2000.0)   # large load + stiff E -> small-strain riding
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    print(f"=== ChartMPM rough-joint direct shear (amp={args.amp} mu={args.mu}) ===")
    hist, summary = run_mpm_shear(amp=args.amp, mu=args.mu, sigma_n=args.sigma_n)
    json.dump({k: v.tolist() for k, v in hist.items()}, open(os.path.join(RUN, "history.json"), "w"))
    json.dump(summary, open(os.path.join(RUN, "metrics.json"), "w"), indent=2)
    print(f"  saved -> {RUN}")


if __name__ == "__main__":
    main()
