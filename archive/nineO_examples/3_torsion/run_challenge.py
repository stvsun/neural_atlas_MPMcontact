#!/usr/bin/env python3
"""Challenge 3: Torsion with Robin DD + CrackTipDecoder enrichment.

Geometry: Thin-walled tube, L=5mm, r_mid=2.925mm, t_wall=0.15mm
Loading: Twist angle alpha at z=L, fixed at z=0
Material: Soda-lime glass (E=70GPa, nu=0.22, sigma_ss=44.4MPa)
Charts: 4 TubeSectorDecoder bulk + 1 CrackTipDecoder at 45-deg
"""
import os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import TubeSectorDecoder, CrackTipDecoder
from solvers.fracture_criteria import drucker_prager_F, derived_shear_strength

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_VTU = os.path.join(ROOT, "runs", "challenge_3")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; mu = E / (2*(1+nu))
    sigma_ts = 40.0; sigma_hs = 27.8
    sigma_ss = derived_shear_strength(sigma_ts, sigma_hs)
    r_mid = 2.925; t_wall = 0.15; L = 5.0

    n_cells = 8; n_circ = 4
    theta_span = math.pi / 1.5  # 120 deg

    # ── Bulk charts: 4 TubeSectorDecoders ──
    solvers = []; decoders_list = []; seeds = []
    for ti in range(n_circ):
        theta_c = ti * 2 * math.pi / n_circ
        dec = TubeSectorDecoder(theta_center=theta_c, theta_span=theta_span,
                                 r_mid=r_mid, t_half=t_wall/2,
                                 z_center=L/2, L_half=L/2).double()
        s = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec,
                                  decoder_kwargs={}, device="cpu", dtype=torch.float64)
        solvers.append(s); decoders_list.append(dec)
        seeds.append([r_mid * math.cos(theta_c), r_mid * math.sin(theta_c), L/2])

    # ── CrackTipDecoder at expected 45-degree crack location ──
    # Crack under torsion is at 45 deg to axis, on outer surface
    crack_z = L / 2
    crack_theta = 0.0  # arbitrary azimuthal position
    tip_pos = [r_mid * math.cos(crack_theta), r_mid * math.sin(crack_theta), crack_z]
    # 45-degree crack: crack direction is 45 deg between z and theta
    crack_dir = [
        -math.sin(crack_theta) / math.sqrt(2),
        math.cos(crack_theta) / math.sqrt(2),
        1.0 / math.sqrt(2),
    ]
    # Opening direction perpendicular to tube surface at that point
    open_dir = [math.cos(crack_theta), math.sin(crack_theta), 0]

    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=tip_pos, crack_direction=crack_dir,
        opening_direction=open_dir, radius=t_wall * 2, power=2.0,
    ).double()
    crack_solver = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        device="cpu", dtype=torch.float64)
    solvers.append(crack_solver); decoders_list.append(crack_dec); seeds.append(tip_pos)

    n_charts = len(solvers)
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    # Neighbors: each bulk chart connects to adjacent bulk + crack tip
    neighbors = [[(i+1)%n_circ, (i-1)%n_circ, n_circ] for i in range(n_circ)]
    neighbors.append(list(range(n_circ)))  # crack tip connects to all bulk

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: {n_charts} (4 TubeSector + 1 CrackTip), {total_nodes} nodes")
    print(f"  sigma_ss = {sigma_ss:.1f} MPa")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # ── Loading ──
    tau_max = sigma_ss * 1.2
    alpha_max = tau_max * L / (mu * r_mid)
    n_steps = 20
    alpha_steps = np.linspace(0, alpha_max, n_steps + 1)[1:]

    def make_bc(alpha):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            z = np_phys[:, 2]; tol = L * 0.05
            rho = np.sqrt(np_phys[:, 0]**2 + np_phys[:, 1]**2)
            theta = np.arctan2(np_phys[:, 1], np_phys[:, 0])
            # z=0: fixed
            z0 = z < tol; m[z0] = True
            # z=L: twist
            zL = z > L - tol; m[zL] = True
            u_th = alpha * rho[zL]
            u[zL, 0] = -u_th * np.sin(theta[zL])
            u[zL, 1] = u_th * np.cos(theta[zL])
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, alpha in enumerate(alpha_steps):
        tau_expected = mu * alpha * r_mid / L

        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders_list,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=4)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(alpha), max_iters=25, tol=1e-2)

        # Collect shear stress
        tau_all = []
        for ci in range(n_charts):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            tau = np.sqrt(sig[:, 0, 2]**2 + sig[:, 1, 2]**2)
            tau_all.extend(tau.tolist())

        tau_avg = np.mean(tau_all) if tau_all else 0
        # Build shear stress tensor for DP check
        sigma_test = np.zeros((1, 3, 3))
        sigma_test[0, 0, 1] = tau_avg; sigma_test[0, 1, 0] = tau_avg
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(alpha)
        stress_hist.append(tau_avg if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: tau={tau_avg:.2f}, sigma_ss={sigma_ss:.1f}")

        err = abs(tau_avg - tau_expected)/tau_expected*100 if nuc_step is None and tau_expected > 0 else 0
        print(f"  Step {step:2d}/{n_steps-1} | alpha={alpha:.4e} | tau={tau_avg:.2f} ({err:.1f}%) | F_DP={F_dp:.1f}")

        final_data = (solvers, u_charts, stress_fn, n_charts)
        if nuc_step is not None and step > nuc_step + 1: break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s")

    # ── PyVista plot ──
    from nineO_examples.pyvista_utils import collect_chart_data, plot_von_mises_deformed
    if final_data:
        s_list, u_list, sfn, nc = final_data
        nodes, u_disp, sigma, cids = collect_chart_data(s_list, u_list, sfn, nc)
        if nodes is not None:
            plot_von_mises_deformed(
                nodes, u_disp, sigma, cids,
                title="Challenge 3: Torsion — von Mises stress",
                filename="challenge_3_von_mises.png",
                warp_factor=20.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
