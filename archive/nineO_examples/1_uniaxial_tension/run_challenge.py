#!/usr/bin/env python3
"""Challenge 1: Uniaxial tension with Robin DD + CrackTipDecoder enrichment.

Geometry: Cylindrical rod, L=15mm, R=2mm
Loading: Axial displacement, u_z = eps*z
Material: Soda-lime glass (E=70GPa, nu=0.22, sigma_ts=40MPa)
Charts: 2 BoxDecoder bulk + 1 CrackTipDecoder at midpoint
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder
from solvers.fracture_criteria import drucker_prager_F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_VTU = os.path.join(ROOT, "runs", "challenge_1")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; sigma_ts = 40.0; sigma_hs = 27.8
    L = 15.0; R = 2.0

    class RodSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R; d_z = np.maximum(-x_np[:, 2], x_np[:, 2] - L)
            out = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            v = out + ins
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = RodSDF()
    n_cells = 10

    # ── Bulk charts: 2 BoxDecoders along z ──
    overlap = 0.3; z_span = L / 2 * (1 + overlap)
    solvers = []; decoders = []; seeds = []
    for ci in range(2):
        z_c = L * (ci + 0.5) / 2
        dec = BoxDecoder(center=(0, 0, z_c), half_extents=(R*1.2, R*1.2, z_span/2)).double()
        s = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec,
                                  decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                  device="cpu", dtype=torch.float64)
        solvers.append(s); decoders.append(dec); seeds.append([0, 0, z_c])

    # ── CrackTipDecoder enrichment at rod midpoint ──
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[0, 0, L/2],
        crack_direction=[1, 0, 0],  # crack in xy-plane
        opening_direction=[0, 0, 1],  # opens along z
        radius=R * 0.8,
        power=2.0,
    ).double()
    crack_solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64,
    )
    solvers.append(crack_solver); decoders.append(crack_dec); seeds.append([0, 0, L/2])

    n_charts = len(solvers)
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: {n_charts} (2 Box + 1 CrackTip), {total_nodes} nodes")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # ── Loading ──
    eps_ts = sigma_ts / E
    n_steps = 20; delta_max = eps_ts * L * 1.2
    delta_steps = np.linspace(0, delta_max, n_steps + 1)[1:]

    def make_bc(delta):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.ones(n, dtype=bool)
            eps = delta / L
            u[:, 0] = -nu * eps * np_phys[:, 0]
            u[:, 1] = -nu * eps * np_phys[:, 1]
            u[:, 2] = eps * np_phys[:, 2]
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, delta in enumerate(delta_steps):
        eps = delta / L
        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=2)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(delta), max_iters=25, tol=1e-3)

        # Collect stress
        szz_all = []
        for ci in range(n_charts):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            szz_all.extend(sig[:, 2, 2].tolist())

        szz = np.mean(szz_all) if szz_all else 0
        sigma_test = np.zeros((1, 3, 3)); sigma_test[0, 2, 2] = szz
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(eps)
        stress_hist.append(szz if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: S_zz={szz:.2f}, F_DP={F_dp:.2f}")

        err = abs(szz - E*eps)/(E*eps)*100 if nuc_step is None and E*eps > 0 else 0
        print(f"  Step {step:2d}/{n_steps-1} | eps={eps:.4e} | S_zz={szz:.2f} ({err:.1f}%) | F_DP={F_dp:.1f}")

        final_data = (solvers, u_charts, stress_fn, n_charts)
        if nuc_step is not None and step > nuc_step + 1: break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s")

    # ── PyVista plot ──
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nineO_examples.pyvista_utils import collect_chart_data, plot_von_mises_deformed

    if final_data:
        s_list, u_list, sfn, nc = final_data
        nodes, u_disp, sigma, cids = collect_chart_data(s_list, u_list, sfn, nc)
        if nodes is not None:
            plot_von_mises_deformed(
                nodes, u_disp, sigma, cids,
                title="Challenge 1: Uniaxial Tension — von Mises stress",
                filename="challenge_1_von_mises.png",
                warp_factor=50.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
