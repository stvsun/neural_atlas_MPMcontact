#!/usr/bin/env python3
"""Challenge 2: Biaxial tension with Robin DD + CrackTipDecoder enrichment.

Geometry: Circular plate, R=5mm, thickness=0.5mm
Loading: Equi-biaxial, u_x = eps*x, u_y = eps*y
Material: Soda-lime glass (E=70GPa, nu=0.22, sigma_bs=27MPa)
Charts: 1 BoxDecoder bulk + 1 CrackTipDecoder at center
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder
from solvers.fracture_criteria import drucker_prager_F, derived_biaxial_strength
from benchmarks.fracture.biaxial_tension import sdf_circular_plate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_VTU = os.path.join(ROOT, "runs", "challenge_2")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; sigma_ts = 40.0; sigma_hs = 27.8
    sigma_bs = derived_biaxial_strength(sigma_ts, sigma_hs)
    R = 5.0; T = 0.5

    class PlateOracle:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            v = sdf_circular_plate(x_np, R=R, L=T)
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = PlateOracle()
    n_cells = 10

    # ── Bulk chart: BoxDecoder covering full plate ──
    dec_bulk = BoxDecoder(center=(0, 0, 0), half_extents=(R*1.1, R*1.1, T*0.6)).double()
    solver_bulk = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, chart_decoder=dec_bulk, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at plate center (expected nucleation site) ──
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[0, 0, 0],
        crack_direction=[1, 0, 0],  # crack propagates radially
        opening_direction=[0, 1, 0],  # opens in y
        radius=R * 0.5,
        power=2.0,
    ).double()
    crack_solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [solver_bulk, crack_solver]
    decoders_list = [dec_bulk, crack_dec]
    seeds = [[0, 0, 0], [0, 0, 0]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1], [0]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 2 (1 Box + 1 CrackTip), {total_nodes} nodes")
    print(f"  sigma_bs = {sigma_bs:.2f} MPa")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # ── Loading ──
    eps_bs = sigma_bs * (1 - nu) / E
    n_steps = 20; eps_max = eps_bs * 1.2
    eps_steps = np.linspace(0, eps_max, n_steps + 1)[1:]

    def make_bc(eps):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.ones(n, dtype=bool)
            u[:, 0] = eps * np_phys[:, 0]
            u[:, 1] = eps * np_phys[:, 1]
            u[:, 2] = -2 * nu / (1 - nu) * eps * np_phys[:, 2]
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, eps in enumerate(eps_steps):
        S_expected = E * eps / (1 - nu)

        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders_list,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=2)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(eps), max_iters=25, tol=1e-3)

        # Collect stress
        sxx_all = []; syy_all = []
        for ci in range(len(solvers)):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            sxx_all.extend(sig[:, 0, 0].tolist())
            syy_all.extend(sig[:, 1, 1].tolist())

        S_biax = (np.mean(sxx_all) + np.mean(syy_all)) / 2 if sxx_all else 0
        sigma_test = np.zeros((1, 3, 3))
        sigma_test[0, 0, 0] = np.mean(sxx_all) if sxx_all else 0
        sigma_test[0, 1, 1] = np.mean(syy_all) if syy_all else 0
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(eps)
        stress_hist.append(S_biax if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: S={S_biax:.2f}, sigma_bs={sigma_bs:.2f}, F_DP={F_dp:.2f}")

        err = abs(S_biax - S_expected)/S_expected*100 if nuc_step is None and S_expected > 0 else 0
        print(f"  Step {step:2d}/{n_steps-1} | eps={eps:.4e} | S_biax={S_biax:.2f} ({err:.1f}%) | F_DP={F_dp:.1f}")

        final_data = (solvers, u_charts, stress_fn, len(solvers))
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
                title="Challenge 2: Biaxial Tension — von Mises stress",
                filename="challenge_2_von_mises.png",
                warp_factor=100.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
