#!/usr/bin/env python3
"""Challenge 6: Indentation with Robin DD + CrackTipDecoder enrichment.

Geometry: Cylindrical block R=25mm, L=25mm, flat punch R_punch=1mm
Loading: Indenter displacement delta at top surface
Material: Soda-lime glass (E=70GPa, nu=0.22)
Charts: 2 BoxDecoder bulk + 1 CrackTipDecoder at ring crack location
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
OUT_VTU = os.path.join(ROOT, "runs", "challenge_6")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; sigma_ts = 40.0; sigma_hs = 27.8
    R_block = 25.0; L_block = 25.0; R_punch = 1.0

    class BlockSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R_block
            d_z = np.maximum(-x_np[:, 2], x_np[:, 2] - L_block)
            out = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            v = out + ins
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = BlockSDF()
    n_cells = 8

    # ── Bulk charts: 2 BoxDecoders (near punch + far field) ──
    dec_top = BoxDecoder(center=(0, 0, L_block*0.8), half_extents=(R_punch*5, R_punch*5, L_block*0.25)).double()
    s_top = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_top,
                                  decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                  device="cpu", dtype=torch.float64)

    dec_bot = BoxDecoder(center=(0, 0, L_block*0.3), half_extents=(R_block*0.5, R_block*0.5, L_block*0.35)).double()
    s_bot = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_bot,
                                  decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                  device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at expected ring crack location ──
    # Ring crack nucleates at r ~ R_punch on top surface
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[R_punch * 1.2, 0, L_block],
        crack_direction=[0, 1, 0],  # along ring circumference
        opening_direction=[0, 0, -1],  # opens downward
        radius=R_punch * 0.5,
        power=2.0,
    ).double()
    s_crack = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [s_top, s_bot, s_crack]
    decoders_list = [dec_top, dec_bot, crack_dec]
    seeds = [[0, 0, L_block*0.8], [0, 0, L_block*0.3], [R_punch*1.2, 0, L_block]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 3 (2 Box + 1 CrackTip), {total_nodes} nodes")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # ── Loading: indenter displacement ──
    n_steps = 15; delta_max = 0.01  # mm
    delta_steps = np.linspace(0, delta_max, n_steps + 1)[1:]

    def make_bc(delta):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            z = np_phys[:, 2]; r = np.sqrt(np_phys[:, 0]**2 + np_phys[:, 1]**2)
            tol_z = L_block * 0.05
            # Bottom: fixed
            bot = z < tol_z; m[bot] = True
            # Top under punch: u_z = -delta
            top_punch = (z > L_block - tol_z) & (r < R_punch * 1.1)
            m[top_punch] = True; u[top_punch, 2] = -delta
            # Top outside punch: free (traction-free)
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, delta in enumerate(delta_steps):
        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders_list,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=2)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(delta), max_iters=25, tol=1e-2)

        szz_all = []
        for ci in range(len(solvers)):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            szz_all.extend(sig[:, 2, 2].tolist())

        szz = np.mean(szz_all) if szz_all else 0
        sigma_test = np.zeros((1, 3, 3)); sigma_test[0, 2, 2] = szz
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(delta)
        stress_hist.append(szz if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: delta={delta*1e3:.3f}um, S_zz={szz:.2f}")

        print(f"  Step {step:2d}/{n_steps-1} | delta={delta*1e3:.3f}um | S_zz={szz:.2f} | F_DP={F_dp:.1f}")

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
                title="Challenge 6: Indentation — von Mises stress",
                filename="challenge_6_von_mises.png",
                warp_factor=1000.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
