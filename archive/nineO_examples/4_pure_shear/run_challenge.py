#!/usr/bin/env python3
"""Challenge 4: Pure-shear fracture with Robin DD + CrackTipDecoder enrichment.

Geometry: Strip L=50mm, H=5mm, B=0.5mm, edge crack A=10mm
Loading: Grip separation h (clamped top/bottom)
Material: Soda-lime glass (E=70GPa, nu=0.22, G_c=10 N/m)
Charts: 2 BoxDecoder bulk + 1 CrackTipDecoder at crack front
"""
import os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.linear_elastic import make_linear_elastic_small_strain
from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder
from solvers.fracture_criteria import drucker_prager_F, griffith_K_Ic
from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_VTU = os.path.join(ROOT, "runs", "challenge_4")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; Gc = 0.01
    sigma_ts = 40.0; sigma_hs = 27.8
    K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)

    W = 25.0; H = 5.0; B = 0.5; A = 10.0
    # Critical grip separation from Griffith:
    # h_crit = sqrt(G_c * H * 4 * (1-nu^2) / E)
    h_crit = math.sqrt(Gc * H * 4 * (1 - nu**2) / E)

    sdf = CrackedPlateSDFOracle(a=A, W=W, H=H, T=B, delta=0.02)
    n_cells = 10

    # ── Bulk charts: 2 BoxDecoders (left and right of crack) ──
    # Left (contains crack)
    dec_left = BoxDecoder(center=(-W/2, 0, 0), half_extents=(W/2+0.1, H/2+0.1, B/2+0.1)).double()
    s_left = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_left,
                                   decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                   device="cpu", dtype=torch.float64)
    # Right
    dec_right = BoxDecoder(center=(W/2, 0, 0), half_extents=(W/2+0.1, H/2+0.1, B/2+0.1)).double()
    s_right = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_right,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at crack front ──
    tip_x = -W + A
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[tip_x, 0, 0],
        crack_direction=[1, 0, 0],
        opening_direction=[0, 1, 0],
        radius=min(A * 0.3, 2.0),
        power=2.0,
    ).double()
    s_crack = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [s_left, s_right, s_crack]
    decoders_list = [dec_left, dec_right, crack_dec]
    seeds = [[-W/2, 0, 0], [W/2, 0, 0], [tip_x, 0, 0]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 3 (2 Box + 1 CrackTip), {total_nodes} nodes")
    print(f"  K_Ic = {K_Ic:.2f}, h_crit = {h_crit*1e3:.2f} um")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # ── Loading: grip separation ──
    n_steps = 20; h_max = h_crit * 2.0
    h_steps = np.linspace(0, h_max, n_steps + 1)[1:]

    def make_bc(h):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            y = np_phys[:, 1]; tol = H/2 * 0.05
            # Top: u_y = +h/2
            top = y > H/2 - tol; m[top] = True; u[top, 1] = h/2
            # Bottom: u_y = -h/2
            bot = y < -H/2 + tol; m[bot] = True; u[bot, 1] = -h/2
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, h in enumerate(h_steps):
        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders_list,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=2)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(h), max_iters=25, tol=1e-2)

        # Collect stress near crack tip
        syy_all = []
        for ci in range(len(solvers)):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            syy_all.extend(sig[:, 1, 1].tolist())

        syy = np.mean(syy_all) if syy_all else 0
        sigma_test = np.zeros((1, 3, 3)); sigma_test[0, 1, 1] = syy
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(h)
        stress_hist.append(syy if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: h={h*1e3:.2f}um, S_yy={syy:.2f}")

        print(f"  Step {step:2d}/{n_steps-1} | h={h*1e3:.3f}um | S_yy={syy:.2f} | F_DP={F_dp:.1f}")

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
                title="Challenge 4: Pure Shear — von Mises stress",
                filename="challenge_4_von_mises.png",
                warp_factor=500.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
