#!/usr/bin/env python3
"""Challenge 8: Double cantilever beam with Robin DD + CrackTipDecoder.

Geometry: Bar L=55mm, H=20mm, B=2.5mm, pre-crack A=25mm
Loading: Pin displacement delta on cantilever arms
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
OUT_VTU = os.path.join(ROOT, "runs", "challenge_8")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    E = 70e3; nu = 0.22; Gc = 0.01
    sigma_ts = 40.0; sigma_hs = 27.8
    K_Ic = griffith_K_Ic(E, Gc, nu, plane_strain=True)

    L = 55.0; H = 20.0; B = 2.5; A = 25.0
    h_arm = H / 2  # arm height
    W_half = L / 2

    # Use cracked plate SDF (crack along left edge at y=0)
    sdf = CrackedPlateSDFOracle(a=A, W=W_half, H=H/2, T=B, delta=0.05)
    n_cells = 10

    # ── Bulk charts: 2 BoxDecoders (left and right halves) ──
    dec_left = BoxDecoder(center=(-L/4, 0, 0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
    s_left = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_left,
                                   decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                   device="cpu", dtype=torch.float64)

    dec_right = BoxDecoder(center=(L/4, 0, 0), half_extents=(L/4+0.1, H/2+0.1, B/2+0.1)).double()
    s_right = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_right,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at crack front ──
    tip_x = -W_half + A  # crack tip location
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[tip_x, 0, 0],
        crack_direction=[1, 0, 0],
        opening_direction=[0, 1, 0],
        radius=2.0,
        power=2.0,
    ).double()
    s_crack = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [s_left, s_right, s_crack]
    decoders_list = [dec_left, dec_right, crack_dec]
    seeds = [[-L/4, 0, 0], [L/4, 0, 0], [tip_x, 0, 0]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 3 (2 Box + 1 CrackTip), {total_nodes} nodes")
    print(f"  K_Ic = {K_Ic:.2f}, A = {A} mm")

    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    # Beam theory critical displacement
    I_arm = B * h_arm**3 / 12
    F_crit = B * math.sqrt(E * Gc * h_arm**3 / (12 * A**2))
    C_A = 2 * A**3 / (3 * E * I_arm)
    delta_crit = F_crit * C_A
    print(f"  Beam theory: F_crit={F_crit:.4f} N, delta_crit={delta_crit*1e3:.2f} um")

    # ── Loading: pin displacement ──
    n_steps = 20; delta_max = delta_crit * 3.0
    delta_steps = np.linspace(0, delta_max, n_steps + 1)[1:]

    # Pin locations: at x = -L/2 + 1.5mm, y = +-H/3
    pin_x = -W_half + 1.5
    pin_y_top = H / 6; pin_y_bot = -H / 6

    def make_bc(delta):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            x = np_phys[:, 0]; y = np_phys[:, 1]
            tol_x = 2.0; tol_y = 2.0
            # Right end: fixed
            right = x > W_half - 1.0; m[right] = True
            # Top pin: u_y = +delta
            top_pin = (np.abs(x - pin_x) < tol_x) & (np.abs(y - pin_y_top) < tol_y) & (y > 0)
            m[top_pin] = True; u[top_pin, 1] = delta
            # Bottom pin: u_y = -delta
            bot_pin = (np.abs(x - pin_x) < tol_x) & (np.abs(y - pin_y_bot) < tol_y) & (y < 0)
            m[bot_pin] = True; u[bot_pin, 1] = -delta
            return u, m
        return bc_fn

    strain_hist = []; stress_hist = []; nuc_step = None
    final_data = None; t0 = time.time()

    for step, delta in enumerate(delta_steps):
        robin = RobinSchwarzSolver(chart_solvers=solvers, seeds=seeds_t, decoders=decoders_list,
                                    neighbors=neighbors, robin_delta=E*0.5, parallel=True, n_workers=2)
        u_charts = robin.solve(stress_fn, tangent_fn, make_bc(delta), max_iters=25, tol=1e-2)

        syy_all = []
        for ci in range(len(solvers)):
            if u_charts[ci] is None: continue
            F = solvers[ci].compute_F(u_charts[ci])
            sig = stress_fn(F).detach().numpy()
            syy_all.extend(sig[:, 1, 1].tolist())

        syy = np.mean(syy_all) if syy_all else 0
        sigma_test = np.zeros((1, 3, 3)); sigma_test[0, 1, 1] = syy
        F_dp = drucker_prager_F(sigma_test, sigma_ts, sigma_hs)[0]

        strain_hist.append(delta)
        stress_hist.append(syy if nuc_step is None else 0.0)

        if F_dp >= 0 and nuc_step is None:
            nuc_step = step
            print(f"  *** NUCLEATION step {step}: delta={delta*1e3:.3f}um, S_yy={syy:.2f}")

        print(f"  Step {step:2d}/{n_steps-1} | delta={delta*1e3:.3f}um | S_yy={syy:.2f} | F_DP={F_dp:.1f}")

        final_data = (solvers, u_charts, stress_fn, len(solvers))
        if nuc_step is not None and step > nuc_step + 1: break

    total_time = time.time() - t0
    print(f"  Total: {total_time:.1f}s")

    from nineO_examples.pyvista_utils import collect_chart_data, plot_von_mises_deformed
    if final_data:
        s_list, u_list, sfn, nc = final_data
        nodes, u_disp, sigma, cids = collect_chart_data(s_list, u_list, sfn, nc)
        if nodes is not None:
            plot_von_mises_deformed(
                nodes, u_disp, sigma, cids,
                title="Challenge 8: DCB — von Mises stress",
                filename="challenge_8_von_mises.png",
                warp_factor=500.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
