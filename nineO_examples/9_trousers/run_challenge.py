#!/usr/bin/env python3
"""Challenge 9: Trousers test with Robin DD + CrackTipDecoder enrichment.

Geometry: Sheet L=100mm, W=40mm, B=1mm, pre-crack A=50mm
Loading: Legs separated vertically
Material: PU elastomer (mu=0.52MPa, Lambda=85.77MPa)
Charts: 2 BoxDecoder bulk + 1 CrackTipDecoder at crack front
Material model: Neo-Hookean hyperelastic (compressible formulation)
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
from solvers.fem.robin_schwarz import RobinSchwarzSolver
from solvers.fem.analytic_decoders import BoxDecoder, CrackTipDecoder
from solvers.fracture_criteria import drucker_prager_F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_VTU = os.path.join(ROOT, "runs", "challenge_9")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    # PU elastomer — Neo-Hookean hyperelastic
    mu = 0.52; lam = 85.77
    E = mu * (3*lam + 2*mu) / (lam + mu)
    nu = lam / (2*(lam + mu))
    sigma_ts = 0.3; sigma_hs = 1.0

    L = 100.0; W = 40.0; B = 1.0; A = 50.0
    W_half = W / 2

    # SDF for sheet with pre-crack (crack along y-axis at x=0)
    class SheetSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            # Sheet: [-W/2, W/2] x [0, L] x [-B/2, B/2]
            dx = np.maximum(np.abs(x_np[:, 0]) - W_half, 0)
            dy = np.maximum(-x_np[:, 1], x_np[:, 1] - L)
            dz = np.abs(x_np[:, 2]) - B / 2
            out = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
            ins = np.minimum(np.maximum(np.maximum(np.maximum(np.abs(x_np[:, 0]) - W_half, 0),
                                                     np.maximum(-x_np[:, 1], x_np[:, 1] - L)),
                                         np.abs(x_np[:, 2]) - B/2), 0)
            # Subtract crack slit: x=0, y in [0, A], all z
            crack_dx = np.abs(x_np[:, 0]) - 0.02  # thin slit
            crack_dy = np.maximum(-x_np[:, 1], x_np[:, 1] - A)
            crack = np.sqrt(np.maximum(crack_dx, 0)**2 + np.maximum(crack_dy, 0)**2)
            crack_in = np.minimum(np.maximum(crack_dx, crack_dy), 0)
            crack_sdf = crack + crack_in
            v = np.maximum(out + ins, -crack_sdf)
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = SheetSDF()
    n_cells = 8

    # ── Bulk charts: 2 BoxDecoders (left and right legs) ──
    dec_left = BoxDecoder(center=(-W/4, L/2, 0), half_extents=(W/4+0.1, L/2+0.1, B/2+0.1)).double()
    s_left = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_left,
                                   decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                   device="cpu", dtype=torch.float64)

    dec_right = BoxDecoder(center=(W/4, L/2, 0), half_extents=(W/4+0.1, L/2+0.1, B/2+0.1)).double()
    s_right = ChartVectorFEMSolver(n_cells=n_cells, support_r=1.0, chart_decoder=dec_right,
                                    decoder_kwargs={}, sdf_oracle=sdf, sdf_threshold=-0.01,
                                    device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at crack front ──
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[0, A, 0],
        crack_direction=[0, 1, 0],  # crack propagates along y
        opening_direction=[0, 0, 1],  # Mode III: opens out of plane
        radius=B * 2.0,
        power=2.0,
    ).double()
    s_crack = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [s_left, s_right, s_crack]
    decoders_list = [dec_left, dec_right, crack_dec]
    seeds = [[-W/4, L/2, 0], [W/4, L/2, 0], [0, A, 0]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 3 (2 Box + 1 CrackTip), {total_nodes} nodes")
    print(f"  E={E:.2f} MPa, nu={nu:.4f}")
    # Neo-Hookean hyperelastic model (finite-strain capable)
    K_bulk = lam + 2*mu/3  # bulk modulus
    stress_fn, tangent_fn = solvers[0].make_neo_hookean(mu, K_bulk)
    print(f"  Neo-Hookean: mu={mu}, K={K_bulk:.3f} MPa")

    # ── Loading: leg separation ──
    n_steps = 15; delta_max = 1.0  # mm
    delta_steps = np.linspace(0, delta_max, n_steps + 1)[1:]

    def make_bc(delta):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            y = np_phys[:, 1]; x = np_phys[:, 0]
            tol = L * 0.02
            # Top of each leg (y ~ L): pull in z
            top = y > L - tol
            m[top] = True
            # Left leg: z = -delta, right leg: z = +delta
            u[top & (x < 0), 2] = -delta
            u[top & (x >= 0), 2] = delta
            # Bottom (y ~ 0): fixed where there's no crack
            bot = y < tol
            m[bot] = True
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
            print(f"  *** NUCLEATION step {step}: delta={delta:.4f}, S_zz={szz:.4f}")

        print(f"  Step {step:2d}/{n_steps-1} | delta={delta:.4f} | S_zz={szz:.4f} | F_DP={F_dp:.2f}")

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
                title="Challenge 9: Trousers — von Mises stress",
                filename="challenge_9_von_mises.png",
                warp_factor=2.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
