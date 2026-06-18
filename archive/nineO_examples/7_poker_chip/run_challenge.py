#!/usr/bin/env python3
"""Challenge 7: Poker-chip with Robin DD + CrackTipDecoder enrichment.

Geometry: Circular disk D=10mm, variable thickness 1.0-1.7mm
Loading: Vertical pull (hydrostatic tension at center)
Material: PU elastomer (mu=0.52MPa, Lambda=85.77MPa)
Charts: 1 BoxDecoder bulk + 1 CrackTipDecoder at center
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
OUT_VTU = os.path.join(ROOT, "runs", "challenge_7")
os.makedirs(OUT_VTU, exist_ok=True)


def run():
    # PU elastomer — Neo-Hookean hyperelastic
    mu = 0.52; lam = 85.77
    E = mu * (3*lam + 2*mu) / (lam + mu)  # ~1.56 MPa
    nu = lam / (2*(lam + mu))  # ~0.4997
    sigma_ts = 0.3; sigma_hs = 1.0

    D = 10.0; R = D/2; L_min = 1.0; L_max = 1.7
    L_avg = (L_min + L_max) / 2

    class DiskSDF:
        def sdf(self, x):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            r = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)
            d_r = r - R
            d_z = np.abs(x_np[:, 2]) - L_avg/2
            out = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
            ins = np.minimum(np.maximum(d_r, d_z), 0)
            v = out + ins
            return torch.tensor(v, dtype=x.dtype, device=x.device) if isinstance(x, torch.Tensor) else v

    sdf = DiskSDF()
    n_cells = 8

    # ── Bulk chart ──
    dec_bulk = BoxDecoder(center=(0, 0, 0), half_extents=(R*1.1, R*1.1, L_avg*0.6)).double()
    s_bulk = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=1.0, chart_decoder=dec_bulk, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    # ── CrackTipDecoder at disk center (hydrostatic tension peak) ──
    crack_dec = CrackTipDecoder.from_crack_tip(
        tip_position=[0, 0, 0],
        crack_direction=[1, 0, 0],
        opening_direction=[0, 0, 1],  # opens vertically
        radius=R * 0.3,
        power=2.0,
    ).double()
    s_crack = ChartVectorFEMSolver(
        n_cells=6, support_r=1.0, chart_decoder=crack_dec, decoder_kwargs={},
        sdf_oracle=sdf, sdf_threshold=-0.01, device="cpu", dtype=torch.float64)

    solvers = [s_bulk, s_crack]
    decoders_list = [dec_bulk, crack_dec]
    seeds = [[0, 0, 0], [0, 0, 0]]
    seeds_t = torch.tensor(seeds, dtype=torch.float64)
    neighbors = [[1], [0]]

    total_nodes = sum(s.n_nodes for s in solvers)
    print(f"  Charts: 2 (1 Box + 1 CrackTip), {total_nodes} nodes")
    print(f"  E={E:.2f} MPa, nu={nu:.4f}")
    # Neo-Hookean hyperelastic model (finite-strain capable)
    K_bulk = lam + 2*mu/3  # bulk modulus
    stress_fn, tangent_fn = solvers[0].make_neo_hookean(mu, K_bulk)
    print(f"  Neo-Hookean: mu={mu}, K={K_bulk:.3f} MPa")

    # ── Loading: vertical pull ──
    n_steps = 15; delta_max = 0.1  # mm
    delta_steps = np.linspace(0, delta_max, n_steps + 1)[1:]

    def make_bc(delta):
        def bc_fn(np_phys):
            n = len(np_phys); u = np.zeros((n, 3)); m = np.zeros(n, dtype=bool)
            z = np_phys[:, 2]; tol = L_avg/2 * 0.1
            # Top: u_z = +delta/2
            top = z > L_avg/2 - tol; m[top] = True; u[top, 2] = delta/2
            # Bottom: u_z = -delta/2
            bot = z < -L_avg/2 + tol; m[bot] = True; u[bot, 2] = -delta/2
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
                title="Challenge 7: Poker-Chip — von Mises stress",
                filename="challenge_7_von_mises.png",
                warp_factor=5.0,
            )

    return {"strain": strain_hist, "stress": stress_hist, "nuc_step": nuc_step}


if __name__ == "__main__":
    run()
