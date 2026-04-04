#!/usr/bin/env python3
"""Export chart FEM results as VTU files.

Generates VTU files for each task showing the actual FEM mesh,
displacement field, and stress computed by ChartVectorFEMSolver.

Usage:
    python postprocessing/export_chart_fem_vtu.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocessing.utils import write_vtu_points

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "chart_fem_vtu")
os.makedirs(OUT_DIR, exist_ok=True)


def write_fem_result_vtu(path, solver, u, stress_fn, extra_fields=None):
    """Write a FEM result to VTU with mesh, displacement, and stress."""
    nodes = solver.nodes.detach().cpu().numpy()
    nodes_phys = solver.nodes_phys.detach().cpu().numpy() if solver.nodes_phys is not None else nodes
    u_np = u.detach().cpu().numpy()

    F = solver.compute_F(u)
    P = stress_fn(F)
    P_np = P.detach().cpu().numpy()
    F_np = F.detach().cpu().numpy()

    # Compute element-average stress and strain, then scatter to nodes
    n_nodes = solver.n_nodes
    n_elements = solver.n_elements
    elements = solver.elements.detach().cpu().numpy()

    # Cauchy stress
    J = np.linalg.det(F_np)
    sigma = np.einsum("eij,ekj->eik", P_np, F_np) / J[:, None, None]

    # Green-Lagrange strain
    I3 = np.eye(3)
    C = np.einsum("eji,ejk->eik", F_np, F_np)
    E_GL = 0.5 * (C - I3[None, :, :])

    # Average element values to nodes
    sigma_nodal = np.zeros((n_nodes, 3, 3))
    strain_nodal = np.zeros((n_nodes, 3, 3))
    count = np.zeros(n_nodes)

    for e in range(n_elements):
        for ni in elements[e]:
            sigma_nodal[ni] += sigma[e]
            strain_nodal[ni] += E_GL[e]
            count[ni] += 1

    count = np.maximum(count, 1)
    sigma_nodal /= count[:, None, None]
    strain_nodal /= count[:, None, None]

    # Von Mises
    s = sigma_nodal
    von_mises = np.sqrt(0.5 * ((s[:, 0, 0] - s[:, 1, 1])**2
                                + (s[:, 1, 1] - s[:, 2, 2])**2
                                + (s[:, 2, 2] - s[:, 0, 0])**2
                                + 6 * (s[:, 0, 1]**2 + s[:, 1, 2]**2 + s[:, 0, 2]**2)))

    fields = {
        "displacement": u_np,
        "sigma_xx": sigma_nodal[:, 0, 0],
        "sigma_yy": sigma_nodal[:, 1, 1],
        "sigma_zz": sigma_nodal[:, 2, 2],
        "sigma_xy": sigma_nodal[:, 0, 1],
        "von_Mises": von_mises,
        "strain_xx": strain_nodal[:, 0, 0],
        "strain_yy": strain_nodal[:, 1, 1],
        "strain_zz": strain_nodal[:, 2, 2],
    }

    if extra_fields:
        fields.update(extra_fields)

    # Use physical-space nodes for the VTU
    deformed = nodes_phys + u_np
    write_vtu_points(path, nodes_phys, fields)

    # Also write deformed configuration
    path_def = path.replace(".vtu", "_deformed.vtu")
    write_vtu_points(path_def, deformed, fields)

    return fields


def export_task1():
    """Task 1: uniaxial strain on identity cube."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic

    E, nu, delta = 100.0, 0.3, 0.01
    solver = ChartVectorFEMSolver(n_cells=8, support_r=1.0, chart_decoder=None,
                                   device="cpu", dtype=torch.float64)
    stress_fn, tangent_fn = make_linear_elastic(E, nu)

    bc_mask = solver.boundary_mask
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
    x_np = solver.nodes[:, 0].numpy()
    u_bc[:, 0] = torch.tensor((x_np + 1.0) / 2.0 * delta, dtype=torch.float64)
    f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

    u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-10)

    fields = write_fem_result_vtu(
        os.path.join(OUT_DIR, "task1_identity.vtu"),
        solver, u, stress_fn,
    )
    print(f"  Task 1: {solver.n_nodes} nodes, sigma_xx={fields['sigma_xx'].mean():.4f}")


def export_task2():
    """Task 2: with ChartDecoder."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic
    from common.models import ChartDecoder

    E, nu, delta = 100.0, 0.3, 0.005
    decoder = ChartDecoder(width=16, depth=2).double()
    decoder.raw_scale = torch.nn.Parameter(torch.tensor(-5.0, dtype=torch.float64))

    seed = torch.zeros(3, dtype=torch.float64)
    t1 = torch.tensor([1, 0, 0], dtype=torch.float64)
    t2 = torch.tensor([0, 1, 0], dtype=torch.float64)
    n = torch.tensor([0, 0, 1], dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64)

    solver = ChartVectorFEMSolver(
        n_cells=8, support_r=1.0, chart_decoder=decoder,
        decoder_kwargs={"seed": seed, "t1": t1, "t2": t2, "n": n, "chart_scale": scale},
        device="cpu", dtype=torch.float64,
    )
    stress_fn, tangent_fn = make_linear_elastic(E, nu)

    bc_mask = solver.boundary_mask
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
    x_np = solver.nodes[:, 0].numpy()
    u_bc[:, 0] = torch.tensor((x_np + 1.0) / 2.0 * delta, dtype=torch.float64)
    f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

    u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=20, tol=1e-8)

    # Add decoder distortion as extra field
    distortion = np.linalg.norm(
        solver.nodes_phys.numpy() - solver.nodes.numpy(), axis=1)
    detJ = np.zeros(solver.n_nodes)  # scatter element detJ to nodes
    elem_detJ = solver.geom_detJ.numpy()
    elements = solver.elements.numpy()
    count = np.zeros(solver.n_nodes)
    for e in range(solver.n_elements):
        for ni in elements[e]:
            detJ[ni] += elem_detJ[e]
            count[ni] += 1
    detJ /= np.maximum(count, 1)

    fields = write_fem_result_vtu(
        os.path.join(OUT_DIR, "task2_decoder.vtu"),
        solver, u, stress_fn,
        extra_fields={
            "decoder_distortion": distortion,
            "jacobian_det": detJ,
        },
    )
    print(f"  Task 2: {solver.n_nodes} nodes, detJ=[{detJ.min():.3f},{detJ.max():.3f}], "
          f"sigma_xx={fields['sigma_xx'].mean():.4f}")


def export_task4_incremental():
    """Task 4: incremental loading — one VTU per load step."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    from solvers.fem.linear_elastic import make_linear_elastic
    from solvers.fracture_criteria import drucker_prager_F, cauchy_from_first_piola

    E, nu = 100.0, 0.3
    sigma_ts, sigma_hs = 2.0, 1.5
    stress_fn, tangent_fn = make_linear_elastic(E, nu)

    solver = ChartVectorFEMSolver(n_cells=6, support_r=1.0, chart_decoder=None,
                                   device="cpu", dtype=torch.float64)
    nodes = solver.nodes

    vtu_files = []
    for step in range(10):
        delta = 0.1 * (step + 1) / 10
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc[:, 0] = torch.tensor((nodes[:, 0].numpy() + 1.0) / 2.0 * delta, dtype=torch.float64)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, bc_mask, max_iter=10, tol=1e-10)

        F = solver.compute_F(u)
        P = stress_fn(F)
        sigma = cauchy_from_first_piola(P.detach().numpy(), F.detach().numpy())
        F_dp = drucker_prager_F(sigma, sigma_ts, sigma_hs)

        # Scatter F_dp to nodes
        F_dp_nodal = np.zeros(solver.n_nodes)
        count = np.zeros(solver.n_nodes)
        elements = solver.elements.numpy()
        for e in range(solver.n_elements):
            for ni in elements[e]:
                F_dp_nodal[ni] += F_dp[e]
                count[ni] += 1
        F_dp_nodal /= np.maximum(count, 1)

        fname = f"task4_step{step:02d}.vtu"
        fields = write_fem_result_vtu(
            os.path.join(OUT_DIR, fname),
            solver, u, stress_fn,
            extra_fields={
                "F_drucker_prager": F_dp_nodal,
                "load_step": np.full(solver.n_nodes, float(step)),
                "applied_delta": np.full(solver.n_nodes, delta),
            },
        )
        vtu_files.append(fname)

        nucleated = F_dp.max() >= 0
        print(f"  Task 4 step {step}: delta={delta:.4f}, sigma_xx={fields['sigma_xx'].mean():.4f}, "
              f"F_DP_max={F_dp.max():.4f}{' *** NUCLEATION' if nucleated else ''}")
        if nucleated:
            break

    # Write PVD
    pvd_path = os.path.join(OUT_DIR, "task4_incremental.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for i, fname in enumerate(vtu_files):
            f.write(f'    <DataSet timestep="{float(i)}" file="{fname}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    print(f"  PVD: {pvd_path}")


def main():
    print(f"Exporting chart FEM results to {OUT_DIR}/\n")

    export_task1()
    export_task2()
    export_task4_incremental()

    print(f"\nAll VTU files saved to {OUT_DIR}/")
    print("Open in ParaView:")
    print(f"  paraview {OUT_DIR}/task1_identity.vtu")
    print(f"  paraview {OUT_DIR}/task2_decoder.vtu")
    print(f"  paraview {OUT_DIR}/task4_incremental.pvd")


if __name__ == "__main__":
    main()
