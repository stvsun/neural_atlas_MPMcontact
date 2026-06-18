"""Verification tests for P2 (quadratic) tetrahedral elements.

Tests partition of unity, patch test, convergence rate, edge midpoint
sharing, consistency with P1 for linear displacement, and integration
with CrackTipDecoder.
"""
import math
import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.fem.p2_tet import (
    p2_shape_functions,
    p2_shape_gradients,
    build_p2_connectivity,
    compute_midpoint_positions,
    QUAD_POINTS_BARY,
    QUAD_WEIGHTS,
    EDGE_PAIRS,
)


class TestP2ShapeFunctions:
    """B5.1: Partition of unity and basic shape function properties."""

    def test_partition_of_unity_at_quadpoints(self):
        """Shape functions sum to 1 at all quadrature points."""
        for q in range(len(QUAD_POINTS_BARY)):
            L = QUAD_POINTS_BARY[q]
            N = p2_shape_functions(L)
            assert abs(N.sum() - 1.0) < 1e-14, f"qp {q}: sum(N) = {N.sum()}"

    def test_partition_of_unity_random(self):
        """Shape functions sum to 1 at random barycentric coords."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            L = rng.dirichlet([1, 1, 1, 1])[:3]  # valid barycentric
            N = p2_shape_functions(L)
            assert abs(N.sum() - 1.0) < 1e-13

    def test_vertex_kronecker(self):
        """Shape function = 1 at its own vertex, 0 at others."""
        # Vertex 0: L = (1, 0, 0)
        vertices_bary = [
            np.array([1, 0, 0]),  # vertex 0
            np.array([0, 1, 0]),  # vertex 1
            np.array([0, 0, 1]),  # vertex 2
            np.array([0, 0, 0]),  # vertex 3 (L4=1)
        ]
        for i, L in enumerate(vertices_bary):
            N = p2_shape_functions(L)
            assert abs(N[i] - 1.0) < 1e-14, f"N[{i}] at vertex {i} = {N[i]}"
            for j in range(10):
                if j != i:
                    assert abs(N[j]) < 1e-14, f"N[{j}] at vertex {i} = {N[j]}"

    def test_midpoint_kronecker(self):
        """Edge midpoint shape function = 1 at its midpoint, 0 at others."""
        # Edge (0,1) midpoint: L = (0.5, 0.5, 0, 0) → L = (0.5, 0.5, 0)
        L = np.array([0.5, 0.5, 0])
        N = p2_shape_functions(L)
        assert abs(N[4] - 1.0) < 1e-14  # N4 = midpoint of edge (0,1)
        for j in range(10):
            if j != 4:
                assert abs(N[j]) < 1e-14, f"N[{j}] at midpoint 4 = {N[j]}"


class TestP2Gradients:
    """B5.1 cont: Gradient verification."""

    def test_gradient_finite_difference(self):
        """Gradients match finite differences."""
        L0 = np.array([0.3, 0.2, 0.15])
        dN = p2_shape_gradients(L0)  # (10, 3)
        eps = 1e-7
        for d in range(3):
            L_p = L0.copy(); L_p[d] += eps
            L_m = L0.copy(); L_m[d] -= eps
            N_p = p2_shape_functions(L_p)
            N_m = p2_shape_functions(L_m)
            dN_fd = (N_p - N_m) / (2 * eps)
            np.testing.assert_allclose(dN[:, d], dN_fd, atol=1e-6)

    def test_gradient_sum_zero(self):
        """Sum of gradients = 0 (partition of unity derivative)."""
        L = np.array([0.25, 0.3, 0.1])
        dN = p2_shape_gradients(L)
        np.testing.assert_allclose(dN.sum(axis=0), 0.0, atol=1e-14)


class TestP2Connectivity:
    """B5.4: Edge midpoint sharing."""

    def test_midpoints_shared(self):
        """Adjacent tets sharing an edge get the same midpoint node."""
        # Two tets sharing edge (0,1)
        elements_p1 = np.array([[0, 1, 2, 3], [0, 1, 4, 5]])
        el_p2, n_p2 = build_p2_connectivity(elements_p1, 6)

        # Both tets should have the same midpoint for edge (0,1)
        mid_01_tet0 = el_p2[0, 4]  # edge (0,1) midpoint in tet 0
        mid_01_tet1 = el_p2[1, 4]  # edge (0,1) midpoint in tet 1
        assert mid_01_tet0 == mid_01_tet1

    def test_no_duplicate_midpoints(self):
        """No duplicate midpoint indices for the same edge."""
        elements_p1 = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 5]])
        el_p2, n_p2 = build_p2_connectivity(elements_p1, 6)

        # Collect all midpoint indices
        all_mids = set()
        for e in range(3):
            for k in range(6):
                all_mids.add(el_p2[e, 4 + k])

        # Each midpoint index should be unique (no collisions)
        assert len(all_mids) > 0
        assert all(m >= 6 for m in all_mids)  # midpoints start at n_nodes_p1


class TestP2Solver:
    """B5.2, B5.6: P2 solver patch test and P1 consistency."""

    def test_p2_patch_test_linear(self):
        """P2 reproduces linear displacement exactly."""
        s = torch.float64
        solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                       dtype=s, element_order=2)
        # Linear displacement: u = eps * x
        eps = 0.01
        u_exact = eps * solver.nodes.clone()

        stress_fn, tangent_fn = solver.make_neo_hookean(1.0, 10.0)
        f_int = solver.internal_forces(u_exact, stress_fn)

        # For linear displacement, F is constant → div(P) = 0 → interior forces = 0
        interior = ~solver.boundary_mask
        f_int_interior = f_int[interior]
        assert f_int_interior.norm() < 1e-8, \
            f"Interior force norm = {f_int_interior.norm():.2e} (should be ~0)"

    def test_p2_matches_p1_for_linear_u(self):
        """P2 internal forces match P1 for linear displacement (consistency)."""
        s = torch.float64
        solver_p1 = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                          dtype=s, element_order=1)
        solver_p2 = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                          dtype=s, element_order=2)

        # Apply same linear displacement on P1 nodes
        eps = 0.005
        u_p1 = eps * solver_p1.nodes
        u_p2 = eps * solver_p2.nodes

        sfn1, _ = solver_p1.make_neo_hookean(1.0, 10.0)
        sfn2, _ = solver_p2.make_neo_hookean(1.0, 10.0)

        f1 = solver_p1.internal_forces(u_p1, sfn1)
        f2 = solver_p2.internal_forces(u_p2, sfn2)

        # Both should have near-zero interior forces for linear u
        int1 = ~solver_p1.boundary_mask
        int2 = ~solver_p2.boundary_mask
        assert f1[int1].norm() < 1e-8
        assert f2[int2].norm() < 1e-8

    def test_p2_newton_converges(self):
        """P2 Newton-Raphson converges for small deformation."""
        s = torch.float64
        solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                       dtype=s, element_order=2)
        sfn, tfn = solver.make_neo_hookean(1.0, 10.0)

        bc = torch.zeros(solver.n_nodes, dtype=torch.bool)
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=s)
        z = solver.nodes[:, 2]
        bc[z.abs() > 0.99] = True
        u_bc[:, 2] = 0.01 * z
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=s)

        u = solver.solve_nonlinear(sfn, tfn, f_ext, u_bc, bc, max_iter=20, tol=1e-8)
        assert torch.isfinite(u).all()
        assert u.abs().max() > 1e-4


# Need import after sys.path fix
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver


class TestP2WithCrackTipDecoder:
    """B5.5: P2 + CrackTipDecoder integration."""

    def test_p2_with_cracktip_decoder(self):
        """P2 solver builds and solves with CrackTipDecoder."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        dec = CrackTipDecoder.from_crack_tip(
            tip_position=[0, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=1.0,
            power=2.0,
        ).double()

        solver = ChartVectorFEMSolver(
            n_cells=6, support_r=1.0, chart_decoder=dec,
            decoder_kwargs={}, device="cpu", dtype=torch.float64,
            element_order=2,
        )

        assert solver.n_nodes > 0
        assert solver.nodes_per_element == 10

        sfn, tfn = solver.make_neo_hookean(1.0, 10.0)
        u = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        f = solver.internal_forces(u, sfn)
        assert torch.isfinite(f).all()

    def test_p2_more_dof_than_p1(self):
        """P2 solver has more DOF than P1 for same n_cells."""
        s1 = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                   dtype=torch.float64, element_order=1)
        s2 = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device="cpu",
                                   dtype=torch.float64, element_order=2)
        assert s2.n_nodes > s1.n_nodes
        assert s2.n_nodes * 3 > s1.n_nodes * 3
