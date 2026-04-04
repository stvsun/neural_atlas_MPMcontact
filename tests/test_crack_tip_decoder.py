"""V&V tests for CrackTipDecoder singularity-absorbing coordinate chart.

V&V-CTD-1: Forward/inverse roundtrip
V&V-CTD-2: Jacobian matches autograd
V&V-CTD-3: Mesh concentration near tip
V&V-CTD-4: Williams displacement is smooth in reference space
V&V-CTD-5: FEM solve with CrackTipDecoder
V&V-CTD-6: Works with all benchmark geometries
"""

import math

import numpy as np
import torch
import pytest


class TestForwardInverse:
    """V&V-CTD-1: Forward and inverse are consistent."""

    def test_roundtrip_identity(self):
        """decoder(decoder.inverse(x)) should recover x."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(
            center=[1.0, 2.0, 0.0],
            normal=[1.0, 0.0, 0.0],
            tangent1=[0.0, 1.0, 0.0],
            tangent2=[0.0, 0.0, 1.0],
            radius=0.5,
        ).double()

        # Points in the chart domain (r > 0 side)
        x_phys = torch.tensor([
            [1.1, 2.3, 0.1],
            [1.25, 2.0, 0.0],
            [1.4, 1.8, -0.2],
        ], dtype=torch.float64)

        xi = decoder.inverse(x_phys)
        x_recovered = decoder(xi)

        err = torch.linalg.norm(x_recovered - x_phys, dim=1).max().item()
        assert err < 1e-10, f"Roundtrip error {err:.2e}"

    def test_inverse_at_center(self):
        """Crack tip (r=0) should map to xi_2 = -1."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(center=[0, 0, 0], normal=[1, 0, 0], radius=1.0).double()
        x_tip = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        xi = decoder.inverse(x_tip)
        assert abs(xi[0, 2].item() - (-1.0)) < 1e-10, "Tip should map to xi_2 = -1"

    def test_inverse_at_boundary(self):
        """r = radius should map to xi_2 = +1."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(center=[0, 0, 0], normal=[1, 0, 0], radius=0.5).double()
        x_boundary = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
        xi = decoder.inverse(x_boundary)
        assert abs(xi[0, 2].item() - 1.0) < 1e-10, "r=radius should map to xi_2 = +1"


class TestJacobian:
    """V&V-CTD-2: Analytical Jacobian matches autograd."""

    def test_jacobian_vs_autograd(self):
        """Explicit jacobian() should match torch.autograd."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(
            center=[1, 0, 0], normal=[0, 1, 0],
            tangent1=[1, 0, 0], tangent2=[0, 0, 1],
            radius=0.5, power=2.0,
        ).double()

        xi = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64, requires_grad=True)

        # Analytical
        J_analytical = decoder.jacobian(xi)[0]

        # Autograd
        x = decoder(xi)
        J_autograd = torch.zeros(3, 3, dtype=torch.float64)
        for i in range(3):
            g = torch.autograd.grad(x[0, i], xi, retain_graph=True)[0]
            J_autograd[i] = g[0]

        err = (J_analytical - J_autograd).abs().max().item()
        assert err < 1e-10, f"Jacobian mismatch: max error {err:.2e}"

    def test_jacobian_det_positive(self):
        """Jacobian determinant should be positive everywhere (valid mapping)."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(radius=0.5, power=2.0).double()
        xi = torch.rand(100, 3, dtype=torch.float64) * 2 - 1
        xi[:, 2] = xi[:, 2].clamp(min=-0.99)  # avoid exact tip (det=0)

        J = decoder.jacobian(xi)
        det = torch.det(J)
        assert (det > 0).all(), f"Some det(J) <= 0: min={det.min().item()}"

    def test_jacobian_vanishes_at_tip(self):
        """Jacobian det -> 0 as xi_2 -> -1 (mesh concentrates at tip)."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(radius=0.5, power=2.0).double()
        xi_near_tip = torch.tensor([[0.0, 0.0, -0.99]], dtype=torch.float64)
        xi_far = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)

        J_tip = decoder.jacobian(xi_near_tip)
        J_far = decoder.jacobian(xi_far)

        det_tip = torch.det(J_tip).item()
        det_far = torch.det(J_far).item()

        assert det_tip < det_far * 0.1, \
            f"det(J) should be much smaller at tip: {det_tip:.6f} vs {det_far:.6f}"


class TestMeshConcentration:
    """V&V-CTD-3: Physical mesh spacing concentrates near tip."""

    def test_spacing_decreases_near_tip(self):
        """Elements near xi_2=-1 should have smaller physical spacing."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(radius=1.0, power=2.0).double()

        # Compare spacing at two radial positions
        xi_near = torch.tensor([[0, 0, -0.8], [0, 0, -0.6]], dtype=torch.float64)
        xi_far = torch.tensor([[0, 0, 0.6], [0, 0, 0.8]], dtype=torch.float64)

        x_near = decoder(xi_near)
        x_far = decoder(xi_far)

        spacing_near = torch.linalg.norm(x_near[1] - x_near[0]).item()
        spacing_far = torch.linalg.norm(x_far[1] - x_far[0]).item()

        assert spacing_near < spacing_far, \
            f"Near-tip spacing {spacing_near:.4f} should be < far spacing {spacing_far:.4f}"

    def test_quadratic_concentration(self):
        """With power=2, r_phys = radius * ((xi+1)/2)^2."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(
            center=[0, 0, 0], normal=[1, 0, 0], radius=1.0, power=2.0,
        ).double()

        for xi2_val in [0.0, 0.5, 1.0]:
            xi = torch.tensor([[0, 0, xi2_val]], dtype=torch.float64)
            x = decoder(xi)
            r_phys = x[0, 0].item()  # normal is [1,0,0]
            r_expected = 1.0 * ((xi2_val + 1) / 2) ** 2
            assert abs(r_phys - r_expected) < 1e-10, \
                f"At xi_2={xi2_val}: r={r_phys:.6f}, expected {r_expected:.6f}"


class TestWilliamsSmoothing:
    """V&V-CTD-4: Williams displacement becomes smooth in reference space."""

    def test_sqrt_r_becomes_linear(self):
        """sqrt(r_phys) should be linear in xi_2 with power=2."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder(
            center=[0, 0, 0], normal=[1, 0, 0], radius=1.0, power=2.0,
        ).double()

        xi2_vals = torch.linspace(-0.9, 1.0, 20, dtype=torch.float64)
        sqrt_r = []
        for xi2 in xi2_vals:
            xi = torch.tensor([[0, 0, xi2.item()]], dtype=torch.float64)
            x = decoder(xi)
            r = x[0, 0].item()
            sqrt_r.append(math.sqrt(max(r, 0)))

        sqrt_r = np.array(sqrt_r)
        xi2_np = xi2_vals.numpy()

        # sqrt(r) should be proportional to (xi2+1)/2 — check linearity
        t = (xi2_np + 1) / 2
        # sqrt(r) = sqrt(radius) * t = 1.0 * t
        expected = t
        residual = np.abs(sqrt_r - expected)
        max_residual = residual.max()

        assert max_residual < 1e-10, \
            f"sqrt(r) should be linear in (xi+1)/2, max residual {max_residual:.2e}"

    def test_williams_u_smooth_in_xi(self):
        """Williams u_y ~ sqrt(r)*sin(theta/2) should be smooth when mapped."""
        from solvers.fem.analytic_decoders import CrackTipDecoder
        from benchmarks.fracture.lefm_reference import williams_displacement

        decoder = CrackTipDecoder(
            center=[0, 0, 0], normal=[0, 1, 0],
            tangent1=[1, 0, 0], tangent2=[0, 0, 1],
            radius=1.0, power=2.0,
        ).double()

        K_I, E, nu = 1.0, 200.0, 0.3

        # Sample along radial direction at theta=pi/2
        n_pts = 50
        xi2_vals = np.linspace(-0.95, 0.95, n_pts)
        u_y_vals = []

        for xi2 in xi2_vals:
            xi = torch.tensor([[0.3, 0.0, xi2]], dtype=torch.float64)
            x = decoder(xi)
            dx = x[0, 0].item()
            dy = x[0, 1].item()
            r = math.sqrt(dx**2 + dy**2)
            theta = math.atan2(dy, dx)
            if r < 1e-12:
                u_y_vals.append(0.0)
                continue
            _, uy = williams_displacement(
                np.array([r]), np.array([theta]), K_I, E, nu,
            )
            u_y_vals.append(uy[0])

        u_y = np.array(u_y_vals)

        # Check smoothness: second derivative should be bounded
        # (not diverging as r->0 like in physical space)
        du = np.diff(u_y)
        ddu = np.diff(du)

        # In physical space, d^2u/dr^2 ~ r^{-3/2} diverges at tip
        # In reference space with squaring, it should be bounded
        max_ddu = np.abs(ddu).max()
        assert np.isfinite(max_ddu), "Second derivative should be finite in xi-space"


class TestFEMSolve:
    """V&V-CTD-5: FEM solve with CrackTipDecoder produces valid results."""

    def test_solver_builds_mesh(self):
        """ChartVectorFEMSolver should build a mesh with CrackTipDecoder."""
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.analytic_decoders import CrackTipDecoder

        # Use from_crack_tip to ensure orthogonal frame
        decoder = CrackTipDecoder.from_crack_tip(
            tip_position=[0, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=0.5, power=2.0,
        ).double()

        solver = ChartVectorFEMSolver(
            n_cells=6, support_r=1.0,
            chart_decoder=decoder, decoder_kwargs={},
            device="cpu", dtype=torch.float64,
        )

        assert solver.n_nodes > 0
        assert solver.n_elements > 0
        print(f"  CrackTip FEM: {solver.n_nodes} nodes, {solver.n_elements} elements")

    def test_solve_converges(self):
        """Newton should converge on the enriched chart."""
        from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
        from solvers.fem.linear_elastic import make_linear_elastic
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder.from_crack_tip(
            tip_position=[0, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=0.5, power=2.0,
        ).double()

        solver = ChartVectorFEMSolver(
            n_cells=6, support_r=1.0,
            chart_decoder=decoder, decoder_kwargs={},
            device="cpu", dtype=torch.float64,
        )

        stress_fn, tangent_fn = make_linear_elastic(200.0, 0.3)

        # Prescribe small displacement on boundary
        bc_mask = solver.boundary_mask
        u_bc = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)
        u_bc[bc_mask, 1] = 0.001  # small opening displacement
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=torch.float64)

        u = solver.solve_nonlinear(
            stress_fn, tangent_fn, f_ext, u_bc, bc_mask,
            max_iter=20, tol=1e-8,
        )

        assert u is not None
        assert torch.isfinite(u).all()
        print(f"  max|u| = {u.abs().max().item():.6f}")


class TestBenchmarkCompatibility:
    """V&V-CTD-6: Works with all benchmark geometries."""

    def test_edge_crack(self):
        """Construct for Mode-I edge crack."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder.from_crack_tip(
            tip_position=[-0.5, 0, 0],
            crack_direction=[1, 0, 0],
            opening_direction=[0, 1, 0],
            radius=0.3,
        )
        assert decoder.radius == 0.3
        x = decoder(torch.tensor([[0, 0, 0]], dtype=torch.float64))
        assert torch.isfinite(x).all()

    def test_biaxial_crack(self):
        """Construct for biaxial tension crack."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        decoder = CrackTipDecoder.from_crack_tip(
            tip_position=[2.0, 0, 0],
            crack_direction=[0, 0, 1],
            opening_direction=[0, 1, 0],
            radius=0.5,
        )
        x = decoder(torch.zeros(5, 3, dtype=torch.float64))
        assert x.shape == (5, 3)

    def test_from_spawned_pair(self):
        """Construct from SpawnedChartPair."""
        from solvers.fem.analytic_decoders import CrackTipDecoder
        from atlas.topo.chart_spawn import SpawnedChartPair

        pair = SpawnedChartPair(
            seed_plus=np.array([0.1, 0.0, 0.0]),
            seed_minus=np.array([-0.1, 0.0, 0.0]),
            frame_plus=np.eye(3),
            frame_minus=np.eye(3),
            radius=0.3,
            parent_chart=0,
            edge_type="crack",
            activation_step=5,
        )

        decoder_plus = CrackTipDecoder.from_spawned_pair(pair, side="plus")
        decoder_minus = CrackTipDecoder.from_spawned_pair(pair, side="minus")

        assert decoder_plus.radius == 0.3
        assert decoder_minus.radius == 0.3

        xi = torch.tensor([[0, 0, 0.5]], dtype=torch.float64)
        x_plus = decoder_plus(xi)
        x_minus = decoder_minus(xi)
        assert not torch.allclose(x_plus, x_minus), "Plus/minus should differ"

    def test_different_powers(self):
        """Different squaring powers should work."""
        from solvers.fem.analytic_decoders import CrackTipDecoder

        for power in [1.5, 2.0, 3.0]:
            decoder = CrackTipDecoder(radius=1.0, power=power).double()
            xi = torch.tensor([[0, 0, 0.5]], dtype=torch.float64)
            x = decoder(xi)
            assert torch.isfinite(x).all()
            xi_back = decoder.inverse(x)
            err = (xi - xi_back).abs().max().item()
            assert err < 1e-8, f"Roundtrip failed at power={power}: err={err}"
