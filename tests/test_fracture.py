"""V&V tests for Phase 4: Fracture benchmarks.

V&V-4.1: Mode-I edge crack stress intensity factor validation
    - Geometry + SDF correctness
    - LEFM reference solution correctness
    - K_I extraction from analytical displacement field
    - Topology detection of crack feature

V&V-4.2: No-crack false positive (in test_vandv.py)
"""

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Geometry & SDF tests
# ---------------------------------------------------------------------------
class TestPlateGeometry:
    def test_sdf_plate_interior_negative(self):
        """Center of plate should have negative SDF."""
        from benchmarks.fracture.plate_crack_sdf import sdf_plate
        x = np.array([[0.0, 0.0, 0.0]])
        assert sdf_plate(x, W=1.0, H=4.0, T=0.5)[0] < 0

    def test_sdf_plate_exterior_positive(self):
        """Point outside plate should have positive SDF."""
        from benchmarks.fracture.plate_crack_sdf import sdf_plate
        x = np.array([[2.0, 0.0, 0.0]])  # outside W=1
        assert sdf_plate(x, W=1.0, H=4.0, T=0.5)[0] > 0

    def test_sdf_cracked_plate_crack_region_positive(self):
        """Point inside the crack slit should have positive SDF (exterior)."""
        from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
        # Crack runs from x1=-1 to x1=-0.5 (a=0.5, W=1) along x2=0
        x = np.array([[-0.8, 0.0, 0.0]])  # inside crack slit
        val = sdf_cracked_plate(x, a=0.5, W=1.0, H=4.0, T=0.5, delta=0.02)
        assert val[0] > 0, f"Crack interior should be positive (outside domain), got {val[0]}"

    def test_sdf_cracked_plate_solid_region_negative(self):
        """Point in the solid (away from crack) should be negative."""
        from benchmarks.fracture.plate_crack_sdf import sdf_cracked_plate
        x = np.array([[0.0, 1.0, 0.0]])  # well inside plate, away from crack
        val = sdf_cracked_plate(x, a=0.5, W=1.0, H=4.0, T=0.5, delta=0.02)
        assert val[0] < 0, f"Solid region should be negative, got {val[0]}"

    def test_sdf_oracle_interface(self):
        """CrackedPlateSDFOracle should implement .sdf() with torch tensors."""
        import torch
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
        oracle = CrackedPlateSDFOracle(a=0.3, W=1.0)
        x = torch.tensor([[0.0, 0.5, 0.0], [-0.8, 0.0, 0.0]], dtype=torch.float64)
        vals = oracle.sdf(x)
        assert vals.shape == (2,)
        assert vals[0].item() < 0  # inside solid
        assert vals[1].item() > 0  # inside crack

    def test_sdf_grid_shape(self):
        """sdf_grid() should return correct shape."""
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
        oracle = CrackedPlateSDFOracle(a=0.3)
        grid = oracle.sdf_grid(resolution=16)
        assert grid.shape == (16, 16, 16)


# ---------------------------------------------------------------------------
# LEFM reference tests
# ---------------------------------------------------------------------------
class TestLEFMReference:
    def test_geometry_factor_short_crack(self):
        """F(0) should be approximately 1.122 (Tada et al.)."""
        from benchmarks.fracture.lefm_reference import geometry_factor_edge_crack
        F_0 = geometry_factor_edge_crack(0.0)
        assert abs(F_0 - 1.122) < 0.001

    def test_K_I_dimensional_analysis(self):
        """K_I should scale as sigma * sqrt(a)."""
        from benchmarks.fracture.lefm_reference import stress_intensity_factor
        K1 = stress_intensity_factor(sigma_inf=1.0, a=0.25, W=1.0)
        K2 = stress_intensity_factor(sigma_inf=2.0, a=0.25, W=1.0)
        assert abs(K2 / K1 - 2.0) < 1e-10, "K_I should scale linearly with sigma"

        K3 = stress_intensity_factor(sigma_inf=1.0, a=1.0, W=2.0)
        K4 = stress_intensity_factor(sigma_inf=1.0, a=0.25, W=2.0)
        # K ~ sqrt(a), so K3/K4 ~ sqrt(1/0.25) = 2 (approximately, modulo F)
        ratio = K3 / K4
        assert ratio > 1.5, f"K_I should increase with crack length, ratio={ratio}"

    def test_williams_displacement_symmetry(self):
        """u_x should be symmetric, u_y antisymmetric about crack line."""
        from benchmarks.fracture.lefm_reference import williams_displacement
        r = np.array([0.1, 0.1])
        theta = np.array([math.pi / 4, -math.pi / 4])
        u_x, u_y = williams_displacement(r, theta, K_I=1.0, E=200.0, nu=0.3)
        # u_x(θ) = u_x(-θ) (symmetric)
        assert abs(u_x[0] - u_x[1]) < 1e-12
        # u_y(θ) = -u_y(-θ) (antisymmetric)
        assert abs(u_y[0] + u_y[1]) < 1e-12

    def test_reference_table(self):
        """Reference table should produce sensible K_I values."""
        from benchmarks.fracture.lefm_reference import mode_i_reference_table
        table = mode_i_reference_table(W=1.0, sigma_inf=1.0)
        assert len(table["K_I"]) == 6
        # K_I should increase with crack length
        for i in range(1, len(table["K_I"])):
            assert table["K_I"][i] > table["K_I"][i - 1], \
                f"K_I should increase: K_I[{i}]={table['K_I'][i]} <= K_I[{i-1}]={table['K_I'][i-1]}"


# ---------------------------------------------------------------------------
# V&V-4.1: K_I extraction from analytical displacement
# ---------------------------------------------------------------------------
class TestVandV_4_1:
    """Validate K_I extraction by feeding the exact Williams displacement
    into the extraction routine and checking it recovers the correct K_I.

    This tests the extraction method itself (not the FEM solver), ensuring
    the validation infrastructure is correct before using it on numerical
    solutions.
    """

    def test_K_I_recovery_from_exact_williams(self):
        """Extract K_I from exact Williams displacement -> should match K_I_exact."""
        from benchmarks.fracture.lefm_reference import (
            stress_intensity_factor,
            williams_displacement,
            extract_K_I_from_displacement,
        )

        W, a = 1.0, 0.3
        sigma_inf, E, nu = 1.0, 200.0, 0.3
        K_I_exact = stress_intensity_factor(sigma_inf, a, W)

        # Generate evaluation points in an annulus around the crack tip
        n_r, n_theta = 20, 36
        r_vals = np.linspace(0.05, 0.4, n_r)
        theta_vals = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
        rr, tt = np.meshgrid(r_vals, theta_vals)
        r_flat = rr.ravel()
        theta_flat = tt.ravel()

        # Exact Williams displacement
        u_x, u_y = williams_displacement(
            r_flat, theta_flat, K_I_exact, E, nu, plane_strain=True,
        )

        # Extract K_I
        K_I_recovered = extract_K_I_from_displacement(
            u_y, r_flat, theta_flat, E, nu, plane_strain=True,
        )

        rel_error = abs(K_I_recovered - K_I_exact) / K_I_exact
        assert rel_error < 0.05, (
            f"V&V-4.1 FAIL: K_I extraction error {rel_error*100:.1f}% > 5%. "
            f"K_I_exact={K_I_exact:.4f}, K_I_recovered={K_I_recovered:.4f}"
        )

    def test_K_I_recovery_multiple_crack_lengths(self):
        """K_I extraction should work across a/W range [0.1, 0.5]."""
        from benchmarks.fracture.lefm_reference import (
            stress_intensity_factor,
            williams_displacement,
            extract_K_I_from_displacement,
        )

        W, E, nu, sigma_inf = 1.0, 200.0, 0.3, 1.0
        a_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        for a in a_values:
            K_I_exact = stress_intensity_factor(sigma_inf, a, W)

            n_r, n_theta = 15, 24
            r_vals = np.linspace(0.02 * a, 0.5 * a, n_r)
            theta_vals = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
            rr, tt = np.meshgrid(r_vals, theta_vals)
            r_flat, theta_flat = rr.ravel(), tt.ravel()

            u_x, u_y = williams_displacement(
                r_flat, theta_flat, K_I_exact, E, nu, plane_strain=True,
            )
            K_I_recovered = extract_K_I_from_displacement(
                u_y, r_flat, theta_flat, E, nu, plane_strain=True,
            )

            rel_error = abs(K_I_recovered - K_I_exact) / K_I_exact
            assert rel_error < 0.05, (
                f"K_I error {rel_error*100:.1f}% at a/W={a/W:.1f} "
                f"(K_exact={K_I_exact:.4f}, K_recovered={K_I_recovered:.4f})"
            )

    def test_benchmark_harness(self):
        """ModeIEdgeCrackBenchmark should initialize and produce references."""
        from benchmarks.fracture.mode_i_edge_crack import ModeIEdgeCrackBenchmark

        bench = ModeIEdgeCrackBenchmark(a=0.3, W=1.0)
        assert bench.a_over_W == 0.3
        assert bench.K_I_analytical > 0
        assert bench.crack_tip[0] == pytest.approx(-0.7)

        # Reference table
        results = bench.run_validation()
        assert len(results["K_I"]) == 5
        assert all(k > 0 for k in results["K_I"])


# ---------------------------------------------------------------------------
# Topology detection of crack
# ---------------------------------------------------------------------------
class TestCrackTopology:
    def test_full_crack_splits_domain(self):
        """A crack spanning the full width severs the plate into 2 components.

        An edge crack that goes all the way through (a >= 2W) splits the
        domain, producing beta_0 = 2 (two connected components). This is
        the topology change that the monitor should detect.
        """
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
        from atlas.topo.filtration import clip_to_interior

        # Full-width crack: a = 2.0 in plate of half-width W=1.0
        oracle = CrackedPlateSDFOracle(a=2.0, W=1.0, H=4.0, T=0.5, delta=0.15)
        grid = clip_to_interior(oracle.sdf_grid(resolution=48))

        try:
            from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
            diagrams = compute_persistence_diagrams(grid, max_dimension=1)
            betti = betti_numbers_at(diagrams, t=-1e-6)

            assert betti.get(0, 0) >= 2, (
                f"Full crack should split domain (beta_0 >= 2), got betti={betti}"
            )
        except ImportError:
            pytest.skip("GUDHI not available")

    def test_intact_plate_single_component(self):
        """An intact plate should be a single connected component."""
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
        from atlas.topo.filtration import clip_to_interior

        oracle = CrackedPlateSDFOracle(a=0.0, W=1.0, H=4.0, T=0.5)
        grid = clip_to_interior(oracle.sdf_grid(resolution=32))

        try:
            from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
            diagrams = compute_persistence_diagrams(grid, max_dimension=1)
            betti = betti_numbers_at(diagrams, t=-1e-6)

            assert betti.get(0, 0) == 1, (
                f"Intact plate should have beta_0=1, got betti={betti}"
            )
        except ImportError:
            pytest.skip("GUDHI not available")

    def test_topology_monitor_detects_domain_splitting(self):
        """TopologyMonitor should detect when a growing crack severs the domain.

        An edge crack does not create an H1 loop (the domain remains simply
        connected until the crack reaches the other edge). The topological
        change is a new H0 component (domain splits). We monitor dimension 0.
        """
        from atlas.topo.monitor import TopologyMonitor
        from benchmarks.fracture.plate_crack_sdf import CrackedPlateSDFOracle
        from atlas.topo.filtration import clip_to_interior

        monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1),  # watch both H0 and H1
        )

        # Step 0: intact plate
        oracle_intact = CrackedPlateSDFOracle(a=0.0, W=1.0, H=4.0, T=0.5)
        grid_intact = clip_to_interior(oracle_intact.sdf_grid(resolution=32))
        monitor.update(grid_intact, load_step=0)

        # Step 1: full crack severs the domain
        oracle_cracked = CrackedPlateSDFOracle(a=2.0, W=1.0, H=4.0, T=0.5, delta=0.15)
        grid_cracked = clip_to_interior(oracle_cracked.sdf_grid(resolution=32))
        events = monitor.update(grid_cracked, load_step=1)

        # Should detect topology change (new H0 component or other event)
        assert len(events) >= 1, (
            f"Topology monitor should detect domain splitting, "
            f"got {len(events)} events"
        )
