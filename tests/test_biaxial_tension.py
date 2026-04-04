"""V&V tests for biaxial tension benchmark (Nine Circles Challenge Problem 2).

Validates topology-aware atlas against the biaxial tension test from
Kamarei, Zeng, Dolbow & Lopez-Pamies (2026) CMAME 448, 118449.

Tests verify:
1. Circular plate SDF geometry correctness
2. Exact stress-strain solution matches Table 2 material constants
3. Topology: intact plate = 1 component, cracked plate = 2 components
4. TopologyMonitor detects crack nucleation (H0 event)
"""

import math

import numpy as np
import pytest


class TestCircularPlateSDF:
    """Verify the circular plate SDF geometry."""

    def test_center_inside(self):
        from benchmarks.fracture.biaxial_tension import sdf_circular_plate
        x = np.array([[0.0, 0.0, 0.0]])
        assert sdf_circular_plate(x, R=5.0, L=0.25)[0] < 0

    def test_outside_radially(self):
        from benchmarks.fracture.biaxial_tension import sdf_circular_plate
        x = np.array([[6.0, 0.0, 0.0]])
        assert sdf_circular_plate(x, R=5.0, L=0.25)[0] > 0

    def test_outside_axially(self):
        from benchmarks.fracture.biaxial_tension import sdf_circular_plate
        x = np.array([[0.0, 0.0, 0.5]])
        assert sdf_circular_plate(x, R=5.0, L=0.25)[0] > 0

    def test_on_lateral_boundary(self):
        from benchmarks.fracture.biaxial_tension import sdf_circular_plate
        x = np.array([[5.0, 0.0, 0.0]])
        assert abs(sdf_circular_plate(x, R=5.0, L=0.25)[0]) < 1e-10

    def test_cracked_plate_slit_is_exterior(self):
        """Point on the crack slit should be outside the domain."""
        from benchmarks.fracture.biaxial_tension import sdf_cracked_circular_plate
        # Crack at angle=0 extends along x1, slit at y=0
        x = np.array([[0.0, 0.0, 0.0]])  # center of crack slit
        val = sdf_cracked_circular_plate(x, R=5.0, L=0.25, crack_angle=0.0, delta=0.02)
        assert val[0] > 0, f"Crack slit center should be exterior, got {val[0]}"


class TestExactSolution:
    """Verify the exact stress-strain response against Table 2 constants."""

    def test_stress_strain_linear_regime(self):
        """Pre-fracture: S = E*delta/((1-nu)*R), linear elastic."""
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark, GLASS

        bench = BiaxialTensionBenchmark(material=GLASS)
        result = bench.exact_stress_strain()

        # At delta=0, stress=0
        assert result["stress"][0] == 0.0

        # At small delta, should be linear
        E, nu, R = GLASS["E"], GLASS["nu"], 5.0
        delta_small = bench.delta_bs * 0.5
        S_expected = E * delta_small / ((1 - nu) * R)
        # Find closest point
        idx = np.argmin(np.abs(result["delta"] - delta_small))
        S_actual = result["stress"][idx]
        assert abs(S_actual - S_expected) / S_expected < 0.02

    def test_fracture_at_sigma_bs(self):
        """Stress should drop to zero at sigma_bs = 27 MPa."""
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark, GLASS

        bench = BiaxialTensionBenchmark(material=GLASS)
        assert bench.sigma_bs == 27.0

        result = bench.exact_stress_strain()
        # Peak stress should be sigma_bs
        assert abs(max(result["stress"]) - 27.0) < 0.5

    def test_delta_bs_consistent(self):
        """Critical displacement should be consistent with material props."""
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark, GLASS

        bench = BiaxialTensionBenchmark(R=5.0, material=GLASS)
        E, nu = GLASS["E"], GLASS["nu"]
        expected = GLASS["sigma_bs"] * (1 - nu) * 5.0 / E
        assert abs(bench.delta_bs - expected) < 1e-10

    def test_strain_at_fracture(self):
        """Strain at fracture for glass: sigma_bs*(1-nu)/E."""
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark, GLASS

        bench = BiaxialTensionBenchmark(material=GLASS)
        expected_strain = GLASS["sigma_bs"] * (1 - GLASS["nu"]) / GLASS["E"]
        assert abs(bench.strain_at_fracture - expected_strain) < 1e-10


class TestBiaxialTopology:
    """V&V: topology detection of crack nucleation in biaxial tension."""

    def test_intact_plate_single_component(self):
        """Intact circular plate should be one connected component."""
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark

        bench = BiaxialTensionBenchmark(R=5.0, L=0.25)
        topo = bench.topology_check(cracked=False, resolution=32)

        if topo["has_gudhi"]:
            assert topo["betti"].get(0, 0) == 1, (
                f"Intact plate should have beta_0=1, got {topo['betti']}"
            )
            assert not topo["domain_split"]
        else:
            pytest.skip("GUDHI not available")

    def test_cracked_plate_two_components(self):
        """Cracked plate (diametral crack) should split into 2 components.

        Uses a thicker plate (L=2.0) so the grid resolves both the crack
        and the thickness. The physical problem has L=0.25mm but the
        topology is the same regardless of aspect ratio.
        """
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark

        bench = BiaxialTensionBenchmark(R=3.0, L=2.0)
        topo = bench.topology_check(cracked=True, resolution=48)

        if topo["has_gudhi"]:
            assert topo["betti"].get(0, 0) >= 2, (
                f"Cracked plate should have beta_0 >= 2, got {topo['betti']}"
            )
            assert topo["domain_split"]
        else:
            pytest.skip("GUDHI not available")

    def test_monitor_detects_nucleation(self):
        """TopologyMonitor should detect domain splitting when crack nucleates.

        This simulates the physical process: under increasing biaxial
        tension, the plate remains intact until S = sigma_bs, at which
        point a through-thickness crack nucleates. The topology changes
        from beta_0=1 to beta_0>=2.
        """
        from atlas.topo.monitor import TopologyMonitor
        from atlas.topo.filtration import clip_to_interior
        from benchmarks.fracture.biaxial_tension import BiaxialTensionBenchmark

        monitor = TopologyMonitor(
            lifetime_threshold=0.05,
            bottleneck_threshold=0.02,
            monitor_dimensions=(0, 1),
        )

        # Use thicker plate so grid resolves both crack and thickness
        bench = BiaxialTensionBenchmark(R=3.0, L=2.0)

        # Step 0: intact plate (pre-fracture)
        grid_intact = clip_to_interior(bench.sdf_intact())
        monitor.update(grid_intact, load_step=0)

        # Step 1: still intact (loading but below sigma_bs)
        events_1 = monitor.update(grid_intact, load_step=1)
        assert len(events_1) == 0, "No events before fracture"

        # Step 2: crack nucleates (post-fracture)
        grid_cracked = clip_to_interior(bench.sdf_cracked(delta=0.1))
        events_2 = monitor.update(grid_cracked, load_step=2)

        assert len(events_2) >= 1, (
            f"Monitor should detect crack nucleation, "
            f"got {len(events_2)} events"
        )

        # Verify it's an H0 event (domain splitting)
        h0_events = [e for e in events_2 if e.dimension == 0]
        assert len(h0_events) >= 1, (
            f"Should detect H0 event (domain splitting), "
            f"got events: {[(e.event_type, e.dimension) for e in events_2]}"
        )
