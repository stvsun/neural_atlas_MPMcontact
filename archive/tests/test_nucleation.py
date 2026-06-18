"""V&V tests for crack nucleation solver and propagation law.

V&V-N1: Drucker-Prager elastic regime (F < 0)
V&V-N2: Drucker-Prager at uniaxial tensile strength (F = 0)
V&V-N3: Drucker-Prager at biaxial tensile strength (F = 0)
V&V-N4: Crack direction from principal stress
V&V-N5: Nucleation at sigma_bs for soda-lime glass
V&V-N6: K_Ic = sqrt(E * G_c / (1 - nu^2))
V&V-N7: MultiCrackSDFOracle correctly subtracts cracks
"""

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# V&V-N1: Elastic regime
# ---------------------------------------------------------------------------
class TestDruckerPragerElastic:
    def test_zero_stress_elastic(self):
        """F(0) < 0: zero stress is inside the strength surface."""
        from solvers.fracture_criteria import drucker_prager_F
        sigma = np.zeros((1, 3, 3))
        F = drucker_prager_F(sigma, sigma_ts=40.0, sigma_hs=27.8)
        assert F[0] < 0, f"F should be negative at zero stress, got {F[0]}"

    def test_small_uniaxial_elastic(self):
        """Small uniaxial tension (sigma < sigma_ts) should be elastic."""
        from solvers.fracture_criteria import drucker_prager_F
        sigma = np.zeros((1, 3, 3))
        sigma[0, 0, 0] = 20.0  # well below sigma_ts=40
        F = drucker_prager_F(sigma, sigma_ts=40.0, sigma_hs=27.8)
        assert F[0] < 0, f"F should be negative at 20 MPa, got {F[0]}"

    def test_small_biaxial_elastic(self):
        """Small biaxial tension should be elastic."""
        from solvers.fracture_criteria import drucker_prager_F
        sigma = np.zeros((1, 3, 3))
        sigma[0, 0, 0] = 10.0
        sigma[0, 1, 1] = 10.0
        F = drucker_prager_F(sigma, sigma_ts=40.0, sigma_hs=27.8)
        assert F[0] < 0


# ---------------------------------------------------------------------------
# V&V-N2: Uniaxial tensile strength
# ---------------------------------------------------------------------------
class TestDruckerPragerUniaxial:
    def test_at_sigma_ts(self):
        """F(diag(sigma_ts, 0, 0)) = 0 at the uniaxial tensile strength."""
        from solvers.fracture_criteria import drucker_prager_F
        sigma_ts, sigma_hs = 40.0, 27.8
        sigma = np.zeros((1, 3, 3))
        sigma[0, 0, 0] = sigma_ts
        F = drucker_prager_F(sigma, sigma_ts, sigma_hs)
        assert abs(F[0]) < 0.5, f"F should be ~0 at sigma_ts, got {F[0]}"

    def test_above_sigma_ts_nucleates(self):
        """F > 0 above sigma_ts."""
        from solvers.fracture_criteria import drucker_prager_F
        sigma = np.zeros((1, 3, 3))
        sigma[0, 0, 0] = 45.0  # above sigma_ts=40
        F = drucker_prager_F(sigma, sigma_ts=40.0, sigma_hs=27.8)
        assert F[0] > 0, f"F should be positive above sigma_ts, got {F[0]}"


# ---------------------------------------------------------------------------
# V&V-N3: Biaxial tensile strength
# ---------------------------------------------------------------------------
class TestDruckerPragerBiaxial:
    def test_at_sigma_bs(self):
        """F(diag(sigma_bs, sigma_bs, 0)) = 0 at biaxial strength."""
        from solvers.fracture_criteria import drucker_prager_F, derived_biaxial_strength
        sigma_ts, sigma_hs = 40.0, 27.8
        sigma_bs = derived_biaxial_strength(sigma_ts, sigma_hs)
        sigma = np.zeros((1, 3, 3))
        sigma[0, 0, 0] = sigma_bs
        sigma[0, 1, 1] = sigma_bs
        F = drucker_prager_F(sigma, sigma_ts, sigma_hs)
        assert abs(F[0]) < 0.5, f"F should be ~0 at sigma_bs={sigma_bs:.1f}, got {F[0]}"

    def test_derived_sigma_bs_matches_table(self):
        """Derived sigma_bs should match Table 2: 27 MPa for glass."""
        from solvers.fracture_criteria import derived_biaxial_strength
        sigma_bs = derived_biaxial_strength(sigma_ts=40.0, sigma_hs=27.8)
        assert abs(sigma_bs - 27.0) < 0.5, f"sigma_bs should be ~27, got {sigma_bs:.1f}"

    def test_derived_sigma_ss_matches_table(self):
        """Derived sigma_ss should match Table 2: 44.4 MPa for glass."""
        from solvers.fracture_criteria import derived_shear_strength
        sigma_ss = derived_shear_strength(sigma_ts=40.0, sigma_hs=27.8)
        assert abs(sigma_ss - 44.4) < 1.0, f"sigma_ss should be ~44.4, got {sigma_ss:.1f}"


# ---------------------------------------------------------------------------
# V&V-N4: Crack direction from principal stress
# ---------------------------------------------------------------------------
class TestCrackDirection:
    def test_uniaxial_x_direction(self):
        """Uniaxial tension in x: crack normal should be [1,0,0]."""
        from solvers.fracture_criteria import crack_normal_from_stress
        sigma = np.diag([40.0, 0.0, 0.0])
        normal = crack_normal_from_stress(sigma)
        assert abs(abs(normal[0]) - 1.0) < 1e-10, f"Normal should be +-[1,0,0], got {normal}"

    def test_uniaxial_y_direction(self):
        """Uniaxial tension in y: crack normal should be [0,1,0]."""
        from solvers.fracture_criteria import crack_normal_from_stress
        sigma = np.diag([0.0, 40.0, 0.0])
        normal = crack_normal_from_stress(sigma)
        assert abs(abs(normal[1]) - 1.0) < 1e-10

    def test_biaxial_degenerate(self):
        """Equi-biaxial: crack normal should be in x-y plane."""
        from solvers.fracture_criteria import crack_normal_from_stress
        sigma = np.diag([27.0, 27.0, 0.0])
        normal = crack_normal_from_stress(sigma)
        # Both x and y eigenvalues are equal, so normal is either [1,0,0] or [0,1,0]
        # (or any combination in the x-y plane)
        assert abs(normal[2]) < 1e-10, "Normal should be in x-y plane"

    def test_max_hoop_pure_mode_I(self):
        """Pure Mode I: propagation angle should be 0."""
        from solvers.fracture_criteria import max_hoop_stress_angle
        theta = max_hoop_stress_angle(K_I=1.0, K_II=0.0)
        assert abs(theta) < 1e-12


# ---------------------------------------------------------------------------
# V&V-N5: Nucleation at sigma_bs for glass
# ---------------------------------------------------------------------------
class TestNucleationDetection:
    def test_nucleation_check_uniform_biaxial(self):
        """Uniform biaxial stress at sigma_bs should trigger nucleation."""
        from solvers.fracture_criteria import check_nucleation_pointwise, derived_biaxial_strength
        n_elements = 100
        sigma_bs = derived_biaxial_strength(40.0, 27.8)  # exact DP value
        stress_field = np.zeros((n_elements, 3, 3))
        stress_field[:, 0, 0] = sigma_bs
        stress_field[:, 1, 1] = sigma_bs
        centroids = np.random.randn(n_elements, 3)

        sites = check_nucleation_pointwise(stress_field, centroids, sigma_ts=40.0, sigma_hs=27.8)
        assert len(sites) >= 1, "Should detect nucleation at sigma_bs"
        assert sites[0]["F_value"] >= 0

    def test_no_nucleation_below_strength(self):
        """Stress below strength should not trigger nucleation."""
        from solvers.fracture_criteria import check_nucleation_pointwise
        n_elements = 50
        stress_field = np.zeros((n_elements, 3, 3))
        stress_field[:, 0, 0] = 10.0  # well below sigma_ts=40 and sigma_bs=27
        stress_field[:, 1, 1] = 10.0
        centroids = np.random.randn(n_elements, 3)

        sites = check_nucleation_pointwise(stress_field, centroids, sigma_ts=40.0, sigma_hs=27.8)
        assert len(sites) == 0, "No nucleation expected below strength"


# ---------------------------------------------------------------------------
# V&V-N6: Griffith fracture toughness
# ---------------------------------------------------------------------------
class TestGriffithKIc:
    def test_K_Ic_glass(self):
        """K_Ic for soda-lime glass: sqrt(E*G_c/(1-nu^2))."""
        from solvers.fracture_criteria import griffith_K_Ic
        E, G_c, nu = 70e3, 0.01, 0.22
        K_Ic = griffith_K_Ic(E, G_c, nu, plane_strain=True)
        expected = math.sqrt(E * G_c / (1 - nu**2))
        assert abs(K_Ic - expected) < 1e-10

    def test_K_Ic_positive(self):
        """K_Ic should always be positive."""
        from solvers.fracture_criteria import griffith_K_Ic
        assert griffith_K_Ic(200.0, 0.01, 0.3) > 0


# ---------------------------------------------------------------------------
# V&V-N7: Multi-crack SDF
# ---------------------------------------------------------------------------
class TestMultiCrackSDF:
    def test_intact_domain(self):
        """No cracks: SDF equals base SDF."""
        from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle

        def base_sdf(x):
            return np.linalg.norm(x, axis=1) - 1.0  # unit sphere

        oracle = MultiCrackSDFOracle(base_sdf, bbox=(np.array([-2, -2, -2]), np.array([2, 2, 2])))
        x = np.array([[0.0, 0.0, 0.0]])
        assert oracle.sdf_np(x)[0] < 0, "Center of sphere should be inside"

    def test_crack_makes_center_exterior(self):
        """A crack through the center should make the origin exterior."""
        from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle

        def base_sdf(x):
            return np.linalg.norm(x, axis=1) - 1.0

        oracle = MultiCrackSDFOracle(
            base_sdf,
            bbox=(np.array([-2, -2, -2]), np.array([2, 2, 2])),
            delta=0.05,
        )
        oracle.add_crack(
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            half_length=1.5,
        )

        # Origin is on the crack slit -> exterior
        x = np.array([[0.0, 0.0, 0.0]])
        assert oracle.sdf_np(x)[0] > 0, "Origin on crack should be exterior"

    def test_multiple_cracks(self):
        """Two perpendicular cracks should both subtract from domain."""
        from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle

        def base_sdf(x):
            # Large box
            return np.max(np.abs(x) - 2.0, axis=1)

        oracle = MultiCrackSDFOracle(
            base_sdf,
            bbox=(np.array([-3, -3, -3]), np.array([3, 3, 3])),
            delta=0.05,
        )
        # Crack 1: along x-axis
        oracle.add_crack(np.array([0, 0, 0]), np.array([0, 1, 0]), 1.0)
        # Crack 2: along y-axis
        oracle.add_crack(np.array([0, 0, 0]), np.array([1, 0, 0]), 1.0)

        assert oracle.n_cracks == 2
        # Origin is on both cracks
        assert oracle.sdf_np(np.array([[0.0, 0.0, 0.0]]))[0] > 0

    def test_advance_crack(self):
        """advance_crack should increase half_length."""
        from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle

        oracle = MultiCrackSDFOracle(
            lambda x: np.max(np.abs(x) - 5.0, axis=1),
            bbox=(np.array([-6, -6, -6]), np.array([6, 6, 6])),
        )
        cid = oracle.add_crack(np.array([0, 0, 0]), np.array([0, 1, 0]), 1.0)
        assert oracle.cracks[cid]["half_length"] == 1.0

        oracle.advance_crack(cid, da=0.5)
        assert oracle.cracks[cid]["half_length"] == 1.5

    def test_sdf_grid_shape(self):
        """sdf_grid should produce correct shape."""
        from benchmarks.fracture.multi_crack_sdf import MultiCrackSDFOracle

        oracle = MultiCrackSDFOracle(
            lambda x: np.linalg.norm(x, axis=1) - 1.0,
            bbox=(np.array([-2, -2, -2]), np.array([2, 2, 2])),
        )
        grid = oracle.sdf_grid(resolution=16)
        assert grid.shape == (16, 16, 16)


# ---------------------------------------------------------------------------
# V&V: Cauchy stress conversion
# ---------------------------------------------------------------------------
class TestCauchyConversion:
    def test_identity_F(self):
        """At F=I, P=0 => sigma=0."""
        from solvers.fracture_criteria import cauchy_from_first_piola
        P = np.zeros((1, 3, 3))
        F = np.eye(3)[None, :, :]
        sigma = cauchy_from_first_piola(P, F)
        assert np.allclose(sigma, 0)

    def test_small_strain_P_equals_sigma(self):
        """For small strains (F ~ I), P ~ sigma."""
        from solvers.fracture_criteria import cauchy_from_first_piola
        P = np.zeros((1, 3, 3))
        P[0, 0, 0] = 40.0
        F = np.eye(3)[None, :, :]
        F[0, 0, 0] = 1.001  # small strain
        sigma = cauchy_from_first_piola(P, F)
        assert abs(sigma[0, 0, 0] - 40.0) < 0.1
