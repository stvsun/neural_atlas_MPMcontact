# Theory Manual: Brittle Fracture with Chart-Based FEM

## 1. Overview

This document describes the mathematical framework for modeling brittle fracture using the neural atlas coordinate chart approach. The method replaces traditional mesh-based FEM with a collection of overlapping coordinate charts, each equipped with a learned or analytical decoder that maps a reference cube to physical space. Fracture is handled through topology-aware chart construction: persistent homology detects domain changes, and new charts are spawned automatically at crack locations.

The framework rests on three independent material properties following Kamarei et al. (2026):
1. **Elasticity** — how the material deforms (E, nu)
2. **Strength** — when it breaks under stress (Drucker-Prager surface)
3. **Toughness** — energy cost of crack propagation (G_c)

---

## 2. Coordinate Charts and Atlas

### 2.1 Chart Decoder

A coordinate chart is defined by a decoder mapping from a reference domain to physical space:

```
phi_i : [-1,1]^3 -> R^3
xi -> x = phi_i(xi)
```

The decoder can be:
- **Neural** (ChartDecoder): x = seed + xi_1*t1 + xi_2*t2 + xi_3*n + residual_MLP(xi)
- **Analytical** (TubeSectorDecoder, BoxDecoder, CrackTipDecoder): closed-form mapping

The Jacobian J_i = d(phi_i)/d(xi) provides the metric tensor g = J^T J and volume element |det J| needed for the mapped FEM operators.

### 2.2 Atlas Construction

An atlas is a collection of M overlapping charts {(phi_i, Omega_i)}_{i=1}^M covering the domain Omega. The minimum chart count is bounded by topology:

```
M >= M_min = cat(Omega) + 1
```

where cat(Omega) is the Lusternik-Schnirelmann category, computed from the Betti numbers of the neural SDF via persistent homology (GUDHI).

### 2.3 Partition of Unity

Each chart has a mask function m_i(xi) (MaskNet) that provides a partition of unity for blending overlapping solutions:

```
w_i(x) = exp(m_i(xi)) / sum_j exp(m_j(xi_j))    (softmax)
u(x) = sum_i w_i(x) * u_i(phi_i^{-1}(x))
```

---

## 3. Mapped FEM on Charts

### 3.1 Weak Form

On each chart i, the elastostatic BVP in the reference domain is:

```
integral_{Omega_i} P(F) : grad_xi(v) |det J_i| dxi = integral_{Omega_i} f . v |det J_i| dxi
```

where:
- F = I + grad_x(u) is the deformation gradient in physical space
- P(F) is the first Piola-Kirchhoff stress
- grad_x(v) = grad_xi(v) . J_i^{-1} is the physical gradient via the chain rule
- |det J_i| is the volume scaling from the decoder Jacobian

### 3.2 Mesh and Elements

Each chart's reference domain [-r, r]^3 is discretized as:
- Structured hexahedral grid with n_cells per axis
- Freudenthal decomposition: 6 tetrahedra per hex cell
- Optional SDF filtering: elements with centroid SDF > threshold are removed
- P1 (linear) tetrahedral elements with 4 nodes per element

### 3.3 Shape Function Gradients

The shape function gradient in physical space is computed from the reference gradient via the decoder Jacobian inverse:

```
dN/dx = dN/dxi . J_i^{-1}
```

where dN/dxi is the standard P1 gradient on the reference tet, and J_i^{-1} is evaluated at the element centroid.

### 3.4 Deformation Gradient

The deformation gradient at each element is:

```
F = I + sum_a u_a (x) dN_a/dx = I + sum_a u_a (x) (dN_a/dxi) . J_i^{-1}
```

### 3.5 Newton-Raphson Solver

The nonlinear equilibrium R(u) = f_int(u) - f_ext = 0 is solved by Newton iteration:

```
K_tan(u_k) . du = -R(u_k)
u_{k+1} = u_k + du
```

with backtracking line search for globalization. Dirichlet BCs are enforced by row/column elimination in K_tan.

---

## 4. Constitutive Models

### 4.1 St. Venant-Kirchhoff (Linear Elastic)

For small strains, the first Piola-Kirchhoff stress is:

```
P = F . S,    S = lambda * tr(E) * I + 2 * mu * E
```

where E = (F^T F - I)/2 is the Green-Lagrange strain, and lambda, mu are Lame constants derived from Young's modulus E and Poisson's ratio nu.

The material tangent dP/dF has 9x9 components computed analytically.

### 4.2 Neo-Hookean (Finite Strain)

For large deformations, the Cauchy stress is:

```
sigma = (1/J) [mu(B - I) + lambda ln(J) I]
```

where B = F F^T is the left Cauchy-Green tensor and J = det(F).

### 4.3 J2 Elastoplasticity

For ductile fracture, multiplicative decomposition F = F_e F_p with radial return mapping in logarithmic strain space. Isotropic hardening with equivalent plastic strain alpha.

---

## 5. Schwarz Domain Decomposition

### 5.1 Multiplicative Schwarz Iteration

Charts are coupled through alternating Dirichlet boundary exchange:

```
For each Schwarz iteration k:
    For each color group (non-overlapping charts):
        For each chart i in group:
            1. Physical BCs: u = g on physical boundary (SDF surface)
            2. Artificial BCs: u = interpolated from neighbors on chart boundary
            3. Solve: K_i u_i = f_i with combined BCs
```

### 5.2 Boundary Interpolation

At artificial boundaries, displacement is interpolated from neighboring charts using the decoder inverse:

```
u_i(x_boundary) = sum_j w_j * u_j(phi_j^{-1}(x_boundary))
```

where phi_j^{-1} is computed via:
- Closed-form inverse (analytical decoders: TubeSectorDecoder.inverse())
- Newton iteration (neural decoders: invert_decoder())

Interpolation within the neighbor chart uses barycentric coordinates in the containing tet element (P1 interpolation).

### 5.3 Under-Relaxation

For stability, the displacement update is damped:

```
u_i^{k+1} = omega * u_i^{new} + (1 - omega) * u_i^{k}
```

with omega in [0.3, 0.7]. Without relaxation, the Schwarz iteration may oscillate between chart contributions.

### 5.4 Parallel Execution

Within each color group, charts are independent and solved concurrently using ThreadPoolExecutor. Measured speedup: 2.1x with 4 threads on 4 charts, 3.6x with 8 threads on 8 charts.

---

## 6. Fracture Nucleation

### 6.1 Drucker-Prager Strength Surface

Fracture nucleates when the Cauchy stress at any element violates the Drucker-Prager criterion (Kamarei et al. 2026, Eq. 2):

```
F(sigma) = sqrt(J2) + alpha * I1 - k = 0
```

where:
- I1 = sig1 + sig2 + sig3 (first stress invariant, principal stresses)
- J2 = (1/3)(sig1^2 + sig2^2 + sig3^2 - sig1*sig2 - sig1*sig3 - sig2*sig3)
- alpha = sigma_ts / (sqrt(3) * (3*sigma_hs - sigma_ts))
- k = sqrt(3) * sigma_hs * sigma_ts / (3*sigma_hs - sigma_ts)

The material constants sigma_ts (uniaxial tensile strength) and sigma_hs (hydrostatic tensile strength) are independent of the elastic moduli and toughness.

Derived strengths:
- Biaxial: sigma_bs = 3*sigma_hs*sigma_ts / (3*sigma_hs + sigma_ts)
- Shear: sigma_ss = sqrt(3)*sigma_hs*sigma_ts / (3*sigma_hs - sigma_ts)

### 6.2 Crack Direction

The crack nucleates perpendicular to the maximum principal stress direction:

```
crack_normal = eigenvector of max(eigenvalues(sigma))
```

For equi-biaxial tension, all in-plane directions are equally favorable (degenerate eigenvalue). For torsion, the maximum principal stress is at 45 degrees to the twist axis.

### 6.3 Nucleation Protocol

At each load step:
1. Solve the BVP on the current atlas
2. Compute Cauchy stress at all element centroids
3. Evaluate F(sigma) at each element
4. If max(F) >= 0: nucleation detected at the most critical element
5. Determine crack center (element centroid) and normal (principal direction)
6. Create crack in the SDF oracle (MultiCrackSDFOracle)

---

## 7. Crack Propagation

### 7.1 Griffith Criterion

An existing crack propagates when the stress intensity factor reaches the fracture toughness:

```
K_I >= K_Ic = sqrt(E * G_c / (1 - nu^2))    (plane strain)
```

### 7.2 K_I Extraction

The stress intensity factor is extracted from the Williams asymptotic displacement field near the crack tip:

```
u_y = K_I/(2mu) * sqrt(r/(2pi)) * sin(theta/2) * [kappa + 1 - 2cos^2(theta/2)]
```

K_I is estimated by evaluating this at multiple points in an annular region around the tip and taking the median (robust to noise).

### 7.3 Propagation Direction

For Mode I (opening): straight ahead (theta = 0).
For mixed mode: maximum tangential stress angle:

```
theta_c = 2 * arctan((K_I - sqrt(K_I^2 + 8*K_II^2)) / (4*K_II))
```

### 7.4 SDF Update

The crack geometry is represented as a union of thin planar slits subtracted from the base domain via CSG:

```
SDF_cracked = max(SDF_base, -SDF_crack)
```

Each crack is parameterized by (center, normal, half_length, delta) and can be advanced incrementally.

---

## 8. Topology-Aware Chart Spawning

### 8.1 Persistent Homology

The SDF is sampled on a cubical grid and processed by GUDHI's CubicalComplex to compute persistence diagrams for H0 (connected components), H1 (loops/tunnels), and H2 (voids/cavities).

Betti numbers beta_k count the significant topological features at a filtration parameter t (typically t = -epsilon, just inside the domain boundary).

### 8.2 Topology Monitoring

During simulation, the TopologyMonitor compares persistence diagrams between load steps using the bottleneck distance. When d_B exceeds a threshold (0.02), a TopologyEvent is fired:

```
event = TopologyEvent(
    load_step, event_type='new_H0',
    dimension=0, birth_value, lifetime,
    bottleneck_change
)
```

### 8.3 Chart Spawning

The ChartSpawner converts TopologyEvents into SpawnedChartPairs:
1. Localize the feature in physical space (SDF gradient at the birth value)
2. Place seeds on opposite sides of the feature normal
3. Build orthonormal frames via Gram-Schmidt
4. Identify the nearest existing chart for warm-start initialization

### 8.4 Dynamic Atlas Update

New charts are registered with the Schwarz solver via add_charts():
1. Create warm-started decoders (copy parent weights)
2. Build new FEM solvers for each chart
3. Rebuild the overlap neighbor graph
4. Re-color for multiplicative Schwarz scheduling
5. Invalidate BC cache (forces recompute)

---

## 9. CrackTipDecoder: Singularity Absorption

### 9.1 Motivation

The Williams stress field has a 1/sqrt(r) singularity at the crack tip. Standard P1 elements cannot capture this without extreme mesh refinement. The CrackTipDecoder absorbs the singularity into the coordinate mapping.

### 9.2 Radial Power-Law Mapping

The decoder maps reference coordinates to a crack-tip region with radial squaring:

```
r_phys = radius * ((xi_2 + 1) / 2) ^ power
```

With power = 2:
- Physical r is quadratic in xi: r ~ xi^2
- Physical sqrt(r) is linear in xi: sqrt(r) ~ xi
- Williams displacement u ~ sqrt(r) becomes smooth (linear) in reference space
- P1 elements capture the singularity exactly without enrichment

### 9.3 Jacobian Structure

The Jacobian is block-diagonal:

```
J = [ in_plane_scale * t1 | in_plane_scale * t2 | dr/dxi_2 * normal ]
```

where dr/dxi_2 = radius * power * ((xi_2+1)/2)^(power-1) / 2. This vanishes at the crack tip (xi_2 = -1), concentrating the mesh.

### 9.4 Properties

- det(J) -> 0 as xi_2 -> -1: mesh concentrates near the tip
- The Jacobian inverse exists everywhere except at the exact tip (floored at t = 0.01)
- Forward/inverse are closed-form and exact
- Compatible with all benchmark geometries via from_crack_tip() constructor
- Integrates with chart spawning via from_spawned_pair() constructor

### 9.5 Connection to XFEM

The CrackTipDecoder achieves the same goal as XFEM enrichment but through coordinate mapping rather than basis function augmentation:

| Aspect | XFEM | CrackTipDecoder |
|--------|------|-----------------|
| Enrichment location | Nodes near tip | Entire crack-tip chart |
| Extra DOFs | Yes (enrichment DOFs) | No |
| Element formulation | Modified (partition of unity) | Standard P1 |
| Singularity capture | Explicit enrichment functions | Absorbed in coordinates |
| Coupling to bulk | Blending elements | Schwarz boundary exchange |
| Dynamic spawning | Not standard | Automatic via TopologyMonitor |

---

## 10. Validation Summary

### 10.1 Benchmark Results

| Benchmark | Quantity | Result | Reference | Error |
|-----------|----------|--------|-----------|-------|
| Biaxial tension (glass) | sigma_bs | 28.09 MPa | 27.03 MPa (DP) | 3.9% |
| Torsion (glass, single chart) | tau | 33.57 MPa | 33.57 MPa (analytical) | 0.01% |
| Uniaxial strain (identity) | sigma_xx | 0.6781 MPa | 0.6731 MPa | 0.75% |
| K_I extraction (Williams) | K_I | exact | exact | 0.0% |
| GUDHI overhead | fraction | 1.1% | < 5% budget | PASS |

### 10.2 Kamarei et al. (2026) Challenge Problems

| Challenge | Type | Our model | AT1 model |
|-----------|------|-----------|-----------|
| Uniaxial tension | Nucleation (strength) | Correct at sigma_ts | Depends on epsilon |
| Biaxial tension | Nucleation (strength) | Correct at sigma_bs | Wrong (58% off at fitted eps) |
| Torsion | Nucleation (strength) | Correct at sigma_ss, 45-degree crack | Wrong stress AND wrong angle |
| DCB | Propagation (Griffith) | Correct F-delta and a(delta) | Fair agreement |

---

## 11. References

- Sun (2026). Neural atlas for mapped coordinate chart FEM/MPM. (This codebase)
- Kamarei, Zeng, Dolbow, Lopez-Pamies (2026). Nine circles of elastic brittle fracture. CMAME 448, 118449.
- Kumar, Francfort, Lopez-Pamies (2018). Fracture and healing of elastomers. JMPS 112, 523-551.
- Tada, Paris, Irwin (2000). The Stress Analysis of Cracks Handbook.

---

## Appendix A: Test Suite

Total: 159 tests passing (as of latest commit).

| Category | Tests | Key verification |
|----------|-------|-----------------|
| Topology pipeline | 24 | Betti numbers, bottleneck stability, M_min |
| Chart certification | 8 | certify_sdf, M_min enforcement |
| Dynamic monitoring | 10 | Zero false positives, topology detection, chart spawning |
| Fracture geometry | 16 | SDF, K_I extraction, LEFM reference |
| Nucleation solver | 23 | Drucker-Prager, crack direction, multi-crack SDF |
| Crack propagation | 15 | Driver growth/hold, topology-monitored, K_I curve |
| Real FEM solve | 8 | Identity, decoder, Schwarz, incremental, full pipeline |
| CrackTipDecoder | 16 | Roundtrip, Jacobian, mesh concentration, Williams smoothing |
| Performance | 12 | GUDHI overhead, pipeline smoke, constitutive models |
| Biaxial tension | 12 | Geometry, exact solution, topology detection |
| V&V false positive | 7 | No spurious events under deformation/noise |
