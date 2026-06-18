# Neural Atlas MPM Codebase Exploration Report

## Executive Summary

This is a **production-ready topology-aware neural atlas framework** for solving brittle fracture problems on complex 3D geometries. The codebase achieves **98.9% (890/900)** on the Nine Circles Elastic Brittle Fracture benchmark suite (Kamarei et al. 2026, CMAME 448), with 8 of 9 challenges scoring perfect 100/100.

**Key differentiators**: topology-driven chart spawning via persistent homology + coordinate map-based crack tip enrichment (singularity absorption) + Robin parallel domain decomposition.

---

## Part 1: TOPOLOGY PIPELINE (atlas/topo/)

### 1.1 Overview
The topology module detects and drives automatic chart adaptation during fracture simulation through persistent homology and topological change monitoring.

**Files**:
- `filtration.py` (159 lines)
- `persistence.py` (236 lines)
- `ls_category.py` (232 lines)
- `monitor.py` (280 lines)
- `chart_spawn.py` (276 lines)
- `certify.py` (221 lines)

**Status**: ✅ COMPLETE AND VALIDATED

### 1.2 Filtration (atlas/topo/filtration.py)

**Purpose**: Sublevel-set filtration of neural SDF for persistent homology input.

**Key Functions**:
```python
sample_sdf_on_grid(sdf_net, bbox_min, bbox_max, resolution=32, batch_size=4096)
  → (grid_vals, coords): Grid evaluation of SDF on regular cubical grid

clip_to_interior(grid_vals, t_max=0.0, t_min=None)
  → clipped_array: Removes exterior features (above boundary t=0)

filtration_value_range(grid_vals) → (min, max): Range of SDF values
```

**Analytic SDFs for verification**:
- `sdf_ball()`: sphere (contractible, beta_0=1, beta_1=0, beta_2=0, cat=0, M_min=1)
- `sdf_solid_torus()`: homotopy ~ S^1 (beta_1=1, cat=1, M_min=2)
- `sdf_thick_spherical_shell()`: homotopy ~ S^2 (beta_2=1, cat=1, M_min=2)

**Key Design**:
- Batch evaluation avoids OOM on large grids
- Clipping at t=0 (domain boundary) removes noise from exterior
- Verification contract: ball must have exactly 1 H_0 pair with death=inf, no H_1/H_2

### 1.3 Persistence (atlas/topo/persistence.py)

**Purpose**: Compute birth-death pairs from SDF sublevel-set filtration via GUDHI.

**Key Functions**:
```python
compute_persistence_diagrams(grid_vals, max_dimension=2) 
  → {k: [(birth, death), ...] for k in 0..max_dimension}
    Uses GUDHI's CubicalComplex (exact, O(N³ log N))

filter_by_lifetime(diagrams, threshold, relative=True, filtration_range=None)
  → filtered_diagrams: Retains pairs with |death - birth| > threshold
    
betti_numbers_at(diagrams, t=-1e-6) → {k: count}
  Evaluates beta_k(t) = |{(b,d) : b <= t < d}|

bottleneck_distance(dgm_a, dgm_b) → float
  L∞ matching distance (certified via gudhi.bottleneck_distance if available)
```

**Contracts**:
- Ball: 1 essential H_0 pair (death=inf), empty H_1/H_2
- Solid torus: 1 H_0 (inf), 1+ H_1 pair(s) (significant lifetime), empty H_2
- Spherical shell: 1 H_0 (inf), empty H_1, 1 H_2 pair

**Noise Filtering**: threshold = 0.05 * |SDF_max - SDF_min| (empirically robust)

### 1.4 Lusternik-Schnirelmann Category (atlas/topo/ls_category.py)

**Purpose**: Compute minimum chart count M_min from topology via cup-length lower bound.

**Key Functions**:
```python
cup_length_lower_bound(betti) → int
  Lower bound: sum(1 for k >= 1 if beta_k > 0)
  Tight for all orientable surfaces

compute_m_min(betti) → int
  M_min = cat(Omega) + 1
  Hard topological floor from nerve theorem

quality_driven_m_practical(betti, kappa_target=10.0, alpha_overlap=0.20, ...)
  → (M_min, M_practical, explanation)
  Balances topology (M >= M_min) and numerics (condition number, overlap)

certify_atlas(M_actual, betti, quality_metrics=None) → report dict
  Checks: M_actual >= M_min, coverage, fold-over, transition error
```

**Topology Bounds Table**:
| Domain | cat | M_min | Notes |
|--------|-----|-------|-------|
| Ball/convex | 0 | 1 | simply connected |
| Solid torus | 1 | 2 | homotopy ~ S¹ |
| Spherical shell | 1 | 2 | homotopy ~ S² |
| Torus surface T² | 2 | 3 | beta_1=2 |

**Practical Gap**: M_practical >> M_min due to Jacobian conditioning. Example: Solid torus has M_min=2 but paper uses M_practical=8 due to numerical requirements.

### 1.5 Topology Monitor (atlas/topo/monitor.py)

**Purpose**: Dynamic detection of topology changes (cracks, voids) during simulation.

**Data Structures**:
```python
@dataclass TopologyEvent:
  load_step: int
  event_type: str  # 'new_H1' | 'new_H2' | 'feature_died'
  dimension: int
  birth_value: float
  lifetime: float
  localization: Optional[ndarray]  # physical coords of feature
  bottleneck_change: float

class TopologyMonitor:
  lifetime_threshold: float (default 0.05)
  bottleneck_threshold: float (default 0.02)
  monitor_dimensions: tuple (default (1,) for cracks only)
  relative_threshold: bool
```

**Key Methods**:
```python
monitor.update(grid_vals, load_step) → List[TopologyEvent]
  Detects new H1/H2 pairs, fires TopologyEvent when bottleneck distance changes
  
monitor.current_betti → dict of current Betti numbers
monitor.event_history → immutable list of all detected events
monitor.summary() → formatted event log
```

**Verification Contract**:
- On fixed domain: monitor.update() returns [] every step after first
- On propagating crack: exactly one new_H1 event fires within 1-2 steps of crack reaching topological significance

**Implementation Details**:
- Greedy matching of persistence pairs between consecutive steps (L∞ distance)
- Projects unmatched pairs to diagonal to detect truly new features
- Handles inf-death pairs (essential features) separately

### 1.6 Chart Spawner (atlas/topo/chart_spawn.py)

**Purpose**: Convert TopologyEvents into concrete chart pair spawn requests with persistence-driven mesh refinement.

**Key Structures**:
```python
@dataclass SpawnedChartPair:
  seed_plus, seed_minus: [3] seed locations on +/- sides
  frame_plus, frame_minus: [3,3] orthonormal frames
  radius: float
  parent_chart: int (for warm-start)
  edge_type: str  # 'crack' or 'interface'
  activation_step: int
  recommended_n_cells: int  # persistence-driven
  recommended_element_order: int  # 1 or 2

class ChartSpawner:
  default_radius: float (default 0.3)
  sdf_net: optional neural SDF for localization
```

**Key Methods**:
```python
spawner.spawn_from_event(event, existing_seeds, existing_frames, grid_vals, bbox_min, bbox_max)
  → SpawnedChartPair with all parameters for atlas.add_charts()
```

**Spawn Protocol**:
1. **Localize feature**: Find grid cells near event.birth_value, compute SDF gradient as normal
2. **Seed placement**: seed_± = center ± 0.5*r*normal
3. **Frame construction**: Gram-Schmidt from normal to build orthonormal basis
4. **Warm-start**: Find nearest existing chart (smallest transition error)
5. **Adaptive mesh**: n_cells ∈ [base, 3*base], driven by normalized lifetime
6. **Element order**: P2 recommended for H1/H2 features

**Persistence-Driven Refinement**:
```python
normalized_lifetime = event.lifetime / max(sdf_range, 1e-10)
adaptive_n_cells = int(base_n_cells * (1.0 + 2.0 * min(normalized_lifetime, 1.0)))
# Range: [base_n_cells, 3*base_n_cells]
# H1 features (cracks): min 12 cells
# H2 features (voids): min 10 cells
```

---

## Part 2: SOLVERS - FEM (solvers/fem/)

### 2.1 Overview

**Files**:
- `chart_fem_solver.py` (669 lines): P1 scalar FEM (Poisson, diffusion)
- `chart_vector_fem.py` (1292 lines): P1/P2 vector FEM (elasticity, Neo-Hookean)
- `schwarz_fem.py` (637 lines): Multiplicative Schwarz multi-chart solver
- `schwarz_vector_fem.py` (269 lines): Vector elasticity Schwarz variant
- `robin_schwarz.py` (299 lines): Parallel Robin domain decomposition
- `linear_elastic.py` (162 lines): Small-strain linear elastic model
- `analytic_decoders.py` (400 lines): CrackTipDecoder, BoxDecoder, TubeSector
- `neural_crack_decoder.py` (348 lines): Neural network crack-tip enrichment
- `p2_tet.py` (342 lines): P2 quadratic tet shape functions & 4-point Gauss quadrature
- `k_extraction.py` (211 lines): K_I extraction via Williams displacement fitting
- `crack_propagation.py` (204 lines): Quasi-static fracture driver
- `parallel_solver.py` (250 lines): Multi-threaded chart solving

**Status**: ✅ COMPLETE, VALIDATED ON 9 FRACTURE BENCHMARKS (890/900)

### 2.2 ChartFEMSolver - Scalar Poisson (chart_fem_solver.py)

**Purpose**: P1 tetrahedral FEM on mapped coordinate chart reference domain.

**Equation Solved** (mapped Poisson):
```
-div_ξ [ A_i(ξ) ∇_ξ u ] = j_i(ξ) f(φ_i(ξ))
where A_i = j_i J_i^{-1} J_i^{-T},  j_i = |det J_i|
```

**Key Components**:
```python
class ChartFEMSolver:
  chart_id: int
  decoder: torch.nn.Module  # φ: ξ → x
  seed, t1, t2, n_vec: torch.Tensor  # chart frame [3]
  support_r: float  # chart radius
  n_cells: int  # grid resolution
  sdf_oracle: optional SDF for mesh filtering
  
  nodes: ndarray (n_nodes, 3)  # ξ coordinates
  elements: ndarray (n_els, 4)  # tet connectivity (P1)
  
  K: sparse CSR matrix (stiffness)
  F: dense vector (RHS)
  u: solution vector
```

**Mesh Construction**:
- Structured hex grid [-r, r]³ with n_cells³ cells (r = support_r * mesh_extent)
- Freudenthal decomposition: 6 tets per hex
- SDF-filtered: elements with centroid SDF > threshold removed
- Boundary classification: physical (domain boundary) vs artificial (chart boundary)

**Assembly**:
```python
compute_diffusion_tensors()
  Precomputes A_elem [n_els, 3, 3] and det_elem [n_els] via decoder Jacobian

assemble_system(forcing_fn, bc_fn)
  Vectorized einsum assembly: K_ij = ∫_e A ∇φ_i · ∇φ_j dx
  Row/column elimination for Dirichlet BCs
```

**Solver**:
```python
solve(forcing_fn, bc_fn, method='spsolve')
  Uses scipy.sparse.linalg.spsolve (LU decomposition)
```

### 2.3 ChartVectorFEMSolver - Elasticity (chart_vector_fem.py)

**Purpose**: P1/P2 vector FEM for 3D nonlinear elastostatics on mapped chart.

**Equation** (nonlinear elasticity):
```
Find u: -div(σ(∇u)) = f  with σ = stress_fn(F) where F = ∇_ξ u
```

**Key Classes**:
```python
class ChartVectorFEMSolver:
  n_cells, support_r: mesh parameters
  chart_decoder: optional φ: ξ → x
  decoder_kwargs: dict (seed, t1, t2, n, chart_scale)
  sdf_oracle, sdf_threshold: mesh filtering
  
  nodes: torch.Tensor (n_nodes, 3)  [reference coords]
  nodes_phys: torch.Tensor (n_nodes, 3) [physical coords, if mapped]
  elements: ndarray (n_els, 4 or 10)  # P1 or P2 tet
  
  element_volume: [n_els]
  boundary_mask: [n_nodes]  # physical boundary
```

**P2 Support**:
- 10-node tets (4 corners + 6 edge midpoints)
- 4-point Gauss quadrature (O(h²) convergence vs O(h) for P1)
- Handled by `p2_tet.py` (shape functions, integration weights)

**Newton-Raphson Nonlinear Solver**:
```python
solve(stress_fn, tangent_fn, bc_fn, f_ext_fn, max_iter=10, tol=1e-8)
  Iterates: [K(u^n)] δu = -r(u^n)  where r = internal - external forces
  Assembles tangent K via autodiff (PyTorch) or analytical tangent
```

**Force/Tangent Assembly**:
```python
assemble_forces(u, stress_fn, volume, nodes_phys, elements)
  For each tet: compute F = ∇_ξ u, σ = stress_fn(F)
  f_i^ext = ∫_e σ : ∇_ξ φ_i dV (weighted by element volume and Jacobian)

assemble_tangent(u, tangent_fn, volume, nodes_phys, elements)
  K_ij = ∫_e (dσ/dF : ∇φ_i) : ∇φ_j dV
```

**Material Models Supported**:
- Linear elastic (small strain)
- Neo-Hookean hyperelastic (finite strain)
- J2 elastoplastic (with internal state)

### 2.4 CrackTipDecoder (analytic_decoders.py)

**Purpose**: Singularity-absorbing coordinate map for Mode I cracks.

**Design**: Absorbs √r singularity into coordinate map via power law.

```python
class CrackTipDecoder:
  center: [3]  # crack tip location
  normal: [3]  # crack opening direction
  tangent1, tangent2: [3]  # in-plane directions
  radius: float  # support radius
  power: float  # exponent (default 2.0)
  in_plane_scale: float  # stretching factor
  
  def forward(xi: Tensor) → x: Tensor
    Maps ξ ∈ [-1,1]³ to physical space with r_phys ~ |ξ_in_plane|^power
    
  def jacobian(xi: Tensor) → J: Tensor
    Analytical Jacobian ∂x/∂ξ
```

**Key Property**:
```
r_phys = radius * |ξ_in_plane|^power
=> sqrt(r_phys) ~ |ξ_in_plane|^(power/2)
   With power=2: sqrt(r) ~ |ξ|, so Williams displacement is linear in ξ
   => P1 elements capture singularity exactly; P2 gives O(h²)
```

**NeuralCrackDecoder** (neural_crack_decoder.py):
- Analytical base + learned residual MLP
- Trains on Williams asymptotic field to further smooth
- Architecture: 3 → width → ... → 3 (MLP)
- Initialization: zero residual, output weights init to zero
- Pre-trained checkpoint saved to `runs/pretrained_decoders/crack_tip_decoder_default.pt`

### 2.5 Robin Parallel Domain Decomposition (robin_schwarz.py)

**Purpose**: Fast convergence for multi-chart coupled problems via Robin interface conditions.

**Reference**: Du (2002) "Optimization Based Nonoverlapping Domain Decomposition Algorithms", SIAM J. Numer. Anal. 39(3).

**Robin Transmission Condition**:
```
∂u_i/∂n + δ u_i = ∂u_neighbor/∂n + δ λ  on interface
```

**Parallel Algorithm** (Du 2002, Eqs. 5.9-5.12):
```
Step 0 (Update): 
  λ^{n+1} = (u_1^n + u_2^n) / 2
  g^{n+1} = g^n + (u_2^n - u_1^n) / 2 * δ

Step 1 (Solve in parallel):
  For each chart i: solve with Robin BCs
    ∂u_i/∂n + δ u_i = δ λ + (-1)^{i+1} g  on interface
```

**Convergence**:
- For affine fields (uniaxial tension): converges in **2 iterations** (exact)
- For general nonlinear problems: converges in **2–5 iterations** vs **10–25** for multiplicative Schwarz
- Robin parameter δ ~ E/h is empirically optimal (paper uses δ = E * 0.5)

**Implementation**:
```python
class RobinSchwarzSolver:
  chart_solvers: List[ChartVectorFEMSolver]
  robin_delta: float
  parallel: bool  # ThreadPoolExecutor for concurrent chart solves
  
  solve(stress_fn, tangent_fn, phys_bc_fn, max_iters=30, tol=1e-4)
    → List[u_charts]: per-chart displacement fields
```

### 2.6 K_I Extraction (k_extraction.py)

**Purpose**: Extract stress intensity factor from displacement field via Williams fitting.

**Method**:
1. Sample displacement at points near crack tip
2. Fit to Williams asymptotic solution: u(r,θ) = K_I/(2μ) √(r/(2π)) f(θ)
3. Extract K_I from fitting coefficient

**Key Functions**:
```python
extract_K_from_displacement(r, theta, u_r, u_theta, K_I_ref=1.0, E=1e5, nu=0.3)
  Least-squares fit of {u_r, u_theta} to Williams field
  
extract_K_from_charts(chart_solvers, u_charts, crack_tip, crack_dir, opening_dir, E, nu, plane_strain)
  Orchestrates multi-chart K_I extraction
```

### 2.7 P2 Tetrahedral Elements (p2_tet.py)

**Purpose**: 10-node quadratic tet shape functions and Gauss quadrature.

**Shape Functions**:
- 4 corner basis: Lagrange P1
- 6 edge midpoint basis: edge-based quadratic
- All satisfy φ_i(x_j) = δ_ij

**Quadrature**:
- 4-point Gauss rule (O(h³) error)
- Weights and points precomputed

---

## Part 3: SOLVERS - MPM (solvers/mpm/)

### 3.1 Overview

**Files**:
- `chart_mpm_solver.py` (151 lines): single-chart MPM time stepper
- `particles.py` (130 lines): MaterialPointCloud
- `grid.py` (104 lines): ChartGrid
- `transfers.py` (199 lines): P2G/G2P transfers
- `constitutive.py` (196 lines): Neo-Hookean, elastoplastic models
- `schwarz_mpm.py` (570 lines): multi-chart Schwarz MPM solver

**Status**: ✅ IMPLEMENTED (not heavily used in fracture benchmarks, but available)

### 3.2 ChartMPMSolver - Single Chart

**Purpose**: Explicit MPM time stepper on mapped coordinate chart.

**Algorithm** (standard MPM cycle):
```
1. P2G: scatter particle data to grid
2. Grid: update momentum with forces, compute velocity, apply BCs
3. G2P: gather grid velocities, update particle velocity and F
4. Stress update: evaluate constitutive model at new F
```

**Classes**:
```python
class MaterialPointCloud:
  xi, v, F, stress: particle state in reference coords
  mass, volume: material properties

class ChartGrid:
  momentum, force, velocity: nodal quantities
  
class ChartMPMSolver:
  grid: ChartGrid
  constitutive: ConstitutiveModel
  gravity: optional
```

**Key Methods**:
```python
solver.step(particles, dt) → diagnostics: dict
  Execute one MPM step, return kinetic energy, max velocity, etc.
```

### 3.3 Constitutive Models (constitutive.py)

**NeoHookeanModel**:
```python
Ψ = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2)(ln J)²
σ = (1/J)[μ(B - I) + λ ln(J) I]  where B = FF^T
```

**ElastoplasticModel**:
- J2 yield criterion with isotropic hardening
- Radial return mapping in Hencky strain space
- State: {Fp: plastic deformation gradient, equiv_plastic_strain}

---

## Part 4: COMMON UTILITIES (common/)

### 4.1 geometry.py

**Purpose**: Jacobian operations, local coordinates, metric tensors on mapped charts.

**Key Functions**:
```python
local_coords(x, seed, t1, t2, n) → xi: Tensor
  Convert physical coords x to chart-local coords xi via frame

chart_map_and_jacobian(decoder, xi, seed, t1, t2, n, chart_scale)
  → (x, xi, J): physical coords, ref coords, Jacobian via autograd

stabilized_jacobian_ops(J, sigma_floor=1e-6, det_floor=1e-10)
  → (inv_J, det_abs, kappa, valid)
  SVD-based inversion with safeguarded singular values

invert_decoder(decoder, x_target, seed, t1, t2, n, chart_scale, xi_init=None, max_iter=20)
  → xi_star: Find reference coords ξ* via Newton iteration s.t. decoder(ξ*) ≈ x_target
```

### 4.2 schwarz.py

**Purpose**: Atlas loading, chart coloring for multiplicative Schwarz.

**Key Functions**:
```python
load_atlas_models(checkpoint, device, dtype)
  → (decoders, masks, ckpt): Load trained ChartDecoder + MaskNet

choose_color_groups(meta_json, n_charts, membership)
  → List[List[int]]: Greedy graph coloring of overlap graph
  Used to decompose charts into independent groups for parallel Schwarz
```

**Graph Coloring**:
- Builds overlap adjacency: i ~ j if they share nodes
- Greedy coloring: assigns minimum color not used by neighbors
- Output: color groups [G0, G1, ...] where charts in same group are independent

---

## Part 5: KEY BENCHMARKS & EXPERIMENTS

### 5.1 Nine Circles Fracture Benchmarks (nineO_examples/)

**Status**: ✅ **890/900 (98.9%)** with 8/9 at perfect 100/100

**Challenges**:
1. **Uniaxial Tension** (100/100): σ_zz exact, Drucker-Prager nucleation 0.8% error
2. **Biaxial Tension** (100/100): σ_bs 5% error, GUDHI H₀ detection
3. **Torsion** (100/100): τ 0% error, 45° crack angle
4. **Pure Shear Fracture** (100/100): K_I linear, G = G_c exact
5. **Single Edge Notch** (100/100): Strength-Griffith transition
6. **Indentation** (100/100): Ring crack r > R_punch, 44° cone angle
7. **Poker Chip** (100/100): Hydrostatic p > 0, Neo-Hookean + F-bar
8. **DCB** (90/100): Stable crack growth
9. **Trousers Test** (100/100): Mode III, G = 2F/B (Rivlin-Thomas)

### 5.2 Fracture Benchmarks (benchmarks/fracture/)

**Files**:
- `solve_cracked_plate.py`: Single-chart solve on pre-cracked geometry
- `plate_crack_sdf.py`: CrackedPlateSDFOracle (SDF for edge crack)
- `mode_i_edge_crack.py`: Mode I reference solutions
- `lefm_reference.py`: Williams field, stress intensity factor
- `run_biaxial_nucleation.py`: Drucker-Prager crack nucleation
- `run_biaxial_propagation.py`: Griffith criterion quasi-static growth
- `run_dcb.py`: Double cantilever beam test
- `run_torsion.py`: Torsional fracture
- `run_torsion_chart_fem.py`: Multi-chart torsion

---

## Part 6: WHAT'S IMPLEMENTED ✅

### Topology & Geometry
- [x] Persistent homology pipeline (GUDHI integration)
- [x] Birth-death pair extraction, filtering, Betti number computation
- [x] Bottleneck distance for change detection
- [x] Lusternik-Schnirelmann category bounds (M_min calculation)
- [x] Dynamic topology monitoring (TopologyMonitor)
- [x] Persistence-driven chart spawning (ChartSpawner)

### FEM Solvers
- [x] P1 scalar FEM (Poisson, diffusion)
- [x] P1/P2 vector FEM (elasticity, Neo-Hookean)
- [x] Mapped geometry (Jacobian-based metric tensors)
- [x] SDF-filtered mesh generation
- [x] Newton-Raphson nonlinear solve
- [x] Multiplicative Schwarz (sequential)
- [x] Robin parallel domain decomposition (Du 2002)

### Crack Enrichment
- [x] Analytical CrackTipDecoder (power-law singularity absorption)
- [x] Neural crack decoder (base + learned residual MLP)
- [x] K_I extraction via Williams fitting
- [x] Quasi-static crack propagation driver
- [x] Drucker-Prager crack nucleation criterion

### MPM Solver
- [x] Explicit time stepper on mapped charts
- [x] Neo-Hookean hyperelasticity
- [x] J2 elastoplasticity with return mapping
- [x] Multi-chart Schwarz MPM

### Advanced Numerics
- [x] P2 (10-node tet) quadratic elements with 4-point Gauss quadrature
- [x] F-bar volumetric locking prevention (nearly incompressible)
- [x] SVD-stabilized Jacobian inversion
- [x] Autodiff-based tangent assembly

---

## Part 7: WHAT'S PLANNED/MISSING ❌

### Topology Pipeline Extensions
- [ ] **3D crack path prediction**: Use persistent homology to predict next crack propagation direction (currently: max hoop stress only)
- [ ] **Multi-scale homology**: Track Betti numbers across multiple filtration levels for load-path awareness
- [ ] **Adaptive thresholding**: Automatic lifetime_threshold selection via statistical hypothesis testing (currently: manual 0.05 heuristic)
- [ ] **Instability detection**: Bifurcation detection via H_1 pair birth clustering

### Parallel Domain Decomposition
- [ ] **Full asynchronous Robin**: True async update of λ/g without global synchronization (currently: synchronous)
- [ ] **Multilevel DD**: Coarse space based on kernel of interface operator (currently: single level)
- [ ] **FETI (Dual Lagrangian)**: Alternative to Robin; potentially better conditioning
- [ ] **GPU-accelerated Schwarz**: CUDA kernels for overlapping solves (currently: CPU-based SciPy LU)

### Crack Tip Enrichment
- [ ] **3D Williams enrichment**: T-stress and higher-order multipole terms (currently: leading order only)
- [ ] **Dynamic basis selection**: Automatically choose enrichment type (analytical vs neural) based on local geometry
- [ ] **Multi-tip interaction**: Coupled enrichment for propagating crack pairs (currently: single-tip decoders)
- [ ] **Kink/branch transitions**: Support for crack path changes in anisotropic materials

### MPM Integration with Topology
- [ ] **Material point topology tracking**: Tag MPM particles by Betti number (which connected component, tunnel, void)
- [ ] **Topology-driven refinement**: Spawn charts when persistent H_1 is detected in particle mass distribution
- [ ] **Hybrid FEM-MPM coupling**: Use topology to decide which regions use FEM vs MPM

### Schwarz Crack Interface Conditions
- [ ] **Prescribed displacement jump (opening)**: Currently: standard Schwarz transmission (continuity). Should support: u_+ - u_- = COD (crack opening displacement)
- [ ] **Traction contrast**: Non-physical transmission (λ_+ ≠ λ_-) to model cohesive zone

### Extended Finite Element Method (XFEM) Alternative
- [ ] Implement classical XFEM as a comparison baseline
- [ ] Hybrid: CrackTipDecoder for mapping + Level-set enrichment for DoF selection

---

## Part 8: INTEGRATION POINTS & CURRENT GAPS

### Gap 1: TOPOLOGY → CHART SPAWNING → SCHWARZ SOLVER LOOP

**Current State**:
- ✅ TopologyMonitor detects events
- ✅ ChartSpawner generates SpawnedChartPair
- ⚠️ **Missing**: Integration into live simulation loop
  - No integration hook in crack propagation driver
  - No overlap graph rebuild after spawning
  - No automatic Schwarz color group recomputation

**What's Needed**:
```python
# Pseudo-code integration:
for step in range(n_steps):
    u_charts = schwarz.solve(...)
    grid_vals = sdf_oracle.evaluate(u_charts)
    events = monitor.update(grid_vals, step)
    for event in events:
        pair = spawner.spawn_from_event(event, ...)
        atlas.add_charts([pair])
        schwarz.rebuild_neighbors(atlas.membership)
        schwarz.color_groups = choose_color_groups(atlas.meta_json, ...)
```

### Gap 2: CRACK TIP ENRICHMENT ↔ ADAPTIVE MESH

**Current State**:
- ✅ CrackTipDecoder (analytical & neural)
- ✅ Persistence-driven n_cells in ChartSpawner
- ⚠️ **No coupling**: persistence info not passed to decoder initialization

**What's Needed**:
- Decoder weight warm-start from nearby charts based on event location
- Adaptive polynomial order selection (P1 vs P2) from persistence
- Persistence-aware Robin parameter selection

### Gap 3: PARALLEL DOMAIN DECOMPOSITION OPTIMALITY

**Current State**:
- ✅ Robin parallel DD (Du 2002)
- ✅ ThreadPoolExecutor for chart solves
- ⚠️ **Limitations**:
  - No GPU acceleration
  - No multilevel coarse space
  - Delta parameter requires manual tuning
  - No convergence diagnostics in output

### Gap 4: CRACK OPENING DISPLACEMENT IN SCHWARZ

**Current State**:
- Schwarz transmission: u_+ = u_- (continuity)
- ❌ **Missing**: Prescribed jump for crack faces
  - No u_+ - u_- = [COD] support
  - Charts on opposite sides of crack treated as continuous

**Why It Matters**: 
For visualization and postprocessing, need to track crack opening. Currently, the "crack" edge type in overlap graph is a label only — actual BCs are still continuous.

---

## Part 9: TEST COVERAGE

### Topology Tests (topo_atlas/tests/)

**✅ Unit Level**:
- Filtration: grid sampling, clipping, analytic shapes
- LS Category: cup_length, compute_m_min, certify_atlas
- Analytic SDFs: ball, torus, shell

**✅ Integration Level** (requires GUDHI):
- Persistence diagrams on analytic shapes
- Betti numbers at t = -1e-6 (just inside boundary)
- Bottleneck distance computation

**✅ Verification Level**:
- Nerve theorem compliance (M_actual >= M_min)
- Contract verification (ball has 1 H_0 pair, torus has 1 H_1, etc.)

**✅ Monitor Tests**:
- Static domain: no false events
- Propagating crack: detects new_H1 event

### FEM Tests (tests/)

**✅ ChartFEMSolver**:
- Poisson on unit domain
- Manufactured solution convergence

**✅ ChartVectorFEMSolver**:
- Uniaxial tension (P1 & P2)
- Bending, torsion

**✅ Crack Tests**:
- Williams field K_I extraction
- CrackTipDecoder forward/inverse consistency

**Total**: 78 tests + 1 xpass, all passing

---

## Part 10: PERFORMANCE NOTES

### Topology Overhead

GUDHI persistent homology at 16³ grid with monitoring every 50 steps:
- **1.1% computational overhead** (paper measurement)
- O(N³ log N) ~ 0.05 seconds per computation

### Schwarz Convergence

- **Robin parallel**: 2–5 iterations (vs 10–25 for multiplicative)
- Example (uniaxial tension): 2 iterations (exact for affine field)
- Example (nonlinear): 3–4 iterations to 1e-4 tolerance

### Memory

- Atlas with 8 charts: ~50 MB (decoders + masks)
- FEM system (16³ mesh per chart): ~10 MB per chart (sparse matrix)
- Total: ~130 MB for 8-chart solver

---

## Part 11: CODE QUALITY & DOCUMENTATION

### Strengths
- ✅ Comprehensive docstrings (NumPy format)
- ✅ Type hints throughout
- ✅ Verification contracts in module docs
- ✅ Mathematical notation in comments
- ✅ Test suite covers key paths
- ✅ README + CLAUDE.md onboarding guide

### Documentation Files
- `docs/theory_manual.md`: 14 sections, rigorous mathematics
- `docs/VandV.md`: 78 tests, per-phase verification exercises
- `docs/nine_circles.md`: Benchmark results with figures
- `docs/benchmark_readiness.md`: Infrastructure assessment

### Code Organization
- Clear module boundaries (topo, fem, mpm, common)
- Consistent naming (ChartX for chart-mapped solvers)
- Backwards compatible: old checkpoint format still loadable

---

## Part 12: RECOMMENDED NEXT STEPS

### Priority 1: Live Topology-Driven Adaptation Loop

**Why**: Integration is the final piece to fully automate fracture mesh adaptation.

**Implementation**:
```python
# In crack_driver.py or fracture benchmarks:
for step in range(max_steps):
    u = solver.solve(...)
    grid_vals = sdf_oracle.evaluate_at_solution(u, step)
    events = monitor.update(grid_vals, step)
    if events:
        for e in events:
            new_pair = spawner.spawn_from_event(e, ...)
            solver.add_charts([new_pair])
```

**Effort**: 2–3 days (integrate existing pieces)

### Priority 2: Multilevel Robin DD

**Why**: Convergence plateaus for highly nonlinear problems; coarse space fixes this.

**Implementation**: Build projection to H_1(Ω) kernel, solve coarse problem for acceleration.

**Effort**: 1–2 weeks

### Priority 3: GPU Schwarz Solver

**Why**: Parallel solve currently CPU-bound; GPU would accelerate 8-chart solve from ~1s to ~100ms.

**Implementation**: PyTorch-based assembly/solve or CuPy sparse LU.

**Effort**: 2–3 weeks

---

## Summary Table

| Component | Status | Lines | Key Files |
|-----------|--------|-------|-----------|
| Topology Pipeline | ✅ Complete | 1,202 | filtration, persistence, ls_category, monitor, chart_spawn |
| FEM Scalar Solver | ✅ Complete | 669 | chart_fem_solver.py |
| FEM Vector Solver | ✅ Complete | 1,292 | chart_vector_fem.py |
| Robin Parallel DD | ✅ Complete | 299 | robin_schwarz.py |
| Multiplicative Schwarz | ✅ Complete | 637 | schwarz_fem.py |
| CrackTipDecoder | ✅ Complete | 400+348 | analytic_decoders.py + neural_crack_decoder.py |
| P2 Elements | ✅ Complete | 342 | p2_tet.py |
| MPM Solver | ✅ Complete | 570 | chart_mpm_solver + transfers + constitutive |
| Crack Propagation Driver | ⚠️ Partial | 204 | crack_driver.py (no topo loop integration) |
| **Total Core** | | **7,853** | |

**Benchmark Validation**: 98.9% (890/900) on Nine Circles fracture suite — 8/9 perfect 100/100.

