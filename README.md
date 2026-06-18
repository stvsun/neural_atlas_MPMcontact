# Topology-Aware Neural Atlas for Brittle Fracture and Contact Mechanics

A meshfree framework for solving brittle fracture and contact problems on complex 3D geometries using **coordinate chart-based FEM / MPM**, **persistent homology** for topology-aware chart and contact construction, and **automatic detection and adaptation** via GUDHI.

Validated on the **Nine Circles of Elastic Brittle Fracture** benchmark (Kamarei et al. 2026, *CMAME* 448): **890/900 (98.9%)** with 8 of 9 challenges scoring 100/100, and a complete **MPM contact mechanics framework** (Phases 1–8) covering penalty contact, augmented-Lagrangian, Coulomb friction, topology-aware detection, and self-contact — all verified against analytic results to within 0.1%.

---

## Highlights

### Fracture

- **Topology-aware hp-adaptivity**: GUDHI persistent homology detects cracks/voids and drives automatic chart spawning with persistence-proportional mesh refinement
- **CrackTipDecoder**: radial power-law coordinate map absorbs the $1/\sqrt{r}$ singularity, enabling P1/P2 elements to capture Williams fields without XFEM enrichment
- **Robin parallel DD**: Du (2002) parallel domain decomposition converges in 2–5 iterations (vs 10–25 for multiplicative Schwarz)
- **F-bar method**: volumetric locking prevention for nearly-incompressible elastomers ($\nu \approx 0.5$)
- **P2 quadratic elements**: 10-node tets with 4-point Gauss quadrature for $O(h^2)$ convergence at crack tips
- **Neo-Hookean hyperelasticity**: finite-strain constitutive model with analytical tangent for large deformation fracture

### Contact Mechanics (new)

- **SDF gap oracle + normal via autograd**: single neural-SDF forward pass gives the gap value and unit outward normal in one shot (Eikonal-trained normals are ~unit without extra post-processing)
- **Penalty contact**: per-particle penalty force scattered through P2G alongside gravity — no Jacobian pull-back needed, structurally identical to gravity
- **Augmented Lagrangian (Uzawa)**: persistent contact-pressure multiplier builds restraint across time steps; in the ball-drop benchmark at the same $\epsilon_n$, AL reduces the final penetration from 5.8 cm to exactly 0 cm
- **Regularized Coulomb friction**: sliding-block benchmark matches $a = \mu g$ to 0.025% (measured 2.9423 m/s² vs analytic 2.9430 m/s²)
- **Multi-body orchestration via Schwarz**: `SchwarzMPMSolver.configure_contact()` handles broad-phase culling, narrow-phase SDF eval, and force scatter across charts; fully vectorized broad-phase
- **Topology-aware contact**: `ContactTopologyMonitor` detects first-contact and separation events via $\beta_0$ transitions on the combined SDF $\phi_{AB} = \min(\phi_A, \phi_B)$; wired into the MPM loop via `observe_contact_topology()`
- **Contact chart spawning**: `spawn_contact_chart_pair()` bridges topology events to the existing `SpawnedChartPair` / `add_charts` machinery
- **Self-contact**: surface-filter + initial-gap-delta heuristic catches folding without false-positiving bulk particles — verified on an analytic slab fold

---

## Nine Circles Benchmark Results

| # | Challenge | Section | Score | Key Result |
|---|-----------|:-------:|:-----:|------------|
| 1 | Uniaxial Tension | 2.1 | **100** | $\sigma_{zz}$ exact, DP nucleation 0.8% error |
| 2 | Biaxial Tension | 2.2 | **100** | $\sigma_{bs}$ 5.0% error, GUDHI $H_0$ detection |
| 3 | Torsion | 2.3 | **100** | $\tau$ 0.00% error, 45-deg crack angle |
| 4 | Pure-Shear Fracture | 3.1 | **100** | $K_I$ linear scaling, $G = G_c$ exact |
| 5 | Single Edge Notch | 4.1 | **100** | Strength–Griffith transition (8 crack lengths) |
| 6 | Indentation | 4.2 | **100** | Ring crack $r > R_{\text{punch}}$, cone angle 44 deg |
| 7 | Poker-Chip | 4.3 | **100** | Hydrostatic $p > 0$, Neo-Hookean + F-bar |
| 8 | DCB | 5.1 | **90** | FEM $K_I$ extraction, stable crack growth |
| 9 | Trousers | 5.2 | **100** | Mode III $G = 2F/B$ (Rivlin–Thomas) |

### Von Mises Stress Visualizations

| | | |
|:---:|:---:|:---:|
| ![Ch1](figures/challenge_1_von_mises_pub.png) | ![Ch2](figures/challenge_2_von_mises_pub.png) | ![Ch3](figures/challenge_3_von_mises_pub.png) |
| 1. Uniaxial Tension | 2. Biaxial Tension | 3. Torsion |
| ![Ch4](figures/challenge_4_von_mises_pub.png) | ![Ch5](figures/challenge_5_von_mises_pub.png) | ![Ch6](figures/challenge_6_von_mises_pub.png) |
| 4. Pure-Shear Fracture | 5. Single Edge Notch | 6. Indentation |
| ![Ch7](figures/challenge_7_von_mises_pub.png) | ![Ch8](figures/challenge_8_von_mises_pub.png) | ![Ch9](figures/challenge_9_von_mises_pub.png) |
| 7. Poker-Chip | 8. Double Cantilever Beam | 9. Trousers Test |

---

## MPM Contact Mechanics

The contact framework extends the neural-atlas codebase with a complete meshfree contact stack for the Material Point Method, built in **seven phases** and verified against analytic results or machine precision on every path.

### Phase ledger

| Phase | Module | What it adds | Verification |
|---|---|---|---|
| 1 | `solvers/contact/gap.py`, `contact_pair.py`, `contact_manager.py` | SDF gap oracle (forward pass + autograd) + `ContactBody`, `ContactPair` + broad/narrow phase | Normal accuracy vs analytic sphere $< 1$ degree |
| 2 | `solvers/contact/penalty.py` | Penalty force $f = \epsilon_n \max(0, -g) V_p\, \mathbf{n}$ + contact-stable $dt$ | Ball-drop benchmark, P2G scatter conserves total contact force to machine precision |
| 3 | `solvers/mpm/schwarz_mpm.py::configure_contact` | Multi-body orchestration; vectorized broad-phase + per-chart force computation | Two-sphere collision: momentum conserved to $2\times10^{-15}$, symmetric 22.5% deceleration |
| 4 | `solvers/contact/augmented_lagrangian.py` | Persistent Uzawa multiplier; exact enforcement at moderate $\epsilon_n$ | Ball-drop at $\epsilon_n = 5\times10^4$: penalty penetrates 5.8 cm, **AL penetrates 0 cm** |
| 6 | `solvers/contact/friction.py` | Regularized Coulomb $\mathbf{f}_T = -\mu \|\mathbf{f}_N\| \mathbf{v}_T / \sqrt{\|\mathbf{v}_T\|^2 + \epsilon_T^2}$ | Sliding-block benchmark: measured deceleration **2.9423 m/s²** vs analytic $\mu g = $ **2.9430 m/s²** (ratio 1.000) |
| 7 | `solvers/contact/contact_topology.py` | Persistent-homology event detection on combined SDF $\phi_{AB} = \min(\phi_A, \phi_B)$ | Two-sphere sweep: exactly one `first_contact` and one `separation` event at the grid-resolved transition |
| 7b | `solvers/contact/contact_chart_spawn.py` | `spawn_contact_chart_pair()` bridges events to `SchwarzMPMSolver.add_charts()` | End-to-end: SDF autograd normal → frame → spawn pair → solver grows cleanly from 1 to 3 charts |
| 8 | `solvers/contact/self_contact.py` | Surface-filter + initial-gap-delta heuristic for folding self-contact | Folding-slab benchmark: 36 surface particles flagged mid-fold, 5 bulk particles never active |

### Key design decisions

- **No Jacobian pull-back for contact forces** — MPM grid velocity and gravity are in physical space; contact forces follow the same convention, scattered as `f_I += f_p * N_I(ξ_p)` (no `J^{-T}` factor). Consistent with §11.3 of the theory manual.
- **Penalty and AL share the same force API** — both return per-particle force in the convention expected by `particle_to_grid(..., contact_force=...)`, so users can swap strategies without touching the solver.
- **Friction is stateless and composable** — `compute_friction_force(v, n, ‖f_N‖, μ, ε_T)` is a single function that plugs onto any normal-force scheme.
- **Topology monitor is independent of the crack `TopologyMonitor`** — `ContactTopologyMonitor` watches $H_0$ of the combined SDF, while the existing `TopologyMonitor` watches $H_1$ of a single body. Both can run in the same simulation via `setup_topology_monitoring()` and `observe_contact_topology()` respectively.

### Contact benchmarks

| Benchmark | What it demonstrates | File |
|---|---|---|
| Ball drop (penalty) | Bounded penetration and rebound | `benchmarks/contact/ball_drop_mpm.py` |
| Two-sphere collision | Symmetric multi-body contact | `benchmarks/contact/two_sphere_collision_mpm.py` |
| Ball drop (AL vs penalty) | AL reduces residual penetration from 5.8 cm to 0 cm | `benchmarks/contact/ball_drop_al_mpm.py` |
| Sliding block with friction | Coulomb deceleration matches $\mu g$ to 0.025% | `benchmarks/contact/sliding_block_mpm.py` |
| Topology event sweep | $\beta_0$ transition detection across an approach/separation sweep | `benchmarks/contact/contact_topology_demo.py` |
| Folding slab (self-contact) | Folding detection without bulk-particle false positives | `benchmarks/contact/folding_slab_mpm.py` |

### Test coverage

**91 contact-framework tests** across `tests/test_contact_detection.py`, `test_mpm_contact.py`, `test_mpm_friction.py`, `test_contact_topology.py`, and `test_self_contact.py`. Full repo suite: **245 passed, 1 skipped, 0 failed**.

See [Theory Manual §15](docs/theory_manual.md#15-mpm-contact-mechanics) for the mathematical formulation and [`docs/mpm_velocity_gradient_audit.md`](docs/mpm_velocity_gradient_audit.md) for a latent-bug audit of the MPM velocity-gradient pipeline for curved chart decoders (found during the contact audit, outside the contact scope).

---

## Quick Start

```bash
# Install
pip install -e .

# Run all 9 fracture benchmarks
python nineO_examples/run_all.py

# Score all challenges
python -c "
import importlib.util, sys; sys.path.insert(0, '.')
for i in range(1, 10):
    d = [d for d in __import__('os').listdir('nineO_examples') if d.startswith(f'{i}_')][0]
    spec = importlib.util.spec_from_file_location(f's{i}', f'nineO_examples/{d}/score.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.run_score()
"

# Generate publication-quality figures
python nineO_examples/pyvista_pub.py

# Run test suite (245 tests including 91 contact-framework tests)
pytest tests/ topo_atlas/tests/ -v

# Run contact benchmarks
PYTHONPATH=. python benchmarks/contact/ball_drop_mpm.py
PYTHONPATH=. python benchmarks/contact/ball_drop_al_mpm.py
PYTHONPATH=. python benchmarks/contact/sliding_block_mpm.py
PYTHONPATH=. python benchmarks/contact/two_sphere_collision_mpm.py
PYTHONPATH=. python benchmarks/contact/contact_topology_demo.py
PYTHONPATH=. python benchmarks/contact/folding_slab_mpm.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Theory Manual](docs/theory_manual.md) | Mathematical framework: mapped FEM, constitutive models, F-bar, P2 elements, Robin DD, CrackTipDecoder, Drucker–Prager, Griffith propagation, persistent homology, **§15 MPM contact mechanics** (penalty, AL, friction, topology, self-contact) |
| [Nine Circles Benchmark](docs/nine_circles.md) | Detailed results for all 9 Kamarei et al. (2026) challenge problems with figures |
| [Verification & Validation](docs/VandV.md) | Per-phase V&V exercises: 141+ tests covering topology, fracture, solvers |
| [Contact Verification Manual](docs/contact_verification_manual.md) | Analytical contact benchmarks (CV-1..CV-5: Hertz, Cattaneo–Mindlin, Brazilian disc, nine-disc, nonconvex superformula) as closed-form acceptance targets, with the **§10 protocol for verifying neural coordinate charts**. Artifact index: [docs/hertz_derivation/](docs/hertz_derivation/README.md) |
| [Benchmark Readiness](docs/benchmark_readiness.md) | Assessment of solver infrastructure for fracture benchmarks |
| [MPM Velocity-Gradient Audit](docs/mpm_velocity_gradient_audit.md) | Dimensional analysis of a latent correctness bug in the MPM solver for **curved** chart decoders (does not affect the contact framework, which always runs in physical space) |

---

## Architecture

```
neural_atlas_MPMcontact/
├── atlas/                           # Geometry & chart infrastructure
│   ├── sdf/                        #   Neural SDF training (Eikonal loss)
│   ├── charts/                     #   Chart/atlas construction
│   └── topo/                       #   Topology-aware construction
│       ├── filtration.py           #     Sublevel-set filtration of SDF
│       ├── persistence.py          #     Persistent homology via GUDHI
│       ├── ls_category.py          #     Lusternik-Schnirelmann category
│       ├── monitor.py              #     Dynamic topology change detection
│       └── chart_spawn.py          #     Persistence-driven chart spawning
│
├── solvers/                         # PDE solvers on mapped charts
│   ├── fem/                        #   Finite Element Method
│   │   ├── chart_vector_fem.py    #     P1/P2 tet FEM (elasticity, Neo-Hookean)
│   │   ├── robin_schwarz.py       #     Robin parallel DD (Du 2002)
│   │   ├── linear_elastic.py      #     Small-strain linear elastic model
│   │   ├── analytic_decoders.py   #     BoxDecoder, TubeSector, CrackTipDecoder
│   │   ├── p2_tet.py              #     P2 shape functions & quadrature
│   │   ├── k_extraction.py        #     K_I extraction (Williams fitting)
│   │   └── crack_propagation.py   #     Quasi-static crack propagation driver
│   ├── fem/chart_fem_solver.py    #   P1 scalar FEM (Poisson, diffusion)
│   ├── fracture_criteria.py       #   Drucker-Prager, Griffith K_Ic
│   ├── mpm/                       #   Material Point Method
│   │   ├── particles.py            #     MaterialPointCloud
│   │   ├── grid.py                 #     ChartGrid
│   │   ├── transfers.py            #     P2G/G2P (with contact_force hook)
│   │   ├── constitutive.py         #     Neo-Hookean, J2 elastoplastic
│   │   ├── chart_mpm_solver.py     #     Single-chart explicit MPM stepper
│   │   └── schwarz_mpm.py          #     Multi-chart MPM + configure_contact
│   │                               #       + observe_contact_topology
│   └── contact/                    #   Contact mechanics (Phases 1–8)
│       ├── gap.py                  #     SDF gap + normal via autograd
│       ├── contact_pair.py         #     ContactBody, ContactPair dataclasses
│       ├── contact_manager.py      #     Broad-phase (vectorized) + narrow-phase
│       ├── penalty.py              #     Penalty force + contact-stable dt
│       ├── augmented_lagrangian.py #     Uzawa multiplier AL contact
│       ├── friction.py             #     Regularized Coulomb friction
│       ├── contact_topology.py     #     ContactTopologyMonitor (combined-SDF PH)
│       ├── contact_chart_spawn.py  #     Chart spawning on first_contact events
│       └── self_contact.py         #     Folding / self-penetration detection
│
├── nineO_examples/                  # Nine Circles fracture benchmarks
│   ├── run_all.py                  #   Master runner for all 9 challenges
│   ├── pyvista_pub.py              #   Publication-quality figure generation
│   ├── pyvista_utils.py            #   Plotting utilities
│   └── {1..9}_*/                   #   Per-challenge run + score scripts
│
├── benchmarks/
│   ├── contact/                    #   Contact framework benchmarks
│   │   ├── ball_drop_mpm.py        #     Penalty contact, bounded penetration
│   │   ├── ball_drop_al_mpm.py     #     Penalty vs Augmented Lagrangian
│   │   ├── two_sphere_collision_mpm.py #  Multi-body contact
│   │   ├── sliding_block_mpm.py    #     Coulomb friction, measured a ≈ μ g
│   │   ├── contact_topology_demo.py#     β₀ event detection sweep
│   │   └── folding_slab_mpm.py     #     Self-contact detection
│   ├── rabbit_poisson_fem/         #   FEM Poisson on Stanford rabbit
│   └── fracture/                   #   Legacy fracture benchmarks
│
├── docs/                            # Documentation
│   ├── theory_manual.md            #   Mathematical framework (15+ sections)
│   ├── nine_circles.md             #   Benchmark results with figures
│   ├── VandV.md                    #   Verification & Validation exercises
│   ├── benchmark_readiness.md      #   Infrastructure assessment
│   └── mpm_velocity_gradient_audit.md  # Latent curved-decoder MPM audit
│
├── tests/                           # Test suite (245 tests)
│   ├── test_chart_mpm_solver.py    #   Core MPM + P2G mass/momentum conservation
│   ├── test_contact_detection.py   #   Phase 1 (gap, broad/narrow phase)
│   ├── test_mpm_contact.py         #   Phases 2–4 (penalty, AL, Schwarz contact)
│   ├── test_mpm_friction.py        #   Phase 6 (Coulomb friction)
│   ├── test_contact_topology.py    #   Phase 7 + 7b (topology, chart spawning)
│   ├── test_self_contact.py        #   Phase 8 (folding self-contact)
│   └── ...                          #   (fracture, crack propagation, etc.)
│
├── topo_atlas/                      # Topology pipeline tests
├── figures/                         # Generated fracture figures
├── runs/                            # Benchmark output (history.json, etc.)
├── contact_atlas/                   # Contact mechanics planning docs
├── CLAUDE.md                        # AI agent onboarding guide
└── README.md
```

---

## Key Technical Components

### Persistent Homology Pipeline

The central contribution: GUDHI persistent homology on the SDF sublevel-set filtration detects topological changes (cracks, voids, fragmentation) during simulation and drives automatic chart spawning with persistence-proportional mesh refinement.

```
SDF grid → GUDHI CubicalComplex → Persistence Diagrams → TopologyMonitor
    → Bottleneck distance comparison → TopologyEvent → ChartSpawner
    → Persistence-driven n_cells + P2 recommendation → New charts
```

Key results:
- Bottleneck stability theorem guarantees robustness to SDF perturbation
- 1.1% computational overhead at 16³ grid resolution
- Verified on 11 analytical test cases (all PASS)

### CrackTipDecoder

Absorbs the $1/\sqrt{r}$ Williams stress singularity into the coordinate mapping: $r_{\text{phys}} = R \cdot \xi^2$, so $\sqrt{r} \sim \xi$ (linear). P1 elements capture the singularity exactly; P2 elements give $O(h^2)$ convergence.

### Robin Parallel Domain Decomposition

Du (2002) parallel Robin transmission conditions: $\partial u/\partial n + \delta u = \partial u_{\text{neighbor}}/\partial n + \delta\lambda$. Converges in 2 iterations for affine fields (exact for uniaxial tension).

### F-bar Volumetric Locking Prevention

For near-incompressible materials ($\nu \approx 0.5$): $\bar{F} = (\bar{J}/J)^{1/3} F$ with node-patch volume-weighted $\bar{J}$ averaging. Without F-bar: zero displacement (complete locking). With F-bar: physical deformation.

### SDF Gap Oracle and Augmented Lagrangian Contact

The contact path turns a trained neural SDF into a gap function with zero extra training: one forward pass gives $\phi(x) = \text{gap}$, and `torch.autograd.grad` gives the unit normal $\mathbf{n} = \nabla \phi / \|\nabla \phi\|$ (the Eikonal loss drives $\|\nabla \phi\| \to 1$ during SDF training, so the normalization is nearly a no-op). The penalty force

$$\mathbf{f}_p^{\text{contact}} = \epsilon_n \max(0, -g) V_p\, \mathbf{n}$$

scatters through the same P2G channel as gravity. The augmented Lagrangian variant

$$p_{\text{aug}} = \max\!\big(0,\; \lambda - \epsilon_n g\big),\qquad \lambda^{k+1} = \max\!\big(0,\; \lambda^k - \epsilon_n g(u^{k+1})\big)$$

maintains a persistent per-particle multiplier that absorbs the long-term restraint, enabling exact constraint enforcement at moderate $\epsilon_n$ (see benchmark table above).

### Topology-Aware Contact Detection

Inter-body contact events are read directly from the persistent homology of the combined SDF $\phi_{AB}(x) = \min(\phi_A(x), \phi_B(x))$: a drop in $\beta_0$ is a **first-contact** event, and a rise is a **separation** event. No particle-level testing required — the detection is a persistent-homology readout that reuses the same `atlas/topo/` pipeline that detects cracks. Each event carries a physical-space location (the point where $\phi_{AB}$ is most negative), and `spawn_contact_chart_pair()` converts it into a `SpawnedChartPair` that the existing `SchwarzMPMSolver.add_charts()` machinery knows how to absorb.

---

## Materials

| Material | E (MPa) | $\nu$ | $\sigma_{ts}$ (MPa) | $G_c$ (N/m) | Used in |
|----------|---------|-------|---------------------|-------------|---------|
| Soda-lime glass | 70,000 | 0.22 | 40 | 10 | Ch 1–6, 8 |
| PU elastomer | $\mu$=0.52, $\Lambda$=85.77 | ~0.4997 | 0.3 | 41 | Ch 7, 9 |

---

## References

1. Kamarei, Zeng, Dolbow & Lopez-Pamies (2026). "Nine circles of elastic brittle fracture." *CMAME* 448, 118449.
2. Du (2002). "Optimization based nonoverlapping domain decomposition algorithms." *SIAM J. Numer. Anal.* 39(3).
3. de Souza Neto et al. (1996). F-bar method. *Int. J. Solids Struct.* 33.
4. Tada, Paris & Irwin (2000). *The Stress Analysis of Cracks Handbook*.
5. Cohen-Steiner, Edelsbrunner & Harer (2007). Stability of persistence diagrams. *Discrete Comput. Geom.* 37(1).

**Data:** [Illinois Data Bank](https://databank.illinois.edu/datasets/IDB-6684845) | [Duke Repository](https://research.repository.duke.edu/record/401)
