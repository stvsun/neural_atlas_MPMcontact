# Topology-Aware Neural Atlas for FEM/MPM on Complex 3D Geometries

A meshfree framework for solving boundary value problems on complex 3D geometries using **learned signed distance functions (SDFs)**, **overlapping coordinate charts**, and **FEM or MPM solvers** in mapped chart-local coordinates. Topology-aware chart construction via persistent homology automatically determines the minimum number of charts needed and can spawn new charts when the geometry evolves (e.g., crack propagation).

---

## Quick Start

```bash
# Install (editable mode)
pip install -e .

# Run MPM tests
pytest tests/test_chart_mpm_solver.py -v

# Run FEM tests
pytest tests/test_chart_fem_solver.py -v

# Train a neural SDF on geometry
python atlas/sdf/train_sdf.py

# Build an atlas of overlapping charts
python atlas/charts/train_atlas.py

# Run FEM Poisson benchmark on rabbit
python benchmarks/rabbit_poisson_fem/run_rabbit_poisson_fem.py
```

---

## Key Concepts

1. **Neural SDF** — Geometry encoded as a learned signed distance function (Eikonal-trained MLP).
2. **Coordinate Charts** — Overlapping balls on the surface, each with a TNB frame and a decoder network mapping local 2D/3D coords to physical 3D space.
3. **FEM on Charts** — P1 tetrahedral FEM solver operating in each chart's local coordinate system, with mapped diffusion tensors from the chart Jacobian.
4. **MPM on Charts** — Material Point Method with particles in chart-local coordinates, B-spline shape functions, and metric-aware P2G/G2P transfers.
5. **Multiplicative Schwarz** — Charts coupled through alternating value+flux matching at overlap interfaces.
6. **Topology-Aware Construction** — Persistent homology of the neural SDF computes Betti numbers; the Lusternik-Schnirelmann category gives the minimum chart count M_min = cat(Omega) + 1.
7. **Dynamic Chart Spawning** — A topology monitor detects changes (cracks, voids) via bottleneck distance and spawns new chart pairs at topological features.

---

## Repository Structure

```
neural_atlas_MPM/
├── atlas/                        # Geometry & chart infrastructure
│   ├── sdf/                     #   Neural SDF training
│   │   ├── train_sdf.py         #   SDF network training (Eikonal loss)
│   │   ├── train_sdf_chartwise.py
│   │   └── build_mesh_sdf.py    #   Mesh-based SDF construction
│   ├── charts/                  #   Chart/atlas construction
│   │   ├── train_atlas.py       #   Atlas training (chart decoders + masks)
│   │   ├── train_mapping.py     #   Chart decoder training from SDF
│   │   └── poisson_disk_seeding.py  # Poisson-disk atlas seeding
│   └── topo/                    #   Topology-aware construction
│       ├── filtration.py        #   Sublevel-set filtration of SDF
│       ├── persistence.py       #   Persistent homology via GUDHI
│       ├── ls_category.py       #   Lusternik-Schnirelmann category
│       ├── monitor.py           #   Dynamic topology change detection
│       └── chart_spawn.py       #   Automatic chart spawning
│
├── solvers/                      # PDE solvers on mapped charts
│   ├── fem/                     #   Finite Element Method
│   │   ├── chart_fem_solver.py  #   P1 scalar FEM (Poisson, diffusion)
│   │   └── chart_vector_fem.py  #   Vector FEM (elasticity, Neo-Hookean)
│   └── mpm/                     #   Material Point Method
│       ├── chart_mpm_solver.py  #   MPM time stepper on a single chart
│       ├── particles.py         #   MaterialPointCloud data structure
│       ├── grid.py              #   Background grid in chart coords
│       ├── transfers.py         #   P2G and G2P with metric-aware shapes
│       └── constitutive.py      #   Neo-Hookean, J2 elastoplasticity
│
├── common/                       # Shared utilities
│   ├── models.py                #   ChartDecoder, MaskNet, MLP
│   ├── geometry.py              #   Jacobian ops, local coords, metric
│   ├── schwarz.py               #   Atlas loading, chart coloring
│   └── utils.py                 #   Device, seeding, plotting, metrics
│
├── benchmarks/                   # Numerical examples
│   ├── rabbit_poisson_fem/      #   Poisson on Stanford rabbit (FEM Schwarz)
│   ├── mpm_basic/               #   MPM verification benchmarks
│   └── fracture/                #   Fracture benchmarks (Phase 4)
│
├── tests/                        # Test suite
│   ├── test_chart_fem_solver.py #   FEM solver tests
│   └── test_chart_mpm_solver.py #   MPM solver tests (14 tests)
│
├── topo_atlas/                   # Topology pipeline (docs, CI, tests)
│   ├── docs/PLAN.md             #   40-week development roadmap
│   └── tests/test_topo_pipeline.py
│
├── configs/                      # YAML experiment configs
├── postprocessing/               # Visualization & analysis
├── pyproject.toml                # Package definition
├── CLAUDE.md                     # AI agent onboarding guide
└── README.md
```

---

## Solvers

### FEM (Finite Element Method)

The `ChartFEMSolver` operates on a single chart's reference domain:
- Structured hex grid subdivided into 6 tetrahedra per hex (Freudenthal decomposition)
- SDF-filtered mesh: only elements inside the geometry are retained
- Mapped diffusion tensor: A_i(xi) = |det J_i| * J_i^{-1} J_i^{-T}
- P1 (linear) tetrahedral elements with vectorized assembly
- Dirichlet BC via row/column elimination

The `ChartVectorFEMSolver` extends this to 3D vector-valued nonlinear elastostatics with Neo-Hookean materials and Newton-Raphson iteration.

### MPM (Material Point Method)

The `ChartMPMSolver` implements MPM in chart-local coordinates:
- Material points carry position (in xi-space), velocity, deformation gradient F, stress, and history
- Background grid in chart-local coordinates with linear B-spline shape functions
- P2G: scatter mass, momentum, and internal forces to grid
- Grid solve: explicit momentum update with gravity and boundary conditions
- G2P: gather velocities, update F and positions
- Constitutive models: compressible Neo-Hookean, J2 elastoplasticity with return mapping

---

## Development Roadmap

See [topo_atlas/docs/PLAN.md](topo_atlas/docs/PLAN.md) for the 40-week phased development plan:

| Phase | Weeks | Focus |
|-------|-------|-------|
| 0 | 1-2 | Fork setup, reproducible environment |
| 1 | 3-8 | Persistent homology pipeline |
| 2 | 9-14 | Chart count certification on benchmarks |
| 3 | 15-22 | Dynamic topology monitoring + chart spawning |
| 4 | 23-32 | Fracture benchmarks (Mode-I, penny-shaped, cyclic) |
| 5 | 33-40 | Optimization & release |

---

## Dependencies

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **NumPy**, **SciPy**, **Matplotlib**
- **GUDHI** >= 3.8 (optional, for topology pipeline)

```bash
pip install -e .            # core dependencies
pip install -e ".[topo]"    # + GUDHI for topology
pip install -e ".[dev]"     # + pytest for development
```
