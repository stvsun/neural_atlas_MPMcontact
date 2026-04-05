# Topology-Aware Neural Atlas for Brittle Fracture

A meshfree framework for solving brittle fracture problems on complex 3D geometries using **coordinate chart-based FEM**, **persistent homology** for topology-aware chart construction, and **automatic crack detection and mesh adaptation** via GUDHI.

Validated on the **Nine Circles of Elastic Brittle Fracture** benchmark (Kamarei et al. 2026, *CMAME* 448): **890/900 (98.9%)** with 8 of 9 challenges scoring 100/100.

---

## Highlights

- **Topology-aware hp-adaptivity**: GUDHI persistent homology detects cracks/voids and drives automatic chart spawning with persistence-proportional mesh refinement
- **CrackTipDecoder**: radial power-law coordinate map absorbs the $1/\sqrt{r}$ singularity, enabling P1/P2 elements to capture Williams fields without XFEM enrichment
- **Robin parallel DD**: Du (2002) parallel domain decomposition converges in 2–5 iterations (vs 10–25 for multiplicative Schwarz)
- **F-bar method**: volumetric locking prevention for nearly-incompressible elastomers ($\nu \approx 0.5$)
- **P2 quadratic elements**: 10-node tets with 4-point Gauss quadrature for $O(h^2)$ convergence at crack tips
- **Neo-Hookean hyperelasticity**: finite-strain constitutive model with analytical tangent for large deformation fracture

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

# Run test suite
pytest tests/ topo_atlas/tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Theory Manual](docs/theory_manual.md) | Mathematical framework: mapped FEM, constitutive models, F-bar, P2 elements, Robin DD, CrackTipDecoder, Drucker–Prager, Griffith propagation, persistent homology |
| [Nine Circles Benchmark](docs/nine_circles.md) | Detailed results for all 9 Kamarei et al. (2026) challenge problems with figures |
| [Verification & Validation](docs/VandV.md) | Per-phase V&V exercises: 141+ tests covering topology, fracture, solvers |
| [Benchmark Readiness](docs/benchmark_readiness.md) | Assessment of solver infrastructure for fracture benchmarks |

---

## Architecture

```
neural_atlas_MPM/
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
│   └── mpm/                       #   Material Point Method
│
├── nineO_examples/                  # Nine Circles fracture benchmarks
│   ├── run_all.py                  #   Master runner for all 9 challenges
│   ├── pyvista_pub.py              #   Publication-quality figure generation
│   ├── pyvista_utils.py            #   Plotting utilities
│   └── {1..9}_*/                   #   Per-challenge run + score scripts
│
├── docs/                            # Documentation
│   ├── theory_manual.md            #   Mathematical framework (14 sections)
│   ├── nine_circles.md             #   Benchmark results with figures
│   ├── VandV.md                    #   Verification & Validation exercises
│   └── benchmark_readiness.md      #   Infrastructure assessment
│
├── benchmarks/                      # Additional numerical examples
├── tests/                           # Test suite (P2 elements, etc.)
├── topo_atlas/                      # Topology pipeline tests
├── figures/                         # Generated figures
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
