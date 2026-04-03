# Implementation, Verification and Validation Plan
## Topology-Aware Neural Atlas via Persistent Homology
### Forked from `stvsun/neural_atlas_MPM`

---

## 1. Project Overview

This project extends the neural atlas framework of Sun (2026) with a topology-aware chart construction pipeline. The central contribution is replacing the user-specified chart count `M` with a mathematically certified minimum `M_min = cat(Ω) + 1` derived automatically from the persistent homology of the neural signed distance function `s_θ`. A secondary contribution is a dynamic atlas protocol that detects topology changes (crack propagation, phase transitions) and autonomously spawns new charts.

**Core mathematical claim to verify:** For any domain Ω covered by the atlas, `M_actual ≥ M_min` at all times, where `M_min` is the Lusternik-Schnirelmann category bound computed from the SDF filtration.

---

## 2. Repository Structure After Fork

```
neural_atlas_MPM/          ← forked from stvsun/neural_atlas_MPM
│
├── atlas/
│   ├── sdf.py             (existing) SDF training
│   ├── charts.py          (existing) chart construction
│   ├── schwarz.py         (existing) Schwarz coupling
│   ├── quality_gates.py   (existing) g_cov, g_fold, g_ov, g_rmse
│   │
│   └── topo/              ← NEW MODULE (this project)
│       ├── __init__.py
│       ├── filtration.py      sublevel-set filtration + analytic SDFs
│       ├── persistence.py     GUDHI interface, Betti numbers, bottleneck distance
│       ├── ls_category.py     LS category, M_min, certify_atlas()
│       ├── monitor.py         TopologyMonitor for dynamic simulations
│       └── chart_spawn.py     ChartSpawner for topology-change events
│
├── tests/
│   ├── test_topo_pipeline.py  ← NEW (Levels 1-4, see §4)
│   └── (existing tests)
│
├── benchmarks/
│   ├── ellipsoid/         (existing)
│   ├── rabbit/            (existing)
│   ├── torus/             (existing)
│   └── fracture/          ← NEW
│       ├── edge_crack/        Mode-I edge crack, plane strain
│       ├── penny_crack/       Penny-shaped crack, 3D
│       └── cyclic_fracture/   Fatigue crack under cyclic loading
│
├── docs/
│   └── PLAN.md            (this file)
│
└── .github/workflows/
    └── ci.yml             ← NEW (3-tier CI)
```

---

## 3. Implementation Plan

### Phase 0 — Fork Setup (Week 1–2)

**Goals:** Reproducible environment, CI green on existing tests.

| Task | Owner | Acceptance criterion |
|------|-------|---------------------|
| Fork `stvsun/neural_atlas_MPM` | PhD student | Fork exists, all existing tests pass |
| Audit Python/CUDA dependencies | PhD student | `requirements.txt` pinned and documented |
| Confirm existing benchmarks reproduce | PhD student | Ellipsoid L² error ≤ 2.26×10⁻³ (Table 5, Sun 2026) |
| Add GUDHI to `requirements.txt` | PhD student | `pip install gudhi` in CI succeeds |
| Create `atlas/topo/` scaffold | PhD student | `from atlas.topo import filtration` imports without error |

**Risks:** The existing codebase uses PyTorch Metal backend on Apple Silicon (Appendix A, Sun 2026). Confirm CUDA/CPU fallback works in CI.

---

### Phase 1 — Persistent Homology Pipeline (Weeks 3–8)

**Goals:** Correct computation of Betti numbers from the neural SDF.

| Week | Task | Verification gate |
|------|------|------------------|
| 3–4 | Implement `filtration.py`: grid sampling, `clip_to_interior`, analytic SDFs | `TestFiltration` all pass |
| 4–5 | Implement `persistence.py`: GUDHI CubicalComplex interface, `betti_numbers_at` | `TestPersistence` all pass |
| 5–6 | Implement `ls_category.py`: `compute_m_min`, `certify_atlas` | `TestLSCategory` all pass |
| 6–7 | Differentiable persistence loss `L_topo` in SDF training | Unit test: gradient w.r.t. SDF weights non-zero |
| 7–8 | Integrate into existing atlas quality gate pipeline | `certify_atlas()` called automatically after atlas build |

**Key implementation decision:** GUDHI's `CubicalComplex` operates on top-dimensional cells (the grid cube values), not the vertices. Ensure filtration values are assigned to cubes, not grid points, to avoid off-by-one in birth values.

---

### Phase 2 — Chart Count Certification (Weeks 9–14)

**Goals:** All new and existing atlas builds automatically produce `M ≥ M_min`.

| Week | Task | Verification gate |
|------|------|------------------|
| 9–10 | Wire `M_min` into seed-placement algorithm (atlas construction § 4.1.2) | `M_actual ≥ M_min` asserted in atlas builder |
| 10–11 | Run certification on all three existing benchmarks | See §5.1 Validation Val-1 |
| 11–12 | Reproduce Sun (2026) torus result: verify M=8 ≥ M_min=2 | `TestVerification::test_nerve_theorem_torus` passes |
| 12–13 | Reproduce Stanford Bunny: verify M=12 ≥ M_min=1 | `certify_atlas` report shows PASS |
| 13–14 | Quantify gap M_practical / M_min across benchmark suite | Table: M_min, M_paper, gap for each geometry |

---

### Phase 3 — Dynamic Topology Monitoring (Weeks 15–22)

**Goals:** `TopologyMonitor` reliably detects topology changes with ≤ 2 load-step latency.

| Week | Task | Verification gate |
|------|------|------------------|
| 15–16 | Implement `monitor.py`: `TopologyMonitor.update()`, event detection | `TestMonitor::test_no_events_on_fixed_domain` passes |
| 16–17 | Implement `chart_spawn.py`: feature localization, seed placement | `TestMonitor::test_topology_change_detected` passes |
| 17–19 | Benchmark: propagating crack in an elastic plate (synthetic SDF) | Event fires within 2 steps of topological change |
| 19–21 | Integrate with J₂ elastoplastic solver (Sun 2026 Example 3 framework) | Crack-path topology matches prescribed reference |
| 21–22 | Validate chart spawning does not degrade Schwarz convergence | Interface displacement jump remains O(10⁻⁷) after spawning |

---

### Phase 4 — Fracture Benchmarks (Weeks 23–32)

**Goals:** Validate the dynamic atlas on three canonical fracture problems.

| Benchmark | Reference | Success criterion |
|-----------|-----------|------------------|
| Mode-I edge crack, linear elastic | LEFM K_I solution | Stress intensity factor error < 5% |
| Penny-shaped crack, 3D | Sneddon (1951) analytical solution | L² displacement error < 2% |
| Fatigue crack, cyclic loading | Paris law da/dN | Crack-growth rate within 10% of reference |

---

### Phase 5 — Integration, Optimization, Dissemination (Weeks 33–40)

- Performance profiling: GUDHI call overhead per load step (target < 5% of Newton solve time)
- Open-source release: `neural-atlas-topo` Python package on PyPI
- Paper submission: *J. Comput. Phys.* (topology pipeline) + *Comput. Methods Appl. Mech. Engrg.* (fracture benchmarks)

---

## 4. Verification Plan

Verification establishes that the code correctly implements the stated mathematical specification. Tests are organized in four levels of increasing integration.

### Level 1 — Unit Tests (no external dependencies)

**Location:** `tests/test_topo_pipeline.py::TestFiltration`, `::TestLSCategory`
**Run with:** `pytest -k "TestFiltration or TestLSCategory"`

| Test ID | What is verified | Expected result |
|---------|-----------------|-----------------|
| V1.1 | `sdf_ball` returns negative values in interior | `SDF(0,0,0) < 0` |
| V1.2 | `sdf_ball` returns zero on surface | `|SDF(1,0,0)| < 1e-6` |
| V1.3 | `clip_to_interior` removes all exterior values | `clipped.max() ≤ 0` |
| V1.4 | Grid sampling produces correct shape `(N,N,N)` | Shape assertion |
| V1.5 | `compute_m_min` for convex body = 1 | `M_min({0:1,1:0,2:0}) == 1` |
| V1.6 | `compute_m_min` for solid torus = 2 | `M_min({0:1,1:1,2:0}) == 2` |
| V1.7 | `compute_m_min` for torus surface = 3 | `M_min({0:1,1:2,2:1}) == 3` |
| V1.8 | `certify_atlas` detects foldover failure | `quality_pass=False` when `g_fold>0` |
| V1.9 | Sun (2026) torus benchmark: M=8 passes M_min=2 | `topology_pass=True` |

### Level 2 — Integration Tests (require GUDHI)

**Location:** `tests/test_topo_pipeline.py::TestPersistence`
**Run with:** `pytest -k "TestPersistence"`

| Test ID | What is verified | Expected result |
|---------|-----------------|-----------------|
| V2.1 | Ball has no H₁ or H₂ features after filtering | `len(filtered[1]) == 0` |
| V2.2 | Ball is connected (beta_0 = 1) | `betti[0] == 1` |
| V2.3 | Solid torus has exactly 1 H₁ loop | `len(filtered[1]) == 1` |
| V2.4 | Spherical shell has exactly 1 H₂ void | `len(filtered[2]) == 1` |
| V2.5 | Torus Betti numbers match theory | `{0:1, 1:1, 2:0}` |

### Level 3 — Mathematical Verification (nerve theorem contracts)

**Location:** `tests/test_topo_pipeline.py::TestVerification`
**Run with:** `pytest -k "TestVerification"`

These are the highest-priority tests. A failure here indicates a fundamental correctness bug.

| Test ID | Mathematical statement verified |
|---------|--------------------------------|
| V3.1 | Nerve theorem: ball M_min = 1 (cat(ball) = 0) |
| V3.2 | Nerve theorem: solid torus M_min = 2 (cat(S¹) = 1) |
| V3.3 | Bottleneck stability: ‖f - g‖∞ ≤ ε ⟹ d_B(Dgm(f), Dgm(g)) ≤ ε |

### Level 4 — Dynamic Monitor Verification

**Location:** `tests/test_topo_pipeline.py::TestMonitor`

| Test ID | What is verified |
|---------|-----------------|
| V4.1 | Fixed domain: zero events after initial step |
| V4.2 | Ball → torus switch: at least 1 H₁ event on transition step |
| V4.3 | Monitor correctly resets to zero-event baseline on `reset()` |

---

## 5. Validation Plan

Validation establishes that the correct mathematics solves the right physical problem. Each benchmark has a ground truth from an independent source.

### Val-1 — Reproduce Sun (2026) Existing Benchmarks

**Purpose:** Confirm that adding the topology module does not regress existing results.

| Benchmark | Paper result | Pass criterion |
|-----------|-------------|----------------|
| Ellipsoid Poisson | rel. L² = 2.26×10⁻³ | within 5% of paper value |
| Rabbit Poisson (PINN) | rel. L² = 2.21×10⁻² | within 10% |
| Rabbit Poisson (FEM, n=56) | rel. L² = 1.79×10⁻² | within 5% |
| Torus inverse, displacement mode | μ error < 10⁻¹¹% | within 1 order of magnitude |

### Val-2 — Topology-Certified Atlas Benchmark

**Purpose:** Verify that `certify_atlas()` is correct and that the topology pipeline adds no accuracy penalty.

**Protocol:**
1. Build torus atlas using the existing pipeline (M=8). Run `certify_atlas`.
2. Build a new atlas with M=2 (topological minimum). Compare PDE solution accuracy.
3. Expected: M=2 should still satisfy M_min but will have higher L² error due to poor Jacobian conditioning. This quantifies the topology–numerics gap.

| Quantity | M=8 (paper) | M=2 (topological min) | Expected |
|---------|-------------|----------------------|---------|
| M_min certification | PASS | PASS | Both ≥ 2 |
| Torus Poisson L² error | ~2% | >10% (expected) | Quality gates explain the gap |
| g_fold | 0.0 | expected > 0 | Numerical instability at M_min |

### Val-3 — Mode-I Edge Crack Validation

**Purpose:** Validate dynamic chart spawning against analytical Linear Elastic Fracture Mechanics.

**Setup:** Rectangular plate, width W=2, height H=4, initial edge crack length a=0.5, far-field tension σ∞. Linear elastic material (E=200, ν=0.3). Load stepped until a/W = 0.5.

**Ground truth:** Stress intensity factor K_I = σ∞ √(πa) F(a/W), where F is the standard shape factor from Tada, Paris & Irwin (2000).

| Quantity | Target | Fail threshold |
|---------|--------|---------------|
| K_I relative error | < 5% | > 10% |
| Crack-tip H₁ detection latency | ≤ 2 load steps | > 5 steps |
| Interface jump after spawning | O(10⁻⁶) | > 10⁻³ |
| Schwarz sweeps to convergence | ≤ 10 | > 30 |

### Val-4 — No-Crack Regression (False Positive Rate)

**Purpose:** Verify the monitor does not fire spurious events on an intact domain under large but elastic deformation.

**Protocol:** Apply 20% compressive strain to the torus benchmark (Sun 2026 Example 3) without exceeding yield. Monitor should report zero topology events.

| Quantity | Expected | Fail criterion |
|---------|----------|---------------|
| Total topology events | 0 | > 0 |
| Betti numbers change | None | Any change |

---

## 6. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R1 | GUDHI CubicalComplex birth values off by grid-spacing `h` | High | Medium | Use relative lifetime threshold 0.05×range; verified by V3.3 |
| R2 | LS category bound underestimates cat for product manifolds | Low | Medium | Accepted limitation; documented in `ls_category.py`; affects S¹×S² but not paper's benchmark suite |
| R3 | Chart spawning corrupts internal variables in J₂ plasticity | Medium | High | Per §3.2.1 of paper: only displacement traces exchanged; internal vars remain private; test explicitly in Val-3 |
| R4 | GUDHI overhead > 10% per load step | Medium | Medium | Profile in Phase 4; if needed, subsample to every N steps and use bottleneck-distance threshold |
| R5 | Topology monitor fires false positive under large elastic deformation | Medium | High | Validated by Val-4 before deploying to fracture benchmarks |
| R6 | M_min = 1 for Bunny (genus-0) means existing M=12 is far above floor | None | Low | Expected and documented; M_practical driven by numerics, not topology for genus-0 |

---

## 7. Getting Started

### Step 1: Fork and clone
```bash
# Fork stvsun/neural_atlas_MPM on GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/neural_atlas_MPM.git
cd neural_atlas_MPM
```

### Step 2: Install dependencies
```bash
pip install torch numpy pytest
pip install gudhi          # required for Levels 2-4 tests
# Copy the atlas/topo/ directory from this project into the fork
```

### Step 3: Run Level 1 tests (no GUDHI, instant feedback)
```bash
pytest tests/test_topo_pipeline.py -k "TestFiltration or TestLSCategory" -v
```

### Step 4: Run full test suite (with GUDHI)
```bash
pytest tests/test_topo_pipeline.py -v
```

### Step 5: Certify the existing torus atlas
```python
from atlas.topo.persistence import compute_persistence_diagrams, filter_by_lifetime, betti_numbers_at
from atlas.topo.filtration import sdf_solid_torus, clip_to_interior
from atlas.topo.ls_category import certify_atlas
import numpy as np

N = 32
lin = np.linspace(-1.8, 1.8, N)
gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
vals = sdf_solid_torus(coords, R=1.0, r=0.35).reshape(N, N, N).astype('float32')
grid = clip_to_interior(vals)

raw = compute_persistence_diagrams(grid, max_dimension=2)
filtered = filter_by_lifetime(raw, threshold=0.08, filtration_range=(float(grid.min()), 0.0))
betti = betti_numbers_at(filtered, t=-1e-6)
report = certify_atlas(M_actual=8, betti=betti,
                       quality_metrics={"g_fold": 0.0, "g_cov": 1.0, "g_ov": 0.026})
for msg in report["messages"]:
    print(msg)
```

Expected output:
```
PASS: M_actual=8 >= M_min=2
PASS: All quality gates satisfied.
```

---

## 8. Timeline Summary

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| 0: Fork setup | 1–2 | Reproducible environment, existing benchmarks pass |
| 1: Persistence pipeline | 3–8 | `atlas/topo/` module, Levels 1–2 tests green |
| 2: Chart count certification | 9–14 | `certify_atlas()` integrated, all benchmarks certified |
| 3: Dynamic monitoring | 15–22 | `TopologyMonitor` + `ChartSpawner`, Level 4 tests green |
| 4: Fracture benchmarks | 23–32 | Val-3 and Val-4 pass |
| 5: Release + paper | 33–40 | PyPI package, two journal submissions |
