# Benchmark Readiness Assessment

Status of the chart-based FEM solver against each Kamarei et al. (2026) challenge problem.

---

## Ready Benchmarks

### Biaxial Tension (Challenge 2) — READY

- **Single chart, SDF-filtered circular plate, 1331 nodes, 6000 tets**
- Stress-strain matches exact solution across 27 load steps
- Drucker-Prager nucleation at sigma = 28.09 MPa (expected 27.03, error 3.9%)
- The 3.9% error is due to load-step discretization (delta ~ 1 MPa per step)
- All stress components are correct: equi-biaxial gives sigma_xx = sigma_yy, sigma_zz ~ 0

### Uniaxial Strain (Patch Test) — READY

- **Single chart, identity decoder, 343 nodes**
- sigma_xx = 0.6781 vs expected 0.6731 (0.75% error)
- Uniform stress across all elements (std = 0)
- Newton converges in 3 iterations

### DCB Force-Displacement (Challenge 8) — READY (analytical)

- Beam theory + Griffith criterion produces exact F vs delta and a vs delta
- Crack growth follows a = (3Eh^3 delta^2 / (16 G_c))^{1/4}
- Not yet validated with chart FEM solve (would require multi-chart bar with pre-crack)

---

## Partially Ready Benchmarks

### Torsion (Challenge 3) — NOT READY

**Root cause:** St. Venant-Kirchhoff constitutive model produces spurious normal stresses under torsion due to geometric nonlinearity in the Green-Lagrange strain.

**Evidence:**
- Single-chart shear stress tau is exact (0.00% error)
- But the Cauchy stress tensor has normal components (sigma_xx ~ 12 MPa at tau ~ 34 MPa)
- These arise from the quadratic term in E = (F^T F - I)/2 when displacement involves rotation
- Drucker-Prager trips at tau = 33.7 MPa instead of sigma_ss = 44.4 MPa

**What's needed:**
1. **Linear strain formulation**: Use epsilon = sym(grad u) instead of Green-Lagrange E = (F^T F - I)/2 for the torsion benchmark (small strain assumption)
2. Or: **True linear elastic constitutive**: sigma = lambda tr(epsilon) I + 2 mu epsilon, bypassing the Piola-Kirchhoff route entirely
3. The Schwarz multi-chart coupling also needs more iterations (20 with omega=0.3 reaches 0.9% convergence, not yet tight enough)

### Mode-I Edge Crack (Challenge 4-like) — PARTIALLY READY

- K_I extraction from exact Williams data: 0.00% error
- CrackTipDecoder absorbs the sqrt(r) singularity
- FEM solver builds mesh and Newton converges with CrackTipDecoder
- Not yet validated end-to-end: chart FEM solve on cracked plate with K_I comparison to LEFM

---

## Not Ready

### Torsion Multi-Chart — NOT READY

Two issues compound:
1. Constitutive spurious normals (single-chart issue above)
2. Schwarz convergence is slow (0.9% residual after 20 iters with omega=0.3)

### Trousers Test (Challenge 9) — NOT FEASIBLE

Requires finite-strain Neo-Hookean, 3D large deformation (leg folding), Mode III tearing, near-incompressibility handling. Beyond current solver capabilities.

---

## Summary of Gaps

| Gap | Impact | Effort to Fix |
|-----|--------|---------------|
| Linear strain formulation (epsilon vs E_GL) | Torsion nucleation stress wrong | Low: add make_linear_elastic_small_strain() |
| Schwarz convergence rate | Multi-chart stress ~10% low at 20 iters | Medium: better interpolation or Krylov acceleration |
| End-to-end cracked plate FEM | K_I not validated from numerical solve | Medium: wire CrackTipDecoder into Schwarz solver |
| Neo-Hookean for FEM | PU elastomer benchmarks not possible | Medium: port from MPM constitutive |
| Near-incompressibility | Lambda/mu = 165 causes volumetric locking | Hard: mixed formulation needed |

## Validated Capabilities

| Capability | Evidence |
|------------|----------|
| Single-chart FEM with decoder Jacobian | 0.00%-0.75% error |
| Analytical decoders (TubeSector, Box, CrackTip) | All pass forward/inverse/Jacobian V&V |
| Schwarz multi-chart coupling with decoder inverse | Convergent with under-relaxation |
| Parallel chart solves (ThreadPoolExecutor) | 2-4x speedup |
| Drucker-Prager nucleation from FEM stress | Correct for biaxial (3.9% error) |
| CrackTipDecoder singularity absorption | sqrt(r) becomes linear in xi |
| Topology monitoring (GUDHI) | 1.1% overhead, detects domain splitting |
| Chart spawning from topology events | Full pipeline demonstrated |
