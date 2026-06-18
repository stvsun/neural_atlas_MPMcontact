# Stage 3: XFEM-Critic Scoring System

## Design Philosophy

This scoring system was designed from the perspective of a senior XFEM researcher reviewing the neural atlas fracture framework. It exposes **structural limitations** that a mature XFEM implementation (e.g., MOOSE/XFEM, Abaqus XFEM, Code_Aster) would not have.

The goal is NOT to penalize the neural atlas approach — it has genuine advantages (meshfree, topology-aware, differentiable). The goal is to **precisely quantify** what capabilities are missing so they can be addressed systematically.

## Test Catalog

### X1: Williams Enrichment Completeness (15 pts, crack challenges only)

**What XFEM does**: Enriches nodes within a radius of the crack tip with 4 branch functions:
- F1 = sqrt(r) * cos(theta/2)
- F2 = sqrt(r) * sin(theta/2)
- F3 = sqrt(r) * sin(theta/2) * sin(theta)
- F4 = sqrt(r) * cos(theta/2) * sin(theta)

These capture both the radial singularity AND the angular variation of the Williams field.

**What neural atlas does**: CrackTipDecoder absorbs only sqrt(r) via power-law radial mapping. Angular modes must be captured by the P1 basis functions, which are linear — they cannot represent cos(theta/2) exactly.

**Consequence**: Poor accuracy at theta != 0, pi/2 on coarse meshes. Interior nodes that are off-axis from the crack get progressively worse displacement estimates.

**How to fix**: Add explicit angular enrichment functions to the decoder, or implement DENNs-style displacement enrichment with Williams basis.

---

### X2: Crack-Face Traction-Free Enforcement (10 pts, crack challenges only)

**What XFEM does**: Heaviside enrichment H(x) = sign(phi(x)) creates a displacement jump across the crack face. Combined with the variational formulation, this automatically enforces sigma_nn = 0 on crack faces.

**What neural atlas does**: SDF-filtered mesh removes elements inside the crack slit. But elements adjacent to the crack face are standard P1 tets with no special treatment. Traction-free condition is only approximately satisfied through the FEM residual.

**Consequence**: Non-zero sigma_yy on elements near the crack face, especially on coarse meshes.

---

### X3: Mixed-Mode Capability (10 pts)

**What XFEM does**: Routinely extracts both K_I and K_II via the interaction integral (M-integral). This separates modes using auxiliary fields corresponding to pure Mode I and pure Mode II.

**What neural atlas does**: Only extracts K_I. The J-integral gives a scalar J = (K_I^2 + K_II^2) * (1-nu^2)/E but cannot separate the modes.

**Consequence**: Cannot analyze mixed-mode problems. Cannot predict crack kinking angle. Cannot model crack approaching an interface.

---

### X4: Curved Crack Path (10 pts)

**What XFEM does**: At each propagation step, computes the maximum hoop stress angle theta_c from K_I and K_II, rotates the crack direction, and advances the tip along the new direction. Level-set update tracks the evolving crack geometry.

**What neural atlas does**: propagate_crack() advances by a fixed increment in a fixed direction. The max_hoop_stress_angle function EXISTS in fracture_criteria.py but is NEVER CALLED by the propagation loop.

**Consequence**: Can only simulate straight cracks. Any problem with non-zero K_II or geometric asymmetry will produce wrong crack paths.

---

### X5: Displacement Discontinuity at Chart Overlaps (10 pts)

**What XFEM does**: Heaviside enrichment ensures the displacement field is discontinuous across the crack surface, continuous everywhere else. No special overlap treatment needed.

**What neural atlas does**: Schwarz interpolation in _interpolate_from_neighbors() uses standard barycentric interpolation, which assumes continuity. If two charts overlap across a crack face, the interpolation will produce the average of the two sides, not the correct discontinuous value.

**Consequence**: Incorrect artificial boundary conditions in Schwarz iterations when charts overlap across crack surfaces. This contributes to the DD convergence problems observed in C8.

---

### X6: Stiffness Matrix Conditioning (10 pts)

**What XFEM does**: Monitors condition number. Uses preconditioning (e.g., Cholesky, geometric multigrid). Warns when cond(K) > 10^6.

**What neural atlas does**: No conditioning check. Dense stiffness matrix assembly. No preconditioning.

**Consequence**: Silent convergence failures possible for highly anisotropic decoders or very fine near-tip meshes.

---

### X7: Integration Accuracy Near Singularity (10 pts)

**What XFEM does**: Subtriangulates elements cut by the crack. Uses enriched quadrature rules (7-13 Gauss points) near the crack tip. Maps the singularity analytically in the enrichment.

**What neural atlas does**: Standard 1-point (P1) or 4-point (P2) Gauss quadrature on all elements. SDF filtering removes entire elements inside the crack, leaving elements adjacent to the crack with standard quadrature.

**Consequence**: Quadrature error accumulates near the crack tip. Elements with high aspect ratios (from power-law mapping) may have inadequate integration.

---

### X8: Nucleation Mesh Independence (15 pts, non-crack challenges)

**What XFEM does (with phase-field or gradient damage)**: Nucleation location converges with mesh refinement due to nonlocal regularization. The damage variable has a finite width controlled by a length scale parameter.

**What neural atlas does**: Pointwise Drucker-Prager check on each element. The nucleation location is the element with the highest F_dp value, which depends on mesh density.

**Consequence**: Nucleation location can shift between meshes. No guarantee of convergence. Mesh-dependent failure patterns.

---

### X9: K_I Accuracy vs XFEM Benchmark (10 pts, crack challenges)

**What XFEM achieves**: K_I error < 1% on meshes with comparable DOF count (~1000-5000 nodes), using interaction integral.

**What neural atlas achieves**: K_I error of 26% (C8 DCB) using displacement correlation / J-integral.

This test simply compares the extracted K_I against the analytical value with XFEM-standard error bars.

---

## Expected Scores (Honest Assessment)

| Challenge | Max Pts | Expected Score | Assessment |
|-----------|---------|----------------|------------|
| C1 | 25 | 10-15 | Conditioning OK, nucleation mesh-dep |
| C2 | 25 | 10-15 | Same as C1 |
| C3 | 35 | 15-20 | DD discontinuity fails, conditioning OK |
| C4 | 85 | 15-25 | Williams angular fails, no K_II, no curved |
| C5 | 100 | 20-30 | All crack tests fail, some integration OK |
| C6 | 35 | 15-20 | Integration OK, nucleation mesh-dep |
| C7 | 25 | 10-15 | Same as C1 |
| C8 | 95 | 15-25 | K_I 26% err, no K_II, no curved, DD fails |
| C9 | 40 | 10-15 | No K_II, no curved, DD discontinuity fails |
| **TOTAL** | **465** | **~120-180** | **~26-39%** |

The low expected score (~30%) is BY DESIGN. It represents the gap between the neural atlas approach and a mature XFEM implementation. Each failed test is a roadmap item for future development.
