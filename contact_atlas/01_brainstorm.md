# Contact Mechanics for the Neural Atlas Framework: Research Brainstorm

> **Historical note.** Design-history brainstorm. Some options below reference FEM modules
> (`ChartFEMSolver`, `SchwarzFEMSolver`, `ChartVectorFEMSolver`, `robin_schwarz.py`, …) that now
> live under `archive/solvers/fem/`; the framework was ultimately built on the **MPM** path
> (see `docs/contact_theory_manual.md`). Kept for rationale, not as an active spec.

## Context

This document brainstorms the techniques required to implement contact mechanics (self-contact and multi-body) within the neural atlas FEM/MPM framework. The framework uses neural SDFs for geometry, learned chart decoders for coordinate mappings, and Schwarz domain decomposition for multi-chart coupling. Contact introduces inequality-constrained inter-body coupling — a fundamentally new interface condition type beyond the existing Schwarz continuity conditions.

---

## 1. Contact Detection

### 1.1 SDF-Based Gap Function (Core Primitive)

For bodies A and B with neural SDFs `phi_A`, `phi_B`, the **gap function** for a point `x_B` on body B's surface is:

```
g(x_B) = phi_A(x_B)       (g < 0 => penetration)
n(x_B) = grad(phi_A) / |grad(phi_A)|   (contact normal)
```

**Why this works here:** The neural SDF (`SDFNet` in `atlas/sdf/train_sdf.py`) is trained with Eikonal loss (`|grad phi| ~ 1`), giving a smooth, differentiable gap function with unit-gradient normals. A single forward pass + autograd gives both gap and normal. No surface mesh extraction needed.

**Limitations:** SDF accuracy (~1e-3 relative error from training) sets a floor on gap precision. Near the medial axis `|grad phi|` dips below 1, degrading normal quality. For large deformation, the reference-config SDF must be composed with the inverse deformation map (see Section 7.3).

**Refs:** Kolev & Shakib (2003), Springer — level-set interface treatment.

### 1.2 Broad-Phase: Chart Bounding Volume Culling

Each chart has seed point, frame, and support radius. Chart pair (i on body A, j on body B) can only be in contact if `dist(seed_i, seed_j) < r_i + r_j + margin`. This is O(M_A * M_B) — cheap, and mirrors the neighbor-graph construction in `SchwarzFEMSolver._build_neighbors`.

### 1.3 Narrow-Phase: Batched SDF Evaluation

After broad-phase, evaluate `phi_A(x_B)` at candidate points:
- **FEM:** Physical boundary nodes (`phys_bc_nodes` from `ChartFEMSolver._classify_nodes`)
- **MPM:** Particles near surface (`|phi_B(x_p)| < threshold`, pushed to physical space via `particles.push_to_physical`)

All candidates batched into one GPU forward pass through the SDF network.

### 1.4 Self-Contact Detection

The chart structure provides a natural spatial partition. Two charts on the *same* body that are **not neighbors** in the overlap graph but whose deformed physical images become proximate indicate self-contact. Detection: track deformed boundary node positions across all charts; flag non-neighbor chart pairs with `min_dist < threshold`.

### 1.5 Multi-Body Contact

Each body has its own atlas + SDF. A **contact manager** sits above the per-body Schwarz solvers and mediates inter-body coupling by cross-evaluating SDFs. The existing per-body solver internals are unchanged.

---

## 2. Contact Formulations

### 2.1 Penalty Method (Simplest, Phase 1)

Contact pressure: `p_n = epsilon_n * max(0, -g)`. Adds forces to FEM residual / MPM P2G without new unknowns.

- **FEM:** Contact nodal force `f_c = p_n * n * area_node` added to `ChartVectorFEMSolver` residual. Tangent: `K_c = epsilon_n * n (x) n * area`.
- **MPM:** Contact body force added during P2G in `particle_to_grid` (`transfers.py`), analogous to gravity.
- **Pros:** Minimal code invasion. No saddle-point system.
- **Cons:** Finite penetration. Time step restriction in explicit MPM: `dt ~ 1/sqrt(epsilon_n)`. Condition number scales with `epsilon_n`.

**Refs:** Wriggers (2006), Ch. 5; Kikuchi & Oden (1988), SIAM.

### 2.2 Lagrange Multiplier Method (Exact)

Introduce multiplier `lambda` (contact pressure) with complementarity: `g >= 0, lambda >= 0, lambda * g = 0`. Extends system to saddle-point: `[K, B; B^T, 0][u; lambda] = [F; 0]`.

- **Pros:** Exact constraint; direct pressure access.
- **Cons:** Indefinite system; needs specialized solver; incompatible with explicit MPM; active set tracking required.

**Refs:** Laursen (2002), Springer; Alart & Curnier (1991), CMAME.

### 2.3 Augmented Lagrangian / Uzawa (Phase 2)

Iterate: `lambda_{k+1} = max(0, lambda_k + epsilon_n * g_k)`, then solve penalty subproblem with current `lambda_k`. Nests inside Schwarz iteration. Moderate `epsilon_n ~ E` avoids ill-conditioning while achieving exact enforcement after 3-10 Uzawa steps.

**Refs:** Simo & Laursen (1992), Computers & Structures; Fortin & Glowinski (1983).

### 2.4 Mortar Method (Non-Matching Meshes)

Weak contact enforcement via intermediate mortar surface. Integral: `int(lambda * [u_A - u_B] dA) = 0`. Natural here because charts inherently have non-matching grids. The PoU blending (MaskNet softmax) provides ready-made mortar weights. Projection uses SDF: closest point on A = `x - phi_A(x) * n_A(x)`.

- **Pros:** Passes contact patch test; variationally consistent.
- **Cons:** Complex implementation; mortar integral construction in 3D with curved surfaces.

**Refs:** Puso & Laursen (2004), CMAME.

### 2.5 Nitsche's Method (Variationally Consistent Penalty)

Adds consistency terms `{sigma * n}` and jump `[u]` with stabilization `gamma ~ E/h`. **Structurally identical to Robin Schwarz** — the Robin parameter `robin_delta` in `robin_schwarz.py` plays the same role as `gamma`. Extending Robin Schwarz to contact-Robin requires replacing equality (`u_A = u_B`) with inequality (`u_n >= 0`).

- **Pros:** Optimal convergence rates; no extra unknowns; close to existing Robin code.
- **Cons:** Requires stress evaluation at contact surface; mesh-dependent stabilization.

**Refs:** Chouly, Hild & Renard (2013), Math. Comp.; Wriggers & Zavarise (2008), Comp. Mech.

### 2.6 Summary: Mapping to Schwarz

| Formulation | Schwarz integration | Subproblem type |
|---|---|---|
| Penalty | Add contact forces to chart residual | Standard FEM with extra forces |
| Augmented Lagrangian | Uzawa outer loop wrapping Schwarz | Penalty subproblem + multiplier update |
| Mortar | PoU-weighted mortar integral at inter-body interfaces | Mixed system per chart |
| Nitsche/Robin | Inequality Robin transmission condition | Robin subproblem with projection |
| Active set | Dirichlet BC at active contact nodes | Standard FEM with augmented BC set |

---

## 3. Normal and Tangential Contact

### 3.1 Signorini Condition (Normal)

KKT complementarity: `g >= 0, p_n >= 0, p_n * g = 0`. Gap linearization for Newton: `delta_g = n . delta_u_B + curvature_terms` (curvature from SDF Hessian `H_phi = d^2 phi/dx^2`, computable via second autograd pass).

### 3.2 Coulomb Friction (Tangential)

Friction cone: `|t_t| <= mu * |t_n|`. Stick: `v_t = 0`. Slip: `t_t = -mu * |t_n| * v_t / |v_t|`.

**Contact frame from SDF:** Normal `n = grad(phi)/|grad(phi)|`. Tangent basis via `_frame_from_normal` in `chart_spawn.py` (Gram-Schmidt). Decompose: `delta_u_n = (delta_u . n) * n`, `delta_u_t = delta_u - delta_u_n`.

Regularized friction for numerical tractability: `t_t = -mu * |t_n| * v_t / max(|v_t|, epsilon_reg)`.

### 3.3 SDF-Derived Contact Normals

```python
x_t = x.requires_grad_(True)
phi = sdf_net(x_t)
grad_phi = torch.autograd.grad(phi, x_t, create_graph=True)[0]
n = grad_phi / grad_phi.norm(dim=1, keepdim=True).clamp(min=1e-12)
```

Same computation as Eikonal normal loss during SDF training. Analytically smooth (C-inf from tanh activations). Near sharp edges/corners, MLP smooths out features — a limitation for knife-edge contact.

---

## 4. Integration with Schwarz Domain Decomposition

### 4.1 Contact as Special Schwarz Interface Condition

Standard Schwarz: continuity (`u_A = u_B`) at chart overlaps. Contact: inequality (`g = (x_B + u_B - x_A - u_A) . n >= 0`). A chart on body A in contact with chart on body B receives **contact force BCs** rather than continuity BCs. Extends `_compute_schwarz_bc` with a `_compute_contact_bc` method.

### 4.2 Robin Contact Transmission (Phase 3)

Replace Robin equality `du/dn + delta * u = data` with Robin complementarity:
```
du/dn + delta * u >= 0,   lambda >= 0,   lambda * (du/dn + delta * u) = 0
```

The `_lambda` and `_g` arrays in `RobinSchwarzSolver` become contact multiplier and gap. Update: `lambda^{n+1} = max(0, lambda^n + epsilon * g^n)` (projection onto non-negative cone).

**Refs:** Bayada, Sabil & Sassi (2008), Appl. Math. Lett.; Dostal (2009), Springer.

### 4.3 Active Set Within Schwarz

At each Schwarz iteration: solve chart subproblem with active contact nodes as Dirichlet (`u_n = 0`), inactive as free. After solve, update active set: add penetrating nodes, remove tensile-pressure nodes. Uses existing `phys_bc_nodes` BC machinery via row/column elimination.

**Risk:** Active set oscillation between Schwarz iterations. Mitigate with damping/relaxation.

**Refs:** Hintermuller, Ito & Kunisch (2002), SIAM J. Optim.

---

## 5. Integration with MPM

### 5.1 Background Grid Contact (Single Grid)

Standard MPM: particles from different materials sharing a grid node automatically interact. Works within a single chart but **not** across chart grids.

### 5.2 Multi-Material MPM Contact (Bardenhagen et al.)

After P2G for both bodies on a shared grid:
1. Identify nodes with mass from both A and B
2. Compute separate velocities `v_A = p_A/m_A`, `v_B = p_B/m_B`
3. If `(v_A - v_B) . n < 0` (approaching): apply no-slip or friction
4. G2P uses corrected grid velocities

**Challenge:** Chart-local grids are misaligned. Requires shared "contact grid" or direct cross-chart interpolation.

**Refs:** Bardenhagen, Brackbill & Sulsky (2000), CMAME; Huang et al. (2011), IJNME.

### 5.3 Penalty Contact via SDF (Recommended for MPM)

For each particle `p_B` near body A: compute `g = phi_A(x_pB)`, apply `f_contact = epsilon_n * max(0, -g) * n_A`. Added as body force during P2G, analogous to gravity in `particle_to_grid`. Force in physical space pulled back to chart-local grid via `J^{-T}`.

**Pros:** No shared grid needed. Each body's solver independent. Natural for chart-local MPM design.
**Cons:** Time step restriction `dt ~ 1/sqrt(epsilon_n)`.

---

## 6. Integration with FEM

### 6.1 Contact Elements at Chart Boundaries

Boundary faces (triangles from boundary tets) identified via `phys_bc_nodes`. Contact surface elements contribute forces and stiffness from gap function evaluation at face quadrature points.

### 6.2 Node-to-Surface (NTS) with SDF Projection (Recommended)

For each "slave" node `x_B` on body B's boundary: project onto body A's surface via SDF. Closest point: `x_A = x_B - phi_A(x_B) * n_A(x_B)`. Smooth, differentiable, no explicit triangle intersection.

**Segment-to-segment (STS)** is more accurate but vastly more complex. The SDF projection gives NTS the smoothness advantages typically associated with STS.

**Refs:** Zavarise & Wriggers (1998); Fischer & Wriggers (2005), Comp. Mech.

---

## 7. SDF-Based Techniques

### 7.1 SDF as Gap + Normal Oracle

- Gap: `g(x) = phi_A(x)` — one forward pass
- Normal: `n(x) = grad(phi_A)(x)` — one autograd call
- Gap rate: `dg/dt = grad(phi_A) . v_B`
- Eikonal property `|grad phi| ~ 1` means gap has unit gradient in normal direction

### 7.2 Level-Set Contact (Kolev-Shakib)

Contact surface = zero level set of `phi_A` restricted to body B's surface. Find by evaluating `phi_A` at B's boundary nodes and interpolating zero crossings.

### 7.3 Deformation-Composed SDF (Key Technique)

The reference-config SDF must be updated for large deformation. Options:

| Approach | How | Cost | Accuracy |
|---|---|---|---|
| **(a) Re-train** | Re-train SDF at each step | Very high | Best |
| **(b) Explicit tracking** | Track deformed boundary node positions | Low | No smooth SDF field |
| **(c) Deformation-composed** | `phi_deformed(x) = phi_ref(decoder^{-1}(x) - u(decoder^{-1}(x)))` | Moderate (decoder inversion via Newton in `common/geometry.py`) | Good |
| **(d) Particle-based** (MPM) | Reconstruct SDF from deformed particle positions via KD-tree | Moderate | Approximate |

**Recommendation:** (c) for FEM, (d) for MPM.

---

## 8. Topology Considerations

### 8.1 Contact as Topological Event

Two disjoint bodies (two H0 components) touching = H0 merge. Monitor combined SDF `min(phi_A, phi_B)` via `TopologyMonitor`. A new H0 death in the persistence diagram signals contact. This is the **topological dual of fracture** (H0 split).

**Limitation:** Grid-based homology (16^3-32^3) gives coarse spatial resolution. Suitable for event detection every N steps, not every Newton iteration.

### 8.2 Chart Spawning at Contact Interfaces

Extend `ChartSpawner` with `edge_type='contact'`:
1. Localize contact (where `phi_A ~ 0` on B's surface)
2. Seed two charts on opposite sides (one from A's atlas, one from B's)
3. Warm-start decoders from nearest existing charts
4. Schwarz treats contact edges with inequality constraints

### 8.3 Separation Detection

H0 split in combined SDF = contact loss. At algorithmic level: nodes with tensile contact pressure removed from active set. Contact charts deactivated when contact lost everywhere.

---

## 9. Large Deformation Contact

### 9.1 Lagrangian Formulation Choice

- **FEM:** Total Lagrangian (consistent with `ChartVectorFEMSolver`). Reference SDF used directly; deformation composed via `F`.
- **MPM:** Updated Lagrangian (particles track deformed positions). Deformed SDF from particle cloud.

### 9.2 Surface Area Evolution (Nanson's Formula)

`dA_deformed = |det(F)| * ||F^{-T} n_ref|| * dA_ref`. The stabilized Jacobian ops in `common/geometry.py` provide `F^{-1}` and `det(F)` needed for contact force transformation.

### 9.3 Chart Quality Monitoring

Monitor condition number `kappa = s_max/s_min` (from `stabilized_jacobian_ops`) near contact surfaces. Trigger chart refinement/respawning if `kappa > threshold`.

---

## 10. Algorithmic Considerations

### 10.1 Semi-Smooth Newton

NCP function: `C(g, lambda) = lambda - max(0, lambda + c*g)`. Generalized derivative:
- Active (`lambda + c*g > 0`): `dC/dg = -c`, `dC/dlambda = 0`
- Inactive: `dC/dg = 0`, `dC/dlambda = 1`

Adds contact rows to Newton Jacobian in `ChartVectorFEMSolver`. Superlinear convergence near solution.

### 10.2 GPU Acceleration

SDF evaluation is naturally batched on GPU (already torch-based throughout MPM solver). Main bottleneck: autograd for SDF gradients (`create_graph=True`) is 2-3x slower than forward pass. For thousands of contact points, GPU batching provides major speedup over CPU point-by-point.

### 10.3 Parallel Contact Detection

Chart-level parallelism: contact detection for different chart pairs is independent. Reuse `ThreadPoolExecutor` from `RobinSchwarzSolver`.

---

## 11. Recommended Phased Implementation

| Phase | Technique | Key Files Modified |
|---|---|---|
| **1** | Penalty contact + SDF gap function | `chart_vector_fem.py`, `transfers.py`, new `contact_manager.py` |
| **2** | Augmented Lagrangian (Uzawa) | `schwarz_fem.py`, `robin_schwarz.py` |
| **3** | Robin contact transmission | `robin_schwarz.py` |
| **4** | Topology-aware contact detection | `monitor.py`, `chart_spawn.py` |
| **5** | Coulomb friction | `contact_manager.py` |

---

## 12. Open Research Questions

1. **Schwarz convergence with inequality constraints:** Does multiplicative still converge faster than Robin for contact? How many Schwarz iterations needed?
2. **Neural SDF accuracy for contact:** Is ~1e-3 Eikonal error sufficient for stiff contact (`epsilon_n ~ E`)?
3. **Contact chart spawning:** Beneficial to spawn dedicated contact charts (like crack charts), or sufficient to use existing charts with contact forces?
4. **Self-contact + chart overlaps:** Deformation-aware neighbor updates needed when non-neighbor charts collide.
5. **Topology-driven detection:** Can bottleneck distance threshold in `TopologyMonitor` reliably distinguish contact from noise at 16^3 grid resolution?
6. **MPM force pull-back:** Is the contact force conservative in xi-space after Jacobian pull-back through the chart decoder?
