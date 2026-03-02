# PINN Coordinate-Chart Framework for 3D Complex Geometry

A **meshfree**, atlas-based Physics-Informed Neural Network (PINN) framework for solving
**forward and inverse PDEs on complex three-dimensional geometries** using overlapping
coordinate charts, multiplicative Schwarz domain decomposition, and automatic differentiation.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Formulation](#mathematical-formulation)
   - [Domain and SDF Representation](#1-domain-and-sdf-representation)
   - [Coordinate Charts and Atlas](#2-coordinate-charts-and-atlas)
   - [Chart Decoder and TNB Frame](#3-chart-decoder-and-tnb-frame)
   - [PDE Pullback to Chart Coordinates](#4-pde-pullback-to-chart-coordinates)
   - [Direct-Coordinate Mode (TNB)](#5-direct-coordinate-mode-tnb)
   - [Multiplicative Schwarz Decomposition](#6-multiplicative-schwarz-decomposition)
   - [Multi-Objective PINN Loss](#7-multi-objective-pinn-loss)
   - [PCGrad Gradient Surgery](#8-pcgrad-gradient-surgery)
   - [CompactChartNet — Voronoi Sub-Atlas](#9-compactchartnet--voronoi-sub-atlas)
   - [Chart-Partitioned SDF](#10-chart-partitioned-sdf)
3. [Key Algorithms — Pseudocode](#key-algorithms--pseudocode)
   - [Algorithm 1: Neural SDF Training](#algorithm-1-neural-sdf-training)
   - [Algorithm 1b: Chart-Partitioned SDF Training](#algorithm-1b-chart-partitioned-sdf-training)
   - [Algorithm 2: Atlas Construction](#algorithm-2-atlas-construction)
   - [Algorithm 3: Multiplicative Schwarz PINN Loop](#algorithm-3-multiplicative-schwarz-pinn-loop)
   - [Algorithm 4: PCGrad K=2 per Chart Step](#algorithm-4-pcgrad-k2-per-chart-step)
   - [Algorithm 5: CompactChartNet Forward Pass](#algorithm-5-compactchartnet-forward-pass)
4. [Numerical Examples](#numerical-examples)
   - [Example 1: Forward Poisson — 3D Ellipsoid / Star Domain](#example-1-forward-poisson--3d-ellipsoid--star-domain)
   - [Example 2: Forward Poisson — Stanford Rabbit (Schwarz)](#example-2-forward-poisson--stanford-rabbit-schwarz)
   - [Example 3: Inverse Neo-Hookean Elasticity — Torus (Original Atlas)](#example-3-inverse-neo-hookean-elasticity--torus-original-atlas)
   - [Example 4: Inverse Neo-Hookean Elasticity — Torus (Schwarz Dual)](#example-4-inverse-neo-hookean-elasticity--torus-schwarz-dual)
   - [Example 5: Inverse Elder-Like Flow — Stanford Rabbit](#example-5-inverse-elder-like-flow--stanford-rabbit)
   - [Example 6: Chart-Partitioned SDF — Stanford Bunny (In Progress)](#example-6-chart-partitioned-sdf--stanford-bunny-in-progress)
5. [Repository Structure](#repository-structure)
6. [Prerequisites and Installation](#prerequisites-and-installation)
7. [Geometry Preparation Pipeline](#geometry-preparation-pipeline)
8. [Running the Solvers](#running-the-solvers)
9. [Configuration Reference](#configuration-reference)
10. [Post-Processing and Visualization](#post-processing-and-visualization)
11. [Known Limitations](#known-limitations)
12. [TODO / Open Problems](#todo--open-problems)
13. [Citation](#citation)

---

## Overview

Solving PDEs on geometries as complex as a Stanford rabbit interior, a torus, or a
three-dimensional star domain is challenging for classical mesh-based methods because
mesh generation is expensive and mesh quality is hard to control. This codebase takes a
**fully meshfree** approach:

1. The geometry is represented implicitly by a learned **Signed Distance Function (SDF)**.
2. The SDF is used to construct a set of overlapping **coordinate charts** (a smooth atlas)
   that together cover the entire domain.
3. On each chart, a small PINN — either a dense MLP or a spatially-localized
   **CompactChartNet** (Voronoi sub-atlas) — solves a local version of the PDE in
   chart-local coordinates.
4. The local solvers are coupled through a **multiplicative Schwarz alternating procedure**
   that enforces value and normal-derivative continuity across chart boundaries.

**Key capabilities**

| Feature | Description |
|---------|-------------|
| Meshfree | No structured mesh; geometry encoded by neural SDF |
| Multi-chart atlas | Handles genus-0 and multiply-connected 3D bodies |
| Forward problems | Poisson equation as verification benchmark |
| Inverse problems | Recover spatially varying material parameters (shear/bulk modulus, permeability tensor) |
| Two PINN architectures | Dense MLP (`--pinn-arch mlp`) or CompactChartNet (`--pinn-arch compact`) |
| PCGrad gradient surgery | Resolves L_pde vs L_sup gradient conflicts per chart |
| Hardware-portable | CUDA GPU, Apple MPS (M1–M4), or CPU |
| Tested geometries | Stanford rabbit, torus (genus-1), 3D star, 3D ellipsoid |

---

## Mathematical Formulation

### 1. Domain and SDF Representation

Let **Ω ⊂ ℝ³** be the physical domain with boundary **∂Ω**. The geometry is
represented by a neural Signed Distance Function:

```
SDF_θ : ℝ³ → ℝ,   SDF_θ(x) < 0 ⟺ x ∈ Ω,   SDF_θ(x) = 0 ⟺ x ∈ ∂Ω
```

The SDF network is trained to satisfy the **Eikonal equation** with surface and normal
constraints:

```
‖∇ SDF_θ(x)‖ = 1                    (eikonal)
SDF_θ(x_s)   = 0  for x_s ∈ ∂Ω     (surface constraint)
∇ SDF_θ(x_s) = n_s                  (normal alignment)
```

Coordinates are normalized so that the bounding box of Ω maps to ≈ [−0.55, 0.55]³.

---

### 2. Coordinate Charts and Atlas

We cover Ω with **N overlapping coordinate charts** {(Ω_i, φ_i)}_{i=1}^N. Each chart consists of:

- **Chart seed** `s_i ∈ ∂Ω` — a surface point that anchors the chart
- **Support ball** `Ω_i = B(s_i, r_i) ∩ Ω` — the subdomain associated with chart *i*
- **Support radius** `r_i` — chosen so that the N balls achieve ≥99% coverage with ~20% overlap
- **Chart coordinate space** `ξ ∈ B̄(0, r_i) ⊂ ℝ³` — the local parameter domain

**Atlas construction** uses **Poisson-disk sampling** to place N seeds on ∂Ω with mutual
separation ≥ d_min, then assigns support radii to ensure full volumetric coverage:

```
Seeds: {s_i}_{i=1}^N ⊂ ∂Ω,   ‖s_i − s_j‖ ≥ d_min for all i ≠ j
Neighbors: 𝒩(i) = { j ≠ i : B(s_i, r_i) ∩ B(s_j, r_j) ≠ ∅ }
```

---

### 3. Chart Decoder and TNB Frame

Each seed `s_i` carries a local **Tangent-Normal-Binormal (TNB) frame**:

```
t₁ᵢ, t₂ᵢ  — tangent vectors at s_i (computed from SDF gradient)
nᵢ         — inward surface normal = −∇SDF_θ(s_i) / ‖∇SDF_θ(s_i)‖
Fᵢ = [t₁ᵢ | t₂ᵢ | nᵢ]  — orthonormal frame matrix (3×3)
```

The **chart decoder** φ_i maps local chart coordinates ξ ∈ ℝ³ to physical space x ∈ ℝ³:

```
φᵢ(ξ) = sᵢ  +  Fᵢ · ξ  +  Δφᵢ(ξ)
```

where `Δφᵢ` is a small nonlinear correction learned by a MLP with amplitude
`≈ 0.20 · tanh(raw_scale) · rᵢ ≈ 0.08` in normalized coordinates.
The Jacobian `Jᵢ(ξ) = ∂φᵢ/∂ξ ∈ ℝ^{3×3}` is used for coordinate transformations.

---

### 4. PDE Pullback to Chart Coordinates

The **Poisson equation** on Ω:

```
−Δₓ u(x) = f(x),    x ∈ Ω
u(x)       = g(x),    x ∈ ∂Ω
```

is pulled back to local chart coordinates via the chain rule.
Let **Jᵢ = ∂φᵢ/∂ξ** be the 3×3 Jacobian. Then the Laplacian transforms as:

```
Δₓ u = Σⱼₖ aⱼₖ(ξ) · ∂²u/∂ξⱼ∂ξₖ  +  lower-order terms

where the metric tensor is:  a(ξ) = |det Jᵢ| · (Jᵢ⁻ᵀ Jᵢ⁻¹)
```

The PDE residual in chart coordinates is:

```
R_pde(ξ) = Δₓ u_net(ξ) + f(φᵢ(ξ))
```

**Jacobian inversion** uses `torch.linalg.inv` (LU decomposition) in the backward pass
for numerical stability. SVD is reserved for condition-number diagnostics under
`torch.no_grad()` to avoid gradient instability when singular values coincide.

---

### 5. Direct-Coordinate Mode (TNB)

Since the TNB frame `Fᵢ` is **orthonormal** (det Jᵢ = 1, condition number κ = 1 for the
rigid part), the Laplacian simplifies exactly when the decoder residual Δφᵢ is dropped:

```
Δₓ u = Σⱼ ∂²u/∂ξⱼ²          (exact for orthogonal frames)
```

Enabled with `--direct-coord-pde`. This eliminates all Jacobian issues and permits
isolating the Schwarz convergence from decoder quality. All training and evaluation in
the Schwarz Poisson experiments (W1–W5, CompactChartNet) use this mode.

---

### 6. Multiplicative Schwarz Decomposition

The Schwarz method decomposes the global problem into local sub-problems on each
subdomain Ω_i, coupled through interface conditions on overlapping regions:

**Interface value continuity** (Robin-penalty):
```
L_iv(i,j) = E_{x ∈ Ω_i ∩ Ω_j} [ (uᵢ(ξᵢ(x)) − uⱼ(ξⱼ(x)))² ]
```

**Interface flux continuity** (normal-derivative matching):
```
L_if(i,j) = E_{x ∈ Ω_i ∩ Ω_j} [ (∇uᵢ · nᵢⱼ − ∇uⱼ · nᵢⱼ)² ]
```

**Multiplicative Schwarz sweep**: charts are updated sequentially (or in graph-colored
groups). At iteration `k`, chart i's PINN is retrained with all neighbor solutions
`{u_j}_{j ∈ 𝒩(i)}` held fixed:

```
uᵢᵏ⁺¹ = argmin_{uᵢ} [ Lᵢ(uᵢ; {uⱼᵏ}_{j ∈ 𝒩(i)}) ]    for i = 1, …, N
```

A **trust-region filter** rejects Schwarz updates that increase the global rel_l2 error
beyond a threshold; the learning rate is halved on rejection.

**Plateau detection** (W3): the stale counter increments when `rel_l2_eval` does not
improve by more than `plateau_tol`; training stops when `stale ≥ plateau_patience`.

---

### 7. Multi-Objective PINN Loss

On each chart *i* the total loss at Schwarz iteration *k* is:

```
Lᵢ  =  w_pde · L_pde  +  w_bc · L_bc  +  w_sup · L_sup
     +  w_iv  · Σⱼ∈𝒩(i) L_iv(i,j)
     +  w_if  · Σⱼ∈𝒩(i) L_if(i,j)
     +  w_h1  · Σⱼ∈𝒩(i) L_h1(i,j)
```

| Term | Expression | Role |
|------|-----------|------|
| **L_pde** | `E_ξ[ (Δₓuᵢ + f)² ]` | PDE residual over interior points |
| **L_bc** | `E_ξ[ (uᵢ(ξ_bc) − g(x_bc))² ]` | Dirichlet boundary condition |
| **L_sup** | `E_ξ[ (uᵢ(ξ) − u*(φ_rig(ξ)))² ]` | Manufactured-solution anchor (W2) |
| **L_iv** | `E_{Ω_i∩Ω_j}[ (uᵢ − uⱼ)² ]` | Interface value continuity |
| **L_if** | `E_{Ω_i∩Ω_j}[ (∂_n uᵢ − ∂_n uⱼ)² ]` | Interface flux continuity |
| **L_h1** | `E_{B_i∩B_j,vol}[ (uᵢ − uⱼ)² ]` | Volumetric H1 overlap penalty (W5) |

**Recommended weights** (validated in attempt19_w4):
`w_pde=5, w_bc=1, w_sup=0.5, w_iv=2, w_if=2, w_h1=0.5`

**PDE warmup**: `w_pde` is ramped from 0 to its target value over `pde_warmup_iters`
Schwarz iterations to allow boundary conditions to settle first.

---

### 8. PCGrad Gradient Surgery

When L_pde and L_sup have **conflicting gradients** (dot product < 0), naïve combined
minimization makes one objective worse. **PCGrad** (K=2) resolves this per-chart:

```
For each parameter θ:
  g_pde ← ∇_θ (w_pde · L_pde)       # backward pass 1
  g_sup ← ∇_θ (w_sup · L_sup)       # backward pass 2
  g_oth ← ∇_θ (L_bc + L_iv + L_if + L_h1)  # backward pass 3

  # Symmetric projection when conflicting:
  if g_pde · g_sup < 0:
    g_pde ← g_pde − (g_pde·g_sup / ‖g_sup‖²) · g_sup
    g_sup ← g_sup − (g_sup·g_pde_orig / ‖g_pde_orig‖²) · g_pde_orig

  θ.grad ← g_pde + g_sup + g_oth
```

PCGrad requires 3 backward passes per local step (vs 1 for the combined loss) but
enables simultaneous progress on both the PDE and supervision objectives, reducing
Schwarz oscillation amplitude. Enabled with `--use-pcgrad`.

---

### 9. CompactChartNet — Voronoi Sub-Atlas

Instead of a single dense MLP per chart, **CompactChartNet** uses M=9 small Tanh-MLPs
whose outputs are blended by a **softmax partition-of-unity (POU)**:

```
u_chart(ξ) = Σ_{m=0}^{M-1} φₘ(ξ) · uₘ(ξ − sₘ)

φₘ(ξ)  = softmax_m( −‖ξ − sₘ‖² / 2τ² )       # POU weights (C∞)

τ = τ_scale × r_i                               # POU bandwidth
sₘ  — sub-seed positions in local frame:
       s₀ = 0  (chart centre)
       s₁…s₈ = ±r_i/3 cube corners (8 vertices)
uₘ  — small Tanh MLP:  ℝ³ → ℝ,  width=32,  depth=2
```

**Key properties**:
- Smooth (C∞) everywhere → autograd Laplacian well-defined
- Spatially localized → gradients of `uₘ` concentrated near sub-seed `sₘ`
- Parameter count: 10,953 per chart vs 12,801 for the dense MLP (width=64, depth=4)
- **Critical hyper-parameter**: τ_scale must be ≥ 0.5 for full chart coverage;
  τ_scale=0.125 (τ ≈ 0.051) leaves isolated Gaussian neighbourhoods → pretrain diverges

**Monotone convergence property**: spatial localization means local PDE updates disturb
neighboring charts less than a dense MLP, yielding monotone Schwarz descent over 60
iterations vs oscillation with the dense MLP.

---

### 10. Chart-Partitioned SDF

The global SDF uses a single **anchor offset** δ for sign-anchor supervision across the
entire domain.  For geometries with thin features (Stanford Bunny ears ~5 mm), a single
offset forces a trade-off: small δ gives poor interior coverage, large δ places anchors
outside thin features → wrong signs → corrupted interface normals.

**Chart-partitioned SDF** resolves this by training N **local** SDF networks `{φᵢ}`, one
per atlas chart, each with a **geometry-adaptive offset**:

```
offset_i = f_factor × dist(seed_i, nearest surface point)
```

For the Stanford Bunny with f_factor=0.30 this gives:
- Ear-region charts: offset ≈ 0.2–1.8 mm (within ear thickness)
- Body charts: offset ≈ 4–15 mm (deep interior coverage)

vs the global offset of 38.9 mm that places anchors outside all thin features.

**Training procedure** (implemented in `src/train_sdf_chartwise.py`):

1. **Voxel SDT initialization** — for each chart i, compute the exact unsigned distance
   from a local query grid to the nearest PLY surface point via `cKDTree`.  Sign is
   determined by the outward-normal dot-product test:
   `sign(q) = +1 if (q - p_nearest) · n_nearest ≥ 0 else -1`.
   The local `SDFNetLocal` is pre-trained by MSE regression to this SDT.

2. **Eikonal reinitialization fine-tuning** — the Eikonal loss `mean((‖∇φᵢ‖ - 1)²)` is
   the steady-state form of the governing equation of the reinitialization PDE:
   `∂φ/∂t + sign(φ₀)(‖∇φ‖ - 1) = 0`.
   The zero level set is preserved by the surface loss `|φᵢ(x_s)|`.

3. **Partition-of-unity blending** for inference — the global SDF is reconstructed as:
   ```
   φ(x) = Σᵢ wᵢ(x) φᵢ(x) / Σᵢ wᵢ(x),   wᵢ(x) = exp(-‖x-seedᵢ‖²/2σᵢ²)
   ```
   This is smooth and fully differentiable through PyTorch autograd.

4. **Global adapter** — a single `SDFNet(128/6)` is optionally fitted to the blended
   chartwise SDF via regression (`--fit-adapter`), providing a drop-in replacement for
   the existing atlas builder and PINN code with no modifications required.

---

## Key Algorithms — Pseudocode

### Algorithm 1: Neural SDF Training

```
Input:  surface point cloud {x_s, n_s}, volume exterior points {x_ext}
Output: SDF network SDF_θ : ℝ³ → ℝ

Initialize θ (MLP, width=128, depth=6, Softplus activations)

for epoch = 1 to E_sdf:
    Sample x_s (surface), x_int (interior random), x_ext (exterior random)

    loss_surface  = mean( SDF_θ(x_s)² )                    # SDF = 0 on surface
    loss_eikonal  = mean( (‖∇SDF_θ(x)‖ − 1)² )            # |∇SDF| = 1 everywhere
    loss_normal   = mean( ‖∇SDF_θ(x_s) − n_s‖² )          # normal alignment
    loss_exterior = mean( ReLU(−SDF_θ(x_ext)) )            # sign anchor

    loss = w_surf*loss_surface + w_eik*loss_eikonal
         + w_norm*loss_normal + w_ext*loss_exterior

    Update θ via Adam(loss, lr=8e-4)

return SDF_θ
```

---

### Algorithm 1b: Chart-Partitioned SDF Training

```
Input:  PLY surface point cloud {x_s, n_s} (or PCA-estimated normals),
        atlas {seed_i, support_radius_i}, N charts
Output: SDFNetChartwise — blended partition-of-unity SDF

for i = 1 to N:

    # Step 1: Geometry-adaptive offset
    d_i ← dist(seed_i, nearest PLY point)
    offset_i ← f_factor × d_i     # f_factor=0.30 recommended
    # e.g. ear-region: d≈0.6mm → offset≈0.2mm; body: d≈14mm → offset≈4.2mm

    # Step 2: Voxel SDT initialization on local grid
    for q in local_grid(seed_i, radius=1.5 × support_radius_i):
        (d_unsigned, k) ← cKDTree({x_s}).query(q)
        sign ← +1 if (q - x_s[k]) · n_s[k] ≥ 0 else -1
        sdt[q] ← sign × d_unsigned

    # Step 3: Pre-train SDFNetLocal_i to match SDT
    Initialize SDFNetLocal_i (width=64, depth=4, tanh)
    for epoch = 1 to E_pretrain:   # default 400
        L_sdt = mean( (φᵢ(q) - sdt[q])² )
        Update via Adam(L_sdt)

    # Step 4: Eikonal fine-tuning (reinitialization PDE at steady state)
    for epoch = 1 to E_finetune:   # default 3000
        Sample x_s_local ← local PLY points near seed_i
        Sample x_eik     ← random points in ball B(seed_i, 1.5 × support_radius_i)

        L_surface ← mean( |φᵢ(x_s_local)| )              # zero-level-set
        L_eikonal ← mean( (‖∇φᵢ(x_eik)‖ - 1)² )          # |∇φ|=1 (reinit PDE)
        L_normal  ← mean( 1 - ∇φᵢ(x_s_local)·n_s_local ) # normal alignment
        L_sign    ← softplus loss at x_s ± offset_i × n_s # adaptive sign anchor

        L_i = w_surf·L_surface + w_eik·L_eikonal + w_norm·L_normal + w_sign·L_sign
        Update SDFNetLocal_i via Adam(L_i)

# Assemble: Gaussian PoU blending
SDFNetChartwise.forward(x):
    w_i(x) = exp(-‖x-seed_i‖² / (2 support_radius_i²))
    return Σ_i w_i(x) φᵢ(x) / Σ_i w_i(x)   # differentiable via autograd

# Optional: fit a single global adapter SDFNet(128/6) by regression on chartwise values
```

**Script**: `src/train_sdf_chartwise.py`

---

### Algorithm 2: Atlas Construction

```
Input:  surface point cloud {x_s}, SDF_θ, N (number of charts)
Output: seeds {sᵢ}, frames {Fᵢ}, radii {rᵢ}, neighbor sets {𝒩(i)}

1. SEED SELECTION (Poisson-disk):
   Start with random seed s₁ ∈ {x_s}
   While |seeds| < N:
     Candidate c ~ Uniform({x_s})
     If min_{sᵢ already placed} ‖c − sᵢ‖ ≥ d_min:
       Accept c as new seed sₖ

2. TNB FRAME CONSTRUCTION (per seed sᵢ):
   nᵢ ← −∇SDF_θ(sᵢ) / ‖∇SDF_θ(sᵢ)‖          # inward normal
   t₁ᵢ ← unit vector ⊥ nᵢ  (stable Gram-Schmidt)
   t₂ᵢ ← nᵢ × t₁ᵢ
   Fᵢ ← [t₁ᵢ | t₂ᵢ | nᵢ]

3. SUPPORT RADIUS:
   rᵢ ← radius such that ball B(sᵢ, rᵢ) covers all interior points
         within the Voronoi cell of sᵢ, plus overlap_target fraction

4. NEIGHBOR GRAPH:
   𝒩(i) ← { j ≠ i : B(sᵢ, rᵢ) ∩ B(sⱼ, rⱼ) ≠ ∅ }

5. QUALITY GATES:
   Check: coverage ≥ 99%, max_chart_rel_l2_init < 20%, zero foldover

return {sᵢ, Fᵢ, rᵢ, 𝒩(i)}
```

---

### Algorithm 3: Multiplicative Schwarz PINN Loop

```
Input:  atlas {sᵢ, Fᵢ, rᵢ, 𝒩(i)}, pretrained u_nets = {uᵢ_θ}
        loss weights, local_steps, max_iters, plateau_patience
Output: trained u_nets, best_rel_l2 checkpoint

# --- Pretrain phase ---
BC warm-start: minimize L_bc over 300 epochs for all charts
Interior supervised pretrain: minimize L_sup over 1000 epochs (all charts)

# --- PDE warmup schedule ---
for k = 1 to pde_warmup_iters:
    w_pde_k ← w_pde * k / pde_warmup_iters

best_rel_l2 ← ∞,  stale ← 0

# --- Schwarz outer loop ---
for k = 1 to max_schwarz_iters:

    # Update each chart in sequence
    for i = 1 to N:
        Sample ξ_pde (interior of Ω_i, SDF rejection)
        Sample ξ_bc  (near ∂Ω ∩ B(sᵢ, rᵢ))
        Sample ξ_if  (interface Ω_i ∩ Ω_j for each j ∈ 𝒩(i))

        for step = 1 to local_steps:
            Compute L_i(uᵢ_θ; {uⱼ_θ_fixed}_{j∈𝒩(i)})

            if use_pcgrad:
                Run Algorithm 4 (PCGrad)
            else:
                L_i.backward(); clip grads; Adam step

    # Evaluate global error
    rel_l2 ← eval_rel_l2(u_nets, x_interior_50K, u_true)

    # Trust-region filter
    if rel_l2 > best_rel_l2_so_far * trust_factor:
        Restore u_nets from best_rel_l2 checkpoint
        lr ← lr * 0.7          # decay on rejection
        continue

    # Update best checkpoint
    if rel_l2 < best_rel_l2:
        best_rel_l2 ← rel_l2
        Save u_nets → best_rel_l2.pt
        stale ← 0
    else:
        stale ← stale + 1

    # Plateau detection (W3)
    if stale ≥ plateau_patience:
        break

return u_nets, best_rel_l2.pt
```

---

### Algorithm 4: PCGrad K=2 per Chart Step

```
Input:  u_net_i, batch (ξ_pde, ξ_bc, ξ_sup, ξ_iv, ξ_if, ξ_h1)
        neighbor solutions {uⱼ} (detached)

# Three separate forward+backward passes (no retain_graph needed)

g_pde ← ∇_θ (w_pde · L_pde)          # Pass 1: PDE only
g_sup ← ∇_θ (w_sup · L_sup)          # Pass 2: supervision only
g_oth ← ∇_θ (L_bc + L_iv + L_if + L_h1)  # Pass 3: boundary + interface terms

# PCGrad symmetric projection
g_pde_orig ← copy(g_pde)
for each parameter p:
    dot ← g_pde[p] · g_sup[p]
    if dot < 0:
        # Project g_pde onto orthogonal complement of g_sup
        g_pde[p] ← g_pde[p] − dot / ‖g_sup[p]‖² · g_sup[p]
        # Project g_sup onto orthogonal complement of g_pde_orig
        g_sup[p] ← g_sup[p] − (g_sup[p]·g_pde_orig[p]) / ‖g_pde_orig[p]‖² · g_pde_orig[p]

# Accumulate into θ.grad and step
θ.grad ← g_pde + g_sup + g_oth
clip_grad_norm(θ, max_norm=1.0)
Adam.step()
```

---

### Algorithm 5: CompactChartNet Forward Pass

```
Input:  ξ ∈ ℝ^{N×3}  (N query points in local chart frame)
        sub-seeds {sₘ}_{m=0}^{M-1} ∈ ℝ^{M×3}  (fixed, precomputed)
        sub-nets {uₘ_θ}  (M small Tanh-MLPs)
        τ (POU bandwidth)

# Step 1: Compute POU weights
dist² ← ‖ξ.unsqueeze(1) − sₘ.unsqueeze(0)‖²    # shape (N, M)
logits ← −dist² / (2τ²)                          # shape (N, M)
φ      ← softmax(logits, dim=1)                   # shape (N, M), sums to 1

# Step 2: Evaluate each sub-net (with optional freezing in exclusive zones)
for m = 0 to M-1:
    vₘ ← uₘ_θ(ξ − sₘ)           # translate input to sub-seed frame; shape (N,1)
    if frozen[m]:
        vₘ ← vₘ.detach()         # no gradient through frozen sub-nets

vals ← stack([v₀, …, v_{M-1}], dim=1)            # shape (N, M, 1)

# Step 3: Weighted sum (partition-of-unity blend)
u ← (φ.unsqueeze(-1) * vals).sum(dim=1)          # shape (N, 1)

return u
```

---

## Numerical Examples

### Example 1: Forward Poisson — 3D Ellipsoid / Star Domain

**Description**: Baseline verification of the chart decoder + PINN pipeline on
analytic domains (no Schwarz coupling needed). A single chart covers the full domain.

**Domain**: 3D ellipsoid `x²/a² + y²/b² + z²/c² ≤ 1` (a=1.2, b=0.9, c=0.7);
3D star domain with 5-pointed star cross-section.

**PDE**: Poisson equation with manufactured solution
`u*(x,y,z) = sin(πx)sin(πy)sin(πz)`, RHS `f = 3π²u*`.

**Method**: Single MLP PINN, sphere-mapping coordinate chart, no Schwarz coupling.

**Results**:

| Domain | Metric | Value |
|--------|--------|-------|
| 3D Ellipsoid | rel_L2 | < 1% after 5000 epochs |
| 3D Star domain | rel_L2 | < 2% after 5000 epochs |

**Scripts**: `src/pinn_3d_ellipsoid_mapped_sphere.py`, `src/run_poisson_star3d_mapped.py`

---

### Example 2: Forward Poisson — Stanford Rabbit (Schwarz)

**Description**: Poisson equation inside the Stanford rabbit volumetric interior,
solved by a 12-chart Schwarz PINN. Two PINN architectures are benchmarked.

**Domain**: Stanford rabbit volumetric interior, SDF-normalized coordinates ≈ [−0.55, 0.55]³.
Atlas: 12 interior ball-charts, support radius r ≈ 0.41, ~20% overlap, 100% coverage.

**PDE**: `−Δu = 3π² sin(πx₁)sin(πx₂)sin(πx₃)` with Dirichlet BC `u = 0` on ∂Ω.
Evaluation: `rel_l2 = ‖u_PINN − u*‖₂ / ‖u*‖₂` over 50K interior reference points.

**Validated improvements** (W-series):

| ID | Improvement | Effect |
|----|------------|--------|
| W1 | Consistent PDE operator in eval (TNB-frame Laplacian) | 28× speedup, −14% rel_l2 |
| W2 | Manufactured-solution anchor during Schwarz (`w_sup=0.5`) | −14.4% rel_l2 vs W1 |
| W3 | Plateau detection tracks rel_l2 (not composite score) | Correct stale counter |
| W4 | PCGrad K=2 gradient surgery (L_pde vs L_sup) | **−18.8% rel_l2 vs W3** |
| W5 | Stronger coupling (`w_if=2.0`, H1 volumetric overlap) | max_error −36%, uniformity ↑ |

**Architecture A — Dense MLP (W1+W2+W3+W4+W5)**:

```
PINN:       width=64, depth=4 Tanh MLP per chart, 12,801 params/chart
Algorithm:  PCGrad K=2, 15 local Adam steps per Schwarz iter, lr=2e-4
Schwarz:    23 iterations total, best at iter 8
```

| Metric | Value |
|--------|-------|
| **rel_l2** | **2.531%** |
| max_error | 5.303% |
| mean_interface_residual | 0.001511 |
| Runtime | 404 s (~7 min, Apple M4) |
| Schwarz iters | 23 (plateau patience=15) |

**Architecture B — CompactChartNet (Voronoi sub-atlas, τ_scale=0.5)**:

```
PINN:       9 sub-nets per chart (M=9, width=32, depth=2, Tanh), 10,953 params/chart
POU:        φₘ = softmax(−‖ξ−sₘ‖²/2τ²), τ = 0.5 × 0.41 ≈ 0.205
Algorithm:  PCGrad K=2 + CompactChartNet, lr=2e-4
Schwarz:    60 iterations (monotone descent), best at iter 56
```

| Metric | Value |
|--------|-------|
| **rel_l2** | **2.207%** ← all-time best |
| max_error | 6.783% |
| mean_interface_residual | 0.001979 |
| Runtime | 3659 s (~61 min, Apple M4) |
| Schwarz iters | 60 (hit max_schwarz_iters, plateau never fired) |
| Convergence | Monotone — no oscillation |

**Per-chart "U-net" (`u_nets`) used by the solver**:

In the Poisson rabbit code, `u_nets` denotes the list of local chart solution networks
(one per chart). This is the field model optimized inside Schwarz iterations.

- Implementation location: `experiments/run_poisson_rabbit_atlas_schwarz.py`
- Runtime selection: `--pinn-arch {mlp,resnet,compact}`
- Stored in checkpoints as chart `u_states` and recoverable by warm-start flags
  (`--init-u-checkpoint`, optional `--u-remap-json`)

Note: in this script, `u_nets` is a per-chart PINN collection (MLP/ResNet/CompactChartNet),
not a convolutional image U-Net encoder-decoder.

**Benchmark progression**:

| Run | Architecture / Features | rel_l2 | max_error | Runtime |
|-----|------------------------|--------|-----------|---------|
| attempt14b | Dense MLP baseline (broken plateau) | 4.31% | 10.86% | 10,133 s |
| attempt15b_w1 | Dense MLP + W1 | 3.698% | **4.107%** | 364 s |
| attempt16_w2 | Dense MLP + W1+W2 | 3.165% | 4.993% | 335 s |
| attempt17_w3 | Dense MLP + W1+W2+W3 | 3.117% | 8.485% | 328 s |
| attempt18_w5 | Dense MLP + W1+W2+W3+W5 | 3.631% | 5.450% | 371 s |
| attempt19_w4 | Dense MLP + W1+W2+W3+W4+W5 | 2.531% | 5.303% | 404 s |
| **attempt20c_compact** | **CompactChartNet + all W1–W5** | **2.207%** | 6.783% | 3659 s |

**Script**: `experiments/run_poisson_rabbit_atlas_schwarz.py`
**Best checkpoint**: `runs/attempt20c_compact/`

---

### Example 3: Inverse Neo-Hookean Elasticity — Torus (Original Atlas)

**Description**: Recover shear modulus μ and bulk modulus K of an incompressible
neo-Hookean solid from traction observations on the torus surface.

**Domain**: Torus (major radius R=1.0, minor radius r=0.35, genus-1 topology).
Atlas: 8 coordinate charts covering the torus surface.

**Forward model** (incompressible neo-Hookean):
```
P = μ F − μ F⁻ᵀ + K(J−1) J F⁻ᵀ      (1st Piola-Kirchhoff stress)
Div P = 0   in Ω,     P·N = t   on ∂Ω_N
```

**Inverse problem**: Given boundary traction observations `t_obs` on ∂Ω,
find scalar parameters μ and K.

**True parameters**: μ = 1.8, K = 25.0

**Method**: Joint PINN optimization — minimize
`L = w_pde · L_pde + w_bc · L_traction + w_param · ‖{μ,K}‖_reg`
with μ, K as trainable scalars initialized near the true values.

**Results** (run: `torus_inverse_mps`, 300 epochs, Apple M4 MPS):

| Parameter | True | Estimated | Rel. Error |
|-----------|------|-----------|-----------|
| μ (shear modulus) | 1.800 | 1.7999998 | **9.3 × 10⁻⁶ %** |
| K (bulk modulus) | 25.00 | 25.0000057 | **2.3 × 10⁻⁵ %** |
| traction rel_l2 | — | — | 2.35 × 10⁻⁷ |

| Metric | Value |
|--------|-------|
| Epochs | 300 |
| Runtime | 201 s |
| Composite score | **3.96 × 10⁻⁷** |

**Script**: `src/run_torus_inverse_neohookean_atlas.py`

---

### Example 4: Inverse Neo-Hookean Elasticity — Torus (Schwarz Dual)

**Description**: Same inverse problem on the torus but with a Schwarz-coupled
multi-chart solver supporting two observation types: displacement observations
and traction observations.

**True parameters**: μ = 1.8, K = 25.0

#### Variant A — Displacement observations

| Parameter | True | Estimated | Rel. Error |
|-----------|------|-----------|-----------|
| μ | 1.800 | 1.8000000 | **4.1 × 10⁻¹² %** |
| K | 25.00 | 25.2498 | 1.00% |

| Metric | Value |
|--------|-------|
| Best epoch | 405 |
| Displacement rel_l2 | 0.174% |
| Traction rel_l2 | 0.959% |
| Runtime | 38.6 s |
| Target met | ✅ |

#### Variant B — Traction observations

| Parameter | True | Estimated | Rel. Error |
|-----------|------|-----------|-----------|
| μ | 1.800 | 1.7938 | **0.345%** |
| K | 25.00 | 25.135 | 0.541% |

| Metric | Value |
|--------|-------|
| Best epoch | 300 |
| Traction rel_l2 | 0.580% |
| Displacement rel_l2 | 0.336% |
| Runtime | 76.9 s |
| Target met | ✅ |

**Script**: `src/run_torus_inverse_neohookean_schwarz_dual.py`

---

### Example 5: Inverse Elder-Like Flow — Stanford Rabbit

**Description**: Recover the **anisotropic permeability tensor** K(x) of a porous medium
inside the Stanford rabbit from sparse pressure and concentration observations.
This is a coupled inverse problem for Darcy flow + species transport.

**Forward model**:
```
−∇·(K(x)∇p) = 0                         (pressure, Darcy flow)
−∇·(D∇c − Ra · K(x)c∇p) = 0             (concentration, Elder-like transport)
```

**Permeability parameterization**:
```
K(x) = k₀ · R(q) Λ Rᵀ(q)
```
where `k₀` is the isotropic scale, `R(q)` is a rotation matrix parameterized by
quaternion `q`, and `Λ = diag(λ₁, λ₂, λ₃)` is a diagonal anisotropy matrix.

**Inverse problem**: Given sparse `(p, c)` observations, recover `k₀, q, Λ`.

**Domain**: Stanford rabbit volumetric interior, 12 atlas charts.

**Results** (run: `rabbit_inverse_elder_globalfield_small`, CPU float64):

| Metric | Value |
|--------|-------|
| k₀ relative error | **3.0%** |
| Eigenvalue rel. error (mean) | 2.97% |
| Eigenvalue rel. error (max) | 6.22% |
| Axis orientation error (mean) | **2.70°** |
| Axis orientation error (max) | 3.35° |
| Pressure field rel_l2 | 0.0% (teacher-student locked) |
| Runtime | 642 s |
| Target met | ✅ |

**Script**: `src/run_rabbit_inverse_elder_atlas_schwarz.py`

---

### Example 6: Chart-Partitioned SDF — Stanford Bunny (In Progress)

**Motivation**: All prior Stanford Bunny Poisson PINN runs diverged (best: 28.1% rel-L²
at Schwarz iter 1, interface flux increasing every iteration).  Root cause diagnosis
identified the **global SDF sign quality** as the bottleneck: the global SDF v3 used a
fixed anchor offset of 38.9 mm, which placed sign-supervision anchors outside the ~5 mm
bunny ears, causing 31% sign errors.  With corrupt interface normals on 31% of surface
points, the Schwarz iteration diverges regardless of atlas quality.

**Proposed fix** (implemented, pending experimental validation):

| Component | Global SDF (v3) | Chart-partitioned SDF |
|-----------|----------------|-----------------------|
| Networks | 1 global SDFNet (128/6) | 12 local SDFNetLocal (64/4) |
| Anchor offset | 38.9 mm (fixed) | 0.2–4.2 mm (adaptive per chart) |
| Ear-safe charts | 0/12 | 10/12 |
| SDT initialization | None (random Xavier) | KD-tree SDT pre-training (400 epochs) |
| Eikonal fine-tuning | Standard | Reinitialization PDE formulation |
| Inference | Single forward pass | Gaussian PoU blending |
| Drop-in compatibility | Yes | Yes (via global adapter `--fit-adapter`) |

**Key result** (from adaptive offset measurement):
- Chart 5 (ear region): seed is only **0.6 mm** from the nearest surface point →
  adaptive offset = 0.2 mm → sign anchors safely inside the 5 mm ear volume
- Chart 0 (body region): seed is **14.1 mm** from surface → adaptive offset = 4.2 mm →
  good deep interior coverage

**Status**: Script implemented (`src/train_sdf_chartwise.py`), validation run pending.

**Expected outcome** (if sign_error < 0.05):
- Interface normals at surface points corrected
- Schwarz interface flux should decrease monotonically
- Expected rel-L² on Stanford Bunny < 10% (vs current 28.1%)

**Script**: `bash scripts/experimental/run_chartwise_sdf_example.sh`

**Detailed results**: `RESULTS.md § Chart-partitioned SDF`

---

## Repository Structure

```
PINN_coordinate_chart_3Dgeometry/
│
├── src/                              # Core verified solvers
│   ├── pinn_3d_ellipsoid_mapped_sphere.py     # Forward Poisson on 3D ellipsoid
│   ├── pinn_gradient_surgery.py               # 2D PINN with PCGrad (reference impl)
│   ├── run_poisson_star3d_mapped.py           # Forward Poisson on 3D star domain
│   ├── run_torus_inverse_neohookean_atlas.py  # Inverse neo-Hookean on torus (Example 3)
│   ├── run_torus_inverse_neohookean_schwarz_dual.py  # Schwarz dual torus inverse (Example 4)
│   ├── run_rabbit_inverse_elder_atlas_schwarz.py     # Inverse Elder flow on rabbit (Example 5)
│   ├── run_rabbit_inverse_neohookean_mapped.py       # Neo-Hookean inverse on rabbit
│   ├── train_sdf_rabbit.py                    # Global neural SDF training (Algorithm 1)
│   ├── train_sdf_chartwise.py                 # Chart-partitioned SDF training (Algorithm 1b, Example 6)
│   ├── train_mapping_from_sdf.py              # Sphere-to-domain mapping network
│   └── export_rabbit_elder_inverse_paraview.py
│
├── experiments/                      # Research-stage solvers
│   ├── run_poisson_rabbit_atlas_schwarz.py    # ← MAIN Schwarz Poisson solver (Example 2)
│   ├── compact_chart_net.py                   # CompactChartNet (Algorithm 5)
│   ├── train_rabbit_atlas.py                  # Atlas decoder + mask network training
│   ├── build_rabbit_atlas_poissondisk.py      # Atlas seed selection (Algorithm 2)
│   ├── build_rabbit_atlas_volumetric.py       # Volumetric interior-seed atlas build
│   ├── run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz.py  # TI Arruda-Boyce inverse
│   ├── postprocess_rabbit_poisson_dense_fields.py
│   ├── export_rabbit_atlas_paraview.py
│   ├── export_rabbit_error_paraview.py
│   └── experimental_ideas/                    # Archived exploratory variants
│
├── configs/                          # YAML configuration files
│   ├── rabbit_atlas_poisson.yaml
│   ├── rabbit_inverse_elder_atlas.yaml
│   └── rabbit_inverse_ti_arruda_boyce.yaml
│
├── docs/
│   ├── chatgpt52_paper_starter.md             # Paper outline (draft)
│   └── chatgpt52_handoff_mindmap.md           # Research mindmap
│
├── scripts/
│   ├── successful/                    # Canonical reproduction scripts
│   │   ├── run_poisson_rabbit_best.sh          # Reproduces 2.21% Poisson (procedural rabbit)
│   │   └── build_bunny_sdf_atlas_v3.sh         # Reproduces SDF v3 + atlas v3 + decoders
│   └── experimental/                  # Failed / in-progress attempts
│       ├── run_chartwise_sdf_example.sh        # Example 6: chart-partitioned SDF pipeline
│       ├── fix_decoders.sh            # Failed: standard atlas (gate failed, pde=165k)
│       ├── fix_decoders_v2.sh         # Partial: high overlap weight (gate=True, PINN diverges)
│       └── diagnostic_supervised.sh   # Failed: manufactured supervision negligible vs PDE
│
├── RESULTS.md                         # Comprehensive results summary and analysis
├── schwarz_poisson_attempt_status.md  # Detailed Schwarz Poisson attempt log
├── pinn_only_improvement_plan.md      # W-series and architecture roadmap
└── runs/                              # Saved checkpoints and metrics
    ├── attempt19_w4/                  # Dense MLP best run (rel_l2=2.531%)
    ├── attempt20c_compact/            # CompactChartNet best run (rel_l2=2.207%)
    ├── bunny_sdf_v3/                  # Stanford Bunny SDF v3 (sign=0.311)
    ├── atlas_bunny_vol_v3_hiW/        # Stanford Bunny atlas decoders (gate=True, overlap=0.018)
    ├── torus_inverse_mps/             # Torus inverse best run (score=3.96e-7)
    └── rabbit_inverse_elder_globalfield_small/  # Elder inverse best run
```

---

## Prerequisites and Installation

### Python Version

Python ≥ 3.9 (3.10 or 3.11 recommended).

### Required Libraries

```bash
# CUDA (NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS):
pip install torch torchvision
```

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Neural networks, autograd, linear algebra |
| `numpy` | ≥ 1.24 | Array operations, point cloud I/O |
| `matplotlib` | ≥ 3.7 | Convergence plots |
| `scipy` | ≥ 1.10 | (optional) Poisson-disk sampling utilities |

No mesh generation library required — the method is fully meshfree.

### Hardware

| Platform | Notes |
|----------|-------|
| **NVIDIA GPU (CUDA)** | Recommended; float64 supported; `--amp` reduces memory |
| **Apple M1–M4 (MPS)** | Supported; use `--dtype float32`; some linalg ops use CPU fallback |
| **CPU** | Works; float64 default; slow for large runs |

Pass `--device auto` (default) to automatically select CUDA → MPS → CPU.

### Clone

```bash
git clone https://github.com/stvsun/PINN_coordinate_chart_3Dgeometry.git
cd PINN_coordinate_chart_3Dgeometry
```

---

## Geometry Preparation Pipeline

### Stage 0: Point Cloud

```bash
# Stanford rabbit: download PLY, convert to NPZ
python - <<'EOF'
import open3d as o3d, numpy as np
mesh = o3d.io.read_triangle_mesh("bunny/reconstruction/bun_zipper.ply")
pcd  = mesh.sample_points_uniformly(30000)
np.savez("rabbit_pointcloud.npz",
         points=np.asarray(pcd.points),
         normals=np.asarray(pcd.normals))
EOF
```

Analytic domains (torus, ellipsoid, star) generate their own point clouds internally.

### Stage 1: Train Neural SDF

```bash
python src/train_sdf_rabbit.py \
    --point-cloud rabbit_pointcloud.npz \
    --output-dir  runs/sdf_rabbit \
    --epochs      5000  --width 128  --depth 6 \
    --lr 8e-4     --device auto  --seed 42
```

Output: `runs/sdf_rabbit/rabbit_sdf.pt`

### Stage 2: Build Atlas

```bash
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint runs/sdf_rabbit/rabbit_sdf.pt \
    --output-dir     runs/atlas_vol \
    --n-charts       12 \
    --overlap-target 0.20 \
    --seed           42
```

Output: `runs/atlas_vol/rabbit_atlas_data.npz`, `runs/atlas_vol/rabbit_atlas_meta.json`

### Stage 3: Train Atlas Decoder

```bash
python experiments/train_rabbit_atlas.py \
    --atlas-data  runs/atlas_vol/rabbit_atlas_data.npz \
    --sdf-checkpoint runs/sdf_rabbit/rabbit_sdf.pt \
    --volumetric \
    --output-dir  runs/atlas_vol_trained \
    --epochs      3000  --width 64  --depth 4 \
    --lr 8e-4     --device auto  --seed 42
```

Output: `runs/atlas_vol_trained/rabbit_atlas_trained.pt`, `runs/atlas_vol_trained/rabbit_atlas_gate_report.json`

---

## Running the Solvers

### Forward Poisson — Dense MLP + PCGrad (W1–W5 recommended)

```bash
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data          runs/atlas_vol/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/atlas_vol_trained/rabbit_atlas_trained.pt \
    --atlas-meta          runs/atlas_vol/rabbit_atlas_meta.json \
    --sdf-checkpoint      runs/sdf_rabbit/rabbit_sdf.pt \
    --volumetric-atlas \
    --output-dir          runs/my_poisson_run \
    --device auto --seed 42 \
    \
    --pinn-arch mlp \
    --pinn-width 64 --pinn-depth 4 \
    --lr 2e-4 \
    --direct-coord-pde \
    --interior-pretrain-epochs 1000 \
    --interior-pretrain-bc-weight 0.5 \
    --interior-pretrain-grad-weight 0.5 \
    --bc-pretrain-epochs 300 \
    --bc-pretrain-grad-weight 0.05 \
    --bc-pretrain-interface-weight 0.2 \
    --pde-warmup-iters 50 --plateau-patience 15 \
    --w-manufactured-supervision 0.5 --plateau-use-rel-l2 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --w-overlap-h1 0.5 --overlap-h1-batch 64 \
    --checkpoint-policy best_rel_l2 \
    --use-pcgrad
```

### Forward Poisson — CompactChartNet (monotone convergence)

```bash
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data          runs/atlas_vol/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/atlas_vol_trained/rabbit_atlas_trained.pt \
    --atlas-meta          runs/atlas_vol/rabbit_atlas_meta.json \
    --sdf-checkpoint      runs/sdf_rabbit/rabbit_sdf.pt \
    --volumetric-atlas \
    --output-dir          runs/my_compact_run \
    --device auto --seed 42 \
    \
    --pinn-arch compact \
    --compact-n-subseed 9 \
    --compact-sub-width 32 --compact-sub-depth 2 \
    --compact-tau-scale 0.5 \
    --lr 2e-4 \
    --max-schwarz-iters 60 \
    --exclusive-finetune-steps 200 \
    --exclusive-finetune-lr 5e-5 \
    --direct-coord-pde \
    --interior-pretrain-epochs 1000 \
    --interior-pretrain-bc-weight 0.5 \
    --interior-pretrain-grad-weight 0.5 \
    --bc-pretrain-epochs 300 \
    --pde-warmup-iters 50 --plateau-patience 15 \
    --w-manufactured-supervision 0.5 --plateau-use-rel-l2 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --w-overlap-h1 0.5 --overlap-h1-batch 64 \
    --checkpoint-policy best_rel_l2 \
    --use-pcgrad
```

### Inverse Neo-Hookean Elasticity (Torus)

```bash
# Original atlas (fastest, most accurate)
python src/run_torus_inverse_neohookean_atlas.py \
    --output-dir  runs/torus_inv \
    --device auto --dtype float32 --seed 42 --epochs 300

# Schwarz dual — displacement observations
python src/run_torus_inverse_neohookean_schwarz_dual.py \
    --inverse-mode displacement \
    --output-dir   runs/torus_schwarz_disp \
    --device auto  --seed 42

# Schwarz dual — traction observations
python src/run_torus_inverse_neohookean_schwarz_dual.py \
    --inverse-mode traction \
    --output-dir   runs/torus_schwarz_trac \
    --device auto  --seed 42
```

### Inverse Elder-Like Flow (Rabbit)

```bash
python src/run_rabbit_inverse_elder_atlas_schwarz.py \
    --atlas-data     runs/atlas_vol/rabbit_atlas_data.npz \
    --atlas-checkpoint  runs/atlas_vol_trained/rabbit_atlas_trained.pt \
    --output-dir     runs/rabbit_elder \
    --device cpu --dtype float64 --seed 42
```

### Simpler Baselines (No Schwarz)

```bash
python src/pinn_3d_ellipsoid_mapped_sphere.py  --output-dir runs/ellipsoid --device auto
python src/run_poisson_star3d_mapped.py        --output-dir runs/star      --device auto
```

---

## Configuration Reference

Key flags for `run_poisson_rabbit_atlas_schwarz.py` (parser defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--atlas-data` | required | Path to `rabbit_atlas_data.npz` |
| `--atlas-checkpoint` | required | Path to `rabbit_atlas_trained.pt` |
| `--atlas-meta` | none | Optional atlas metadata (`rabbit_atlas_meta.json`) |
| `--pinn-arch` | `mlp` | `mlp` / `resnet` / `compact` |
| `--pinn-width` | 64 | Hidden width (dense MLP) |
| `--pinn-depth` | 4 | Hidden layers (dense MLP) |
| `--compact-n-subseed` | 9 | Sub-seeds in CompactChartNet |
| `--compact-sub-width` | 32 | Sub-net hidden width |
| `--compact-tau-scale` | 0.125 | POU bandwidth (use ≥ 0.5 in practice) |
| `--lr` | `8e-4` | Adam learning rate |
| `--max-schwarz-iters` | 60 | Schwarz outer iterations |
| `--local-steps` | 15 | Adam steps per chart per Schwarz iter |
| `--pde-warmup-iters` | 10 | Ramp `w_pde` over this many iters |
| `--plateau-patience` | 15 | Stop after this many stale Schwarz iters |
| `--plateau-use-rel-l2` | off | W3: track rel_l2_eval for plateau |
| `--direct-coord-pde` | off | W1: bypass decoder Jacobian |
| `--w-manufactured-supervision` | 0.0 | W2: supervision anchor weight |
| `--w-interface-value` | 0.8 | Interface value coupling weight |
| `--w-interface-flux` | 0.2 | Interface flux coupling weight |
| `--w-overlap-h1` | 0.0 | W5: H1 volumetric overlap weight |
| `--use-pcgrad` | off | W4: PCGrad K=2 gradient surgery |
| `--volumetric-atlas` | off | Enables volumetric overlap sampling and SDF interior sampling |
| `--use-sdf-sampling` | off | SDF-based interior collocation sampling |
| `--checkpoint-policy` | `last` | `last` / `best_rel_l2` / `best_target` / `best_flux` / `best_score` / `best_pareto` |
| `--exclusive-finetune-steps` | 0 | Post-Schwarz exclusive-zone fine-tuning |
| `--exclusive-finetune-lr` | `2e-4` | Learning rate for exclusive-zone finetune |

---

## Post-Processing and Visualization

```bash
# Export solution to ParaView (.vtu / .vtp)
python experiments/export_rabbit_atlas_paraview.py \
    --solution-npz        runs/my_poisson_run/rabbit_poisson_schwarz_solution.npz \
    --output-dir          runs/my_poisson_run/paraview

# Dense post-processed pressure/velocity fields
python experiments/postprocess_rabbit_poisson_dense_fields.py \
    --atlas-data          runs/atlas_vol/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/atlas_vol_trained/rabbit_atlas_trained.pt \
    --solver-checkpoint   runs/my_poisson_run/rabbit_poisson_schwarz_best_rel_l2.pt \
    --output-dir          runs/my_poisson_run/dense_field

# Elder inverse — export to ParaView
python src/export_rabbit_elder_inverse_paraview.py \
    --checkpoint  runs/rabbit_elder/best_target.pt \
    --atlas-data  runs/atlas_vol/rabbit_atlas_data.npz \
    --output-dir  runs/rabbit_elder/paraview
```

**Output files per solver run:**

| File | Contents |
|------|----------|
| `*_checkpoint.pt` | Per-chart PINN weights + optimizer state |
| `*_best_rel_l2.pt` | Weights at best global rel_l2 checkpoint |
| `*_best_target.pt` / `*_best_flux.pt` / `*_best_score.pt` | Alternative checkpoint-policy snapshots |
| `*_solution.npz` | Merged field arrays used by exporters (`points`, `u_pred`, `u_true`, `u_error`, `chart_weights`, etc.) |
| `*_metrics.json` | Per-iteration: rel_l2, PDE residual, BC loss, interface residuals, runtime |
| `*_history.json` | Full time-series of all logged quantities |
| `*_curves.png` | Matplotlib convergence plot |
| `*.vtu` / `*.vtp` | ParaView export (requires running export script) |

---

## Known Limitations

1. **CompactChartNet runtime**: 9× slower than dense MLP due to M=9 sub-nets in the
   double-autograd computation graph for the Laplacian. Reduce `--compact-n-subseed` to
   4–5 for a 2× speedup at some loss of expressiveness.

2. **CompactChartNet exclusive-zone fine-tuning**: With `tau_scale=0.5`, the safety margin
   is too large and all sub-nets are frozen. Requires `safety_factor ≤ 1.5` or a
   point-coverage-based masking approach to have any effect.

3. **Schwarz oscillation (dense MLP)**: After reaching the best iterate (~iter 8 for W4+W5),
   the dense MLP Schwarz oscillates at 3–6%. Hard-capping with `--max-schwarz-iters 10`
   or tighter `--plateau-patience 5` helps preserve the optimum.

4. **MPS float32 only**: Apple MPS does not reliably support float64. Inverse problems
   are run on CPU with float64 for accuracy. Forward Poisson runs on MPS with float32.

5. **TI Arruda-Boyce identifiability**: The transversely-isotropic material inverse
   problem has near-degenerate parameter directions; single traction load families may
   be insufficient for four-parameter identification.

6. **Seam artifacts in gradient fields**: The partition-of-unity blending is C⁰ but not
   C¹; gradient-based observables (Darcy velocity = −K∇p) show visible seams at chart
   boundaries. Use `weighted_detached` blending policy for visualization.

7. **Global SDF sign quality on thin-featured geometry**: The global SDF uses a single
   anchor offset across all charts.  For the Stanford Bunny (~5 mm ears), an offset large
   enough for body coverage (38.9 mm) places anchors outside the ears, causing 31% sign
   errors.  This corrupts interface normals in all 12 charts simultaneously and causes
   Schwarz iteration to diverge.  The **chart-partitioned SDF** (Example 6) addresses
   this directly via geometry-adaptive per-chart offsets (0.2–4.2 mm for the bunny).

---

## TODO / Open Problems

The following tasks are planned or in progress, roughly ordered by priority.

### High Priority — Stanford Bunny Poisson convergence

- [ ] **[EXP] Run `scripts/experimental/run_chartwise_sdf_example.sh`** (~5.5 h)
  Train chart-partitioned SDF on Stanford Bunny, rebuild atlas, run Poisson PINN.
  Target metric: `sign_error < 0.05` (vs current 0.311 for global SDF v3).

- [ ] **[EVAL] Verify Schwarz convergence with chartwise SDF**
  Monitor `interface_flux` per iteration — it should decrease (not increase as before).
  Record best `rel_L2` and compare to current 28.1% (iter=1 checkpoint).

- [ ] **[TUNE] If chartwise PINN still diverges**: add cross-chart consistency loss
  in overlap regions (`mean((φᵢ(x) - φⱼ(x))²)` for chart pairs sharing overlap points).
  This is already stubbed in `train_sdf_chartwise.py` as `compute_overlap_consistency`.

- [ ] **[BENCH] If chartwise PINN converges**: document results in `RESULTS.md`,
  add a row to the `scripts/successful/` table, and run postprocessing to generate
  publication figures (`paraview/bunny_poisson_chartwise/`, `figures/bunny_chartwise/`).

### Medium Priority — SDF quality improvements

- [ ] **[CODE] Normal estimation for PLY without per-vertex normals**
  The current PCA k-NN implementation uses a centroid-based global orientation flip,
  which misclassifies ~50% of normals on concave surfaces (e.g., between the ears).
  Better approach: use a minimum spanning tree (MST) propagation algorithm (Hoppe et al.)
  to propagate consistent orientation from a seed point.

- [ ] **[CODE] Multi-scale anchor offsets in global SDF**
  Train the global `SDFNet` with a mix of three offsets (e.g., 0.02, 0.15, 0.50)
  rather than a single value.  This is an alternative to the full chartwise SDF approach
  and simpler to integrate with the existing pipeline.

- [ ] **[CODE] Replace centroid-estimated normals with global SDF gradient**
  When `--sdf-checkpoint` is provided to `train_sdf_chartwise.py`, use
  `n(x) = ∇φ(x) / ‖∇φ(x)‖` at PLY surface points as the normal estimate.
  The global SDF v3 already has `final_normal = 0.032` (good gradient alignment),
  so its gradients are reliable normals even if the sign is wrong.

### Medium Priority — PINN architecture

- [ ] **[CODE] Replace manufactured solution with PLY-mesh BCs**
  Currently the Poisson PINN enforces `u=sin(πx₁)sin(πx₂)sin(πx₃)` on `∂Ω` using the
  SDF zero-crossing for BC sampling.  A wrong-sign SDF places BC samples inside Ω.
  Using the actual PLY mesh vertices as BC sample points removes the SDF-sign dependence
  for boundary conditions entirely.

- [ ] **[CODE] Supervised warmup before Schwarz iterations**
  Add a pre-Schwarz phase where all charts are initialized to the manufactured solution
  via MSE regression.  This gives a good global starting point before the Schwarz
  coupling begins, avoiding the cold-start divergence observed in the Stanford Bunny runs.

- [ ] **[EXP] CompactChartNet on Stanford Bunny**
  Once the sign quality is resolved (via chartwise SDF or PLY-mesh BCs), benchmark
  the CompactChartNet architecture on the Stanford Bunny domain.
  Hypothesis: monotone Schwarz convergence carries over from the procedural rabbit.

### Low Priority — Paper and documentation

- [ ] **[DOC] Finalize paper draft** (`docs/chatgpt52_paper_starter.md`)
  Key sections remaining: Example 6 results table, comparison with global SDF baseline,
  discussion of adaptive offset derivation.

- [ ] **[DOC] Generate publication figures for Stanford Bunny**
  Once chartwise PINN converges, run `postprocessing/render_3d_figures.py` to produce:
  - `figures/3d/bunny_poisson_error_surface.png`
  - `figures/3d/bunny_poisson_charts.png`
  - `figures/3d/bunny_adaptive_offsets.png`

- [ ] **[CODE] Git push `PINN_coordinate_chart_3Dgeometry` main branch**
  The sub-repo is 7 commits ahead of `origin/main` (as of 2026-03-01).
  Push after the chartwise SDF experiment validates.

- [ ] **[CODE] Vectorize `estimate_normals_pca` in `train_sdf_chartwise.py`**
  The current per-point `np.linalg.eigh` loop is O(N·k²) and takes ~0.8s for 36k
  points.  A batched `np.linalg.eigh` on the full covariance stack would be 10–20× faster.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{sun2026pinn_atlas,
  title   = {Inverse Problems in Three-Dimensional Multiply-Connected Bodies
             via Neural Network Coordinate Charts},
  author  = {Sun, W.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

*Maintained by Steve Sun. Contributions and bug reports welcome via GitHub Issues.*
