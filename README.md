# PINN Coordinate Chart Framework for 3D Complex Geometry

A meshfree, atlas-based Physics-Informed Neural Network (PINN) framework for solving **forward and inverse PDEs on complex three-dimensional geometries** using overlapping coordinate charts, Schwarz domain decomposition, and automatic differentiation.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Theory](#mathematical-theory)
   - [Coordinate Charts and Atlas](#coordinate-charts-and-atlas)
   - [PDE Formulation in Chart Coordinates](#pde-formulation-in-chart-coordinates)
   - [Multiplicative Schwarz Coupling](#multiplicative-schwarz-coupling)
   - [Multi-Objective PINN Loss](#multi-objective-pinn-loss)
3. [Numerical Examples](#numerical-examples)
   - [Manufactured Solution (Poisson)](#manufactured-solution-poisson)
   - [Forward Poisson on the Rabbit Geometry](#forward-poisson-on-the-rabbit-geometry)
   - [Inverse Neo-Hookean Elasticity (Torus)](#inverse-neo-hookean-elasticity-torus)
   - [Inverse Elder-Like Flow (Rabbit)](#inverse-elder-like-flow-rabbit)
4. [Repository Structure](#repository-structure)
5. [Prerequisites and Installation](#prerequisites-and-installation)
6. [Geometry Preparation](#geometry-preparation)
   - [Training a Neural SDF](#training-a-neural-sdf)
   - [Building the Atlas](#building-the-atlas)
   - [Training the Atlas Decoder Networks](#training-the-atlas-decoder-networks)
7. [Running the Solvers](#running-the-solvers)
   - [Forward Poisson (Schwarz)](#forward-poisson-schwarz)
   - [Inverse Neo-Hookean Elasticity](#inverse-neo-hookean-elasticity)
   - [Inverse Elder Flow](#inverse-elder-flow)
8. [Configuration Reference](#configuration-reference)
9. [Recent Improvements (M1–M8)](#recent-improvements-m1m8)
10. [Post-Processing and Visualization](#post-processing-and-visualization)
11. [Known Limitations and Open Problems](#known-limitations-and-open-problems)
12. [Citation](#citation)

---

## Overview

Solving PDEs on geometries as complex as a rabbit surface, a torus with a fillet, or a three-dimensional star domain is challenging for classical mesh-based methods because mesh generation is expensive and mesh quality is hard to control. This codebase takes a **meshfree** approach: the geometry is represented implicitly by a learned **Signed Distance Function (SDF)**, which is then used to define a set of overlapping **coordinate charts** that together cover the entire domain. On each chart, a small PINN solves a local version of the PDE; the charts are coupled together through a **Schwarz alternating procedure** that enforces value and flux continuity across chart boundaries.

**Key capabilities:**
- Meshfree: no structured mesh is required
- Multi-chart atlas: naturally handles genus-0 and multiply-connected 3D bodies
- Forward problems: Poisson equation as a verification benchmark
- Inverse problems: recover spatially varying material parameters (shear/bulk modulus, anisotropic permeability, transversely-isotropic Arruda–Boyce parameters) from sparse observations
- Hardware-portable: runs on CUDA GPUs, Apple MPS (M1/M4), and CPU
- Tested geometries: Stanford rabbit, torus, star domain, ellipsoid

---

## Mathematical Theory

### Coordinate Charts and Atlas

Let Ω ⊂ ℝ³ be the physical domain. We cover it with *N* **coordinate charts** {(U_i, φ_i)}. Each chart consists of:

- **Chart seed** s_i ∈ ∂Ω — a surface point that anchors the chart
- **TNB frame** (t₁_i, t₂_i, n_i) — orthonormal tangent, co-tangent, and normal vectors at s_i
- **Local chart coordinates** ξ_i = (ξ, η, ζ) ∈ B_r ⊂ ℝ³, where B_r is a ball of radius r_i (the *support radius*)
- **Chart decoder** φ_i : B_r → ℝ³, a learned neural network that maps local coordinates to physical space:

```
x = φ_i(ξ) = s_i + [t₁_i | t₂_i | n_i] · ξ  +  Δφ_i(ξ)
```

where Δφ_i is a small nonlinear residual learned by the MLP, with tanh activations ensuring the map stays injective near the origin. The TNB frame is computed from the SDF surface normals.

**Partition of unity.** Each chart carries a soft mask w_i(x) ≥ 0 implemented as a learned MaskNet. The masks satisfy ∑_i w_i(x) ≈ 1 everywhere in Ω and are used to blend solutions across overlapping regions.

**Atlas construction** (Poisson-disk sampling):
1. Sample *N* seeds on the surface ∂Ω via Poisson-disk sampling with mutual separation ≥ d_min
2. For each seed, compute the TNB frame from the local SDF gradient and a consistent tangent frame
3. Initialize decoder φ_i as the identity map in the TNB basis; train to minimize reconstruction error + Jacobian smoothness + coverage loss

### PDE Formulation in Chart Coordinates

Consider the Poisson equation on Ω:

```
−Δ_x u(x) = f(x),    x ∈ Ω
u(x) = g(x),          x ∈ ∂Ω
```

On chart i, the PDE is pulled back to local coordinates via the chain rule. Let **J**_i = ∂φ_i/∂ξ be the 3×3 Jacobian of the decoder. Then:

```
Δ_x u = Σ_{j,k} a_{jk}(ξ) · ∂²u/∂ξ_j∂ξ_k  +  lower-order terms
```

where the metric tensor coefficients are:

```
a(ξ) = |det J| · (J⁻ᵀ J⁻¹)
```

**Jacobian inversion.** Computing **J**⁻¹ for backpropagation requires a numerically stable method. We use `torch.linalg.inv` (LU decomposition) for the inversion gradient path, reserving `svdvals` for condition-number diagnostics under `torch.no_grad()`. This avoids the backward singularity of SVD when two singular values coincide — which is common for near-isometric coordinate maps.

**Direct-coordinate mode (M1).** For diagnosis, the PDE can also be evaluated directly in physical space using only the rigid TNB frame (no learned decoder). Since the TNB frame is orthogonal with det J = 1 and condition number κ = 1:

```
Δ_x u = Σ_j ∂²u/∂ξ_j²    (exact identity for orthogonal frames)
```

This mode eliminates all Jacobian issues and lets you test whether the Schwarz decomposition itself is correct, independently of decoder quality.

### Multiplicative Schwarz Coupling

The Schwarz method decomposes the global PDE into local sub-problems on each chart's subdomain Ω_i, coupled through interface conditions on the overlapping regions Ω_i ∩ Ω_j.

**Interface value continuity** (Robin-penalty form):

```
loss_iv(i,j) = Σ_{x ∈ Ω_i ∩ Ω_j} [u_i(x) − u_j(x)]²
```

**Interface flux continuity** (normal derivative matching):

```
loss_if(i,j) = Σ_{x ∈ Ω_i ∩ Ω_j} [(∇u_i · n_ij) − (∇u_j · n_ij)]²
```

where n_ij is a consistent normal computed from the mask-level-set gradient.

**Multiplicative Schwarz update** proceeds in a graph-colored sequence: charts are grouped into non-adjacent color classes, and within each color all charts are updated simultaneously (or in parallel using CUDA streams). Each update performs `local_steps` gradient steps of Adam with the current neighbor solutions held fixed.

**Relaxation:** After each full sweep, a damped update `u_i ← ω · u_i^new + (1-ω) · u_i^old` is optionally applied (default ω = 0.8) to stabilize convergence.

### Multi-Objective PINN Loss

On each chart i the total loss is:

```
L_i = w_pde · L_pde_i  +  w_bc · L_bc_i  +  w_iv · L_iv_i  +  w_if · L_if_i
```

where:
- **L_pde** — Huber-clipped mean of |residual|² over interior collocation points
- **L_bc** — mean squared error of u against the Dirichlet condition on boundary points
- **L_iv** — interface value continuity (Robin penalty)
- **L_if** — interface flux continuity (normal derivative penalty)

**Recommended weights (M4):** `w_pde = 5`, `w_bc = 1`, `w_iv = 2`, `w_if = 2`. The PDE term must dominate for the solver to behave as a PDE solver rather than a regression problem; the original default of w_pde = 1 < w_bc = 2 was the primary cause of elevated interior residuals.

**PDE warmup:** The PDE weight is ramped from 0 to w_pde over `pde_warmup_iters` Schwarz iterations, giving the boundary conditions time to settle before the interior residual dominates.

---

## Numerical Examples

### Manufactured Solution (Poisson)

All Poisson experiments use a manufactured solution with known closed form:

```
u*(x, y, z) = sin(πx) · sin(πy) · sin(πz)
f(x, y, z)  = 3π² · u*(x, y, z)
```

This satisfies the homogeneous Dirichlet condition u* = 0 on the faces of the unit cube [0,1]³. The exact solution is used to compute the relative L² error:

```
rel_L2 = ‖u_PINN − u*‖₂ / ‖u*‖₂
```

Typical achieved accuracy on 12-chart rabbit atlas: **rel L² ≈ 7–8%** after 60 Schwarz iterations at default settings; improves to **< 5%** with M3+M4 rebalancing.

### Forward Poisson on the Rabbit Geometry

The Stanford rabbit surface encloses a simply-connected 3D domain. The Poisson equation −Δu = f is solved inside the rabbit body with:
- 12 overlapping coordinate charts (Poisson-disk sampled seeds, overlap ~20%)
- 15 Adam steps per chart per Schwarz iteration
- Batch size: 192 PDE points + 192 BC points + 128 interface points per chart

The solution is validated against the manufactured solution. Seam artifacts near chart boundaries are the dominant error source; M1 (direct-coordinate mode) isolates whether these originate from the Schwarz decomposition or from the learned decoder.

### Inverse Neo-Hookean Elasticity (Torus)

The **torus** (major radius R = 1.0, minor radius r = 0.35) with multiply-connected topology is the primary inverse benchmark:

**Forward model:** incompressible neo-Hookean elasticity
```
P = μ F − μ F⁻ᵀ + K(J−1) J F⁻ᵀ
Div P = 0  in Ω,    P·N = t  on ∂Ω_N,    u = 0  on ∂Ω_D
```

**Inverse problem:** given sparse displacement or traction observations, recover the shear modulus μ and bulk modulus K as spatially homogeneous unknowns (or piecewise constant by chart).

**Typical results:** parameter recovery to within 0.01% relative error on the torus, composite score ~4×10⁻⁷ at best.

### Inverse Elder-Like Flow (Rabbit)

**Forward model:** Darcy flow + species transport with anisotropic permeability:
```
−∇·(K(x)∇p) = 0          (pressure)
−∇·(D∇c − Ra·K(x)c∇p) = 0  (concentration)
```

**Inverse problem:** recover the permeability tensor K(x) = k₀ R(q) Λ Rᵀ(q) where R(q) is a rotation parameterized by quaternion q and Λ is a diagonal matrix, from sparse pressure/concentration observations on the rabbit geometry.

**Typical results:** parameter metric ~0.04 relative error; strong performance on concentration field recovery.

---

## Repository Structure

```
PINN_coordinate_chart_3Dgeometry/
│
├── src/                              # Core verified solvers
│   ├── pinn_3d_ellipsoid_mapped_sphere.py     # Poisson on 3D ellipsoid (sphere mapping)
│   ├── pinn_gradient_surgery.py               # 2D PINN with PCGrad conflict resolution
│   ├── run_poisson_star3d_mapped.py           # Poisson on 3D star domain
│   ├── run_torus_inverse_neohookean_atlas.py  # Inverse neo-Hookean elasticity (torus atlas)
│   ├── run_torus_inverse_neohookean_schwarz_dual.py  # Schwarz-coupled torus inverse
│   ├── run_rabbit_inverse_elder_atlas_schwarz.py     # Inverse Elder-like flow on rabbit
│   ├── run_rabbit_inverse_neohookean_mapped.py       # Neo-Hookean inverse on rabbit
│   ├── train_sdf_rabbit.py                    # Neural SDF training for arbitrary geometry
│   ├── train_mapping_from_sdf.py              # Sphere-to-domain mapping network
│   └── export_rabbit_elder_inverse_paraview.py
│
├── experiments/                      # Research-stage and advanced variants
│   ├── run_poisson_rabbit_atlas_schwarz.py    # ← MAIN Schwarz Poisson solver (modified by M1–M8)
│   ├── train_rabbit_atlas.py                  # Atlas decoder + mask network training
│   ├── build_rabbit_atlas_poissondisk.py      # Atlas seed selection (Poisson disk)
│   ├── run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz.py       # TI Arruda-Boyce inverse
│   ├── run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz_mms.py   #   ↳ manufactured solution
│   ├── run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz_normal_ramp.py
│   ├── run_rabbit_inverse_neohookean_atlas_schwarz_normal_disp.py
│   ├── run_torus_inverse_neohookean_atlas_cube.py
│   ├── run_torus_inverse_neohookean_atlas_fillet_inn.py
│   ├── postprocess_rabbit_poisson_dense_fields.py
│   ├── export_rabbit_atlas_paraview.py
│   ├── export_rabbit_error_paraview.py
│   ├── export_rabbit_ti_inverse_interface_diagnostics.py
│   └── experimental_ideas/                    # Archived exploratory variants
│       ├── atlas_split/
│       └── fixed_focus/
│
├── configs/                          # YAML configuration files
│   ├── rabbit_atlas_poisson.yaml              # Schwarz Poisson solver config
│   ├── rabbit_inverse_elder_atlas.yaml        # Elder flow inverse config
│   ├── rabbit_inverse_ti_arruda_boyce.yaml    # TI Arruda-Boyce config
│   ├── rabbit_inverse_neohookean_normal_disp.yaml
│   ├── rabbit_sdf_inverse.yaml
│   ├── torus_inverse_neohookean_atlas_cube.yaml
│   ├── torus_inverse_neohookean_atlas_fillet_inn.yaml
│   └── rabbit_data_source.yaml                # Paths to geometry data
│
├── docs/                             # Research notes and paper skeleton
│   ├── chatgpt52_paper_starter.md             # Draft paper outline
│   └── chatgpt52_handoff_mindmap.md           # Method overview and decisions
│
├── scripts/
│   └── check_mps_feasibility.py              # Apple MPS compatibility check
│
├── pinn_only_improvement_plan.md     # Detailed roadmap: architecture, training, sampling
├── benchmark_smoke.json              # Quick smoke-test results
└── runs/                             # Saved checkpoints, metrics, and exports
```

---

## Prerequisites and Installation

### Python Version

Python ≥ 3.9 is required (3.10 or 3.11 recommended).

### Required Libraries

Install via pip:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# or for Apple Silicon (MPS backend):
pip install torch torchvision
```

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Neural networks, autograd, linear algebra |
| `numpy` | ≥ 1.24 | Array operations, point cloud I/O |
| `matplotlib` | ≥ 3.7 | Convergence plots, field visualization |
| `scipy` | ≥ 1.10 | (optional) Poisson-disk sampling utilities |

No mesh generation library (Gmsh, FEniCS, etc.) is required — the method is fully meshfree.

### Hardware

| Platform | Notes |
|----------|-------|
| **NVIDIA GPU (CUDA)** | Recommended; float64 fully supported; AMP (`--amp`) reduces memory |
| **Apple M1/M2/M3/M4 (MPS)** | Supported; **must use `--dtype float32`**; some linalg kernels have occasional MPS-specific instabilities |
| **CPU** | Works; float64 default; slow for large runs |

**Auto-detection:** pass `--device auto` (the default) to automatically select CUDA → MPS → CPU.

### Optional: ParaView

For 3D visualization of solution fields:
- Download ParaView ≥ 5.11 from https://www.paraview.org/download/
- Export scripts in `experiments/` write `.vtu` / `.vtp` files readable by ParaView

### Clone the Repository

```bash
git clone https://github.com/stvsun/PINN_coordinate_chart_3Dgeometry.git
cd PINN_coordinate_chart_3Dgeometry
```

---

## Geometry Preparation

The full pipeline to run the Schwarz Poisson solver on a new geometry has three stages.

### Stage 0: Obtain a Point Cloud

For the **Stanford rabbit**, download the PLY file and convert it to a NumPy point cloud:

```bash
# Download
wget https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
tar -xf bunny.tar.gz

# Convert PLY → NPZ (or use Open3D / trimesh in Python)
python - <<'EOF'
import open3d as o3d, numpy as np
mesh = o3d.io.read_triangle_mesh("bunny/reconstruction/bun_zipper.ply")
pcd  = mesh.sample_points_uniformly(number_of_points=30000)
pts  = np.asarray(pcd.points)
nrm  = np.asarray(pcd.normals)
np.savez("rabbit_pointcloud.npz", points=pts, normals=nrm)
EOF
```

For analytic domains (torus, ellipsoid, star), point clouds are generated inside the respective training scripts.

### Stage 1: Training a Neural SDF

The neural SDF provides an implicit volumetric representation of the geometry. It is used for:
- SDF-guided interior sampling (M2)
- Hard BC weighting (M7)
- Boundary distance information during atlas construction

```bash
python src/train_sdf_rabbit.py \
    --point-cloud  rabbit_pointcloud.npz \
    --output-dir   runs/sdf_rabbit \
    --epochs       5000 \
    --width        128 \
    --depth        6 \
    --lr           8e-4 \
    --device       auto \
    --seed         42
```

**Key outputs:**
- `runs/sdf_rabbit/sdf_rabbit_final.pt` — SDF network checkpoint
  Format: `{"model": state_dict, "center": [cx,cy,cz], "scale": s, "width": 128, "depth": 6}`

**Loss terms during SDF training:**
- Surface constraint: `|SDF(x_surf)| = 0`
- Eikonal: `|∇SDF(x)| = 1`
- Normal alignment: `∇SDF(x_surf) ≈ n_surf`
- Sign anchor: `SDF(x_far) > 0` for exterior points

### Stage 2: Building the Atlas

The atlas construction selects Poisson-disk sampled seeds, computes TNB frames, and stores membership assignments:

```bash
python experiments/build_rabbit_atlas_poissondisk.py \
    --point-cloud    rabbit_pointcloud.npz \
    --output-dir     runs/rabbit_atlas_data \
    --n-charts       12 \
    --overlap-target 0.20 \
    --frame-k        96 \
    --normal-k       32 \
    --seed           42
```

**Key outputs:**
- `runs/rabbit_atlas_data/rabbit_atlas_data.npz`
  Arrays: `seed_points`, `support_radii`, `t1`, `t2`, `normals`, `membership`
- `runs/rabbit_atlas_data/meta.json` — color groups for graph-colored Schwarz updates

### Stage 3: Training the Atlas Decoder Networks

Each chart's decoder (MLP: ξ → x_phys) and MaskNet (MLP: ξ → w_i) are trained jointly:

```bash
python experiments/train_rabbit_atlas.py \
    --atlas-data     runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-meta     runs/rabbit_atlas_data/meta.json \
    --output-dir     runs/rabbit_atlas_trained \
    --epochs         3000 \
    --width          64 \
    --depth          4 \
    --mask-width     48 \
    --mask-depth     3 \
    --lr             8e-4 \
    --device         auto \
    --seed           42
```

**Training losses:**
- `w_recon = 1.0`: reconstruction error ‖φ_i(ξ_surf) − x_surf‖²
- `w_mask = 0.5`: mask normalization ∑_i w_i(x) ≈ 1
- `w_overlap = 0.8`: overlap coverage loss
- `w_jac = 2.0`: Jacobian barrier (det J > δ to prevent folding)
- `w_coverage = 0.7`: ensure sufficient interior coverage

**Atlas quality gates** (checked before running PDE solvers):
- Coverage fraction > 99%
- Overlap consistency < 2.5%
- Foldover ratio = 0%
- Boundary RMSE < 3%

**Key outputs:**
- `runs/rabbit_atlas_trained/atlas_checkpoint.pt` — decoder + mask state dicts

---

## Running the Solvers

### Forward Poisson (Schwarz)

The main Schwarz Poisson solver is `experiments/run_poisson_rabbit_atlas_schwarz.py`.

**Minimal run using YAML config:**

```bash
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --config          configs/rabbit_atlas_poisson.yaml \
    --atlas-data      runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-meta      runs/rabbit_atlas_data/meta.json \
    --atlas-checkpoint runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir      runs/poisson_schwarz_rabbit \
    --device          auto \
    --seed            42
```

**With all M1–M8 features enabled:**

```bash
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --config               configs/rabbit_atlas_poisson.yaml \
    --atlas-data           runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-meta           runs/rabbit_atlas_data/meta.json \
    --atlas-checkpoint     runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir           runs/poisson_m1_m8 \
    --device               auto \
    --seed                 42 \
    \
    --direct-coord-pde              \  # M1: bypass decoder Jacobian
    --sdf-checkpoint       runs/sdf_rabbit/sdf_rabbit_final.pt \
    --use-sdf-sampling              \  # M2: interior collocation via SDF rejection
    --sdf-interior-threshold  0.0   \
    --sdf-rejection-factor    6     \
    --hard-bc                       \  # M7: SDF-based hard BC weighting
    --hard-bc-scale           0.05  \
    --rar-period              10    \  # M8: RAR adaptive sampling every 10 Schwarz iters
    --rar-candidates          512   \
    --rar-top-k               64    \
    --rar-pool-max            256   \
    --rar-mix-n               32
```

**Key argparse flags for the Poisson solver:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML config (values overridden by CLI flags) |
| `--atlas-data` | — | `.npz` from `build_rabbit_atlas_poissondisk.py` |
| `--atlas-meta` | — | `meta.json` from atlas build |
| `--atlas-checkpoint` | — | `.pt` from `train_rabbit_atlas.py` |
| `--output-dir` | — | Directory for checkpoints and metrics |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--dtype` | `auto` | `auto` / `float32` / `float64` |
| `--amp` | off | Mixed-precision (CUDA autocast + GradScaler) |
| `--seed` | 42 | Global random seed |
| `--pinn-width` | 64 | Hidden width of each chart PINN |
| `--pinn-depth` | 4 | Number of hidden layers |
| `--lr` | `8e-4` | Adam learning rate |
| `--max-schwarz-iters` | 60 | Total Schwarz outer iterations |
| `--local-steps` | 15 | Adam steps per chart per Schwarz iteration |
| `--pde-batch` | 192 | PDE collocation points per chart per step |
| `--bc-batch` | 192 | Boundary condition points per chart per step |
| `--if-batch` | 128 | Interface points per chart-pair per step |
| `--w-pde` | 5.0 | PDE loss weight |
| `--w-bc` | 1.0 | BC loss weight |
| `--w-interface-value` | 2.0 | Interface value continuity weight |
| `--w-interface-flux` | 2.0 | Interface flux continuity weight |
| `--pde-warmup-iters` | 50 | Ramp PDE weight over this many Schwarz iters |
| `--direct-coord-pde` | off | M1: bypass decoder Jacobian |
| `--sdf-checkpoint` | None | M2/M7: SDF network path |
| `--use-sdf-sampling` | off | M2: SDF rejection interior sampling |
| `--hard-bc` | off | M7: SDF-based BC weighting |
| `--hard-bc-scale` | 0.05 | M7: tanh length scale |
| `--rar-period` | 0 | M8: RAR update period (0 = disabled) |
| `--log-every` | 1 | Log metrics every N Schwarz iters |

### Inverse Neo-Hookean Elasticity

```bash
# Torus geometry — atlas variant (most stable)
python src/run_torus_inverse_neohookean_atlas.py \
    --output-dir  runs/torus_inv_mu_K \
    --device      auto \
    --dtype       float32 \
    --seed        42 \
    --epochs      250

# Schwarz dual-physics variant (displacement + traction)
python src/run_torus_inverse_neohookean_schwarz_dual.py \
    --output-dir  runs/torus_schwarz_dual \
    --device      auto \
    --seed        42

# Rabbit geometry
python experiments/run_rabbit_inverse_neohookean_atlas_schwarz_normal_disp.py \
    --config      configs/rabbit_inverse_neohookean_normal_disp.yaml \
    --atlas-data  runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-checkpoint runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir  runs/rabbit_inv_neohookean
```

### Inverse Elder Flow

```bash
python src/run_rabbit_inverse_elder_atlas_schwarz.py \
    --config      configs/rabbit_inverse_elder_atlas.yaml \
    --atlas-data  runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-meta  runs/rabbit_atlas_data/meta.json \
    --atlas-checkpoint runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir  runs/rabbit_inv_elder \
    --device      auto \
    --seed        42
```

### 3D Ellipsoid / Star Domain (Simpler Baselines)

```bash
# Ellipsoid (sphere mapping — no atlas needed)
python src/pinn_3d_ellipsoid_mapped_sphere.py \
    --output-dir runs/ellipsoid_poisson \
    --device     auto

# 3D star domain
python src/run_poisson_star3d_mapped.py \
    --output-dir runs/star_poisson \
    --device     auto
```

---

## Configuration Reference

All major hyperparameters can be set in YAML and overridden on the command line.

**`configs/rabbit_atlas_poisson.yaml`** (annotated excerpt):

```yaml
run:
  seed: 42
  n_charts: 12            # Number of atlas charts
  overlap_target: 0.20    # Target fraction of points in ≥ 2 charts

atlas_train:
  epochs: 3000
  lr: 8.0e-4
  width: 64               # Decoder MLP hidden width
  depth: 4                # Decoder MLP depth
  mask_width: 48          # MaskNet hidden width
  mask_depth: 3
  w_recon: 1.0            # Reconstruction loss weight
  w_jac: 2.0              # Jacobian determinant barrier

poisson_schwarz:
  pinn_width: 64          # Chart PINN hidden width
  pinn_depth: 4
  lr: 8.0e-4
  max_schwarz_iters: 60   # Total Schwarz outer iterations
  local_steps: 15         # Adam steps per chart per iteration
  omega: 0.8              # Schwarz relaxation parameter

  pde_batch: 192          # PDE collocation batch size
  bc_batch: 192           # Boundary condition batch size
  if_batch: 128           # Interface batch size

  pde_warmup_iters: 50    # Ramp PDE weight over N iters
  w_pde: 5.0              # PDE residual weight (dominant)
  w_bc: 1.0               # Dirichlet BC weight
  w_interface_value: 2.0  # Schwarz value continuity weight
  w_interface_flux: 2.0   # Schwarz flux continuity weight

  sigma_floor: 1.0e-3     # Minimum singular value for Jacobian masking
  det_floor: 1.0e-6       # Minimum |det J| for valid chart points
  jac_kappa_max: 1.0e3    # Maximum condition number κ = σ_max/σ_min

  # M1: direct-coordinate PDE (bypass learned decoder Jacobian)
  direct_coord_pde: false

  # M2: SDF-guided interior sampling
  sdf_checkpoint: null
  use_sdf_sampling: false
  sdf_interior_threshold: 0.0    # Accept points with SDF < this value
  sdf_rejection_factor: 6        # Oversampling factor for rejection

  # M7: SDF-based hard BC weighting
  hard_bc: false
  hard_bc_scale: 0.05            # tanh length scale

  # M8: Residual-based Adaptive Refinement (RAR)
  rar_period: 0                  # Update RAR pool every N Schwarz iters (0=off)
  rar_candidates: 512            # Candidate points to score per RAR update
  rar_top_k: 64                  # Top-k residual points added to pool
  rar_pool_max: 256              # Maximum pool size per chart
  rar_mix_n: 32                  # RAR pool points mixed per training batch
```

---

## Recent Improvements (M1–M8)

The following improvements were applied to `experiments/run_poisson_rabbit_atlas_schwarz.py` and `configs/rabbit_atlas_poisson.yaml` in the current version. Each is individually opt-in (except M3 and M4 which are always active):

| ID | Name | Status | Effect |
|----|------|--------|--------|
| **M3** | Stable Jacobian inversion | Always-on | Replace SVD backward with `torch.linalg.inv` (LU). Eliminates gradient NaN/inf when singular values cluster. |
| **M4** | Rebalanced loss weights | Always-on | `w_pde 1→5`, `w_bc 2→1`, `w_iv 0.8→2`, `w_if 0.2→2`, `warmup 10→50`. PDE now correctly dominates. |
| **M1** | Direct-coordinate PDE | `--direct-coord-pde` | Compute Δu directly via rigid TNB frame. Since the TNB frame is orthogonal, Δ_x u = Δ_ξ u exactly — no learned Jacobian, no SVD. |
| **M2** | SDF interior sampling | `--use-sdf-sampling` | Sample collocation points in the volumetric interior using SDF rejection, fixing the fundamental bug where surface-noise sampling only covers a thin surface shell. |
| **M7** | Hard BC via SDF weight | `--hard-bc` | Weight BC loss by `tanh(−SDF(x)/scale)`: smooth zero at boundary, one in interior. Prevents near-boundary BC points from dominating the gradient. |
| **M8** | Residual-based Adaptive Refinement (RAR) | `--rar-period N` | Every N Schwarz iters, evaluate residual on candidate points and add top-k highest-residual points to a persistent pool that is mixed into training batches. Focuses training on high-error regions. |

**M5** (PCGrad gradient surgery for multi-objective Schwarz) and **M6** (volumetric atlas rebuild) are planned for a future release.

---

## Post-Processing and Visualization

**Export to ParaView:**

```bash
# Poisson solution field
python experiments/export_rabbit_atlas_paraview.py \
    --schwarz-checkpoint  runs/poisson_schwarz_rabbit/schwarz_checkpoint.pt \
    --atlas-data          runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir          runs/poisson_schwarz_rabbit/paraview

# Error field (requires reference manufactured solution)
python experiments/export_rabbit_error_paraview.py \
    --schwarz-checkpoint  runs/poisson_schwarz_rabbit/schwarz_checkpoint.pt \
    --atlas-data          runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir          runs/poisson_schwarz_rabbit/paraview_error

# Dense post-processed field
python experiments/postprocess_rabbit_poisson_dense_fields.py \
    --schwarz-checkpoint  runs/poisson_schwarz_rabbit/schwarz_checkpoint.pt \
    --atlas-data          runs/rabbit_atlas_data/rabbit_atlas_data.npz \
    --atlas-checkpoint    runs/rabbit_atlas_trained/atlas_checkpoint.pt \
    --output-dir          runs/poisson_schwarz_rabbit/dense_field \
    --n-query             50000
```

**Output files written per solver run:**
- `schwarz_checkpoint.pt` — per-chart PINN weights and optimizer state
- `metrics.json` — per-iteration: `global_residual`, `bc_loss`, `interface_value`, `interface_flux`, `rel_l2_eval`, `lr`
- `training_curve.png` — matplotlib convergence plot
- `*.vtu` / `*.vtp` — ParaView field exports (if export scripts are run)

---

## Known Limitations and Open Problems

1. **Volumetric sampling gap (M2).** The default `sample_local_xi` places points by adding Gaussian noise to surface point coordinates, which covers only a thin shell near ∂Ω. The M2 SDF-rejection sampler (`--use-sdf-sampling`) fixes this but requires a pre-trained SDF network.

2. **Seam artifacts in gradient fields.** The partition-of-unity blending is C⁰ but not C¹; gradient-based observables (velocity = −K∇p, stress) show visible seams at chart boundaries. Addressed in M1 (TNB diagnostic) but not yet fully resolved.

3. **MPS float32 only.** Apple MPS does not support float64 reliably. The MPS code path uses a hand-written adjugate-based Jacobian inverse to avoid unstable MPS `linalg.inv` kernels, but some edge cases may still produce NaN on very ill-conditioned charts.

4. **TI Arruda-Boyce identifiability.** The transversely-isotropic material inverse problem has near-degenerate parameter directions; small changes in the fibre-direction parameter q produce nearly identical observations, causing parameter drift and high variance.

5. **M6 (volumetric atlas rebuild).** The current atlas seeds are placed on the surface ∂Ω with normals pointing inward. A fully volumetric atlas would place additional seed points in the interior, with local coordinate frames decoupled from the surface TNB frame. This requires rewriting `build_rabbit_atlas_poissondisk.py` and `train_rabbit_atlas.py`.

6. **M5 (PCGrad for Schwarz).** Project Conflicting Gradients resolves multi-objective conflicts but adds roughly 4× backward computation cost per Schwarz step. Validate M1–M4 accuracy improvements first before enabling.

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
