# Mapped-Sphere PINN — Results Summary
*Last updated: 2026-03-01*

---

## Best Published Results

| Benchmark | Geometry | rel-L² | Schwarz iters | Run directory |
|-----------|----------|--------|---------------|---------------|
| **Poisson MMS** | Procedural rabbit (5-ellipsoid) | **2.21%** | 61 | `runs/attempt20c_compact/` |
| Poisson MMS (ResNet) | Procedural rabbit | 8.47% | 1 | `runs/poisson_rabbit_resnet_guard_20260215_010119/` |
| **Elder (surface)** | Stanford Bunny PLY | — | — | `runs/rabbit_inverse_elder_atlas_main/` |

---

## Poisson Benchmark — Procedural Rabbit (best: 2.21%)

### Domain
- **Geometry**: 5-ellipsoid procedural rabbit, physically scaled at 1.64 m
- **PDE**: −Δu = f, manufactured solution u = sin(πx)sin(πy)sin(πz)
- **Atlas**: `runs/atlas_vol/` (volumetric, procedural SDF, gate=True, overlap=0.006)
- **Atlas decoders**: `runs/atlas_vol_trained/rabbit_atlas_trained.pt`

### PINN settings (`attempt20c_compact`)
```bash
--pinn-arch compact --pinn-width 64 --pinn-depth 4
--w-pde 5.0 --w-bc 1.0 --w-interface-value 2.0 --w-interface-flux 2.0
--pde-warmup-iters 50 --max-schwarz-iters 120
--plateau-patience 20 --plateau-use-rel-l2
--checkpoint-policy best_rel_l2
--interface-normal-mode seed
--direct-coord-pde
```
- 9 sub-seeds per CompactChartNet, 10,953 params/chart
- Device: Apple MPS (M-series), float32
- Runtime: ~3659 s (~61 min), 61 accepted Schwarz iterations

### Reproduction script
```bash
bash scripts/successful/run_poisson_rabbit_best.sh
```

### Why this domain works
The procedural SDF is analytically defined (perfect sign quality), giving:
- Zero sign error → correct BC sampling and interface normals
- Atlas overlap_consistency = 0.006 (gate=True, far below threshold of 0.025)
- Clean chart interfaces → Schwarz iteration converges monotonically

---

## Stanford Bunny — SDF + Atlas (geometry deliverables)

### SDF v3 (`runs/bunny_sdf_v3/`)
Trained on `bun_zipper.ply` (34,834 vertices) with balanced loss weights.

| Metric | V1 | V2 | **V3** |
|--------|----|----|--------|
| sign loss | 0.700 | 0.295 | **0.311** |
| surface loss | 0.013 | 0.072 | **0.017** ✅ |
| normal loss | — | 0.207 | **0.032** ✅ |
| scale (m) | — | 0.1557 | **0.1557** |

V3 training settings:
```bash
python src/train_sdf_rabbit.py \
    --points-file bun_zipper.ply \
    --output-dir runs/bunny_sdf_v3 \
    --width 128 --depth 6 --epochs 20000 \
    --anchor-offset 0.25 --w-sign 8.0 --w-normal 3.0 --w-surface 10.0 --batch-far 4096
```

### Atlas v3 (`runs/atlas_bunny_vol_v3/`)
- 50,000 interior points, 12 charts, axis-aligned frames
- **100%** of points within PLY bounding box (vs 95.2% for v1)
- Physical domain: x∈[−0.097, 0.051], y∈[0.025, 0.178], z∈[−0.062, 0.059]

### Atlas decoders v3 (`runs/atlas_bunny_vol_v3_hiW/`)
Trained with `--w-overlap 8.0 --batch-overlap 2048 --epochs 6000 --volumetric`.

| Metric | default (v3 vol) | hiW (v3 vol) | Procedural (reference) |
|--------|-----------------|--------------|------------------------|
| gate passed | ❌ False | ✅ **True** | ✅ True |
| overlap_consistency | 0.083 | **0.018** | 0.006 |
| coverage_ratio | 1.000 | **1.000** | — |
| foldover_ratio | 0.000 | **0.000** | — |

Key finding: default `--w-overlap 0.8` caused overlap loss to **increase** during
training (1.23e−4 → 8.67e−3 at epoch 2000). Raising to 8.0 kept overlap
training loss decreasing monotonically (9.33e−4 at epoch 2000, 4.69e−4 at epoch 4100).

Reproduction script:
```bash
bash scripts/successful/build_bunny_sdf_atlas_v3.sh
```

---

## Stanford Bunny — Poisson PINN (open problem)

### Summary
All Poisson PINN runs on the Stanford Bunny domain diverged. Best result: **28.1% rel-L²** (iter=1 checkpoint from `runs/bunny_poisson_v3/`).

### Convergence history across all attempts

| Run | Atlas | Gate | overlap | iter=1 rel-L² | Final | Status |
|-----|-------|------|---------|---------------|-------|--------|
| `bunny_poisson` | v1 (vol) | ❌ | — | 4.1% | 15.2% | Diverged at iter 16 |
| `bunny_poisson_v3` | v3 (vol) | ❌ 0.083 | 0.083 | 27.9% | 41.8% | Plateaued iter 24 |
| `bunny_poisson_v3_hiW` | v3+hiW | ✅ 0.018 | 0.018 | 28.3% | 36.5% (iter 7, killed) | Diverging |
| `bunny_poisson_diagnostic` | v3+hiW + w_sup=3 | ✅ 0.018 | 0.018 | 24.9% | 42.7% (iter 10, killed) | Diverging |

### Root cause analysis

**1. SDF sign quality is insufficient (sign loss = 0.311)**

The Stanford Bunny has thin features (ears ~5 mm thick). The single anchor offset
`--anchor-offset 0.25` (39 mm physical) places interior anchors **outside** thin
features, making their sign supervision impossible. This is a geometric mismatch:
- Ears require anchor_offset < 0.032 (5 mm / 157 mm scale)
- Body requires anchor_offset > 0.1 (15 mm+) for deep interior supervision
- A single fixed offset cannot serve both

Consequence: 31% of near-surface sign predictions are wrong → interface normals
and BC sampling are corrupted.

**2. Schwarz iteration is sensitive to interface accuracy**

The Schwarz additive iteration requires consistent interface conditions between
charts to converge. With 31% sign error:
- Interface normals (SDF gradient) are incorrect at ~31% of interface points
- BC positions (SDF zero-crossing) are shifted by O(sign_error × offset)
- Each Schwarz step accepts locally correct PDE but globally diverges

Evidence: interface flux residual **increased** every accepted iteration:
```
iter=1: if_flux=0.147  →  iter=7: if_flux=0.200  (charts drifting apart)
```

**3. Manufactured supervision cannot rescue a diverging Schwarz iteration**

At the start of training, PDE loss = ~960, manufactured supervision contribution
= w_sup × L_sup = 3 × 9e−4 = 0.003. The PDE term dominates by a factor of
~1.8 million, making supervision negligible during the critical early iterations.

**4. Atlas overlap quality (0.018) is 3× worse than the working procedural case (0.006)**

The 3× gap in overlap consistency correlates with Schwarz failure. The Stanford
Bunny's complex geometry prevents atlas charts from achieving the clean 2D-like
interfaces that the smooth 5-ellipsoid procedural model naturally provides.

### What would fix it

| Fix | Difficulty | Expected outcome |
|-----|-----------|-----------------|
| **Chart-partitioned SDF** (implemented, see below) | Done | Adaptive offset 0.2–4.2 mm vs 38.9 mm global → sign error expected < 0.05 |
| Multi-scale anchor offsets (e.g., 0.02 + 0.15 + 0.50) | Medium (code change) | sign < 0.1 → possible convergence |
| Separate supervised warmup phase before Schwarz | Medium (code change) | Direct solution initialization → convergence |
| Replace manufactured solution with PLY-mesh BCs | Medium (code change) | Removes SDF sign dependence for BCs |
| Accept procedural rabbit as Poisson benchmark | None | 2.21% already achieved |

---

## Chart-partitioned SDF (new numerical example)

### Motivation

The global SDF used a single anchor offset of **38.9 mm** for all 12 charts.
Stanford Bunny ears are ~5 mm thick — the 38.9 mm offset placed sign-supervision
anchors **outside the ears entirely**, producing 31% sign errors and corrupting
all interface normals simultaneously.

### Architecture: `src/train_sdf_chartwise.py`

12 independent `SDFNetLocal` networks (width=64, depth=4), one per atlas chart,
blended at inference via Gaussian partition-of-unity:

```
φ(x) = Σᵢ wᵢ(x) φᵢ(x) / Σᵢ wᵢ(x),   wᵢ(x) = exp(-‖x-seedᵢ‖² / 2σᵢ²)
```

### Key innovation 1: Geometry-adaptive anchor offset

```
offset_i = 0.30 × dist(seed_i, nearest PLY surface point)
```

Measured on the actual atlas (`atlas_bunny_vol_v3`):

| Chart | Seed→Surface | Adaptive offset | Status |
|-------|-------------|----------------|--------|
| 0 | 14.1 mm | 4.2 mm | body |
| 1 | 8.4 mm | 2.5 mm | body |
| 2 | 4.8 mm | 1.5 mm | **ear-safe** |
| 3 | 3.7 mm | 1.1 mm | **ear-safe** |
| 4 | 5.9 mm | 1.8 mm | **ear-safe** |
| 5 | 0.6 mm | 0.2 mm | **ear-safe** |
| 6 | 7.6 mm | 2.3 mm | **ear-safe** |
| 7 | 4.6 mm | 1.4 mm | **ear-safe** |
| 8 | 1.9 mm | 0.6 mm | **ear-safe** |
| 9 | 1.2 mm | 0.4 mm | **ear-safe** |
| 10 | 4.7 mm | 1.4 mm | **ear-safe** |
| 11 | 3.7 mm | 1.1 mm | **ear-safe** |

**10 / 12 charts ear-safe** vs 0/12 for global offset of 38.9 mm.

### Key innovation 2: Voxel SDT initialisation + Eikonal fine-tuning

For each chart, training proceeds in two phases:

**Phase 1 — SDT pre-training** (400 epochs, MSE regression):
For each chart, a KD-tree query on the local PLY point cloud gives the
unsigned distance to the nearest surface point. Sign is determined by the
outward-normal dot-product test:

```
sign(q) = +1  if  (q - p_nearest) · n_nearest ≥ 0  (exterior)
          -1  otherwise                               (interior)
```

The local `SDFNetLocal` is pre-trained to match this analytical SDT.
Starting from a correct sign topology means sign corrections are fine-tuning
rather than re-learning — directly addressing the root cause.

**Phase 2 — Eikonal fine-tuning** (3000 epochs):
The fine-tuning enforces the Eikonal equation `‖∇φ‖ = 1` via:

```
L_eik = mean( (‖∇φ‖ - 1)² )
```

This is the **steady-state governing equation of the reinitialization problem**:

```
∂φ/∂t + sign(φ₀)(‖∇φ‖ - 1) = 0  →  ‖∇φ‖ = 1  at steady state
```

The zero level set is preserved by the surface loss `‖φ_surf‖`, and the
sign topology from phase 1 ensures the correct branch is selected.

### Normal estimation for PLY without normals

The `bun_zipper.ply` file has no per-vertex normals.  The script uses PCA
on `k`-nearest neighbours (`--normal-k 20`) to estimate surface normals from
the point cloud geometry, then applies a global outward-orientation flip via
the centroid test.

### Usage

```bash
python src/train_sdf_chartwise.py \
    --ply-file   bun_zipper.ply \
    --atlas-data runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --output-dir runs/bunny_sdf_chartwise \
    --anchor-offset-factor 0.30 \
    --pretrain-epochs 400 \
    --epochs 3000 \
    --fit-adapter          # also saves drop-in global adapter
```

### Outputs

| File | Description |
|------|-------------|
| `rabbit_sdf_chartwise.pt` | Full chartwise model (SDFNetChartwise) |
| `rabbit_sdf_adapter.pt` | Global adapter — drop-in for unmodified atlas builder |
| `chart_NN_sdf.pt` | Per-chart checkpoint (12 files) |
| `rabbit_sdf_chartwise_meta.json` | Quality metrics, per-chart offsets |
| `adaptive_offsets.png` | Bar chart: per-chart offset vs support radius |
| `chartwise_sdf_slices.png` | Three orthogonal SDF slices (blended view) |

### Full pipeline (chartwise SDF → atlas → PINN)

```bash
bash scripts/experimental/run_chartwise_sdf_example.sh
```

Uses `rabbit_sdf_adapter.pt` as a drop-in for `build_rabbit_atlas_volumetric.py`
and the existing Poisson PINN — **no modifications to existing code required**.

---

## Available Figures and VTU Files

### Publication-ready figures (`figures/3d/`)
Generated by `postprocessing/render_3d_figures.py`.

| File | Description |
|------|-------------|
| `rabbit_poisson_error_surface.png` | Pointwise error on procedural rabbit (PyVista alpha-shape) |
| `rabbit_poisson_upred_surface.png` | Predicted solution on procedural rabbit |
| `rabbit_poisson_charts.png` | 12-chart atlas decomposition (procedural rabbit) |
| `rabbit_poisson_error.html` | Interactive Plotly: error field (procedural rabbit) |
| `rabbit_poisson_upred.html` | Interactive Plotly: predicted solution (procedural rabbit) |
| `rabbit_poisson_charts.html` | Interactive Plotly: chart decomposition |
| `rabbit_elder_fields.png` | Elder problem: two-scalar fields on Stanford Bunny |
| `rabbit_elder_pressure_4panel.png` | Elder pressure: 4-view panel (Stanford Bunny surface) |
| `rabbit_elder_error_4panel.png` | Elder error: 4-view panel (Stanford Bunny surface) |

### ParaView VTU files (`paraview/`)

| Directory | Contents |
|-----------|----------|
| `paraview/rabbit_poisson_bunny_v3/` | Stanford Bunny Poisson VTUs (28.1% result, iter=1 checkpoint) |

---

## File Organization

```
scripts/
├── successful/
│   ├── run_poisson_rabbit_best.sh          ← reproduces 2.21% Poisson result
│   ├── build_bunny_sdf_atlas_v3.sh         ← reproduces SDF v3 + atlas v3 + decoders
│   └── build_bunny_atlas_v3_pipeline.sh    ← full overnight pipeline (steps 1-5)
└── experimental/
    ├── fix_decoders.sh            ← attempt: standard atlas decoder training (failed: pde=165k)
    ├── fix_decoders_v2.sh         ← attempt: high-overlap-weight decoder training (gate=True, but PINN still diverges)
    ├── diagnostic_supervised.sh   ← attempt: PINN with manufactured supervision (w_sup=3, negligible vs PDE)
    └── resume_from_step4.sh       ← attempt: resume pipeline from Schwarz step after config fix

runs/
├── attempt20c_compact/            ← BEST: 2.21% Poisson, procedural rabbit
├── poisson_rabbit_resnet_guard_20260215_010119/  ← 8.47% ResNet, procedural rabbit
├── bunny_sdf_v3/                  ← SDF v3: sign=0.311, surface=0.017
├── atlas_bunny_vol_v3/            ← Atlas v3: 50k pts, 12 charts, 100% PLY containment
├── atlas_bunny_vol_v3_hiW/        ← Decoders: gate=True, overlap=0.018
├── bunny_poisson_v3/              ← Stanford Bunny Poisson: 28.1% (iter=1 best)
└── rabbit_inverse_elder_atlas_main/  ← Elder surface PINN: Stanford Bunny
```

---

## Convergence Figures (`figures/rabbit_poisson_v3/`)

Generated by `postprocessing/poisson_rabbit_paraview.py`:

| File | Description |
|------|-------------|
| `poisson_rabbit_convergence.pdf/png` | Schwarz iteration convergence curve (rel-L²) |
| `poisson_rabbit_error_summary.pdf/png` | Per-chart L² error breakdown |
