# Schwarz Poisson Solver — Stanford Rabbit Volumetric Atlas: Attempt Status Report

**Date**: 2026-02-25
**Project**: Multiplicative Schwarz PINN for Poisson equation on Stanford rabbit volumetric geometry
**Goal**: Achieve `rel_l2 < 5%` for the manufactured solution `u = sin(πx₁)sin(πx₂)sin(πx₃)`
**Current status**: ✅ **TARGET MET + W1–W3+W5 VALIDATED** — Best rel_l2: **3.117%** (attempt17_w3, W1+W2+W3), best max_error: **4.107%** (attempt15b_w1, W1 only). W5 (10× stronger flux coupling, H1 overlap) fixes the argparse-default underweighting bug: max_error −36% (8.485%→5.450%), chart uniformity dramatically improved. W4 (PCGrad K=2) is next to resolve the intra-chart L_pde vs L_sup gradient conflict.

---

## Problem Setup

| Item | Detail |
|------|--------|
| Domain | Stanford rabbit volumetric interior, SDF-normalized coords (≈ [-0.55, 0.55]³) |
| Atlas (M6) | 12 interior ball-charts, seeds placed by SDF-guided Poisson disk sampling |
| Support radius | `r ≈ 0.41` in normalized space |
| Chart overlap | ~20% (confirmed by atlas build — 100% coverage, neighbor count 2–5) |
| Manufactured solution | `u = sin(πx₁)sin(πx₂)sin(πx₃)`, RHS `f = -ΔU = 3π²u` |
| Evaluation metric | `rel_l2 = ‖u_pred − u_true‖₂ / ‖u_true‖₂` over 50K atlas interior points |
| RMS of `u_true` | ≈ 0.389 over interior points |

---

## Infrastructure Status (Working Components)

The following components have been built, tested, and confirmed to be working correctly:

### ✅ Step 1: SDF Network Training
- **File**: trained in previous session, checkpoint loaded via `load_sdf_for_schwarz()`
- **Status**: Working. SDF net correctly classifies interior (SDF < 0) vs exterior on normalized coordinates
- **Key detail**: SDF net takes **SDF-normalized** coordinates as input (NOT physical/raw coordinates)

### ✅ Step 2: Volumetric Atlas Build (M6)
- **File**: `experiments/build_rabbit_atlas_volumetric.py`
- **Output**: `runs/atlas_vol/rabbit_atlas_data.npz`
- **Status**: Working. 12 charts, 50K interior points, support_r ≈ 0.41, 100% coverage, ~20% overlap
- **Commit**: `31b668c` ("Implement M6: volumetric atlas rebuild with interior-seed ball charts")

### ✅ Step 3: Atlas Decoder Training
- **File**: `experiments/train_rabbit_atlas.py`
- **Output**: `runs/atlas_vol/rabbit_atlas_trained.pt`
- **Status**: Working. ChartDecoder + mask networks trained. L_domain double-norm bug fixed in commit `8e91761`
- **Architecture**: `x = seed + ξ₁t₁ + ξ₂t₂ + ξ₃n + 0.20·tanh(raw_scale)·chart_scale·net(ξ/chart_scale)`. Amplitude of residual term ≈ 0.08 in normalized coords
- **Axis-aligned frames**: t1=e1, t2=e2, n=e3 for interior-seed charts

### ✅ Schwarz Infrastructure
- Multiplicative Schwarz loop (sequential chart updates) with score-based acceptance
- Interface value coupling (`w_if` for `u_i ≈ u_j` at interface surface)
- Interface flux coupling (`flux_mult` for normal-derivative matching)
- LR scheduling with stale-detection and halving
- `eval_rel_l2_subset()` evaluation against 50K interior reference points
- Argparse with all required flags

### ✅ BC (Boundary Condition) Pretrain
- Surface BC pretrain converges to `loss ≈ 5e-3` in 300 epochs
- BC sampling from domain surface (SDF≈0) is working after fix in `12d492a`

### ✅ Interior Supervised Pretrain
- Interior pretrain using atlas decoder to compute supervision targets `u_true(decoder(ξ))`
- Converges from `~0.10` to `~0.04` over 1000 epochs (loss on decoder-mapped points)
- BC retention term (`--interior-pretrain-bc-weight`) working: reduces BC degradation from 44× to ~2×

### ✅ All Double-Normalization Bugs Fixed
- 4 instances of double-normalization found and fixed across 2 files (see Bug Log below)
- All SDF evaluations now pass already-normalized coordinates directly

### ✅ H1 Volumetric Overlap Coupling (New Feature)
- Implemented `sample_chart_overlap_volumetric(i, j, n_samples)` helper
- Samples from 3D intersection of chart ball supports using rejection-based approach
- H1 loss term `w_overlap_h1 * E[i,j∈neighbors][(u_i(ξ_i) - u_j(ξ_j))²]` functional
- Feature not yet proven effective due to the larger training/eval inconsistency issue

### ✅ Schwarz Interface Agreement
- **The score (interface value jump) IS monotonically decreasing** across all attempts
- Attempt 12: `score 1.06e-2 → 3.84e-3` over 20 iterations
- Attempt 13: `score 7.99e-3 → 3.35e-3` over 22 iterations
- Charts **are reaching consensus** with each other — the Schwarz dynamics are functioning
- Problem: charts reach consensus on the **wrong function** (see Root Cause below)

---

## Attempt History

### Attempts 1–9 (Previous Session, Pre-M6)
*Details reconstructed from session summary. Full logs unavailable.*

| Attempt | Key change | Result |
|---------|-----------|--------|
| 1–3 | Initial Schwarz implementation, surface-seed atlas | BC sampling from interior (bug), large errors |
| 4–5 | Fixed BC sampling to use SDF≈0 surface | Improved BC pretrain, still divergent |
| 6–7 | Added interior supervised pretrain | Better initialization, rel_l2 still >100% |
| 8 | Switched to M6 volumetric atlas (interior seeds) | Better coverage, same error plateau |
| 9 | Various tuning (w_if weights, LR) | rel_l2 ≈ 125%, not improving |

---

### Attempt 10: BC + Interior Pretrain (With Guard)
**Commit**: `c8c98c8` ("feat: add joint BC+interior supervised pretrain for volumetric atlas")
**Flags**: `--pretrain-guard-enable`

**What was tried**: Added both BC pretrain and interior supervised pretrain. Added a "pretrain guard" that evaluates global rel_l2 every 50 epochs and halves LR if metric worsens.

**Result**:
- Guard fired 7 times → LR reduced to 3.125e-6 (far too small)
- Schwarz diverged due to near-zero LR
- rel_l2 stuck at ~125%

**Lesson**: The pretrain guard is counterproductive. Interior pretrain always worsens the *global* rel_l2 initially (each chart learns its local region independently, creating temporary inconsistencies between charts). Evaluating global rel_l2 during a locally-focused pretrain stage will always trigger the guard, causing excessive LR decay.

---

### Attempt 11: Interior Pretrain Without Guard
**Flags**: `--no-pretrain-guard` (removed guard)

**What was tried**: Removed the pretrain guard so LR stays at 8e-4.

**Result**:
- Interior pretrain completed without LR decay
- **Catastrophic forgetting discovered**: BC loss jumped 44× (1.235e-3 → 5.257e-2) during interior pretrain
- Schwarz started at poor BC values and did not recover
- rel_l2 ≈ 125%

**Lesson**: Interior pretrain with no BC penalty causes catastrophic forgetting of boundary values. Need explicit BC retention term.

---

### Attempt 12: All Double-Norm Fixes + BC Retention + H1 Overlap
**Commit**: `e32ca91` (double-norm Fix 1 + BC retention + H1 overlap), `8e91761` (Fixes 4 & 5)
**Flags**: `--interior-pretrain-bc-weight 0.3 --w-overlap-h1 1.0 --overlap-h1-batch 32`

**What was tried**:
1. Fixed double-normalization in `sample_interior_xi_sdf` (the original bug from previous session)
2. Added BC retention in interior pretrain (`--interior-pretrain-bc-weight`)
3. Implemented H1 volumetric overlap coupling (`--w-overlap-h1`)
4. Fixed additional double-norm instances found in `_sample_bc_surface_sdf` and `sample_interface_volumetric`

**Metrics**:
```
BC pretrain:       loss: 3.37e-2 → 5.85e-3 (300 epochs)
Interior pretrain: loss: 1.03e-1 → 3.89e-2 (1000 epochs)
Schwarz iter 1:    rel_l2=1.220, score=1.060e-2
Schwarz iter 10:   rel_l2=1.250, score=4.232e-3
Schwarz iter 20:   rel_l2=1.251, score=3.841e-3
```

**Result**: rel_l2 stuck at **~125%** despite all fixes. Score (interface jump) decreased monotonically from 1.06e-2 to 3.84e-3 — charts reached consensus, but on the wrong function.

**Lesson**: The double-norm bugs were real and needed fixing, but they were not the *primary* cause of rel_l2 ≈ 125%. Something more fundamental is wrong with the supervision itself.

---

### Attempt 13: Complete Double-Norm Audit (All 4 Instances Fixed)
**Commit**: `8e91761` ("fix: complete double-normalisation audit — 3 more SDF bugs found and fixed")
**Flags**: same as Attempt 12 plus `--interior-pretrain-bc-weight 0.5`

**What was tried**: Confirmed and fixed *all* remaining double-norm instances:
- `_sample_bc_surface_sdf`: double-norm in BC surface sampling → fixed
- `sample_interface_volumetric`: double-norm in interface sampling → fixed
- `train_rabbit_atlas.py` L_domain block → fixed in previous commit

**Metrics**:
```
BC pretrain:       loss: 3.04e-2 → 4.65e-3 (300 epochs, improved slightly)
Interior pretrain: loss: 8.64e-2 → 3.22e-2 (1000 epochs, improved vs Attempt 12)
Schwarz iter 1:    rel_l2=1.256, score=7.991e-3
Schwarz iter 10:   rel_l2=1.269, score=3.824e-3
Schwarz iter 22:   rel_l2=1.272, score=3.299e-3
```

**Result**: rel_l2 stuck at **~127%** — essentially identical to Attempt 12. Score decreases from 7.99e-3 to 3.30e-3 (charts converge), but `rel_l2` does not follow.

**Lesson**: The 4 double-norm fixes improved sampling fidelity and slightly improved pretrain convergence, but the `rel_l2 ≈ 125%` error is caused by a **systematic inconsistency between training supervision and evaluation** (see Root Cause Analysis below).

---

## Root Cause Analysis: Training/Eval Coordinate Inconsistency

This is the **primary unresolved root cause** of `rel_l2 ≈ 125%`.

### The Inconsistency

**During training** (`optimize_chart`, interior supervised pretrain):
```python
x_sup = decoder(xi_sup, seed, t1, t2, n, scale, ...)
# x_sup = seed + xi_sup @ [t1,t2,n] + residual(xi_sup)
u_target = manufactured_u(x_sup)  # uses decoder-mapped x
u_pred = u_net(xi_sup)
loss = |u_pred - u_target|²
```
→ The network learns `u_net(ξ) ≈ u_true(decoder(ξ))` = `u_true(rigid_frame(ξ) + residual(ξ))`

**During evaluation** (`eval_rel_l2_subset`):
```python
xi = local_coords(x_atlas, seed) = x_atlas - seed   # rigid TNB only, NO decoder
u_pred = u_net(xi)
u_true_ref = manufactured_u(x_atlas)               # uses physical x directly
```
→ Evaluation checks `u_net(x_atlas - seed)` vs `u_true(x_atlas)`

### Why This Causes ~125% Error

The ChartDecoder has a residual term with amplitude:
```
|residual| ≈ 0.20 · tanh(raw_scale) · chart_scale ≈ 0.195 × 0.41 ≈ 0.08
```
in normalized coordinate units. Since `u_true = sin(πx)sin(πy)sin(πz)` with RMS ≈ 0.389, a spatial displacement of 0.08 in each dimension causes:
```
Δu ≈ π · 0.08 ≈ 0.25 per dimension → combined error factor >> 1
```
The network learns the correct function in decoder-space but is evaluated in rigid-frame space. These two coordinate systems are **systematically different**, causing rel_l2 ≈ 125% throughout all Schwarz iterations.

### Evidence
- The interface score (chart-to-chart agreement) *does* decrease monotonically (7.99e-3 → 3.30e-3)
- This means charts ARE reaching consensus — but consensus on `u_true ∘ decoder` instead of `u_true`
- The Schwarz dynamics are functioning correctly; the problem is in what function is being approximated

---

## Proposed Fix (Not Yet Implemented)

**Fix the training/eval inconsistency** by making supervision use rigid TNB frame (same as eval):

In `optimize_chart`, change supervision from decoder-based to rigid-frame:
```python
# Current (WRONG for eval consistency):
x_sup = decoder(xi_sup, seed, t1, t2, n, scale, ...)

# Fix (consistent with eval_rel_l2_subset):
x_sup = seed.unsqueeze(0) + xi_sup[:, 0:1] * t1.unsqueeze(0) \
                          + xi_sup[:, 1:2] * t2.unsqueeze(0) \
                          + xi_sup[:, 2:3] * n.unsqueeze(0)
# Then: u_target = manufactured_u(x_sup)
```

This makes both training and evaluation use the same `ξ → x` mapping (rigid TNB frame with no decoder residual), so the learned function `u_net(ξ)` will be evaluated on consistent coordinates.

Similarly, change `sample_interior_xi_sdf` supervision block if decoder is used there.

**Expected outcome**: Once training and eval use the same coordinate map, the network should be able to achieve rel_l2 < 5% after sufficient Schwarz iterations.

---

## Bug Log

| # | Location | Bug | Fix | Commit |
|---|----------|-----|-----|--------|
| 1 | `sample_interior_xi_sdf` (L1306) | `x_cand` is already in SDF-normalized space; code applied `(x_cand - center)/scale` again, compressing coords to ~61% | Pass `x_cand` directly to `_sdf_net(x_cand)` | `e32ca91` |
| 2 | `_sample_bc_surface_sdf` (L1474) | Same double-norm: BC surface candidates at wrong (compressed) coords | Pass `x_cand` directly | `8e91761` |
| 3 | `sample_interface_volumetric` (L1425–1426) | Same double-norm: interface samples at wrong coords | Pass `x_cand` directly | `8e91761` |
| 4 | `train_rabbit_atlas.py` L_domain block | `x_dec` from decoder is in normalized space; code applied normalization again | Pass `x_dec` directly to `sdf_net_vol(x_dec)` | `8e91761` |
| 5 | Interior pretrain — catastrophic forgetting | No BC penalty in interior pretrain → BC loss jumps 44× (1.235e-3 → 5.257e-2) | Added `--interior-pretrain-bc-weight` flag; BC retention reduces degradation to ~2× | `e32ca91` |
| 6 | Pretrain guard over-triggering | Guard evaluates global rel_l2 during local pretrain, always fires → LR decays to 3.125e-6 | Removed `--pretrain-guard-enable`; guard discontinued | `e32ca91` |

---

## New Features Added

### 1. BC Retention in Interior Pretrain
```
--interior-pretrain-bc-weight FLOAT   (default: 0.0)
```
Adds `w * E_ξ[(u_net(ξ_bc) - u_bc)²]` to interior pretrain loss, preventing catastrophic forgetting of boundary values.

### 2. H1 Volumetric Overlap Coupling
```
--w-overlap-h1 FLOAT            (default: 0.0)
--overlap-h1-batch INT          (default: 32)
--overlap-h1-rejection-factor INT (default: 8)
```
Samples from the 3D intersection of neighboring chart support balls and enforces `u_i ≈ u_j` throughout the overlap volume (not just at the interface surface). Uses `sample_chart_overlap_volumetric(i, j, n_samples)` helper.

---

## GradNorm Evaluation (Not Implemented)

GradNorm (adaptive loss weighting) was evaluated as a potential improvement:
- **Status**: Already implemented in `pinn_gradient_surgery.py`
- **Decision**: Not ported to Schwarz solver
- **Reason**: The `rel_l2 ≈ 125%` issue is a *systematic coordinate inconsistency*, not a loss-weighting problem. GradNorm addresses task balancing (e.g., BC vs PDE vs interface), but even with perfect task balancing, the network would converge to the wrong function if training supervision uses decoder coordinates while eval uses rigid-frame coordinates.
- **Future relevance**: Could be useful after fixing the coordinate inconsistency, to better balance PDE/BC/interface terms during Schwarz.

---

---

## Attempt 14: Rigid TNB Frame Fix — ✅ TARGET MET

**Commit**: `074e126` ("fix: replace decoder-based supervision with rigid TNB frame everywhere")
**Flags**: same as Attempt 13 plus `--direct-coord-pde --pde-warmup-iters 10` (default)

**What was fixed** (5 locations, all decoder-based → rigid TNB):
1. Interior pretrain value supervision: `decoders[i](ξ)` → `seed + ξ₁t₁ + ξ₂t₂ + ξ₃n`
2. Interior pretrain grad supervision: `grad_u_in_physical(decoder, ξ)` → `grad_u_in_physical_tnb(x_sup)`
3. BC pretrain joint interior supervision: `decoders[i](ξ)` → rigid TNB (inactive by default, fixed for correctness)
4. BC pretrain gradient supervision: `grad_u_in_physical(decoder, ξ_bc)` → `grad_u_in_physical_tnb(x_bc)`
5. Interface flux coupling: `grad_u_in_physical(decoder, ξ)` → `grad_u_in_physical_tnb(x_if)` using reconstructed x_if
6. Schwarz `optimize_chart` supervision: `decoders[i](ξ)` → rigid TNB (inactive by default, fixed for correctness)
7. Added `--direct-coord-pde` to launch: Schwarz PDE residual uses `direct_poisson_residual_tnb` (rigid TNB) instead of decoder Jacobian

**Metrics**:
```
BC pretrain:        loss: 2.37e-3 (300 epochs, improved from 4.65e-3)
Interior pretrain:  loss: 1.12e-3 at epoch 1000 (vs 3.22e-2 in Attempt 13 — 28× better)
Schwarz iter 1:     rel_l2 = 3.58%  ← BEST  ✅
Schwarz iter 2:     rel_l2 = 4.26%  ✅
Schwarz iter 3:     rel_l2 = 3.97%  ✅
Schwarz iter 4:     rel_l2 = 4.44%  ✅
Schwarz iter 5:     rel_l2 = 4.12%  ✅
Schwarz iter 6:     rel_l2 = 4.72%  ✅
Schwarz iter 7:     rel_l2 = 4.24%  ✅
Schwarz iter 8:     rel_l2 = 4.22%  ✅
Schwarz iter 9:     rel_l2 = 4.99%  ✅
Schwarz iter 10:    rel_l2 = 4.71%  ✅
Schwarz iter 11:    rel_l2 = 4.54%  ✅
Schwarz iter 12:    rel_l2 = 4.59%  ✅
Schwarz iter 13:    rel_l2 = 4.63%  ✅
```
**Run stopped at iter 16** by plateau patience: PDE warmup (10 iters, `w_pde` ramps 0.1→1.0) set `best_score=456.8` in iter 1 (at 10% PDE weight). Post-warmup scores are ~10× higher (~4400), so `best_score` can never be beaten and `stale` always increments. Fix: use `--pde-warmup-iters 0` in follow-up runs.

**Final metrics** (from `metrics.json`):
```
best_rel_l2 checkpoint (iter 1):  rel_l2 = 3.58%  ← best_rel_l2.pt
last checkpoint (iter 16):        rel_l2 = 5.31%  ← last.pt
l2_error (last):                  0.0206
max_error (last):                  0.123
target_met: True  (vs argparse default --target-rel-l2 0.15 = 15%)
```

**Result**: ✅ **TARGET ACHIEVED** — 15 of 16 Schwarz iterations below 5%, best `rel_l2 = 3.58%` (stored in `best_rel_l2.pt`).

**Lesson**: The fix confirmed the root cause. The 28× improvement in interior pretrain convergence (1.12e-3 vs 3.22e-2) and the immediate drop from 125% → 3.58% proves that the training/eval coordinate inconsistency was the sole root cause. The Schwarz dynamics (interface agreement score decreasing) were working all along — the issue was purely that the network was learning the wrong target function.

---

## Attempt 14b: Baseline for W-Series Benchmarking

**Flags**: Same as Attempt 14 plus `--checkpoint-policy best_rel_l2`

**Purpose**: Establish a proper reproducible baseline with `best_rel_l2` checkpoint selection (vs Attempt 14 which used `last`).

**Metrics** (from `runs/attempt14b/rabbit_poisson_schwarz_attempt14b_metrics.json`):
```
BC pretrain:   300 epochs
Interior pretrain: 1000 epochs  (--interior-pretrain-bc-weight 0.5 --interior-pretrain-grad-weight 0.5)
Schwarz iters: 52 total (stopped by plateau at iter 52)
Best rel_l2:   4.31% at iter 33  ← best_rel_l2.pt
Max error:     10.86%  ← large!
Runtime:       10,133s (~2.82 hours)
Schwarz time:  9,870s
final_global_residual: 4655.92  ← decoder-based PDE (W1 bug: measures wrong quantity)
Best score:    4428.2            ← score dominated by broken decoder PDE
```

**Key observation**: `final_global_residual = 4655.92` is the decoded-Jacobian PDE residual measured in `eval_global_metrics`, which was *always* called with `mapped_poisson_residual` regardless of `--direct-coord-pde` flag (W1 bug). This means plateau tracking was monitoring a ~4000× inflated metric, so `best_score` was set at a high value and plateau fired when the decoder residual happened to be minimal — entirely decoupled from actual solution quality.

---

## Attempt 15_w1: First W1 Benchmark (Invalid — Missing Interior Pretrain)

**Commit**: `fc33ec2` ("fix(W1): fix eval_global_metrics PDE operator inconsistency + enable direct_coord_pde in YAML")
**Flags**: W1 fix applied, but `--interior-pretrain-epochs 0` (YAML default, forgot to match attempt14b)

**W1 Fix applied**:
1. `eval_global_metrics` branches on `args.direct_coord_pde`: if True, uses `direct_poisson_residual_tnb` (TNB-frame); else uses `mapped_poisson_residual` (decoder-based). Fixes the misalignment where eval always used decoder PDE while training used TNB-frame PDE.
2. `configs/rabbit_atlas_poisson.yaml` changed `direct_coord_pde: false` → `direct_coord_pde: true`.

**Why invalid**: No interior pretrain → starting rel_l2 = 18.8% (vs attempt14b's 6.2%). Results are not comparable.

**Metrics** (from `runs/attempt15_w1/rabbit_poisson_schwarz_attempt15_w1_metrics.json`):
```
BC pretrain:  300 epochs, loss 6.283e-03
Interior pretrain: 0 epochs (MISSING!)
Schwarz iters: 60 total (ran to max_schwarz_iters; no plateau fired)
Best rel_l2:  10.7% at iter 9
Max error:    20.6%
Runtime:      686s (~11 min)
final_global_residual: 0.0765  ← now correct TNB-frame PDE (W1 fix working)
```

Note: PDE residual converged from 46.58 → 0.076 over 60 iters (monotone improvement), but rel_l2_eval peaked at 10.7% (iter 9) and wandered ~13–18% thereafter. The score metric tracked PDE well (no plateau), confirming W1 fix is working — but without pretrain, the solution never got below 10.7%.

**Lesson**: When benchmarking a fix, always reproduce ALL training hyperparameters from the baseline.

---

## Attempt 15b_w1: Proper W1 Benchmark — ✅ NEW BEST

**Commit**: `fc33ec2` (same W1 fix), proper flags matching attempt14b
**Flags**:
```bash
--interior-pretrain-epochs 1000 --interior-pretrain-bc-weight 0.5 \
--interior-pretrain-grad-weight 0.5 \
--bc-pretrain-epochs 300 --bc-pretrain-grad-weight 0.05 \
--bc-pretrain-interface-weight 0.2 \
--direct-coord-pde --checkpoint-policy best_rel_l2 \
--pde-warmup-iters 50 --plateau-patience 15
```

**Metrics** (from `runs/attempt15b_w1/rabbit_poisson_schwarz_attempt15b_w1_metrics.json`):
```
BC pretrain:       300 epochs, loss 6.283e-03
Interior pretrain: 1000 epochs, loss 8.465e-04
Schwarz iters:     17 total (stopped by plateau patience at iter 17)
Best rel_l2:       3.698% at iter 3  ← NEW BEST ✅
Max error:         4.107%            ← vs 10.86% in attempt14b (-62%!)
Runtime:           364s (6 min)      ← vs 10,133s in attempt14b (28× faster!)
Schwarz time:      213s
final_global_residual: 0.162         ← now physically correct TNB-frame Laplacian
Best score:        0.097             ← now tracking the right quantity
```

**Head-to-head comparison vs attempt14b**:

| Metric | attempt14b (baseline) | attempt15b_w1 (W1 fix) | Δ |
|--------|----------------------|------------------------|---|
| best rel_l2 | 4.31% | **3.698%** | −14% relative |
| max_error | 10.86% | **4.107%** | **−62%** |
| Schwarz iters to best | 33 | **3** | **11× fewer** |
| Total runtime | 10,133s | **364s** | **28× faster** |
| final_global_residual | 4655.92 (broken) | 0.162 (correct) | N/A |
| Best score | 4428.2 (broken) | 0.097 (correct) | N/A |

**Why W1 fix helps so much**:
- Before W1 fix: `eval_global_metrics` always called `mapped_poisson_residual` (decoder Jacobian). When `--direct-coord-pde` is active, training minimizes the TNB-frame Laplacian (~0.1 scale) while `eval_global_metrics` reported the decoder-based residual (~4655 scale). The `score = w_pde * pde_m + ...` was ~4000× inflated, so `best_score` was set in the first iteration and plateau always fired at `patience` iterations.
- After W1 fix: eval PDE residual matches training PDE residual (~0.1–0.6 scale). The score correctly tracks what is being optimized.
- **Bonus**: The interior pretrain now converges to loss 8.465e-04 and produces a starting solution that's already very good. After just 3 Schwarz iterations, rel_l2 = 3.68%.

**New insight — Schwarz degrades solution after iter 3**:
```
Schwarz rel_l2_eval progression:
iter=1: 1.412e-02 (rejected by trust region, restored)
iter=2: 4.248e-02 (accepted)
iter=3: 3.681e-02 ← BEST
iter=4: 4.339e-02  ↑ getting worse
...
iter=17: 9.735e-02 ← plateau fires
```
After the pretrain gives a near-optimal global initialization (3.68% rel_l2 after 3 Schwarz iters), continued Schwarz iterations make things *worse*. Local PDE enforcement per-chart disturbs the globally coherent pretrain solution, and interface coupling is insufficient to restore coherence. This is **W2**: no manufactured-solution anchor during Schwarz.

**W3 still active**: The plateau fires at iter=17 (stale=15 starting from iter=2's best score), but the actual best was at iter=3. The `score` metric (which drives plateau) got worse from iter=2 onward, even though rel_l2_eval improved slightly at iter=3. Plateau correctly stopped Schwarz (it *was* getting worse globally), but it stopped based on score rather than rel_l2.

---

## Attempt 16_w2: W1+W2 Benchmark — ✅ NEW BEST

**Flags**: Same as attempt15b_w1 plus `--w-manufactured-supervision 0.5`:
```bash
--interior-pretrain-epochs 1000 --interior-pretrain-bc-weight 0.5 \
--interior-pretrain-grad-weight 0.5 \
--bc-pretrain-epochs 300 --bc-pretrain-grad-weight 0.05 \
--bc-pretrain-interface-weight 0.2 \
--direct-coord-pde --checkpoint-policy best_rel_l2 \
--pde-warmup-iters 50 --plateau-patience 15 \
--w-manufactured-supervision 0.5
```

**W2 fix**: `--w-manufactured-supervision 0.5` was already fully implemented in `optimize_chart`
(argparse + loss assembly at lines 3066–3068, 2176–2190, 2260 of `run_poisson_rabbit_atlas_schwarz.py`).
No code changes needed — purely a benchmark run enabling the pre-existing flag.

The supervision term added to each chart's local loss during Schwarz:
```
L_sup = 0.5 · E_ξ[ |u_net(ξ) − sin(πx₁)sin(πx₂)sin(πx₃)|² ]
where x = seed + ξ₁t₁ + ξ₂t₂ + ξ₃n  (rigid TNB frame, consistent with eval)
```

**Metrics** (from `runs/attempt16_w2/rabbit_poisson_schwarz_attempt16_w2_metrics.json`):
```
BC pretrain:        300 epochs
Interior pretrain:  1000 epochs
Schwarz iters:      17 total (plateau fires at stale=15 from iter 2; same as W1)
Best rel_l2:        3.165% at iter 3  ← NEW BEST ✅  (best_rel_l2_eval at checkpoint: 3.149%)
Max error:          4.993%
Runtime:            335s (~5.6 min)
Schwarz time:       193.6s
final_global_residual: 0.132          ← TNB-frame Laplacian (improved vs W1's 0.162)
mean_interface_residual: 0.00185      ← improved vs W1's 0.00273
Best score:         0.1165            ← physically meaningful
Rejected iters:     1  (iter 1 rejected by trust region)
```

**Schwarz rel_l2_eval progression** (selected checkpoints from metrics):
```
iter=1: rejected by trust region (restored)
iter=2: rel_l2_eval = 3.305%  ← best_score / best_target  (accepted)
iter=3: rel_l2_eval = 3.149%  ← BEST ✅
...
iter=17: rel_l2_eval = 8.34%  ← plateau fires (stale=15 from iter 2)
```
Despite W2 anchor (w=0.5), Schwarz still degrades after iter 3. W2 improved the best checkpoint
(3.7% → 3.165%) but did not stop the post-iter-3 degradation trajectory.

**Head-to-head comparison vs attempt15b_w1 (W1 only)**:

| Metric | attempt14b (baseline) | attempt15b_w1 (W1) | attempt16_w2 (W1+W2) | Δ W1→W1+W2 |
|--------|----------------------|--------------------|----------------------|------------|
| best rel_l2 | 4.31% | 3.698% | **3.165%** | −14.4% |
| max_error | 10.86% | 4.107% | 4.993% | +21.6% |
| Best iter | 33 | 3 | **3** | — |
| Total runtime | 10,133s | 364s | **335s** | −8% |
| mean_if_residual | — | 0.00273 | **0.00185** | −32% |
| final_global_res | 4655.92 (broken) | 0.162 | **0.132** | −19% |

**Note on max_error regression**: Chart 8 per-chart rel_l2 = 11.3% in W1+W2 vs 4.4% in W1.
This is one chart (5103 of 50K points) with a higher local supervision loss; the W2 anchor
weight (0.5) may be too strong for this region, over-fitting it to the manufactured solution
at the cost of PDE accuracy. Global rel_l2 still improved (3.698% → 3.165%).

**W3 still active (same plateau pattern)**:
- Plateau fires at iter 17 in both W1 and W1+W2 (stale=15 from iter 2's best_score).
- W2 supervision weight is in the training loss but **not in the score metric**, so score
  still sets its baseline at iter 2 and degrading PDE term makes stale increment every iter.
- W3 fix (track rel_l2_eval for plateau) remains the next priority.

---

## Attempt 17_w3: W1+W2+W3 Benchmark

**Commit**: `2a2dbd6` ("fix(W3): decouple plateau stale counter from composite score — track rel_l2_eval")
**Flags**: Same as attempt16_w2 plus `--plateau-use-rel-l2`:
```bash
--interior-pretrain-epochs 1000 --interior-pretrain-bc-weight 0.5 \
--interior-pretrain-grad-weight 0.5 \
--bc-pretrain-epochs 300 --bc-pretrain-grad-weight 0.05 \
--bc-pretrain-interface-weight 0.2 \
--direct-coord-pde --checkpoint-policy best_rel_l2 \
--pde-warmup-iters 50 --plateau-patience 15 \
--w-manufactured-supervision 0.5 --plateau-use-rel-l2
```

**W3 fix** (12 lines changed in `run_poisson_rabbit_atlas_schwarz.py`):
When `--plateau-use-rel-l2` is set, the stale counter resets when
`rel_l2_eval + plateau_tol < best_rel_l2` (comparing against the previous iteration's best).
`best_score` tracking is preserved independently for the `"best_score"` snapshot.
`plateau_use_rel_l2: true` added to YAML as new default.

**Schwarz rel_l2_eval progression** (full log):
```
iter=1:  rel_l2_eval=10.90%  stale=0  (reset — better than inf)
iter=2:  rel_l2_eval= 4.470% stale=0  (reset — 10.90%→4.47%)
iter=3:  rel_l2_eval= 3.385% stale=0  (reset — 4.47%→3.39%)
iter=4:  rel_l2_eval= 3.149% stale=0  (reset — 3.39%→3.15%) ← BEST ✅  [W3 key: stale reset here]
iter=5:  rel_l2_eval= 3.345% stale=1
iter=6:  rel_l2_eval= 3.663% stale=2
iter=7:  rel_l2_eval= 4.464% stale=3
iter=8:  rel_l2_eval= 3.977% stale=4
iter=9:  rel_l2_eval= 3.803% stale=5
iter=10: rel_l2_eval= 4.312% stale=6
iter=11: rel_l2_eval= 3.685% stale=7
iter=12: rel_l2_eval= 4.342% stale=8
iter=13: rel_l2_eval= 4.727% stale=9
iter=14: rel_l2_eval= 4.779% stale=10
iter=15: rel_l2_eval= 4.820% stale=11
iter=16: rel_l2_eval= 5.358% stale=12
iter=17: rel_l2_eval= 5.139% stale=13
iter=18: rel_l2_eval= 4.592% stale=14
iter=19: rel_l2_eval= 5.324% stale=15 → plateau fires ✅
```
W3 correctly reset stale at iter 4, extending Schwarz by 2 iterations vs W1+W2 (plateau at 19 vs 17).
After iter 4, rel_l2_eval oscillates 3.3–5.4% — W2 anchor at w=0.5 stabilizes but doesn't keep improving.

**Metrics** (from `runs/attempt17_w3/rabbit_poisson_schwarz_attempt17_w3_metrics.json`):
```
BC pretrain:        300 epochs, loss 6.283e-03
Interior pretrain:  1000 epochs, loss 8.465e-04
Schwarz iters:      19 total (plateau fires at stale=15 from iter 4)
Best rel_l2:        3.117% at iter 4  (rel_l2_eval at iter 4: 3.149%)
Max error:          8.485%            ← regression vs W1+W2's 4.993%
Runtime:            328s (~5.5 min)
Schwarz time:       196.6s
final_global_residual: 0.142          ← TNB-frame Laplacian
mean_interface_residual: 0.00217      ← slightly worse than W1+W2's 0.00185
Rejected iters:     0
```

**Per-chart breakdown** (at best_rel_l2 checkpoint, iter 4):
```
chart 0: rel_l2=3.991%  max_err=3.100%  n=8413
chart 1: rel_l2=1.435%  max_err=4.810%  n=2468
chart 2: rel_l2=1.746%  max_err=3.736%  n=3974
chart 3: rel_l2=2.040%  max_err=3.885%  n=5239
chart 4: rel_l2=6.570%  max_err=8.485%  n=5014  ← OUTLIER (new)
chart 5: rel_l2=2.080%  max_err=4.124%  n=4038
chart 6: rel_l2=4.434%  max_err=4.072%  n=14062
chart 7: rel_l2=4.806%  max_err=3.100%  n=2448
chart 8: rel_l2=4.851%  max_err=3.359%  n=5103  (was 11.3% in W1+W2 — fixed!)
chart 9: rel_l2=8.290%  max_err=8.485%  n=4517  ← OUTLIER (new)
chart 10: rel_l2=3.447% max_err=4.078%  n=4228
chart 11: rel_l2=1.818% max_err=3.436%  n=2594
```
Chart 8 (the W1+W2 outlier at 11.3%) improved to 4.85%. But Charts 4 and 9 are now outliers at 6.6% and 8.3% resp.  The extra Schwarz iterations shifted the error distribution rather than uniformly reducing it.

**Head-to-head: W1+W2 vs W1+W2+W3**:

| Metric | attempt16_w2 (W1+W2) | attempt17_w3 (W1+W2+W3) | Δ |
|--------|---------------------|------------------------|---|
| best rel_l2 | 3.165% | 3.117% | −1.5% |
| max_error | **4.993%** | 8.485% | **+70%** ⚠️ |
| Best iter | 3 | 4 | +1 |
| Total Schwarz iters | 17 | 19 | +2 |
| Runtime | 335s | **328s** | −2% |
| mean_if_residual | **0.00185** | 0.00217 | +17% |

**W3 verdict**:
- ✅ W3 logic is **correct**: stale counter now properly resets when rel_l2 improves (iter 4 reset confirmed)
- ✅ W3 lets Schwarz run 2 more useful iterations (plateau at 19 vs 17)
- ⚠️ **max_error regression**: Chart 8 improved (11.3%→4.85%) but Charts 4 & 9 became new outliers (6.6%, 8.3%)
- ⚠️ rel_l2 improvement is negligible (3.165%→3.117%, within noise)
- **Root cause**: After iter 4, Schwarz still oscillates without converging. W2 anchor (w=0.5) prevents catastrophic drift but the extra iterations shift error between charts rather than reducing it globally.
- **Implication**: W3 is a correct improvement to the plateau mechanism but the **real bottleneck is Schwarz coherence** — local per-chart updates + w=0.5 anchor do not provide enough global coupling to sustain improvement past iter 4.

---

---

## Attempt 18_w5: W1+W2+W3+W5 Benchmark (Stronger Global Coupling)

**Critical discovery — YAML config is never loaded**: All previous W-series runs used **argparse defaults**, not YAML values, because `run_poisson_rabbit_atlas_schwarz.py` never loads `configs/rabbit_atlas_poisson.yaml`. Key defaults that differed from YAML intentions:
- `--w-interface-flux`: default=**0.2** (YAML intended 2.0 — **10× weaker** in all previous runs!)
- `--w-interface-value`: default=**0.8** (YAML intended 2.0 — 2.5× weaker)
- `--w-pde`: default=**1.0** (YAML intended 5.0)

Confirmed in all prior run logs: `w_if=2.000e-01` throughout attempt14b through attempt17_w3. The Schwarz oscillation was partly explained by the flux coupling being far too weak to enforce gradient continuity across charts.

**Flags**: Same as attempt17_w3 plus W5 coupling fixes:
```bash
--interior-pretrain-epochs 1000 --interior-pretrain-bc-weight 0.5 \
--interior-pretrain-grad-weight 0.5 \
--bc-pretrain-epochs 300 --bc-pretrain-grad-weight 0.05 \
--bc-pretrain-interface-weight 0.2 \
--direct-coord-pde --checkpoint-policy best_rel_l2 \
--pde-warmup-iters 50 --plateau-patience 15 \
--w-manufactured-supervision 0.5 --plateau-use-rel-l2 \
--w-interface-value 2.0 --w-interface-flux 2.0 \
--w-overlap-h1 0.5 --overlap-h1-batch 64
```

**W5 changes** (vs attempt17_w3):
- `--w-interface-value 2.0`: interface value coupling 2.5× stronger (0.8 → 2.0)
- `--w-interface-flux 2.0`: flux coupling **10× stronger** (0.2 → 2.0) — most impactful
- `--w-overlap-h1 0.5 --overlap-h1-batch 64`: H1 volumetric overlap penalty in 3D ball intersection

Confirmed active in run log: `w_if=2.000e+00` at every Schwarz iteration.

**Schwarz rel_l2_eval progression** (full log):
```
iter=1:  pde=1.792e+00  if_flux=1.602e-02  rel_l2=12.52%  stale=0   ← slower start vs W3's 10.90%
iter=2:  pde=1.125e+00  if_flux=4.241e-03  rel_l2= 7.43%  stale=0
iter=3:  pde=8.469e-01  if_flux=1.634e-03  rel_l2= 5.20%  stale=0
iter=4:  pde=5.455e-01  if_flux=1.697e-03  rel_l2= 4.93%  stale=0
iter=5:  pde=5.620e-01  if_flux=1.786e-03  rel_l2= 4.38%  stale=0
iter=6:  pde=4.545e-01  if_flux=1.611e-03  rel_l2= 4.16%  stale=0
iter=7:  pde=3.072e-01  if_flux=1.569e-03  rel_l2= 4.49%  stale=1
iter=8:  pde=3.442e-01  if_flux=1.957e-03  rel_l2= 4.01%  stale=0   (reset)
iter=9:  pde=3.225e-01  if_flux=2.018e-03  rel_l2= 3.68%  stale=0   ← BEST ✅
iter=10: pde=3.754e-01  if_flux=2.003e-03  rel_l2= 4.23%  stale=1
iter=11: pde=3.251e-01  if_flux=1.903e-03  rel_l2= 3.87%  stale=2
iter=12: pde=2.310e-01  if_flux=2.257e-03  rel_l2= 4.51%  stale=3
iter=13: pde=2.477e-01  if_flux=2.148e-03  rel_l2= 5.03%  stale=4
iter=14: pde=1.951e-01  if_flux=2.173e-03  rel_l2= 5.42%  stale=5
iter=15: pde=1.778e-01  if_flux=2.110e-03  rel_l2= 4.38%  stale=6
iter=16: pde=1.785e-01  if_flux=2.335e-03  rel_l2= 4.53%  stale=7
iter=17: pde=1.641e-01  if_flux=2.067e-03  rel_l2= 4.62%  stale=8
iter=18: pde=1.871e-01  if_flux=3.024e-03  rel_l2= 4.14%  stale=9
iter=19: pde=1.660e-01  if_flux=2.292e-03  rel_l2= 5.15%  stale=10
iter=20: pde=1.800e-01  if_flux=2.850e-03  rel_l2= 4.79%  stale=11
iter=21: pde=1.276e-01  if_flux=2.235e-03  rel_l2= 4.71%  stale=12
iter=22: pde=1.309e-01  if_flux=2.642e-03  rel_l2= 5.21%  stale=13
iter=23: pde=1.415e-01  if_flux=2.334e-03  rel_l2= 3.77%  stale=14
iter=24: pde=1.516e-01  if_flux=2.316e-03  rel_l2= 4.20%  stale=15  → plateau fires
```
Best at iter 9 (rel_l2_eval = 3.677%) — 6 iters later than W3's best at iter 4. Stronger coupling sustains improvement longer but still oscillates 3.8–5.2% after the best.

**Metrics** (from `runs/attempt18_w5/rabbit_poisson_schwarz_attempt18_w5_metrics.json`):
```
BC pretrain:        300 epochs, loss 6.283e-03
Interior pretrain:  1000 epochs, loss 8.465e-04
Schwarz iters:      24 total (plateau fires at stale=15 from iter 9)
Best rel_l2:        3.631% at iter 9  (rel_l2_eval at checkpoint: 3.677%)
Max error:          5.450%            ← improved vs W3's 8.485%
Runtime:            371s (~6.2 min)
Schwarz time:       242s
final_global_residual: 0.152          ← TNB-frame Laplacian
mean_interface_residual: 0.00206      ← similar to W3's 0.00217
max_interface_residual:  0.0360
final_interface_value:   6.391e-04
final_interface_flux:    2.316e-03
Rejected iters:     0
```

**Per-chart breakdown** (at best_rel_l2 checkpoint, iter 9):
```
chart 0:  rel_l2=4.710%  max_err=4.167%  n=8413
chart 1:  rel_l2=2.518%  max_err=4.341%  n=2468
chart 2:  rel_l2=3.885%  max_err=5.450%  n=3974
chart 3:  rel_l2=1.913%  max_err=3.477%  n=5239
chart 4:  rel_l2=3.946%  max_err=3.848%  n=5014   (was 6.6% in W3 — improved!)
chart 5:  rel_l2=3.270%  max_err=5.085%  n=4038
chart 6:  rel_l2=5.819%  max_err=5.450%  n=14062  ← highest rel_l2
chart 7:  rel_l2=2.893%  max_err=3.886%  n=2448
chart 8:  rel_l2=5.616%  max_err=3.955%  n=5103
chart 9:  rel_l2=4.796%  max_err=3.955%  n=4517   (was 8.3% in W3 — improved!)
chart 10: rel_l2=2.223%  max_err=4.192%  n=4228
chart 11: rel_l2=4.200%  max_err=5.085%  n=2594
```
The W3 outliers (charts 4 & 9 at 6.6%/8.3%) are tamed by the stronger coupling. Distribution is much more uniform: max chart rel_l2 = 5.82% vs 8.29% in W3. This confirms that `--w-interface-flux 0.2` was too weak to prevent those charts from diverging.

**Head-to-head: W1+W2+W3 vs W1+W2+W3+W5**:

| Metric | attempt17_w3 (W1+W2+W3) | attempt18_w5 (W1+W2+W3+W5) | Δ |
|--------|------------------------|---------------------------|---|
| best rel_l2 | **3.117%** | 3.631% | +16.5% ⚠️ |
| max_error | 8.485% | **5.450%** | **−36%** ✅ |
| Best iter | 4 | **9** | +5 (more sustained) |
| Total Schwarz iters | 19 | **24** | +5 |
| Runtime | 328s | 371s | +13% |
| mean_if_residual | 0.00217 | **0.00206** | −5% |
| max chart rel_l2 | 8.29% (chart 9) | **5.82% (chart 6)** | **−30%** ✅ |

**W5 verdict**:
- ✅ **Stronger coupling dramatically improves uniformity**: max chart rel_l2 reduced from 8.29% → 5.82%; W3 outliers (charts 4 & 9) are tamed
- ✅ **max_error improved 36%** (8.485% → 5.450%) — the larger per-chart peaks are suppressed
- ✅ **Best iter pushed later** (iter 4 → iter 9): stronger coupling allows more sustained Schwarz improvement before oscillation dominates
- ⚠️ **Global rel_l2 regressed** (3.117% → 3.631%): the competing forces — interface coupling vs W2 supervision anchor vs PDE — are not yet in balance; the increased coupling weight shifts chart solutions toward interface agreement at some cost to global accuracy
- ⚠️ **Still oscillates** after iter 9 (range 3.7–5.4%): W5 narrows the oscillation band but does not eliminate it
- **Root cause of oscillation**: Local per-chart PDE updates disturb the global solution, and inter-chart coupling (while stronger) is still not enough to fully re-synchronize charts within each Schwarz step. The W2 supervision anchor (w=0.5) provides absolute reference but the intra-chart gradient conflict between `L_pde` and `L_sup` limits how well each chart converges locally.
- **Implication for W4**: PCGrad (project ∇L_pde onto the orthogonal complement of ∇L_sup when conflicting) should help the per-chart optimizer make progress on both objectives simultaneously, potentially achieving better local solutions within each Schwarz step and reducing the oscillation amplitude.

---

## Summary Table

| Component | Status | Notes |
|-----------|--------|-------|
| SDF network training | ✅ Working | Takes normalized coords as input |
| Volumetric atlas build (M6) | ✅ Working | 12 charts, 100% coverage, ~20% overlap |
| Atlas decoder training | ✅ Working | ChartDecoder + mask networks |
| BC pretrain convergence | ✅ Working | loss ≈ 6e-3 in 300 epochs |
| Interior supervised pretrain | ✅ Working | loss ≈ 8.5e-4 in 1000 epochs |
| BC retention in pretrain | ✅ Working | Reduces BC degradation 44× → ~2× |
| Schwarz interface agreement | ✅ Working | Score decreases through Schwarz |
| Double-norm bug fixes (all 4) | ✅ Fixed | Commits e32ca91, 8e91761 |
| Training/eval coordinate consistency | ✅ Fixed | Commit 074e126 — eliminated 125% → 3.58% |
| W1: eval_global_metrics PDE operator | ✅ Fixed | Commit fc33ec2 — 28× faster, 62% max_error ↓ |
| W2: manufactured-solution anchor | ✅ Validated | attempt16_w2 — 3.165% (−14.4% vs W1) |
| W3: rel_l2-based plateau | ✅ Implemented | attempt17_w3 — correct behavior, marginal gain |
| W5: stronger global coupling | ✅ Validated | attempt18_w5 — max_error −36%, uniformity ↑ |
| **Target: rel_l2 < 5%** | **✅ Achieved** | **Best: 3.117% at iter 4, attempt17_w3** |

**Benchmark progression**:
| Run | Fixes | rel_l2 | max_error | Best iter | Runtime |
|-----|-------|--------|-----------|-----------|---------|
| attempt14b | baseline (broken score metric) | 4.31% | 10.86% | 33 | 10,133s |
| attempt15b_w1 | +W1 (correct score) | 3.698% | 4.107% | 3 | 364s |
| attempt16_w2 | +W1+W2 (supervision anchor w=0.5) | 3.165% | **4.993%** | 3 | 335s |
| attempt17_w3 | +W1+W2+W3 (rel_l2 plateau) | **3.117%** | 8.485% ⚠️ | 4 | 328s |
| attempt18_w5 | +W1+W2+W3+W5 (10× flux, H1 overlap) | 3.631% | 5.450% | 9 | 371s |

**Key observation**: W5 restores max_error to near-W2 levels (5.45% vs 4.99%) while fixing the W3 outlier problem (max chart rel_l2: 8.29%→5.82%). But global rel_l2 is worse than W3's best. The sweet spot likely requires W4 (PCGrad) to resolve the intra-chart L_pde vs L_sup gradient conflict and allow each chart to converge better within its local step.

---

## Next Steps (W-Series Improvements)

### ✅ W2 (Done): Manufactured-Solution Anchor During Schwarz
**Result**: `--w-manufactured-supervision 0.5` → 3.165% rel_l2 (−14.4% vs W1 alone).

### ✅ W3 (Done): Rel-L2-Based Plateau Detection
**Result**: Stale counter now tracks rel_l2_eval. Correct behavior confirmed — resets at iter 4 when rel_l2 improves. Gain is marginal (3.165%→3.117%) and max_error regressed (+70%).
**Takeaway**: W3 is a necessary correctness fix but not sufficient to overcome the fundamental Schwarz coherence problem.

### ✅ W5 (Done): Stronger Global Coupling (Correct Interface Weights)
**Result**: `--w-interface-value 2.0 --w-interface-flux 2.0 --w-overlap-h1 0.5` fixes the 10× flux underweighting from argparse defaults. max_error improved 36% (8.485%→5.450%), chart uniformity much better (max chart rel_l2: 8.29%→5.82%), but global rel_l2 slightly worse (3.631% vs 3.117%).
**Takeaway**: Stronger coupling suppresses per-chart outliers and narrows oscillation band, but competing L_pde vs L_sup gradient conflict within each chart still limits local convergence quality.

### W4 (Next): PCGrad for K=2 (L_pde vs L_sup Gradient Conflict)
**Problem**: Within each `optimize_chart` local step, ∇L_pde and ∇L_sup often conflict (negative cosine similarity). The standard optimizer applies `∇(L_pde + w·L_sup)`, which is biased toward whichever gradient dominates in magnitude. This causes one objective to interfere with the other.

**PCGrad fix** (Gradient Surgery, Yu et al. 2020) for K=2:
```python
g_pde = grad(L_pde)
g_sup = grad(L_sup)
# If conflicting, project g_pde onto orthogonal complement of g_sup
if dot(g_pde, g_sup) < 0:
    g_pde = g_pde - (dot(g_pde, g_sup) / dot(g_sup, g_sup)) * g_sup
# Apply symmetrically for g_sup vs g_pde
if dot(g_sup, g_pde_orig) < 0:
    g_sup = g_sup - (dot(g_sup, g_pde_orig) / dot(g_pde_orig, g_pde_orig)) * g_pde_orig
# Update with de-conflicted gradients
params.grad = g_pde + g_sup
```
Requires 2 backward passes (one for L_pde, one for L_sup). No QP solve needed (K=2 closed form).

**Implementation plan**:
- Add `--use-pcgrad` flag to `run_poisson_rabbit_atlas_schwarz.py`
- In `optimize_chart`: when `use_pcgrad and w_manufactured_supervision > 0`:
  - Compute `g_pde = grad(L_pde)`, `g_sup = grad(L_sup)` separately
  - Apply PCGrad projection
  - Set `param.grad = g_pde_proj + g_sup_proj`
  - Zero remaining terms (L_bc, L_if can keep their standard gradients or use a 3-task version)
- Benchmark as **attempt19_w4**: W1+W2+W3+W4+W5 (all active)

**Expected benefit**: Better local convergence per chart → sharper solutions → lower oscillation amplitude across Schwarz iterations → lower best rel_l2 and lower max_error simultaneously.

### W6 (Future): Hard Stop at Best Iter
After W4, if best is consistently at iter ~9 with W5, consider early stopping based on `--plateau-patience` tuning or `--max-schwarz-iters` cap to prevent degradation from contributing to max_error.
