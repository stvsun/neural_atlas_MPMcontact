#!/usr/bin/env bash
# ============================================================================
# New numerical example: Chart-partitioned SDF for Stanford Bunny
#
# Motivation
# ----------
# The global SDF (train_sdf_rabbit.py) uses a single anchor offset (0.25
# normalised) for sign supervision.  For the Stanford Bunny, this places
# anchors OUTSIDE the ~5mm ears, causing 31% sign errors and preventing
# Schwarz iteration convergence.
#
# This script demonstrates the chart-partitioned approach:
#   - 12 local SDFNetLocal networks (width=64, depth=4), one per atlas chart
#   - Geometry-adaptive anchor offset: offset_i = 0.30 × dist(seed_i, PLY)
#     → ear-region charts get ~1.5 mm offset, body charts get ~15 mm offset
#   - Voxel SDT initialisation (KD-tree based) before fine-tuning
#   - Eikonal fine-tuning = reinitialization PDE at steady state
#   - Gaussian PoU blending for smooth global inference
#
# Steps
# -----
# Step 1: Train 12 local SDFs + optional global adapter (~3 h)
# Step 2: Rebuild atlas using the adapter SDF           (~5 min)
# Step 3: Train atlas decoders (high overlap weight)    (~25 min)
# Step 4: Run Poisson PINN                              (~2 h)
# Step 5: Generate figures                              (~5 min)
#
# Total: ~5.5 hours
#
# Run from: PINN_coordinate_chart_3Dgeometry/
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"
ATLAS_REF="runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz"   # existing atlas for chart layout
OUT_CW="runs/bunny_sdf_chartwise"
OUT_ATLAS="runs/atlas_chartwise_vol"
OUT_ATLAS_DEC="runs/atlas_chartwise_vol_hiW"
OUT_PINN="runs/bunny_poisson_chartwise"
LOG="runs/chartwise_pipeline.log"

echo "[$(date)] ===== STEP 1: Train chart-partitioned SDF =====" | tee -a "$LOG"
# Uses the existing v3 atlas (atlas_bunny_vol_v3) for chart layout only.
# Trains 12 local networks + fits a drop-in global adapter.
#
# Key hyperparameters:
#   --anchor-offset-factor 0.30  → adaptive offset = 30% × dist(seed→surface)
#   --pretrain-epochs 400        → SDT initialisation pre-training
#   --epochs 3000                → Eikonal fine-tuning (reinitialization PDE)
#   --fit-adapter                → also fit a global SDFNet adapter for pipeline compatibility
python src/train_sdf_chartwise.py \
    --ply-file      "$PLY" \
    --atlas-data    "$ATLAS_REF" \
    --output-dir    "$OUT_CW" \
    --local-width   64 \
    --local-depth   4 \
    --sdt-grid-size 48 \
    --pretrain-epochs 400 \
    --pretrain-lr   1e-3 \
    --epochs        3000 \
    --lr            1e-3 \
    --batch-surface 1024 \
    --batch-eikonal 1024 \
    --anchor-offset-factor 0.30 \
    --coverage-factor 1.5 \
    --w-surface 10.0 \
    --w-eikonal 1.0 \
    --w-normal  3.0 \
    --w-sign    8.0 \
    --fit-adapter \
    --adapter-width  128 \
    --adapter-depth  6 \
    --adapter-epochs 3000 \
    --log-every 500
python3 -c "
import json
with open('$OUT_CW/rabbit_sdf_chartwise_meta.json') as f: m = json.load(f)
print(f'  overlap_consistency = {m[\"overlap_consistency\"]:.4f}')
print(f'  sign_error          = {m[\"sign_error\"]:.4f}')
print(f'  Per-chart offsets (mm):')
scale = m['args']['']  # args may not contain scale; read from meta
for c in m['per_chart']:
    offset_mm = c['adaptive_offset'] * 0.1557 * 1000
    print(f'    chart {c[\"chart_idx\"]:2d}: offset={c[\"adaptive_offset\"]:.4f} ({offset_mm:.1f} mm)  sign={c[\"final_sign\"]:.3e}')
" 2>/dev/null || python3 -c "
import json
with open('$OUT_CW/rabbit_sdf_chartwise_meta.json') as f: m = json.load(f)
print(f'  sign_error = {m[\"sign_error\"]:.4f}  overlap_consistency = {m[\"overlap_consistency\"]:.4f}')
for c in m['per_chart']:
    print(f'    chart {c[\"chart_idx\"]:2d}: offset={c[\"adaptive_offset\"]:.4f}  sign={c[\"final_sign\"]:.3e}')
"

echo "[$(date)] ===== STEP 2: Rebuild atlas from adapter SDF =====" | tee -a "$LOG"
# The adapter SDF is a drop-in replacement: same checkpoint format as
# train_sdf_rabbit.py (keys: model_state, model_kwargs, center, scale).
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint     "$OUT_CW/rabbit_sdf_adapter.pt" \
    --n-interior-samples 50000 \
    --n-charts           12 \
    --frame-mode         axis_aligned \
    --output-dir         "$OUT_ATLAS"

echo "[$(date)] ===== STEP 3: Train atlas decoders (w_overlap=8) =====" | tee -a "$LOG"
python experiments/train_rabbit_atlas.py \
    --atlas-data     "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --output-dir     "$OUT_ATLAS_DEC" \
    --volumetric \
    --sdf-checkpoint "$OUT_CW/rabbit_sdf_adapter.pt" \
    --epochs         6000 \
    --w-overlap      8.0 \
    --batch-overlap  2048
python3 -c "
import json
with open('$OUT_ATLAS_DEC/rabbit_atlas_gate_report.json') as f: g = json.load(f)
print(f'  gate={g[\"passed\"]}  overlap={g[\"overlap_consistency\"]:.4f}  coverage={g[\"coverage_ratio\"]:.4f}')
"

echo "[$(date)] ===== STEP 4: Poisson PINN with chartwise atlas =====" | tee -a "$LOG"
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data        "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --atlas-checkpoint  "$OUT_ATLAS_DEC/rabbit_atlas_trained.pt" \
    --output-dir        "$OUT_PINN" \
    --run-tag           bunny_poisson_chartwise \
    --pinn-arch compact --pinn-width 64 --pinn-depth 4 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --pde-warmup-iters 50 \
    --max-schwarz-iters 120 \
    --plateau-patience 20 --plateau-use-rel-l2 \
    --checkpoint-policy best_rel_l2 \
    --interface-normal-mode seed \
    --direct-coord-pde \
    --allow-failed-gate

echo "[$(date)] ===== STEP 5: Postprocess + figures =====" | tee -a "$LOG"
SOLN_NPZ=$(ls -t "$OUT_PINN"/*_solution.npz 2>/dev/null | head -1 || \
           ls -t "$OUT_PINN"/**/*_solution.npz 2>/dev/null | head -1 || echo "")
if [[ -n "$SOLN_NPZ" ]]; then
    python postprocessing/poisson_rabbit_paraview.py \
        --solution-npz "$SOLN_NPZ" \
        --atlas-npz    "$OUT_ATLAS/rabbit_atlas_data.npz" \
        --ply-file     "$PLY" \
        --output-dir   "paraview/bunny_poisson_chartwise" \
        --prefix       bunny_poisson_chartwise
    python postprocessing/render_3d_figures.py
fi

echo "[$(date)] ===== CHARTWISE PIPELINE COMPLETE =====" | tee -a "$LOG"
python3 -c "
import json, glob
mf = glob.glob('$OUT_PINN/*_metrics.json')
if mf:
    with open(mf[0]) as f: m = json.load(f)
    g = m['global']
    b = m['checkpoint_triplet']['best_rel_l2']
    print(f'  rel-L2 = {g[\"relative_l2_error\"]*100:.2f}%  best_iter = {b[\"iter\"]}')
" 2>/dev/null || echo "  (metrics not available)"
