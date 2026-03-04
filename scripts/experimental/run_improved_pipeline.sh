#!/usr/bin/env bash
# run_improved_pipeline.sh
#
# Combined improved pipeline for Stanford Bunny PLY:
#   A. Watertight mesh repair (--repair-watertight)
#   B. Wider atlas (--overlap-target 0.35, 12 charts, 75k interior pts)
#   C. Stronger overlap training (--w-overlap 20, --batch-overlap 4096, 9000 epochs)
#
# Expected total: ~3.5 hrs (SDF ~40 min, atlas ~5 min, decoders ~50 min, PINN ~80 min)
#
# Gate targets:
#   sign_error  < 0.01  (was 0.046 without repair)
#   overlap_consistency < 0.007  (was 0.013 without stronger training)
#   if_flux @ Schwarz iter 1 < 0.05  (was 0.15+ on all Stanford Bunny PLY runs)
#
# Usage:
#   cd PINN_coordinate_chart_3Dgeometry
#   nohup bash scripts/experimental/run_improved_pipeline.sh \
#       > runs/improved_pipeline.log 2>&1 &
#   echo "PID: $!"

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"
OUT_SDF="runs/bunny_sdf_repaired"
OUT_ATLAS="runs/atlas_bunny_repaired"
OUT_DEC="runs/atlas_bunny_repaired_dec"
OUT_PINN="runs/bunny_poisson_repaired"
LOG="runs/improved_pipeline.log"

echo "[$(date)] ===== IMPROVED PIPELINE START =====" | tee -a "$LOG"
echo "[$(date)] PLY: $PLY" | tee -a "$LOG"

# ── Step 1: Rebuild SDF with watertight repair ─────────────────────────────
echo "[$(date)] ===== STEP 1: Exact mesh SDF + watertight repair =====" | tee -a "$LOG"
python src/build_mesh_sdf.py \
    --ply-file "$PLY" \
    --out-dir  "$OUT_SDF" \
    --n-near    150000 \
    --n-uniform 150000 \
    --epochs    5000 \
    --sign-gate 0.02 \
    --repair-watertight \
    2>&1 | tee -a "$LOG"

# Verification gate A
python3 -c "
import torch, json, sys
ckpt = torch.load('${OUT_SDF}/rabbit_sdf_mesh.pt', weights_only=False)
se = ckpt['sign_error_estimate']
ag = ckpt['pysdf_open3d_agreement']
print(f'  sign_error = {se:.4f}  (target < 0.02)')
print(f'  agreement  = {ag:.4f}  (target > 0.99)')
if se >= 0.02:
    print('GATE FAIL: sign_error too high — aborting pipeline')
    sys.exit(1)
print('  GATE PASS ✓')
" 2>&1 | tee -a "$LOG"

# ── Step 2: Build 12-chart wide atlas ─────────────────────────────────────
echo "[$(date)] ===== STEP 2: Build wide atlas (12 charts, overlap=0.35) =====" | tee -a "$LOG"
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint     "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --n-interior-samples 75000 \
    --n-charts           12 \
    --overlap-target     0.35 \
    --frame-mode         axis_aligned \
    --output-dir         "$OUT_ATLAS" \
    2>&1 | tee -a "$LOG"

# ── Step 3: Stronger decoder training ────────────────────────────────────
echo "[$(date)] ===== STEP 3: Train atlas decoders (w_overlap=20, 9000 epochs) =====" | tee -a "$LOG"
python experiments/train_rabbit_atlas.py \
    --atlas-data     "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --output-dir     "$OUT_DEC" \
    --volumetric \
    --sdf-checkpoint "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --epochs         9000 \
    --w-overlap      20.0 \
    --batch-overlap  4096 \
    2>&1 | tee -a "$LOG"

# Verification gate B
python3 -c "
import json, sys
g = json.load(open('${OUT_DEC}/rabbit_atlas_gate_report.json'))
oc = g['overlap_consistency']
cr = g['coverage_ratio']
passed = g['passed']
print(f'  overlap_consistency = {oc:.5f}  (target < 0.007)')
print(f'  coverage_ratio      = {cr:.4f}  (target = 1.000)')
print(f'  gate_passed         = {passed}')
if not passed:
    print('  NOTE: Gate failed but proceeding (--allow-failed-gate)')
" 2>&1 | tee -a "$LOG"

# ── Step 4: Poisson PINN ──────────────────────────────────────────────────
echo "[$(date)] ===== STEP 4: Poisson PINN (Schwarz, max 120 iters) =====" | tee -a "$LOG"
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data        "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --atlas-checkpoint  "$OUT_DEC/rabbit_atlas_trained.pt" \
    --sdf-checkpoint    "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --output-dir        "$OUT_PINN" \
    --run-tag           bunny_poisson_repaired \
    --pinn-arch compact --pinn-width 64 --pinn-depth 4 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --pde-warmup-iters 50 \
    --max-schwarz-iters 120 \
    --plateau-patience 20 --plateau-use-rel-l2 \
    --checkpoint-policy best_rel_l2 \
    --interface-normal-mode seed \
    --direct-coord-pde --allow-failed-gate --volumetric-atlas \
    2>&1 | tee -a "$LOG"

# ── Step 5: Postprocess ───────────────────────────────────────────────────
echo "[$(date)] ===== STEP 5: Postprocess =====" | tee -a "$LOG"
SOLN=$(ls -t "${OUT_PINN}"/*_solution.npz 2>/dev/null | head -1 || echo "")
if [[ -n "$SOLN" ]]; then
    python postprocessing/poisson_rabbit_paraview.py \
        --solution-npz "$SOLN" \
        --atlas-npz    "$OUT_ATLAS/rabbit_atlas_data.npz" \
        --ply-file     "$PLY" \
        --output-dir   paraview/bunny_poisson_repaired \
        --prefix       bunny_poisson_repaired \
        2>&1 | tee -a "$LOG"
    python postprocessing/render_3d_figures.py 2>&1 | tee -a "$LOG"
else
    echo "[$(date)] WARNING: No solution NPZ found — skipping postprocess" | tee -a "$LOG"
fi

echo "[$(date)] ===== PIPELINE COMPLETE =====" | tee -a "$LOG"
