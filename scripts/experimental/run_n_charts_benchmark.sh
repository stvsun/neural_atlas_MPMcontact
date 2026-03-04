#!/usr/bin/env bash
# run_n_charts_benchmark.sh
#
# Benchmark PINN accuracy vs number of coordinate charts: 8, 12, 16, 32.
#
# Scaling rule: keep min chart size ≥ 2500 interior points.
#   n=8  → 50k samples, overlap=0.35
#   n=12 → 75k samples, overlap=0.35
#   n=16 → 100k samples, overlap=0.30
#   n=32 → 200k samples, overlap=0.25
#
# Prerequisites:
#   - runs/bunny_sdf_repaired/rabbit_sdf_mesh.pt  (from run_improved_pipeline.sh)
#
# After all configs finish, run:
#   python postprocessing/benchmark_n_charts.py \
#       --ply-file  runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply \
#       --configs 8 12 16 32 \
#       --output-dir runs/benchmark_summary
#
# Usage:
#   cd PINN_coordinate_chart_3Dgeometry
#   nohup bash scripts/experimental/run_n_charts_benchmark.sh \
#       > runs/n_charts_benchmark.log 2>&1 &
#   echo "PID: $!"

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

SDF="runs/bunny_sdf_repaired/rabbit_sdf_mesh.pt"
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"
LOG="runs/n_charts_benchmark.log"

# Verify SDF checkpoint exists
if [[ ! -f "$SDF" ]]; then
    echo "ERROR: SDF checkpoint not found: $SDF"
    echo "Run run_improved_pipeline.sh first to build the repaired SDF."
    exit 1
fi

echo "[$(date)] ===== n-CHARTS BENCHMARK START =====" | tee -a "$LOG"
echo "[$(date)] SDF: $SDF" | tee -a "$LOG"

declare -A SAMPLES=([8]=50000 [12]=75000 [16]=100000 [32]=200000)
declare -A OVERLAP=([8]=0.35  [12]=0.35  [16]=0.30   [32]=0.25)

for N in 8 12 16 32; do
    echo "" | tee -a "$LOG"
    echo "[$(date)] ========================================" | tee -a "$LOG"
    echo "[$(date)] ===== n_charts = ${N} =====" | tee -a "$LOG"
    echo "[$(date)] ========================================" | tee -a "$LOG"

    ATLAS_DIR="runs/benchmark_atlas_${N}chart"
    DEC_DIR="runs/benchmark_dec_${N}chart"
    PINN_DIR="runs/benchmark_pinn_${N}chart"

    # ── Step 1: Build atlas ───────────────────────────────────────────────
    echo "[$(date)] Step 1/3: Build atlas (${SAMPLES[$N]} pts, ${N} charts, overlap=${OVERLAP[$N]})" | tee -a "$LOG"
    python experiments/build_rabbit_atlas_volumetric.py \
        --sdf-checkpoint     "$SDF" \
        --n-interior-samples "${SAMPLES[$N]}" \
        --n-charts           "$N" \
        --overlap-target     "${OVERLAP[$N]}" \
        --frame-mode         axis_aligned \
        --output-dir         "$ATLAS_DIR" \
        2>&1 | tee -a "$LOG"

    # ── Step 2: Train decoders ────────────────────────────────────────────
    echo "[$(date)] Step 2/3: Train decoders (w_overlap=20, 9000 epochs)" | tee -a "$LOG"
    python experiments/train_rabbit_atlas.py \
        --atlas-data     "$ATLAS_DIR/rabbit_atlas_data.npz" \
        --output-dir     "$DEC_DIR" \
        --volumetric \
        --sdf-checkpoint "$SDF" \
        --epochs         9000 \
        --w-overlap      20.0 \
        --batch-overlap  4096 \
        2>&1 | tee -a "$LOG"

    # ── Step 3: Run PINN ──────────────────────────────────────────────────
    echo "[$(date)] Step 3/3: Run Poisson PINN (max 120 Schwarz iters)" | tee -a "$LOG"
    python experiments/run_poisson_rabbit_atlas_schwarz.py \
        --atlas-data        "$ATLAS_DIR/rabbit_atlas_data.npz" \
        --atlas-checkpoint  "$DEC_DIR/rabbit_atlas_trained.pt" \
        --sdf-checkpoint    "$SDF" \
        --output-dir        "$PINN_DIR" \
        --run-tag           "benchmark_${N}chart" \
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

    echo "[$(date)] DONE n_charts=${N}" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[$(date)] ===== ALL CONFIGS DONE — generating benchmark report =====" | tee -a "$LOG"

python postprocessing/benchmark_n_charts.py \
    --ply-file   "$PLY" \
    --configs    8 12 16 32 \
    --output-dir runs/benchmark_summary \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ===== n-CHARTS BENCHMARK COMPLETE =====" | tee -a "$LOG"
echo "  Summary figures: runs/benchmark_summary/" | tee -a "$LOG"
