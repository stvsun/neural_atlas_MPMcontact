#!/usr/bin/env bash
# ============================================================================
# Enhanced rabbit Poisson PINN: FEM-initialized with decaying guidance loss
#
# Strategy:
# 1. Load FEM n=48 solution as target data
# 2. Pretrain each chart network to match FEM values (interior pretraining)
# 3. Run Schwarz solver with FEM guidance that decays over iterations
# 4. Use larger PDE batch for high-error charts (4, 6, 0)
# 5. Increase local steps per sweep
# 6. No manufactured-solution supervision — FEM guidance replaces it
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1

ATLAS_DATA="runs/atlas_vol/rabbit_atlas_data.npz"
ATLAS_CKPT="runs/atlas_vol_trained/rabbit_atlas_trained.pt"
FEM_SOL="runs/fem_highres/rabbit_poisson_schwarz_fem_highres_n48_solution.npz"
OUT_DIR="runs/poisson_rabbit_fem_init"

echo "[$(date)] Starting FEM-initialized rabbit Poisson PINN"
echo "  Atlas: $ATLAS_DATA"
echo "  FEM solution: $FEM_SOL"
echo "  Output: $OUT_DIR"

python experiments/rabbit_atlas_variants/run_poisson_rabbit_fem_init.py \
    --atlas-data        "$ATLAS_DATA" \
    --atlas-checkpoint  "$ATLAS_CKPT" \
    --fem-solution      "$FEM_SOL" \
    --output-dir        "$OUT_DIR" \
    --run-tag           fem_init \
    --max-schwarz-iters 120 \
    --local-steps       30 \
    --pde-batch         384 \
    --bc-batch          256 \
    --if-batch          192 \
    --bc-pretrain-epochs 300 \
    --fem-pretrain-epochs 500 \
    --fem-guidance-start 1.0 \
    --fem-guidance-end   0.0 \
    --fem-guidance-decay-iters 60 \
    --high-error-charts "4,6,0" \
    --high-error-pde-batch 768 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --pde-warmup-iters 50

echo "[$(date)] Done. Output: $OUT_DIR"
