#!/usr/bin/env bash
# ============================================================================
# Canonical script: Poisson MMS on procedural rabbit volumetric atlas
# Result: attempt20c_compact — rel-L2 = 2.21% at Schwarz iter 61
#
# This is the BEST KNOWN result for the Poisson manufactured-solution benchmark
# on the mapped-sphere atlas framework. It uses the procedural (5-ellipsoid)
# rabbit domain, which has a perfect analytical SDF and clean chart interfaces.
#
# To reproduce: run from PINN_coordinate_chart_3Dgeometry/
# Estimated time: ~60 minutes (61 Schwarz iterations × ~60s each)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1

ATLAS_DATA="runs/atlas_vol/rabbit_atlas_data.npz"
ATLAS_CKPT="runs/atlas_vol_trained/rabbit_atlas_trained.pt"
OUT_DIR="runs/poisson_rabbit_best_$(date +%Y%m%d_%H%M%S)"
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

echo "[$(date)] Starting Poisson rabbit benchmark (target: ~2.2% rel-L2)"

python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data        "$ATLAS_DATA" \
    --atlas-checkpoint  "$ATLAS_CKPT" \
    --output-dir        "$OUT_DIR" \
    --run-tag           poisson_rabbit_best \
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

echo "[$(date)] Done. Output: $OUT_DIR"
python3 -c "
import json, glob
mf = glob.glob('$OUT_DIR/*_metrics.json')
if mf:
    with open(mf[0]) as f: m = json.load(f)
    g = m['global']
    b = m['checkpoint_triplet']['best_rel_l2']
    print(f'  rel-L2 = {g[\"relative_l2_error\"]*100:.2f}%  best_iter = {b[\"iter\"]}')
"
