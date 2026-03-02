#!/usr/bin/env bash
# DIAGNOSTIC: Run PINN with manufactured-solution supervision to test whether
# the Schwarz iteration CAN converge on Stanford Bunny (given good initialization).
# If this converges to <10%, the issue is initialization/SDF quality, NOT the method.
# Uses atlas_bunny_vol_v3_hiW decoders (gate=True, overlap=0.018).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[$(date)] Diagnostic: supervised PINN on Stanford Bunny"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data       runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --atlas-checkpoint runs/atlas_bunny_vol_v3_hiW/rabbit_atlas_trained.pt \
    --sdf-checkpoint   runs/bunny_sdf_v3/rabbit_sdf.pt \
    --output-dir       runs/bunny_poisson_diagnostic \
    --run-tag          bunny_diag_supervised \
    --pinn-arch compact --pinn-width 64 --pinn-depth 4 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --w-manufactured-supervision 3.0 \
    --manufactured-supervision-batch 256 \
    --pde-warmup-iters 50 \
    --max-schwarz-iters 120 \
    --plateau-patience 20 --plateau-use-rel-l2 \
    --checkpoint-policy best_rel_l2 \
    --interface-normal-mode seed \
    --volumetric-atlas --direct-coord-pde \
    --allow-failed-gate

echo "[$(date)] Diagnostic done."
python3 -c "
import json
with open('runs/bunny_poisson_diagnostic/rabbit_poisson_schwarz_bunny_diag_supervised_metrics.json') as f:
    m = json.load(f)
print(f'  rel_l2:     {m[\"global\"][\"relative_l2_error\"]*100:.2f}%')
print(f'  Schwarz iters: {m[\"checkpoint_triplet\"][\"best_rel_l2\"][\"iter\"]}')
" 2>/dev/null
