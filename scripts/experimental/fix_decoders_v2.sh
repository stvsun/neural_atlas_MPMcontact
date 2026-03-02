#!/usr/bin/env bash
# Fix v2: retrain atlas decoders with HIGH overlap weight to prevent overlap from
# being sacrificed to mask/coverage losses.
# Default w_overlap=0.8 means overlap is only 1% of loss → gets ignored.
# This run uses w_overlap=8.0 (10×) + larger overlap batch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[$(date)] Working directory: $(pwd)"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

echo ""
echo "[$(date)] ===== Retrain atlas decoders (w_overlap=8.0, batch_overlap=2048, 6000 ep) ====="
python experiments/train_rabbit_atlas.py \
    --atlas-data     runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --output-dir     runs/atlas_bunny_vol_v3_hiW \
    --volumetric \
    --sdf-checkpoint runs/bunny_sdf_v3/rabbit_sdf.pt \
    --epochs         6000 \
    --w-overlap      8.0 \
    --batch-overlap  2048

echo "[$(date)] Decoder training done."
python3 -c "
import json
with open('runs/atlas_bunny_vol_v3_hiW/rabbit_atlas_gate_report.json') as f:
    g = json.load(f)
print(f'  gate passed:         {g[\"passed\"]}')
print(f'  overlap_consistency: {g[\"overlap_consistency\"]:.4f}')
print(f'  boundary_rmse:       {g[\"boundary_rmse\"]:.4f}')
print(f'  coverage_ratio:      {g[\"coverage_ratio\"]:.4f}')
print(f'  foldover_ratio:      {g[\"foldover_ratio\"]:.4f}')
" 2>/dev/null

echo ""
echo "[$(date)] ===== Retrain Poisson PINN (volumetric-atlas) ====="
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data       runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --atlas-checkpoint runs/atlas_bunny_vol_v3_hiW/rabbit_atlas_trained.pt \
    --sdf-checkpoint   runs/bunny_sdf_v3/rabbit_sdf.pt \
    --output-dir       runs/bunny_poisson_v3_hiW \
    --run-tag          bunny_poisson_v3_hiW \
    --pinn-arch compact --pinn-width 64 --pinn-depth 4 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --pde-warmup-iters 50 \
    --max-schwarz-iters 120 \
    --plateau-patience 20 --plateau-use-rel-l2 \
    --checkpoint-policy best_rel_l2 \
    --interface-normal-mode seed \
    --volumetric-atlas --direct-coord-pde \
    --allow-failed-gate
echo "[$(date)] PINN done."

echo ""
echo "[$(date)] ===== Postprocess + figures ====="
SOLN_NPZ="runs/bunny_poisson_v3_hiW/rabbit_poisson_schwarz_bunny_poisson_v3_hiW_solution.npz"
if [[ ! -f "$SOLN_NPZ" ]]; then
    SOLN_NPZ=$(find runs/bunny_poisson_v3_hiW -name "*_solution.npz" | sort | tail -1)
fi
python postprocessing/poisson_rabbit_paraview.py \
    --solution-npz "$SOLN_NPZ" \
    --atlas-npz    runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --ply-file     "$PLY" \
    --output-dir   paraview/rabbit_poisson_bunny_v3_hiW \
    --prefix       rabbit_poisson_bunny_v3_hiW \
    --grid --surface --figures-dir figures/rabbit_poisson_v3_hiW

python postprocessing/render_3d_figures.py

echo "[$(date)] ===== PIPELINE COMPLETE ====="
