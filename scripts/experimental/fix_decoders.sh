#!/usr/bin/env bash
# Fix: retrain atlas decoders WITHOUT --volumetric flag (match procedural atlas approach)
# The volumetric training causes overlap loss to INCREASE (bad), std training decreases it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[$(date)] Working directory: $(pwd)"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

echo ""
echo "[$(date)] ===== Retrain atlas decoders (standard, no --volumetric) ====="
# Use same v3 atlas data but train WITHOUT --volumetric
# This matches the approach of atlas_vol_trained (procedural) that achieved gate=True
python experiments/train_rabbit_atlas.py \
    --atlas-data     runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --output-dir     runs/atlas_bunny_vol_v3_std \
    --epochs         10000

echo "[$(date)] Decoder training done."
python3 -c "
import json
with open('runs/atlas_bunny_vol_v3_std/rabbit_atlas_gate_report.json') as f:
    g = json.load(f)
print(f'  gate passed:         {g[\"passed\"]}')
print(f'  overlap_consistency: {g[\"overlap_consistency\"]:.4f}')
print(f'  boundary_rmse:       {g[\"boundary_rmse\"]:.4f}')
print(f'  coverage_ratio:      {g[\"coverage_ratio\"]:.4f}')
" 2>/dev/null

echo ""
echo "[$(date)] ===== Retrain Poisson PINN (no --volumetric-atlas) ====="
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data       runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --atlas-checkpoint runs/atlas_bunny_vol_v3_std/rabbit_atlas_trained.pt \
    --sdf-checkpoint   runs/bunny_sdf_v3/rabbit_sdf.pt \
    --output-dir       runs/bunny_poisson_v3_std \
    --run-tag          bunny_poisson_v3_std \
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
echo "[$(date)] PINN done."

echo ""
echo "[$(date)] ===== Postprocess + figures ====="
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"
SOLN_NPZ="runs/bunny_poisson_v3_std/rabbit_poisson_schwarz_bunny_poisson_v3_std_solution.npz"
if [[ ! -f "$SOLN_NPZ" ]]; then
    SOLN_NPZ=$(find runs/bunny_poisson_v3_std -name "*_solution.npz" | sort | tail -1)
fi

python postprocessing/poisson_rabbit_paraview.py \
    --solution-npz "$SOLN_NPZ" \
    --atlas-npz    runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --ply-file     "$PLY" \
    --output-dir   paraview/rabbit_poisson_bunny_v3_std \
    --prefix       rabbit_poisson_bunny_v3_std \
    --grid --surface --figures-dir figures/rabbit_poisson_v3_std

python postprocessing/render_3d_figures.py

echo ""
echo "[$(date)] ===== FIX PIPELINE COMPLETE ====="
