#!/usr/bin/env bash
# Overnight publication pipeline: SDF v3 → Atlas v3 → Decoders → PINN → Figures
# Launch: nohup bash runs/overnight_pipeline.sh > runs/overnight_pipeline.log 2>&1 &
set -euo pipefail

# Ensure we always run from PINN_coordinate_chart_3Dgeometry/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[$(date)] Working directory: $(pwd)"

export PYTORCH_ENABLE_MPS_FALLBACK=1
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 1: Train SDF v3 (~60 min) ====="
# V3 improvements vs V2:
#   anchor-offset 0.25 (deeper: 39mm, was 23mm) for better deep-interior sign
#   w-sign 8.0 (balanced, was 20 which crowded out surface loss)
#   w-normal 3.0 (3x default for cleaner gradient field)
#   batch-far 4096 (same as v2, 4x more far-field coverage than v1)
#   epochs 20000 (2x v2 for better convergence)
python src/train_sdf_rabbit.py \
    --points-file  "$PLY" \
    --output-dir   runs/bunny_sdf_v3 \
    --width 128 --depth 6 \
    --epochs 20000 \
    --anchor-offset 0.25 \
    --w-sign   8.0 \
    --w-normal 3.0 \
    --w-surface 10.0 \
    --batch-far 4096
echo "[$(date)] SDF v3 done. Losses:"
python3 -c "
import json
with open('runs/bunny_sdf_v3/rabbit_sdf_meta.json') as f:
    d = json.load(f)
print(f'  sign={d[\"final_sign\"]:.4f}  surface={d[\"final_surface\"]:.4f}  normal={d[\"final_normal\"]:.4f}')
" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 2: Build atlas v3 (~5 min) ====="
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint    runs/bunny_sdf_v3/rabbit_sdf.pt \
    --n-interior-samples 50000 \
    --n-charts          12 \
    --frame-mode        axis_aligned \
    --output-dir        runs/atlas_bunny_vol_v3
echo "[$(date)] Atlas v3 done."

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 3: Train atlas decoders v3 (~30 min) ====="
# REQUIRED before PINN: trains ChartDecoder + MaskNet networks, creates gate dict
python experiments/train_rabbit_atlas.py \
    --atlas-data     runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --output-dir     runs/atlas_bunny_vol_v3 \
    --volumetric \
    --sdf-checkpoint runs/bunny_sdf_v3/rabbit_sdf.pt \
    --epochs         2000
echo "[$(date)] Atlas decoders v3 done: runs/atlas_bunny_vol_v3/rabbit_atlas_trained.pt"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 4: Retrain Poisson PINN (~120 min) ====="
# Optimal settings from attempt20c_compact (best known: 2.2% rel-L2)
# --allow-failed-gate bypasses atlas gate check for unattended run
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data       runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --atlas-checkpoint runs/atlas_bunny_vol_v3/rabbit_atlas_trained.pt \
    --sdf-checkpoint   runs/bunny_sdf_v3/rabbit_sdf.pt \
    --output-dir       runs/bunny_poisson_v3 \
    --run-tag          bunny_poisson_v3 \
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

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 5: Postprocess + publication figures (~30 min) ====="

# Solution NPZ: run_stem = "rabbit_poisson_schwarz_bunny_poisson_v3"
SOLN_NPZ="runs/bunny_poisson_v3/rabbit_poisson_schwarz_bunny_poisson_v3_solution.npz"
if [[ ! -f "$SOLN_NPZ" ]]; then
    # Fallback: find any solution NPZ in the output directory
    SOLN_NPZ=$(find runs/bunny_poisson_v3 -name "*_solution.npz" | sort | tail -1)
fi
echo "[$(date)] Using solution: $SOLN_NPZ"

python postprocessing/poisson_rabbit_paraview.py \
    --solution-npz "$SOLN_NPZ" \
    --atlas-npz    runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --ply-file     "$PLY" \
    --output-dir   paraview/rabbit_poisson_bunny_v3 \
    --prefix       rabbit_poisson_bunny_v3 \
    --grid --surface --figures-dir figures/rabbit_poisson_v3

python postprocessing/render_3d_figures.py

echo ""
echo "[$(date)] ===== PIPELINE COMPLETE ====="
echo "  VTUs:    paraview/rabbit_poisson_bunny_v3/"
echo "  Figures: figures/rabbit_poisson_v3/  +  render_3d_figures output"
echo "  PINN log: runs/bunny_poisson_v3/"
