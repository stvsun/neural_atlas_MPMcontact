#!/usr/bin/env bash
# Resume pipeline from Step 4 (SDF v3, Atlas v3, Decoders are already done)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[$(date)] Working directory: $(pwd)"
echo "[$(date)] Resuming from Step 4 (Steps 1-3 already complete)"

export PYTORCH_ENABLE_MPS_FALLBACK=1
PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ===== STEP 4: Retrain Poisson PINN (~120 min) ====="
# Optimal settings from attempt20c_compact (best known: 2.2% rel-L2)
# Note: no --config flag; all weights passed directly
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

SOLN_NPZ="runs/bunny_poisson_v3/rabbit_poisson_schwarz_bunny_poisson_v3_solution.npz"
if [[ ! -f "$SOLN_NPZ" ]]; then
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
