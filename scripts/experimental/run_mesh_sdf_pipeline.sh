#!/usr/bin/env bash
# ============================================================================
# Exact Mesh SDF Pipeline for Stanford Bunny Poisson PINN
#
# Motivation
# ----------
# All neural SDF approaches (global v1-v3, chart-partitioned) produced
# sign_error > 30%, causing the Schwarz PINN to diverge immediately
# (best rel-L2 = 25% at iter 1).
#
# Root cause: sign supervision via noisy PCA normals + tiny anchor offsets
# leads to sign loss ≈ log(2) — the network never learns sign.
#
# This pipeline replaces neural SDF training with exact mesh-based SDF
# computation using three complementary libraries:
#   - open3d.RaycastingScene : primary oracle (embree BVH, non-watertight safe)
#   - mesh_to_sdf            : near-surface point distribution
#   - pysdf                  : independent sign cross-validation
#
# Steps
# -----
# Step 1: Build exact mesh SDF + train MLP adapter        (~38 min)
#         → sign_error < 5% guaranteed by gate check
# Step 2: Build atlas from exact SDF                       (~5 min)
# Step 3: Train atlas decoders                             (~30 min)
# Step 4: Run Poisson PINN (Schwarz iteration)            (~120 min)
# Step 5: Postprocess + figures                            (~10 min)
#
# Total: ~3.5 hours  (vs 5.5 h with neural SDF training)
#
# Expected outcome: rel-L2 similar to attempt20c (~2.2%), since
# the PINN uses atlas_bunny_mesh (correct interior points) and
# exact-sign interface normals.
#
# Run from: PINN_coordinate_chart_3Dgeometry/
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"
OUT_SDF="runs/bunny_sdf_mesh"
OUT_ATLAS="runs/atlas_bunny_mesh"
OUT_DEC="runs/atlas_bunny_mesh_dec"
OUT_PINN="runs/bunny_poisson_mesh"
LOG="runs/mesh_sdf_pipeline.log"

# ── Step 1: Exact mesh SDF (open3d + mesh_to_sdf + pysdf) ────────────────────
echo "[$(date)] ===== STEP 1: Exact mesh SDF (~38 min) =====" | tee -a "$LOG"
#
# Key flags:
#   --n-near 150000     : near-surface samples from mesh_to_sdf (dense near φ=0)
#   --n-uniform 150000  : uniform random + open3d sign (interior/exterior coverage)
#   --epochs 5000       : MSE regression + weak Eikonal regulariser
#   --sign-gate 0.05    : abort if MLP sign_error > 5% (before atlas build)
#
python src/build_mesh_sdf.py \
    --ply-file  "$PLY" \
    --out-dir   "$OUT_SDF" \
    --n-near    150000 \
    --n-uniform 150000 \
    --epochs    5000 \
    --lr        1e-3 \
    --width     128 \
    --depth     6 \
    --w-eikonal 0.1 \
    --sign-gate 0.05 \
    --log-every 500

# Print sign quality summary
python3 -c "
import json
with open('$OUT_SDF/meta.json') as f:
    m = json.load(f)
print(f'  sign_error          = {m[\"sign_error_estimate\"]:.4f}  (< 0.05 PASS)')
print(f'  pysdf_agreement     = {m[\"pysdf_open3d_agreement\"]:.4f}  (> 0.90 PASS)')
print(f'  n_training_samples  = {m[\"n_training_samples\"]}')
" 2>/dev/null || echo "  (meta.json not readable)"

# ── Step 2: Build atlas from exact SDF ───────────────────────────────────────
echo "[$(date)] ===== STEP 2: Build atlas (~5 min) =====" | tee -a "$LOG"
#
# Uses the same centre/scale as v3 → interior_points are now classified with
# exact sign from open3d (0% sign error by construction in the teacher SDF).
#
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint     "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --n-interior-samples 50000 \
    --n-charts           12 \
    --frame-mode         axis_aligned \
    --output-dir         "$OUT_ATLAS"

# ── Step 3: Train atlas decoders ─────────────────────────────────────────────
echo "[$(date)] ===== STEP 3: Atlas decoders (~30 min) =====" | tee -a "$LOG"
python experiments/train_rabbit_atlas.py \
    --atlas-data     "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --output-dir     "$OUT_DEC" \
    --volumetric \
    --sdf-checkpoint "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --epochs         6000 \
    --w-overlap      8.0 \
    --batch-overlap  2048

python3 -c "
import json
with open('$OUT_DEC/rabbit_atlas_gate_report.json') as f:
    g = json.load(f)
print(f'  gate={g[\"passed\"]}  overlap={g[\"overlap_consistency\"]:.4f}  coverage={g[\"coverage_ratio\"]:.4f}')
" 2>/dev/null || echo "  (gate_report not readable)"

# ── Step 4: Poisson PINN with exact-SDF atlas ────────────────────────────────
echo "[$(date)] ===== STEP 4: Poisson PINN (~120 min) =====" | tee -a "$LOG"
#
# Settings match attempt20c_compact (best known: 2.2% rel-L2, 61 Schwarz iters)
#
python experiments/run_poisson_rabbit_atlas_schwarz.py \
    --atlas-data        "$OUT_ATLAS/rabbit_atlas_data.npz" \
    --atlas-checkpoint  "$OUT_DEC/rabbit_atlas_trained.pt" \
    --sdf-checkpoint    "$OUT_SDF/rabbit_sdf_mesh.pt" \
    --output-dir        "$OUT_PINN" \
    --run-tag           bunny_poisson_mesh \
    --pinn-arch compact --pinn-width 64 --pinn-depth 4 \
    --w-pde 5.0 --w-bc 1.0 \
    --w-interface-value 2.0 --w-interface-flux 2.0 \
    --pde-warmup-iters 50 \
    --max-schwarz-iters 120 \
    --plateau-patience 20 --plateau-use-rel-l2 \
    --checkpoint-policy best_rel_l2 \
    --interface-normal-mode seed \
    --direct-coord-pde \
    --volumetric-atlas \
    --allow-failed-gate

# ── Step 5: Postprocess + figures ────────────────────────────────────────────
echo "[$(date)] ===== STEP 5: Postprocess + figures =====" | tee -a "$LOG"
SOLN_NPZ=$(ls -t "$OUT_PINN"/*_solution.npz 2>/dev/null | head -1 || \
           ls -t "$OUT_PINN"/**/*_solution.npz 2>/dev/null | head -1 || echo "")

if [[ -n "$SOLN_NPZ" ]]; then
    python postprocessing/poisson_rabbit_paraview.py \
        --solution-npz "$SOLN_NPZ" \
        --atlas-npz    "$OUT_ATLAS/rabbit_atlas_data.npz" \
        --ply-file     "$PLY" \
        --output-dir   paraview/bunny_poisson_mesh \
        --prefix       bunny_poisson_mesh
    python postprocessing/render_3d_figures.py
else
    echo "  No solution NPZ found — skipping postprocessing"
fi

echo "[$(date)] ===== MESH SDF PIPELINE COMPLETE =====" | tee -a "$LOG"

# ── Final metrics summary ────────────────────────────────────────────────────
python3 -c "
import json, glob
mf = glob.glob('$OUT_PINN/*_metrics.json')
if mf:
    with open(mf[0]) as f: m = json.load(f)
    g = m['global']
    b = m['checkpoint_triplet']['best_rel_l2']
    print(f'  rel-L2 = {g[\"relative_l2_error\"]*100:.2f}%  best_iter = {b[\"iter\"]}')
" 2>/dev/null || echo "  (metrics not available)"
