#!/usr/bin/env bash
# ============================================================================
# Canonical script: Train SDF v3 + build volumetric atlas v3 for Stanford Bunny
# Results:
#   SDF v3:   sign=0.311  surface=0.017  normal=0.032  (runs/bunny_sdf_v3/)
#   Atlas v3: 50k interior pts, 12 charts, 100% PLY bbox containment
#             (runs/atlas_bunny_vol_v3/)
#   Decoders: gate=True, overlap=0.018, coverage=1.000
#             (runs/atlas_bunny_vol_v3_hiW/)
#
# Note: The Poisson PINN on this atlas does NOT converge to publication quality
# (see RESULTS.md for root cause analysis). The atlas IS geometrically correct
# and can be used for visualization or alternative PDE solvers.
#
# Run from PINN_coordinate_chart_3Dgeometry/
# Total time: ~50 min (SDF 25min + atlas 1min + decoders 25min)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

PLY="runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply"

# ── Step 1: SDF v3 (balanced weights, good surface+normal quality) ───────────
echo "[$(date)] Step 1: Train SDF v3 (20k epochs)"
python src/train_sdf_rabbit.py \
    --points-file   "$PLY" \
    --output-dir    runs/bunny_sdf_v3 \
    --width 128 --depth 6 --epochs 20000 \
    --anchor-offset 0.25 \
    --w-sign   8.0 \
    --w-normal 3.0 \
    --w-surface 10.0 \
    --batch-far 4096
python3 -c "
import json
with open('runs/bunny_sdf_v3/rabbit_sdf_meta.json') as f: d=json.load(f)
print(f'  sign={d[\"final_sign\"]:.4f}  surface={d[\"final_surface\"]:.4f}  normal={d[\"final_normal\"]:.4f}')
"

# ── Step 2: Build volumetric atlas ──────────────────────────────────────────
echo "[$(date)] Step 2: Build atlas v3"
python experiments/build_rabbit_atlas_volumetric.py \
    --sdf-checkpoint     runs/bunny_sdf_v3/rabbit_sdf.pt \
    --n-interior-samples 50000 \
    --n-charts           12 \
    --frame-mode         axis_aligned \
    --output-dir         runs/atlas_bunny_vol_v3

# ── Step 3: Train atlas decoders (high overlap weight) ──────────────────────
echo "[$(date)] Step 3: Train atlas decoders (w_overlap=8, 6000 epochs)"
python experiments/train_rabbit_atlas.py \
    --atlas-data     runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \
    --output-dir     runs/atlas_bunny_vol_v3_hiW \
    --volumetric \
    --sdf-checkpoint runs/bunny_sdf_v3/rabbit_sdf.pt \
    --epochs         6000 \
    --w-overlap      8.0 \
    --batch-overlap  2048
python3 -c "
import json
with open('runs/atlas_bunny_vol_v3_hiW/rabbit_atlas_gate_report.json') as f: g=json.load(f)
print(f'  gate={g[\"passed\"]}  overlap={g[\"overlap_consistency\"]:.4f}  coverage={g[\"coverage_ratio\"]:.4f}')
"
echo "[$(date)] Done. Atlas and decoders ready."
