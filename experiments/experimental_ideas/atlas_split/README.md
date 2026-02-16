# Atlas Split Archive

This folder archives the chart-splitting experiment so the main rabbit workflow can focus on fixed-atlas refinement.

## Archived Files
- `build_rabbit_atlas_adaptive_split.py`
- `train_rabbit_atlas_warmstart.py`
- `run_poisson_rabbit_atlas_schwarz_adaptive_split.py`
- `rabbit_poisson_adaptive_split.yaml`

## Status
- Archived as experimental (not active path).
- Fixed-atlas refinement is now the active strategy.

## Known Failure Modes Observed
- Refined stages could silently run with zero trainable charts due train/freeze JSON parsing mismatch.
- MPS backend instability in some linear algebra and indexed sampling paths can produce brittle runs.
- Split refinement introduced strong interface constraints that frequently regressed global field quality.

## Last Known Adaptive Run Roots
- `runs/atlas_schwarz_adaptive_main`
- `runs/atlas_schwarz_adaptive_main_mpsfix3`
- `runs/atlas_schwarz_adaptive_main_mpsfix4`
- `runs/atlas_schwarz_adaptive_main_mpsfix5`
- `runs/atlas_schwarz_adaptive_main_mpsfix6`
- `runs/atlas_schwarz_adaptive_main_mpsfix7`
- `runs/atlas_schwarz_adaptive_fastmps_20260214_210804`
- `runs/atlas_schwarz_adaptive_fastmps_20260214_214012`
- `runs/atlas_schwarz_adaptive_smoke_20260214_124446`
- `runs/atlas_schwarz_adaptive_smoke_20260214_124723`
- `runs/atlas_schwarz_adaptive_quickcheck_20260214_130130`
