#!/usr/bin/env python3
"""
Utility: summarize interface diagnostics from rabbit TI inverse run outputs.

Usage:
  python3 export_rabbit_ti_inverse_interface_diagnostics.py \
    --metrics <..._metrics.json> \
    --history <..._history.json> \
    --out <..._interface_diag.json>
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_last(arr):
    if not isinstance(arr, list) or len(arr) == 0:
        return None
    return arr[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Export compact interface diagnostics JSON.")
    p.add_argument("--metrics", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics = load_json(args.metrics)
    history = load_json(args.history)

    stage_a = history.get("stage_A", {})
    stage_b = history.get("stage_B", {})
    stage_c = history.get("stage_C", {}) if isinstance(history.get("stage_C"), dict) else {}

    out = {
        "source": {
            "metrics": args.metrics,
            "history": args.history,
        },
        "final_interface": metrics.get("interface", {}),
        "stage_A": {
            "if_jump_last": safe_last(stage_a.get("if_jump", [])),
            "if_val_last": safe_last(stage_a.get("if_val", [])),
            "if_flux_last": safe_last(stage_a.get("if_flux", [])),
            "rejected_iters": int(sum(stage_a.get("iter_rejected", []))) if isinstance(stage_a.get("iter_rejected", []), list) else None,
        },
        "stage_B": {
            "if_jump_last": safe_last(stage_b.get("if_jump", [])),
            "if_val_last": safe_last(stage_b.get("if_val", [])),
            "if_flux_last": safe_last(stage_b.get("if_flux", [])),
            "rejected_iters": int(sum(stage_b.get("iter_rejected", []))) if isinstance(stage_b.get("iter_rejected", []), list) else None,
        },
        "stage_C": {
            "if_jump_last": safe_last(stage_c.get("if_jump", [])),
            "if_val_last": safe_last(stage_c.get("if_val", [])),
            "if_flux_last": safe_last(stage_c.get("if_flux", [])),
            "rejected_iters": int(sum(stage_c.get("iter_rejected", []))) if isinstance(stage_c.get("iter_rejected", []), list) else None,
        },
        "acceptance": metrics.get("acceptance", {}),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved interface diagnostics: {args.out}")


if __name__ == "__main__":
    main()
