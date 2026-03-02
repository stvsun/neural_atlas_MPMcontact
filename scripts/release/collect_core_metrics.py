#!/usr/bin/env python3
"""Collect normalized metrics from the canonical core benchmark registry."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


ELLIPSOID_REL_RE = re.compile(r"Relative L2:\s*([0-9eE+\-.]+)")
ELLIPSOID_L2_RE = re.compile(r"L2 error:\s*([0-9eE+\-.]+)")
ELLIPSOID_MAX_RE = re.compile(r"Max error:\s*([0-9eE+\-.]+)")
ELLIPSOID_LOSS_RE = re.compile(r"Final total loss:\s*([0-9eE+\-.]+)")
ELLIPSOID_TIME_RE = re.compile(r"Epoch\s+\d+/\d+\s+\|.*?Time:\s*([0-9eE+\-.]+)s")


STANDARD_NUMERIC_KEYS = [
    "relative_l2_error",
    "l2_error",
    "max_error",
    "obs_rel_l2",
    "traction_rel_l2",
    "disp_rel_l2",
    "mu_rel_error_percent",
    "K_rel_error_percent",
    "k0_rel_error",
    "eig_rel_error_mean",
    "eig_rel_error_max",
    "axis_error_deg_mean",
    "field_rel_global",
    "interface_flux",
    "runtime_seconds",
]


@dataclass
class ParseResult:
    metrics: Dict[str, Any]
    warnings: List[str]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ellipsoid_log(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    warnings: List[str] = []

    rel = ELLIPSOID_REL_RE.search(text)
    l2 = ELLIPSOID_L2_RE.search(text)
    mx = ELLIPSOID_MAX_RE.search(text)
    loss = ELLIPSOID_LOSS_RE.search(text)
    times = [float(m.group(1)) for m in ELLIPSOID_TIME_RE.finditer(text)]

    if rel is None:
        warnings.append("Relative L2 not found in ellipsoid log.")

    metrics = {
        "relative_l2_error": _to_float(rel.group(1)) if rel else None,
        "l2_error": _to_float(l2.group(1)) if l2 else None,
        "max_error": _to_float(mx.group(1)) if mx else None,
        "final_total_loss": _to_float(loss.group(1)) if loss else None,
        "runtime_seconds": max(times) if times else None,
        "target_met": None,
    }
    return ParseResult(metrics=metrics, warnings=warnings)


def parse_poisson_star_json(path: Path) -> ParseResult:
    d = _load_json(path)
    metrics = {
        "relative_l2_error": _to_float(d.get("relative_l2_error")),
        "l2_error": _to_float(d.get("l2_error")),
        "max_error": _to_float(d.get("max_error")),
        "final_total_loss": _to_float(d.get("final_total_loss")),
        "runtime_seconds": _to_float(d.get("train_time_sec")),
        "target_met": d.get("target_met"),
    }
    return ParseResult(metrics=metrics, warnings=[])


def parse_rabbit_poisson_json(path: Path) -> ParseResult:
    d = _load_json(path)
    g = d.get("global", {}) if isinstance(d.get("global"), dict) else {}
    triplet = d.get("checkpoint_triplet", {}) if isinstance(d.get("checkpoint_triplet"), dict) else {}
    best_rel = triplet.get("best_rel_l2", {}) if isinstance(triplet.get("best_rel_l2"), dict) else {}

    rel = _to_float(g.get("relative_l2_error"))
    if rel is None:
        rel = _to_float(best_rel.get("rel_l2_eval"))

    metrics = {
        "relative_l2_error": rel,
        "l2_error": _to_float(g.get("l2_error")),
        "max_error": _to_float(g.get("max_error")),
        "interface_flux": _to_float(d.get("final_interface_flux")),
        "runtime_seconds": _to_float(d.get("total_runtime_sec") or d.get("runtime_seconds")),
        "target_met": d.get("target_met"),
    }
    return ParseResult(metrics=metrics, warnings=[])


def parse_torus_original_json(path: Path) -> ParseResult:
    d = _load_json(path)
    metrics = {
        "mu_rel_error_percent": _to_float(d.get("mu_rel_error_percent")),
        "K_rel_error_percent": _to_float(d.get("K_rel_error_percent")),
        "traction_rel_l2": _to_float(d.get("traction_rel_l2")),
        "runtime_seconds": _to_float(d.get("runtime_seconds")),
        "target_met": d.get("target_met"),
    }
    return ParseResult(metrics=metrics, warnings=[])


def parse_torus_schwarz_json(path: Path) -> ParseResult:
    d = _load_json(path)
    metrics = {
        "obs_rel_l2": _to_float(d.get("obs_rel_l2_final")),
        "traction_rel_l2": _to_float(d.get("traction_rel_l2_final")),
        "disp_rel_l2": _to_float(d.get("disp_rel_l2_final")),
        "mu_rel_error_percent": _to_float(d.get("mu_rel_error_percent")),
        "K_rel_error_percent": _to_float(d.get("K_rel_error_percent")),
        "runtime_seconds": _to_float(d.get("runtime_seconds")),
        "target_met": d.get("target_met"),
    }
    return ParseResult(metrics=metrics, warnings=[])


def parse_rabbit_elder_json(path: Path) -> ParseResult:
    d = _load_json(path)
    metrics = {
        "k0_rel_error": _to_float(d.get("k0_rel_error")),
        "eig_rel_error_mean": _to_float(d.get("eig_rel_error_mean")),
        "eig_rel_error_max": _to_float(d.get("eig_rel_error_max")),
        "axis_error_deg_mean": _to_float(d.get("axis_angle_error_deg_mean")),
        "field_rel_global": _to_float(d.get("field_rel_global")),
        "interface_flux": _to_float(d.get("final_interface_flux")),
        "runtime_seconds": _to_float(d.get("runtime_seconds")),
        "target_met": d.get("target_met"),
    }
    return ParseResult(metrics=metrics, warnings=[])


PARSERS = {
    "ellipsoid_log": parse_ellipsoid_log,
    "poisson_star_json": parse_poisson_star_json,
    "rabbit_poisson_json": parse_rabbit_poisson_json,
    "torus_original_json": parse_torus_original_json,
    "torus_schwarz_json": parse_torus_schwarz_json,
    "rabbit_elder_json": parse_rabbit_elder_json,
}


def pick_primary_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if key == "relative_l2_error":
        return _to_float(metrics.get("relative_l2_error"))
    return _to_float(metrics.get(key))


def summarize_numeric(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in STANDARD_NUMERIC_KEYS:
        vals = [_to_float(r.get("metrics", {}).get(key)) for r in records]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        out[key] = {
            "n": float(len(vals)),
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def collect_case(case: Dict[str, Any]) -> Dict[str, Any]:
    parsed_records: List[Dict[str, Any]] = []
    warnings: List[str] = []

    primary_key = case.get("primary_metric", {}).get("key")
    lower_is_better = bool(case.get("primary_metric", {}).get("lower_is_better", True))

    for rec in case.get("records", []):
        mpath = Path(rec["metrics_path"])
        r = {
            "record_id": rec.get("record_id"),
            "seed": rec.get("seed"),
            "run_dir": rec.get("run_dir"),
            "metrics_path": rec.get("metrics_path"),
            "metrics_type": rec.get("metrics_type"),
            "checkpoint_path": rec.get("checkpoint_path"),
            "figures": rec.get("figures", []),
            "status": "ok",
            "warnings": [],
            "metrics": {},
            "primary_metric_value": None,
        }

        if not mpath.is_file():
            r["status"] = "missing_metrics"
            r["warnings"].append(f"Missing metrics file: {mpath}")
            warnings.extend(r["warnings"])
            parsed_records.append(r)
            continue

        parser_name = rec.get("metrics_type")
        parser = PARSERS.get(parser_name)
        if parser is None:
            r["status"] = "unknown_metrics_type"
            r["warnings"].append(f"Unknown metrics_type: {parser_name}")
            warnings.extend(r["warnings"])
            parsed_records.append(r)
            continue

        try:
            parsed = parser(mpath)
            r["metrics"] = parsed.metrics
            r["warnings"].extend(parsed.warnings)
        except Exception as exc:  # pragma: no cover
            r["status"] = "parse_error"
            r["warnings"].append(f"Parse error: {exc}")

        r["primary_metric_value"] = pick_primary_value(r["metrics"], primary_key)
        if r["primary_metric_value"] is None and r["status"] == "ok":
            r["warnings"].append(
                f"Primary metric '{primary_key}' missing in parsed metrics for record {r['record_id']}"
            )

        warnings.extend(r["warnings"])
        parsed_records.append(r)

    ok_records = [r for r in parsed_records if r["status"] == "ok" and _to_float(r["primary_metric_value"]) is not None]
    best_record = None
    if ok_records:
        best_record = sorted(
            ok_records,
            key=lambda rr: float(rr["primary_metric_value"]),
            reverse=not lower_is_better,
        )[0]

    return {
        "case_id": case.get("case_id"),
        "family": case.get("family"),
        "include_in_main": bool(case.get("include_in_main", False)),
        "description": case.get("description"),
        "script_path": case.get("script_path"),
        "primary_metric": case.get("primary_metric"),
        "records": parsed_records,
        "record_count": len(parsed_records),
        "ok_record_count": len(ok_records),
        "best_record": best_record,
        "summary": summarize_numeric(ok_records),
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and normalize metrics for core CMAME benchmarks.")
    parser.add_argument(
        "--registry",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_benchmarks_registry.json",
        help="Path to benchmark registry JSON.",
    )
    parser.add_argument(
        "--output",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_metrics_collected.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    output_path = Path(args.output)
    registry = _load_json(registry_path)

    cases = registry.get("main_cases", [])
    collected_cases = [collect_case(case) for case in cases]

    with_three_or_more = [
        {
            "case_id": c["case_id"],
            "n": c["ok_record_count"],
            "primary_metric": c["primary_metric"],
            "summary": c["summary"],
        }
        for c in collected_cases
        if c["ok_record_count"] >= 3
    ]

    out = {
        "registry_path": str(registry_path),
        "scope": registry.get("scope"),
        "project_root": registry.get("project_root"),
        "cases": collected_cases,
        "three_run_or_more_cases": with_three_or_more,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote collected metrics: {output_path}")
    print(f"Cases: {len(collected_cases)} | with >=3 parsed records: {len(with_three_or_more)}")


if __name__ == "__main__":
    main()
