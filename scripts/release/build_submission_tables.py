#!/usr/bin/env python3
"""Build CMAME submission tables from collected core benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


BEST_COLUMNS = [
    "case_id",
    "record_id",
    "seed",
    "primary_metric_key",
    "primary_metric_value",
    "relative_l2_error",
    "obs_rel_l2",
    "traction_rel_l2",
    "mu_rel_error_percent",
    "K_rel_error_percent",
    "k0_rel_error",
    "eig_rel_error_mean",
    "axis_error_deg_mean",
    "field_rel_global",
    "interface_flux",
    "runtime_seconds",
    "run_dir",
    "metrics_path",
    "checkpoint_path",
]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x: Any, digits: int = 4) -> str:
    v = _to_float(x)
    if v is None:
        return "-"
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1e3 or av < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def _fmt_mean_std(summary: Dict[str, Any], key: str, digits: int = 4) -> str:
    info = summary.get(key)
    if not isinstance(info, dict):
        return "-"
    m = _to_float(info.get("mean"))
    s = _to_float(info.get("std"))
    if m is None or s is None:
        return "-"
    return f"{_fmt(m, digits)} ± {_fmt(s, digits)}"


def build_best_rows(collected: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in collected.get("cases", []):
        best = case.get("best_record")
        if not isinstance(best, dict):
            continue
        metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
        row = {
            "case_id": case.get("case_id"),
            "record_id": best.get("record_id"),
            "seed": best.get("seed"),
            "primary_metric_key": (case.get("primary_metric") or {}).get("key"),
            "primary_metric_value": best.get("primary_metric_value"),
            "run_dir": best.get("run_dir"),
            "metrics_path": best.get("metrics_path"),
            "checkpoint_path": best.get("checkpoint_path"),
        }
        for k in [
            "relative_l2_error",
            "obs_rel_l2",
            "traction_rel_l2",
            "mu_rel_error_percent",
            "K_rel_error_percent",
            "k0_rel_error",
            "eig_rel_error_mean",
            "axis_error_deg_mean",
            "field_rel_global",
            "interface_flux",
            "runtime_seconds",
        ]:
            row[k] = metrics.get(k)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def write_markdown(path: Path, collected: Dict[str, Any], best_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Core Benchmark Tables")
    lines.append("")
    lines.append("## Best-Seed Results")
    lines.append("")
    lines.append(
        "| Case | Primary metric | Field rel-L2 | Obs rel-L2 | Traction rel-L2 | mu err (%) | K err (%) | k0 err | eig err mean | axis err (deg) | Interface flux | Runtime (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in best_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("case_id", "-")),
                    _fmt(r.get("primary_metric_value")),
                    _fmt(r.get("relative_l2_error")),
                    _fmt(r.get("obs_rel_l2")),
                    _fmt(r.get("traction_rel_l2")),
                    _fmt(r.get("mu_rel_error_percent")),
                    _fmt(r.get("K_rel_error_percent")),
                    _fmt(r.get("k0_rel_error")),
                    _fmt(r.get("eig_rel_error_mean")),
                    _fmt(r.get("axis_error_deg_mean")),
                    _fmt(r.get("interface_flux")),
                    _fmt(r.get("runtime_seconds"), digits=2),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Multi-seed Statistics (n >= 3)")
    lines.append("")
    lines.append("| Case | n | Primary (mean ± std) | Field rel-L2 (mean ± std) | Obs rel-L2 (mean ± std) | Traction rel-L2 (mean ± std) | Interface flux (mean ± std) | Runtime (s, mean ± std) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in collected.get("cases", []):
        n = int(case.get("ok_record_count", 0))
        if n < 3:
            continue
        pk = ((case.get("primary_metric") or {}).get("key")) or ""
        summary = case.get("summary", {}) if isinstance(case.get("summary"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("case_id", "-")),
                    str(n),
                    _fmt_mean_std(summary, pk),
                    _fmt_mean_std(summary, "relative_l2_error"),
                    _fmt_mean_std(summary, "obs_rel_l2"),
                    _fmt_mean_std(summary, "traction_rel_l2"),
                    _fmt_mean_std(summary, "interface_flux"),
                    _fmt_mean_std(summary, "runtime_seconds", digits=2),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, best_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("% Auto-generated by build_submission_tables.py")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Case & Primary & Field rel-L2 & Obs rel-L2 & $\\mu$ err (\\%) & $K$ err (\\%) & Runtime (s)\\\\")
    lines.append("\\midrule")
    for r in best_rows:
        lines.append(
            f"{r.get('case_id','-')} & "
            f"{_fmt(r.get('primary_metric_value'))} & "
            f"{_fmt(r.get('relative_l2_error'))} & "
            f"{_fmt(r.get('obs_rel_l2'))} & "
            f"{_fmt(r.get('mu_rel_error_percent'))} & "
            f"{_fmt(r.get('K_rel_error_percent'))} & "
            f"{_fmt(r.get('runtime_seconds'), digits=2)}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission tables from collected core metrics.")
    parser.add_argument(
        "--input",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_metrics_collected.json",
        help="Collected metrics JSON from collect_core_metrics.py",
    )
    parser.add_argument(
        "--output-md",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_submission_tables.md",
        help="Output markdown table path",
    )
    parser.add_argument(
        "--output-best-csv",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_best_seed_results.csv",
        help="Output CSV for best-seed rows",
    )
    parser.add_argument(
        "--output-tex",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_best_seed_table.tex",
        help="Output LaTeX table path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    collected = json.loads(input_path.read_text(encoding="utf-8"))

    best_rows = build_best_rows(collected)
    write_csv(Path(args.output_best_csv), best_rows, BEST_COLUMNS)
    write_markdown(Path(args.output_md), collected, best_rows)
    write_latex(Path(args.output_tex), best_rows)

    print(f"Wrote markdown table: {args.output_md}")
    print(f"Wrote CSV table: {args.output_best_csv}")
    print(f"Wrote LaTeX table: {args.output_tex}")


if __name__ == "__main__":
    main()

