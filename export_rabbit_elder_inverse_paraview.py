#!/usr/bin/env python3
"""
Export rabbit inverse Elder atlas solution to ParaView-compatible VTU files.

Reads:
- rabbit_inverse_elder_atlas_schwarz*_solution.npz
- optional atlas_data.npz for full chart-support exports

Writes:
- merged VTU with p/c/u predictions and errors
- per-chart VTU files
- manifest JSON
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np


def write_vtu_points(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape [N,3]")
    n = int(points.shape[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)  # VTK_VERTEX

    def arr_ascii(a: np.ndarray) -> str:
        return " ".join(map(str, np.asarray(a).reshape(-1)))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + arr_ascii(points) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + arr_ascii(connectivity) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + arr_ascii(offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + arr_ascii(cell_types) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for key, value in point_data.items():
            v = np.asarray(value)
            if v.ndim == 1:
                n_comp = 1
            elif v.ndim == 2:
                n_comp = int(v.shape[1])
            else:
                raise ValueError(f"Unsupported shape for PointData '{key}': {v.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{key}" NumberOfComponents="{n_comp}" format="ascii">\n'
            )
            f.write("          " + arr_ascii(v) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def get_array(data: np.lib.npyio.NpzFile, key: str, ndim: Optional[int] = None) -> Optional[np.ndarray]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Array '{key}' expected ndim={ndim}, got {arr.ndim}")
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rabbit inverse Elder results to VTU")
    parser.add_argument("--solution-npz", required=True, help="Path to *_solution.npz")
    parser.add_argument("--output-dir", required=True, help="Output directory for VTU files")
    parser.add_argument("--atlas-data", default=None, help="Optional atlas_data.npz for full chart support")
    parser.add_argument("--prefix", default="rabbit_inverse_elder", help="Output filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.solution_npz)
    points = np.asarray(data["points"], dtype=float)
    chart_id = np.asarray(data["chart_id"], dtype=np.int32).reshape(-1)

    p_true = np.asarray(data["p_true"], dtype=float).reshape(-1)
    p_pred = np.asarray(data["p_pred"], dtype=float).reshape(-1)
    p_error = np.asarray(data["p_error"], dtype=float).reshape(-1)
    p_error_mag = np.asarray(data["p_error_mag"], dtype=float).reshape(-1)

    c_true = np.asarray(data["c_true"], dtype=float).reshape(-1)
    c_pred = np.asarray(data["c_pred"], dtype=float).reshape(-1)
    c_error = np.asarray(data["c_error"], dtype=float).reshape(-1)
    c_error_mag = np.asarray(data["c_error_mag"], dtype=float).reshape(-1)

    u_true = np.asarray(data["u_true"], dtype=float)
    u_pred = np.asarray(data["u_pred"], dtype=float)
    u_error = np.asarray(data["u_error"], dtype=float)
    u_error_mag = np.asarray(data["u_error_mag"], dtype=float).reshape(-1)

    detail_score = get_array(data, "detail_score", ndim=1)
    high_detail_mask = get_array(data, "high_detail_mask", ndim=1)

    merged_data: Dict[str, np.ndarray] = {
        "chart_id": chart_id.astype(float),
        "p_true": p_true,
        "p_pred": p_pred,
        "p_error": p_error,
        "p_error_mag": p_error_mag,
        "pressure_true": p_true,
        "pressure": p_pred,
        "pressure_error": p_error,
        "pressure_error_mag": p_error_mag,
        "c_true": c_true,
        "c_pred": c_pred,
        "c_error": c_error,
        "c_error_mag": c_error_mag,
        "u_true": u_true,
        "u_pred": u_pred,
        "u_error": u_error,
        "u_error_mag": u_error_mag,
        "velocity_true": u_true,
        "velocity": u_pred,
        "velocity_error": u_error,
        "velocity_error_mag": u_error_mag,
    }
    if detail_score is not None:
        merged_data["detail_score"] = detail_score
    if high_detail_mask is not None:
        merged_data["high_detail_mask"] = high_detail_mask

    merged_path = os.path.join(args.output_dir, f"{args.prefix}_merged.vtu")
    write_vtu_points(merged_path, points, merged_data)

    membership = None
    n_charts = int(np.max(chart_id) + 1) if chart_id.size else 0
    if args.atlas_data is not None and os.path.isfile(args.atlas_data):
        atlas = np.load(args.atlas_data)
        if "membership" in atlas.files:
            membership = np.asarray(atlas["membership"]).astype(bool)
            n_charts = int(membership.shape[1])

    per_chart_paths: List[str] = []
    for i in range(n_charts):
        if membership is not None:
            mask = membership[:, i]
        else:
            mask = chart_id == i
        if not np.any(mask):
            continue
        p = os.path.join(args.output_dir, f"{args.prefix}_chart_{i:02d}.vtu")
        pdata: Dict[str, np.ndarray] = {
            "chart_id": chart_id[mask].astype(float),
            "p_true": p_true[mask],
            "p_pred": p_pred[mask],
            "p_error": p_error[mask],
            "p_error_mag": p_error_mag[mask],
            "pressure_true": p_true[mask],
            "pressure": p_pred[mask],
            "pressure_error": p_error[mask],
            "pressure_error_mag": p_error_mag[mask],
            "c_true": c_true[mask],
            "c_pred": c_pred[mask],
            "c_error": c_error[mask],
            "c_error_mag": c_error_mag[mask],
            "u_true": u_true[mask],
            "u_pred": u_pred[mask],
            "u_error": u_error[mask],
            "u_error_mag": u_error_mag[mask],
            "velocity_true": u_true[mask],
            "velocity": u_pred[mask],
            "velocity_error": u_error[mask],
            "velocity_error_mag": u_error_mag[mask],
        }
        if detail_score is not None:
            pdata["detail_score"] = detail_score[mask]
        if high_detail_mask is not None:
            pdata["high_detail_mask"] = high_detail_mask[mask]
        write_vtu_points(p, points[mask], pdata)
        per_chart_paths.append(p)

    manifest = {
        "solution_npz": args.solution_npz,
        "atlas_data": args.atlas_data,
        "n_points": int(points.shape[0]),
        "n_charts": int(n_charts),
        "merged_vtu": merged_path,
        "per_chart_vtu": per_chart_paths,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("ParaView export complete")
    print(f"  merged_vtu: {merged_path}")
    print(f"  per_chart:  {len(per_chart_paths)} files")
    print(f"  manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
