#!/usr/bin/env python3
"""
Export atlas Schwarz Poisson results to ParaView-compatible VTU files.

Reads:
- rabbit_poisson_schwarz_solution.npz from run_poisson_rabbit_atlas_schwarz.py

Writes:
- merged point-cloud VTU
- per-chart point-cloud VTU files
- manifest json
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def write_vtu_points(
    path: str,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray],
) -> None:
    n = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)  # VTK_VERTEX

    def ascii_data(arr: np.ndarray) -> str:
        return " ".join(map(str, np.asarray(arr).reshape(-1)))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + ascii_data(points) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + ascii_data(connectivity) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + ascii_data(offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + ascii_data(cell_types) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = arr.shape[1]
            else:
                raise ValueError(f"Unsupported data shape for '{name}': {arr.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + ascii_data(arr) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rabbit atlas Schwarz result to ParaView VTU")
    parser.add_argument("--solution-npz", required=True, help="Path to rabbit_poisson_schwarz_solution.npz")
    parser.add_argument("--output-dir", required=True, help="Directory for .vtu outputs")
    parser.add_argument("--prefix", default="rabbit_atlas_poisson", help="Output file prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.solution_npz)

    points = np.asarray(data["points"], dtype=float)
    u_pred = np.asarray(data["u_pred"], dtype=float).reshape(-1)
    u_true = np.asarray(data["u_true"], dtype=float).reshape(-1)
    u_error = np.asarray(data["u_error"], dtype=float).reshape(-1)
    u_error_mag = np.asarray(data["u_error_mag"], dtype=float).reshape(-1)
    chart_id = np.asarray(data["chart_id"], dtype=np.int32).reshape(-1)
    blend_weight = np.asarray(data["blend_weight"], dtype=float).reshape(-1)
    interface_residual = np.asarray(data["interface_residual"], dtype=float).reshape(-1)

    chart_weights = np.asarray(data["chart_weights"], dtype=float)
    chart_values = np.asarray(data["chart_values"], dtype=float)
    n_charts = int(chart_weights.shape[1])

    os.makedirs(args.output_dir, exist_ok=True)

    merged_path = os.path.join(args.output_dir, f"{args.prefix}_merged.vtu")
    write_vtu_points(
        merged_path,
        points=points,
        point_data={
            "u_pred": u_pred,
            "u_true": u_true,
            "u_error": u_error,
            "u_error_mag": u_error_mag,
            "chart_id": chart_id.astype(float),
            "blend_weight": blend_weight,
            "interface_residual": interface_residual,
        },
    )

    per_chart_paths: List[str] = []
    for i in range(n_charts):
        mask = chart_id == i
        if not np.any(mask):
            continue
        p = os.path.join(args.output_dir, f"{args.prefix}_chart_{i:02d}.vtu")
        write_vtu_points(
            p,
            points=points[mask],
            point_data={
                "u_pred": u_pred[mask],
                "u_true": u_true[mask],
                "u_error": u_error[mask],
                "u_error_mag": u_error_mag[mask],
                "chart_id": chart_id[mask].astype(float),
                "blend_weight": blend_weight[mask],
                "interface_residual": interface_residual[mask],
                "chart_weight_local": chart_weights[mask, i],
                "chart_value_local": chart_values[mask, i],
            },
        )
        per_chart_paths.append(p)

    manifest = {
        "solution_npz": args.solution_npz,
        "n_points": int(points.shape[0]),
        "n_charts": n_charts,
        "merged_vtu": merged_path,
        "per_chart_vtu": per_chart_paths,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("ParaView export complete")
    print(f"  merged_vtu: {merged_path}")
    print(f"  charts:     {len(per_chart_paths)} files")
    print(f"  manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
