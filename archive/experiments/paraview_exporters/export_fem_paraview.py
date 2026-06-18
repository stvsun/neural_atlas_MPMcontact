#!/usr/bin/env python3
"""
Export FEM Schwarz Poisson results to ParaView-compatible VTU files.

Reads:
- Solution NPZ files from run_poisson_rabbit_atlas_schwarz_fem.py
- Metrics JSON files for summary info

Writes:
- Per-resolution merged point-cloud VTU (blended solution on rabbit surface)
- Manifest JSON listing all exported files
"""

import argparse
import json
import os
from typing import Dict

import numpy as np


def write_vtu_points(
    path: str,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray],
) -> None:
    """Write a VTK UnstructuredGrid (.vtu) file with VTK_VERTEX cells."""
    n = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

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


def export_solution_npz(
    npz_path: str,
    metrics_path: str,
    output_dir: str,
    prefix: str,
) -> Dict:
    """Export one solution NPZ to VTU + return summary info."""
    data = np.load(npz_path)
    points = np.asarray(data["points"], dtype=float)
    u_pred = np.asarray(data["u_pred"], dtype=float).ravel()
    u_true = np.asarray(data["u_true"], dtype=float).ravel()
    u_error = np.asarray(data["u_error"], dtype=float).ravel()
    u_error_mag = np.asarray(data["u_error_mag"], dtype=float).ravel()

    # Load metrics for extra info
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # Write merged VTU
    vtu_path = os.path.join(output_dir, f"{prefix}.vtu")
    write_vtu_points(
        vtu_path,
        points=points,
        point_data={
            "u_pred": u_pred,
            "u_true": u_true,
            "u_error": u_error,
            "u_error_mag": u_error_mag,
        },
    )

    return {
        "npz": npz_path,
        "vtu": vtu_path,
        "n_points": int(points.shape[0]),
        "n_cells": metrics.get("n_cells"),
        "h": metrics.get("h"),
        "total_dofs": metrics.get("total_dofs"),
        "rel_l2": metrics.get("relative_l2_error"),
        "max_error": metrics.get("max_error"),
        "runtime_s": metrics.get("runtime_seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FEM Schwarz results to ParaView VTU files"
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Directory containing solution NPZ and metrics JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for VTU files (default: {sweep-dir}/paraview)",
    )
    parser.add_argument(
        "--n-cells",
        nargs="*",
        type=int,
        default=None,
        help="Specific n_cells values to export (default: auto-detect all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir
    output_dir = args.output_dir or os.path.join(sweep_dir, "paraview")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect available resolutions
    if args.n_cells:
        n_cells_list = args.n_cells
    else:
        # Find all solution NPZ files
        n_cells_list = []
        for fname in sorted(os.listdir(sweep_dir)):
            if fname.endswith("_solution.npz") and "_n" in fname:
                # Extract n_cells from filename like ..._n16_solution.npz
                parts = fname.replace("_solution.npz", "").split("_n")
                if parts and parts[-1].isdigit():
                    n_cells_list.append(int(parts[-1]))
        n_cells_list = sorted(set(n_cells_list))

    if not n_cells_list:
        print("No solution files found. Check --sweep-dir path.")
        return

    print(f"Exporting FEM solutions for n_cells = {n_cells_list}")
    print(f"Output directory: {output_dir}")

    exports = []
    for nc in n_cells_list:
        npz_path = os.path.join(
            sweep_dir,
            f"rabbit_poisson_schwarz_fem_sweep_n{nc}_solution.npz",
        )
        metrics_path = os.path.join(
            sweep_dir,
            f"rabbit_poisson_schwarz_fem_sweep_n{nc}_metrics.json",
        )

        if not os.path.isfile(npz_path):
            print(f"  [n_cells={nc}] solution NPZ not found: {npz_path}")
            continue

        prefix = f"fem_poisson_n{nc}"
        info = export_solution_npz(npz_path, metrics_path, output_dir, prefix)
        exports.append(info)
        print(
            f"  [n_cells={nc:>3d}] {info['n_points']:>6d} pts  "
            f"rel-L2={info['rel_l2']:.4e}  "
            f"max-err={info['max_error']:.4e}  "
            f"-> {info['vtu']}"
        )

    # Write manifest
    manifest = {
        "description": "FEM P1 Schwarz Poisson on rabbit atlas — h-refinement sweep",
        "n_resolutions": len(exports),
        "exports": exports,
    }
    manifest_path = os.path.join(output_dir, "fem_paraview_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Total VTU files: {len(exports)}")
    print("Done.")


if __name__ == "__main__":
    main()
