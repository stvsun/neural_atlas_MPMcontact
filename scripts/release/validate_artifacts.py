#!/usr/bin/env python3
"""Validate canonical core benchmark artifacts and emit checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _add_artifact(
    out: List[Dict[str, Any]],
    *,
    case_id: str,
    record_id: str,
    role: str,
    path_str: str,
    required: bool,
) -> Tuple[bool, str]:
    p = Path(path_str)
    exists = p.exists()
    entry: Dict[str, Any] = {
        "case_id": case_id,
        "record_id": record_id,
        "role": role,
        "path": str(p),
        "required": required,
        "exists": exists,
        "size_bytes": None,
        "sha256": None,
    }
    if exists:
        try:
            entry["size_bytes"] = p.stat().st_size
            if p.is_file():
                entry["sha256"] = _sha256(p)
        except Exception as exc:  # pragma: no cover
            entry["error"] = str(exc)
    out.append(entry)

    if not exists:
        return False, f"{'REQUIRED' if required else 'OPTIONAL'} missing {role}: {p}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical core benchmark artifacts.")
    parser.add_argument(
        "--registry",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_benchmarks_registry.json",
        help="Path to core benchmark registry JSON",
    )
    parser.add_argument(
        "--manifest-out",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_artifact_manifest.json",
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--checksums-out",
        default="/Users/wsun/Documents/Softwares/Mapped_sphere_method_for_complex_geometry/PINN_coordinate_chart_3Dgeometry/runs/core_artifact_checksums.txt",
        help="Output checksum text path",
    )
    parser.add_argument(
        "--strict-figures",
        action="store_true",
        help="Treat missing figure files as errors (default: warnings only)",
    )
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    artifacts: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    for case in registry.get("main_cases", []):
        case_id = str(case.get("case_id"))
        for rec in case.get("records", []):
            record_id = str(rec.get("record_id"))
            ok, msg = _add_artifact(
                artifacts,
                case_id=case_id,
                record_id=record_id,
                role="run_dir",
                path_str=str(rec.get("run_dir", "")),
                required=True,
            )
            if not ok:
                errors.append(msg)

            ok, msg = _add_artifact(
                artifacts,
                case_id=case_id,
                record_id=record_id,
                role="metrics",
                path_str=str(rec.get("metrics_path", "")),
                required=True,
            )
            if not ok:
                errors.append(msg)

            ckpt = rec.get("checkpoint_path")
            if ckpt:
                ok, msg = _add_artifact(
                    artifacts,
                    case_id=case_id,
                    record_id=record_id,
                    role="checkpoint",
                    path_str=str(ckpt),
                    required=True,
                )
                if not ok:
                    errors.append(msg)

            for fig in rec.get("figures", []):
                ok, msg = _add_artifact(
                    artifacts,
                    case_id=case_id,
                    record_id=record_id,
                    role="figure",
                    path_str=str(fig),
                    required=bool(args.strict_figures),
                )
                if not ok:
                    if args.strict_figures:
                        errors.append(msg)
                    else:
                        warnings.append(msg)

    summary = {
        "registry_path": args.registry,
        "total_artifacts": len(artifacts),
        "existing_artifacts": sum(1 for a in artifacts if a.get("exists")),
        "missing_artifacts": sum(1 for a in artifacts if not a.get("exists")),
        "errors": errors,
        "warnings": warnings,
    }

    manifest = {
        "summary": summary,
        "artifacts": artifacts,
    }

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    checksum_lines: List[str] = []
    for a in artifacts:
        if a.get("exists") and a.get("sha256"):
            checksum_lines.append(f"{a['sha256']}  {a['path']}")
    checksums_path = Path(args.checksums_out)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote checksums: {checksums_path}")
    print(
        "Validation summary: "
        f"{summary['existing_artifacts']}/{summary['total_artifacts']} exist; "
        f"errors={len(errors)}, warnings={len(warnings)}"
    )
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

