#!/usr/bin/env python3
"""
Check whether PyTorch MPS (Apple GPU) is feasible in the current environment.

This script reports:
- hardware and Python architecture
- torch version and MPS build/availability flags
- optional MPS smoke tests (tensor op + tiny autograd step)

Exit code:
- 0: MPS feasible and smoke test passed
- 1: MPS not feasible or smoke test failed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Tuple


def parse_version(v: str) -> Tuple[int, int, int]:
    core = v.split("+", 1)[0]
    parts = core.split(".")
    out = []
    for p in parts[:3]:
        digits = []
        for ch in p:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        out.append(int("".join(digits) or "0"))
    while len(out) < 3:
        out.append(0)
    return tuple(out)  # type: ignore[return-value]


def uname_m() -> str:
    try:
        return (
            subprocess.check_output(["uname", "-m"], text=True, stderr=subprocess.DEVNULL)
            .strip()
            .lower()
        )
    except Exception:
        return "unknown"


def detect_rosetta() -> bool:
    # Heuristic: arm64 kernel but x86_64 Python process.
    return uname_m() == "arm64" and platform.machine().lower() == "x86_64"


def run_smoke(torch_mod: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tensor_op_ok": False,
        "autograd_ok": False,
        "autocast_mps_ok": False,
        "error": None,
        "elapsed_seconds": None,
    }
    start = time.time()
    try:
        device = torch_mod.device("mps")
        x = torch_mod.randn((1024, 128), device=device, dtype=torch_mod.float32, requires_grad=True)
        w = torch_mod.randn((128, 64), device=device, dtype=torch_mod.float32, requires_grad=True)
        y = x @ w
        z = torch_mod.tanh(y).mean()
        out["tensor_op_ok"] = True
        z.backward()
        out["autograd_ok"] = True

        # MPS autocast may vary by torch version; probe safely.
        if hasattr(torch_mod, "autocast"):
            try:
                with torch_mod.autocast(device_type="mps", dtype=torch_mod.float16):
                    xa = torch_mod.randn((256, 64), device=device, dtype=torch_mod.float32)
                    wa = torch_mod.randn((64, 32), device=device, dtype=torch_mod.float32)
                    _ = xa @ wa
                out["autocast_mps_ok"] = True
            except Exception:
                out["autocast_mps_ok"] = False
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["error"] = repr(exc)
    out["elapsed_seconds"] = round(time.time() - start, 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether PyTorch MPS is feasible.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Only report environment capability flags, skip runtime smoke tests.",
    )
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "os": platform.platform(),
        "uname_m": uname_m(),
        "python_machine": platform.machine().lower(),
        "python_executable": sys.executable,
        "rosetta_suspected": detect_rosetta(),
        "torch_import_ok": False,
        "torch_version": None,
        "torch_version_min_recommended": "1.12.0",
        "mps_built": False,
        "mps_available": False,
        "mps_feasible": False,
        "smoke": None,
        "recommendations": [],
    }

    try:
        import torch  # type: ignore

        report["torch_import_ok"] = True
        report["torch_version"] = torch.__version__
        ver_ok = parse_version(torch.__version__) >= parse_version("1.12.0")
        mps_backend = getattr(torch.backends, "mps", None)
        mps_built = bool(mps_backend is not None and torch.backends.mps.is_built())
        mps_available = bool(mps_backend is not None and torch.backends.mps.is_available())
        report["mps_built"] = mps_built
        report["mps_available"] = mps_available

        if not ver_ok:
            report["recommendations"].append("Upgrade PyTorch to >= 1.12.0 (2.x recommended).")
        if report["python_machine"] != "arm64":
            report["recommendations"].append("Use a native arm64 Python interpreter on Apple Silicon.")
        if report["rosetta_suspected"]:
            report["recommendations"].append("Current Python appears under Rosetta. Use native arm64 terminal/env.")
        if not mps_built:
            report["recommendations"].append("Install a PyTorch build compiled with MPS support.")
        if mps_built and not mps_available:
            report["recommendations"].append(
                "MPS is built but unavailable. Check macOS version, arm64 env, and run with "
                "PYTORCH_ENABLE_MPS_FALLBACK=1."
            )

        if mps_available and not args.skip_smoke:
            smoke = run_smoke(torch)
            report["smoke"] = smoke
            report["mps_feasible"] = bool(smoke["tensor_op_ok"] and smoke["autograd_ok"])
            if not report["mps_feasible"]:
                report["recommendations"].append("MPS runtime smoke test failed; use CPU fallback and inspect error.")
        else:
            report["mps_feasible"] = bool(mps_available and mps_built and ver_ok and report["python_machine"] == "arm64")

    except Exception as exc:  # pragma: no cover - runtime dependent
        report["recommendations"].append(f"PyTorch import failed: {exc!r}")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("MPS Feasibility Report")
        print(f"  OS:                {report['os']}")
        print(f"  uname -m:          {report['uname_m']}")
        print(f"  Python machine:    {report['python_machine']}")
        print(f"  Python executable: {report['python_executable']}")
        print(f"  Rosetta suspected: {report['rosetta_suspected']}")
        print(f"  Torch import ok:   {report['torch_import_ok']}")
        print(f"  Torch version:     {report['torch_version']}")
        print(f"  MPS built:         {report['mps_built']}")
        print(f"  MPS available:     {report['mps_available']}")
        if report["smoke"] is not None:
            smoke = report["smoke"]
            print("  Smoke test:")
            print(f"    tensor op:       {smoke['tensor_op_ok']}")
            print(f"    autograd:        {smoke['autograd_ok']}")
            print(f"    autocast(mps):   {smoke['autocast_mps_ok']}")
            print(f"    elapsed (s):     {smoke['elapsed_seconds']}")
            if smoke["error"] is not None:
                print(f"    error:           {smoke['error']}")
        print(f"  MPS feasible:      {report['mps_feasible']}")
        if report["recommendations"]:
            print("  Recommendations:")
            for rec in report["recommendations"]:
                print(f"    - {rec}")

    return 0 if report["mps_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
