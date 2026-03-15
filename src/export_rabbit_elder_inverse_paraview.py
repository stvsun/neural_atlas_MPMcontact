"""Compatibility shim for the archived rabbit Elder ParaView exporter.

The maintained implementation now lives in ``experiments``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.export_rabbit_elder_inverse_paraview import *  # noqa: F401,F403


if __name__ == "__main__":
    from experiments.export_rabbit_elder_inverse_paraview import main

    main()
