"""Compatibility shim for the archived rabbit Elder inverse benchmark.

The implementation now lives in ``experiments`` because it is no longer part
of the manuscript's core validated benchmark set. This wrapper keeps older
entrypoints working.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_rabbit_inverse_elder_atlas_schwarz import *  # noqa: F401,F403


if __name__ == "__main__":
    from experiments.run_rabbit_inverse_elder_atlas_schwarz import main

    main()
