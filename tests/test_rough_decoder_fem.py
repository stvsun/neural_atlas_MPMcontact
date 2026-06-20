"""Verify the GENUINE rough-geometry atlas (manual 11.11) BEFORE trusting any contact result:
  * the Fourier boundary-fitted decoder reconstructs the rough surface (and beats a plain-MLP decoder);
  * the chart-FEM on the decoder has NO element foldover (det J > 0);
  * MMS O(h^2) convergence on the curved rough geometry (the FEM solves correctly on it).

These are the gates the PI mandate requires (train the decoders and verify first; no shortcuts).
Slow (trains small nets) — runs as a benchmark-grade test, like the other CV-7 checks.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")

from solvers.fem.rough_block_decoder import (  # noqa: E402
    band_limited_rough_surface, train_rough_decoder, verify_decoder)


def _surf_rms(amp=0.10):
    r = np.random.RandomState(3)
    return float(np.std(band_limited_rough_surface(r.uniform(-1, 1, 5000), r.uniform(-1, 1, 5000), amp=amp)))


def test_fourier_decoder_reconstructs_and_no_foldover():
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.10)     # noqa: E731
    dec, rmse, dk = train_rough_decoder(tgt, n_freq=20, iters=2500)
    assert rmse / _surf_rms(0.10) < 0.08, rmse                       # resolves the rough surface
    v = verify_decoder(dec, dk, n_cells=10)
    assert v["all_valid"] and v["detJ_min"] > 0, v                   # chart-FEM well-posed (no foldover)


def test_fourier_beats_plain_decoder():
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.10)     # noqa: E731
    _, rmse_f, _ = train_rough_decoder(tgt, iters=2500, plain=False)
    _, rmse_p, _ = train_rough_decoder(tgt, iters=2500, plain=True)
    assert rmse_f < 0.3 * rmse_p, (rmse_f, rmse_p)                    # Fourier features essential


def test_decoder_mms_convergence_O_h2():
    from benchmarks.contact.cv_numerical.cv7_decoder_verify import mms_convergence
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.10)     # noqa: E731
    dec, _, dk = train_rough_decoder(tgt, iters=2500)
    errs, rates = mms_convergence(dec, dk, cells=(4, 8, 12))
    assert errs[-1] < errs[0]                                         # refinement helps
    assert max(rates) > 1.7                                          # a clean O(h^2) step
