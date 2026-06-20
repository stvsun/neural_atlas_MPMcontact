"""Trainable boundary-fitted ChartDecoder for a ROUGH-faced elastic block (the genuine neural atlas).

The neural-atlas thesis is that geometry should be a learned boundary-fitted COORDINATE CHART, not an
ambient level set.  This decoder maps a reference cube xi in [-1,1]^3 to a physical block whose ONE
face is a rough rock-joint surface z = h(x,y); the chart-FEM (`chart_vector_fem.py`) then solves
elasticity on the *actual rough geometry* via the chart-Jacobian pushforward — so dilation/friction
emerge from the real asperities, NOT from an imposed effective dilation angle (manual 11.10's flat
benchmark).

    x = seed + xi_x t1 + xi_y t2 + xi_z n  +  relief(xi_x, xi_y) * ramp(xi_z) * n_hat,
    relief = MLP( FOURIER-FEATURES(xi_x, xi_y) )   trained to fit the rough surface,
    ramp(xi_z) = (xi_z+1)/2  (rough TOP face) or (1-xi_z)/2  (rough BOTTOM face), 0 at the flat face.

CRITICAL (honesty): the relief net uses FOURIER FEATURES.  A vanilla tanh MLP (the stock
``common.models.ChartDecoder``) suffers the SAME spectral bias as an ambient neural SDF and would
SMOOTH the asperities — so a fair "atlas beats level set" claim requires the chart to actually resolve
the roughness.  `train_rough_decoder` fits the relief; `verify_decoder` checks reconstruction AND that
the chart-FEM sees no foldover (det J > 0) before any contact is attempted.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from common.models import MLP


# --------------------------------------------------------------------------------------------------
def band_limited_rough_surface(x, y, n_modes=6, amp=0.12, k_min=0.6, k_max=2.2, seed=0):
    """A band-limited rough height field h(x,y) the FEM mesh can resolve (a few asperity wavelengths).
    Representative of a rock-joint surface at the resolved scale; deterministic for a given seed."""
    rng = np.random.RandomState(seed)
    f = np.zeros(np.shape(x))
    for kx, ky, ph in zip(rng.uniform(k_min, k_max, n_modes), rng.uniform(k_min, k_max, n_modes),
                          rng.uniform(0, 2 * np.pi, n_modes)):
        f = f + np.sin(math.pi * (kx * x + ky * y) + ph)
    f = f / np.abs(f).max()
    return amp * f


class RoughBlockDecoder(torch.nn.Module):
    """ChartDecoder-compatible map (xi -> x) for a rough-faced block; Fourier-feature relief net."""

    def __init__(self, rough_face: str = "top", n_freq: int = 20, width: int = 64, depth: int = 3,
                 k_max: float = 6.0, plain: bool = False):
        super().__init__()
        assert rough_face in ("top", "bottom")
        self.rough_face = rough_face
        self.n_freq = int(n_freq)
        self.plain = bool(plain)                                     # plain=True -> NO Fourier features
        if self.plain:
            self.register_buffer("B", torch.zeros(0, 2, dtype=torch.float64))
            self.net = MLP(in_dim=2, out_dim=1, width=width, depth=depth)
        else:
            # deterministic 2-D Fourier frequency bank (cycles across xi in [-1,1]) -> resolves roughness
            g = torch.Generator().manual_seed(0)
            B = torch.randn(int(n_freq), 2, generator=g, dtype=torch.float64) * (k_max / 3.0)
            self.register_buffer("B", B)
            self.net = MLP(in_dim=2 * int(n_freq), out_dim=1, width=width, depth=depth)

    def _relief(self, xi_xy: torch.Tensor) -> torch.Tensor:
        if self.plain:                                              # bare coordinate MLP (spectral bias)
            return self.net(xi_xy).squeeze(-1)
        proj = math.pi * (xi_xy @ self.B.t())                       # (N, n_freq)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.net(feat).squeeze(-1)

    def forward(self, xi: torch.Tensor, seed: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
                n: torch.Tensor, chart_scale: torch.Tensor) -> torch.Tensor:
        base = (seed.unsqueeze(0) + xi[:, 0:1] * t1.unsqueeze(0) + xi[:, 1:2] * t2.unsqueeze(0)
                + xi[:, 2:3] * n.unsqueeze(0))
        n_hat = n / torch.clamp(n.norm(), min=1e-9)
        ramp = (xi[:, 2] + 1.0) * 0.5 if self.rough_face == "top" else (1.0 - xi[:, 2]) * 0.5
        relief = self._relief(xi[:, :2]) * ramp
        return base + relief.unsqueeze(-1) * n_hat.unsqueeze(0)

    def rough_face_xi_z(self) -> float:
        return 1.0 if self.rough_face == "top" else -1.0


# --------------------------------------------------------------------------------------------------
def train_rough_decoder(target_fn, rough_face="top", L=1.0, n_freq=20, width=64, depth=3,
                        k_max=6.0, iters=4000, lr=2e-3, n_pts=6000, device="cpu", plain=False,
                        verbose=False) -> Tuple["RoughBlockDecoder", float, dict]:
    """Fit the relief net so the decoder's rough face matches target_fn(x,y) (x,y = L*xi_xy in [-L,L]).
    ``plain=True`` fits the Fourier-free coordinate-MLP decoder (the spectral-bias baseline).
    Returns (decoder, surface RMSE in physical units, decoder_kwargs for the FEM)."""
    torch.manual_seed(0)
    dev = torch.device(device)
    dec = RoughBlockDecoder(rough_face=rough_face, n_freq=n_freq, width=width, depth=depth,
                            k_max=k_max, plain=plain).to(dev).double()
    dk = dict(seed=torch.zeros(3, dtype=torch.float64, device=dev),
              t1=torch.tensor([L, 0, 0], dtype=torch.float64, device=dev),
              t2=torch.tensor([0, L, 0], dtype=torch.float64, device=dev),
              n=torch.tensor([0, 0, L], dtype=torch.float64, device=dev),
              chart_scale=torch.tensor(1.0, dtype=torch.float64, device=dev))
    # supervise the relief directly: at the rough face, z_phys = (+/-L) + relief, target = target_fn
    rng = np.random.RandomState(0)
    xy = rng.uniform(-1, 1, (n_pts, 2))
    tgt = target_fn(L * xy[:, 0], L * xy[:, 1])                      # target relief (physical height)
    xy_t = torch.from_numpy(xy).to(dev); tgt_t = torch.from_numpy(tgt).to(dev)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        idx = torch.randint(0, n_pts, (2048,), device=dev)
        loss = torch.mean((dec._relief(xy_t[idx]) - tgt_t[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 800 == 0 or it == iters - 1):
            with torch.no_grad():
                rmse = float(torch.sqrt(torch.mean((dec._relief(xy_t) - tgt_t) ** 2)))
            print(f"    decoder fit it {it:5d}  rmse={rmse:.4e}")
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((dec._relief(xy_t) - tgt_t) ** 2)))
    return dec, rmse, dk


def verify_decoder(dec, dk, n_cells=10, support_r=1.0):
    """Build the chart-FEM on the decoder and check it is well-posed: every element valid, det J > 0
    (no foldover), and report the worst Jacobian conditioning.  Run BEFORE any contact."""
    from solvers.fem.chart_vector_fem import ChartVectorFEMSolver
    solver = ChartVectorFEMSolver(n_cells=n_cells, support_r=support_r, chart_decoder=dec,
                                  decoder_kwargs=dk, dtype=torch.float64)
    return dict(solver=solver, n_valid=int(solver.geom_valid.sum()),
                n_elem=int(solver.n_elements),
                all_valid=bool(solver.geom_valid.all()),
                detJ_min=float(solver.geom_detJ.min()),
                detJ_max=float(solver.geom_detJ.max()))


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print("=== train + verify a rough-block ChartDecoder ===")
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.12)     # noqa: E731
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", iters=3000, verbose=True)
    rms_surf = float(np.std(tgt(np.random.RandomState(1).uniform(-1, 1, 4000),
                                np.random.RandomState(2).uniform(-1, 1, 4000))))
    print(f"  surface reconstruction RMSE = {rmse:.4e}  ({rmse/rms_surf*100:.2f}% of surface RMS)")
    v = verify_decoder(dec, dk, n_cells=10)
    print(f"  chart-FEM on decoder: {v['n_valid']}/{v['n_elem']} elements valid, "
          f"det J in [{v['detJ_min']:.3e}, {v['detJ_max']:.3e}]  all_valid={v['all_valid']}")
    assert rmse / rms_surf < 0.08, "decoder does not resolve the rough surface"
    assert v["all_valid"] and v["detJ_min"] > 0, "decoder geometry foldover — reduce amplitude / refine"
    print("  rough_block_decoder self-test PASSED")
