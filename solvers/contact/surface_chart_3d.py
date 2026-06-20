"""Open 2-D neural HEIGHT chart  h_theta : (x,y) -> z  — the 3-D rock-joint surface chart.

The 3-D extension of ``profile_chart_2d.py`` (the 1-D height chart h(x)).  A single-valued rough
surface (a 3-D rock-joint face) is carried as a learned graph

    z = h_theta(x, y),     h_theta = MLP( Gaussian-random-Fourier-features(x, y) ),

NOT an ambient level set.  Random Fourier features (Tancik et al. 2020) hand the network the high
spatial frequencies directly, defeating the 2-D spectral bias that smooths a plain coordinate MLP —
the same mechanism that makes the chart resolve the asperities a neural SDF erases (manual §11.9).

Contact contract (surface = graph z=h(x,y)):
    vertical gap of a point (x,y,z):  g_vert = z - h(x,y),
    upward unit normal:               n(x,y) = (-h_x, -h_y, 1) / sqrt(1 + h_x^2 + h_y^2),
with h_x, h_y from autograd.

Analytic anchors for verification (closed-form Patton + anisotropy):
  * ``RidgedSawtooth3D`` — sawtooth in x, CONSTANT in y (parallel ridges).  Shearing ACROSS the
    ridges (x, "in-plane") rides up the flanks -> Patton mu_app = tan(phi_b + i) with dilation;
    shearing ALONG the ridges (y, "out-of-plane"/anti-plane) sees no asperity -> pure Coulomb mu,
    zero dilation.  The cleanest in-plane-vs-out-of-plane anisotropy test.
  * ``PyramidSawtooth3D`` — sawtooth in both x and y (egg-crate); dilation in any in-plane direction.

Scope: single-valued surfaces z=h(x,y) (real topography maps are single-valued).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from common.models import MLP


def _eps(dtype: torch.dtype) -> float:
    return max(torch.finfo(dtype).eps, 1e-12)


class NeuralHeight2D(torch.nn.Module):
    """Trained height field h:(x,y)->R as an MLP of GAUSSIAN RANDOM FOURIER FEATURES of (x,y).

    Inputs are normalized to ~[-1,1]^2, then encoded as [cos(2*pi*Xn @ B^T), sin(2*pi*Xn @ B^T)]
    with a fixed Gaussian frequency bank B ~ N(0, sigma^2) of shape (n_freq, 2).  ``sigma`` sets the
    spatial bandwidth (cycles across the normalized domain); larger sigma resolves finer asperities.
    A learnable ``base`` carries the mean height."""

    def __init__(self, xy_lo, xy_hi, sigma: float = 8.0, n_freq: int = 256, width: int = 128,
                 depth: int = 4, base: float = 0.0, seed: int = 0):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(xy_lo, dtype=torch.float64))
        self.register_buffer("hi", torch.as_tensor(xy_hi, dtype=torch.float64))
        span = (self.hi - self.lo).clamp_min(1e-12)
        self.register_buffer("span", span)
        g = torch.Generator().manual_seed(int(seed))
        B = torch.randn(int(n_freq), 2, generator=g, dtype=torch.float64) * float(sigma)
        self.register_buffer("B", B)                                   # (K, 2)
        self.net = MLP(in_dim=2 * int(n_freq), out_dim=1, width=width, depth=depth)
        self.raw_base = torch.nn.Parameter(torch.tensor(float(base), dtype=torch.float64))

    def _xn(self, xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - self.lo) / self.span - 1.0                  # -> ~[-1,1]^2

    def _features(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (self._xn(xy) @ self.B.t())            # (N, K)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2K)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.raw_base + self.net(self._features(xy)).squeeze(-1)


class RidgedSawtooth3D(torch.nn.Module):
    """Parallel-ridge sawtooth: z(x,y) = sawtooth_i(x), constant in y.  Flank slope tan(i)."""

    def __init__(self, wavelength=5.0, angle_deg=20.0):
        super().__init__()
        self.wavelength = float(wavelength)
        self.amp = 0.25 * self.wavelength * math.tan(math.radians(float(angle_deg)))
        self.angle_deg = float(angle_deg)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        s = xy[..., 0] / self.wavelength
        tri = 2.0 * torch.abs(2.0 * (s - torch.floor(s + 0.5))) - 1.0
        return self.amp * tri


class PyramidSawtooth3D(torch.nn.Module):
    """Egg-crate sawtooth: z = amp*(tri(x)+tri(y))/2; flank slope tan(i) along each axis."""

    def __init__(self, wavelength=5.0, angle_deg=20.0):
        super().__init__()
        self.wavelength = float(wavelength)
        self.amp = 0.25 * self.wavelength * math.tan(math.radians(float(angle_deg)))
        self.angle_deg = float(angle_deg)

    def _tri(self, u):
        s = u / self.wavelength
        return 2.0 * torch.abs(2.0 * (s - torch.floor(s + 0.5))) - 1.0

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.amp * 0.5 * (self._tri(xy[..., 0]) + self._tri(xy[..., 1]))


def height_and_grad(xy: torch.Tensor, height: torch.nn.Module
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (h, [h_x, h_y]) with the gradient from autograd.  Safe in an outer no_grad block."""
    with torch.enable_grad():
        p = xy.detach().reshape(-1, 2).requires_grad_(True)
        h = height(p)
        if h.dim() > 1:
            h = h.squeeze(-1)
        gh = torch.autograd.grad(h, p, grad_outputs=torch.ones_like(h),
                                 create_graph=False, retain_graph=False)[0]
    return h.detach(), gh.detach()


def surface_normal_3d(xy: torch.Tensor, height: torch.nn.Module) -> torch.Tensor:
    """Upward unit normal of z=h(x,y): n = (-h_x, -h_y, 1)/sqrt(1+|grad h|^2).  Shape (N,3)."""
    _, gh = height_and_grad(xy, height)
    n = torch.cat([-gh, torch.ones_like(gh[:, :1])], dim=-1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(_eps(xy.dtype))


def fit_surface_chart(xy: np.ndarray, z: np.ndarray, sigma: float = 8.0, n_freq: int = 256,
                      width: int = 128, depth: int = 4, iters: int = 4000, lr: float = 2e-3,
                      batch: int = 8192, plain: bool = False, device: str = "cpu",
                      verbose: bool = False) -> Tuple["NeuralHeight2D", float]:
    """Fit a NeuralHeight2D to sampled surface points (xy (N,2), z (N,)).  Returns (chart, RMSE).
    ``plain=True`` -> sigma forced tiny (a near-Fourier-free coordinate MLP, the ambient-SDF class)."""
    torch.manual_seed(0)
    dev = torch.device(device)
    XY = torch.from_numpy(np.asarray(xy, float)).to(dev)
    Z = torch.from_numpy(np.asarray(z, float)).to(dev)
    s = 0.5 if plain else sigma
    chart = NeuralHeight2D(XY.min(0).values, XY.max(0).values, sigma=s, n_freq=n_freq,
                           width=width, depth=depth, base=float(Z.mean())).to(dev).double()
    opt = torch.optim.Adam(chart.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    n = XY.shape[0]
    for it in range(iters):
        idx = torch.randint(0, n, (min(batch, n),), device=dev)
        loss = torch.mean((chart(XY[idx]) - Z[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 500 == 0 or it == iters - 1):
            with torch.no_grad():
                rmse = float(torch.sqrt(torch.mean((chart(XY) - Z) ** 2)))
            print(f"    it {it:5d}  rmse={rmse:.4e}")
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((chart(XY) - Z) ** 2)))
    return chart, rmse


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # self-test 1: ridged sawtooth slopes — tan(i) across ridges (x), ~0 along ridges (y)
    saw = RidgedSawtooth3D(wavelength=5.0, angle_deg=20.0).double()
    g = torch.tensor(np.stack([np.linspace(0.3, 49.7, 3000), np.linspace(0.0, 30.0, 3000)], 1))
    _, gh = height_and_grad(g, saw)
    med_x = float(torch.median(torch.abs(gh[:, 0]))); max_y = float(torch.abs(gh[:, 1]).max())
    print(f"  ridged: |h_x| median = {med_x:.4f} (tan20={math.tan(math.radians(20)):.4f}), "
          f"|h_y| max = {max_y:.2e}")
    assert abs(med_x - math.tan(math.radians(20.0))) < 1e-6 and max_y < 1e-9
    # self-test 2: NeuralHeight2D fits a rough synthetic surface to a small fraction of RMS
    rng = np.random.RandomState(0)
    xs = np.linspace(0, 20, 120); ys = np.linspace(0, 20, 120)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros_like(XX)
    for _ in range(40):
        kx, ky = rng.uniform(0.2, 2.0, 2); ph = rng.uniform(0, 2 * np.pi)
        Z += (1.0 / (kx + ky)) * np.sin(kx * XX + ky * YY + ph)
    xy = np.stack([XX.ravel(), YY.ravel()], 1)
    chart, rmse = fit_surface_chart(xy, Z.ravel(), sigma=4.0, n_freq=256, iters=1500)
    print(f"  NeuralHeight2D fit RMSE = {rmse:.4e}  (surface RMS = {Z.std():.4e}, "
          f"ratio {rmse/Z.std()*100:.2f}%)")
    assert rmse / Z.std() < 0.08, rmse / Z.std()
    n = surface_normal_3d(torch.from_numpy(xy[:200]), chart)
    assert torch.allclose(n.norm(dim=1), torch.ones(200, dtype=torch.float64), atol=1e-9)
    assert (n[:, 2] > 0).all()
    print("  surface_chart_3d self-test PASSED")
