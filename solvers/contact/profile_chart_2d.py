"""Open 1-D neural HEIGHT chart  h_theta : x -> z  (the rock-joint capstone's coordinate chart).

The 1-D, *non-periodic* companion to ``radial_chart_2d.py`` (the periodic radial chart rho:S^1->R+).
A single-valued rough surface (a rock-joint profile) is carried as a learned graph

    z = h_theta(x),      h_theta = MLP( random-Fourier-features(x) ),

NOT as an ambient level set.  This is the neural-atlas thesis specialized to a surface-of-a-graph:
the geometry is a *function on the surface's own parameter* x, so it resolves every roughness scale
with O(surface) storage and O(1) evaluation, and **random Fourier features hand the network the high
frequencies directly**, defeating the 1-D spectral bias that smooths a plain coordinate MLP (the same
mechanism that let ``NeuralRho2D`` beat the ambient neural SDF on the cusped superformula, manual
§11.4, and that the CV-6 Koch ceiling measures from the SDF side).

The contact contract is the usual one.  For a query point ``p=(x,z)`` above/below the surface the
*vertical* gap and the chart normal are

    gap_vert(x, z) = z - h_theta(x),
    normal(x)      = ( -h'(x), 1 ) / sqrt(1 + h'(x)^2),      h'(x) = d h_theta / dx  (autograd),

with the true (minimal) normal gap obtained by projecting the vertical gap onto the surface normal,
``gap_n ~ gap_vert / sqrt(1 + h'^2)`` (small-slope-exact; the node-to-surface penalty in the shear
driver uses the surface normal directly).  ``AnalyticSawtooth1D`` is the closed-form regular-asperity
surface whose direct shear has the Patton law tau/sigma = tan(phi_b + i) — the L1 verification anchor.

Scope: single-valued profiles (every x has one height) — real rock-joint topography maps are
single-valued, so this covers the capstone; genuine overhangs would need the transition-map chart.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from common.models import MLP


def _eps(dtype: torch.dtype) -> float:
    return max(torch.finfo(dtype).eps, 1e-12)


class NeuralHeight1D(torch.nn.Module):
    """Trained height function h:[x0-? , ?]->R as an MLP of RANDOM FOURIER FEATURES of x.

    The input is encoded as ``[cos(2*pi*B*x_tilde), sin(2*pi*B*x_tilde)]`` where ``x_tilde`` is x
    mapped to roughly [-1,1] and ``B`` is a fixed bank of frequencies spanning DC up to ``f_max``
    half-wavelengths across the domain (geometric spacing → equal log-coverage of scales).  This is
    the deterministic Fourier-feature encoding (Tancik et al. 2020) that defeats spectral bias; the
    rough Inada profile needs ~L/dx ≈ 3000 resolvable wavelengths, so ``f_max`` is set accordingly.
    A learnable ``base`` carries the mean height so the net learns only the deviation."""

    def __init__(self, x_lo: float, x_hi: float, f_max: float = 1500.0, n_freq: int = 192,
                 width: int = 128, depth: int = 4, base: float = 0.0, plain: bool = False):
        super().__init__()
        self.x_lo = float(x_lo)
        self.x_hi = float(x_hi)
        self.span = float(x_hi - x_lo) if x_hi > x_lo else 1.0
        self.plain = bool(plain)                                # plain=True -> NO Fourier features
        if self.plain:
            # a bare coordinate MLP of x (the representation an ambient neural SDF is built from):
            # subject to spectral bias -> cannot resolve high-frequency asperities (the ceiling).
            self.register_buffer("freqs", torch.zeros(0, dtype=torch.float64))
            self.net = MLP(in_dim=1, out_dim=1, width=width, depth=depth)
        else:
            # geometric frequency bank (cycles across the normalized domain), DC handled by `base`
            freqs = torch.from_numpy(np.geomspace(0.5, float(f_max), int(n_freq))).double()
            self.register_buffer("freqs", freqs)
            self.net = MLP(in_dim=2 * int(n_freq), out_dim=1, width=width, depth=depth)
        self.raw_base = torch.nn.Parameter(torch.tensor(float(base), dtype=torch.float64))

    def _xtilde(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.x_lo) / self.span - 1.0          # -> ~[-1, 1]

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        xt = self._xtilde(x).unsqueeze(-1)                      # (N,1)
        if self.plain:
            return xt                                          # raw coordinate -> spectral bias
        ang = 2.0 * math.pi * xt * self.freqs                  # (N,K)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)   # (N,2K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw_base + self.net(self._features(x)).squeeze(-1)


class AnalyticSawtooth1D(torch.nn.Module):
    """Closed-form regular sawtooth (triangular) asperity surface — the Patton L1 anchor.

    z(x) = amplitude * triangle(x / wavelength), a symmetric triangle wave whose flank slope equals
    ``tan(i)`` for asperity angle ``i`` (degrees): ``amplitude = (wavelength/4) * tan(i)`` (a triangle
    of period lambda has flank slope 4*amplitude/lambda).  Direct shear of two mating sawtooth
    faces under plain Coulomb friction (base angle phi_b) must emergently give the Patton law
    tau/sigma = tan(phi_b + i) in the ride-up (dilation) regime — a real closed form, not imposed."""

    def __init__(self, wavelength: float = 5.0, angle_deg: float = 15.0, x_lo: float = 0.0,
                 x_hi: float = 50.0):
        super().__init__()
        self.wavelength = float(wavelength)
        self.angle_deg = float(angle_deg)
        self.amp = 0.25 * self.wavelength * math.tan(math.radians(self.angle_deg))
        self.x_lo, self.x_hi = float(x_lo), float(x_hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # symmetric triangle wave in [-amp, amp], period = wavelength, slope magnitude = tan(i)
        s = x / self.wavelength
        tri = 2.0 * torch.abs(2.0 * (s - torch.floor(s + 0.5))) - 1.0   # in [-1,1]
        return self.amp * tri


def height_and_grad(x: torch.Tensor, height: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (h(x), h'(x)) with h' from autograd.  Safe inside an outer no_grad block; returns
    detached tensors."""
    with torch.enable_grad():
        xg = x.detach().reshape(-1).requires_grad_(True)
        h = height(xg)
        if h.dim() > 1:
            h = h.squeeze(-1)
        dh = torch.autograd.grad(h, xg, grad_outputs=torch.ones_like(h),
                                 create_graph=False, retain_graph=False)[0]
    return h.detach(), dh.detach()


def surface_normal(x: torch.Tensor, height: torch.nn.Module) -> torch.Tensor:
    """Upward unit normal of the graph z=h(x): n = (-h', 1)/sqrt(1+h'^2).  Shape (N,2)."""
    _, dh = height_and_grad(x, height)
    n = torch.stack([-dh, torch.ones_like(dh)], dim=-1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(_eps(x.dtype))


def vertical_gap(x: torch.Tensor, z: torch.Tensor, height: torch.nn.Module
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vertical gap of points (x,z) above the surface z=h(x): gap=z-h(x) (<0 => below/penetrating),
    and the surface unit normal at x.  Returns (gap (N,), normal (N,2))."""
    h, dh = height_and_grad(x, height)
    n = torch.stack([-dh, torch.ones_like(dh)], dim=-1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(_eps(x.dtype))
    return (z.reshape(-1) - h), n


def fit_height_chart(x_data: np.ndarray, z_data: np.ndarray, f_max: float = 1500.0,
                     n_freq: int = 192, width: int = 128, depth: int = 4, iters: int = 4000,
                     lr: float = 2e-3, batch: int = 4096, device: str = "cpu",
                     plain: bool = False, verbose: bool = False) -> Tuple["NeuralHeight1D", float]:
    """Fit a NeuralHeight1D to sampled profile points (x_data, z_data) by MSE regression.  Returns
    (trained chart, final RMSE in the same units as z_data).  ``plain=True`` fits the Fourier-free
    coordinate MLP (the spectral-bias-limited ambient-SDF-class representation) at matched width/depth
    — the controlled chart-vs-SDF ablation.  CPU/float64 by default (tiny net, better conditioning)."""
    torch.manual_seed(0)
    dev = torch.device(device)
    xt = torch.from_numpy(np.asarray(x_data, float)).to(dev)
    zt = torch.from_numpy(np.asarray(z_data, float)).to(dev)
    chart = NeuralHeight1D(float(xt.min()), float(xt.max()), f_max=f_max, n_freq=n_freq,
                           width=width, depth=depth, base=float(zt.mean()), plain=plain).to(dev).double()
    opt = torch.optim.Adam(chart.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    n = xt.shape[0]
    for it in range(iters):
        idx = torch.randint(0, n, (min(batch, n),), device=dev)
        pred = chart(xt[idx])
        loss = torch.mean((pred - zt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 500 == 0 or it == iters - 1):
            with torch.no_grad():
                rmse = float(torch.sqrt(torch.mean((chart(xt) - zt) ** 2)))
            print(f"    it {it:5d}  rmse={rmse:.4e}")
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((chart(xt) - zt) ** 2)))
    return chart, rmse


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # self-test 1: analytic sawtooth slope magnitude == tan(i) everywhere (away from the apex kinks)
    saw = AnalyticSawtooth1D(wavelength=5.0, angle_deg=20.0).double()
    x = torch.linspace(0.3, 49.7, 4000, dtype=torch.float64)
    _, dh = height_and_grad(x, saw)
    expect = math.tan(math.radians(20.0))
    # most points sit on a flank with |slope|=tan(i); the few near apexes differ — check the median
    med = float(torch.median(torch.abs(dh)))
    print(f"  sawtooth |slope| median = {med:.4f}  (tan 20deg = {expect:.4f})")
    assert abs(med - expect) < 1e-6, med
    # self-test 2: a NeuralHeight1D fits a moderately rough synthetic profile to < few % of its RMS
    rng = np.random.RandomState(0)
    xs = np.linspace(0.0, 50.0, 4000)
    ks = np.arange(1, 60)
    amp = ks ** (-1.0)                                       # ~self-affine-ish spectrum
    ph = rng.uniform(0, 2 * np.pi, ks.size)
    zs = sum(a * np.sin(2 * np.pi * k * xs / 50.0 + p) for a, k, p in zip(amp, ks, ph))
    chart, rmse = fit_height_chart(xs, zs, f_max=120.0, n_freq=128, iters=2500, verbose=False)
    print(f"  neural height-chart fit RMSE = {rmse:.4e}  (profile RMS = {zs.std():.4e}, "
          f"ratio {rmse/zs.std()*100:.2f}%)")
    assert rmse / zs.std() < 0.05, rmse / zs.std()
    # normal is unit and points up
    n = surface_normal(torch.from_numpy(xs[:200]), chart)
    assert torch.allclose(n.norm(dim=1), torch.ones(200, dtype=torch.float64), atol=1e-10)
    assert (n[:, 1] > 0).all()
    print("  profile_chart_2d self-test PASSED")
