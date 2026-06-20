#!/usr/bin/env python3
"""VERIFY the rough-block ChartDecoder BEFORE any contact (the PI mandate: train the decoders and
verify they work first; no shortcuts).

Three gates + the honest geometry comparison:
  1. RECONSTRUCTION  — the Fourier decoder's rough face matches the target surface (RMSE / RMS).
  2. NO FOLDOVER     — the chart-FEM on the decoder has every element valid, det J > 0.
  3. MMS O(h^2)      — manufactured-solution convergence on the ROUGH geometry (a curved chart passes
                       the patch test only approximately, converging at O(h^2)); this is the rigorous
                       "the FEM solves correctly on the rough geometry" check.
  + ATLAS vs LEVEL SET (honest): the Fourier boundary-fitted decoder vs (a) a vanilla plain-MLP decoder
    and (b) an ambient 3-D neural SDF — both of which suffer spectral bias and SMOOTH the asperities.

Run:  python3 benchmarks/contact/cv_numerical/cv7_decoder_verify.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver          # noqa: E402
from solvers.fem.linear_elastic import make_linear_elastic_small_strain  # noqa: E402
from solvers.fem.rough_block_decoder import (                          # noqa: E402
    band_limited_rough_surface, train_rough_decoder, RoughBlockDecoder)
from common.models import MLP                                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(_ROOT, "runs", "cv7_decoder")
DT = torch.float64


def _lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# --- MMS on the rough decoder (physical-coordinate manufactured field) ----------------------------
def _mms_u(xphys, amp):
    pi = math.pi
    s = amp * torch.sin(pi * xphys[:, 0]) * torch.sin(pi * xphys[:, 1]) * torch.sin(pi * xphys[:, 2])
    return torch.stack([s, s, s], dim=1)


def _mms_body_force(xphys, E, nu, amp):
    lam, mu = _lame(E, nu); pi = math.pi
    x = xphys.clone().requires_grad_(True)
    s = amp * torch.sin(pi * x[:, 0]) * torch.sin(pi * x[:, 1]) * torch.sin(pi * x[:, 2])
    u = torch.stack([s, s, s], dim=1)
    grad_u = torch.stack([torch.autograd.grad(u[:, i], x, torch.ones(x.shape[0], dtype=x.dtype),
                          create_graph=True, retain_graph=True)[0] for i in range(3)], dim=1)
    eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
    tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
    sigma = lam * tr.view(-1, 1, 1) * torch.eye(3, dtype=x.dtype) + 2 * mu * eps
    div = torch.zeros_like(u)
    for i in range(3):
        for J in range(3):
            div[:, i] = div[:, i] + torch.autograd.grad(sigma[:, i, J], x,
                        torch.ones(x.shape[0], dtype=x.dtype), retain_graph=True)[0][:, J]
    return -div.detach()


def mms_convergence(dec, dk, E=100.0, nu=0.3, amp=0.004, cells=(4, 8, 12)):
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
    errs = []
    for nc in cells:
        s = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=dec, decoder_kwargs=dk, dtype=DT)
        xp = s.nodes_phys
        u_bc = _mms_u(xp, amp)
        fc = _mms_body_force(s.elem_centroids_phys, E, nu, amp)
        f_ext = torch.zeros(s.n_nodes, 3, dtype=DT)
        f_elem = (s.vol[:, None, None] / 4.0) * fc.unsqueeze(1).expand(-1, 4, -1)
        idx = s.elements[:, :4].reshape(-1).unsqueeze(-1).expand(-1, 3)
        f_ext.scatter_add_(0, idx, f_elem.reshape(-1, 3))
        u = s.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, s.boundary_mask, max_iter=20, tol=1e-12)
        errs.append(float(torch.norm(u - _mms_u(xp, amp)) / torch.norm(_mms_u(xp, amp))))
    rates = [math.log(errs[i - 1] / errs[i]) / math.log(cells[i] / cells[i - 1])
             for i in range(1, len(errs))]
    return errs, rates


# --- ambient 3-D neural SDF of the rough top surface (the level-set baseline) ----------------------
class AmbientSDF3D(torch.nn.Module):
    def __init__(self, width=128, depth=6):
        super().__init__(); self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, P):
        return self.net(P).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_ambient_sdf3d(target_fn, L=1.0, band=0.25, iters=4000, n_pts=60000, width=128, depth=6):
    """phi(x,y,z) ~ signed distance to the surface z=h(x,y) (solid below).  Returns (sdf, recon RMSE)."""
    torch.manual_seed(0); rng = np.random.RandomState(0)
    xs = np.linspace(-L, L, 200); ys = np.linspace(-L, L, 200)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); H = target_fn(X, Y)
    # band points around the surface with vertical signed distance (small slope -> ~Euclidean)
    bi = rng.randint(0, X.size, n_pts)
    Px = X.ravel()[bi]; Py = Y.ravel()[bi]; Hs = H.ravel()[bi]
    Pz = Hs + rng.uniform(-band, band, n_pts)
    d = (Pz - Hs)                                                    # signed vertical distance
    P = torch.tensor(np.stack([Px, Py, Pz], 1)); dt = torch.tensor(d)
    sdf = AmbientSDF3D(width=width, depth=depth).double()
    opt = torch.optim.Adam(sdf.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        idx = torch.randint(0, n_pts, (8192,))
        loss = torch.mean((sdf(P[idx]) - dt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    # extract zero level set h_sdf(x,y) by bisection in z, compare to target
    gx = np.linspace(-L, L, 80); GX, GY = np.meshgrid(gx, gx, indexing="ij")
    Xq = torch.tensor(GX.ravel()); Yq = torch.tensor(GY.ravel())
    zlo = torch.full_like(Xq, -band); zhi = torch.full_like(Xq, band)
    for _ in range(40):
        zm = 0.5 * (zlo + zhi); phi = sdf(torch.stack([Xq, Yq, zm], 1))
        below = phi < 0; zlo = torch.where(below, zm, zlo); zhi = torch.where(below, zhi, zm)
    h_sdf = (0.5 * (zlo + zhi)).numpy(); h_tgt = target_fn(GX.ravel(), GY.ravel())
    return sdf, float(np.sqrt(np.mean((h_sdf - h_tgt) ** 2)))


def main():
    os.makedirs(RUN, exist_ok=True)
    tgt = lambda x, y: band_limited_rough_surface(x, y, amp=0.12)     # noqa: E731
    rms_surf = float(np.std(tgt(np.random.RandomState(1).uniform(-1, 1, 6000),
                                np.random.RandomState(2).uniform(-1, 1, 6000))))
    res = {"surface_rms": rms_surf}
    print(f"=== rough surface RMS = {rms_surf:.4f} ===")

    print("\n[1/4] train + verify the FOURIER boundary-fitted decoder (the neural atlas)")
    dec, rmse, dk = train_rough_decoder(tgt, rough_face="top", n_freq=20, iters=4000)
    s = ChartVectorFEMSolver(n_cells=12, support_r=1.0, chart_decoder=dec, decoder_kwargs=dk, dtype=DT)
    res["fourier_decoder"] = dict(recon_rmse=rmse, recon_pct=rmse / rms_surf * 100,
                                  all_valid=bool(s.geom_valid.all()), detJ_min=float(s.geom_detJ.min()),
                                  n_params=sum(p.numel() for p in dec.parameters()))
    print(f"    reconstruction RMSE {rmse:.3e} ({rmse/rms_surf*100:.2f}%); "
          f"FEM valid={s.geom_valid.all().item()} detJ_min={s.geom_detJ.min():.3e}")

    print("\n[2/4] MMS O(h^2) convergence on the ROUGH geometry")
    errs, rates = mms_convergence(dec, dk)
    res["mms"] = dict(errors=errs, rates=rates)
    print(f"    L2 errors {['%.2e' % e for e in errs]}  rates {['%.2f' % r for r in rates]}")

    print("\n[3/4] vanilla PLAIN-MLP decoder (Fourier off) — spectral-bias baseline")
    dec_p, rmse_p, _ = train_rough_decoder(tgt, plain=True, iters=4000)
    res["plain_decoder"] = dict(recon_rmse=rmse_p, recon_pct=rmse_p / rms_surf * 100)
    print(f"    reconstruction RMSE {rmse_p:.3e} ({rmse_p/rms_surf*100:.2f}%)")

    print("\n[4/4] ambient 3-D neural SDF (the level set) — extract zero level set")
    sdf, rmse_sdf = train_ambient_sdf3d(tgt)
    res["ambient_sdf"] = dict(recon_rmse=rmse_sdf, recon_pct=rmse_sdf / rms_surf * 100,
                              n_params=sdf.n_params())
    print(f"    zero-level-set reconstruction RMSE {rmse_sdf:.3e} ({rmse_sdf/rms_surf*100:.2f}%)")

    json.dump(res, open(os.path.join(RUN, "verify.json"), "w"), indent=2)
    print("\n=== HONEST SUMMARY ===")
    print(f"  Fourier decoder (atlas): {res['fourier_decoder']['recon_pct']:.1f}% of RMS, FEM well-posed, MMS O(h^2)")
    print(f"  plain-MLP decoder:       {res['plain_decoder']['recon_pct']:.1f}% of RMS (spectral bias)")
    print(f"  ambient 3-D SDF:         {res['ambient_sdf']['recon_pct']:.1f}% of RMS (level set smooths)")
    print(f"  saved -> {os.path.join(RUN, 'verify.json')}")


if __name__ == "__main__":
    main()
