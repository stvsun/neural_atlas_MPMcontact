"""Train a 2-D NEURAL RADIAL chart rho_theta : S^1 -> R+ on an analytical star-shaped body
(plan A4) — the transition-map detector that gives the ACCURATE CV-5 path.

Where a neural SDF degrades on the cusped, concave superformula (CV-5's documented SDF weakness,
manual §11.4), fitting the 1-D radius rho(psi) is easy and accurate: no medial axis, no ambient
spectral bias.  This trains solvers/contact/radial_chart_2d.py::NeuralRho2D to the analytical
superformula radius and records the fit error; the trained chart reproduces the analytical RADIAL
gap/normal (supershape.radial_gap) to ~1e-3, vs the neural SDF's ~8e-3 gap + degraded normal.

Run:  python3 atlas/charts/train_radial_chart.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solvers.contact import supershape as ss                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(_ROOT, "runs", "neural_radial_chart")

# the canonical CV-5 superformula (matches tests/test_neural_chart_verification.py)
CV5_PARAMS = dict(m=6, n1=0.7, n2=0.7, n3=0.7, a=1.0, b=1.0, scale=1.0)


def train_superformula_radial(width: int = 96, depth: int = 3, n_freq: int = 16, epochs: int = 4000,
                              lr: float = 2e-3, n_train: int = 6000, n_eval: int = 4000, seed: int = 0,
                              params: dict = None, name: str = "supershape",
                              save: bool = True, verbose: bool = True) -> Tuple[object, Dict]:
    """Fit NeuralRho2D to the analytical superformula radius rho(psi); return (model, metrics)."""
    import torch
    from atlas.sdf.train_sdf import set_seed
    from solvers.contact.radial_chart_2d import NeuralRho2D
    set_seed(seed)
    dtype = torch.float64
    pr = params or CV5_PARAMS
    sp = ss.SuperParams(**pr)

    # 1-D regression targets rho(psi); psi sampled uniformly (the (cos,sin) input is periodic).
    rng = np.random.RandomState(seed)
    psi_tr = rng.uniform(0, 2 * np.pi, n_train)
    rho_tr = ss.radius(psi_tr, sp)
    base = float(np.mean(rho_tr))
    Pt = torch.tensor(psi_tr, dtype=dtype)
    Rt = torch.tensor(rho_tr, dtype=dtype)

    model = NeuralRho2D(width=width, depth=depth, base=base, n_freq=n_freq).to(dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-2)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = torch.mean(torch.abs(model(Pt) - Rt))
        loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == 1):
            print(f"    superformula-radial epoch {epoch}/{epochs}  L1(rho)={loss.item():.3e}")

    # ---- L0 evaluation: radius fit + radial gap/normal vs the ANALYTICAL radial chart ----
    L = float(ss.radius(np.linspace(0, 2 * np.pi, 1024), sp).max())
    psi_e = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    rho_e_ana = ss.radius(psi_e, sp)
    rho_e_nn = model(torch.tensor(psi_e, dtype=dtype)).detach().numpy()
    radius_rmse = float(np.sqrt(np.mean((rho_e_nn - rho_e_ana) ** 2)))

    # near-boundary gap/normal vs supershape.radial_gap (the like-for-like radial reference)
    from solvers.contact.radial_chart_2d import evaluate_radial_gap_2d
    rng2 = np.random.RandomState(seed + 7)
    xy = ss.boundary(rng2.uniform(0, 2 * np.pi, n_eval), np.zeros(2), 0.0, sp) + rng2.randn(n_eval, 2) * 0.1
    g_ana, grad_ana = ss.radial_gap(xy, np.zeros(2), 0.0, sp)
    n_ana = grad_ana / np.clip(np.linalg.norm(grad_ana, axis=1, keepdims=True), 1e-12, None)
    g_nn, n_nn = evaluate_radial_gap_2d(torch.tensor(xy, dtype=dtype), model, center=(0.0, 0.0), alpha=0.0)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    gap_rmse = float(np.sqrt(np.mean((g_nn - g_ana) ** 2)))
    ang = np.degrees(np.arccos(np.clip(np.sum(n_nn * n_ana, axis=1), -1, 1)))

    m = {
        "shape": name, "object": "neural_radial_chart", "params": pr,
        "width": width, "depth": depth, "n_params": int(sum(p.numel() for p in model.parameters())),
        "L": L, "radius_rmse": radius_rmse, "radius_rmse_rel": radius_rmse / L,
        "gap_rmse": gap_rmse, "gap_rmse_rel": gap_rmse / L,
        "normal_angle_median_deg": float(np.median(ang)), "normal_angle_max_deg": float(np.max(ang)),
        "train_seconds": round(time.time() - t0, 2),
    }
    if save:
        os.makedirs(RUN_DIR, exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "model_kwargs": {"width": width, "depth": depth, "base": base, "n_freq": n_freq},
                    "params": pr, "metrics": m},
                   os.path.join(RUN_DIR, f"{name}_radial.pt"))
        with open(os.path.join(RUN_DIR, f"{name}_radial_meta.json"), "w") as f:
            json.dump(m, f, indent=2)
    if verbose:
        print(f"  [supershape radial chart] radiusRMSE/L={m['radius_rmse_rel']:.3e}  "
              f"gapRMSE/L={m['gap_rmse_rel']:.3e}  angle(med)={m['normal_angle_median_deg']:.3f}deg "
              f"(max {m['normal_angle_max_deg']:.2f})  ({m['train_seconds']}s)")
    return model, m


def load_trained_radial_chart(shape: str = "supershape"):
    """Load a cached trained NeuralRho2D for the harness; return the model or None."""
    path = os.path.join(RUN_DIR, f"{shape}_radial.pt")
    if not os.path.isfile(path):
        return None
    import torch
    from solvers.contact.radial_chart_2d import NeuralRho2D
    ckpt = torch.load(path, map_location="cpu")
    model = NeuralRho2D(**ckpt["model_kwargs"]).to(dtype=torch.float64)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model


def parse_args():
    ap = argparse.ArgumentParser(description="Train a 2-D neural radial chart on the superformula")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    kw = dict(width=args.width, depth=args.depth, epochs=args.epochs)
    if args.quick:
        kw.update(width=48, depth=3, epochs=1500, n_train=2000, n_eval=2000)
    train_superformula_radial(**kw)


if __name__ == "__main__":
    main()
