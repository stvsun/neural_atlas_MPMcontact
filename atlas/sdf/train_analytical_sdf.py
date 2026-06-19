"""Train a neural SDF on an ANALYTICAL CV shape (plan A2) — the detection object for the
numerical CV suite.

Generalizes benchmarks/contact/koch_neural_ceiling.py from Koch to any
atlas/shapes/analytical.py shape, and unifies the 3-D and 2-D cases:

  * 3-D shape (sphere): supervise the native 3-D signed distance directly.
  * 2-D shape (disc / superformula): supervise a 3-D PRISM via extrusion_sdf — the in-plane
    signed distance lifted to |z|<=H and replicated across z-slices, so a 3-D SDFNet learns a
    true prism field (|grad phi|=1, n_z~0 at z=0).  evaluate_gap (the production contact
    detector) then consumes the trained net unchanged, embedding 2-D queries at z=0.

The net is atlas/sdf/train_sdf.py::SDFNet (in_dim=3, Eikonal-trained).  Per shape we record the
final Eikonal residual (the tau_g driver the manual §11.3 lists as "not yet measured") and the
held-out L0 metrics (gap RMSE / L, normal angle, max|n_z|).  Checkpoints cache to
runs/neural_sdf/<shape>_sdf.pt so the verification harness can load them.

Run:
  python3 atlas/sdf/train_analytical_sdf.py --shape sphere|disc|supershape [--quick]
  python3 atlas/sdf/train_analytical_sdf.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atlas.shapes.analytical import get_shape, AnalyticalShape      # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(_ROOT, "runs", "neural_sdf")


def extrusion_sdf(d2: np.ndarray, z: np.ndarray, H: float) -> np.ndarray:
    """Exact 3-D signed distance of a 2-D shape (in-plane signed distance ``d2``) extruded to a
    finite prism |z|<=H.  For |d2|<H at z=0 this is ``d2`` exactly; the caps teach dphi/dz=0."""
    dz = np.abs(z) - H
    outside = np.sqrt(np.maximum(d2, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    inside = np.minimum(np.maximum(d2, dz), 0.0)
    return outside + inside


def build_dataset(shape: AnalyticalShape, n_near: int, n_bulk: int, band: float,
                  seed: int, H: float, z_range: float, n_z_slices: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """(X3, g) training set as 3-D points + signed distance.  3-D shapes are sampled in space;
    2-D shapes are sampled in-plane then lifted across z-slices via extrusion_sdf."""
    if shape.dim == 3:
        near = shape.sample_near(n_near, band, seed)
        bulk = shape.sample_bulk(n_bulk, seed + 1)
        X3 = np.vstack([near, bulk])
        g = shape.sdf(X3)
        return X3, g
    # 2-D shape -> prism
    near = shape.sample_near(n_near, band, seed)
    bulk = shape.sample_bulk(n_bulk, seed + 1)
    xy = np.vstack([near, bulk])
    d2 = shape.sdf(xy)
    zs = np.linspace(-z_range, z_range, n_z_slices)            # symmetric, includes z=0 (odd)
    xy_rep = np.tile(xy, (n_z_slices, 1))
    d2_rep = np.tile(d2, n_z_slices)
    z_rep = np.repeat(zs, len(xy))
    X3 = np.column_stack([xy_rep, z_rep])
    g = extrusion_sdf(d2_rep, z_rep, H)
    return X3, g


def _eval_L0(shape: AnalyticalShape, model, n_eval: int, band: float, seed: int) -> Dict:
    """Held-out L0 metrics vs the analytical reference (gap RMSE/L, normal angle, max|n_z|)."""
    import torch
    from solvers.contact.gap import evaluate_gap
    dtype = torch.float64
    L = shape.body_size()
    if shape.dim == 3:
        xq = shape.sample_near(n_eval, band, seed + 999)
        g_ana = shape.sdf(xq)
        n_ana = shape.normal(xq) if hasattr(shape, "normal") else None
        X3 = xq
    else:
        xy = shape.sample_near(n_eval, band, seed + 999)
        g_ana = shape.sdf(xy)
        X3 = np.column_stack([xy, np.zeros(len(xy))])
        n_ana = shape.normal(xy) if hasattr(shape, "normal") else None
    g_nn, n_nn = evaluate_gap(torch.tensor(X3, dtype=dtype), model)
    g_nn = g_nn.numpy(); n_nn = n_nn.numpy()
    gap_rmse = float(np.sqrt(np.mean((g_nn - g_ana) ** 2)))
    out = {"gap_rmse": gap_rmse, "gap_rmse_rel": gap_rmse / L, "L": L}
    if shape.dim == 2:
        out["max_abs_nz"] = float(np.max(np.abs(n_nn[:, 2])))
    if n_ana is not None:
        nn = n_nn[:, :shape.dim]
        nn = nn / np.clip(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12, None)
        cos = np.clip(np.sum(nn * n_ana, axis=1), -1, 1)
        ang = np.degrees(np.arccos(cos))
        out["normal_angle_median_deg"] = float(np.median(ang))
        out["normal_angle_max_deg"] = float(np.max(ang))
    return out


def train_shape_sdf(shape_name: str, *, width: int = 128, depth: int = 5, epochs: int = 4000,
                    lr: float = 1e-3, n_near: int = 6000, n_bulk: int = 1500, band: float = 0.08,
                    H: float = 0.5, z_range: float = 0.75, n_z_slices: int = 5, w_eik: float = 0.05,
                    w_surf: float = 0.0, batch_eik: int = 2048, n_eval: int = 2000,
                    eval_band: float = 0.05,
                    seed: int = 0, save: bool = True, verbose: bool = True, **shape_kwargs
                    ) -> Tuple[object, Dict]:
    """Train a fixed SDFNet on the named analytical shape; return (model, metrics)."""
    import torch
    from atlas.sdf.train_sdf import SDFNet, set_seed
    set_seed(seed)
    dtype = torch.float64
    shape = get_shape(shape_name, **shape_kwargs)

    X3, g = build_dataset(shape, n_near, n_bulk, band, seed, H, z_range, n_z_slices)
    Xt = torch.tensor(X3, dtype=dtype)
    gt = torch.tensor(g, dtype=dtype)

    # ON-SURFACE supervision (target phi=0): the strongest constraint on the zero-level-set,
    # i.e. directly on the gap error near contact.  2-D shapes are pinned at z=0.
    surf_pts, _ = shape.sample_surface(max(n_near // 2, 1500), seed + 5)
    if shape.dim == 2:
        surf_pts = np.column_stack([surf_pts, np.zeros(len(surf_pts))])
    Xs = torch.tensor(surf_pts, dtype=dtype)

    model = SDFNet(width=width, depth=depth).to(dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-2)
    eik_rng = np.random.RandomState(seed + 99)
    half = shape.body_size() * 1.4
    zr = z_range if shape.dim == 2 else half
    # near-surface points matter most for contact; weight them above the far-field bulk so the
    # L1 data loss is not dominated by large-magnitude bulk targets (narrow-band emphasis).
    w_pt = torch.where(gt.abs() < band * 1.5,
                       torch.tensor(4.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    w_pt = w_pt / w_pt.mean()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss_surf = torch.mean(torch.abs(model(Xs)))          # phi = 0 on the boundary
        loss_data = torch.mean(w_pt * torch.abs(model(Xt) - gt))
        pe = np.empty((batch_eik, 3))
        pe[:, :2] = eik_rng.uniform(-half, half, size=(batch_eik, 2))
        pe[:, 2] = eik_rng.uniform(-zr, zr, size=batch_eik) if shape.dim == 2 else \
            eik_rng.uniform(-half, half, size=batch_eik)
        pet = torch.tensor(pe, dtype=dtype, requires_grad=True)
        ge = torch.autograd.grad(model(pet), pet, torch.ones(batch_eik, dtype=dtype),
                                 create_graph=True)[0]
        loss_eik = torch.mean((torch.linalg.norm(ge, dim=1) - 1.0) ** 2)
        loss = w_surf * loss_surf + loss_data + w_eik * loss_eik
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == 1):
            print(f"    {shape_name} epoch {epoch}/{epochs} surf={loss_surf.item():.3e} "
                  f"data={loss_data.item():.3e} eik={loss_eik.item():.3e}")

    m = _eval_L0(shape, model, n_eval, eval_band, seed)
    m.update({"shape": shape_name, "dim": shape.dim, "width": width, "depth": depth,
              "n_params": int(sum(p.numel() for p in model.parameters())),
              "final_eikonal": float(loss_eik.item()), "train_seconds": round(time.time() - t0, 2)})

    if save:
        os.makedirs(RUN_DIR, exist_ok=True)
        ckpt = {"model_state": model.state_dict(),
                "model_kwargs": {"width": width, "depth": depth},
                "shape": shape_name, "dim": shape.dim, "metrics": m}
        torch.save(ckpt, os.path.join(RUN_DIR, f"{shape_name}_sdf.pt"))
        with open(os.path.join(RUN_DIR, f"{shape_name}_sdf_meta.json"), "w") as f:
            json.dump(m, f, indent=2)
    if verbose:
        print(f"  [{shape_name}] gapRMSE/L={m['gap_rmse_rel']:.3e} "
              f"eik={m['final_eikonal']:.3e} "
              + (f"|nz|max={m.get('max_abs_nz', float('nan')):.2e} " if shape.dim == 2 else "")
              + (f"angle(med)={m.get('normal_angle_median_deg', float('nan')):.2f}deg "
                 if 'normal_angle_median_deg' in m else "")
              + f"({m['train_seconds']}s)")
    return model, m


def load_trained_sdf(shape_name: str):
    """Load a cached trained SDFNet for the harness; return the model or None if not trained."""
    path = os.path.join(RUN_DIR, f"{shape_name}_sdf.pt")
    if not os.path.isfile(path):
        return None
    import torch
    from atlas.sdf.train_sdf import SDFNet
    ckpt = torch.load(path, map_location="cpu")
    model = SDFNet(**ckpt["model_kwargs"]).to(dtype=torch.float64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train a neural SDF on an analytical CV shape")
    p.add_argument("--shape", default="sphere", choices=["sphere", "disc", "cylinder", "supershape"])
    p.add_argument("--all", action="store_true", help="train sphere, disc, supershape")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    kw = dict(width=args.width, depth=args.depth, epochs=args.epochs)
    if args.quick:
        kw.update(width=64, depth=4, epochs=1200, n_near=2000, n_bulk=1000, n_eval=1000)
    shapes = ["sphere", "disc", "supershape"] if args.all else [args.shape]
    for s in shapes:
        train_shape_sdf(s, **kw)


if __name__ == "__main__":
    main()
