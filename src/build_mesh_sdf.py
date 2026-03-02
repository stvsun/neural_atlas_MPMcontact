#!/usr/bin/env python3
"""
Build an exact signed distance function for the Stanford Bunny from its PLY mesh,
using three complementary libraries for robustness:

  1. open3d.t.geometry.RaycastingScene  — primary SDF oracle (embree BVH, handles
                                          non-watertight meshes via ray casting)
  2. mesh_to_sdf.sample_sdf_near_surface — near-surface point sampling (gives
                                           training points concentrated near the
                                           zero level-set where sign matters most)
  3. pysdf.SDF                           — independent cross-validation of sign
                                           quality (aborts if disagreement > 10%)

The script trains a standard MLP (width=128, depth=6, tanh activations) by pure
MSE regression to the exact SDF values — **no sign loss, no PCA normals**.
Because the teacher SDF has geometrically correct sign, the MLP inherits < 5%
sign error (vs 31–46% from the neural sign-anchor approach).

Output checkpoint is drop-in compatible with the existing pipeline
(build_rabbit_atlas_volumetric.py, run_poisson_rabbit_atlas_schwarz.py).

Sign conventions:
  open3d   : negative = inside, positive = outside  ← our convention
  mesh_to_sdf: negative = inside, positive = outside  ← same
  pysdf    : positive = inside, negative = outside  ← OPPOSITE; negate before use

Usage
-----
python src/build_mesh_sdf.py \\
    --ply-file  runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply \\
    --out-dir   runs/bunny_sdf_mesh \\
    --n-near    150000 \\
    --n-uniform 150000 \\
    --epochs    5000 \\
    --sign-gate 0.05
"""

import argparse
import json
import math
import os
import random
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# MLP — identical architecture to train_sdf_rabbit.SDFNet (ensures checkpoint
#        compatibility with load_sdf_net() in build_rabbit_atlas_volumetric.py)
# ──────────────────────────────────────────────────────────────────────────────

class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(width, out_dim)
        for layer in self.hidden:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)


class SDFNet(torch.nn.Module):
    """Drop-in replacement for train_sdf_rabbit.SDFNet."""

    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# open3d RaycastingScene — primary SDF oracle
# ──────────────────────────────────────────────────────────────────────────────

def build_open3d_scene(ply_path: str):
    """Load PLY into an open3d RaycastingScene.  Returns (scene, o3d_mesh)."""
    import open3d as o3d
    mesh_o3d = o3d.io.read_triangle_mesh(ply_path)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    print(
        f"  open3d mesh: {len(mesh_o3d.vertices)} verts, "
        f"{len(mesh_o3d.triangles)} triangles, "
        f"watertight={mesh_o3d.is_watertight()}"
    )
    return scene, mesh_o3d


def query_open3d(scene, pts_phys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Query the open3d scene for signed distance and occupancy.

    Parameters
    ----------
    pts_phys : (N, 3) float32 array in physical (metres) coordinates.

    Returns
    -------
    sdf  : (N,) float32 — signed distance; **negative = inside**
    occ  : (N,) float32 — occupancy; 1 = inside, 0 = outside
    """
    import open3d as o3d
    pts_f32 = pts_phys.astype(np.float32)
    t = o3d.core.Tensor(pts_f32, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(t).numpy().astype(np.float64)
    occ = scene.compute_occupancy(t).numpy().astype(np.float64)
    return sdf, occ


# ──────────────────────────────────────────────────────────────────────────────
# pysdf — independent cross-validation (sign convention: positive = inside)
# ──────────────────────────────────────────────────────────────────────────────

def build_pysdf(mesh_trimesh) -> object:
    """Build a pysdf.SDF evaluator from a trimesh mesh."""
    import pysdf
    verts = mesh_trimesh.vertices.astype(np.float32)
    tris = mesh_trimesh.faces.astype(np.uint32)
    return pysdf.SDF(verts, tris)


def query_pysdf_negated(pysdf_f, pts_phys: np.ndarray) -> np.ndarray:
    """Query pysdf and NEGATE to match our convention (negative = inside).

    pysdf.SDF returns positive values for interior points.
    """
    return -pysdf_f(pts_phys.astype(np.float32)).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Training sample generation
# ──────────────────────────────────────────────────────────────────────────────

def get_near_surface_samples(
    mesh_trimesh,
    open3d_scene,
    our_center: np.ndarray,
    our_scale: float,
    n_near: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample near-surface points using mesh_to_sdf; query exact SDF via open3d.

    Uses mesh_to_sdf only for the SPATIAL DISTRIBUTION of points (concentrated
    near the zero level-set).  Signs/distances come from the more reliable
    open3d embree BVH.

    Returns
    -------
    pts_norm  : (M, 3) in our normalised space
    sdf_norm  : (M,)  exact SDF in our normalised space (negative = inside)
    """
    try:
        import mesh_to_sdf

        print(f"  Sampling {n_near} near-surface points via mesh_to_sdf…")
        # mesh_to_sdf normalises the mesh to a unit sphere internally:
        #   ms_center = mean(vertices)
        #   ms_scale  = max distance from ms_center to any vertex
        ms_center = np.mean(mesh_trimesh.vertices, axis=0)
        ms_scale = float(
            np.max(np.linalg.norm(mesh_trimesh.vertices - ms_center, axis=1))
        )

        pts_unitsp, _ = mesh_to_sdf.sample_sdf_near_surface(
            mesh_trimesh,
            number_of_points=n_near,
            surface_point_method="scan",
            sign_method="normal",
            scan_count=100,
            scan_resolution=400,
        )
        # Convert mesh_to_sdf unit-sphere coords → physical space
        pts_phys = pts_unitsp * ms_scale + ms_center

        # Exact SDF from open3d (ignore mesh_to_sdf's sign estimate)
        sdf_phys, _ = query_open3d(open3d_scene, pts_phys.astype(np.float32))

        # Convert to our normalised space
        pts_norm = (pts_phys - our_center) / our_scale
        sdf_norm = sdf_phys / our_scale

        print(
            f"  Near-surface: {len(pts_norm)} pts, "
            f"sdf ∈ [{sdf_norm.min():.4f}, {sdf_norm.max():.4f}]"
        )
        return pts_norm, sdf_norm

    except Exception as exc:
        print(f"  mesh_to_sdf failed ({exc}); falling back to empty near-surface set.")
        return np.empty((0, 3)), np.empty((0,))


def get_uniform_samples(
    open3d_scene,
    our_center: np.ndarray,
    our_scale: float,
    n_uniform: int,
    rng: np.random.Generator,
    bbox_half: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample uniformly in normalised space; classify with open3d.

    Returns
    -------
    pts_norm  : (N, 3)
    sdf_norm  : (N,)  negative = inside
    occ       : (N,)  1 = inside, 0 = outside
    """
    pts_norm = rng.uniform(-bbox_half, bbox_half, (n_uniform, 3))
    pts_phys = pts_norm * our_scale + our_center
    sdf_phys, occ = query_open3d(open3d_scene, pts_phys.astype(np.float32))
    sdf_norm = sdf_phys / our_scale
    return pts_norm, sdf_norm, occ


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate_sign(
    pysdf_f,
    open3d_scene,
    our_center: np.ndarray,
    our_scale: float,
    rng: np.random.Generator,
    n_validate: int = 20_000,
) -> float:
    """Check pysdf vs open3d sign agreement on interior points.

    Returns fraction where both agree (sign = negative = inside).
    """
    # Sample candidate interior points
    pts_norm = rng.uniform(-0.55, 0.55, (n_validate * 4, 3))
    pts_phys = pts_norm * our_scale + our_center
    _, occ = query_open3d(open3d_scene, pts_phys.astype(np.float32))
    interior_mask = occ.astype(bool)
    interior_phys = pts_phys[interior_mask][:n_validate]
    if len(interior_phys) < 100:
        print("  WARNING: fewer than 100 interior points found for cross-validation")
        return 1.0

    sdf_pysdf = query_pysdf_negated(pysdf_f, interior_phys)
    agreement = float(np.mean(sdf_pysdf < 0.0))
    return agreement


# ──────────────────────────────────────────────────────────────────────────────
# MLP training
# ──────────────────────────────────────────────────────────────────────────────

def train_mlp(
    pts_norm: np.ndarray,
    sdf_norm: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[SDFNet, dict]:
    """Train an SDFNet by MSE regression to exact SDF + optional Eikonal.

    Parameters
    ----------
    pts_norm : (N, 3)  training points in normalised space
    sdf_norm : (N,)    exact SDF values (negative inside)

    Returns
    -------
    model, history dict
    """
    N = len(pts_norm)
    print(f"\nTraining MLP on {N} samples ({args.epochs} epochs)…")

    model = SDFNet(width=args.width, depth=args.depth).to(device=device, dtype=dtype)

    # Cosine learning-rate schedule: lr_0 → lr_min
    lr_min = args.lr * 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr_min
    )

    pts_t = torch.tensor(pts_norm, dtype=dtype, device=device)
    sdf_t = torch.tensor(sdf_norm, dtype=dtype, device=device)

    history = {"mse": [], "eikonal": [], "total": []}
    batch_size = min(args.batch_size, N)
    rng = np.random.default_rng(args.seed + 1)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # ── MSE regression batch ──────────────────────────────────────────────
        idx = rng.choice(N, batch_size, replace=False)
        p_t = pts_t[idx]
        s_t = sdf_t[idx]
        phi = model(p_t)
        mse_loss = torch.mean((phi - s_t) ** 2)

        # ── Eikonal regulariser (optional) ───────────────────────────────────
        if args.w_eikonal > 0:
            idx_eik = rng.choice(N, batch_size // 2, replace=False)
            p_eik = pts_t[idx_eik].detach().requires_grad_(True)
            phi_eik = model(p_eik)
            grad_phi = torch.autograd.grad(
                phi_eik,
                p_eik,
                grad_outputs=torch.ones_like(phi_eik),
                create_graph=True,
            )[0]
            eik_loss = torch.mean((torch.norm(grad_phi, dim=1) - 1.0) ** 2)
        else:
            eik_loss = torch.zeros(1, dtype=dtype, device=device)

        total_loss = mse_loss + args.w_eikonal * eik_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            history["mse"].append(float(mse_loss))
            history["eikonal"].append(float(eik_loss))
            history["total"].append(float(total_loss))
            elapsed = time.time() - t0
            print(
                f"  ep {epoch:5d}/{args.epochs}  "
                f"MSE={float(mse_loss):.3e}  "
                f"Eik={float(eik_loss):.3e}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"t={elapsed:.0f}s"
            )

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Sign-error evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sign_error(
    model: SDFNet,
    open3d_scene,
    our_center: np.ndarray,
    our_scale: float,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
    n_test: int = 20_000,
) -> float:
    """Measure fraction of test points where MLP sign disagrees with open3d.

    Tests n_test interior points (open3d occ=1 → expected phi < 0) and
    n_test exterior points (occ=0 → expected phi > 0).
    """
    model.eval()

    # Collect interior + exterior points
    pts_norm_cand = rng.uniform(-0.55, 0.55, (n_test * 8, 3))
    pts_phys_cand = pts_norm_cand * our_scale + our_center
    _, occ = query_open3d(open3d_scene, pts_phys_cand.astype(np.float32))
    interior_pts = pts_norm_cand[occ.astype(bool)][:n_test]
    exterior_pts = pts_norm_cand[~occ.astype(bool)][:n_test]

    errors = []
    for pts, expected_sign in [(interior_pts, "negative"), (exterior_pts, "positive")]:
        if len(pts) == 0:
            continue
        x_t = torch.tensor(pts, dtype=dtype, device=device)
        phi = model(x_t).cpu().numpy()
        if expected_sign == "negative":
            wrong = np.mean(phi >= 0.0)
        else:
            wrong = np.mean(phi <= 0.0)
        errors.append(float(wrong))

    return float(np.mean(errors)) if errors else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# SDF slice plot
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_slice_plot(
    model: SDFNet,
    out_path: str,
    grid_n: int = 200,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> None:
    model.eval()
    lin = np.linspace(-0.55, 0.55, grid_n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    slice_specs = [
        ("XY (z=0)", 0, 1, 0.0),
        ("XZ (y=0.11)", 0, 2, 0.11 / 0.156),  # ~approx bunny waist in norm coords
        ("YZ (x=0)", 1, 2, 0.0),
    ]
    for ax, (title, ai, bi, cv) in zip(axes, slice_specs):
        G = np.zeros((grid_n, grid_n, 3))
        G[:, :, ai] = lin[None, :]
        G[:, :, bi] = lin[:, None]
        G[:, :, 3 - ai - bi] = cv
        pts = G.reshape(-1, 3)
        x_t = torch.tensor(pts, dtype=dtype, device=device)
        phi = model(x_t).cpu().numpy().reshape(grid_n, grid_n)
        cf = ax.contourf(lin, lin, phi.T, levels=40, cmap="RdBu_r")
        ax.contour(lin, lin, phi.T, levels=[0.0], colors="k", linewidths=1.5)
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, shrink=0.85)
    fig.suptitle("Exact Mesh SDF — MLP (normalised coordinates)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Slice plot saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print("BUILD EXACT MESH SDF — Stanford Bunny")
    print("=" * 60)

    # ── Load trimesh (for mesh_to_sdf and pysdf) ──────────────────────────────
    print(f"\nLoading PLY: {args.ply_file}")
    mesh_trimesh = trimesh.load(args.ply_file, force="mesh")
    print(
        f"  trimesh: {len(mesh_trimesh.vertices)} verts, "
        f"{len(mesh_trimesh.faces)} faces, "
        f"watertight={mesh_trimesh.is_watertight}"
    )

    # Optional: attempt to fill small holes for better watertightness
    if not mesh_trimesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh_trimesh)
            print(
                f"  After fill_holes: watertight={mesh_trimesh.is_watertight}"
            )
        except Exception as e:
            print(f"  fill_holes failed ({e}); continuing with original mesh")

    # ── Compute normalisation (same formula as train_sdf_rabbit.py) ───────────
    verts = np.array(mesh_trimesh.vertices, dtype=np.float64)
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    our_center = 0.5 * (mins + maxs)
    our_scale = float(np.max(maxs - mins))
    our_scale = max(our_scale, 1e-6)
    print(f"\nNormalisation:")
    print(f"  center = {our_center.tolist()}")
    print(f"  scale  = {our_scale:.6f}")
    print(f"  PLY normalised: x∈[{(mins[0]-our_center[0])/our_scale:.3f}, "
          f"{(maxs[0]-our_center[0])/our_scale:.3f}]")

    # ── Build open3d RaycastingScene (primary oracle) ─────────────────────────
    print("\nBuilding open3d RaycastingScene (primary SDF oracle)…")
    open3d_scene, _ = build_open3d_scene(args.ply_file)

    # Quick sanity check: bunny centre should be inside
    centroid_phys = our_center.reshape(1, 3).astype(np.float32)
    sdf_c, occ_c = query_open3d(open3d_scene, centroid_phys)
    print(
        f"  Sanity check at bounding-box centre: "
        f"sdf={sdf_c[0]:.4f}  occ={occ_c[0]:.0f}  "
        f"({'inside ✓' if occ_c[0] > 0.5 else 'OUTSIDE — check mesh!'})"
    )

    # ── Build pysdf (cross-validator) ─────────────────────────────────────────
    print("\nBuilding pysdf evaluator (cross-validator)…")
    pysdf_f = build_pysdf(mesh_trimesh)

    # ── Cross-validate open3d vs pysdf BEFORE generating training data ────────
    print(f"\nCross-validating open3d vs pysdf on 20k interior points…")
    rng = np.random.default_rng(args.seed)
    agreement = cross_validate_sign(
        pysdf_f, open3d_scene, our_center, our_scale, rng, n_validate=20_000
    )
    print(f"  pysdf/open3d agreement = {agreement:.4f}")
    if agreement < 0.90:
        raise RuntimeError(
            f"pysdf and open3d disagree on {1-agreement:.1%} of interior points "
            f"(threshold 10%). Inspect the mesh for severe holes. "
            f"agreement={agreement:.4f}"
        )
    print(f"  {'OK ✓' if agreement > 0.95 else 'WARN: moderate disagreement'}")

    # ── Generate training data ────────────────────────────────────────────────
    print(f"\nGenerating training data ({args.n_near} near-surface + "
          f"{args.n_uniform} uniform)…")

    pts_near, sdf_near = get_near_surface_samples(
        mesh_trimesh, open3d_scene, our_center, our_scale, args.n_near
    )
    pts_uni, sdf_uni, occ_uni = get_uniform_samples(
        open3d_scene, our_center, our_scale, args.n_uniform, rng
    )

    # Combine
    if len(pts_near) > 0:
        pts_all = np.concatenate([pts_near, pts_uni], axis=0).astype(np.float64)
        sdf_all = np.concatenate([sdf_near, sdf_uni], axis=0).astype(np.float64)
    else:
        pts_all = pts_uni.astype(np.float64)
        sdf_all = sdf_uni.astype(np.float64)

    n_interior = int(np.sum(occ_uni > 0.5))
    n_exterior = args.n_uniform - n_interior
    print(
        f"\nTotal training samples: {len(pts_all)}"
        f"  (uniform interior={n_interior}, exterior={n_exterior})"
    )
    print(
        f"  SDF value range: [{sdf_all.min():.4f}, {sdf_all.max():.4f}]"
        f"  (negative inside, positive outside)"
    )

    # ── Train MLP ─────────────────────────────────────────────────────────────
    model, history = train_mlp(pts_all, sdf_all, args, device, dtype)

    # ── Evaluate sign quality ─────────────────────────────────────────────────
    print("\nEvaluating MLP sign quality on 20k held-out test points…")
    rng2 = np.random.default_rng(args.seed + 99)
    sign_error = evaluate_sign_error(
        model, open3d_scene, our_center, our_scale, rng2, device, dtype, n_test=20_000
    )
    print(f"  sign_error = {sign_error:.4f}  "
          f"({'PASS ✓' if sign_error < args.sign_gate else 'FAIL ✗'})")
    print(f"  (reference: global SDF v3 = 0.311, chartwise = 0.460)")

    if sign_error >= args.sign_gate:
        # Save diagnostics and abort
        diag_path = os.path.join(args.out_dir, "sign_error_diagnostic.json")
        with open(diag_path, "w") as f:
            json.dump(
                {
                    "sign_error": float(sign_error),
                    "pysdf_agreement": float(agreement),
                    "sign_gate": args.sign_gate,
                    "status": "FAIL",
                },
                f,
                indent=2,
            )
        raise RuntimeError(
            f"MLP sign_error={sign_error:.4f} exceeds gate threshold "
            f"{args.sign_gate:.2f}. "
            f"Diagnostic saved to {diag_path}. "
            f"Try increasing --n-near / --n-uniform / --epochs."
        )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.out_dir, "rabbit_sdf_mesh.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {"width": args.width, "depth": args.depth},
            "center": our_center.tolist(),
            "scale": float(our_scale),
            "source": f"mesh_exact:{args.ply_file}",
            "sign_error_estimate": float(sign_error),
            "pysdf_open3d_agreement": float(agreement),
            "n_training_samples": len(pts_all),
            "history": history,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── Save slice plot ───────────────────────────────────────────────────────
    slice_path = os.path.join(args.out_dir, "sdf_slices.png")
    save_slice_plot(model, slice_path, device=device, dtype=dtype)

    # ── Save meta JSON ────────────────────────────────────────────────────────
    meta = {
        "checkpoint": ckpt_path,
        "center": our_center.tolist(),
        "scale": float(our_scale),
        "sign_error_estimate": float(sign_error),
        "pysdf_open3d_agreement": float(agreement),
        "n_training_samples": len(pts_all),
        "n_near": len(pts_near),
        "n_uniform": args.n_uniform,
        "status": "PASS",
        "args": vars(args),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXACT MESH SDF BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  sign_error_estimate     = {sign_error:.4f}  (< {args.sign_gate:.2f}  PASS)")
    print(f"  pysdf_open3d_agreement  = {agreement:.4f}")
    print(f"  checkpoint              = {ckpt_path}")
    print(f"  n_training_samples      = {len(pts_all)}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an exact mesh SDF for the Stanford Bunny via open3d + "
            "mesh_to_sdf + pysdf, then train an MLP drop-in replacement."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ply-file",
        required=True,
        help="Path to the Stanford Bunny PLY file",
    )
    p.add_argument(
        "--out-dir",
        default="runs/bunny_sdf_mesh",
        help="Output directory for checkpoint and plots",
    )
    # Training data
    p.add_argument(
        "--n-near",
        type=int,
        default=150_000,
        help="Number of near-surface samples from mesh_to_sdf",
    )
    p.add_argument(
        "--n-uniform",
        type=int,
        default=150_000,
        help="Number of uniform-random samples classified by open3d",
    )
    # MLP architecture
    p.add_argument("--width", type=int, default=128, help="MLP hidden width")
    p.add_argument("--depth", type=int, default=6, help="MLP depth (hidden layers+1)")
    # Training
    p.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for MSE regression",
    )
    p.add_argument(
        "--w-eikonal",
        type=float,
        default=0.1,
        help="Weight for optional Eikonal regulariser (0 to disable)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print loss every N epochs",
    )
    # Quality gate
    p.add_argument(
        "--sign-gate",
        type=float,
        default=0.05,
        help="Abort if MLP sign_error exceeds this fraction (default 5%%)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
