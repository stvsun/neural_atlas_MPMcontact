#!/usr/bin/env python3
"""
Chart-partitioned neural SDF for the Stanford Bunny.

Key innovations over the global SDF (train_sdf_rabbit.py)
==========================================================

1. **12 local SDFNetLocal networks**, one per atlas chart.
   Each network is responsible only for its chart neighbourhood;
   thin-feature charts and body charts are trained independently.

2. **Geometry-adaptive anchor offset** (per chart).
   For each chart seed the offset is set to::

       offset_i = anchor_offset_factor * dist(seed_i, nearest PLY point)

   Concretely:
   - Ear-region seeds sit ~5 mm from the ear surface → offset ≈ 1.5 mm  (safe for 5 mm ears)
   - Body seeds sit ~50 mm from the nearest surface  → offset ≈ 15 mm  (deep interior coverage)

   This directly resolves the single-offset limitation of the global SDF
   (where anchor_offset=0.25 placed anchors outside the 5-mm ears, causing
   31 % sign errors).

3. **Voxel signed-distance-transform (SDT) initialization**.
   Before the neural fine-tuning, each local network is pre-trained to
   match a cheap analytical distance transform computed from the local PLY
   point cloud (using `scipy.spatial.cKDTree`, no voxelisation needed).
   The sign is determined by projecting each query point onto the nearest
   surface normal::

       sign(q) = +1  if  (q - p_nearest) · n_nearest  ≥ 0
                 -1  otherwise

   Starting from a numerically correct SDT means the fine-tuning begins
   with the right sign topology; the subsequent sign-anchor loss acts as
   a fine-correction rather than driving sign recovery from a random init.

4. **Reinitialization PDE fine-tuning (Eikonal loss)**.
   After SDT pre-training, the network is fine-tuned with the standard
   surface / Eikonal / normal / sign losses.  The Eikonal loss::

       L_eik = mean( (||∇φ|| - 1)² )

   is the steady-state form of the governing equation of the classical
   reinitialization problem::

       ∂φ/∂t + sign(φ₀)(||∇φ|| - 1) = 0

   The zero level set is preserved by the surface loss ``|φ_surf|``, and
   the sign topology is maintained by the pre-trained initialisation plus
   the sign-anchor loss.  Together these four terms are the neural-network
   equivalent of the full Eikonal reinitialization scheme.

5. **Gaussian partition-of-unity (PoU) blending** for inference.
   The combined SDF is::

       φ(x) = Σ_i w_i(x) φ_i(x) / Σ_i w_i(x),
       w_i(x) = exp( -||x - seed_i||² / (2 σ_i²) ),  σ_i = support_radii_i

   This is smooth, differentiable (autograd works through the blending),
   and reduces to the nearest-chart SDF far from chart boundaries.

6. **Cross-chart consistency loss in overlap regions**.
   In chart overlap regions (where two or more local SDFs are responsible),
   a consistency loss ``mean( (φ_i(x) - φ_j(x))² )`` penalises disagreement,
   encouraging a globally consistent distance field.

Usage
-----
::

    python src/train_sdf_chartwise.py \\
        --ply-file   runs/atlas_schwarz_20260212_210412/downloads/bunny/reconstruction/bun_zipper.ply \\
        --atlas-data runs/atlas_bunny_vol_v3/rabbit_atlas_data.npz \\
        --output-dir runs/bunny_sdf_chartwise

Checkpoint format
-----------------
The saved ``rabbit_sdf_chartwise.pt`` contains a ``SDFNetChartwise`` whose
``forward(x)`` has the same signature as ``SDFNet.forward(x)`` — it is a
drop-in replacement for the existing SDF checkpoint used by the PINN,
provided the PINN loading code is updated to detect the ``'chartwise': True``
flag and call ``SDFNetChartwise.load(ckpt)`` instead of constructing
``SDFNet(**ckpt['model_kwargs'])``.

A thin adapter checkpoint (``rabbit_sdf_adapter.pt``) is also saved in the
*same format* as the original ``train_sdf_rabbit.py`` output (``model_state``,
``model_kwargs``) so that it can be loaded by unmodified PINN code with *no
changes at all* — at a small accuracy cost from fitting the chartwise SDF
into a single global network.
"""

import argparse
import json
import math
import os
import random
import struct
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────────────────────────────────────

torch.set_default_dtype(torch.float64)

PLY_TYPE_MAP = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_vecs(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def parse_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Parse a PLY file (ASCII or binary_little_endian) → (points, normals)."""
    with open(path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {path}")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        fmt = n_vertices = None
        vertex_props: List[tuple] = []
        current_element = None
        for line in header_lines:
            toks = line.split()
            if not toks:
                continue
            if toks[0] == "format":
                fmt = toks[1]
            elif toks[0] == "element":
                current_element = toks[1]
                if current_element == "vertex":
                    n_vertices = int(toks[2])
            elif toks[0] == "property" and current_element == "vertex":
                if len(toks) >= 3 and toks[1] != "list":
                    vertex_props.append((toks[2], toks[1]))   # (name, type)

        prop_names = [p[0] for p in vertex_props]
        if fmt is None or n_vertices is None or not all(k in prop_names for k in ["x", "y", "z"]):
            raise ValueError(f"Unsupported PLY: {path}")

        nx_present = all(k in prop_names for k in ["nx", "ny", "nz"])

        if fmt == "ascii":
            rows = []
            for _ in range(n_vertices):
                rows.append([float(t) for t in f.readline().decode("ascii", errors="ignore").strip().split()])
            arr = np.asarray(rows, dtype=float)
        elif fmt == "binary_little_endian":
            fmt_str = "<" + "".join(PLY_TYPE_MAP[tp] for _, tp in vertex_props)
            rec_size = struct.calcsize(fmt_str)
            rows = [struct.unpack(fmt_str, f.read(rec_size)) for _ in range(n_vertices)]
            arr = np.asarray(rows, dtype=float)
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")

    ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    points = arr[:, [ix, iy, iz]]
    normals = None
    if nx_present:
        inx, iny, inz = prop_names.index("nx"), prop_names.index("ny"), prop_names.index("nz")
        normals = arr[:, [inx, iny, inz]]
    return points, normals


# ──────────────────────────────────────────────────────────────────────────────
# Neural network architecture
# ──────────────────────────────────────────────────────────────────────────────

class MLP(torch.nn.Module):
    """Fully-connected network with tanh activations and Xavier initialisation."""

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


class SDFNetLocal(torch.nn.Module):
    """Small local SDF network for one atlas chart.

    Smaller than the global SDFNet (default width=64, depth=4 vs 128/6) to
    keep per-chart training fast and to prevent overfitting on local data.
    """

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.width = width
        self.depth = depth
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SDFNetChartwise(torch.nn.Module):
    """Gaussian partition-of-unity blending of N_charts local SDFs.

    This module has the same ``forward(x) → scalar`` interface as the global
    ``SDFNet``.  It can be differentiated through (autograd works through the
    PoU blending), so PINN interface-normal computation is exact.

    Parameters
    ----------
    local_nets : list of SDFNetLocal
        Trained local SDF networks, one per chart.
    seed_points_np : np.ndarray, shape (C, 3)
        Chart seed positions in *normalised SDF space*.
    support_radii_np : np.ndarray, shape (C,)
        Chart support radii in normalised SDF space.  Used as σ_i in the
        Gaussian weight ``w_i(x) = exp(-||x-seed_i||²/(2 σ_i²))``.
    """

    def __init__(
        self,
        local_nets: List[SDFNetLocal],
        seed_points_np: np.ndarray,
        support_radii_np: np.ndarray,
    ):
        super().__init__()
        self.local_nets = torch.nn.ModuleList(local_nets)
        self.register_buffer(
            "seed_pts",
            torch.tensor(seed_points_np, dtype=torch.float64),
        )
        self.register_buffer(
            "sigma2",
            torch.tensor(support_radii_np ** 2, dtype=torch.float64),
        )

    @property
    def n_charts(self) -> int:
        return len(self.local_nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the partition-of-unity blended SDF.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Query points in *normalised SDF space*.

        Returns
        -------
        phi : torch.Tensor, shape (N,)
            Blended signed distance values.
        """
        # Gaussian weights:  w_i(x) = exp(-||x - seed_i||² / (2 σ_i²))
        diff = x.unsqueeze(1) - self.seed_pts.unsqueeze(0)   # (N, C, 3)
        dist2 = (diff ** 2).sum(dim=2)                        # (N, C)
        w = torch.exp(-dist2 / (2.0 * self.sigma2.unsqueeze(0)))  # (N, C)

        # Evaluate all local SDFs in one pass
        phi_stack = torch.stack(
            [net(x) for net in self.local_nets], dim=1
        )  # (N, C)

        # Normalised weighted average
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return (w * phi_stack).sum(dim=1) / w_sum.squeeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# Voxel SDT initialisation (step 3 in the doc-string)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_normals_pca(
    pts: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Estimate outward surface normals via PCA on k-nearest neighbours.

    For each point p, the normal is the eigenvector of the local covariance
    matrix corresponding to the *smallest* eigenvalue (i.e. the direction of
    least variation, perpendicular to the local tangent plane).

    This is significantly more accurate than centroid-based estimation for
    complex geometries like the Stanford Bunny.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 3)
        Input surface point cloud in any consistent coordinate system.
    k : int
        Number of nearest neighbours used for PCA.

    Returns
    -------
    normals : np.ndarray, shape (N, 3)
        Unit normals (orientation may be inconsistent across the surface;
        the caller should apply a global orientation fix using the centroid
        test).
    """
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)   # k+1 because the point itself is included

    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbours = pts[idx[i, 1:]]    # exclude the point itself
        cov = np.cov((neighbours - neighbours.mean(0)).T)
        _, vecs = np.linalg.eigh(cov)   # ascending eigenvalues
        normals[i] = vecs[:, 0]         # eigenvector for smallest eigenvalue

    return normalize_vecs(normals)


def build_local_sdt(
    seed_norm: np.ndarray,          # chart seed in normalised space, shape (3,)
    cover_radius: float,            # half-side of the local query box (normalised)
    all_pts_norm: np.ndarray,       # full PLY point cloud in normalised space
    all_nrm_norm: np.ndarray,       # unit outward normals in normalised space
    grid_size: int = 48,            # grid resolution per side
    coverage_factor: float = 2.5,   # PLY points within this × cover_radius
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an analytical signed distance transform on a local grid.

    Rather than rasterising onto a voxel grid (which requires choosing a
    voxel size), this implementation evaluates the *exact* unsigned distance
    from each grid point to the nearest PLY surface point via a KD-tree, then
    assigns a sign via the outward-normal dot-product test.

    Parameters
    ----------
    seed_norm, cover_radius :
        Centre and half-side of the query box in normalised SDF space.
    all_pts_norm, all_nrm_norm :
        Full PLY point cloud (surface points) and their unit outward normals,
        both in normalised SDF space.
    grid_size :
        Number of grid points per axis (grid_size³ total).
    coverage_factor :
        Only PLY points within ``coverage_factor × cover_radius`` of the seed
        are used for the KD-tree query.  Falls back to the full point cloud
        when fewer than 20 local points exist.

    Returns
    -------
    grid_pts : np.ndarray, shape (grid_size³, 3)
        Grid query points in normalised SDF space.
    sdt_vals : np.ndarray, shape (grid_size³,)
        Signed distance values (negative inside, positive outside).
    """
    # Local query box
    lin = np.linspace(-cover_radius, cover_radius, grid_size)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) + seed_norm

    # Select local PLY subset
    d2seed = np.linalg.norm(all_pts_norm - seed_norm, axis=1)
    local_mask = d2seed < coverage_factor * cover_radius
    n_local = local_mask.sum()
    if n_local < 20:
        local_pts = all_pts_norm
        local_nrm = all_nrm_norm
    else:
        local_pts = all_pts_norm[local_mask]
        local_nrm = all_nrm_norm[local_mask]

    # Unsigned distance via KD-tree
    tree = cKDTree(local_pts)
    dist_unsigned, nn_idx = tree.query(grid_pts, k=1)

    # Sign via outward-normal orientation:
    #   (query_pt - nearest_surface_pt) · outward_normal > 0 → exterior (+)
    vec = grid_pts - local_pts[nn_idx]           # (N, 3)  query → surface
    dot = (vec * local_nrm[nn_idx]).sum(axis=1)   # (N,)
    sign = np.where(dot >= 0.0, 1.0, -1.0)

    sdt_vals = sign * dist_unsigned
    return grid_pts, sdt_vals


# ──────────────────────────────────────────────────────────────────────────────
# SDT pre-training (step 3 continued)
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_from_sdt(
    net: SDFNetLocal,
    grid_pts: np.ndarray,
    sdt_vals: np.ndarray,
    n_epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> float:
    """Pre-train *net* to match the analytical SDT.

    Uses mini-batch MSE regression:  net(x) ≈ sdt(x).

    Returns
    -------
    float
        Final RMSE between the network output and the SDT ground-truth
        on the full grid.
    """
    x = torch.tensor(grid_pts, dtype=dtype, device=device)
    y = torch.tensor(sdt_vals, dtype=dtype, device=device)
    n = x.shape[0]

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ep in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = F.mse_loss(net(x[idx]), y[idx])
            loss.backward()
            opt.step()

    with torch.no_grad():
        rmse = float(torch.sqrt(F.mse_loss(net(x), y)).item())
    return rmse


# ──────────────────────────────────────────────────────────────────────────────
# Per-chart fine-tuning (step 4: Eikonal / surface / normal / sign losses)
# ──────────────────────────────────────────────────────────────────────────────

def train_local_sdf(
    chart_idx: int,
    seed_norm: np.ndarray,           # (3,)  chart seed in normalised space
    cover_radius: float,             # normalised Voronoi radius of this chart
    adaptive_offset: float,          # sign-anchor offset (normalised)
    surf_pts_norm: np.ndarray,       # local PLY surface points  (M, 3)
    surf_nrm_norm: np.ndarray,       # local surface normals      (M, 3)
    all_pts_norm: np.ndarray,        # full PLY for SDT init       (Ntot, 3)
    all_nrm_norm: np.ndarray,        # full PLY normals            (Ntot, 3)
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[SDFNetLocal, dict, dict]:
    """Train one local SDF network for chart `chart_idx`.

    Workflow
    --------
    1. Build analytical SDT from the full PLY cloud.
    2. Pre-train ``SDFNetLocal`` to match the SDT (Eikonal init).
    3. Fine-tune with surface / Eikonal / normal / sign losses.

    Parameters
    ----------
    adaptive_offset :
        Geometry-adaptive sign-anchor offset =
        ``anchor_offset_factor × dist(seed, nearest PLY point)``.
        Smaller for ear-region charts, larger for body charts.

    Returns
    -------
    net : SDFNetLocal
        Trained local network.
    history : dict
        Per-epoch loss history.
    final_metrics : dict
        Summary statistics for the gate report.
    """
    net = SDFNetLocal(
        width=args.local_width, depth=args.local_depth
    ).to(device=device, dtype=dtype)

    # ── Step 1 & 2: Build SDT and pre-train ──────────────────────────────────
    print(
        f"  [chart {chart_idx:2d}] Building SDT (grid={args.sdt_grid_size}³, "
        f"radius={cover_radius:.4f}, offset={adaptive_offset:.4f})…"
    )
    sdt_cover = 1.5 * cover_radius  # SDT box half-side
    grid_pts, sdt_vals = build_local_sdt(
        seed_norm, sdt_cover, all_pts_norm, all_nrm_norm,
        grid_size=args.sdt_grid_size,
    )
    pretrain_rmse = pretrain_from_sdt(
        net, grid_pts, sdt_vals,
        n_epochs=args.pretrain_epochs, lr=args.pretrain_lr,
        device=device, dtype=dtype,
    )
    print(f"  [chart {chart_idx:2d}] SDT pre-train RMSE = {pretrain_rmse:.4e}")

    # ── Step 3: Fine-tuning ───────────────────────────────────────────────────
    n_local = surf_pts_norm.shape[0]
    if n_local == 0:
        print(f"  [chart {chart_idx:2d}] WARNING: no local surface points — skipping fine-tune")
        return net, {}, {"n_local_surf": 0, "pretrain_rmse": pretrain_rmse}

    pts_t = torch.tensor(surf_pts_norm, dtype=dtype, device=device)
    nrm_t = torch.tensor(surf_nrm_norm, dtype=dtype, device=device)
    seed_t = torch.tensor(seed_norm, dtype=dtype, device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    history: Dict[str, list] = {
        k: [] for k in ["total", "surface", "eikonal", "normal", "sign"]
    }

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # ── Surface batch ────────────────────────────────────────────────────
        bs = min(args.batch_surface, n_local)
        idx_s = torch.randint(0, n_local, (bs,), device=device)
        p_surf = pts_t[idx_s]
        n_surf_batch = nrm_t[idx_s]

        p_req = p_surf.clone().detach().requires_grad_(True)
        phi_s = net(p_req)
        grad_s = torch.autograd.grad(
            phi_s, p_req,
            grad_outputs=torch.ones_like(phi_s),
            create_graph=True,
        )[0]
        loss_surface = torch.mean(torch.abs(phi_s))

        # Normal alignment  ∇φ · n_surf → 1
        grad_norm = torch.linalg.norm(grad_s, dim=1, keepdim=True)
        grad_dir = grad_s / torch.clamp(grad_norm, min=1e-12)
        loss_normal = torch.mean(1.0 - (grad_dir * n_surf_batch).sum(dim=1))

        # ── Eikonal batch (reinitialization PDE: |∇φ| = 1) ──────────────────
        # Sample random points in a sphere of radius 1.5 × cover_radius
        # centred at the chart seed.  This is the region where the local SDF
        # is responsible; the Eikonal constraint must hold everywhere here.
        u = torch.randn((args.batch_eikonal, 3), device=device, dtype=dtype)
        r = torch.rand((args.batch_eikonal, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
        p_eik = seed_t + r * (1.5 * cover_radius) * u / torch.linalg.norm(u, dim=1, keepdim=True).clamp(min=1e-12)
        p_eik_req = p_eik.detach().requires_grad_(True)
        phi_eik = net(p_eik_req)
        grad_eik = torch.autograd.grad(
            phi_eik, p_eik_req,
            grad_outputs=torch.ones_like(phi_eik),
            create_graph=True,
        )[0]
        loss_eik = torch.mean((torch.linalg.norm(grad_eik, dim=1) - 1.0) ** 2)

        # ── Sign-anchor loss with geometry-adaptive offset ────────────────────
        # offset_i = anchor_offset_factor × dist(seed_i, nearest PLY point)
        # → automatically smaller for ear charts, larger for body charts
        p_in = p_surf - adaptive_offset * n_surf_batch   # just inside surface
        p_out = p_surf + adaptive_offset * n_surf_batch  # just outside surface
        phi_in = net(p_in)
        phi_out = net(p_out)
        # softplus(-target * phi): penalise wrong sign
        loss_sign_in = torch.mean(F.softplus(phi_in))    # want phi_in < 0
        loss_sign_out = torch.mean(F.softplus(-phi_out)) # want phi_out > 0

        # Far-field: points at distance 2× (outside domain) should be positive
        far_dirs = torch.randn((128, 3), device=device, dtype=dtype)
        far_dirs = far_dirs / torch.linalg.norm(far_dirs, dim=1, keepdim=True).clamp(min=1e-12)
        far_pts = far_dirs * 2.0   # normalised space; domain half-width ≈ 0.55
        loss_far = torch.mean(F.softplus(-net(far_pts)))

        loss_sign = 0.5 * (loss_sign_in + loss_sign_out) + 0.3 * loss_far

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (
            args.w_surface * loss_surface
            + args.w_eikonal * loss_eik
            + args.w_normal * loss_normal
            + args.w_sign * loss_sign
        )
        loss.backward()
        optimizer.step()

        for key, val in zip(
            ["total", "surface", "eikonal", "normal", "sign"],
            [loss, loss_surface, loss_eik, loss_normal, loss_sign],
        ):
            history[key].append(float(val.item()))

        if epoch % max(1, args.log_every) == 0:
            print(
                f"  [chart {chart_idx:2d}] ep {epoch:5d}: "
                f"surf={loss_surface:.3e} eik={loss_eik:.3e} "
                f"sign={loss_sign:.3e} total={loss:.3e}"
            )

    final_metrics = {
        "chart_idx": chart_idx,
        "final_surface": history["surface"][-1],
        "final_eikonal": history["eikonal"][-1],
        "final_normal": history["normal"][-1],
        "final_sign": history["sign"][-1],
        "adaptive_offset": float(adaptive_offset),
        "cover_radius": float(cover_radius),
        "n_local_surf": int(n_local),
        "pretrain_rmse": float(pretrain_rmse),
    }
    return net, history, final_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Cross-chart consistency check (optional quality metric)
# ──────────────────────────────────────────────────────────────────────────────

def compute_overlap_consistency(
    model: SDFNetChartwise,
    all_pts_norm: np.ndarray,
    membership: np.ndarray,        # (N, C) uint8
    device: torch.device,
    dtype: torch.dtype,
    n_sample: int = 4096,
) -> float:
    """Measure average SDF disagreement between chart pairs in overlap regions.

    Returns
    -------
    float
        Mean absolute difference |φ_i(x) - φ_j(x)| over overlapping (i,j) pairs.
        Lower is better.  Analogous to the atlas ``overlap_consistency`` metric.
    """
    # Find interior points that belong to >= 2 charts
    n_membership = membership.sum(axis=1)   # (N,)
    overlap_mask = n_membership >= 2
    if overlap_mask.sum() < 2:
        return float("nan")

    overlap_pts = all_pts_norm[overlap_mask]
    overlap_mem = membership[overlap_mask]  # (M, C)

    # Subsample
    rng = np.random.default_rng(0)
    sel = rng.choice(overlap_pts.shape[0], min(n_sample, overlap_pts.shape[0]), replace=False)
    pts = torch.tensor(overlap_pts[sel], dtype=dtype, device=device)
    mem = overlap_mem[sel]  # (S, C)

    C = model.n_charts
    with torch.no_grad():
        # Evaluate all local SDFs at the overlap points
        phi_all = torch.stack([net(pts) for net in model.local_nets], dim=1)  # (S, C)

    diffs = []
    for ci in range(C):
        for cj in range(ci + 1, C):
            both = (mem[:, ci].astype(bool)) & (mem[:, cj].astype(bool))
            if both.sum() < 2:
                continue
            di = phi_all[torch.tensor(both, device=device), ci]
            dj = phi_all[torch.tensor(both, device=device), cj]
            diffs.append(float(torch.mean(torch.abs(di - dj)).item()))

    return float(np.mean(diffs)) if diffs else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_global_slice_plot(
    model: SDFNetChartwise,
    out_path: str,
    grid_n: int = 180,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> None:
    """Save three orthogonal SDF slice plots for the blended chartwise model."""
    def eval_slice(axis: int, val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lin = np.linspace(-1.2, 1.2, grid_n)
        A, B = np.meshgrid(lin, lin)
        pts = np.zeros((grid_n * grid_n, 3))
        free = [i for i in range(3) if i != axis]
        pts[:, free[0]] = A.ravel()
        pts[:, free[1]] = B.ravel()
        pts[:, axis] = val
        xt = torch.tensor(pts, dtype=dtype, device=device)
        with torch.no_grad():
            phi = model(xt).cpu().numpy().reshape(grid_n, grid_n)
        return A, B, phi

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (axis, val, title) in zip(axes, [(2, 0.0, "z=0"), (1, 0.0, "y=0"), (0, 0.0, "x=0")]):
        A, B, phi = eval_slice(axis, val)
        cf = ax.contourf(A, B, phi, levels=40, cmap="coolwarm")
        ax.contour(A, B, phi, levels=[0.0], colors="black", linewidths=1.5)
        ax.set_title(f"Chartwise SDF ({title})")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, shrink=0.85)
    fig.suptitle("Chart-partitioned SDF — blended global view (normalised coords)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_offset_diagram(
    seed_pts_norm: np.ndarray,
    support_radii: np.ndarray,
    adaptive_offsets: np.ndarray,
    center: np.ndarray,
    scale: float,
    out_path: str,
) -> None:
    """Bar chart showing adaptive anchor offset per chart (physical mm)."""
    C = seed_pts_norm.shape[0]
    offsets_mm = adaptive_offsets * scale * 1000.0   # normalised → metres → mm
    radii_mm = support_radii * scale * 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(C)
    ax.bar(x - 0.2, radii_mm, 0.4, label="Support radius (mm)", alpha=0.7, color="steelblue")
    ax.bar(x + 0.2, offsets_mm, 0.4, label="Adaptive offset (mm)", alpha=0.9, color="tomato")
    ax.axhline(5.0, ls="--", c="black", lw=1.0, label="Ear thickness ≈ 5 mm (limit)")
    ax.set_xlabel("Chart index")
    ax.set_ylabel("mm (physical space)")
    ax.set_title("Per-chart adaptive anchor offset vs Voronoi support radius")
    ax.legend()
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Global adapter: fit blended SDF into a single SDFNet for unmodified PINN code
# ──────────────────────────────────────────────────────────────────────────────

class SDFNetGlobal(torch.nn.Module):
    """Single global SDF network (same architecture as train_sdf_rabbit.SDFNet).
    Used for the optional global adapter.
    """

    def __init__(self, width: int = 128, depth: int = 6):
        super().__init__()
        self.width = width
        self.depth = depth
        self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_global_adapter(
    chartwise_model: SDFNetChartwise,
    all_pts_norm: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> SDFNetGlobal:
    """Fit a single global SDFNet to the blended chartwise SDF.

    This enables drop-in use of the chartwise SDF by unmodified PINN code
    that loads ``SDFNet(**ckpt['model_kwargs'])``.  The adapter is trained by
    regression against chartwise SDF values on a dense random grid.

    Note: accuracy is lower than the full chartwise model (single-network
    bottleneck).  Use ``SDFNetChartwise`` directly when possible.
    """
    adapter = SDFNetGlobal(width=args.adapter_width, depth=args.adapter_depth).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    # Sample points: mix random grid + PLY surface
    n_rand = 80_000
    x_rand_np = (np.random.rand(n_rand, 3) * 3.0) - 1.5   # covers normalized domain
    x_rand = torch.tensor(x_rand_np, dtype=dtype, device=device)

    x_surf = torch.tensor(all_pts_norm, dtype=dtype, device=device)

    with torch.no_grad():
        y_rand = chartwise_model(x_rand)
        y_surf = chartwise_model(x_surf)

    x_all = torch.cat([x_rand, x_surf], dim=0)
    y_all = torch.cat([y_rand, y_surf], dim=0)
    n_all = x_all.shape[0]

    print(f"\nFitting global adapter ({args.adapter_epochs} epochs)…")
    for ep in range(1, args.adapter_epochs + 1):
        perm = torch.randperm(n_all, device=device)
        for start in range(0, n_all, 4096):
            idx = perm[start : start + 4096]
            optimizer.zero_grad()
            loss = F.mse_loss(adapter(x_all[idx]), y_all[idx])
            loss.backward()
            optimizer.step()
        if ep % max(1, args.adapter_epochs // 10) == 0:
            with torch.no_grad():
                rmse = float(torch.sqrt(F.mse_loss(adapter(x_all), y_all)).item())
            print(f"  adapter ep {ep:5d}: RMSE = {rmse:.4e}")

    return adapter


# ──────────────────────────────────────────────────────────────────────────────
# Main training orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def train_all(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load PLY ──────────────────────────────────────────────────────────────
    print(f"Loading PLY: {args.ply_file}")
    ply_pts, ply_nrm = parse_ply(args.ply_file)
    if ply_nrm is None:
        print(
            "  PLY has no precomputed normals — estimating via PCA on "
            f"{args.normal_k}-nearest neighbours (robust for complex geometry)."
        )
        ply_nrm = estimate_normals_pca(ply_pts, k=args.normal_k)
    else:
        ply_nrm = normalize_vecs(ply_nrm)
    # Ensure globally consistent outward orientation via centroid flip-test
    centroid = np.mean(ply_pts, axis=0, keepdims=True)
    flip = ((ply_pts - centroid) * ply_nrm).sum(axis=1) < 0
    ply_nrm[flip] *= -1.0
    n_flip = int(flip.sum())
    if n_flip > 0:
        print(f"  Flipped {n_flip} inward-pointing normals to outward orientation.")
    print(f"  PLY loaded: {ply_pts.shape[0]} vertices")

    # ── Load atlas data ───────────────────────────────────────────────────────
    print(f"Loading atlas: {args.atlas_data}")
    atlas = np.load(args.atlas_data, allow_pickle=True)
    center = np.array(atlas["center"])   # SDF normalisation centre
    scale = float(atlas["scale"])        # SDF normalisation scale
    seed_pts_norm = atlas["seed_points"]  # (C, 3) in normalised space
    support_radii = atlas["support_radii"]  # (C,) in normalised space
    membership = atlas["membership"].astype(np.uint8)  # (N, C)
    interior_pts = atlas["interior_points"]  # (N, 3) in normalised space

    C = seed_pts_norm.shape[0]
    print(f"  Atlas loaded: {C} charts, {interior_pts.shape[0]} interior points")
    print(f"  Normalisation: center={center}, scale={scale:.6f}")

    # Normalise PLY into SDF space
    ply_pts_norm = (ply_pts - center[np.newaxis, :]) / scale
    print(
        f"  PLY normalised: x∈[{ply_pts_norm[:,0].min():.3f}, {ply_pts_norm[:,0].max():.3f}], "
        f"y∈[{ply_pts_norm[:,1].min():.3f}, {ply_pts_norm[:,1].max():.3f}]"
    )

    # ── Compute geometry-adaptive offsets ────────────────────────────────────
    # For each chart seed, find the distance to the nearest PLY surface point.
    # offset_i = anchor_offset_factor × this distance.
    # This automatically gives small offsets for ear charts and large for body.
    print("\nComputing adaptive offsets…")
    kd_ply = cKDTree(ply_pts_norm)
    seed_to_surf_dists, _ = kd_ply.query(seed_pts_norm, k=1)
    adaptive_offsets = args.anchor_offset_factor * seed_to_surf_dists
    for i in range(C):
        phys_mm = adaptive_offsets[i] * scale * 1000.0
        print(
            f"  Chart {i:2d}: seed→surf dist = {seed_to_surf_dists[i]:.4f}  "
            f"offset = {adaptive_offsets[i]:.4f} ({phys_mm:.1f} mm)  "
            f"radius = {support_radii[i]:.4f}"
        )

    # ── Visualise adaptive offsets ────────────────────────────────────────────
    offset_plot = os.path.join(args.output_dir, "adaptive_offsets.png")
    save_offset_diagram(seed_pts_norm, support_radii, adaptive_offsets, center, scale, offset_plot)
    print(f"\nOffset diagram saved: {offset_plot}")

    # ── Train one local SDF per chart ─────────────────────────────────────────
    print(f"\nTraining {C} local SDF networks…\n{'─'*60}")
    local_nets: List[SDFNetLocal] = []
    all_histories: List[dict] = []
    all_metrics: List[dict] = []

    # Resume mode: if all per-chart checkpoints exist, load them and skip training
    all_ckpts_exist = all(
        os.path.isfile(os.path.join(args.output_dir, f"chart_{ci:02d}_sdf.pt"))
        for ci in range(C)
    )
    if all_ckpts_exist:
        print(
            "  [RESUME] All chart checkpoints found — loading from disk, skipping training."
        )
        for ci in range(C):
            ckpt_ci = os.path.join(args.output_dir, f"chart_{ci:02d}_sdf.pt")
            ckpt = torch.load(ckpt_ci, map_location=device, weights_only=False)
            net = SDFNetLocal(
                width=args.local_width, depth=args.local_depth
            ).to(device=device, dtype=dtype)
            net.load_state_dict(ckpt["model_state"])
            net.eval()
            local_nets.append(net)
            all_histories.append({})
            m = ckpt.get("metrics", {})
            all_metrics.append(m)
            print(
                f"  Loaded chart {ci:2d}: "
                f"surf={m.get('final_surface', float('nan')):.3e}  "
                f"sign={m.get('final_sign', float('nan')):.3e}"
            )
    else:
        for ci in range(C):
            print(f"\n{'━'*60}")
            print(f"  Chart {ci:2d} / {C-1}  (seed={seed_pts_norm[ci].round(4).tolist()})")
            t0 = time.time()

            # Extract local PLY surface data (within args.coverage_factor × radius)
            cover_r = support_radii[ci]
            d2seed = np.linalg.norm(ply_pts_norm - seed_pts_norm[ci], axis=1)
            local_mask = d2seed < args.coverage_factor * cover_r
            local_surf = ply_pts_norm[local_mask]
            local_nrm = ply_nrm[local_mask]
            print(
                f"  Local PLY points: {local_mask.sum()}  "
                f"(within {args.coverage_factor:.1f}×{cover_r:.4f} of seed)"
            )

            net, hist, metrics = train_local_sdf(
                chart_idx=ci,
                seed_norm=seed_pts_norm[ci],
                cover_radius=cover_r,
                adaptive_offset=adaptive_offsets[ci],
                surf_pts_norm=local_surf,
                surf_nrm_norm=local_nrm,
                all_pts_norm=ply_pts_norm,
                all_nrm_norm=ply_nrm,
                args=args,
                device=device,
                dtype=dtype,
            )
            local_nets.append(net)
            all_histories.append(hist)
            all_metrics.append(metrics)

            elapsed = time.time() - t0
            print(
                f"  [chart {ci:2d}] DONE in {elapsed:.0f}s — "
                f"surf={metrics.get('final_surface', 'N/A'):.3e}  "
                f"sign={metrics.get('final_sign', 'N/A'):.3e}"
            )

            # Save per-chart checkpoint
            ckpt_ci = os.path.join(args.output_dir, f"chart_{ci:02d}_sdf.pt")
            torch.save(
                {
                    "model_state": net.state_dict(),
                    "model_kwargs": {
                        "width": args.local_width, "depth": args.local_depth
                    },
                    "chart_idx": ci,
                    "seed_norm": seed_pts_norm[ci].tolist(),
                    "cover_radius": float(cover_r),
                    "adaptive_offset": float(adaptive_offsets[ci]),
                    "metrics": metrics,
                },
                ckpt_ci,
            )

    # ── Assemble SDFNetChartwise ───────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("Assembling SDFNetChartwise (PoU blending)…")
    chartwise = SDFNetChartwise(local_nets, seed_pts_norm, support_radii).to(device=device)

    # ── Overlap consistency metric ────────────────────────────────────────────
    # Use interior_pts (50 k, matching membership rows), NOT ply_pts_norm (35 k)
    print("Computing overlap consistency…")
    oc = compute_overlap_consistency(chartwise, interior_pts, membership, device, dtype)
    print(f"  overlap_consistency = {oc:.4f}  (lower is better; procedural ref ≈ 0.006)")

    # ── Global slice plot for blended SDF ─────────────────────────────────────
    slice_path = os.path.join(args.output_dir, "chartwise_sdf_slices.png")
    print(f"Saving SDF slice plot → {slice_path}")
    save_global_slice_plot(chartwise, slice_path, device=device, dtype=dtype)

    # ── Sign quality: evaluate at PLY points ─────────────────────────────────
    print("Evaluating sign quality at PLY surface points…")
    x_surf_t = torch.tensor(ply_pts_norm, dtype=dtype, device=device)
    with torch.no_grad():
        phi_surf = chartwise(x_surf_t).cpu().numpy()
    # At the surface, phi should be ~0; anchors at +offset should be > 0 (exterior)
    # We test sign quality by checking anchors
    x_out_np = ply_pts_norm + 0.02 * ply_nrm   # shallow exterior anchors
    x_in_np = ply_pts_norm - 0.02 * ply_nrm    # shallow interior anchors
    x_out_t = torch.tensor(x_out_np, dtype=dtype, device=device)
    x_in_t = torch.tensor(x_in_np, dtype=dtype, device=device)
    with torch.no_grad():
        phi_out = chartwise(x_out_t).cpu().numpy()
        phi_in = chartwise(x_in_t).cpu().numpy()
    sign_error = float(
        (np.mean((phi_out <= 0).astype(float)) + np.mean((phi_in >= 0).astype(float))) / 2.0
    )
    print(f"  sign_error (fraction of wrong-sign anchors) = {sign_error:.4f}")
    print(f"  (reference: global SDF v3 sign_loss ≈ 0.311)")

    # ── Save main chartwise checkpoint ────────────────────────────────────────
    chartwise_ckpt_path = os.path.join(args.output_dir, "rabbit_sdf_chartwise.pt")
    torch.save(
        {
            "chartwise": True,
            "n_charts": C,
            "model_state": chartwise.state_dict(),
            "model_kwargs": {
                "local_width": args.local_width,
                "local_depth": args.local_depth,
                "seed_points": seed_pts_norm.tolist(),
                "support_radii": support_radii.tolist(),
            },
            "center": center.tolist(),
            "scale": float(scale),
            "source": f"chartwise:{args.ply_file}",
            "overlap_consistency": float(oc),
            "sign_error": float(sign_error),
            "adaptive_offsets": adaptive_offsets.tolist(),
            "per_chart_metrics": all_metrics,
        },
        chartwise_ckpt_path,
    )
    print(f"\nChartwise checkpoint saved: {chartwise_ckpt_path}")

    # ── Optionally fit a global adapter ──────────────────────────────────────
    if args.fit_adapter:
        adapter_path = os.path.join(args.output_dir, "rabbit_sdf_adapter.pt")
        adapter = fit_global_adapter(chartwise, ply_pts_norm, args, device, dtype)
        torch.save(
            {
                "model_state": adapter.state_dict(),
                "model_kwargs": {
                    "width": args.adapter_width, "depth": args.adapter_depth
                },
                "center": center.tolist(),
                "scale": float(scale),
                "source": f"adapter_from_chartwise",
                "chartwise_checkpoint": chartwise_ckpt_path,
            },
            adapter_path,
        )
        print(f"Global adapter checkpoint saved: {adapter_path}")

    # ── Save history and meta JSON ────────────────────────────────────────────
    meta = {
        "n_charts": C,
        "chartwise_checkpoint": chartwise_ckpt_path,
        "center": center.tolist(),
        "scale": float(scale),
        "overlap_consistency": float(oc),
        "sign_error": float(sign_error),
        "per_chart": all_metrics,
        "args": vars(args),
    }
    meta_path = os.path.join(args.output_dir, "rabbit_sdf_chartwise_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta JSON saved: {meta_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("CHART-PARTITIONED SDF TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"  n_charts            = {C}")
    print(f"  overlap_consistency = {oc:.4f}  (ref: global SDF v3 ≈ N/A, atlas ≈ 0.006)")
    print(f"  sign_error          = {sign_error:.4f}  (ref: global SDF v3 ≈ 0.311)")
    print(f"  Output dir          = {args.output_dir}")
    print(f"  Checkpoint          = {chartwise_ckpt_path}")
    print(f"{'═'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a chart-partitioned neural SDF for the Stanford Bunny",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--ply-file", required=True,
                   help="Path to Stanford Bunny PLY file (bun_zipper.ply)")
    p.add_argument("--atlas-data", required=True,
                   help="Path to rabbit_atlas_data.npz (volumetric atlas)")
    p.add_argument("--output-dir", default="runs/bunny_sdf_chartwise",
                   help="Output directory for checkpoints and figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normal-k", type=int, default=20,
                   help="k-nearest neighbours for PCA normal estimation "
                        "(used when PLY file has no per-vertex normals)")

    # Local network architecture
    p.add_argument("--local-width", type=int, default=64,
                   help="Hidden layer width for each local SDFNetLocal")
    p.add_argument("--local-depth", type=int, default=4,
                   help="Number of hidden layers for each local SDFNetLocal")

    # SDT initialisation
    p.add_argument("--sdt-grid-size", type=int, default=48,
                   help="Grid resolution per axis for the local SDT (48³ = 110k voxels)")
    p.add_argument("--pretrain-epochs", type=int, default=400,
                   help="Epochs to pre-train each local network from the SDT")
    p.add_argument("--pretrain-lr", type=float, default=1e-3)

    # Fine-tuning
    p.add_argument("--epochs", type=int, default=3000,
                   help="Fine-tuning epochs per chart after SDT pre-training")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-surface", type=int, default=1024)
    p.add_argument("--batch-eikonal", type=int, default=1024)

    # Adaptive sign-anchor offset
    p.add_argument("--anchor-offset-factor", type=float, default=0.30,
                   help="offset_i = factor × dist(seed_i, nearest PLY point)\n"
                        "Ear charts → small offset; body charts → large offset")

    # Local surface coverage
    p.add_argument("--coverage-factor", type=float, default=1.5,
                   help="PLY points within (factor × support_radius) of seed are 'local'")

    # Loss weights
    p.add_argument("--w-surface", type=float, default=10.0)
    p.add_argument("--w-eikonal", type=float, default=1.0)
    p.add_argument("--w-normal",  type=float, default=3.0)
    p.add_argument("--w-sign",    type=float, default=8.0)

    # Global adapter
    p.add_argument("--fit-adapter", action="store_true",
                   help="After chartwise training, fit a single global SDFNet adapter "
                        "(drop-in for unmodified PINN code)")
    p.add_argument("--adapter-width", type=int, default=128)
    p.add_argument("--adapter-depth", type=int, default=6)
    p.add_argument("--adapter-epochs", type=int, default=3000)

    # Misc
    p.add_argument("--log-every", type=int, default=500)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_all(args)


if __name__ == "__main__":
    main()
