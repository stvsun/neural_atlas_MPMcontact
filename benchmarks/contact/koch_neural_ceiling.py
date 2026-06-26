#!/usr/bin/env python3
"""CV-6 refinement-ceiling experiment: a FIXED-capacity neural SDF vs the exact Koch SDF.

This *measures* the claim that the CV-6 prose and `solvers/contact/koch.py` previously only
*argued*: a fixed-width neural SDF cannot keep resolving self-similar fractal detail as the
Koch level ``n`` grows, while the recursive IFS chart refines to any depth on demand. See
``docs/contact_verification_manual.md`` §11.6.

Protocol (held fixed across levels => the capacity ceiling is the only moving part):
  * Network: ``atlas/sdf/train_sdf.py::SDFNet`` of a SINGLE fixed (width, depth) for every n.
  * Supervision: the EXACT Koch signed distance ``koch.nearest_boundary(x, n)[0]`` (sign from
    ``koch.inside``, magnitude verified == brute-force polyline distance) at points sampled in
    a near-boundary band plus the bulk box. The 2-D shape is embedded at z=0 (thin z-slab,
    z-independent target => a prism SDF: |grad phi|=1 in 3-D and n_z ~ 0).
  * Train separately at n = 1, 2, 3, 4, 5, ... with the SAME compute budget.
  * Per level record, on a HELD-OUT near-boundary point set: gap RMSE vs the exact SDF (the
    quantity §11.2 asks for) and the Eikonal residual mean (|grad phi| - 1)^2 (the tau_g
    driver §11.3 lists as "not yet measured").

The error rises as n outgrows what the fixed net can represent -- the measured ceiling.

Run:
  python3 benchmarks/contact/koch_neural_ceiling.py                 # full figure run (cached)
  python3 benchmarks/contact/koch_neural_ceiling.py --quick         # fast smoke run

Outputs:
  figures/koch_neural_ceiling_pub.png (+ .pdf)   -- the measured error-vs-n curve
  runs/koch_neural_ceiling/metrics.json          -- per-level metrics + hyperparameters
  runs/koch_neural_ceiling/cache/level_*.npz     -- precomputed exact-SDF datasets (reused)

Pure-numpy supervision (matches solvers/contact/koch.py); PyTorch for the net (matches
atlas/sdf/train_sdf.py). CPU/float64 throughout (the Koch SDF references are float64).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solvers.contact import koch                                     # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(_ROOT, "runs", "koch_neural_ceiling")
CACHE_DIR = os.path.join(RUN_DIR, "cache")
FIG_DIR = os.path.join(_ROOT, "figures")


# ---------------------------------------------------------------------------
# Exact Koch signed-distance supervision (pure numpy, via the verified chart)
# ---------------------------------------------------------------------------

def koch_signed_distance(points_xy: np.ndarray, level: int, R: float = 1.0) -> np.ndarray:
    """Exact Koch signed distance (negative inside) at each (x, y) row.

    Loops the verified ``koch.nearest_boundary`` (signed gap == brute-force polyline
    distance, sign from ``koch.inside``)."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    return np.array([koch.nearest_boundary(p, level, R=R)[0] for p in pts])


def koch_sdf_and_normal(points_xy: np.ndarray, level: int, R: float = 1.0):
    """Exact Koch signed distance AND outward unit normal at each (x, y) row (one
    ``koch.nearest_boundary`` pass; the normal is its verified gap-ascent direction)."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    g = np.empty(len(pts))
    nrm = np.empty((len(pts), 2))
    for i, p in enumerate(pts):
        gi, _foot, ni = koch.nearest_boundary(p, level, R=R)
        g[i] = gi
        nrm[i] = ni
    return g, nrm


def body_size(level: int, R: float = 1.0) -> float:
    """Characteristic body size L (max boundary radius), matching the CV-5 convention."""
    V = koch.snowflake_vertices(level, R=R)[:-1]
    c = V.mean(axis=0)
    return float(np.linalg.norm(V - c, axis=1).max())


def sample_near_boundary(level: int, n: int, band: float, seed: int, R: float = 1.0) -> np.ndarray:
    """Points in a band around the level-n boundary: a random spot on a random segment,
    jittered by an isotropic Gaussian of std ``band`` (so |gap| ~ band)."""
    rng = np.random.RandomState(seed)
    V = koch.snowflake_vertices(level, R=R)
    P, Q = V[:-1], V[1:]
    idx = rng.randint(0, len(P), size=n)
    t = rng.rand(n)
    on = P[idx] + t[:, None] * (Q[idx] - P[idx])
    return on + rng.randn(n, 2) * band


def sample_on_boundary(level: int, n: int, seed: int, R: float = 1.0) -> np.ndarray:
    """Points EXACTLY on the level-n boundary (random spot on a random segment, no jitter).
    The exact signed distance there is 0, so |phi_nn| at these points IS the zero-level-set
    deviation -- the distance of the net's learned surface from the true boundary (manual
    §11.2). This is the scale-robust ceiling probe: a fixed near-band stops "seeing" features
    once they shrink below it, but on-boundary points always sit on the finest detail."""
    rng = np.random.RandomState(seed)
    V = koch.snowflake_vertices(level, R=R)
    P, Q = V[:-1], V[1:]
    idx = rng.randint(0, len(P), size=n)
    t = rng.rand(n)
    return P[idx] + t[:, None] * (Q[idx] - P[idx])


def sample_bulk(n: int, seed: int, half: float = 1.3) -> np.ndarray:
    """Uniform points in the bounding box (far-field sign/magnitude supervision)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-half, half, size=(n, 2))


def build_dataset(
    level: int, n_near: int, n_bulk: int, band: float, seed: int, R: float = 1.0,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """(xy, signed_gap) training set for a level, cached to NPZ (the exact-SDF precompute
    is the experiment's only slow step and is identical across re-runs)."""
    tag = f"L{level}_near{n_near}_bulk{n_bulk}_band{band:g}_R{R:g}_s{seed}"
    path = os.path.join(CACHE_DIR, f"{tag}.npz")
    if use_cache and os.path.isfile(path):
        d = np.load(path)
        return d["xy"], d["g"]
    near = sample_near_boundary(level, n_near, band, seed, R)
    bulk = sample_bulk(n_bulk, seed + 1)
    xy = np.vstack([near, bulk])
    g = koch_signed_distance(xy, level, R=R)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, xy=xy, g=g)
    return xy, g


# ---------------------------------------------------------------------------
# Fixed-capacity neural SDF training + evaluation
# ---------------------------------------------------------------------------

def extrusion_sdf(d2: np.ndarray, z: np.ndarray, H: float) -> np.ndarray:
    """Exact 3-D signed distance of the 2-D Koch shape extruded to a finite prism |z| <= H.

    Standard extrusion SDF: with ``d2`` = the in-plane signed distance and ``dz = |z| - H``,
        sdf = min(max(d2, dz), 0) + |max([d2, dz], 0)|.
    For near-boundary points (|d2| < H) at z=0 this reduces to ``d2`` exactly, so the z=0
    evaluation still compares to the 2-D ``koch.nearest_boundary`` reference -- but supervising
    a TRUE 3-D SDF (caps included) is what teaches the net dphi/dz = 0 at z=0 (small n_z) and
    keeps the Eikonal term from leaking gradient into z (a flat z-slab target does not)."""
    dz = np.abs(z) - H
    outside = np.sqrt(np.maximum(d2, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    inside = np.minimum(np.maximum(d2, dz), 0.0)
    return outside + inside


def _grad_norm(model, x3: np.ndarray) -> np.ndarray:
    """|grad phi| at each 3-D point (raw autograd gradient norm; the Eikonal quantity)."""
    import torch
    x = torch.tensor(x3, dtype=torch.float64, requires_grad=True)
    phi = model(x)
    if phi.dim() > 1:
        phi = phi.squeeze(-1)
    g = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=False)[0]
    return g.norm(dim=1).detach().numpy()


def train_level(
    level: int,
    *,
    width: int,
    depth: int,
    epochs: int,
    lr: float = 1e-3,
    n_near: int = 2500,
    n_bulk: int = 1500,
    band: float = 0.04,
    H: float = 0.5,
    z_range: float = 0.75,
    n_z_slices: int = 5,
    w_eik: float = 0.1,
    batch_eik: int = 2048,
    n_eval: int = 2000,
    eval_band: float = 0.03,
    seed: int = 0,
    R: float = 1.0,
    use_cache: bool = True,
    save_model_dir: str = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train one fixed-capacity SDFNet on the exact level-n Koch SDF and return metrics on a
    held-out near-boundary set: gap RMSE (abs + /L), Eikonal-residual mean & RMS, max |n_z|."""
    import torch
    from atlas.sdf.train_sdf import SDFNet, set_seed

    set_seed(seed)
    dtype = torch.float64

    # ---- training data: in-plane exact SDF (cached, z-independent) lifted to a finite
    #      prism |z|<=H via extrusion_sdf, so the net learns a true 3-D SDF (dphi/dz=0 at z=0).
    #      Each (x,y) is replicated across n_z_slices symmetric z-levels (incl. z=0) so the
    #      net actually learns z-invariance -> n_z ~ 0 at the z=0 evaluation slice.
    xy, d2 = build_dataset(level, n_near, n_bulk, band, seed, R=R, use_cache=use_cache)
    zs = np.linspace(-z_range, z_range, n_z_slices)        # symmetric, includes z=0 when odd
    xy_rep = np.tile(xy, (n_z_slices, 1))
    d2_rep = np.tile(d2, n_z_slices)
    z_rep = np.repeat(zs, len(xy))
    Xt = torch.tensor(np.column_stack([xy_rep, z_rep]), dtype=dtype)
    gt = torch.tensor(extrusion_sdf(d2_rep, z_rep, H), dtype=dtype)

    model = SDFNet(width=width, depth=depth).to(dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eik_rng = np.random.RandomState(seed + 99)
    half = 1.3

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        phi = model(Xt)
        loss_data = torch.mean(torch.abs(phi - gt))
        pe = np.empty((batch_eik, 3))
        pe[:, :2] = eik_rng.uniform(-half, half, size=(batch_eik, 2))
        pe[:, 2] = eik_rng.uniform(-z_range, z_range, size=batch_eik)
        pe_t = torch.tensor(pe, dtype=dtype, requires_grad=True)
        phe = model(pe_t)
        ge = torch.autograd.grad(phe, pe_t, torch.ones_like(phe), create_graph=True)[0]
        loss_eik = torch.mean((torch.linalg.norm(ge, dim=1) - 1.0) ** 2)
        loss = loss_data + w_eik * loss_eik
        loss.backward()
        opt.step()
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == 1):
            print(f"    level {level} epoch {epoch}/{epochs} "
                  f"data={loss_data.item():.3e} eik={loss_eik.item():.3e}")

    # ---- held-out near-boundary evaluation (distinct seed) ----
    from solvers.contact.gap import evaluate_gap
    xy_e = sample_near_boundary(level, n_eval, eval_band, seed + 12345, R=R)
    g_ana, n_ana = koch_sdf_and_normal(xy_e, level, R=R)
    x3 = np.hstack([xy_e, np.zeros((len(xy_e), 1))])
    g_nn, n_nn = evaluate_gap(torch.tensor(x3, dtype=dtype), model)
    g_nn = g_nn.numpy()
    n_nn = n_nn.numpy()

    L = body_size(level, R)
    gap_rmse = float(np.sqrt(np.mean((g_nn - g_ana) ** 2)))
    grad_n = _grad_norm(model, x3)
    eik_mean = float(np.mean((grad_n - 1.0) ** 2))         # mean (|grad phi| - 1)^2  (= loss_eik units)
    eik_rms = float(np.sqrt(eik_mean))
    # in-plane normal angle vs the exact Koch outward normal (degrees)
    nn2 = n_nn[:, :2]
    nn2 = nn2 / np.clip(np.linalg.norm(nn2, axis=1, keepdims=True), 1e-12, None)
    cosang = np.clip(np.sum(nn2 * n_ana, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))

    # PRIMARY ceiling metric: zero-level-set deviation -- RMS |phi_nn| at points ON the true
    # boundary (g_ana = 0), i.e. how far the net's surface sits from the level-n boundary.
    # Scale-robust (always probes the finest detail), unlike the fixed near-band gap RMSE.
    xy_b = sample_on_boundary(level, n_eval, seed + 24680, R=R)
    x3b = np.hstack([xy_b, np.zeros((len(xy_b), 1))])
    g_b, _ = evaluate_gap(torch.tensor(x3b, dtype=dtype), model)
    boundary_rmse = float(np.sqrt(np.mean(g_b.numpy() ** 2)))

    if save_model_dir:
        os.makedirs(save_model_dir, exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "model_kwargs": {"width": width, "depth": depth},
                    "level": int(level)},
                   os.path.join(save_model_dir, f"koch_sdf_level{level}.pt"))

    return {
        "level": int(level),
        "n_segments": int(koch.n_segments(level)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "L": L,
        "boundary_rmse": boundary_rmse,
        "boundary_rmse_rel": boundary_rmse / L,           # PRIMARY: zero-level-set deviation / L
        "gap_rmse": gap_rmse,
        "gap_rmse_rel": gap_rmse / L,                     # secondary: fixed near-band gap RMSE / L
        "eik_residual_mean": eik_mean,
        "eik_residual_rms": eik_rms,
        "normal_angle_median_deg": float(np.median(ang)),
        "normal_angle_mean_deg": float(np.mean(ang)),
        "max_abs_nz": float(np.max(np.abs(n_nn[:, 2]))),
        "feature_scale": float(R * 3.0 ** (-level)),       # smallest resolvable detail / R
        "train_seconds": round(time.time() - t0, 2),
    }


def run_experiment(
    levels: List[int],
    *,
    width: int,
    depth: int,
    epochs: int,
    seed: int = 0,
    use_cache: bool = True,
    verbose: bool = True,
    save_path: str = None,
    save_model_dir: str = None,
    **train_kwargs,
) -> Dict:
    """Train a fixed-capacity net separately at every level; return results + metadata.

    If ``save_path`` is given the (growing) report is written after EACH level, so a long
    background run is crash-safe and the per-level dataset cache is never wasted."""
    results = []
    report = {
        "hyperparameters": {
            "width": width, "depth": depth, "epochs": epochs, "seed": seed, **train_kwargs,
        },
        "levels": results,
    }
    for n in levels:
        if verbose:
            print(f"  [level {n}] training fixed SDFNet (width={width}, depth={depth}) ...")
        m = train_level(n, width=width, depth=depth, epochs=epochs, seed=seed,
                        use_cache=use_cache, save_model_dir=save_model_dir,
                        verbose=verbose, **train_kwargs)
        if verbose:
            print(f"    -> bndRMSE/L={m['boundary_rmse_rel']:.3e}  "
                  f"bandgap/L={m['gap_rmse_rel']:.3e}  "
                  f"eik(mean)={m['eik_residual_mean']:.3e}  "
                  f"|n_z|max={m['max_abs_nz']:.2e}  ({m['train_seconds']}s)")
        results.append(m)
        report["hyperparameters"]["n_params"] = results[0]["n_params"]
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_results(report: Dict, out_path: str) -> str:
    """Three-panel measured error-vs-n curve for a fixed-capacity neural SDF on the Koch SDF:
    (A) magnitude error (zero-level-set deviation + near-band gap RMSE), (B) Eikonal residual,
    (C) median normal-angle error. All three jump off the representable-level (n=1) floor and
    plateau at a capacity ceiling for n>=2 -- the net never recovers n=1 fidelity as the fractal
    detail refines below its resolution (the Koch shape converges in Hausdorff distance, so the
    magnitude error rises to a peak ~ where feature scale = net resolution, then plateaus; the
    normal angle saturates monotonically near ~45 deg = orientation lost)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(_ROOT, "postprocessing"))
    from utils import set_pub_style, PUB_COLORS, DOUBLE_COL_W      # noqa: E402
    set_pub_style()

    lv = report["levels"]
    levels = [r["level"] for r in lv]
    bnd_rel = [r["boundary_rmse_rel"] for r in lv]
    gap_rel = [r["gap_rmse_rel"] for r in lv]
    eik = [r["eik_residual_mean"] for r in lv]
    ang = [r.get("normal_angle_median_deg", float("nan")) for r in lv]
    hp = report["hyperparameters"]
    n_params = hp.get("n_params")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))

    # (A) magnitude error: zero-level-set deviation (primary) + near-band gap RMSE (faded)
    axA.semilogy(levels, bnd_rel, "o-", color=PUB_COLORS[1], label="zero-level-set dev. / L")
    axA.semilogy(levels, gap_rel, "x--", color="0.6", lw=0.9, ms=4, label="near-band gap RMSE / L")
    axA.axhline(bnd_rel[0], ls=":", lw=0.9, color=PUB_COLORS[0],
                label=f"$\\tau_g$ floor (n=1)={bnd_rel[0]:.1e}")
    axA.set_xlabel("fractal level $n$"); axA.set_ylabel("boundary error $/\\,L$")
    axA.set_title("(a)", loc="left"); axA.set_xticks(levels)
    axA.legend(loc="lower right", fontsize=5.6)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # (B) Eikonal residual
    axB.semilogy(levels, eik, "s-", color=PUB_COLORS[2],
                 label="$\\langle(|\\nabla\\phi|-1)^2\\rangle$")
    axB.axhline(eik[0], ls=":", lw=0.9, color=PUB_COLORS[0], label=f"floor (n=1)={eik[0]:.1e}")
    axB.set_xlabel("fractal level $n$"); axB.set_ylabel("Eikonal residual (near boundary)")
    axB.set_title("(b)", loc="left"); axB.set_xticks(levels)
    axB.legend(loc="lower right", fontsize=5.8)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # (C) median normal-angle error (the monotone, cleanest ceiling indicator)
    axC.plot(levels, ang, "^-", color=PUB_COLORS[3], label="median normal angle")
    axC.axhline(ang[0], ls=":", lw=0.9, color=PUB_COLORS[0], label=f"floor (n=1)={ang[0]:.0f}$^\\circ$")
    axC.set_xlabel("fractal level $n$"); axC.set_ylabel("normal-angle error (deg)")
    axC.set_title("(c)", loc="left"); axC.set_xticks(levels)
    axC.set_ylim(0, max(ang) * 1.25)
    axC.legend(loc="lower right", fontsize=5.8)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    cap = f"width={hp['width']}, depth={hp['depth']}"
    if n_params:
        cap += f" ({n_params:,} params, FIXED)"
    # (descriptive sup-title removed; identification moved to the LaTeX caption)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV-6 neural-SDF refinement-ceiling experiment")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--width", type=int, default=64, help="FIXED net width (held across levels)")
    p.add_argument("--depth", type=int, default=4, help="FIXED net depth (held across levels)")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-near", type=int, default=2500)
    p.add_argument("--n-bulk", type=int, default=1500)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--band", type=float, default=0.04)
    p.add_argument("--eval-band", type=float, default=0.03)
    p.add_argument("--w-eik", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (levels 1-3, tiny net, few epochs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.levels = [1, 2, 3]
        args.width, args.depth, args.epochs = 48, 3, 800
        args.n_near, args.n_bulk, args.n_eval = 800, 500, 600

    os.makedirs(RUN_DIR, exist_ok=True)
    metrics_path = os.path.join(RUN_DIR, "metrics.json")
    print(f"CV-6 refinement-ceiling experiment | levels={args.levels} "
          f"width={args.width} depth={args.depth} epochs={args.epochs}")
    report = run_experiment(
        args.levels, width=args.width, depth=args.depth, epochs=args.epochs,
        lr=args.lr, n_near=args.n_near, n_bulk=args.n_bulk, n_eval=args.n_eval,
        band=args.band, eval_band=args.eval_band, w_eik=args.w_eik, seed=args.seed,
        use_cache=not args.no_cache, save_path=metrics_path,
        save_model_dir=os.path.join(RUN_DIR, "models"),
    )
    fig_path = plot_results(report, os.path.join(FIG_DIR, "koch_neural_ceiling_pub.png"))

    print("\nSummary (fixed capacity, rising level):")
    print(f"  {'n':>2} {'segments':>9} {'bndRMSE/L':>11} {'bandgap/L':>11} {'eik(mean)':>11} {'|nz|max':>9}")
    for r in report["levels"]:
        print(f"  {r['level']:>2} {r['n_segments']:>9} {r['boundary_rmse_rel']:>11.3e} "
              f"{r['gap_rmse_rel']:>11.3e} {r['eik_residual_mean']:>11.3e} {r['max_abs_nz']:>9.2e}")
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved figure:  {fig_path}")


if __name__ == "__main__":
    main()
