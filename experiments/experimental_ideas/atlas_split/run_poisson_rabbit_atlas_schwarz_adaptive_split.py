#!/usr/bin/env python3
"""
Adaptive Poisson rabbit pipeline:
baseline solve -> split high-score charts -> warmstart atlas retrain ->
refined solve (split-only) -> refined solve (with hotspot-weighted interface overrides).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_simple_yaml(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def build_run_stem(run_tag: str) -> str:
    run_tag = (run_tag or "").strip()
    if len(run_tag) == 0:
        return "rabbit_poisson_schwarz"
    return f"rabbit_poisson_schwarz_{run_tag}"


def run_cmd(cmd: List[str], log_path: str, cwd: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    child_env = os.environ.copy()
    # Keep MPS fallback enabled by default for PyTorch ops that are not implemented on Metal.
    child_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        logf.write(f"ENV: PYTORCH_ENABLE_MPS_FALLBACK={child_env.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
        code = int(proc.returncode)
        logf.write(f"\nEXIT_CODE={code}\n")
        return code


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def score_charts_from_metrics(metrics_path: str) -> Dict[str, object]:
    m = read_json(metrics_path)
    pcs = m.get("per_chart")
    if not isinstance(pcs, list):
        raise RuntimeError(f"Missing per_chart in {metrics_path}")
    rows = []
    sizes = []
    for row in pcs:
        rel = row.get("relative_l2_error")
        npts = int(row.get("n_points", 0))
        cid = int(row.get("chart_id", -1))
        if rel is None or cid < 0:
            continue
        sizes.append(max(1, npts))
        rows.append((cid, float(rel), npts))
    if not rows:
        raise RuntimeError(f"No valid per-chart rows in {metrics_path}")
    med_sz = float(np.median(np.asarray(sizes, dtype=float)))
    med_sz = max(med_sz, 1.0)
    max_cid = max(cid for cid, _, _ in rows)
    scores = np.zeros((max_cid + 1,), dtype=float)
    entries = []
    for cid, rel, npts in rows:
        scale = (max(float(npts), 1.0) / med_sz) ** 0.20
        napinn = rel * scale
        scores[cid] = napinn
        entries.append(
            {
                "chart_id": int(cid),
                "relative_l2_error": float(rel),
                "n_points": int(npts),
                "naPINN_proxy": float(napinn),
            }
        )
    entries = sorted(entries, key=lambda r: float(r["naPINN_proxy"]), reverse=True)
    return {
        "score_method": "per_chart_relative_l2_proxy",
        "chart_scores": scores.tolist(),
        "per_chart": entries,
    }


def compute_interface_hotspots(
    solution_npz: str,
    atlas_data_npz: str,
    alpha: float,
    top_pairs: int,
    min_shared: int,
) -> Dict[str, object]:
    sol = np.load(solution_npz)
    atlas = np.load(atlas_data_npz)
    vals = np.asarray(sol["chart_values"], dtype=float)
    membership = np.asarray(atlas["membership"]).astype(bool)
    support_r = np.asarray(atlas["support_radii"], dtype=float)
    n_points, n_charts = membership.shape

    pairs = []
    for i in range(n_charts):
        mi = membership[:, i]
        for j in range(i + 1, n_charts):
            shared = mi & membership[:, j]
            n = int(np.sum(shared))
            if n < int(min_shared):
                continue
            d = vals[shared, i] - vals[shared, j]
            if_val_rms = float(np.sqrt(np.mean(d**2)))
            # Proxy for flux mismatch when direct per-pair flux is unavailable.
            scale = max(1e-8, 0.5 * (support_r[i] + support_r[j]))
            if_flux_rms_proxy = float(if_val_rms / scale)
            pairs.append(
                {
                    "pair": [int(i), int(j)],
                    "shared_points": int(n),
                    "if_val_rms": if_val_rms,
                    "if_flux_rms_proxy": if_flux_rms_proxy,
                }
            )

    if not pairs:
        return {
            "pairs": [],
            "top_pairs": [],
            "hotspot_chart_ids": [],
            "note": "no_pairs_with_min_shared",
        }

    max_val = max(p["if_val_rms"] for p in pairs)
    max_flux = max(p["if_flux_rms_proxy"] for p in pairs)
    max_val = max(max_val, 1e-12)
    max_flux = max(max_flux, 1e-12)
    for p in pairs:
        nv = p["if_val_rms"] / max_val
        nf = p["if_flux_rms_proxy"] / max_flux
        p["score"] = float(alpha * nv + (1.0 - alpha) * nf)

    pairs = sorted(pairs, key=lambda r: float(r["score"]), reverse=True)
    top_k = max(1, min(int(top_pairs), len(pairs)))
    top = pairs[:top_k]
    charts = sorted({int(x) for p in top for x in p["pair"]})
    return {
        "pairs": pairs,
        "top_pairs": top,
        "hotspot_chart_ids": charts,
        "alpha": float(alpha),
        "if_flux_note": "proxy_from_value_mismatch_over_support_radius",
    }


def derive_hotspot_overrides(
    hotspot: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    top = hotspot.get("top_pairs", [])
    if not isinstance(top, list) or len(top) == 0:
        return {
            "severity": 0.0,
            "solver_overrides": [],
            "hotspot_chart_ids": [],
        }
    sev = float(np.mean([float(p.get("score", 0.0)) for p in top]))
    sev = float(np.clip(sev, 0.0, 1.0))

    w_if_val = args.w_interface_value_base * min(args.max_if_value_mult, 1.0 + args.beta * sev)
    w_if_flux = args.w_interface_flux_base * min(args.max_if_flux_mult, 1.0 + args.gamma * sev)
    robin = args.robin_lambda_base * min(args.max_robin_mult, 1.0 + args.eta * sev)
    if_batch = int(max(8, round(args.if_batch_base * args.r_hot)))
    detail_topk = int(max(args.detail_chart_topk_base, len(hotspot.get("hotspot_chart_ids", []))))
    detail_boost = args.interface_detail_boost_base * min(args.max_detail_boost_mult, 1.0 + 0.75 * sev)

    overrides = [
        "--w-interface-value",
        f"{w_if_val:.8g}",
        "--w-interface-flux",
        f"{w_if_flux:.8g}",
        "--robin-lambda",
        f"{robin:.8g}",
        "--if-batch",
        str(if_batch),
        "--detail-chart-topk",
        str(detail_topk),
        "--interface-detail-boost",
        f"{detail_boost:.8g}",
    ]
    return {
        "severity": sev,
        "solver_overrides": overrides,
        "hotspot_chart_ids": hotspot.get("hotspot_chart_ids", []),
        "derived_values": {
            "w_interface_value": w_if_val,
            "w_interface_flux": w_if_flux,
            "robin_lambda": robin,
            "if_batch": if_batch,
            "detail_chart_topk": detail_topk,
            "interface_detail_boost": detail_boost,
        },
    }


def read_top_chart_ids_from_scores(chart_score_path: str, topk: int) -> List[int]:
    if topk <= 0 or (not os.path.isfile(chart_score_path)):
        return []
    payload = read_json(chart_score_path)
    per_chart = payload.get("per_chart")
    if not isinstance(per_chart, list):
        return []
    out: List[int] = []
    for row in per_chart:
        cid = int(row.get("chart_id", -1))
        if cid >= 0:
            out.append(cid)
        if len(out) >= int(topk):
            break
    return out


def map_old_to_new_ids(old_ids: List[int], split_map: Dict[str, object]) -> List[int]:
    old_to_new = split_map.get("old_to_new")
    if not isinstance(old_to_new, dict):
        return old_ids
    out: List[int] = []
    for oid in old_ids:
        vals = old_to_new.get(str(int(oid)))
        if isinstance(vals, list):
            out.extend([int(x) for x in vals])
        elif vals is not None:
            out.append(int(vals))
    return sorted({int(x) for x in out if int(x) >= 0})


def expand_with_neighbors(chart_ids: List[int], overlap_graph: object) -> List[int]:
    if not isinstance(overlap_graph, list):
        return sorted({int(x) for x in chart_ids})
    out = {int(x) for x in chart_ids}
    for i in list(out):
        if 0 <= i < len(overlap_graph):
            nbrs = overlap_graph[i]
            if isinstance(nbrs, list):
                out.update(int(x) for x in nbrs)
    return sorted(out)


def build_reuse_policy(
    *,
    seed_dir: str,
    atlas_data_use: str,
    atlas_data_ref: str,
    split_map_path: str,
    chart_score_path: str,
    hotspot: Dict[str, object],
    retrain_topk: int,
    include_neighbor_repair: bool,
) -> Dict[str, object]:
    atlas = np.load(atlas_data_use)
    n_charts = int(np.asarray(atlas["membership"]).shape[1])
    trainable: List[int] = []
    remap_path: Optional[str] = None

    hotspot_old = [int(x) for x in hotspot.get("hotspot_chart_ids", [])] if isinstance(hotspot.get("hotspot_chart_ids"), list) else []
    top_old = read_top_chart_ids_from_scores(chart_score_path, topk=retrain_topk)

    if atlas_data_use == atlas_data_ref and os.path.isfile(split_map_path):
        sm = read_json(split_map_path)
        split_parents = [int(x) for x in sm.get("split_parents", [])] if isinstance(sm.get("split_parents"), list) else []
        split_children = map_old_to_new_ids(split_parents, sm)
        hotspot_new = map_old_to_new_ids(hotspot_old, sm)
        top_new = map_old_to_new_ids(top_old, sm)
        trainable = sorted({*split_children, *hotspot_new, *top_new})
        if include_neighbor_repair:
            trainable = expand_with_neighbors(trainable, sm.get("overlap_graph"))
        remap_path = split_map_path
    else:
        trainable = sorted({*hotspot_old, *top_old})

    trainable = [i for i in trainable if 0 <= int(i) < n_charts]
    if len(trainable) == 0:
        trainable = list(range(n_charts))
    freeze = [i for i in range(n_charts) if i not in set(trainable)]

    trainable_path = os.path.join(seed_dir, "reuse_trainable_charts.json")
    freeze_path = os.path.join(seed_dir, "reuse_frozen_charts.json")
    write_json(
        trainable_path,
        {
            "chart_ids": [int(i) for i in trainable],
            "n_charts": int(n_charts),
        },
    )
    write_json(
        freeze_path,
        {
            "chart_ids": [int(i) for i in freeze],
            "n_charts": int(n_charts),
        },
    )
    summary = {
        "atlas_data_use": atlas_data_use,
        "n_charts": int(n_charts),
        "trainable_count": int(len(trainable)),
        "frozen_count": int(len(freeze)),
        "trainable_path": trainable_path,
        "freeze_path": freeze_path,
        "u_remap_json": remap_path,
    }
    write_json(os.path.join(seed_dir, "reuse_policy_summary.json"), summary)
    return summary


@dataclass
class SolverRunResult:
    metrics_path: str
    history_path: str
    solution_npz: str
    metrics: Dict[str, object]
    log_path: str


def run_solver(
    *,
    python_bin: str,
    solver_script: str,
    output_dir: str,
    run_tag: str,
    atlas_data: str,
    atlas_checkpoint: str,
    atlas_meta: Optional[str],
    init_u_checkpoint: Optional[str],
    seed: int,
    common_args: List[str],
    phase_args: List[str],
    cwd: str,
    log_name: str,
) -> SolverRunResult:
    cmd = [
        python_bin,
        "-u",
        solver_script,
        "--atlas-data",
        atlas_data,
        "--atlas-checkpoint",
        atlas_checkpoint,
        "--output-dir",
        output_dir,
        "--run-tag",
        run_tag,
        "--seed",
        str(seed),
    ]
    if atlas_meta:
        cmd.extend(["--atlas-meta", atlas_meta])
    if init_u_checkpoint:
        cmd.extend(["--init-u-checkpoint", init_u_checkpoint])
    cmd.extend(common_args)
    cmd.extend(phase_args)

    log_path = os.path.join(output_dir, log_name)
    code = run_cmd(cmd, log_path=log_path, cwd=cwd)
    if code != 0:
        raise RuntimeError(f"Solver run failed ({run_tag}), see {log_path}")

    stem = build_run_stem(run_tag)
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.json")
    history_path = os.path.join(output_dir, f"{stem}_history.json")
    solution_npz = os.path.join(output_dir, f"{stem}_solution.npz")
    if not os.path.isfile(metrics_path):
        raise RuntimeError(f"Missing metrics: {metrics_path}")
    if not os.path.isfile(solution_npz):
        raise RuntimeError(f"Missing solution: {solution_npz}")
    return SolverRunResult(
        metrics_path=metrics_path,
        history_path=history_path,
        solution_npz=solution_npz,
        metrics=read_json(metrics_path),
        log_path=log_path,
    )


def metric_rel_l2(metrics: Dict[str, object]) -> float:
    g = metrics.get("global", {})
    if isinstance(g, dict) and g.get("relative_l2_error") is not None:
        return float(g["relative_l2_error"])
    return float("inf")


def metric_interface_flux(metrics: Dict[str, object]) -> float:
    v = metrics.get("final_interface_flux")
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def hotspot_worst_if_val_rms(hotspot: Dict[str, object]) -> float:
    top = hotspot.get("top_pairs")
    if not isinstance(top, list) or len(top) == 0:
        return float("nan")
    try:
        return float(top[0].get("if_val_rms", float("nan")))
    except (TypeError, ValueError, AttributeError):
        return float("nan")


def is_finite(x: float) -> bool:
    return bool(np.isfinite(float(x)))


def summarize_seed_runs(seed_runs: List[Dict[str, object]]) -> Dict[str, object]:
    rels = [float(r["best_refined_rel_l2"]) for r in seed_runs if r.get("best_refined_rel_l2") is not None]
    if len(rels) == 0:
        return {"n": 0}
    base = [float(r["baseline_rel_l2"]) for r in seed_runs]
    nonneg = [1 if float(r["delta_rel_l2"]) <= 0.0 else 0 for r in seed_runs]
    out = {
        "n": len(seed_runs),
        "baseline_rel_l2_mean": float(statistics.mean(base)),
        "baseline_rel_l2_std": float(statistics.pstdev(base) if len(base) > 1 else 0.0),
        "refined_rel_l2_mean": float(statistics.mean(rels)),
        "refined_rel_l2_std": float(statistics.pstdev(rels) if len(rels) > 1 else 0.0),
        "nonnegative_gain_count": int(sum(nonneg)),
        "nonnegative_gain_ratio": float(sum(nonneg) / len(nonneg)),
    }
    worstpair_vals = [
        float(r["worstpair_improvement_ratio"])
        for r in seed_runs
        if r.get("worstpair_improvement_ratio") is not None and is_finite(float(r["worstpair_improvement_ratio"]))
    ]
    flux_vals = [
        float(r["flux_regression_ratio"])
        for r in seed_runs
        if r.get("flux_regression_ratio") is not None and is_finite(float(r["flux_regression_ratio"]))
    ]
    if worstpair_vals:
        out["worstpair_improvement_mean"] = float(statistics.mean(worstpair_vals))
        out["worstpair_improvement_std"] = float(statistics.pstdev(worstpair_vals) if len(worstpair_vals) > 1 else 0.0)
    if flux_vals:
        out["flux_regression_ratio_mean"] = float(statistics.mean(flux_vals))
        out["flux_regression_ratio_std"] = float(statistics.pstdev(flux_vals) if len(flux_vals) > 1 else 0.0)
    return out


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(os.path.dirname(here))
    repo_root = os.path.dirname(experiments_dir)
    parser = argparse.ArgumentParser(description="Adaptive split Poisson rabbit pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=False, default=None)

    parser.add_argument("--atlas-data", required=False, default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_data.npz"))
    parser.add_argument("--atlas-checkpoint", required=False, default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_trained.pt"))
    parser.add_argument("--atlas-meta", required=False, default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_meta.json"))

    parser.add_argument("--solver-script", default=os.path.join(experiments_dir, "run_poisson_rabbit_atlas_schwarz.py"))
    parser.add_argument("--split-build-script", default=os.path.join(here, "build_rabbit_atlas_adaptive_split.py"))
    parser.add_argument("--atlas-train-script", default=os.path.join(here, "train_rabbit_atlas_warmstart.py"))

    parser.add_argument("--seed-list", default="42", help="Comma-separated seeds, e.g. 42,52,62")
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--k-split", type=int, default=2)
    parser.add_argument("--n-children-per-parent", type=int, default=2)
    parser.add_argument("--min-points-to-split", type=int, default=1200)
    parser.add_argument("--split-radius-scale", type=float, default=0.72)
    parser.add_argument("--max-charts", type=int, default=20)

    parser.add_argument("--run-split-only", action="store_true")
    parser.add_argument("--run-full-hotspot", action="store_true")
    parser.add_argument("--fallback-on-gate-fail", dest="fallback_on_gate_fail", action="store_true")
    parser.add_argument("--no-fallback-on-gate-fail", dest="fallback_on_gate_fail", action="store_false")
    parser.set_defaults(fallback_on_gate_fail=True)

    parser.add_argument("--hotspot-alpha", type=float, default=0.35)
    parser.add_argument("--hotspot-top-pairs", type=int, default=4)
    parser.add_argument("--hotspot-min-shared", type=int, default=20)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--r-hot", type=float, default=2.5)
    parser.add_argument("--max-if-value-mult", type=float, default=3.0)
    parser.add_argument("--max-if-flux-mult", type=float, default=4.0)
    parser.add_argument("--max-robin-mult", type=float, default=3.0)
    parser.add_argument("--max-detail-boost-mult", type=float, default=3.0)

    parser.add_argument("--w-interface-value-base", type=float, default=0.8)
    parser.add_argument("--w-interface-flux-base", type=float, default=0.2)
    parser.add_argument("--robin-lambda-base", type=float, default=5.0)
    parser.add_argument("--if-batch-base", type=int, default=160)
    parser.add_argument("--detail-chart-topk-base", type=int, default=4)
    parser.add_argument("--interface-detail-boost-base", type=float, default=1.0)

    parser.add_argument("--atlas-warmstart-epochs", type=int, default=900)
    parser.add_argument("--atlas-warmstart-lr", type=float, default=5e-4)
    parser.add_argument("--atlas-warmstart-log-every", type=int, default=100)

    parser.add_argument("--solver-common-args", default="")
    parser.add_argument("--solver-baseline-args", default="")
    parser.add_argument("--solver-refined-split-args", default="")
    parser.add_argument("--solver-refined-full-args", default="")

    parser.add_argument("--run-tag-prefix", default="adaptive")
    parser.add_argument("--accept-rel-improve-abs", type=float, default=0.007)
    parser.add_argument("--accept-rel-improve-ratio", type=float, default=0.05)
    parser.add_argument("--accept-worstpair-improve", type=float, default=0.15)
    parser.add_argument("--accept-flux-regression-max", type=float, default=0.10)
    parser.add_argument("--target-rel-l2", type=float, default=0.15)
    parser.add_argument("--retrain-topk", type=int, default=2, help="Additional top-score charts to retrain.")
    parser.add_argument(
        "--include-neighbor-repair",
        dest="include_neighbor_repair",
        action="store_true",
        help="Expand retrain set with 1-ring overlap neighbors for interface repair.",
    )
    parser.add_argument(
        "--no-include-neighbor-repair",
        dest="include_neighbor_repair",
        action="store_false",
        help="Disable retrain-set neighbor expansion.",
    )
    parser.set_defaults(include_neighbor_repair=True)

    args = parser.parse_args()
    defaults = parser.parse_args([])
    if args.config is not None:
        cfg = parse_simple_yaml(args.config)
        for k, v in cfg.items():
            attr = k.replace("-", "_")
            if not hasattr(args, attr):
                continue
            # Keep explicit CLI args higher priority than config values.
            if getattr(args, attr) != getattr(defaults, attr):
                continue
            cur = getattr(args, attr)
            if isinstance(cur, bool):
                val = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(v)
            elif isinstance(cur, float):
                val = float(v)
            else:
                val = v
            setattr(args, attr, val)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(repo_root, "runs", f"atlas_schwarz_adaptive_{ts}")
    if not args.run_split_only and not args.run_full_hotspot:
        args.run_split_only = True
        args.run_full_hotspot = True
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    if not seeds:
        raise RuntimeError("seed-list is empty")

    common_args = shlex.split(args.solver_common_args)
    baseline_args = shlex.split(args.solver_baseline_args)
    refined_split_args = shlex.split(args.solver_refined_split_args)
    refined_full_args = shlex.split(args.solver_refined_full_args)

    seed_reports: List[Dict[str, object]] = []

    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n=== Seed {seed} ===")

        # Stage 1: baseline solve.
        baseline_tag = f"{args.run_tag_prefix}_baseline_s{seed}"
        baseline = run_solver(
            python_bin=args.python_bin,
            solver_script=args.solver_script,
            output_dir=seed_dir,
            run_tag=baseline_tag,
            atlas_data=args.atlas_data,
            atlas_checkpoint=args.atlas_checkpoint,
            atlas_meta=args.atlas_meta,
            init_u_checkpoint=None,
            seed=seed,
            common_args=common_args,
            phase_args=baseline_args,
            cwd=cwd,
            log_name=f"{baseline_tag}.log",
        )
        chart_score = score_charts_from_metrics(baseline.metrics_path)
        chart_score_path = os.path.join(seed_dir, f"{baseline_tag}_chart_scores.json")
        write_json(chart_score_path, chart_score)

        hotspot = compute_interface_hotspots(
            solution_npz=baseline.solution_npz,
            atlas_data_npz=args.atlas_data,
            alpha=args.hotspot_alpha,
            top_pairs=args.hotspot_top_pairs,
            min_shared=args.hotspot_min_shared,
        )
        hotspot_path = os.path.join(seed_dir, f"{baseline_tag}_interface_hotspots.json")
        write_json(hotspot_path, hotspot)

        # Stage 2: split atlas.
        split_dir = os.path.join(seed_dir, "split")
        os.makedirs(split_dir, exist_ok=True)
        split_cmd = [
            args.python_bin,
            "-u",
            args.split_build_script,
            "--atlas-data",
            args.atlas_data,
            "--atlas-meta",
            args.atlas_meta,
            "--baseline-metrics",
            baseline.metrics_path,
            "--chart-score-json",
            chart_score_path,
            "--output-dir",
            split_dir,
            "--k-split",
            str(args.k_split),
            "--n-children-per-parent",
            str(args.n_children_per_parent),
            "--min-points-to-split",
            str(args.min_points_to_split),
            "--split-radius-scale",
            str(args.split_radius_scale),
            "--max-charts",
            str(args.max_charts),
            "--seed",
            str(seed),
        ]
        split_log = os.path.join(seed_dir, "split_build.log")
        if run_cmd(split_cmd, split_log, cwd=cwd) != 0:
            raise RuntimeError(f"Split build failed for seed {seed}. See {split_log}")
        atlas_data_ref = os.path.join(split_dir, "rabbit_atlas_data_refined.npz")
        atlas_meta_ref = os.path.join(split_dir, "rabbit_atlas_meta_refined.json")
        split_map = os.path.join(split_dir, "atlas_split_map.json")

        # Stage 3: warmstart atlas retrain.
        atlas_train_dir = os.path.join(seed_dir, "atlas_warmstart")
        os.makedirs(atlas_train_dir, exist_ok=True)
        train_cmd = [
            args.python_bin,
            "-u",
            args.atlas_train_script,
            "--atlas-data",
            atlas_data_ref,
            "--output-dir",
            atlas_train_dir,
            "--init-atlas-checkpoint",
            args.atlas_checkpoint,
            "--split-map-json",
            split_map,
            "--epochs",
            str(args.atlas_warmstart_epochs),
            "--lr",
            str(args.atlas_warmstart_lr),
            "--seed",
            str(seed),
            "--log-every",
            str(args.atlas_warmstart_log_every),
        ]
        train_log = os.path.join(seed_dir, "atlas_warmstart.log")
        if run_cmd(train_cmd, train_log, cwd=cwd) != 0:
            raise RuntimeError(f"Warmstart atlas training failed for seed {seed}. See {train_log}")
        atlas_ckpt_ref = os.path.join(atlas_train_dir, "rabbit_atlas_trained.pt")
        atlas_gate_ref = os.path.join(atlas_train_dir, "rabbit_atlas_gate_report.json")
        gate = read_json(atlas_gate_ref) if os.path.isfile(atlas_gate_ref) else {}
        atlas_ref_ok = bool(gate.get("passed", False))
        atlas_data_use = atlas_data_ref
        atlas_ckpt_use = atlas_ckpt_ref
        atlas_meta_use = atlas_meta_ref
        refine_failed_gate = False
        if (not atlas_ref_ok) and bool(args.fallback_on_gate_fail):
            refine_failed_gate = True
            atlas_data_use = args.atlas_data
            atlas_ckpt_use = args.atlas_checkpoint
            atlas_meta_use = args.atlas_meta

        # Warmstart U checkpoint from baseline best_rel_l2 if available.
        baseline_ckpts = baseline.metrics.get("checkpoint_paths", {})
        init_u_ckpt = None
        if isinstance(baseline_ckpts, dict):
            init_u_ckpt = baseline_ckpts.get("best_rel_l2") or baseline_ckpts.get("selected")

        split_only: Optional[SolverRunResult] = None
        full_refine: Optional[SolverRunResult] = None
        split_only_hotspot: Optional[Dict[str, object]] = None
        full_refine_hotspot: Optional[Dict[str, object]] = None
        split_only_hotspot_path: Optional[str] = None
        full_refine_hotspot_path: Optional[str] = None

        reuse_policy = build_reuse_policy(
            seed_dir=seed_dir,
            atlas_data_use=atlas_data_use,
            atlas_data_ref=atlas_data_ref,
            split_map_path=split_map,
            chart_score_path=chart_score_path,
            hotspot=hotspot,
            retrain_topk=args.retrain_topk,
            include_neighbor_repair=bool(args.include_neighbor_repair),
        )
        reuse_solver_args = [
            "--trainable-charts-json",
            str(reuse_policy["trainable_path"]),
            "--freeze-charts-json",
            str(reuse_policy["freeze_path"]),
        ]
        remap_path = reuse_policy.get("u_remap_json")
        if isinstance(remap_path, str) and len(remap_path) > 0:
            reuse_solver_args.extend(["--u-remap-json", remap_path])

        if args.run_split_only:
            tag = f"{args.run_tag_prefix}_refined_split_s{seed}"
            split_only = run_solver(
                python_bin=args.python_bin,
                solver_script=args.solver_script,
                output_dir=seed_dir,
                run_tag=tag,
                atlas_data=atlas_data_use,
                atlas_checkpoint=atlas_ckpt_use,
                atlas_meta=atlas_meta_use,
                init_u_checkpoint=init_u_ckpt,
                seed=seed,
                common_args=common_args,
                phase_args=list(refined_split_args) + list(reuse_solver_args),
                cwd=cwd,
                log_name=f"{tag}.log",
            )
            split_only_hotspot = compute_interface_hotspots(
                solution_npz=split_only.solution_npz,
                atlas_data_npz=atlas_data_use,
                alpha=args.hotspot_alpha,
                top_pairs=args.hotspot_top_pairs,
                min_shared=args.hotspot_min_shared,
            )
            split_only_hotspot_path = os.path.join(seed_dir, f"{tag}_interface_hotspots.json")
            write_json(split_only_hotspot_path, split_only_hotspot)

        hotspot_cfg = derive_hotspot_overrides(hotspot, args)
        hotspot_cfg_path = os.path.join(seed_dir, f"{baseline_tag}_hotspot_strategy.json")
        write_json(hotspot_cfg_path, hotspot_cfg)

        if args.run_full_hotspot:
            tag = f"{args.run_tag_prefix}_refined_full_s{seed}"
            phase_args = list(refined_full_args) + list(reuse_solver_args) + list(hotspot_cfg.get("solver_overrides", []))
            full_refine = run_solver(
                python_bin=args.python_bin,
                solver_script=args.solver_script,
                output_dir=seed_dir,
                run_tag=tag,
                atlas_data=atlas_data_use,
                atlas_checkpoint=atlas_ckpt_use,
                atlas_meta=atlas_meta_use,
                init_u_checkpoint=init_u_ckpt,
                seed=seed,
                common_args=common_args,
                phase_args=phase_args,
                cwd=cwd,
                log_name=f"{tag}.log",
            )
            full_refine_hotspot = compute_interface_hotspots(
                solution_npz=full_refine.solution_npz,
                atlas_data_npz=atlas_data_use,
                alpha=args.hotspot_alpha,
                top_pairs=args.hotspot_top_pairs,
                min_shared=args.hotspot_min_shared,
            )
            full_refine_hotspot_path = os.path.join(seed_dir, f"{tag}_interface_hotspots.json")
            write_json(full_refine_hotspot_path, full_refine_hotspot)

        candidates = []
        if split_only is not None:
            candidates.append(
                {
                    "name": "split_only",
                    "result": split_only,
                    "hotspot": split_only_hotspot,
                    "hotspot_path": split_only_hotspot_path,
                }
            )
        if full_refine is not None:
            candidates.append(
                {
                    "name": "full_hotspot",
                    "result": full_refine,
                    "hotspot": full_refine_hotspot,
                    "hotspot_path": full_refine_hotspot_path,
                }
            )
        if not candidates:
            raise RuntimeError("No refined run was executed.")
        best_entry = sorted(candidates, key=lambda r: metric_rel_l2(r["result"].metrics))[0]
        best = best_entry["result"]
        best_hotspot = best_entry["hotspot"] if isinstance(best_entry["hotspot"], dict) else {}
        best_hotspot_path = best_entry["hotspot_path"]

        baseline_rel = metric_rel_l2(baseline.metrics)
        best_rel = metric_rel_l2(best.metrics)
        delta = best_rel - baseline_rel
        rel_improve = (-delta / max(1e-12, baseline_rel))
        rel_improved = bool(delta < 0.0)

        baseline_if_flux = metric_interface_flux(baseline.metrics)
        best_if_flux = metric_interface_flux(best.metrics)
        flux_reg_ratio = float("nan")
        if is_finite(baseline_if_flux) and abs(baseline_if_flux) > 1e-12 and is_finite(best_if_flux):
            flux_reg_ratio = float((best_if_flux - baseline_if_flux) / abs(baseline_if_flux))
        flux_ok = (not rel_improved) or (not is_finite(flux_reg_ratio)) or (flux_reg_ratio <= float(args.accept_flux_regression_max))

        baseline_worstpair = hotspot_worst_if_val_rms(hotspot)
        best_worstpair = hotspot_worst_if_val_rms(best_hotspot)
        worstpair_impr = float("nan")
        if is_finite(baseline_worstpair) and baseline_worstpair > 1e-12 and is_finite(best_worstpair):
            worstpair_impr = float((baseline_worstpair - best_worstpair) / baseline_worstpair)
        worstpair_ok = is_finite(worstpair_impr) and (worstpair_impr >= float(args.accept_worstpair_improve))

        seed_report = {
            "seed": int(seed),
            "baseline_metrics_path": baseline.metrics_path,
            "baseline_solution_npz": baseline.solution_npz,
            "baseline_rel_l2": baseline_rel,
            "baseline_target_met": baseline.metrics.get("target_met"),
            "chart_scores_path": chart_score_path,
            "hotspot_path": hotspot_path,
            "hotspot_strategy_path": hotspot_cfg_path,
            "split_only_hotspot_path": split_only_hotspot_path,
            "full_refined_hotspot_path": full_refine_hotspot_path,
            "best_refined_hotspot_path": best_hotspot_path,
            "atlas_split_dir": split_dir,
            "atlas_warmstart_dir": atlas_train_dir,
            "atlas_refined_gate_passed": atlas_ref_ok,
            "refine_failed_gate": refine_failed_gate,
            "split_only_metrics_path": split_only.metrics_path if split_only else None,
            "full_refined_metrics_path": full_refine.metrics_path if full_refine else None,
            "best_refined_variant": best_entry["name"],
            "best_refined_metrics_path": best.metrics_path,
            "best_refined_rel_l2": best_rel,
            "delta_rel_l2": delta,
            "relative_improvement": rel_improve,
            "reuse_policy_summary": os.path.join(seed_dir, "reuse_policy_summary.json"),
            "baseline_interface_flux": baseline_if_flux if is_finite(baseline_if_flux) else None,
            "best_refined_interface_flux": best_if_flux if is_finite(best_if_flux) else None,
            "flux_regression_ratio": flux_reg_ratio if is_finite(flux_reg_ratio) else None,
            "flux_regression_ok_if_rel_l2_improved": bool(flux_ok),
            "baseline_worstpair_if_val_rms": baseline_worstpair if is_finite(baseline_worstpair) else None,
            "best_worstpair_if_val_rms": best_worstpair if is_finite(best_worstpair) else None,
            "worstpair_improvement_ratio": worstpair_impr if is_finite(worstpair_impr) else None,
            "worstpair_improve_ok": bool(worstpair_ok),
        }
        seed_reports.append(seed_report)
        comparison_path = os.path.join(seed_dir, f"{args.run_tag_prefix}_comparison_seed{seed}.json")
        write_json(
            comparison_path,
            {
                "seed": int(seed),
                "baseline": baseline.metrics,
                "refined_split": split_only.metrics if split_only else None,
                "refined_full": full_refine.metrics if full_refine else None,
                "best_refined_variant": best_entry["name"],
                "best_refined_metrics": best.metrics,
                "delta_rel_l2": delta,
                "relative_improvement": rel_improve,
                "baseline_hotspot": hotspot,
                "best_refined_hotspot": best_hotspot,
            },
        )
        seed_report["comparison_path"] = comparison_path

    # Aggregate and acceptance.
    summary = summarize_seed_runs(seed_reports)

    # Acceptance checks on best-per-seed reports.
    deltas = [float(r["delta_rel_l2"]) for r in seed_reports]
    gains = [(-d / max(1e-12, float(r["baseline_rel_l2"]))) for d, r in zip(deltas, seed_reports)]
    gain_abs_ok = sum(1 for d in deltas if d <= -float(args.accept_rel_improve_abs))
    gain_ratio_ok = sum(1 for g in gains if g >= float(args.accept_rel_improve_ratio))
    nonneg = sum(1 for d in deltas if d <= 0.0)
    maj_ok = nonneg >= max(1, math.ceil((2.0 / 3.0) * len(seed_reports)))
    worstpair_ok_count = sum(1 for r in seed_reports if bool(r.get("worstpair_improve_ok", False)))
    rel_improved_count = sum(1 for r in seed_reports if float(r.get("delta_rel_l2", 0.0)) < 0.0)
    flux_guard_count = sum(
        1
        for r in seed_reports
        if (float(r.get("delta_rel_l2", 0.0)) >= 0.0) or bool(r.get("flux_regression_ok_if_rel_l2_improved", False))
    )

    acceptance = {
        "n_seeds": len(seed_reports),
        "abs_improve_threshold": float(args.accept_rel_improve_abs),
        "ratio_improve_threshold": float(args.accept_rel_improve_ratio),
        "worstpair_improve_threshold": float(args.accept_worstpair_improve),
        "flux_regression_max": float(args.accept_flux_regression_max),
        "abs_improve_count": int(gain_abs_ok),
        "ratio_improve_count": int(gain_ratio_ok),
        "majority_nonnegative_gain": bool(maj_ok),
        "majority_nonnegative_gain_count": int(nonneg),
        "worstpair_improve_count": int(worstpair_ok_count),
        "rel_l2_improved_count": int(rel_improved_count),
        "flux_regression_guard_count": int(flux_guard_count),
        "worstpair_majority_ok": bool(worstpair_ok_count >= max(1, math.ceil((2.0 / 3.0) * len(seed_reports)))),
        "flux_guard_all_ok_when_rel_l2_improved": bool(flux_guard_count == len(seed_reports)),
    }

    out = {
        "config": vars(args),
        "seed_reports": seed_reports,
        "aggregate": summary,
        "acceptance": acceptance,
    }
    out_path = os.path.join(args.output_dir, "adaptive_split_summary.json")
    write_json(out_path, out)

    print("\nAdaptive split pipeline complete")
    print(f"  output_dir: {args.output_dir}")
    print(f"  summary:    {out_path}")
    print(f"  seeds:      {len(seed_reports)}")
    if len(seed_reports) > 0:
        print(f"  baseline rel-L2 mean: {summary.get('baseline_rel_l2_mean')}")
        print(f"  refined  rel-L2 mean: {summary.get('refined_rel_l2_mean')}")


if __name__ == "__main__":
    main()
