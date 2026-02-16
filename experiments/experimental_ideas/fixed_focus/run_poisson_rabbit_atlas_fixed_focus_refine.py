#!/usr/bin/env python3
"""
Fixed-atlas rabbit Poisson pipeline:
baseline -> worst-chart focus refinement -> interface hotspot repair.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


def chart_rel_error_map(metrics: Dict[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    pcs = metrics.get("per_chart")
    if not isinstance(pcs, list):
        return out
    for row in pcs:
        if not isinstance(row, dict):
            continue
        cid = row.get("chart_id")
        rel = row.get("relative_l2_error")
        if cid is None or rel is None:
            continue
        try:
            out[int(cid)] = float(rel)
        except (TypeError, ValueError):
            continue
    return out


def rank_worst_charts(metrics: Dict[str, object], k_worst: int) -> List[int]:
    cmap = chart_rel_error_map(metrics)
    if not cmap:
        return []
    ranked = sorted(cmap.items(), key=lambda kv: kv[1], reverse=True)
    return [int(cid) for cid, _ in ranked[: max(1, int(k_worst))]]


def find_flag_value(tokens: Sequence[str], flag: str) -> Optional[str]:
    out: Optional[str] = None
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            out = tokens[i + 1]
    return out


def find_flag_float(tokens: Sequence[str], flag: str, default: float) -> float:
    raw = find_flag_value(tokens, flag)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def sanitize_overlap_graph(graph_obj: object, n_charts: int) -> List[List[int]]:
    graph: List[List[int]] = [[] for _ in range(n_charts)]
    if not isinstance(graph_obj, list):
        return graph
    for i in range(min(n_charts, len(graph_obj))):
        nbrs = graph_obj[i]
        if not isinstance(nbrs, list):
            continue
        vals = sorted({int(x) for x in nbrs if 0 <= int(x) < n_charts and int(x) != i})
        graph[i] = vals
    return graph


def compute_overlap_graph_from_membership(atlas_data: str, min_shared: int) -> List[List[int]]:
    atlas = np.load(atlas_data)
    membership = np.asarray(atlas["membership"]).astype(bool)
    n_points, n_charts = membership.shape
    _ = n_points
    graph: List[List[int]] = [[] for _ in range(n_charts)]
    for i in range(n_charts):
        mi = membership[:, i]
        for j in range(i + 1, n_charts):
            n_shared = int(np.sum(mi & membership[:, j]))
            if n_shared >= int(min_shared):
                graph[i].append(j)
                graph[j].append(i)
    for i in range(n_charts):
        graph[i] = sorted(graph[i])
    return graph


def load_overlap_graph(atlas_data: str, atlas_meta: Optional[str], min_shared: int) -> List[List[int]]:
    n_charts = int(np.asarray(np.load(atlas_data)["membership"]).shape[1])
    if atlas_meta and os.path.isfile(atlas_meta):
        meta = read_json(atlas_meta)
        graph = sanitize_overlap_graph(meta.get("overlap_graph"), n_charts=n_charts)
        if any(len(v) > 0 for v in graph):
            return graph
    return compute_overlap_graph_from_membership(atlas_data=atlas_data, min_shared=min_shared)


def expand_neighbors(roots: Sequence[int], overlap_graph: List[List[int]], depth: int) -> List[int]:
    if not roots:
        return []
    cur: Set[int] = {int(i) for i in roots if 0 <= int(i) < len(overlap_graph)}
    seen: Set[int] = set(cur)
    for _ in range(max(0, int(depth))):
        nxt: Set[int] = set()
        for i in cur:
            nxt.update(overlap_graph[i])
        nxt -= seen
        if not nxt:
            break
        seen.update(nxt)
        cur = nxt
    return sorted(seen)


def write_focus_sets(seed_dir: str, n_charts: int, trainable_ids: Sequence[int]) -> Tuple[str, str, List[int], List[int]]:
    trainable = sorted({int(i) for i in trainable_ids if 0 <= int(i) < int(n_charts)})
    if len(trainable) == 0:
        trainable = [0]
    freeze = [i for i in range(n_charts) if i not in set(trainable)]
    trainable_path = os.path.join(seed_dir, "focus_trainable_charts.json")
    freeze_path = os.path.join(seed_dir, "focus_frozen_charts.json")
    write_json(trainable_path, {"chart_ids": trainable, "n_charts": int(n_charts)})
    write_json(freeze_path, {"chart_ids": freeze, "n_charts": int(n_charts)})
    return trainable_path, freeze_path, trainable, freeze


def compute_interface_hotspots(
    *,
    solution_npz: str,
    atlas_data_npz: str,
    touching_ids: Sequence[int],
    top_pairs: int,
    min_shared: int,
) -> Dict[str, object]:
    sol = np.load(solution_npz)
    atlas = np.load(atlas_data_npz)
    vals = np.asarray(sol["chart_values"], dtype=float)
    membership = np.asarray(atlas["membership"]).astype(bool)
    support_r = np.asarray(atlas["support_radii"], dtype=float)
    _, n_charts = membership.shape

    touch = {int(i) for i in touching_ids if 0 <= int(i) < n_charts}
    pairs: List[Dict[str, object]] = []
    for i in range(n_charts):
        mi = membership[:, i]
        for j in range(i + 1, n_charts):
            if touch and (i not in touch) and (j not in touch):
                continue
            shared = mi & membership[:, j]
            n = int(np.sum(shared))
            if n < int(min_shared):
                continue
            d = vals[shared, i] - vals[shared, j]
            if_val_rms = float(np.sqrt(np.mean(d**2)))
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

    if len(pairs) == 0:
        return {"pairs": [], "top_pairs": [], "severity": 0.0, "touching_ids": sorted(touch)}

    max_val = max(max(float(p["if_val_rms"]), 1e-12) for p in pairs)
    max_flux = max(max(float(p["if_flux_rms_proxy"]), 1e-12) for p in pairs)
    for p in pairs:
        nv = float(p["if_val_rms"]) / max_val
        nf = float(p["if_flux_rms_proxy"]) / max_flux
        p["score"] = 0.35 * nv + 0.65 * nf

    pairs = sorted(pairs, key=lambda r: float(r["score"]), reverse=True)
    k = max(1, min(int(top_pairs), len(pairs)))
    top = pairs[:k]
    sev = float(np.mean([float(p["score"]) for p in top])) if top else 0.0
    return {
        "pairs": pairs,
        "top_pairs": top,
        "severity": float(np.clip(sev, 0.0, 1.0)),
        "touching_ids": sorted(touch),
        "if_flux_note": "proxy_from_value_mismatch_over_support_radius",
    }


def choose_checkpoint_path(metrics: Dict[str, object], key_order: Sequence[str]) -> Optional[str]:
    ckpts = metrics.get("checkpoint_paths", {})
    if not isinstance(ckpts, dict):
        return None
    for k in key_order:
        v = ckpts.get(k)
        if isinstance(v, str) and os.path.isfile(v):
            return v
    return None


@dataclass
class SolverRunResult:
    run_tag: str
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
        run_tag=run_tag,
        metrics_path=metrics_path,
        history_path=history_path,
        solution_npz=solution_npz,
        metrics=read_json(metrics_path),
        log_path=log_path,
    )


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(os.path.dirname(here))
    repo_root = os.path.dirname(experiments_dir)
    parser = argparse.ArgumentParser(description="Fixed-atlas rabbit worst-chart refinement pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--atlas-data", default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_data.npz"))
    parser.add_argument("--atlas-checkpoint", default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_trained.pt"))
    parser.add_argument("--atlas-meta", default=os.path.join(repo_root, "runs/atlas_schwarz_20260213_005232/rabbit_atlas_meta.json"))
    parser.add_argument("--solver-script", default=os.path.join(experiments_dir, "run_poisson_rabbit_atlas_schwarz.py"))

    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed-list", default="42")
    parser.add_argument("--run-tag-prefix", default="fixed_focus")

    parser.add_argument("--k-worst", type=int, default=1)
    parser.add_argument("--neighbor-depth", type=int, default=1)
    parser.add_argument("--overlap-min-shared", type=int, default=20)
    parser.add_argument("--hotspot-top-pairs", type=int, default=4)

    parser.add_argument("--stage-c-lr-mult", type=float, default=0.5)
    parser.add_argument("--stage-d-lr-mult", type=float, default=0.6)
    parser.add_argument("--stage-d-if-value-mult", type=float, default=1.7)
    parser.add_argument("--stage-d-if-flux-mult", type=float, default=2.2)
    parser.add_argument("--stage-d-if-batch-mult", type=float, default=1.8)
    parser.add_argument("--stage-d-robin-mult", type=float, default=1.4)

    parser.add_argument("--allow-global-regression", type=float, default=0.005)
    parser.add_argument("--required-worst-chart-improvement", type=float, default=0.20)
    parser.add_argument("--max-interface-flux-regression", type=float, default=0.10)

    parser.add_argument("--solver-common-args", default="")
    parser.add_argument("--solver-baseline-args", default="")
    parser.add_argument("--solver-stage-c-args", default="")
    parser.add_argument("--solver-stage-d-args", default="")

    args = parser.parse_args()
    defaults = parser.parse_args([])
    if args.config is not None:
        cfg = parse_simple_yaml(args.config)
        for k, v in cfg.items():
            attr = k.replace("-", "_")
            if not hasattr(args, attr):
                continue
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
        args.output_dir = os.path.join(repo_root, "runs", f"atlas_fixed_focus_{ts}")
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
    stage_c_args = shlex.split(args.solver_stage_c_args)
    stage_d_args = shlex.split(args.solver_stage_d_args)

    baseline_lr = find_flag_float(list(common_args) + list(baseline_args), "--lr", 2e-4)
    stage_c_lr = baseline_lr * float(args.stage_c_lr_mult)
    stage_d_lr = stage_c_lr * float(args.stage_d_lr_mult)

    aggregate: List[Dict[str, object]] = []

    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n=== Seed {seed} ===")

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

        per_chart_map = chart_rel_error_map(baseline.metrics)
        worst_roots = rank_worst_charts(baseline.metrics, k_worst=args.k_worst)
        if len(worst_roots) == 0 and per_chart_map:
            worst_roots = [max(per_chart_map.items(), key=lambda kv: kv[1])[0]]
        if len(worst_roots) == 0:
            worst_roots = [0]
        ranking = sorted(
            [{"chart_id": int(k), "relative_l2_error": float(v)} for k, v in per_chart_map.items()],
            key=lambda r: float(r["relative_l2_error"]),
            reverse=True,
        )
        ranking_path = os.path.join(seed_dir, f"{baseline_tag}_per_chart_rank.json")
        write_json(
            ranking_path,
            {
                "seed": int(seed),
                "worst_roots": [int(i) for i in worst_roots],
                "per_chart_rank": ranking,
            },
        )

        overlap_graph = load_overlap_graph(
            atlas_data=args.atlas_data,
            atlas_meta=args.atlas_meta,
            min_shared=args.overlap_min_shared,
        )
        n_charts = len(overlap_graph)
        focus_ids = expand_neighbors(worst_roots, overlap_graph=overlap_graph, depth=args.neighbor_depth)
        if len(focus_ids) == 0:
            focus_ids = list(worst_roots)
        trainable_json, freeze_json, trainable_ids, freeze_ids = write_focus_sets(
            seed_dir=seed_dir,
            n_charts=n_charts,
            trainable_ids=focus_ids,
        )
        write_json(
            os.path.join(seed_dir, f"{args.run_tag_prefix}_focus_policy_s{seed}.json"),
            {
                "n_charts": int(n_charts),
                "worst_roots": [int(i) for i in worst_roots],
                "neighbor_depth": int(args.neighbor_depth),
                "trainable_ids": [int(i) for i in trainable_ids],
                "frozen_ids": [int(i) for i in freeze_ids],
                "overlap_graph": overlap_graph,
                "trainable_json": trainable_json,
                "freeze_json": freeze_json,
            },
        )

        baseline_ckpt = choose_checkpoint_path(
            baseline.metrics,
            key_order=["best_rel_l2", "selected", "best_target", "best_score"],
        )
        stage_c_phase = list(stage_c_args) + [
            "--trainable-charts-json",
            trainable_json,
            "--freeze-charts-json",
            freeze_json,
            "--lr",
            f"{stage_c_lr:.8g}",
        ]
        stage_c_tag = f"{args.run_tag_prefix}_focus_s{seed}"
        stage_c = run_solver(
            python_bin=args.python_bin,
            solver_script=args.solver_script,
            output_dir=seed_dir,
            run_tag=stage_c_tag,
            atlas_data=args.atlas_data,
            atlas_checkpoint=args.atlas_checkpoint,
            atlas_meta=args.atlas_meta,
            init_u_checkpoint=baseline_ckpt,
            seed=seed,
            common_args=common_args,
            phase_args=stage_c_phase,
            cwd=cwd,
            log_name=f"{stage_c_tag}.log",
        )

        hotspot = compute_interface_hotspots(
            solution_npz=stage_c.solution_npz,
            atlas_data_npz=args.atlas_data,
            touching_ids=worst_roots,
            top_pairs=args.hotspot_top_pairs,
            min_shared=args.overlap_min_shared,
        )
        hotspot_path = os.path.join(seed_dir, f"{stage_c_tag}_hotspots.json")
        write_json(hotspot_path, hotspot)

        sev = float(hotspot.get("severity", 0.0))
        sev = float(np.clip(sev, 0.0, 1.0))
        stage_c_if_val = find_flag_float(list(common_args) + stage_c_phase, "--w-interface-value", 1.0)
        stage_c_if_flux = find_flag_float(list(common_args) + stage_c_phase, "--w-interface-flux", 0.2)
        stage_c_if_batch = find_flag_float(list(common_args) + stage_c_phase, "--if-batch", 192.0)
        stage_c_robin = find_flag_float(list(common_args) + stage_c_phase, "--robin-lambda", 5.0)

        stage_d_if_val = stage_c_if_val * (1.0 + (float(args.stage_d_if_value_mult) - 1.0) * sev)
        stage_d_if_flux = stage_c_if_flux * (1.0 + (float(args.stage_d_if_flux_mult) - 1.0) * sev)
        stage_d_if_batch = int(max(8, round(stage_c_if_batch * (1.0 + (float(args.stage_d_if_batch_mult) - 1.0) * sev))))
        stage_d_robin = stage_c_robin * (1.0 + (float(args.stage_d_robin_mult) - 1.0) * sev)

        stage_d_overrides = [
            "--w-interface-value",
            f"{stage_d_if_val:.8g}",
            "--w-interface-flux",
            f"{stage_d_if_flux:.8g}",
            "--if-batch",
            str(stage_d_if_batch),
            "--robin-lambda",
            f"{stage_d_robin:.8g}",
            "--lr",
            f"{stage_d_lr:.8g}",
        ]
        stage_d_phase = list(stage_d_args) + [
            "--trainable-charts-json",
            trainable_json,
            "--freeze-charts-json",
            freeze_json,
        ] + stage_d_overrides
        stage_d_ckpt = choose_checkpoint_path(
            stage_c.metrics,
            key_order=["best_rel_l2", "selected", "best_target", "best_score"],
        )
        stage_d_tag = f"{args.run_tag_prefix}_hotspot_s{seed}"
        stage_d = run_solver(
            python_bin=args.python_bin,
            solver_script=args.solver_script,
            output_dir=seed_dir,
            run_tag=stage_d_tag,
            atlas_data=args.atlas_data,
            atlas_checkpoint=args.atlas_checkpoint,
            atlas_meta=args.atlas_meta,
            init_u_checkpoint=stage_d_ckpt,
            seed=seed,
            common_args=common_args,
            phase_args=stage_d_phase,
            cwd=cwd,
            log_name=f"{stage_d_tag}.log",
        )

        variants: Dict[str, SolverRunResult] = {
            "baseline": baseline,
            "stage_c_focus": stage_c,
            "stage_d_hotspot": stage_d,
        }
        base_rel = metric_rel_l2(baseline.metrics)
        base_flux = metric_interface_flux(baseline.metrics)
        base_chart_map = chart_rel_error_map(baseline.metrics)
        base_worst = max([base_chart_map.get(int(i), float("inf")) for i in worst_roots]) if worst_roots else float("inf")

        variant_scores: Dict[str, Dict[str, float]] = {}
        for name, res in variants.items():
            cmap = chart_rel_error_map(res.metrics)
            worst_rel = max([cmap.get(int(i), float("inf")) for i in worst_roots]) if worst_roots else float("inf")
            variant_scores[name] = {
                "rel_l2": metric_rel_l2(res.metrics),
                "interface_flux": metric_interface_flux(res.metrics),
                "worst_chart_rel_l2": worst_rel,
            }

        best_rel_name = min(variant_scores.keys(), key=lambda n: variant_scores[n]["rel_l2"])
        best_worst_name = min(variant_scores.keys(), key=lambda n: variant_scores[n]["worst_chart_rel_l2"])
        best_if_name = min(variant_scores.keys(), key=lambda n: variant_scores[n]["interface_flux"])

        eligible: List[str] = []
        for name in ["stage_d_hotspot", "stage_c_focus"]:
            s = variant_scores[name]
            glob_ok = s["rel_l2"] <= (base_rel + float(args.allow_global_regression))
            if np.isfinite(base_worst) and base_worst > 1e-12 and np.isfinite(s["worst_chart_rel_l2"]):
                imp = (base_worst - s["worst_chart_rel_l2"]) / base_worst
            else:
                imp = float("-inf")
            worst_ok = imp >= float(args.required_worst_chart_improvement)
            if np.isfinite(base_flux) and abs(base_flux) > 1e-12 and np.isfinite(s["interface_flux"]):
                flux_ratio = (s["interface_flux"] - base_flux) / abs(base_flux)
            else:
                flux_ratio = float("inf")
            flux_ok = flux_ratio <= float(args.max_interface_flux_regression)
            if glob_ok and worst_ok and flux_ok:
                eligible.append(name)

        if eligible:
            final_name = min(eligible, key=lambda n: variant_scores[n]["rel_l2"])
        else:
            final_name = "baseline"

        triplet_dir = os.path.join(seed_dir, "triplet")
        os.makedirs(triplet_dir, exist_ok=True)

        def persist_triplet(name: str, label: str, ckpt_keys: Sequence[str]) -> Optional[str]:
            src = choose_checkpoint_path(variants[name].metrics, key_order=ckpt_keys)
            if src is None:
                return None
            dst = os.path.join(triplet_dir, f"{args.run_tag_prefix}_{label}_s{seed}.pt")
            shutil.copy2(src, dst)
            return dst

        triplet_paths = {
            "best_rel_l2": persist_triplet(best_rel_name, "best_rel_l2", ["best_rel_l2", "selected"]),
            "best_worst_chart": persist_triplet(best_worst_name, "best_worst_chart", ["best_rel_l2", "selected"]),
            "best_interface": persist_triplet(best_if_name, "best_interface", ["best_flux", "best_target", "selected"]),
            "final_selected": persist_triplet(final_name, "final_selected", ["selected", "best_rel_l2", "best_target"]),
        }

        summary_payload = {
            "seed": int(seed),
            "atlas_data": args.atlas_data,
            "atlas_checkpoint": args.atlas_checkpoint,
            "atlas_meta": args.atlas_meta,
            "worst_roots": [int(i) for i in worst_roots],
            "focus_trainable_ids": [int(i) for i in trainable_ids],
            "focus_frozen_ids": [int(i) for i in freeze_ids],
            "focus_trainable_json": trainable_json,
            "focus_frozen_json": freeze_json,
            "baseline_rank_path": ranking_path,
            "stage_c_hotspot_path": hotspot_path,
            "stage_d_overrides": {
                "severity": sev,
                "w_interface_value": stage_d_if_val,
                "w_interface_flux": stage_d_if_flux,
                "if_batch": stage_d_if_batch,
                "robin_lambda": stage_d_robin,
                "lr": stage_d_lr,
            },
            "runs": {
                k: {
                    "run_tag": v.run_tag,
                    "metrics_path": v.metrics_path,
                    "history_path": v.history_path,
                    "solution_npz": v.solution_npz,
                    "log_path": v.log_path,
                }
                for k, v in variants.items()
            },
            "scores": variant_scores,
            "baseline_reference": {
                "rel_l2": base_rel,
                "interface_flux": base_flux,
                "worst_chart_rel_l2": base_worst,
            },
            "triplet": {
                "best_rel_l2": {
                    "variant": best_rel_name,
                    "path": triplet_paths["best_rel_l2"],
                    "score": variant_scores[best_rel_name]["rel_l2"],
                },
                "best_worst_chart": {
                    "variant": best_worst_name,
                    "path": triplet_paths["best_worst_chart"],
                    "score": variant_scores[best_worst_name]["worst_chart_rel_l2"],
                },
                "best_interface": {
                    "variant": best_if_name,
                    "path": triplet_paths["best_interface"],
                    "score": variant_scores[best_if_name]["interface_flux"],
                },
            },
            "acceptance_thresholds": {
                "allow_global_regression": float(args.allow_global_regression),
                "required_worst_chart_improvement": float(args.required_worst_chart_improvement),
                "max_interface_flux_regression": float(args.max_interface_flux_regression),
            },
            "eligible_variants": eligible,
            "final_selected_variant": final_name,
            "final_selected_checkpoint": triplet_paths["final_selected"],
        }
        summary_path = os.path.join(seed_dir, f"{args.run_tag_prefix}_comparison_seed{seed}.json")
        write_json(summary_path, summary_payload)
        print(f"Seed {seed} comparison summary: {summary_path}")

        aggregate.append(
            {
                "seed": int(seed),
                "baseline_rel_l2": base_rel,
                "final_selected_variant": final_name,
                "final_selected_rel_l2": float(variant_scores[final_name]["rel_l2"]),
                "final_selected_worst_chart_rel_l2": float(variant_scores[final_name]["worst_chart_rel_l2"]),
                "final_selected_interface_flux": float(variant_scores[final_name]["interface_flux"]),
                "comparison_path": summary_path,
            }
        )

    agg_path = os.path.join(args.output_dir, f"{args.run_tag_prefix}_summary.json")
    rels = [float(x["final_selected_rel_l2"]) for x in aggregate]
    aggregate_out = {
        "output_dir": args.output_dir,
        "n_seeds": len(aggregate),
        "seeds": aggregate,
        "final_rel_l2_mean": float(statistics.mean(rels)) if rels else None,
        "final_rel_l2_std": float(statistics.pstdev(rels)) if len(rels) > 1 else 0.0 if rels else None,
    }
    write_json(agg_path, aggregate_out)
    print(f"Aggregate summary: {agg_path}")


if __name__ == "__main__":
    main()
