#!/usr/bin/env python3
"""Data storage + hysteresis analysis for the 3-D rock-joint shear capstone.

Traction-separation convention (3-D interface, joint plane = local x-y, normal = z)
-----------------------------------------------------------------------------------
A point on the joint exchanges THREE conjugate traction/separation pairs:

  * one normal traction   t_N  conjugate to the normal separation u_n (dilation),
  * two in-plane shear tractions  t_x, t_y  conjugate to the in-plane slips u_x, u_y.

Sign convention: the normal stress sigma_n (and t_N) is COMPRESSION-POSITIVE, so a
closing/dilating joint under load keeps sigma_n >= 0.  The in-plane shear tractions
form the resolved shear traction tau = sqrt(t_x^2 + t_y^2); the apparent friction is
mu_app = tau / sigma_n.  Energy bookkeeping (FEM cyclic flavor) splits the external
work W_ext into recoverable elastic energy W_elastic and the dissipated parts
W_friction (frictional slip, >= 0) and W_damage (surface degradation, >= 0).

This module is a thin, numpy-only I/O + analysis helper.  It stores per-increment
histories (arbitrary array keys) as ``history.npz`` plus a self-describing
``params.json``, and provides hysteresis-loop diagnostics (enclosed loop area =
dissipated energy/cycle, cumulative frictional work, per-cycle peak/area summary).

Supported history flavors (both handled transparently — keys are not hard-coded):
  (1) RIGID monotonic : u, z, dilation, Tx, Ty, T_par, T_perp, mu_app, n_active, ...
  (2) FEM   cyclic    : step, ux, uy, u_n, t_N, t_x, t_y, mu_app, sigma_n,
                        W_ext, W_elastic, W_friction, W_damage, D_mean, i_mean, ...
                        (+ an optional per-cycle list of summary dicts)
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# JSON encoder for numpy scalars / arrays
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays JSON-serialisable (used as ``default=``)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _is_arraylike(v: Any) -> bool:
    """True if *v* should be stored in the .npz (a real numeric array)."""
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, (list, tuple)) and len(v) > 0:
        # a list of dicts (e.g. per-cycle summary) is NOT array data
        if any(isinstance(x, dict) for x in v):
            return False
        try:
            arr = np.asarray(v)
        except Exception:
            return False
        return arr.dtype != object and arr.ndim >= 1
    return False


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_joint_history(
    out_dir: str,
    history: Dict[str, Any],
    params: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a joint-shear run to *out_dir*.

    Writes:
      * ``history.npz``   — every array-valued item in *history*,
      * ``history_aux.json`` — the leftover scalar / list-of-dict items,
      * ``params.json``   — the (self-describing) *params* dict,
      * ``meta.json``     — only if *meta* is given.

    Array keys are arbitrary; *params* may carry any of E, nu, mu, phi_b, i0,
    k_n, eps_n, protocol, azimuth, n_cycles, surface, sigma_n0, ...  Returns
    *out_dir*.
    """
    os.makedirs(out_dir, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    aux: Dict[str, Any] = {}
    for key, val in history.items():
        if _is_arraylike(val):
            arrays[key] = np.asarray(val)
        else:
            aux[key] = val

    np.savez(os.path.join(out_dir, "history.npz"), **arrays)

    with open(os.path.join(out_dir, "history_aux.json"), "w") as fh:
        json.dump(aux, fh, indent=2, default=_json_default)

    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=_json_default)

    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=_json_default)

    return out_dir


def load_joint_history(path: str) -> Dict[str, Any]:
    """Load a run saved by :func:`save_joint_history`.

    *path* may be the run directory or the ``history.npz`` file itself.
    Returns ``{"history": {...}, "params": {...}, "meta": {...}}`` with all
    arrays restored.
    """
    if os.path.isdir(path):
        run_dir = path
    else:
        run_dir = os.path.dirname(os.path.abspath(path))

    history: Dict[str, Any] = {}
    npz_path = os.path.join(run_dir, "history.npz")
    if os.path.isfile(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for key in data.files:
                history[key] = data[key]

    aux_path = os.path.join(run_dir, "history_aux.json")
    if os.path.isfile(aux_path):
        with open(aux_path) as fh:
            history.update(json.load(fh))

    def _load_json(name: str) -> Dict[str, Any]:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
        return {}

    return {
        "history": history,
        "params": _load_json("params.json"),
        "meta": _load_json("meta.json"),
    }


# ---------------------------------------------------------------------------
# Hysteresis-loop analysis
# ---------------------------------------------------------------------------

def loop_area(u: Sequence[float], tau: Sequence[float]) -> float:
    """Signed enclosed area of the hysteresis loop traced by (u, tau).

    Uses the shoelace formula on the closed path; the path is closed
    automatically if the last point differs from the first.  The dissipated
    energy per cycle is ``abs(loop_area(u, tau))``.  A counter-clockwise loop
    (the physical sense for a shear stress lagging the slip) gives a positive
    area.
    """
    u = np.asarray(u, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    if u.size < 3:
        return 0.0
    # close the path
    if u[0] != u[-1] or tau[0] != tau[-1]:
        u = np.append(u, u[0])
        tau = np.append(tau, tau[0])
    # shoelace: A = 1/2 sum (x_i * y_{i+1} - x_{i+1} * y_i)
    return 0.5 * float(np.sum(u[:-1] * tau[1:] - u[1:] * tau[:-1]))


def cumulative_dissipation(
    slip_increment: Sequence[float],
    tau: Sequence[float],
) -> np.ndarray:
    """Running frictional work: cumulative sum of ``tau * d(slip)``.

    *slip_increment* is the per-increment slip step (d|u| >= 0 for monotonic
    slip); *tau* is the resolved shear traction at each increment.  Returns an
    array the same length as the inputs (cumulative work after each increment).
    """
    slip_increment = np.asarray(slip_increment, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    n = min(slip_increment.size, tau.size)
    return np.cumsum(tau[:n] * slip_increment[:n])


def _resolved_shear(history: Dict[str, Any]) -> Optional[np.ndarray]:
    """Best-effort resolved shear traction tau from a history dict."""
    if "tau" in history:
        return np.asarray(history["tau"], dtype=float).ravel()
    if "t_x" in history and "t_y" in history:
        tx = np.asarray(history["t_x"], dtype=float).ravel()
        ty = np.asarray(history["t_y"], dtype=float).ravel()
        return np.hypot(tx, ty)
    if "Tx" in history and "Ty" in history:
        tx = np.asarray(history["Tx"], dtype=float).ravel()
        ty = np.asarray(history["Ty"], dtype=float).ravel()
        return np.hypot(tx, ty)
    if "T_par" in history:
        return np.asarray(history["T_par"], dtype=float).ravel()
    return None


def _shear_slip(history: Dict[str, Any]) -> Optional[np.ndarray]:
    """Best-effort scalar in-plane slip from a history dict."""
    for key in ("u", "u_par", "ux"):
        if key in history:
            return np.asarray(history[key], dtype=float).ravel()
    if "ux" in history and "uy" in history:
        ux = np.asarray(history["ux"], dtype=float).ravel()
        uy = np.asarray(history["uy"], dtype=float).ravel()
        return np.hypot(ux, uy)
    return None


def _dilation(history: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("dilation", "u_n"):
        if key in history:
            return np.asarray(history[key], dtype=float).ravel()
    return None


def per_cycle_summary(
    history: Dict[str, Any],
    cycle_index_key: str = "cycle",
) -> List[Dict[str, Any]]:
    """Split a per-increment history by cycle index and summarise each cycle.

    Requires a per-increment integer cycle-index array under *cycle_index_key*.
    Returns a list of ``{cycle, tau_peak, loop_area, dilation_range}`` dicts.
    If no cycle-index array is present, returns ``[]``.
    """
    if cycle_index_key not in history:
        return []
    cyc = np.asarray(history[cycle_index_key]).ravel()
    if cyc.size == 0:
        return []

    u = _shear_slip(history)
    tau = _resolved_shear(history)
    dil = _dilation(history)

    out: List[Dict[str, Any]] = []
    for c in sorted(set(int(v) for v in cyc)):
        mask = cyc == c
        entry: Dict[str, Any] = {"cycle": int(c)}
        entry["tau_peak"] = float(np.max(np.abs(tau[mask]))) if tau is not None else float("nan")
        if u is not None and tau is not None:
            entry["loop_area"] = abs(loop_area(u[mask], tau[mask]))
        else:
            entry["loop_area"] = float("nan")
        if dil is not None and np.any(mask):
            entry["dilation_range"] = float(np.ptp(dil[mask]))
        else:
            entry["dilation_range"] = float("nan")
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    out_dir = os.path.join(root, "runs", "_io_selftest")

    # ---- synthesise a 2-cycle hysteresis loop ----------------------------
    # u = triangle wave over 2 cycles in [-U, U]; tau = mu*sigma + a*sin(theta).
    # Here u(theta) is a triangle wave and the oscillating part of tau is a sine
    # that is 90 deg out of phase with u (it peaks where u crosses zero), so the
    # (u, tau) path encloses a lens whose exact area is 8*U*a/pi per cycle.  The
    # chosen phase makes the loop counter-clockwise (positive signed area).
    mu, sigma, a, U = 0.6, 1.0, 0.5, 1.0
    n_per_cycle = 400
    n_cycles = 2
    theta = np.linspace(0.0, 2.0 * np.pi * n_cycles, n_per_cycle * n_cycles, endpoint=False)
    u = U * (2.0 / np.pi) * np.arcsin(np.sin(theta))      # triangle wave in [-U, U]
    tau = mu * sigma - a * np.sin(theta + np.pi / 2.0)    # 90 deg out of phase with u
    cycle = (theta // (2.0 * np.pi)).astype(int)
    slip_inc = np.abs(np.diff(u, prepend=u[0]))

    history = {
        "theta": theta, "u": u, "tau": tau, "dilation": 0.1 * np.abs(u),
        "cycle": cycle, "slip_inc": slip_inc, "mu_app": tau / sigma,
        "n_active": np.full(theta.shape, 42, dtype=int),  # arbitrary array key
        "_cycle_summary": [{"cycle": 0, "tau_peak": 1.1}, {"cycle": 1, "tau_peak": 1.1}],
        "n_steps": int(theta.size),                       # scalar -> aux json
    }
    params = {
        "E": 30e9, "nu": 0.25, "mu": mu, "phi_b": 30.0, "i0": 10.0,
        "k_n": 1e12, "eps_n": 1e10, "protocol": "cyclic", "azimuth": 0.0,
        "n_cycles": n_cycles, "surface": "selftest_ridged", "sigma_n0": sigma,
    }
    meta = {"created_by": "joint_data_io self-test", "version": 1}

    # ---- save / load round-trip ------------------------------------------
    save_joint_history(out_dir, history, params, meta=meta)
    loaded = load_joint_history(out_dir)
    h2 = loaded["history"]

    for key in ("theta", "u", "tau", "dilation", "cycle", "slip_inc",
                "mu_app", "n_active"):
        assert key in h2, f"missing array key {key!r} after round-trip"
        assert np.allclose(np.asarray(history[key], dtype=float),
                           np.asarray(h2[key], dtype=float)), \
            f"array {key!r} did not round-trip"
    assert loaded["params"]["surface"] == "selftest_ridged"
    assert loaded["params"]["n_cycles"] == n_cycles
    assert loaded["meta"]["version"] == 1
    assert h2["n_steps"] == int(theta.size)               # scalar via aux json
    assert isinstance(h2["_cycle_summary"], list) and len(h2["_cycle_summary"]) == 2

    # ---- loop area: numeric vs analytic ----------------------------------
    # Triangle-wave u vs 90-deg-shifted sine tau: exact enclosed area = 8*U*a/pi.
    area_one = abs(loop_area(u[cycle == 0], tau[cycle == 0]))
    area_analytic = 8.0 * U * a / np.pi
    rel_err = abs(area_one - area_analytic) / area_analytic
    assert loop_area(u[cycle == 0], tau[cycle == 0]) > 0.0, "loop area should be positive"
    assert rel_err < 0.01, f"loop area off by {rel_err:.3%} (got {area_one:.6f}, want {area_analytic:.6f})"

    # ---- cumulative dissipation monotone non-decreasing ------------------
    cum = cumulative_dissipation(slip_inc, np.abs(tau))
    assert cum.shape == theta.shape
    assert np.all(np.diff(cum) >= -1e-12), "cumulative dissipation must be non-decreasing"

    # ---- per-cycle summary -----------------------------------------------
    summary = per_cycle_summary(history, cycle_index_key="cycle")
    assert len(summary) == n_cycles, f"expected {n_cycles} cycle summaries, got {len(summary)}"
    for s in summary:
        assert {"cycle", "tau_peak", "loop_area", "dilation_range"} <= set(s)
        assert abs(s["loop_area"] - area_analytic) / area_analytic < 0.01

    # ---- cleanup ---------------------------------------------------------
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"  loop area  : numeric={area_one:.6f}  analytic={area_analytic:.6f}  rel_err={rel_err:.3%}")
    print(f"  cum. dissip: final={cum[-1]:.6f}  ({len(summary)} cycles)")
    print("joint_data_io self-test PASSED")


if __name__ == "__main__":
    _selftest()
