"""General utilities: device resolution, seeding, plotting, metrics."""

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Requested --device cuda but CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested --device mps but MPS is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_arg}")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type in ("cuda", "mps"):
            return torch.float32
        return torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS backend does not support float64 well; use --dtype float32 or auto.")
        return torch.float64
    raise ValueError(f"Unsupported dtype option: {dtype_arg}")


def build_run_stem(run_tag: str, prefix: str = "run") -> str:
    if run_tag is None:
        run_tag = ""
    run_tag = run_tag.strip()
    if len(run_tag) == 0:
        return prefix
    return f"{prefix}_{run_tag}"


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    iters = np.arange(1, len(history["global_residual"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.semilogy(iters, np.maximum(history["global_residual"], 1e-16), label="global_residual")
    ax.semilogy(iters, np.maximum(history["interface_value"], 1e-16), label="interface_value")
    ax.semilogy(iters, np.maximum(history["interface_flux"], 1e-16), label="interface_flux")
    ax.semilogy(iters, np.maximum(history["bc_loss"], 1e-16), label="bc_loss")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Metric")
    ax.set_title("Atlas Schwarz convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def metric_l2(u_pred: np.ndarray, u_true: np.ndarray) -> Dict[str, float]:
    err = u_pred - u_true
    l2 = float(np.sqrt(np.mean(err**2)))
    rel = float(np.sqrt(np.mean(err**2) / max(np.mean(u_true**2), 1e-12)))
    max_e = float(np.max(np.abs(err)))
    return {
        "l2_error": l2,
        "relative_l2_error": rel,
        "max_error": max_e,
    }
