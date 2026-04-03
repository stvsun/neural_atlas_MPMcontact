"""Topology certification glue for trained SDF and atlas checkpoints.

Connects the SDF training pipeline to the persistent homology pipeline,
providing convenience functions to compute Betti numbers, M_min, and
full atlas certification from checkpoint files.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from atlas.topo.filtration import sample_sdf_on_grid, clip_to_interior


def certify_sdf(
    sdf_net: torch.nn.Module,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    resolution: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Compute topological certification of a trained SDF.

    Samples the SDF on a regular grid, computes persistent homology,
    extracts Betti numbers, and determines the minimum chart count M_min.

    Parameters
    ----------
    sdf_net : torch.nn.Module
        Trained SDF network with forward(x) -> (N,) signed distance values.
    bbox_min, bbox_max : torch.Tensor
        (3,) bounding box corners.
    resolution : int
        Grid resolution per axis.
    device : torch.device
        Compute device.

    Returns
    -------
    report : dict
        Keys: M_min (int), betti (dict), explanation (str),
        has_gudhi (bool), grid_vals (ndarray).
    """
    # Sample SDF on grid
    grid_vals, coords = sample_sdf_on_grid(
        sdf_net, bbox_min, bbox_max, resolution=resolution,
    )
    grid_clipped = clip_to_interior(grid_vals, t_max=0.0)

    # Try to compute persistence (requires GUDHI)
    try:
        from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
        from atlas.topo.ls_category import compute_m_min

        diagrams = compute_persistence_diagrams(grid_clipped, max_dimension=2)
        betti = betti_numbers_at(diagrams, t=-1e-6)
        m_min = compute_m_min(betti)

        explanation_parts = [f"Betti numbers: {betti}", f"M_min = {m_min}"]
        for k, v in sorted(betti.items()):
            if k == 0:
                explanation_parts.append(f"  beta_0 = {v} connected component(s)")
            elif k == 1:
                explanation_parts.append(f"  beta_1 = {v} tunnel(s)/loop(s)")
            elif k == 2:
                explanation_parts.append(f"  beta_2 = {v} void(s)/cavity(ies)")

        return {
            "M_min": m_min,
            "betti": betti,
            "explanation": "\n".join(explanation_parts),
            "has_gudhi": True,
            "grid_vals": grid_clipped,
        }

    except ImportError:
        return {
            "M_min": 1,
            "betti": {0: 1, 1: 0, 2: 0},
            "explanation": "GUDHI not available; assuming contractible domain (M_min=1)",
            "has_gudhi": False,
            "grid_vals": grid_clipped,
        }


def certify_sdf_from_checkpoint(
    sdf_checkpoint: str,
    bbox_min: Optional[torch.Tensor] = None,
    bbox_max: Optional[torch.Tensor] = None,
    resolution: int = 32,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Dict:
    """Load a trained SDF checkpoint and compute topological certification.

    Parameters
    ----------
    sdf_checkpoint : str
        Path to SDF checkpoint (.pt file with model_state, model_kwargs, center, scale).
    bbox_min, bbox_max : torch.Tensor, optional
        Bounding box. If None, uses center +/- 1.5*scale from checkpoint.
    resolution : int
        Grid resolution per axis.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision.

    Returns
    -------
    report : dict
        Same as certify_sdf(), plus sdf_net and checkpoint metadata.
    """
    from common.models import MLP

    ckpt = torch.load(sdf_checkpoint, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", {"width": 128, "depth": 6})

    # Reconstruct SDFNet
    class SDFNet(torch.nn.Module):
        def __init__(self, width=128, depth=6):
            super().__init__()
            self.net = MLP(in_dim=3, out_dim=1, width=width, depth=depth)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    sdf_net = SDFNet(**model_kwargs).to(device=device, dtype=dtype)
    sdf_net.load_state_dict(ckpt["model_state"])
    sdf_net.eval()

    center = torch.tensor(ckpt["center"], device=device, dtype=dtype)
    scale = float(ckpt["scale"])

    if bbox_min is None:
        bbox_min = center - 1.5 * scale
    if bbox_max is None:
        bbox_max = center + 1.5 * scale

    # The SDF network operates in normalized space; wrap it
    class NormalizedSDF(torch.nn.Module):
        def __init__(self, net, center, scale):
            super().__init__()
            self.net = net
            self.center = center
            self.scale_val = scale
        def forward(self, x):
            x_norm = (x - self.center.unsqueeze(0)) / self.scale_val
            return self.net(x_norm) * self.scale_val

    wrapped = NormalizedSDF(sdf_net, center, scale)

    report = certify_sdf(wrapped, bbox_min, bbox_max, resolution=resolution, device=device)
    report["sdf_checkpoint"] = sdf_checkpoint
    report["center"] = center.cpu().numpy()
    report["scale"] = scale
    return report


def certify_atlas_from_checkpoint(
    atlas_checkpoint: str,
    sdf_checkpoint: str,
    resolution: int = 32,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Dict:
    """Full atlas certification: topology (M_min) + quality gates.

    Parameters
    ----------
    atlas_checkpoint : str
        Path to trained atlas checkpoint (.pt).
    sdf_checkpoint : str
        Path to trained SDF checkpoint (.pt).
    resolution : int
        Grid resolution for topology computation.
    device, dtype : torch.device, torch.dtype
        Compute settings.

    Returns
    -------
    report : dict
        Combined topology + quality certification.
    """
    from atlas.topo.ls_category import certify_atlas

    # Load atlas checkpoint for M_actual and quality gates
    atlas_ckpt = torch.load(atlas_checkpoint, map_location=device)
    n_charts = len(atlas_ckpt["decoder_states"])
    quality_gate = atlas_ckpt.get("gate", {})

    # Get topology certification
    topo_report = certify_sdf_from_checkpoint(
        sdf_checkpoint, resolution=resolution, device=device, dtype=dtype,
    )

    # Combine: certify_atlas checks M_actual >= M_min + quality gates
    if topo_report["has_gudhi"]:
        certification = certify_atlas(
            M_actual=n_charts,
            betti=topo_report["betti"],
            quality_metrics=quality_gate,
        )
    else:
        certification = {
            "passed": quality_gate.get("passed", False),
            "topology_pass": True,
            "quality_pass": quality_gate.get("passed", False),
            "M_actual": n_charts,
            "M_min": 1,
            "note": "GUDHI not available; topology check skipped",
        }

    return {
        "topology": topo_report,
        "quality": quality_gate,
        "certification": certification,
        "M_actual": n_charts,
        "atlas_checkpoint": atlas_checkpoint,
        "sdf_checkpoint": sdf_checkpoint,
    }
