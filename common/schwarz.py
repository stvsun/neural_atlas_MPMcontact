"""Schwarz domain decomposition utilities.

Functions for atlas model loading and chart coloring for multiplicative
Schwarz iteration on overlapping chart domains.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from common.models import ChartDecoder, MaskNet


def load_atlas_models(
    atlas_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[ChartDecoder], List[MaskNet], Dict[str, object]]:
    ckpt = torch.load(atlas_checkpoint, map_location=device)

    dec_kw = ckpt.get("decoder_kwargs", {"width": 64, "depth": 4})
    mask_kw = ckpt.get("mask_kwargs", {"width": 48, "depth": 3})
    dec_states = ckpt["decoder_states"]
    mask_states = ckpt["mask_states"]

    decoders: List[ChartDecoder] = []
    masks: List[MaskNet] = []
    for ds, ms in zip(dec_states, mask_states):
        d = ChartDecoder(width=dec_kw["width"], depth=dec_kw["depth"]).to(device=device, dtype=dtype)
        m = MaskNet(width=mask_kw["width"], depth=mask_kw["depth"]).to(device=device, dtype=dtype)
        d.load_state_dict(ds)
        m.load_state_dict(ms)
        d.eval()
        m.eval()
        for p in d.parameters():
            p.requires_grad_(False)
        for p in m.parameters():
            p.requires_grad_(False)
        decoders.append(d)
        masks.append(m)

    return decoders, masks, ckpt


def choose_color_groups(meta_json: Optional[str], n_charts: int, membership_np: np.ndarray) -> List[List[int]]:
    if meta_json is not None and os.path.isfile(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        groups = meta.get("color_groups")
        if isinstance(groups, list) and len(groups) > 0:
            out = []
            seen = set()
            for g in groups:
                gi = []
                for x in g:
                    ix = int(x)
                    if 0 <= ix < n_charts:
                        gi.append(ix)
                        seen.add(ix)
                if gi:
                    out.append(sorted(set(gi)))
            missing = [i for i in range(n_charts) if i not in seen]
            if missing:
                out.append(missing)
            return out

    adj = {i: set() for i in range(n_charts)}
    for i in range(n_charts):
        mi = membership_np[:, i].astype(bool)
        for j in range(i + 1, n_charts):
            mj = membership_np[:, j].astype(bool)
            shared = int(np.sum(mi & mj))
            if shared > 0:
                adj[i].add(j)
                adj[j].add(i)
    color: Dict[int, int] = {}
    for i in range(n_charts):
        used = {color[j] for j in adj[i] if j in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    n_colors = max(color.values()) + 1 if color else 1
    groups = [[] for _ in range(n_colors)]
    for i in range(n_charts):
        groups[color[i]].append(i)
    return groups
