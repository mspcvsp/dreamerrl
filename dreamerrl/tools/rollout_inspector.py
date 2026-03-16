# dreamerrl/tools/rollout_inspector.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from dreamerrl.models.world_model import WorldModelState


def summarize_rollout(
    rollout: Iterable[WorldModelState],
) -> Dict[str, Any]:
    """Return cheap, shape‑only summary of a rollout."""
    hs: List[torch.Size] = []
    zs: List[torch.Size] = []

    for s in rollout:
        assert isinstance(s, WorldModelState)
        hs.append(s.h.shape)
        zs.append(s.z.shape)

    return {
        "len": len(hs),
        "h_shapes": hs,
        "z_shapes": zs,
    }


def check_rollout_consistency(
    rollout: Iterable[WorldModelState],
) -> None:
    """Assert that all states in the rollout share consistent batch/latent dims."""
    hs: List[torch.Size] = []
    zs: List[torch.Size] = []

    for s in rollout:
        assert isinstance(s, WorldModelState)
        hs.append(s.h.shape)
        zs.append(s.z.shape)

    if not hs:
        return

    h0, z0 = hs[0], zs[0]
    for i, (h, z) in enumerate(zip(hs, zs)):
        assert h == h0, f"h shape mismatch at t={i}: {h} != {h0}"
        assert z == z0, f"z shape mismatch at t={i}: {z} != {z0}"
