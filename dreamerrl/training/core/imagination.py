from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import (
    WorldModel,
    WorldModelState,
)


def imagination_rollout(
    world_model: WorldModel,
    actor: Optional[Actor],
    critic: Optional[ValueHead],
    state: WorldModelState,
    horizon: int,
) -> Dict[str, Any]:
    """
    Roll out imagined trajectories in latent space.

    Args:
        world_model: the world model.
        actor: policy network (may be None).
        critic: value network (may be None).
        state: initial latent state (B batch size).
        horizon: number of imagination steps T.

    Returns:
        Dict with keys:
            "h": (T, B, deter)
            "z": (T, B, stoch)
            "value": (T, B, 1) or None
            "action": (T, B) or None
    """
    hs, zs, values, actions = [], [], [], []

    for _ in range(horizon):
        state = world_model.imagine_step(state)
        hs.append(state.h)
        zs.append(state.z)

        if actor is not None:
            logits = actor(state.h, state.z)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            actions.append(act)

        if critic is not None:
            values.append(critic(state.h, state.z))

    out: Dict[str, Any] = {
        "h": torch.stack(hs),  # (T, B, deter)
        "z": torch.stack(zs),  # (T, B, stoch)
    }

    if values:
        out["value"] = torch.stack(values)  # (T, B, 1)
    else:
        out["value"] = None

    if actions:
        out["action"] = torch.stack(actions)  # (T, B)
    else:
        out["action"] = None

    return out
