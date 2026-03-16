# dreamerrl/training/core/imagination.py

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.distributions import Categorical

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState


def imagine_trajectory_for_training(
    world_model: WorldModel,
    actor: Actor,
    critic: ValueHead,
    state: WorldModelState,
    horizon: int,
) -> Dict[str, Any]:
    """
    Training-time imagination.
    Always returns actions, values, rewards, h, z.
    """
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs: List[torch.Tensor] = []
    zs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    for _ in range(horizon):
        # reward + value
        r = world_model.predict_reward(s).squeeze(-1)
        v = critic(s.h, s.z).squeeze(-1)

        rewards.append(r)
        values.append(v)

        # action
        logits = actor(s.h, s.z)
        dist = Categorical(logits=logits)
        a = dist.sample()
        actions.append(a)

        # world model transition
        s = world_model.imagine_step(s)
        hs.append(s.h)
        zs.append(s.z)

    return {
        "h": torch.stack(hs, dim=0),
        "z": torch.stack(zs, dim=0),
        "reward": torch.stack(rewards, dim=0),
        "value": torch.stack(values, dim=0),
        "action": torch.stack(actions, dim=0),
    }


def imagine_trajectory_for_testing(
    world_model: WorldModel,
    state: WorldModelState,
    horizon: int,
) -> Dict[str, Any]:
    """
    Testing-time imagination.
    Only returns h, z, reward.
    No actor, no critic.
    """
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs: List[torch.Tensor] = []
    zs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []

    for _ in range(horizon):
        r = world_model.predict_reward(s).squeeze(-1)
        rewards.append(r)

        s = world_model.imagine_step(s)
        hs.append(s.h)
        zs.append(s.z)

    return {
        "h": torch.stack(hs, dim=0),
        "z": torch.stack(zs, dim=0),
        "reward": torch.stack(rewards, dim=0),
    }
