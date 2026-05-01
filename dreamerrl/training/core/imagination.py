from __future__ import annotations

from typing import Any, Dict

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
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs, zs, rewards, actions = [], [], [], []

    for _ in range(horizon):
        reward_logits = world_model.reward_head(s.h, s.z)
        r = world_model.reward_head.readout(reward_logits)

        logits = actor(s.h, s.z)
        dist = Categorical(logits=logits)
        a = dist.sample()

        rewards.append(r)
        actions.append(a)

        s = world_model.imagine_step(s)
        hs.append(s.h)
        zs.append(s.z)

    bootstrap_logits = critic(s.h, s.z)
    bootstrap_value = critic.readout(bootstrap_logits)

    return {
        "h": torch.stack(hs, dim=0),
        "z": torch.stack(zs, dim=0),
        "reward": torch.stack(rewards, dim=0),
        "action": torch.stack(actions, dim=0),
        "bootstrap_value": bootstrap_value,
    }


def imagine_trajectory_for_testing(world_model: WorldModel, state: WorldModelState, horizon: int):
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs, zs, rewards = [], [], []

    for _ in range(horizon):
        reward_logits = world_model.reward_head(s.h, s.z)
        r = world_model.reward_head.readout(reward_logits)

        rewards.append(r)
        hs.append(s.h)
        zs.append(s.z)

        s = world_model.imagine_step(s)

    return {
        "h": torch.stack(hs, dim=0),
        "z": torch.stack(zs, dim=0),
        "reward": torch.stack(rewards, dim=0),
    }
