from __future__ import annotations

from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor, act_in_imagination
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState


def imagine_trajectory_for_training(
    world_model: WorldModel,
    actor: Actor,
    critic: ValueHead,
    state: WorldModelState,
    horizon: int,
    deterministic_imagination: bool = False,
) -> Dict[str, Any]:
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs, zs, rewards, actions = [], [], [], []

    for _ in range(horizon):
        reward_main_logits, _ = world_model.reward_heads(s.h, s.z)
        r = world_model.reward_heads.main.readout(reward_main_logits)

        logits = actor(s.h, s.z)
        a = act_in_imagination(logits, deterministic_imagination=deterministic_imagination)

        rewards.append(r)
        actions.append(a)

        s = world_model.imagine_step(s, actor, deterministic_imagination=deterministic_imagination)

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


def imagine_trajectory_for_testing(world_model: WorldModel, actor: Actor, state: WorldModelState, horizon: int):
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs, zs, rewards = [], [], []

    for _ in range(horizon):
        reward_main_logits, _ = world_model.reward_heads(s.h, s.z)
        r = world_model.reward_heads.main.readout(reward_main_logits)

        rewards.append(r)
        hs.append(s.h)
        zs.append(s.z)

        s = world_model.imagine_step(s, actor)

    return {
        "h": torch.stack(hs, dim=0),
        "z": torch.stack(zs, dim=0),
        "reward": torch.stack(rewards, dim=0),
    }
