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
    action_dim = world_model.net_cfg.action_dim
    if action_dim is None:
        raise ValueError("action_dim is required to sample imagined actions")

    hs, zs, rewards, actions, continues = [], [], [], [], []
    # Discrete REINFORCE uses detached imagined features. Only the policy
    # log-probability evaluated later carries actor gradients.
    with torch.no_grad():
        for _ in range(horizon):
            hs.append(s.h)
            zs.append(s.z)
            logits = actor(s.h, s.z)
            a = act_in_imagination(logits, deterministic_imagination=deterministic_imagination)
            actions.append(a)
            action = torch.nn.functional.one_hot(a, action_dim).float()
            s = world_model.imagine_step(
                s,
                actor,
                deterministic_imagination=deterministic_imagination,
                action=action,
            )
            reward_logits = world_model.reward_heads.main(s.h, s.z)
            rewards.append(world_model.reward_heads.main.readout(reward_logits))
            continues.append(world_model.continue_head.readout(world_model.continue_head(s.h, s.z)))
        bootstrap_value = critic.readout(critic(s.h, s.z))
    return {
        "h": torch.stack(hs),
        "z": torch.stack(zs),
        "reward": torch.stack(rewards),
        "action": torch.stack(actions),
        "continue": torch.stack(continues),
        "bootstrap_value": bootstrap_value,
    }


def imagine_trajectory_for_testing(world_model: WorldModel, actor: Actor, state: WorldModelState, horizon: int):
    device = next(world_model.parameters()).device
    s = state.to(device)

    hs, zs, rewards = [], [], []

    for _ in range(horizon):
        reward_main_logits = world_model.reward_heads.main(s.h, s.z)
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
