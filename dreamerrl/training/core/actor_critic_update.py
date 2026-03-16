from __future__ import annotations

from typing import Dict, Tuple

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import (
    WorldModel,
    WorldModelState,
)

from .lambda_return import lambda_return


def actor_critic_update(
    world_model: WorldModel,
    actor: Actor,
    critic: ValueHead,
    batch: Dict[str, torch.Tensor],
    imagination_horizon: int,
    discount: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = batch["state"].size(0)
    state: WorldModelState = world_model.init_state(B)

    # ---- imagination rollout ----
    imagined_states = []
    for _ in range(imagination_horizon):
        state = world_model.imagine_step(state)
        imagined_states.append(state)

    # ---- rewards ----
    rewards = torch.stack(
        [world_model.predict_reward(s).squeeze(-1) for s in imagined_states],
        dim=0,
    )  # (T, B)

    # ---- values ----
    values = torch.stack(
        [critic(s.h, s.z).squeeze(-1) for s in imagined_states],
        dim=0,
    )  # (T, B)

    # ---- bootstrap ----
    bootstrap = critic(imagined_states[-1].h, imagined_states[-1].z).squeeze(-1)
    value_bootstrap = torch.cat([values, bootstrap.unsqueeze(0)], dim=0)  # (T+1, B)

    # ---- λ-return ----
    returns = lambda_return(
        reward=rewards,
        value=value_bootstrap,
        discount=discount,
        lam=lam,
    )  # (T, B)

    # ---- policy ----
    logits = torch.stack(
        [actor(s.h, s.z) for s in imagined_states],
        dim=0,
    )  # (T, B, A)

    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()  # (T, B)
    logp = dist.log_prob(actions)  # (T, B)

    # ---- losses ----
    actor_loss = -(logp * returns.detach()).mean()
    critic_loss = (values - returns.detach()).pow(2).mean()

    return actor_loss, critic_loss
