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
    """
    Compute actor and critic losses using imagined rollouts and λ-return.

    Args:
        world_model: the world model.
        actor: policy network.
        critic: value network.
        batch: replay batch (only batch size is used here).
        imagination_horizon: number of imagination steps T.
        discount: discount factor γ.
        lam: λ parameter.

    Returns:
        (actor_loss, critic_loss) as scalar tensors.
    """
    B = batch["state"].size(0)
    state: WorldModelState = world_model.init_state(B)

    imagined_states = []
    for _ in range(imagination_horizon):
        state = world_model.imagine_step(state)
        imagined_states.append(state)

    rewards = torch.stack(
        [world_model.predict_reward(s) for s in imagined_states],
        dim=0,
    ).squeeze(-1)  # (T, B)

    values = torch.stack(
        [critic(s.h, s.z) for s in imagined_states],
        dim=0,
    ).squeeze(-1)  # (T, B)

    # Bootstrap value: (T+1, B)
    value_bootstrap = torch.cat([values, values[-1:].detach()], dim=0)

    returns = lambda_return(
        reward=rewards,
        value=value_bootstrap,
        discount=discount,
        lam=lam,
    )  # (T, B)

    logits = torch.stack(
        [actor(s.h, s.z) for s in imagined_states],
        dim=0,
    )  # (T, B, A)

    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    logp = dist.log_prob(actions)  # (T, B)

    actor_loss = -(logp * returns.detach()).mean()
    critic_loss = (values - returns.detach()).pow(2).mean()

    return actor_loss, critic_loss
