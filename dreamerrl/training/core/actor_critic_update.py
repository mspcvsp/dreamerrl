from __future__ import annotations

import torch
from torch.distributions import Categorical

from dreamerrl.training.core.imagination import imagine_trajectory_for_training
from dreamerrl.training.core.lambda_return import lambda_return
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode, value_from_logits


def actor_critic_update(
    world_model,
    actor,
    critic,
    batch,
    imagination_horizon: int,
    discount: float,
    lam: float,  # <--- matches trainer
) -> tuple[torch.Tensor, torch.Tensor]:
    # ---------------------------------------------------------
    # 1. Imagination rollout
    # ---------------------------------------------------------
    start_state = world_model.init_state(batch["state"].shape[0])
    traj = imagine_trajectory_for_training(
        world_model=world_model,
        actor=actor,
        critic=critic,
        state=start_state,
        horizon=imagination_horizon,
    )

    h = traj["h"]  # (T, B, deter)
    z = traj["z"]  # (T, B, stoch)
    reward = traj["reward"]  # (T, B)
    action = traj["action"]  # (T, B)

    T, B = reward.shape

    # ---------------------------------------------------------
    # 2. Critic target: λ-return in symlog space
    # ---------------------------------------------------------
    with torch.no_grad():
        bootstrap_logits = critic(h[-1], z[-1])  # (B, num_bins)
        bootstrap_value = value_from_logits(bootstrap_logits)  # (B,)

        returns = lambda_return(
            reward=reward,
            value=bootstrap_value,
            discount=discount,
            lam=lam,  # <--- correct
        )  # (T, B)

        returns_symlog = symlog(returns)

    # ---------------------------------------------------------
    # 3. Critic loss (two-hot)
    # ---------------------------------------------------------
    critic_logits = critic(
        h.reshape(T * B, -1),
        z.reshape(T * B, -1),
    ).reshape(T, B, -1)

    target_twohot = twohot_encode(returns_symlog)
    log_probs = torch.log_softmax(critic_logits, dim=-1)
    critic_loss = -(target_twohot * log_probs).sum(dim=-1).mean()

    # ---------------------------------------------------------
    # 4. Actor loss (return normalization)
    # ---------------------------------------------------------
    with torch.no_grad():
        value_pred = value_from_logits(critic_logits)
        adv = returns - value_pred

        flat = returns.reshape(-1)
        p5, p95 = torch.quantile(flat, torch.tensor([0.05, 0.95], device=flat.device))
        scale = torch.clamp(p95 - p5, min=1.0)
        adv_norm = adv / scale

    logits = actor(
        h.reshape(T * B, -1),
        z.reshape(T * B, -1),
    ).reshape(T, B, -1)

    dist = Categorical(logits=logits)
    logp = dist.log_prob(action)
    entropy = dist.entropy()

    actor_loss = -(logp * adv_norm.detach()).mean() - 1e-3 * entropy.mean()

    return actor_loss, critic_loss
