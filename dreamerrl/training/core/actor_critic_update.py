"""
Why this function uses closures:
-------------------------------
Dreamer-style actor–critic updates depend on many shared tensors (h, z, reward, action, returns, logits) and
hyperparameters (discount, λ, normalization scale, entropy weight). Passing all of these through every helper function
leads to long, brittle signatures and obscures the mathematical structure of the update.

By defining small inner functions (compute_returns(), compute_critic_loss(), compute_actor_loss()), we let Python
closures capture the rollout tensors and configuration from the outer scope. Each helper becomes a clean,
self-contained unit that directly reflects the equations in the Dreamer-V3 paper, without threading a dozen arguments
through every call.

This pattern keeps the implementation modular, testable, and easy to extend (EMA critics, advantage clipping, entropy
schedules, PopGym-specific shaping, etc.) while preserving the clarity of the underlying algorithm. The closures simply
“see” the trajectory dict and hyperparameters, making the code both concise and faithful to the research formulation.
"""

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
    lam: float,
):
    """
    Clean Dreamer-V3 actor-critic update using closures.
    """

    # ---------------------------------------------------------
    # 0. Imagination rollout
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
    # 1. Closure: compute λ-returns in symlog space
    # ---------------------------------------------------------
    def compute_returns():
        with torch.no_grad():
            bootstrap_logits = critic(h[-1], z[-1])  # (B, num_bins)
            bootstrap_value = value_from_logits(bootstrap_logits)  # (B,)

            returns = lambda_return(
                reward=reward,
                value=bootstrap_value,
                discount=discount,
                lam=lam,
            )  # (T, B)

            return symlog(returns)

    returns_symlog = compute_returns()

    # ---------------------------------------------------------
    # 2. Closure: critic loss (two-hot)
    # ---------------------------------------------------------
    def compute_critic_loss():
        critic_logits = critic(
            h.reshape(T * B, -1),
            z.reshape(T * B, -1),
        ).reshape(T, B, -1)

        target_twohot = twohot_encode(returns_symlog)
        log_probs = torch.log_softmax(critic_logits, dim=-1)
        return -(target_twohot * log_probs).sum(dim=-1).mean()

    critic_loss = compute_critic_loss()

    # ---------------------------------------------------------
    # 3. Closure: actor loss (return normalization)
    # ---------------------------------------------------------
    def compute_actor_loss():
        with torch.no_grad():
            critic_logits = critic(
                h.reshape(T * B, -1),
                z.reshape(T * B, -1),
            ).reshape(T, B, -1)

            value_pred = value_from_logits(critic_logits)
            adv = returns_symlog - symlog(value_pred)

            flat = adv.reshape(-1)
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

        return -(logp * adv_norm.detach()).mean() - 1e-3 * entropy.mean()

    actor_loss = compute_actor_loss()

    return actor_loss, critic_loss
