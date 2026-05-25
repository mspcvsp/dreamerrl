from __future__ import annotations

import torch
from torch.distributions import Categorical

from dreamerrl.training.core.imagination import imagine_trajectory_for_training
from dreamerrl.training.core.lambda_return import lambda_return
from dreamerrl.utils.transforms import symlog


def actor_critic_update(
    world_model,
    actor,
    critic,
    batch,
    imagination_horizon: int,
    discount: float,
    lam: float,
):
    # Determine batch size
    if "state" in batch:
        B = batch["state"].shape[0]
    elif "obs" in batch:
        B = batch["obs"].shape[0]
    else:
        raise KeyError("Batch must contain 'state' or 'obs'")

    # Start from zero state (smoke test)
    start_state = world_model.init_state(B)

    # Imagination rollout
    traj = imagine_trajectory_for_training(
        world_model=world_model,
        actor=actor,
        critic=critic,
        state=start_state,
        horizon=imagination_horizon,
    )

    # After imagination rollout
    h = traj["h"]
    z = traj["z"]
    reward = traj["reward"]
    action = traj["action"]

    # ---------------------------------------------------------
    # Detach imagination for critic update
    # ---------------------------------------------------------
    h_det = h.detach()
    z_det = z.detach()
    reward_det = reward.detach()

    T, B = reward.shape
    _, _, K, C = z.shape

    # ---------------------------------------------------------
    # Compute returns in symlog space
    # ---------------------------------------------------------
    def compute_returns_symlog():
        with torch.no_grad():
            # Flatten time & batch, preserve factored z
            h_tb = h.reshape(T * B, -1)
            z_tb = z.reshape(T * B, K, C)

            critic_logits = critic(h_tb, z_tb)
            values = critic.readout(critic_logits).reshape(T, B)

            # Bootstrap from final state
            bootstrap_logits = critic(h[-1], z[-1])
            bootstrap_value = critic.readout(bootstrap_logits)

            value_seq = torch.zeros(T + 1, B, device=values.device, dtype=values.dtype)
            value_seq[1:] = values
            value_seq[-1] = bootstrap_value

            returns = lambda_return(reward_det, value_seq, discount, lam)
            return symlog(returns)

    returns_symlog = compute_returns_symlog()

    # ---------------------------------------------------------
    # Critic loss
    # ---------------------------------------------------------
    def compute_critic_loss():
        h_tb = h_det.reshape(T * B, -1)
        z_tb = z_det.reshape(T * B, K, C)

        critic_logits = critic(h_tb, z_tb).reshape(T, B, -1)
        return critic.loss(critic_logits, returns_symlog)

    critic_loss = compute_critic_loss()

    # ---------------------------------------------------------
    # Actor loss
    # ---------------------------------------------------------
    def compute_actor_loss():
        # Critic must NOT backprop into actor
        with torch.no_grad():
            h_tb = h.reshape(T * B, -1)
            z_tb = z.reshape(T * B, K, C)

            critic_logits = critic(h_tb, z_tb)
            value_pred = critic.readout(critic_logits).reshape(T, B)

            adv = returns_symlog - symlog(value_pred)

            flat = adv.reshape(-1)
            p5, p95 = torch.quantile(flat, torch.tensor([0.05, 0.95], device=flat.device))
            scale = torch.clamp(p95 - p5, min=1.0)
            adv_norm = adv / scale

        logits = actor(h.reshape(T * B, -1), z.reshape(T * B, K, C)).reshape(T, B, -1)
        dist = Categorical(logits=logits)

        logp = dist.log_prob(action)
        entropy = dist.entropy()

        return -(logp * adv_norm.detach()).mean() - 1e-3 * entropy.mean()

    actor_loss = compute_actor_loss()
    return actor_loss, critic_loss
