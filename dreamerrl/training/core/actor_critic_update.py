"""
Dreamer-V3 actor–critic update (intuitive summary)
--------------------------------------------------

High-level picture:
- The world model imagines trajectories in latent space using the current actor.
- The critic learns to predict *distributional* returns (two-hot over symlog values).
- The actor learns to choose actions that maximize those returns, using normalized advantages.

Key ideas:

1) Imagination in latent space
   We never roll out in pixel/observation space. Instead, we:
   - start from a latent world-model state,
   - repeatedly decode reward from the latent state,
   - sample actions from the actor,
   - advance the latent state with world_model.imagine_step().
   This gives us (T, B, ...) tensors for h, z, reward, action.

2) λ-returns with bootstrap
   The critic is trained on λ-returns, which blend:
   - TD(0): r_t + γ V(s_{t+1})
   - Monte Carlo: r_t + γ r_{t+1} + γ² r_{t+2} + ...
   λ controls the bias–variance tradeoff.
   We compute scalar λ-returns from:
   - reward: (T, B)
   - value:  (T+1, B) from the critic over imagined states + a bootstrap value.

3) Distributional critic (two-hot over symlog)
   The critic outputs logits over value bins.
   We:
   - decode scalar values from logits for λ-returns and advantages,
   - encode target returns as two-hot distributions in symlog space,
   - train the critic with cross-entropy between target two-hot and predicted logits.

4) Actor with normalized advantages
   The actor is trained to maximize expected return via policy gradients:
   - advantages = (target returns) - (critic prediction),
   - we normalize advantages by a robust scale (p5–p95) to stabilize learning,
   - actor loss = -E[log π(a|s) * adv_norm] - entropy_weight * H[π].

This function wires all of that together in a way that mirrors the Dreamer-V3 paper:
- imagination → λ-returns → critic loss (two-hot) → actor loss (advantage-weighted log-prob).
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
    Dreamer-V3 actor–critic update:

      1) Imagine trajectories in latent space with the current actor.
      2) Compute λ-returns from imagined rewards and critic values.
      3) Train the critic to match λ-returns (two-hot, symlog space).
      4) Train the actor to maximize normalized advantages.

    All tensors are time-major: (T, B, ...).
    """

    # ---------------------------------------------------------
    # 0. Imagination rollout (latent space)
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
        """
        Compute scalar λ-returns from imagined rewards and critic values, then
        transform them with symlog for distributional training.

        We:
        - evaluate the critic on all imagined states to get V(s_t),
        - get a bootstrap value from the final state,
        - build a (T+1, B) value sequence for lambda_return(),
        - apply symlog to the resulting returns.
        """
        with torch.no_grad():
            # Critic over all imagined states (T, B, num_bins)
            critic_logits = critic(
                h.reshape(T * B, -1),
                z.reshape(T * B, -1),
            ).reshape(T, B, -1)

            values = value_from_logits(critic_logits)  # (T, B)

            # Bootstrap from final state
            bootstrap_logits = critic(h[-1], z[-1])  # (B, num_bins)
            bootstrap_value = value_from_logits(bootstrap_logits)  # (B,)

            # Build (T+1, B) value sequence:
            # value_seq[t+1] = V(s_{t+1}), matching lambda_return's contract.
            value_seq = torch.zeros(T + 1, B, device=values.device, dtype=values.dtype)
            value_seq[1:] = values
            value_seq[-1] = bootstrap_value  # consistent final bootstrap

            returns = lambda_return(
                reward=reward,  # (T, B)
                value=value_seq,  # (T+1, B)
                discount=discount,
                lam=lam,
            )  # (T, B)

            return symlog(returns)

    returns_symlog = compute_returns()  # (T, B)

    # ---------------------------------------------------------
    # 2. Closure: critic loss (two-hot over symlog returns)
    # ---------------------------------------------------------
    def compute_critic_loss():
        """
        Train the critic to predict the symlog λ-returns as a two-hot distribution
        over value bins, using cross-entropy loss.
        """
        critic_logits = critic(
            h.reshape(T * B, -1),
            z.reshape(T * B, -1),
        ).reshape(T, B, -1)  # (T, B, num_bins)

        target_twohot = twohot_encode(returns_symlog)  # (T, B, num_bins)
        log_probs = torch.log_softmax(critic_logits, dim=-1)

        # Cross-entropy between target two-hot and predicted logits
        return -(target_twohot * log_probs).sum(dim=-1).mean()

    critic_loss = compute_critic_loss()

    # ---------------------------------------------------------
    # 3. Closure: actor loss (advantage-weighted log-prob + entropy)
    # ---------------------------------------------------------
    def compute_actor_loss():
        """
        Train the actor to maximize normalized advantages:

          L_actor = -E[ log π(a|s) * adv_norm ] - α * H[π]

        where:
        - adv = returns_symlog - symlog(V_pred),
        - adv_norm is robustly normalized using the 5th–95th percentile range,
        - entropy term encourages exploration.
        """
        with torch.no_grad():
            critic_logits = critic(
                h.reshape(T * B, -1),
                z.reshape(T * B, -1),
            ).reshape(T, B, -1)

            value_pred = value_from_logits(critic_logits)  # (T, B)
            adv = returns_symlog - symlog(value_pred)  # (T, B)

            # Robust advantage normalization (p5–p95 range)
            flat = adv.reshape(-1)
            p5, p95 = torch.quantile(
                flat,
                torch.tensor([0.05, 0.95], device=flat.device),
            )
            scale = torch.clamp(p95 - p5, min=1.0)
            adv_norm = adv / scale

        logits = actor(
            h.reshape(T * B, -1),
            z.reshape(T * B, -1),
        ).reshape(T, B, -1)

        dist = Categorical(logits=logits)
        logp = dist.log_prob(action)  # (T, B)
        entropy = dist.entropy()  # (T, B)

        # Policy gradient with entropy regularization
        return -(logp * adv_norm.detach()).mean() - 1e-3 * entropy.mean()

    actor_loss = compute_actor_loss()

    return actor_loss, critic_loss
