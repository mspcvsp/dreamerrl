from __future__ import annotations

from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.replay_buffer import DreamerReplayBuffer


class TestDreamerTrainer:
    """
    Minimal trainer used ONLY for unit tests.
    No env, no wandb, no cfg.
    """

    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor | None,
        critic: ValueHead | None,
        replay_buffer: DreamerReplayBuffer,
        device: torch.device,
    ):
        self.world = world_model
        self.actor = actor
        self.critic = critic
        self.replay = replay_buffer
        self.device = device

    # ---------------- World model training step ----------------
    def world_model_training_step(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        batch = self.replay.sample(batch_size, seq_len, device=self.device)

        obs = batch["state"]  # (B, L, obs_dim)
        reward = batch["reward"]  # (B, L)
        B, L, _ = obs.shape

        state = self.world.init_state(B)

        recon_losses = []
        reward_losses = []
        kl_losses = []

        for t in range(L):
            out = self.world.observe_step(state, obs[:, t])
            state = out["state"]

            recon_losses.append(((out["recon"] - obs[:, t]) ** 2).mean())
            reward_losses.append(((out["reward_pred"].squeeze(-1) - reward[:, t]) ** 2).mean())
            kl_losses.append(out["kl"])

        loss = torch.stack(recon_losses).mean() + torch.stack(reward_losses).mean() + torch.stack(kl_losses).mean()

        return {"loss": loss}

    # ---------------- Imagination rollout ----------------
    def imagination_rollout(self, state, horizon: int):
        hs, zs, values, actions = [], [], [], []

        for _ in range(horizon):
            state = self.world.imagine_step(state)
            hs.append(state.h)
            zs.append(state.z)

            if self.actor is not None:
                logits = self.actor(state.h, state.z)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                actions.append(act)

            if self.critic is not None:
                values.append(self.critic(state.h, state.z))

        return {
            "h": torch.stack(hs),
            "z": torch.stack(zs),
            "value": torch.stack(values) if values else None,
            "action": torch.stack(actions) if actions else None,
        }


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: float,
    lam: float,
) -> torch.Tensor:
    """
    λ-return blends short-horizon TD and long-horizon Monte Carlo:

    TD(0) target (λ = 0):
        r_t + γ V(s_{t+1})

    Monte Carlo target (λ = 1):
        r_t + γ r_{t+1} + γ² r_{t+2} + ...

    λ-return mixes all n-step returns with exponentially decaying weights:

       G_t^λ = (1-λ) * [1-step]
               + λ(1-λ) * [2-step]
               + λ²(1-λ) * [3-step]
               + ...

    Visual intuition:

        r_t      r_{t+1}      r_{t+2}      r_{t+3}      ...
        |----------|-----------|-----------|-----------|
        | 1-step   | 2-step    | 3-step    | 4-step    |

        weight: (1-λ)   λ(1-λ)    λ²(1-λ)     λ³(1-λ)   ...

    λ = 0 → trust critic immediately (low variance, high bias)
    λ = 1 → trust full rollout (high variance, low bias)
    0 < λ < 1 → smooth bias–variance tradeoff for stable value learning
    ----------------------------------------------------------------------
    reward: (B, T)
    value:  (B, T+1) or (B, T) depending on usage
    Returns: (B, T)
    """
    B, T = reward.shape
    ret = torch.zeros_like(reward)

    next_val = value[:, -1]
    for t in reversed(range(T)):
        delta = reward[:, t] + discount * next_val - value[:, t]
        next_val = value[:, t] + lam * delta
        ret[:, t] = next_val

    return ret
