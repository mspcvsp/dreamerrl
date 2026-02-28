from __future__ import annotations

from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.replay_buffer import DreamerReplayBuffer


class _TestDreamerTrainer:
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
    λ-return (time-major):

    reward: (T, B)
    value:  (T+1, B

    G_t^λ blends TD(0) and Monte Carlo:

       TD(0):        r_t + γ V_{t+1}
       Monte Carlo:  r_t + γ r_{t+1} + γ² r_{t+2} + ...

    λ mixes n-step returns with exponentially decaying weights:

       G_t^λ = (1-λ)*G_t^{1-step}
             + λ(1-λ)*G_t^{2-step}
             + λ²(1-λ)*G_t^{3-step}
             + ...

    λ = 0 → trust critic (low variance, high bias)
    λ = 1 → trust rollout (high variance, low bias)

    Time-major rollout:
    ------------------
    t = 0      1      2      ...    T-1      T
    |------|------|------|------|------|------|
    s0     s1     s2     ...    s(T-1)  sT
    r0     r1     r2     ...    r(T-1)

    Values:
    V(s0)  V(s1)  V(s2)  ...    V(s(T-1))  V(sT)
    <----------- T+1 values ------------->

    λ-return needs V(s_{t+1}) for every t, so value must be (T+1, B)
    """
    T, B = reward.shape
    ret = torch.zeros_like(reward)

    next_val = value[-1]  # (B,)
    for t in reversed(range(T)):
        delta = reward[t] + discount * next_val - value[t]
        next_val = value[t] + lam * delta
        ret[t] = next_val

    return ret
