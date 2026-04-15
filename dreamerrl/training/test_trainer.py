from __future__ import annotations

import torch

from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.core.world_model_update import world_model_training_step


class _TestDreamerTrainer:
    """
    Minimal trainer used only for unit tests.
    Provides:
      - training_step(batch_size, seq_len)
      - world_model_update(batch_size, seq_len)
    """

    def __init__(self, world_model, actor, critic, replay_buffer, device):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer  # kept for signature compatibility
        self.device = device

        # No pre-filling of replay_buffer; tests don't require it.

    # ---------------------------------------------------------
    # Actor–critic update (Dreamer-V3)
    # ---------------------------------------------------------
    def training_step(self, batch_size: int, seq_len: int):
        B, T = batch_size, seq_len
        obs_dim = self.world_model.flat_obs_dim

        # Synthetic batch in the format expected by actor_critic_update
        batch = {
            "state": torch.randn(B, T, obs_dim, device=self.device),
            "reward": torch.randn(B, T, device=self.device),
        }

        actor_loss, critic_loss = actor_critic_update(
            world_model=self.world_model,
            actor=self.actor,
            critic=self.critic,
            batch=batch,
            imagination_horizon=5,
            discount=0.99,
            lam=0.95,
        )

        return actor_loss + critic_loss

    # ---------------------------------------------------------
    # World model update (Dreamer-V3)
    # ---------------------------------------------------------
    def world_model_update(self, batch_size: int, seq_len: int):
        B, T = batch_size, seq_len
        obs_dim = self.world_model.flat_obs_dim

        batch = {
            "obs": torch.randn(B, T, obs_dim, device=self.device),
            "action": torch.zeros(B, T, dtype=torch.long, device=self.device),
            "reward": torch.randn(B, T, device=self.device),
            "is_first": torch.zeros(B, T, device=self.device),
            "is_last": torch.zeros(B, T, device=self.device),
            "is_terminal": torch.zeros(B, T, device=self.device),
        }

        loss = world_model_training_step(self.world_model, batch)
        return {"loss": loss}
