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
        self.replay_buffer = replay_buffer
        self.device = device  # Pre-fill replay buffer with minimal episodes for tests

        # Pre-fill replay buffer with minimal valid episodes
        num_envs = self.replay_buffer.num_envs
        obs_dim = self.replay_buffer.obs_dim

        for _ in range(5):
            trans = {
                "state": torch.randn(num_envs, obs_dim),
                "action": torch.zeros(num_envs, dtype=torch.long),
                "reward": torch.zeros(num_envs),
                "is_first": torch.zeros(num_envs),
                "is_last": torch.ones(num_envs),
                "is_terminal": torch.zeros(num_envs),
            }
            self.replay_buffer.add_batch(trans)

    # ---------------------------------------------------------
    # Actor–critic update (Dreamer-V3)
    # ---------------------------------------------------------
    def training_step(self, batch_size: int, seq_len: int):
        batch = self.replay_buffer.sample(batch_size, seq_len, device=self.device)

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
        batch = self.replay_buffer.sample(batch_size, seq_len, device=self.device)
        return world_model_training_step(self.world_model, batch)
