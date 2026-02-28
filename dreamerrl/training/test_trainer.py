from __future__ import annotations

from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel

# --- Core functions (single source of truth) ---
from dreamerrl.training.core import (
    imagination_rollout,
    lambda_return,
    world_model_training_step,
)
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

        loss = world_model_training_step(
            world_model=self.world,
            batch=batch,
            kl_scale=1.0,  # tests use equal weighting
        )

        return {"loss": loss}

    # ---------------- Imagination rollout ----------------
    def imagination_rollout(self, state, horizon: int):
        return imagination_rollout(
            world_model=self.world,
            actor=self.actor,
            critic=self.critic,
            state=state,
            horizon=horizon,
        )


# Re-export λ-return for tests
lambda_return = lambda_return
