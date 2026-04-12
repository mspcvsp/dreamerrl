from __future__ import annotations

from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import DreamerReplayBuffer

# --- Core functions (single source of truth) ---
from dreamerrl.training.core import (
    imagine_trajectory_for_training,
    lambda_return,
    world_model_training_step,
)


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
        # For training, actor and critic MUST be provided
        assert self.actor is not None, "Training imagination requires an Actor"
        assert self.critic is not None, "Training imagination requires a Critic"

        return imagine_trajectory_for_training(
            world_model=self.world,
            actor=self.actor,
            critic=self.critic,
            state=state,
            horizon=horizon,
        )


# Re-export λ-return for tests
lambda_return = lambda_return
