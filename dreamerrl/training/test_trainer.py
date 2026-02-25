from typing import Any, Dict

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.replay_buffer import DreamerReplayBuffer


class TestDreamerTrainer:
    """
    Minimal trainer used ONLY for unit tests.
    Does NOT initialize env, wandb, schedulers, or cfg.
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

    # -------------------------------------------------------------
    # World model training step
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Imagination rollout
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # λ-return
    # -------------------------------------------------------------
    @staticmethod
    def lambda_return(reward, value, discount, lam):
        T, B = reward.shape
        ret = torch.zeros_like(reward)

        next_val = value[-1]
        for t in reversed(range(T)):
            delta = reward[t] + discount * next_val - value[t]
            next_val = value[t] + lam * delta
            ret[t] = next_val

        return ret
