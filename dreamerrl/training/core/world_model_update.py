from __future__ import annotations

from typing import Dict

import torch

from dreamerrl.models.world_model import WorldModel


def world_model_training_step(
    world_model: WorldModel,
    batch: Dict[str, torch.Tensor],
    kl_scale: float,
) -> torch.Tensor:
    obs = batch["state"]  # (B, L, obs_dim)
    reward = batch["reward"]  # (B, L)
    B, L, _ = obs.shape

    state = world_model.init_state(B)

    recon_losses = []
    reward_losses = []
    kl_losses = []

    for t in range(L):
        out = world_model.observe_step(state, obs[:, t])
        state = out["state"]

        recon_losses.append(((out["recon"] - obs[:, t]) ** 2).mean())
        reward_losses.append(((out["reward_pred"].squeeze(-1) - reward[:, t]) ** 2).mean())
        kl_losses.append(out["kl"])

    recon_loss = torch.stack(recon_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()

    return recon_loss + reward_loss + kl_scale * kl_loss
