from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from dreamerrl.utils.transforms import symlog


def world_model_training_step(
    world_model,
    batch: Dict[str, torch.Tensor],
    kl_scale: float = 1.0,
) -> torch.Tensor:
    device = next(world_model.parameters()).device

    # -------------------------------------------------------------
    # Extract batch
    # -------------------------------------------------------------
    if "obs" in batch:
        obs = batch["obs"].to(device)
    elif "state" in batch:
        obs = batch["state"].to(device)
    else:
        raise KeyError("Batch must contain 'obs' or 'state'")

    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    cont_target = 1.0 - done  # continuation target

    B, L, _ = obs.shape

    # -------------------------------------------------------------
    # Roll out RSSM over sequence
    # -------------------------------------------------------------
    state = world_model.init_state(B)
    posts = []
    priors = []

    for t in range(L):
        # Convert discrete action IDs → one-hot
        action_t = batch["action"][:, t]  # (B,)
        action_one_hot = F.one_hot(action_t, num_classes=world_model.net_cfg.action_dim).float()  # (B, action_dim)

        out = world_model.observe_step(
            prev_state=state,
            obs=obs[:, t],
            action=action_one_hot,
            reward=batch["reward"][:, t],
        )

        state = out["post"]
        posts.append(out["post"].post_stats)
        priors.append(out["prior"].prior_stats)

    # (B, L, deter), (B, L, K, C)
    h = torch.stack([s["h"] for s in posts], dim=1)
    z = torch.stack([s["z"] for s in posts], dim=1)

    # -------------------------------------------------------------
    # Flatten for heads
    # -------------------------------------------------------------
    z_factored = z.reshape(
        B * L,
        world_model.latent.stoch_size,
        world_model.latent.num_classes,
    )
    h_flat = h.reshape(B * L, -1)

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------
    recon = world_model.decoder(h_flat, z_factored).reshape(B, L, -1)

    reward_logits = world_model.reward_head(h_flat, z_factored).reshape(B, L, world_model.net_cfg.value_bins)

    cont_logits = world_model.continue_head(h_flat, z_factored).reshape(B, L, world_model.net_cfg.value_bins)

    # -------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------
    recon_target = symlog(obs)
    recon_loss = F.mse_loss(recon, recon_target)

    reward_loss = world_model.reward_head.loss_from_logits(reward_logits, reward)
    cont_loss = world_model.continue_head.loss_from_logits(cont_logits, cont_target)

    L_pred = recon_loss + reward_loss + cont_loss

    # -------------------------------------------------------------
    # KL losses (already computed in observe_step)
    # -------------------------------------------------------------
    kl_dyn = torch.stack([p["kl_dyn"] for p in posts], dim=1).mean()
    kl_rep = torch.stack([p["kl_rep"] for p in posts], dim=1).mean()

    return L_pred + kl_scale * (kl_dyn + kl_rep)
