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

    if "obs" in batch:
        obs = batch["obs"].to(device)
    elif "state" in batch:
        obs = batch["state"].to(device)
    else:
        raise KeyError("Batch must contain 'obs' or 'state'")

    reward = batch["reward"].to(device)
    is_terminal = batch["is_terminal"].to(device)
    cont = 1.0 - is_terminal.float()

    B, L, _ = obs.shape

    state = world_model.init_state(B)
    posts, priors = [], []

    for t in range(L):
        out = world_model.observe_step(
            prev_state=state,
            obs=obs[:, t],
            action=batch["action"][:, t],
            reward=batch["reward"][:, t],
            is_first=batch["is_first"][:, t],
            is_last=batch["is_last"][:, t],
            is_terminal=batch["is_terminal"][:, t],
        )
        state = out["post"]
        posts.append(out["post"].post_stats)
        priors.append(out["prior"].prior_stats)

    h = torch.stack([s["h"] for s in posts], dim=1)
    z = torch.stack([s["z"] for s in posts], dim=1)

    recon = world_model.decoder(h.reshape(B * L, -1), z.reshape(B * L, -1)).reshape(B, L, -1)
    reward_logits = world_model.reward_head(h.reshape(B * L, -1), z.reshape(B * L, -1)).reshape(B, L, -1)
    cont_logits = world_model.continue_head(h.reshape(B * L, -1), z.reshape(B * L, -1)).reshape(B, L)

    recon_target = symlog(obs)
    recon_loss = F.mse_loss(recon, recon_target)

    reward_loss = world_model.reward_head.loss(reward_logits, reward)
    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont)

    L_pred = recon_loss + reward_loss + cont_loss

    # Use KL terms already computed in observe_step via structured_kl
    kl_dyn = torch.stack([p["kl_dyn"] for p in posts], dim=1)  # (B, L)
    kl_rep = torch.stack([p["kl_rep"] for p in posts], dim=1)  # (B, L)

    L_dyn = kl_dyn.mean()
    L_rep = kl_rep.mean()

    loss = L_pred + kl_scale * (L_dyn + L_rep)
    return loss
