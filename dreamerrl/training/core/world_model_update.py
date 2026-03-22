from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from dreamerrl.models.reward_head import RewardHead
from dreamerrl.utils.transforms import symlog


def world_model_training_step(
    world_model,
    batch: Dict[str, torch.Tensor],
    kl_scale: float = 1.0,
    free_nats: float = 1.0,
) -> torch.Tensor:
    """
    Dreamer-V3 world model update:
      L = pred + kl_scale * (dyn + rep)

    pred = recon + reward + continue
    dyn  = KL[ sg(q) || p ]
    rep  = KL[ q || sg(p) ]
    """

    device = next(world_model.parameters()).device

    obs = batch["state"].to(device)  # (B, L, obs_dim)
    reward = batch["reward"].to(device)  # (B, L)
    is_terminal = batch["is_terminal"].to(device)  # (B, L)
    cont = 1.0 - is_terminal.float()  # (B, L)

    B, L, _ = obs.shape

    # ---------------------------------------------------------
    # RSSM observe sequence
    # ---------------------------------------------------------
    state = world_model.init_state(B)
    posts = []
    priors = []

    for t in range(L):
        out = world_model.observe_step(state, obs[:, t])
        state = out["state"]
        posts.append(out["state"].post_stats)
        priors.append(out["state"].prior_stats)

    # ---------------------------------------------------------
    # Decode from posterior states
    # ---------------------------------------------------------
    h = torch.stack([s["h"] for s in posts], dim=1)  # (B, L, deter)
    z = torch.stack([s["z"] for s in posts], dim=1)  # (B, L, stoch)

    recon = world_model.decoder(
        h.reshape(B * L, -1),
        z.reshape(B * L, -1),
    ).reshape(B, L, -1)

    reward_logits = world_model.reward_head(
        h.reshape(B * L, -1),
        z.reshape(B * L, -1),
    ).reshape(B, L, -1)

    cont_logits = world_model.continue_head(
        h.reshape(B * L, -1),
        z.reshape(B * L, -1),
    ).reshape(B, L)

    # ---------------------------------------------------------
    # Prediction losses
    # ---------------------------------------------------------
    recon_target = symlog(obs)
    recon_loss = F.mse_loss(recon, recon_target)

    reward_loss = RewardHead.loss_from_logits(reward_logits, reward)

    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont)

    L_pred = recon_loss + reward_loss + cont_loss

    # ---------------------------------------------------------
    # KL losses (dyn + rep)
    # ---------------------------------------------------------
    kl_terms = []
    for post, prior in zip(posts, priors):
        kl = world_model.kl_divergence(post, prior)  # scalar
        kl_terms.append(kl)

    kl = torch.stack(kl_terms).mean()

    # free bits
    dyn = torch.clamp(kl.detach(), min=free_nats)
    rep = torch.clamp(kl, min=free_nats)

    L_dyn = dyn
    L_rep = rep

    # ---------------------------------------------------------
    # Total loss
    # ---------------------------------------------------------
    loss = L_pred + kl_scale * (L_dyn + L_rep)

    return loss
