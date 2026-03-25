from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from dreamerrl.models.reward_head import RewardHead
from dreamerrl.utils.transforms import symlog


def _gaussian_kl(mean_q, std_q, mean_p, std_p):
    """
    KL(q || p) for diagonal Gaussians.
    mean_*, std_*: (..., latent_dim)
    returns: (..., latent_dim)
    """
    var_q = std_q**2
    var_p = std_p**2

    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
    return kl  # no reduction


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

    Free bits are applied per latent dimension before reduction.
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
        posts.append(out["state"].post_stats)  # each: {"mean", "std", "h", "z", ...}
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
    # Structured KL (dyn + rep) with free bits
    # ---------------------------------------------------------
    post_mean = torch.stack([p["mean"] for p in posts], dim=1)  # (B, L, D)
    post_std = torch.stack([p["std"] for p in posts], dim=1)  # (B, L, D)
    prior_mean = torch.stack([p["mean"] for p in priors], dim=1)  # (B, L, D)
    prior_std = torch.stack([p["std"] for p in priors], dim=1)  # (B, L, D)

    # KL_dyn = KL[ sg(q) || p ]
    kl_dyn = _gaussian_kl(
        mean_q=post_mean.detach(),
        std_q=post_std.detach(),
        mean_p=prior_mean,
        std_p=prior_std,
    )  # (B, L, D)

    # KL_rep = KL[ q || sg(p) ]
    kl_rep = _gaussian_kl(
        mean_q=post_mean,
        std_q=post_std,
        mean_p=prior_mean.detach(),
        std_p=prior_std.detach(),
    )  # (B, L, D)

    # Free bits per latent dimension
    latent_dim = kl_dyn.shape[-1]
    fb_per_dim = free_nats / latent_dim

    kl_dyn = torch.clamp(kl_dyn, min=fb_per_dim)
    kl_rep = torch.clamp(kl_rep, min=fb_per_dim)

    # Reduce over latent dim, time, batch
    L_dyn = kl_dyn.sum(dim=-1).mean()  # scalar
    L_rep = kl_rep.sum(dim=-1).mean()  # scalar

    # ---------------------------------------------------------
    # Total loss
    # ---------------------------------------------------------
    loss = L_pred + kl_scale * (L_dyn + L_rep)

    return loss
