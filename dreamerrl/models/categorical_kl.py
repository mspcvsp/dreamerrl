from __future__ import annotations

import torch

from dreamerrl.utils.types import KLConfig


def categorical_kl(q_probs, p_probs, eps=1e-8):
    """
    KL(q || p) per latent factor.
    q_probs, p_probs: (B, K, C)
    Returns: (B, K)
    """
    q = q_probs.clamp_min(eps)
    p = p_probs.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)  # (B, K)


def apply_free_nats(kl_per_factor, free_nats):
    """
    Dreamer‑V3 free‑nats:
      clamp each factor BEFORE summing.
    """
    if free_nats <= 0:
        return kl_per_factor
    return torch.clamp(kl_per_factor, min=free_nats)


def structured_kl(q_probs, p_probs, free_nats=0.0, kl_cfg=KLConfig()):
    """
    Dreamer‑V3 structured KL:
        KL_dyn = KL[ sg(q) || p ]
        KL_rep = KL[ q || sg(p) ]
    """
    # Per-factor KL
    kl_dyn_f = categorical_kl(q_probs.detach(), p_probs)
    kl_rep_f = categorical_kl(q_probs, p_probs.detach())

    # Apply free-nats per factor
    kl_dyn = apply_free_nats(kl_dyn_f, free_nats).mean(dim=-1)
    kl_rep = apply_free_nats(kl_rep_f, free_nats).mean(dim=-1)

    # Validate KL stability
    for name, kl_tensor in [("kl_dyn", kl_dyn), ("kl_rep", kl_rep)]:
        if not torch.isfinite(kl_tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if (kl_tensor < kl_cfg.min_kl).any():
            raise ValueError(f"{name} went negative")
        if kl_tensor.mean() > kl_cfg.max_kl:
            raise ValueError(f"{name} exploded: mean={kl_tensor.mean().item()}")

    return {
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
        "kl_total": kl_dyn + kl_rep,
    }
