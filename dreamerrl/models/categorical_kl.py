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


def apply_free_nats(kl, free_nats):
    if free_nats <= 0:
        return kl

    return torch.clamp(kl, min=free_nats)


def structured_kl(q_probs, p_probs, kl_cfg=KLConfig()):
    """
    Dreamer‑V3 structured KL:
        KL_dyn = KL[ sg(q) || p ]
        KL_rep = KL[ q || sg(p) ]
    """
    # Per-factor KL
    raw_kl_dyn = categorical_kl(q_probs.detach(), p_probs)
    raw_kl_rep = categorical_kl(q_probs, p_probs.detach())

    # Floor the summed categorical KL for each example.
    kl_dyn = apply_free_nats(raw_kl_dyn.sum(dim=-1), kl_cfg.free_nats)
    kl_rep = apply_free_nats(raw_kl_rep.sum(dim=-1), kl_cfg.free_nats)

    # Validate KL stability
    for name, kl_tensor in [("kl_dyn", kl_dyn), ("kl_rep", kl_rep)]:
        if not torch.isfinite(kl_tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if (kl_tensor < kl_cfg.min_kl).any():
            raise ValueError(f"{name} went negative")
        if kl_tensor.mean() > kl_cfg.max_kl:
            raise ValueError(f"{name} exploded: mean={kl_tensor.mean().item()}")

    return {
        "raw_kl_dyn": raw_kl_dyn,
        "raw_kl_rep": raw_kl_rep,
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
        "kl_total": kl_dyn + kl_rep,
    }
