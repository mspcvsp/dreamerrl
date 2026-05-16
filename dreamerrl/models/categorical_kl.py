from __future__ import annotations

import torch

from dreamerrl.utils.types import KLConfig


def categorical_kl(q_probs, p_probs, eps=1e-8):
    """
    KL(q || p) per factor.
    q_probs, p_probs: (B, K, C)
    Returns: (B, K)
    """
    q = q_probs.clamp_min(eps)
    p = p_probs.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def apply_free_bits(kl, free_bits):
    if free_bits <= 0:
        return kl
    return torch.clamp(kl, min=free_bits)


def structured_kl(q_probs, p_probs, free_bits=0.0, kl_cfg=KLConfig()):
    """
    Dreamer‑V3 structured KL:
        KL_dyn = KL[ sg(q) || p ]
        KL_rep = KL[ q || sg(p) ]
    """
    kl_dyn = categorical_kl(q_probs.detach(), p_probs)
    kl_dyn = apply_free_bits(kl_dyn, free_bits)

    kl_rep = categorical_kl(q_probs, p_probs.detach())
    kl_rep = apply_free_bits(kl_rep, free_bits)

    # Validate KL stability
    for name, kl_tensor in [("kl_dyn", kl_dyn), ("kl_rep", kl_rep)]:
        if not torch.isfinite(kl_tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if (kl_tensor < kl_cfg.min_kl).any():
            raise ValueError(f"{name} went negative")
        if kl_tensor.mean() > kl_cfg.max_kl:
            raise ValueError(f"{name} exploded: mean={kl_tensor.mean().item()}")
        if kl_cfg.require_nonzero and kl_tensor.mean() == 0:
            raise ValueError(f"{name} collapsed to zero")

    return {
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
        "kl_total": kl_dyn + kl_rep,
    }
