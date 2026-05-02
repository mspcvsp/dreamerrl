from __future__ import annotations

from typing import Dict

import torch


def categorical_kl(
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Categorical KL(q || p) per factor.

    Args:
        q_probs: (B, stoch_size, num_classes)
        p_probs: (B, stoch_size, num_classes)
    Returns:
        kl: (B, stoch_size) KL per factor (sum over classes)
    """
    q = q_probs.clamp_min(eps)
    p = p_probs.clamp_min(eps)
    kl = (q * (q.log() - p.log())).sum(dim=-1)  # sum over classes
    return kl.sum(dim=-1)


def apply_free_bits(
    kl: torch.Tensor,
    free_bits: float,
) -> torch.Tensor:
    """
    Apply free bits per factor.

    Args:
        kl: (B, stoch_size)
        free_bits: scalar (in nats)
    Returns:
        kl_fb: (B, stoch_size) with free bits applied
    """
    if free_bits <= 0.0:
        return kl
    return torch.clamp(kl, min=free_bits)


def structured_kl(
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    free_bits: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Dreamer-style structured KL with dyn/rep split and optional free bits.

    dyn = KL[ sg(q) || p ]
    rep = KL[ q || sg(p) ]

    Args:
        q_probs: (B, stoch_size, num_classes)
        p_probs: (B, stoch_size, num_classes)
    Returns:
        {
            "kl_dyn":  scalar mean over batch and factors
            "kl_rep":  scalar mean over batch and factors
            "kl_total": scalar mean (dyn + rep)
        }
    """
    # KL[ sg(q) || p ]
    kl_dyn = categorical_kl(q_probs.detach(), p_probs)
    kl_dyn = apply_free_bits(kl_dyn, free_bits)

    # KL[ q || sg(p) ]
    kl_rep = categorical_kl(q_probs, p_probs.detach())
    kl_rep = apply_free_bits(kl_rep, free_bits)

    kl_dyn_mean = kl_dyn.mean()
    kl_rep_mean = kl_rep.mean()
    kl_total_mean = kl_dyn_mean + kl_rep_mean

    return {
        "kl_dyn": kl_dyn_mean,
        "kl_rep": kl_rep_mean,
        "kl_total": kl_total_mean,
    }
