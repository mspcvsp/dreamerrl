from __future__ import annotations

from typing import Dict

import torch

from dreamerrl.utils.types import KLConfig


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
    q_probs: torch.Tensor, p_probs: torch.Tensor, free_bits: float = 0.0, kl_cfg: KLConfig = KLConfig()
) -> Dict[str, torch.Tensor]:
    """
    Dreamer-style structured KL with dyn/rep split and optional free bits.

    dyn = KL[ sg(q) || p ]
    rep = KL[ q || sg(p) ]

    Args:
        q_probs: (B, stoch_size, num_classes)
        p_probs: (B, stoch_size, num_classes)
        free_bits: scalar (in nats)
        kl_cfg: KLConfig object with max_kl, min_kl, require_nonzero

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

    # Invariants reminder:
    # --------------------
    # Dreamer‑V3 splits KL into two terms:
    #   • KL_dyn = "Did my dynamics predict the right latent?"
    #   • KL_rep = "Did my encoder add extra information?"
    #
    # These KLs are the *health signals* of the world model:
    #   • KL_dyn exploding  → RSSM transition is unstable or diverging.
    #   • KL_rep exploding  → encoder is overfitting or leaking future info.
    #   • Either KL going negative → numerical instability (log underflow).
    #   • Either KL collapsing to zero → dead latent model (posterior collapse).
    #
    # We validate KL BEFORE reducing to scalars because a single latent factor
    # can explode while the mean stays small. Catching this early prevents
    # silent corruption of the world model and actor/critic training.
    for name, kl_tensor in [("kl_dyn", kl_dyn), ("kl_rep", kl_rep)]:
        if not torch.isfinite(kl_tensor).all():
            raise ValueError(f"{name} contains NaN or Inf: {kl_tensor}")

        if (kl_tensor < kl_cfg.min_kl).any():
            raise ValueError(f"{name} went negative: {kl_tensor}")

        if kl_tensor.mean() > kl_cfg.max_kl:
            raise ValueError(f"{name} exploded: mean={kl_tensor.mean().item()}")

        if kl_cfg.require_nonzero and kl_tensor.mean() == 0:
            raise ValueError(f"{name} collapsed to zero")

    return {
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
        "kl_total": kl_dyn + kl_rep,
    }
