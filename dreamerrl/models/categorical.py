"""Dreamer categorical distributions and their straight-through estimator."""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def mixed_probs(logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
    if not 0 <= unimix <= 1:
        raise ValueError("unimix must be in [0, 1]")
    probs = logits.float().softmax(-1)
    return (1 - unimix) * probs + unimix / logits.shape[-1]


def sample_onehot(probs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    """Hard forward sample; backward is the mixed-probability Jacobian."""
    indices = probs.argmax(-1) if deterministic else Categorical(probs=probs.detach()).sample()
    hard = F.one_hot(indices, probs.shape[-1]).to(probs.dtype)
    return hard + (probs - probs.detach())
