import torch

from .transforms import symexp

# Bins in symlog space, then mapped back with symexp for readout
# You can tune range/step later.
_SYMLOG_BINS = torch.linspace(-10.0, 10.0, steps=41)  # 41 bins
BINS = symexp(_SYMLOG_BINS)  # (num_bins,)


def twohot_encode(
    y: torch.Tensor,
    bins: torch.Tensor | None = None,
) -> torch.Tensor:
    if bins is None:
        bins = BINS.to(y.device)

    y = y.unsqueeze(-1)  # (..., 1)
    diff = torch.abs(y - bins)  # (..., num_bins)
    idx = torch.argmin(diff, dim=-1)  # (...,)

    # Use squeezed y here so shapes match idx
    delta = torch.sign(y.squeeze(-1) - bins[idx])  # (...,)
    idx2 = idx + delta.to(torch.long)
    idx2 = torch.clamp(idx2, 0, bins.numel() - 1)

    b1 = bins[idx]
    b2 = bins[idx2]
    denom = torch.clamp(torch.abs(b2 - b1), min=1e-8)
    w2 = torch.abs(y.squeeze(-1) - b1) / denom
    w1 = 1.0 - w2

    out = torch.zeros(*idx.shape, bins.numel(), device=y.device)
    out.scatter_(-1, idx.unsqueeze(-1), w1.unsqueeze(-1))
    out.scatter_(-1, idx2.unsqueeze(-1), w2.unsqueeze(-1))
    return out


def value_from_logits(logits: torch.Tensor, bins: torch.Tensor | None = None) -> torch.Tensor:
    """
    logits: (..., num_bins)
    returns: (...,) expected value under softmax(logits) over bins
    """
    if bins is None:
        bins = BINS.to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)
