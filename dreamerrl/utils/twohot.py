import torch


def twohot_encode(targets, bins):
    # targets: (B,)
    # bins: (V,)
    B, V = targets.shape[0], bins.shape[0]

    # Clamp targets to bin range
    t = torch.clamp(targets, bins[0], bins[-1])

    # Find right bin
    idx = torch.searchsorted(bins, t, right=True).clamp(1, V - 1)

    left = idx - 1
    right = idx

    left_bin = bins[left]
    right_bin = bins[right]

    # Linear interpolation weights
    denom = (right_bin - left_bin).clamp(min=1e-8)
    w_right = (t - left_bin) / denom
    w_left = 1.0 - w_right

    out = torch.zeros(B, V, device=targets.device)
    out[torch.arange(B), left] = w_left
    out[torch.arange(B), right] = w_right

    return out


def value_from_logits(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)
