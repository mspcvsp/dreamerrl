import torch


def twohot_encode(targets, bins):
    """
    Dreamer‑V3 produces value targets with shape (T, B), not just (B,), so this function flattens the targets,
    computes two‑hot weights in 1‑D, and reshapes the result back. This guarantees correct behavior for multi‑step
    returns, imagined rollouts, and any higher‑rank value tensor without extra logic.
    """
    orig_shape = targets.shape
    flat = targets.reshape(-1)

    V = bins.shape[0]
    B = flat.shape[0]

    # Clamp
    t = torch.clamp(flat, bins[0], bins[-1])

    # Bin indices
    idx = torch.searchsorted(bins, t, right=True).clamp(1, V - 1)
    left = idx - 1
    right = idx

    left_bin = bins[left]
    right_bin = bins[right]

    denom = (right_bin - left_bin).clamp(min=1e-8)
    w_right = (t - left_bin) / denom
    w_left = 1.0 - w_right

    out = torch.zeros(B, V, device=targets.device)
    out[torch.arange(B), left] = w_left
    out[torch.arange(B), right] = w_right

    return out.reshape(*orig_shape, V)


def value_from_logits(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)
