import torch


def twohot_encode(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    y = y.unsqueeze(-1)  # (..., 1)
    diff = torch.abs(y - bins)  # (..., num_bins)
    idx = torch.argmin(diff, dim=-1)  # (...)

    delta = torch.sign(y.squeeze(-1) - bins[idx])
    idx2 = torch.clamp(idx + delta.to(torch.long), 0, bins.numel() - 1)

    b1 = bins[idx]
    b2 = bins[idx2]
    denom = torch.clamp(torch.abs(b2 - b1), min=1e-8)
    w2 = torch.abs(y.squeeze(-1) - b1) / denom
    w1 = 1.0 - w2

    out = torch.zeros(*idx.shape, bins.numel(), device=y.device)
    out.scatter_(-1, idx.unsqueeze(-1), w1.unsqueeze(-1))
    out.scatter_(-1, idx2.unsqueeze(-1), w2.unsqueeze(-1))
    return out


def value_from_logits(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)
