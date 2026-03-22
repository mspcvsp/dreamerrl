import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
