import torch
from torch import nn


def init_weights(module):
    # Deterministic initialization to ensure stable tests
    torch.manual_seed(0)

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    elif hasattr(module, "mean") and hasattr(module, "std"):
        # Gaussian heads
        nn.init.xavier_uniform_(module.mean.weight, gain=0.01)
        nn.init.zeros_(module.mean.bias)
        nn.init.xavier_uniform_(module.std.weight, gain=0.01)
        nn.init.zeros_(module.std.bias)
