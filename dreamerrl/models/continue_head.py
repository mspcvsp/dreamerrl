import torch
import torch.nn as nn


class ContinueHead(nn.Module):
    def __init__(self, deter_size, stoch_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_size + stoch_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        return self.net(x)  # logits
