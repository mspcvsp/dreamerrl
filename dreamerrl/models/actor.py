import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Dreamer-style actor head.
    Outputs logits for a categorical action distribution.

    Works for:
    - Dreamer-Lite (z = 0)
    - Full Dreamer (z from posterior/prior)
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int, action_dim: int):
        super().__init__()

        input_dim = deter_size + stoch_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim),  # logits
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, stoch_size)
        Returns: logits (B, action_dim)
        """
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
