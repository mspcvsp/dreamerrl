import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Dreamer-style value function.
    Predicts scalar value from latent state (h, z).
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int):
        super().__init__()
        torch.manual_seed(0)

        input_dim = deter_size + stoch_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),  # scalar value
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
        Returns: value estimate (B, 1)
        """
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
