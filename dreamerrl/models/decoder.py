import torch
import torch.nn as nn


class ObsDecoder(nn.Module):
    """
    Dreamer-style observation decoder.
    Reconstructs flattened observations from latent state (h, z).

    Works for:
    - Dreamer-Lite (z = 0)
    - Full Dreamer (z from posterior/prior)
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int, obs_shape):
        super().__init__()

        # Flattened observation dimension
        if isinstance(obs_shape, int):
            self.obs_dim = obs_shape
        else:
            self.obs_dim = int(torch.tensor(obs_shape).prod())

        input_dim = deter_size + stoch_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.obs_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, stoch_size)
        Returns: reconstructed obs (B, obs_dim)
        """
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
