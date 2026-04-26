import torch
import torch.nn as nn


class ObsDecoder(nn.Module):
    def __init__(self, deter_size: int, stoch_size: int, num_classes: int, hidden_size: int, obs_shape):
        super().__init__()

        if isinstance(obs_shape, int):
            self.obs_dim = obs_shape
        else:
            self.obs_dim = int(torch.tensor(obs_shape).prod())

        # PATCH: updated input dimension for discrete latent
        input_dim = deter_size + stoch_size * num_classes

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
