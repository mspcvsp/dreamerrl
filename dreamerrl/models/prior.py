import torch
import torch.nn as nn
import torch.nn.functional as F


class Prior(nn.Module):
    """
    p(z_t | h_{t-1})
    Dreamer-style Gaussian prior over stochastic latent state.
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int = 256):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * stoch_size),  # mean + log_std
        )

        self.stoch_size = stoch_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h):
        stats = self.fc(h)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(log_std) + 1e-5
        z = mean + std * torch.randn_like(std)
        return {"mean": mean, "std": std, "z": z}
