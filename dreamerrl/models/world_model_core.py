import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.types import LatentConfig, NetworkConfig

from .deterministic_layernorm import DeterministicLayerNorm


class RSSMCore(nn.Module):
    """
    RSSM deterministic core: h_{t+1} = f(h_t, z_t).
    """

    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()

        self.latent = latent
        self.net_cfg = net

        input_dim = latent.deter_size + latent.z_dim

        self.fc1 = nn.Linear(input_dim, net.hidden_size)
        self.ln1 = DeterministicLayerNorm(net.hidden_size)
        self.fc2 = nn.Linear(net.hidden_size, latent.deter_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)
        return self.fc2(x)
