import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.types import LatentConfig, NetworkConfig
from .deterministic_layernorm import DeterministicLayerNorm


class RSSMCore(nn.Module):
    """
    Dreamer‑V3 deterministic core:
    h_{t+1} = f(h_t, action_t)

    NOTE:
    - z_t is *not* part of the deterministic update in Dreamer‑V3.
    - z_t is handled entirely by Prior/Posterior.
    """

    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()

        assert net.action_dim is not None, "RSSMCore requires action_dim"

        self.latent = latent
        self.net_cfg = net

        # Dreamer‑V3 input: [h, action]
        input_dim = latent.deter_size + net.action_dim

        self.fc1 = nn.Linear(input_dim, net.hidden_size)
        self.ln1 = DeterministicLayerNorm(net.hidden_size)
        self.fc2 = nn.Linear(net.hidden_size, latent.deter_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Dreamer‑V3 deterministic transition:
        h_{t+1} = f(h_t, action_t)
        """
        x = torch.cat([h, action], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)
        return self.fc2(x)
