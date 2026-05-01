from __future__ import annotations

import torch
import torch.nn as nn

from dreamerrl.utils.twohot import twohot_encode, value_from_logits
from dreamerrl.utils.types import LatentConfig, NetworkConfig


class ValueHead(nn.Module):
    """
    Distributional value head in symlog space with two-hot targets.

    NOTE: Uses the same bins as RewardHead via NetworkConfig,
    so actor-critic update, imagination, and readout are all consistent.
    """

    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()

        assert net.value_bins is not None, "Critic requires value_bins"

        input_dim = latent.deter_size + latent.z_dim
        self.bins = net.make_bins()

        self.net = nn.Sequential(
            nn.Linear(input_dim, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.net(x)  # (B, num_bins)

    def loss_from_logits(
        self,
        logits: torch.Tensor,
        target_returns_symlog: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (T, B, num_bins)
        target_returns_symlog: (T, B) in symlog space
        """
        target_twohot = twohot_encode(target_returns_symlog, self.bins)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(target_twohot * log_probs).sum(dim=-1).mean()
        return loss

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        return value_from_logits(logits, self.bins)
