from __future__ import annotations

import torch
import torch.nn as nn

from dreamerrl.utils.twohot import BINS, twohot_encode, value_from_logits


class ValueHead(nn.Module):
    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        num_classes: int,
        hidden_size: int,
        num_bins: int | None = None,
    ):
        super().__init__()
        input_dim = deter_size + stoch_size * num_classes
        if num_bins is None:
            num_bins = BINS.numel()
        self.num_bins = num_bins

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_bins),
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

    @staticmethod
    def loss_from_logits(
        logits: torch.Tensor,
        target_returns_symlog: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (T, B, num_bins)
        target_returns_symlog: (T, B) in symlog space
        """
        target_twohot = twohot_encode(target_returns_symlog)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(target_twohot * log_probs).sum(dim=-1).mean()
        return loss

    @staticmethod
    def readout(logits: torch.Tensor) -> torch.Tensor:
        return value_from_logits(logits)
