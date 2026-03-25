from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Posterior(nn.Module):
    """
    Posterior q(z_t | h_{t-1}, embed_t) as a diagonal Gaussian.

    deterministic_latent_for_tests:
        - False → full Dreamer sampling (training)
        - True  → z = mean (CPU/GPU numerical equivalence tests)
    """

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        hidden_size: int,
        deterministic_latent_for_tests: bool = False,
    ):
        super().__init__()

        # Deterministic initialization to ensure stable tests
        torch.manual_seed(0)

        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.hidden_size = hidden_size
        self.deterministic_latent_for_tests = deterministic_latent_for_tests

        in_dim = deter_size + hidden_size

        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, stoch_size)
        self.fc2_std = nn.Linear(hidden_size, stoch_size)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([h, embed], dim=-1)
        x = F.silu(self.fc1(x))

        mean = self.fc2_mean(x)
        std = F.softplus(self.fc2_std(x)) + 1e-4

        if self.deterministic_latent_for_tests:
            z = mean
        else:
            eps = torch.randn_like(std)
            z = mean + std * eps

        return {
            "mean": mean,
            "std": std,
            "z": z,
        }
