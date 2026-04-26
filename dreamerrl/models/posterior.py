from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Posterior(nn.Module):
    """
    Factored discrete posterior q(z_t | h_{t-1}, embed_t).

    - stoch_size: number of categorical factors
    - num_classes: number of classes per factor
    - deterministic_latent_for_tests:
        True  -> argmax one-hot (CPU/GPU equivalence tests)
        False -> straight-through Gumbel-Softmax (training)
    """

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        num_classes: int,
        hidden_size: int,
        deterministic_latent_for_tests: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        torch.manual_seed(0)  # deterministic init for tests

        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.deterministic_latent_for_tests = deterministic_latent_for_tests

        in_dim = deter_size + hidden_size

        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc_logits = nn.Linear(hidden_size, stoch_size * num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            logits: (B, stoch_size, num_classes)
            probs:  (B, stoch_size, num_classes)
            z:      (B, stoch_size * num_classes) one-hot (or soft one-hot during training)
        """
        B = h.shape[0]

        x = torch.cat([h, embed], dim=-1)
        x = F.silu(self.fc1(x))

        logits = self.fc_logits(x)  # (B, stoch_size * num_classes)
        logits = logits.view(B, self.stoch_size, self.num_classes)
        probs = F.softmax(logits, dim=-1)

        if self.deterministic_latent_for_tests:
            # Argmax one-hot for deterministic CPU/GPU tests
            idx = probs.argmax(dim=-1)  # (B, stoch_size)
            z = F.one_hot(idx, num_classes=self.num_classes).float()
        else:
            # Straight-through Gumbel-Softmax
            g = -torch.log(-torch.log(torch.rand_like(probs)))
            y = F.softmax((logits + g) / self.temperature, dim=-1)
            # Straight-through: forward = y, backward = argmax
            z = y + (y.argmax(dim=-1) - y).detach()
            z = F.one_hot(z.argmax(dim=-1), num_classes=self.num_classes).float()

        # Flatten factors × classes → (B, stoch_size * num_classes)
        z_flat = z.view(B, -1)

        return {
            "logits": logits,
            "probs": probs,
            "z": z_flat,
        }
