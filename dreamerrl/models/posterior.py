from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.types import LatentConfig, NetworkConfig


class Posterior(nn.Module):
    """
    Factored discrete posterior q(z_t | h_{t-1}, embed_t).
    """

    def __init__(
        self,
        *,
        latent: LatentConfig,
        net: NetworkConfig,
        deterministic_latent_for_tests: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        torch.manual_seed(0)

        self.latent = latent
        self.net_cfg = net
        self.temperature = temperature
        self.deterministic_latent_for_tests = deterministic_latent_for_tests

        in_dim = latent.deter_size + net.hidden_size

        self.fc1 = nn.Linear(in_dim, net.hidden_size)
        self.fc_logits = nn.Linear(net.hidden_size, latent.z_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]

        x = torch.cat([h, embed], dim=-1)
        x = F.silu(self.fc1(x))

        logits = self.fc_logits(x)  # (B, z_dim)
        logits = logits.view(B, self.latent.stoch_size, self.latent.num_classes)
        probs = F.softmax(logits, dim=-1)

        if self.deterministic_latent_for_tests:
            idx = probs.argmax(dim=-1)
            z = F.one_hot(idx, num_classes=self.latent.num_classes).float()
        else:
            g = -torch.log(-torch.log(torch.rand_like(probs)))
            y = F.softmax((logits + g) / self.temperature, dim=-1)
            idx = y.argmax(dim=-1)
            z_hard = F.one_hot(idx, num_classes=self.latent.num_classes).float()
            z = z_hard + (y - y.detach())

        z_flat = z.view(B, -1)

        return {
            "logits": logits,
            "probs": probs,
            "z": z_flat,
        }
