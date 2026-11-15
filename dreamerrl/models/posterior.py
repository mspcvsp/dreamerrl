from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.models.categorical import mixed_probs, sample_onehot
from dreamerrl.utils.types import LatentConfig, NetworkConfig


class Posterior(nn.Module):
    """
    Dreamer‑V3 factored discrete posterior q(z_t | h_t, embed_t).

    Produces:
        logits: (B, K, C)
        probs:  (B, K, C)
        z:      (B, K, C) one‑hot (hard or straight‑through)
    """

    def __init__(
        self,
        *,
        latent: LatentConfig,
        net: NetworkConfig,
        deterministic_latent_for_tests: bool = False,
    ):
        super().__init__()

        self.latent = latent
        self.net_cfg = net
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
        logits = logits.view(B, self.latent.num_classes, self.latent.stoch_size)  # (B, K, C)
        probs = mixed_probs(logits, self.net_cfg.unimix)
        # The same mixed distribution is used for sampling, KL, and diagnostics.
        if self.net_cfg.unimix:
            logits = probs.log()
        z = sample_onehot(probs, deterministic=self.deterministic_latent_for_tests)

        return {"logits": logits, "probs": probs, "z": z}
