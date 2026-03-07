import torch
import torch.nn as nn
import torch.nn.functional as F

from .deterministic_layernorm import DeterministicLayerNorm


class RSSMCore(nn.Module):
    """
    RSSM deterministic core used by both Dreamer-Lite and full Dreamer.

    - Always updates deterministic state h.
    - In Dreamer-Lite: z is always zero.
    - In full Dreamer: z is sampled from posterior/prior.

    Architecture:
    - Concatenate [h, z]
    - LayerNorm
    - SiLU activation
    - Linear → next deterministic state
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int):
        super().__init__()

        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.hidden_size = hidden_size

        # ---------------------------------------------------------
        # Deterministic transition: h' = f(h, z)
        # ---------------------------------------------------------
        self.fc1 = nn.Linear(deter_size + stoch_size, hidden_size)
        self.ln1 = DeterministicLayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, deter_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------
    # Forward pass: deterministic update
    # -------------------------------------------------------------
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h: (B, deter_size)
        z: (B, stoch_size)  -- zero vector in Dreamer-Lite
        """
        x = torch.cat([h, z], dim=-1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)

        h_next = self.fc2(x)
        return h_next
