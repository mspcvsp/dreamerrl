import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.types import LatentConfig, NetworkConfig

from .deterministic_layernorm import DeterministicLayerNorm


class RSSMCore(nn.Module):
    """
    IMPORTANT:
    ---------
    RSSMCore intentionally does NOT use a GRU. Dreamer‑V3 found that GRUs introduce instability (especially with
    discrete latents), are nondeterministic across devices, and add unnecessary complexity. The deterministic
    transition is a simple MLP with LayerNorm:

    h_{t+1} = f(h_t, action_t)

    Recurrence comes from unrolling this function over time, not from a recurrent cell. This design is more stable,
    more reproducible, and easier to train.

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
        IMPORTANT:
        ---------
        RSSMCore consumes a *one‑hot* action vector, not logits and not a discrete action index. Dreamer‑V3 uses a
        categorical policy:

        logits = actor(h, z)
        dist = Categorical(logits)
        a = dist.sample()                     # integer action ID
        action = F.one_hot(a, action_dim)     # (B, action_dim)

        Only this one‑hot action vector should be passed to RSSMCore.forward(). Feeding logits or integer IDs will
        silently corrupt the latent dynamics.

        Dreamer‑V3 deterministic transition:
        -----------------------------------
        h_{t+1} = f(h_t, action_t)
        """
        x = torch.cat([h, action], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)
        return self.fc2(x)
