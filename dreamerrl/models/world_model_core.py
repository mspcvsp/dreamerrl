import torch
import torch.nn as nn
import torch.nn.functional as F

from .deterministic_layernorm import DeterministicLayerNorm


class RSSMCore(nn.Module):
    """Gated deterministic RSSM state: h_next = GRU(embed(z, action), h).

    Uses a standard PyTorch GRUCell. The categorical latent remains factored
    outside the input embedding; the recurrent state is passed explicitly.
    """

    def __init__(self, *, latent, net):
        super().__init__()

        assert net.action_dim is not None, "RSSMCore requires action_dim"

        self.latent = latent
        self.net_cfg = net

        input_dim = latent.z_dim + net.action_dim
        self.fc1 = nn.Linear(input_dim, net.hidden_size)
        self.ln1 = DeterministicLayerNorm(net.hidden_size)
        self.gru = nn.GRUCell(net.hidden_size, latent.deter_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h, z, action):
        if z.dim() > 2:
            z = z.reshape(z.shape[0], -1)

        x = torch.cat(
            [
                z,
                action,
            ],
            dim=-1,
        )

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)

        return self.gru(x, h)
