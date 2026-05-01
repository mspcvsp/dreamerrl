import torch
import torch.nn as nn

from dreamerrl.utils.types import LatentConfig, NetworkConfig


class ObsDecoder(nn.Module):
    def __init__(self, *, latent: LatentConfig, net: NetworkConfig, output_dim):
        super().__init__()

        if isinstance(output_dim, int):
            self.obs_dim = output_dim
        else:
            self.obs_dim = int(torch.tensor(output_dim).prod())

        input_dim = latent.deter_size + latent.z_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, self.obs_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
