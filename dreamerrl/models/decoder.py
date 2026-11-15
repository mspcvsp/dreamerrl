import torch.nn as nn


class ObsDecoder(nn.Module):
    """
    Dreamer‑V3 observation decoder.
    Accepts:
        h: (B, deter_size)
        z: (B, K, C) factored discrete latent
    """

    def __init__(self, *, latent, net, output_dim: int):
        super().__init__()

        self.latent = latent
        self.net_cfg = net
        self.output_dim = output_dim

        self.z_embed = nn.Linear(latent.z_dim, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        # Final decoder MLP
        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, output_dim),
        )

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, K, C)
        """
        # Preserve the identity of every categorical factor.
        z_e = self.z_embed(z.reshape(z.shape[0], -1))
        h_e = self.h_embed(h)
        features = h_e + z_e

        return self.net(features)
