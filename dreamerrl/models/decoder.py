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

        self.z_embed = nn.Linear(latent.stoch_size, net.hidden_size)
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
        # Embed z per factor → (B, K, H)
        z_e = self.z_embed(z)

        # Sum over factors → (B, H)
        z_sum = z_e.sum(dim=1)

        # Embed h → (B, H)
        h_e = self.h_embed(h)

        # Fuse
        features = h_e + z_sum

        return self.net(features)
