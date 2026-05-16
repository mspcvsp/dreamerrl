import torch.nn as nn


class ContinueHead(nn.Module):
    """
    Dreamer‑V3 continuation model.
    Predicts continuation logits from factored z and deterministic h.
    """

    def __init__(self, *, latent, net):
        super().__init__()

        self.latent = latent
        self.net_cfg = net

        # Embed each categorical factor
        self.z_embed = nn.Linear(latent.num_classes, net.hidden_size)

        # Embed deterministic state
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        # Final continuation head
        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, 1),  # continuation logit
        )

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, K, C)
        """
        z_e = self.z_embed(z)  # (B, K, H)
        z_sum = z_e.sum(dim=1)  # (B, H)
        h_e = self.h_embed(h)  # (B, H)

        features = h_e + z_sum
        return self.net(features)  # (B, 1)
