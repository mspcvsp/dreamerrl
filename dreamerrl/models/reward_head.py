import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """
    Dreamer‑V3 reward model.
    Accepts factored z and produces scalar reward logits.
    """

    def __init__(self, *, latent, net):
        super().__init__()

        self.latent = latent
        self.net_cfg = net

        self.z_embed = nn.Linear(latent.num_classes, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, 1),  # scalar reward logit
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert reward logits to scalar rewards.
        For now, treat logits as direct scalar predictions.
        """
        return logits.squeeze(-1)
