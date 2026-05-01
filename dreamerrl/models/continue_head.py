import torch
import torch.nn as nn

from dreamerrl.utils.types import LatentConfig, NetworkConfig


class ContinueHead(nn.Module):
    """
    Predicts continuation logit (not two-hot, no value bins).
    Uses the same latent geometry as the rest of the world model.
    """

    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()
        input_dim = latent.deter_size + latent.z_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.net(x)  # (B, 1) logits
