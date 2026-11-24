import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.transforms import symexp, symlog
from dreamerrl.utils.twohot import twohot_encode
from dreamerrl.utils.types import LatentConfig, NetworkConfig


class RewardHead(nn.Module):
    """
    Dreamer‑V3 reward model.
    Accepts factored z and produces categorical reward logits over bins.
    """

    bin_values: torch.Tensor

    def __init__(self, *, latent, net):
        super().__init__()

        self.latent = latent
        self.net_cfg = net

        self.z_embed = nn.Linear(latent.z_dim, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),  # categorical logits over bins
        )

        self.apply(self._init_weights)

        self.value_bins = net.value_bins

        self.register_buffer("bin_values", torch.linspace(net.bin_min, net.bin_max, self.value_bins))

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
        # Preserve the identity of every categorical factor.
        z_e = self.z_embed(z.reshape(z.shape[0], -1))
        h_e = self.h_embed(h)
        features = h_e + z_e
        return self.net(features)  # (B, value_bins)

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical reward logits to scalar reward prediction.
        Use expectation over bin centers.
        """
        probs = torch.softmax(logits, dim=-1)
        return symexp((probs * self.bin_values).sum(dim=-1))

    def loss(self, logits: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Distributional cross-entropy loss for symlog-transformed rewards.
        """
        with torch.no_grad():
            target = twohot_encode(symlog(reward), self.bin_values)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def loss_from_logits(self, logits: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, reward)


class RewardModel(nn.Module):
    """Single standard reward head, preserving historical `reward_heads.main` keys."""

    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()
        self.main = RewardHead(latent=latent, net=net)
