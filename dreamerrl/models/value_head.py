import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.utils.transforms import symexp, symlog
from dreamerrl.utils.twohot import twohot_encode


class ValueHead(nn.Module):
    """
    Dreamer‑V3 value model.
    Consumes factored z and deterministic h.
    Outputs categorical logits over value bins.
    """

    bin_values: torch.Tensor

    def __init__(self, *, latent, net, probe=None):
        super().__init__()

        self.latent = latent
        self.net_cfg = net
        self.probe = probe

        self.z_embed = nn.Linear(latent.z_dim, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),  # distribution over value bins
        )

        self.value_bins = net.value_bins
        self.register_buffer("bin_values", torch.linspace(net.bin_min, net.bin_max, self.value_bins))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h: (B, deter_size)
        z: (B, K, C)
        """
        # Preserve the identity of every categorical factor.
        z_e = self.z_embed(z.reshape(z.shape[0], -1))
        h_e = self.h_embed(h)
        features = h_e + z_e
        logits = self.net(features)

        if self.probe:
            self.probe.critic(logits)

        return logits

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Turn categorical logits into scalar value by expectation over bin centers.
        """
        probs = torch.softmax(logits, dim=-1)
        return symexp((probs * self.bin_values).sum(dim=-1))  # (B,)

    def loss(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Two-hot cross-entropy; returns are in original reward units.
        """
        with torch.no_grad():
            target = twohot_encode(symlog(returns), self.bin_values)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def loss_from_logits(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, returns)
