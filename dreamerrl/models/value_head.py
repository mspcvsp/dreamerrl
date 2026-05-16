import torch
import torch.nn as nn

from dreamerrl.utils.twohot import twohot_encode, value_from_logits


class ValueHead(nn.Module):
    """
    Dreamer‑V3 distributional value head.
    Consumes factored z (B, K, C) and deterministic h (B, deter_size).
    Produces logits over value bins.
    """

    def __init__(self, *, latent, net):
        super().__init__()

        assert net.value_bins is not None, "ValueHead requires value_bins"

        self.latent = latent
        self.net_cfg = net
        self.bins = net.make_bins()

        # Embed each categorical factor
        self.z_embed = nn.Linear(latent.num_classes, net.hidden_size)

        # Embed deterministic state
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        # Final critic network
        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),
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
        return self.net(features)  # (B, num_bins)

    def loss(self, logits, target_returns_symlog):
        """
        logits: (T, B, num_bins)
        target_returns_symlog: (T, B)
        """
        target_twohot = twohot_encode(target_returns_symlog, self.bins)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(dim=-1).mean()

    def readout(self, logits):
        return value_from_logits(logits, self.bins)
