import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    """
    Dreamer‑V3 reward model.
    Accepts factored z and produces categorical reward logits over bins.
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
            nn.Linear(net.hidden_size, net.value_bins),  # categorical logits over bins
        )

        self.apply(self._init_weights)

        self.value_bins = net.value_bins
        self.bin_values = torch.linspace(-1, 1, self.value_bins)

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
        return self.net(features)  # (B, value_bins)

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical reward logits to scalar reward prediction.
        Use expectation over bin centers.
        """
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bin_values.to(logits.device)).sum(dim=-1)

    def loss(self, logits: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Distributional cross-entropy loss for symlog-transformed rewards.
        """
        target = torch.sign(reward) * torch.log1p(torch.abs(reward))
        with torch.no_grad():
            bin_idx = torch.argmin(
                torch.abs(target.unsqueeze(-1) - self.bin_values.to(target.device)),
                dim=-1,
            )
        return F.cross_entropy(logits.view(-1, self.value_bins), bin_idx.view(-1))

    def loss_from_logits(self, logits: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, reward)
