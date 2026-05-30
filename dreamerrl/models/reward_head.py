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

        self.z_embed = nn.Linear(latent.stoch_size, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),  # categorical logits over bins
        )

        self.apply(self._init_weights)

        self.value_bins = net.value_bins

        # Bins span [-1, +1] in symlog space because symlog compresses magnitudes into a bounded, approximately linear
        # region around 0. Large raw values map to small symlog values, so a fixed symmetric range [-1, +1] covers
        # almost all realistic rewards/returns after symlog, making distributional classification stable and
        # domain‑agnostic.
        #
        # smymg: symlog is defined as sign(x) * log(1 + abs(x)), so bin centers in symlog space correspond to nonlinear
        # bin edges in raw reward space. For example, with 5 bins, the centers are at [-1, -0.5, 0, 0.5, 1] in symlog
        # space, which correspond to approximately [-0.63, -0.20, 0, 0.20, 0.63] in raw reward space.
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
