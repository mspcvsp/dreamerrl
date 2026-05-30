import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinueHead(nn.Module):
    def __init__(self, *, latent, net):
        super().__init__()

        # Explicit type annotation fixes Pylance
        self.bin_values: torch.Tensor

        # register bin centers as a buffer (Tensor, not Module)
        self.register_buffer("bin_values", torch.linspace(-1.0, 1.0, net.value_bins))

        self.latent = latent
        self.net_cfg = net

        self.z_embed = nn.Linear(latent.stoch_size, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),  # distributional continuation
        )

    def forward(self, h, z):
        z_e = self.z_embed(z).sum(dim=1)
        h_e = self.h_embed(h)
        features = h_e + z_e
        return self.net(features)  # (B, value_bins)

    def loss_from_logits(self, logits, cont_target):
        """
        logits: (B, L, value_bins)
        cont_target: (B, L) float in {0,1}
        """
        B, L, C = logits.shape

        # symlog transform
        target = torch.sign(cont_target) * torch.log1p(torch.abs(cont_target))

        # discretize to nearest bin
        with torch.no_grad():
            bin_idx = torch.argmin(
                torch.abs(target.unsqueeze(-1) - self.bin_values.to(target.device)),
                dim=-1,
            )  # (B, L)

        # cross-entropy over bins
        return F.cross_entropy(
            logits.reshape(B * L, C),
            bin_idx.reshape(B * L),
        )
