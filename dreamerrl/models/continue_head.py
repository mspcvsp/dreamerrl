import torch.nn as nn
import torch.nn.functional as F


class ContinueHead(nn.Module):
    """Bernoulli continuation: time limits continue, true terminals do not."""

    def __init__(self, *, latent, net):
        super().__init__()
        self.z_embed = nn.Linear(latent.stoch_size, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, 1),
        )

    def forward(self, h, z):
        return self.net(self.h_embed(h) + self.z_embed(z).sum(dim=1))

    def readout(self, logits):
        return logits.sigmoid().squeeze(-1)

    def loss_from_logits(self, logits, cont_target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), cont_target.float())
