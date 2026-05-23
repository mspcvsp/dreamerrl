import torch.nn as nn
import torch.nn.functional as F


class ContinueHead(nn.Module):
    def __init__(self, *, latent, net):
        super().__init__()
        self.z_embed = nn.Linear(latent.num_classes, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.value_bins),  # categorical binary logits
        )

    def forward(self, h, z):
        z_e = self.z_embed(z).sum(dim=1)
        h_e = self.h_embed(h)
        features = h_e + z_e
        return self.net(features)  # (B, 2)

    def loss_from_logits(self, logits, is_terminal):
        return F.cross_entropy(logits, is_terminal.long())
