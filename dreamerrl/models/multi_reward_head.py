import torch.nn as nn

from .reward_head import RewardHead


class MultiRewardHead(nn.Module):
    """
    Main reward head + N auxiliary reward heads.
    All heads take (h, z) and output logits.
    """

    def __init__(self, latent, net, num_aux=1):
        super().__init__()
        self.main = RewardHead(latent=latent, net=net)
        self.aux = nn.ModuleList([RewardHead(latent=latent, net=net) for _ in range(num_aux)])

    def forward(self, h, z):
        main_logits = self.main(h, z)
        aux_logits = [head(h, z) for head in self.aux]
        return main_logits, aux_logits
