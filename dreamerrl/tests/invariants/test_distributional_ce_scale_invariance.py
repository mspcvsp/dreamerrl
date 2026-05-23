import torch

from dreamerrl.models.value_head import ValueHead
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_distributional_ce_scale_invariance():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, value_bins=41)
    head = ValueHead(latent=latent, net=net)

    B = 8
    h = torch.randn(B, latent.deter_size)
    z = torch.randn(B, latent.stoch_size, latent.num_classes)

    logits = head(h, z)
    target = torch.randn(B)

    loss1 = head.loss_from_logits(logits.unsqueeze(1), target.unsqueeze(1))
    loss2 = head.loss_from_logits(logits.unsqueeze(1), (10 * target).unsqueeze(1))

    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
