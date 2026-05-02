import torch

from dreamerrl.models.value_head import ValueHead, value_from_logits
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_distributional_value_readout_monotonic():
    B = 8
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, value_bins=51)

    head = ValueHead(latent=latent, net=net)
    h = torch.zeros(B, latent.deter_size)
    z = torch.zeros(B, latent.z_dim)

    logits = head(h, z)
    v = value_from_logits(logits, net.make_bins())

    assert v.shape == (B,)
