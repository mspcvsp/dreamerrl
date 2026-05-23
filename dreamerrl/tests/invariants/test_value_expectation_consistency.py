import pytest
import torch

from dreamerrl.models.value_head import ValueHead
from dreamerrl.utils.transforms import symexp
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.invariants
def test_value_expectation_consistency():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, value_bins=41)
    head = ValueHead(latent=latent, net=net)

    B = 6
    h = torch.randn(B, latent.deter_size)
    z = torch.randn(B, latent.stoch_size, latent.num_classes)

    logits = head(h, z)
    probs = logits.softmax(-1)
    bins = head.bin_values

    expected = symexp((probs * bins).sum(-1))
    assert torch.isfinite(expected).all()
