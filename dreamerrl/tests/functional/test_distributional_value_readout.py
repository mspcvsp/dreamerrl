import pytest
import torch

from dreamerrl.models.value_head import ValueHead
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.functional
def test_distributional_value_readout_monotonic():
    B = 8
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, value_bins=51)

    head = ValueHead(latent=latent, net=net)

    # V3 deterministic state
    h = torch.zeros(B, latent.deter_size)

    # V3 factored latent (B, K, C)
    z = torch.zeros(B, latent.stoch_size, latent.num_classes)

    logits = head(h, z)

    # Output shape: (B, value_bins)
    assert logits.shape == (B, net.value_bins)

    # Monotonicity: logits must be non-decreasing across bins
    diffs = logits[:, 1:] - logits[:, :-1]
    assert torch.all(diffs >= -1e-5)
