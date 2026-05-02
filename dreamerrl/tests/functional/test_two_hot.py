import torch

from dreamerrl.utils.twohot import twohot_encode
from dreamerrl.utils.types import NetworkConfig


def test_two_hot_basic_properties():
    cfg = NetworkConfig(hidden_size=256, value_bins=51)
    bins = cfg.make_bins()

    B = 7
    targets = torch.linspace(bins[0].item(), bins[-1].item(), steps=B)
    weights = twohot_encode(targets, bins)

    assert weights.shape == (B, cfg.value_bins)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-6)
