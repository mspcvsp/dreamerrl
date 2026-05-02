import torch

from dreamerrl.utils.types import NetworkConfig


def test_bins_monotonic():
    cfg = NetworkConfig(hidden_size=256, value_bins=41)
    bins = cfg.make_bins()
    assert torch.all(bins[1:] > bins[:-1])
