from dreamerrl.utils.types import NetworkConfig


def test_bin_center_monotonicity():
    cfg = NetworkConfig(hidden_size=256, value_bins=41)
    bins = cfg.make_bins()
    assert (bins[1:] > bins[:-1]).all()
