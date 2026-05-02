from dreamerrl.utils.types import NetworkConfig


def test_network_config_bins_shape():
    cfg = NetworkConfig(hidden_size=256, value_bins=41)
    bins = cfg.make_bins()
    assert bins.shape == (41,)
