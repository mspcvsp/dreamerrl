from dreamerrl.utils.types import LatentConfig


def test_latent_config_z_dim():
    cfg = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    assert cfg.z_dim == 30 * 32
