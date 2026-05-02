import torch

from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_rssm_core_shapes():
    B = 8
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=200, action_dim=5)
    assert net.action_dim is not None, "action_dim must be set for RSSMCore"

    core = RSSMCore(latent=latent, net=net)

    h = torch.zeros(B, latent.deter_size)
    action = torch.zeros(B, net.action_dim)

    h_next = core(h, action)
    assert h_next.shape == (B, latent.deter_size)
