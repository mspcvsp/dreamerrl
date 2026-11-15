"""Prediction heads must retain which categorical factor carries each value."""

import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.continue_head import ContinueHead
from dreamerrl.models.decoder import ObsDecoder
from dreamerrl.models.reward_head import RewardHead
from dreamerrl.models.value_head import ValueHead
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.parametrize("kind", [Actor, RewardHead, ValueHead, ContinueHead, ObsDecoder])
def test_head_distinguishes_latent_factor_permutations(kind):
    latent = LatentConfig(deter_size=2, num_classes=2, stoch_size=3)
    net = NetworkConfig(hidden_size=4, action_dim=3, value_bins=5)
    kwargs = dict(latent=latent, net=net)
    head = kind(**kwargs, output_dim=3) if kind is ObsDecoder else kind(**kwargs)
    # Two valid latents with identical category counts but different factor identities.
    z = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], requires_grad=True)
    h = torch.zeros(2, 2)
    assert torch.equal(z[0].sum(0), z[1].sum(0))
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.z_embed.weight[0, 0] = 1
        head.z_embed.weight[0, 3] = -1
        for layer in head.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight[0, 0] = 1
    out = head(h, z)
    assert out.shape[0] == 2 and torch.isfinite(out).all()
    assert not torch.allclose(out[0], out[1])
    out[:, 0].sum().backward()
    assert z.grad is not None
    assert z.grad[0, 0, 0] != z.grad[0, 1, 0]
