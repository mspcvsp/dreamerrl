import pytest
import torch

from dreamerrl.models.value_head import ValueHead


@pytest.mark.actor_critic
def test_critic_shapes(device):
    B = 4
    deter, stoch = 32, 16

    critic = ValueHead(deter, stoch, 64).to(device)

    h = torch.randn(B, deter, device=device)
    z = torch.randn(B, stoch, device=device)

    values = critic(h, z)
    assert values.shape == (B, critic.num_bins)
