"""GRU equation, memory path, and sequence-gradient regressions."""

import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.fixture
def gated_core():
    return RSSMCore(
        latent=LatentConfig(deter_size=5, num_classes=2, stoch_size=3), net=NetworkConfig(hidden_size=7, action_dim=4)
    )


def test_rssm_matches_standard_gru_equations(gated_core):
    core = gated_core
    h, z, action = torch.randn(3, 5), torch.randn(3, 2, 3), torch.eye(4)[:3]
    x = F.silu(core.ln1(core.fc1(torch.cat([z.flatten(1), action], -1))))
    ir, iz, inn = F.linear(x, core.gru.weight_ih, core.gru.bias_ih).chunk(3, -1)
    hr, hz, hn = F.linear(h, core.gru.weight_hh, core.gru.bias_hh).chunk(3, -1)
    reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
    candidate = (inn + reset * hn).tanh()
    expected = (1 - update) * candidate + update * h
    torch.testing.assert_close(core(h, z, action), expected)
    torch.testing.assert_close(core(h, z.flatten(1), action), expected)


@pytest.mark.parametrize("retain", [False, True])
def test_update_gate_can_retain_or_replace_memory(gated_core, retain):
    core = gated_core
    with torch.no_grad():
        for param in core.parameters():
            param.zero_()
        # PyTorch gate order: reset, update, candidate. Candidate is zero.
        core.gru.bias_ih[5:10].fill_(100 if retain else -100)
    h = torch.randn(2, 5, requires_grad=True)
    current = h
    for _ in range(40):
        current = core(current, torch.zeros(2, 2, 3), torch.zeros(2, 4))
    torch.testing.assert_close(current, h if retain else torch.zeros_like(h), rtol=0, atol=0)
    current.sum().backward()
    assert h.grad is not None
    torch.testing.assert_close(h.grad, torch.ones_like(h) if retain else torch.zeros_like(h), rtol=0, atol=0)


def test_sequence_gradients_reach_state_latents_and_actions(gated_core):
    initial = torch.randn(2, 5, requires_grad=True)
    latents = torch.randn(6, 2, 2, 3, requires_grad=True)
    actions = torch.randn(6, 2, 4, requires_grad=True)
    current = initial
    for z, action in zip(latents, actions):
        current = gated_core(current, z, action)
    current.square().mean().backward()
    for tensor in [initial, latents, actions]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for parameter in gated_core.gru.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
