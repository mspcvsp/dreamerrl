import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.world_model import WorldModelState


@pytest.mark.actor_critic
def test_actor_act(device):
    B = 4
    deter, stoch, action_dim = 32, 16, 5

    actor = Actor(deter, stoch, 64, action_dim).to(device)

    state = WorldModelState(
        h=torch.randn(B, deter, device=device),
        z=torch.randn(B, stoch, device=device),
    )

    actions, logprobs = actor.act(state)

    assert actions.shape == (B,)
    assert logprobs.shape == (B,)
    assert actions.min() >= 0
    assert actions.max() < action_dim
