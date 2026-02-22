import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModelState


@pytest.fixture(scope="session")
def deter_size():
    return 32


@pytest.fixture(scope="session")
def stoch_size():
    return 16


@pytest.fixture(scope="session")
def action_dim():
    return 5


@pytest.fixture
def actor(device, deter_size, stoch_size, action_dim):
    return Actor(deter_size, stoch_size, 64, action_dim).to(device)


@pytest.fixture
def critic(device, deter_size, stoch_size):
    return ValueHead(deter_size, stoch_size, 64).to(device)


@pytest.fixture
def fake_state(device, deter_size, stoch_size):
    return WorldModelState(
        h=torch.randn(4, deter_size, device=device),
        z=torch.randn(4, stoch_size, device=device),
    )
