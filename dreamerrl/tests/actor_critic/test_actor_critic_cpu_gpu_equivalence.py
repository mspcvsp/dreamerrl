import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.tests.helpers.numerical_equivalence import assert_close_cpu_gpu


@pytest.mark.actor_critic
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actor_critic_cpu_gpu_equivalence():
    B = 4
    deter, stoch, action_dim = 32, 16, 5

    def fn(h, z):
        actor = Actor(deter, stoch, 64, action_dim).to(h.device)
        critic = ValueHead(deter, stoch, 64).to(h.device)
        logits = actor(h, z)
        values = critic(h, z)
        return logits, values

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    assert_close_cpu_gpu(fn, h, z)
