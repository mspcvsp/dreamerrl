import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.tests.helpers.numerical_equivalence import assert_close_cpu_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_observe_step_cpu_gpu_equivalence(obs_space, action_dim, world_model_config):
    B, obs_dim = 4, obs_space.shape[0]

    def fn(obs):
        wm = WorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            device=obs.device,
            **world_model_config,
        ).to(obs.device)

        state = wm.init_state(B)
        out = wm.observe_step(state, obs)
        return out["state"].h, out["state"].z

    obs = torch.randn(B, obs_dim)
    assert_close_cpu_gpu(fn, obs)
