import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.tests.helpers.numerical_equivalence import assert_close_cpu_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_imagine_step_cpu_gpu_equivalence(obs_space, action_dim, world_model_config):
    B = 4

    def fn(state):
        wm = WorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            device=state.h.device,
            **world_model_config,
        ).to(state.h.device)

        return wm.imagine_step(state).h, wm.imagine_step(state).z

    state_cpu = WorldModel(
        obs_space=obs_space,
        action_dim=action_dim,
        device=torch.device("cpu"),
        **world_model_config,
    ).init_state(B)

    assert_close_cpu_gpu(fn, state_cpu)
