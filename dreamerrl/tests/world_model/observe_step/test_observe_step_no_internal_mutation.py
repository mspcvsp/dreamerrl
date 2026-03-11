from copy import deepcopy

import torch


def test_observe_step_no_internal_mutation(make_world_model, fake_obs):
    wm = make_world_model()
    state = wm.init_state(batch_size=fake_obs.shape[0])

    state_before = deepcopy(state)
    obs_before = fake_obs.clone()

    wm.observe_step(state, fake_obs)

    torch.testing.assert_close(state.h, state_before.h)
    torch.testing.assert_close(state.z, state_before.z)
    torch.testing.assert_close(fake_obs, obs_before)
