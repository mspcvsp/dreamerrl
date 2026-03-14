from copy import deepcopy

import torch


def test_imagination_rollout_no_internal_mutation(world_model):
    B, horizon = 4, 6
    state0 = world_model.init_state(B)
    before = deepcopy(state0)

    _ = world_model.imagination_rollout(state0, horizon=horizon)

    torch.testing.assert_close(state0.h, before.h)
    torch.testing.assert_close(state0.z, before.z)
