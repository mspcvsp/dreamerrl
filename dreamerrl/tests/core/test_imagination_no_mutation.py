from copy import deepcopy

import torch


def test_imagination_no_mutation(world_model, actor, critic):
    B, T = 3, 5
    state = world_model.init_state(B)
    before = deepcopy(state)

    _ = world_model.imagine_trajectory_for_training(actor, critic, state, T)

    assert torch.allclose(before.h, state.h)
    assert torch.allclose(before.z, state.z)
