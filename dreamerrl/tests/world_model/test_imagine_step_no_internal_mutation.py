from copy import deepcopy

import torch


def test_imagine_step_no_internal_mutation(make_world_model):
    wm = make_world_model()
    B = 4
    state = wm.init_state(B)

    before = deepcopy(state)
    _ = wm.imagine_step(state)

    torch.testing.assert_close(state.h, before.h)
    torch.testing.assert_close(state.z, before.z)
