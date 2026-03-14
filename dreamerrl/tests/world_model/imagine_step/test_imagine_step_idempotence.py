import torch


def test_imagine_step_idempotence(make_world_model):
    wm = make_world_model()
    B = 4
    state = wm.init_state(B)

    next1 = wm.imagine_step(state)
    next2 = wm.imagine_step(state)

    torch.testing.assert_close(next1.h, next2.h)
    torch.testing.assert_close(next1.z, next2.z)
