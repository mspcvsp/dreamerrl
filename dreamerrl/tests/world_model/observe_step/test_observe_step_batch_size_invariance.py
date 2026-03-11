import torch


def test_observe_step_batch_size_invariance(make_world_model, obs_space):
    wm = make_world_model()

    h1 = torch.randn(1, obs_space.shape[0])
    h4 = h1.repeat(4, 1)

    state1 = wm.init_state(1)
    state4 = wm.init_state(4)

    out1 = wm.observe_step(state1, h1)
    out4 = wm.observe_step(state4, h4)

    torch.testing.assert_close(out4["state"].h, out1["state"].h.repeat(4, 1))
