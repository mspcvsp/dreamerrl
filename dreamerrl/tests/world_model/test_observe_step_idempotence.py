import torch


def test_observe_step_idempotence(make_world_model, fake_obs):
    wm = make_world_model()
    state = wm.init_state(batch_size=fake_obs.shape[0])

    out1 = wm.observe_step(state, fake_obs)
    out2 = wm.observe_step(state, fake_obs)

    torch.testing.assert_close(out1["state"].h, out2["state"].h)
    torch.testing.assert_close(out1["state"].z, out2["state"].z)
    torch.testing.assert_close(out1["recon"], out2["recon"])
    torch.testing.assert_close(out1["reward_pred"], out2["reward_pred"])
