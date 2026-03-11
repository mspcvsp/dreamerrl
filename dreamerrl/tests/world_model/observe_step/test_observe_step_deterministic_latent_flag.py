import torch


def test_observe_step_deterministic_latent_flag(make_world_model, fake_obs, monkeypatch):
    monkeypatch.setenv("DREAMER_DETERMINISTIC_TEST", "1")
    wm = make_world_model()

    state = wm.init_state(fake_obs.shape[0])

    out1 = wm.observe_step(state, fake_obs)
    out2 = wm.observe_step(state, fake_obs)

    torch.testing.assert_close(out1["state"].z, out2["state"].z)
