def test_world_model_shapes(world_model, fake_obs):
    B = fake_obs.size(0)
    state = world_model.init_state(B)
    out = world_model.observe_step(state, fake_obs)

    assert out["state"].h.shape == (B, world_model.deter_size)
    assert out["state"].z.shape == (B, world_model.stoch_size)
    assert out["recon"].shape == fake_obs.shape
    assert out["reward_pred"].shape == (B, 1)
