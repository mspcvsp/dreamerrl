def test_observe_step_shapes(make_world_model, fake_obs):
    wm = make_world_model()
    B = fake_obs.shape[0]

    state = wm.init_state(B)
    out = wm.observe_step(state, fake_obs)

    assert out["state"].h.shape == (B, wm.deter_size)
    assert out["state"].z.shape == (B, wm.stoch_size)
    assert out["recon"].shape == (B, wm.flat_obs_dim)
    assert out["reward_pred"].shape == (B, 1)
    assert out["kl"].shape == ()
