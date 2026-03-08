def test_world_model_shapes(make_world_model, fake_obs):
    wm = make_world_model()

    B = fake_obs.shape[0]
    state = wm.init_state(B)

    out = wm.observe_step(state, fake_obs)

    assert out["state"].h.shape == (B, wm.deter_size)
    assert out["state"].z.shape == (B, wm.stoch_size)
    assert out["recon"].shape == fake_obs.shape
    assert out["reward_pred"].shape == (B, 1)

    # imagine_step shapes
    next_state = wm.imagine_step(state)
    assert next_state.h.shape == (B, wm.deter_size)
    assert next_state.z.shape == (B, wm.stoch_size)
