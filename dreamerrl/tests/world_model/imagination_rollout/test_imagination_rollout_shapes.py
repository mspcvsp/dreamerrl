def test_imagination_rollout_shapes(world_model, fake_batch):
    B, _ = fake_batch["state"].shape[:2]
    horizon = 5

    state0 = world_model.init_state(B)
    rollout = world_model.imagination_rollout(state0, horizon=horizon)

    assert rollout["state"].h.shape == (B, horizon, world_model.deter_size)
    assert rollout["state"].z.shape == (B, horizon, world_model.stoch_size)
    assert rollout["reward_pred"].shape == (B, horizon, 1)
