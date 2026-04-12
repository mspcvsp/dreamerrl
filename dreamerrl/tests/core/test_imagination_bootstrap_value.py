def test_imagination_bootstrap_value(world_model, actor, critic):
    B, T = 3, 5
    state = world_model.init_state(B)

    traj = world_model.imagine_trajectory_for_training(actor, critic, state, T)

    assert "bootstrap_value" in traj
    assert traj["bootstrap_value"].shape == (B,)
