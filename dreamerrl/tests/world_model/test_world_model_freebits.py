def test_world_model_freebits(world_model, obs_batch):
    B = obs_batch["obs"].shape[0]
    state = world_model.init_state(B)
    out = world_model.observe_step(state, obs_batch["obs"])

    kl = world_model.kl_divergence(out["post_stats"], out["prior_stats"])
    assert (kl >= 0).all()
