import torch


def test_observe_step_outputs_finite(make_world_model, fake_obs):
    wm = make_world_model()
    state = wm.init_state(fake_obs.shape[0])

    out = wm.observe_step(state, fake_obs)

    assert torch.isfinite(out["state"].h).all()
    assert torch.isfinite(out["state"].z).all()
    assert torch.isfinite(out["recon"]).all()
    assert torch.isfinite(out["reward_pred"]).all()
    assert torch.isfinite(out["kl"]).all()
