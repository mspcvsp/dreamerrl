import torch


def test_observe_step_kl_stability(make_world_model, fake_obs):
    wm = make_world_model()
    state = wm.init_state(fake_obs.shape[0])

    out = wm.observe_step(state, fake_obs)
    kl = out["kl"]

    assert kl.dim() == 0
    assert torch.isfinite(kl)
