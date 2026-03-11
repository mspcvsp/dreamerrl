import torch


def test_observe_step_gradients_finite(make_world_model, fake_obs):
    wm = make_world_model()
    B = fake_obs.shape[0]
    state = wm.init_state(B)

    fake_obs = fake_obs.clone().requires_grad_(True)

    out = wm.observe_step(state, fake_obs)
    loss = out["recon"].mean() + out["reward_pred"].mean() + out["kl"]

    loss.backward()

    for p in wm.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
