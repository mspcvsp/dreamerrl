import torch


def test_critic_gradients_finite(critic, world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    z = torch.randn(B, world_model.stoch_size)

    logits = critic(h, z)
    loss = logits.sum()
    loss.backward()

    for p in critic.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
